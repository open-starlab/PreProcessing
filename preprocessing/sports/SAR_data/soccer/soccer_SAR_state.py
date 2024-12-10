import argparse
import logging
import re
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import copy
import pandas as pd
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from soccer.state_preprocess.preprocess_frame import frames2events
from soccer.state_preprocess.reward_model import RewardModelBase
from soccer.utils.file_utils import load_json, save_as_jsonlines, save_formatted_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_game(game_dir: Path, args: argparse.Namespace, config: dict) -> None:
    # save_dir = args.preprocessed_data_dir / game_dir.name
    # if not save_dir.exists():
    logger.info(f"preprocessing started... {game_dir.name}")
    start = time.time()
    frames = pd.read_json(game_dir / 'frames.jsonl', lines=True, orient='records')
    reward_model = RewardModelBase.from_params(config['reward_model'])
    events = frames2events(
        frames,
        data_type = args.data,
        origin_pos=config['origin_pos'],
        reward_model=reward_model,
        absolute_coordinates=config['absolute_coordinates'],
        min_frame_len_threshold=config['min_frame_len_threshold'],
        max_frame_len_threshold=config['max_frame_len_threshold'],
    )
    save_as_jsonlines(
        [event.to_dict() for event in events], args.preprocessed_data_dir / game_dir.name / 'events.jsonl'
    )
    logger.info(f"preprocessing finished... game_id: {game_dir.name} ({time.time() - start:.2f} sec)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessing_config", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--cleaned_data_dir", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--preprocessed_data_dir", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--data", type=str, default='laliga')
    args = parser.parse_args()

    config = load_json(args.preprocessing_config)

    game_dirs = [dir_ for dir_ in sorted(args.cleaned_data_dir.glob('*')) if re.match(r'\d{7}', dir_.name)]
    args.preprocessed_data_dir.mkdir(parents=True, exist_ok=True)

    if args.num_process == 1:
        for game_dir in tqdm(game_dirs[args.start_index : args.end_index]):
            preprocess_game(game_dir, args=args, config=copy.deepcopy(config))
    else:
        def preprocess_game_with_copy(game_dir, args, config):
            return preprocess_game(game_dir, args=args, config=copy.deepcopy(config))

        with Pool(processes=args.num_process) as pool:
            pool.map(partial(preprocess_game_with_copy, args=args, config=config), game_dirs[args.start_index : args.end_index])
    
    save_formatted_json(config, args.preprocessed_data_dir / 'config.json')

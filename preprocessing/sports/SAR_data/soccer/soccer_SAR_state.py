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

# if __name__ == '__main__':
#     from state_preprocess.preprocess_frame import frames2events
#     from state_preprocess.reward_model import RewardModelBase
#     from utils.file_utils import load_json, save_as_jsonlines, save_formatted_json
# else:
#     from .state_preprocess.preprocess_frame import frames2events
#     from .state_preprocess.reward_model import RewardModelBase
#     from .utils.file_utils import load_json, save_as_jsonlines, save_formatted_json

from state_preprocess.preprocess_frame import frames2events
from state_preprocess.reward_model import RewardModelBase
from utils.file_utils import load_json, save_as_jsonlines, save_formatted_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_single_game(game_dir: str, league: str, save_dir: str, config: dict) -> None:
    save_dir = Path(save_dir)
    game_dir = Path(game_dir)
    config = load_json(config)
    logger.info(f"preprocessing started... {game_dir.name}")
    start = time.time()
    frames = pd.read_json(game_dir / 'frames.jsonl', lines=True, orient='records')
    reward_model = RewardModelBase.from_params(config['reward_model'])
    events = frames2events(
        frames,
        league = league,
        origin_pos=config['origin_pos'],
        reward_model=reward_model,
        absolute_coordinates=config['absolute_coordinates'],
        min_frame_len_threshold=config['min_frame_len_threshold'],
        max_frame_len_threshold=config['max_frame_len_threshold'],
    )
    save_as_jsonlines(
        [event.to_dict() for event in events], save_dir / game_dir.name / 'events.jsonl'
    )
    logger.info(f"preprocessing finished... game_id: {game_dir.name} ({time.time() - start:.2f} sec)")


def preprocess_game(game_dir: str, league: str, save_dir: str, config: dict) -> None:
    save_dir = Path(save_dir)
    game_dir = Path(game_dir)
    logger.info(f"preprocessing started... {game_dir.name}")
    start = time.time()
    frames = pd.read_json(game_dir / 'frames.jsonl', lines=True, orient='records')
    reward_model = RewardModelBase.from_params(config['reward_model'])
    events = frames2events(
        frames,
        data_type = league,
        origin_pos=config['origin_pos'],
        reward_model=reward_model,
        absolute_coordinates=config['absolute_coordinates'],
        min_frame_len_threshold=config['min_frame_len_threshold'],
        max_frame_len_threshold=config['max_frame_len_threshold'],
    )
    save_as_jsonlines(
        [event.to_dict() for event in events], save_dir / game_dir.name / 'events.jsonl'
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

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


from preprocessing.sports.SAR_data.soccer.state_preprocess.preprocess_frame import frames2events
from preprocessing.sports.SAR_data.soccer.state_preprocess.reward_model import RewardModelBase
from preprocessing.sports.SAR_data.soccer.utils.file_utils import load_json, save_as_jsonlines, save_formatted_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_single_game(game_dir: str, league: str, state_def: str, save_dir: str, config: dict, match_id: str) -> None:
    save_dir = Path(save_dir)
    game_dir = Path(game_dir) / match_id
    config = load_json(config)
    logger.info(f"preprocessing started... {game_dir.name}")
    start = time.time()
    frames = pd.read_json(game_dir / 'frames.jsonl', lines=True, orient='records')
    reward_model = RewardModelBase.from_params(config['reward_model'])

    if state_def == 'PVF':
        events = frames2events(
            frames,
            league = league,
            origin_pos=config['origin_pos'],
            reward_model=reward_model,
            absolute_coordinates=config['absolute_coordinates'],
            min_frame_len_threshold=config['min_frame_len_threshold'],
            max_frame_len_threshold=config['max_frame_len_threshold'],
        )
    elif state_def == 'EDMF':
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



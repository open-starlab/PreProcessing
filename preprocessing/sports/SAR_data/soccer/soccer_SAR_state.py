import argparse
import logging
import re
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import copy
import os
import pickle
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from preprocessing.sports.SAR_data.soccer.state_preprocess.preprocess_frame import frames2events
from preprocessing.sports.SAR_data.soccer.state_preprocess.reward_model import RewardModelBase
from preprocessing.sports.SAR_data.soccer.utils.file_utils import load_json, save_as_jsonlines
from preprocessing.sports.SAR_data.soccer.constant import INPUT_EVENT_COLUMNS_JLEAGUE
from preprocessing.sports.SAR_data.soccer.state_preprocess.preprocess_frame import frames2events_pvs, frames2events_edms
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def preprocess_single_game(game_dir: str, state: str, league: str, save_dir: str, config: dict, match_id: str) -> None:
    save_dir = Path(save_dir)
    game_dir = Path(game_dir) / match_id
    config = load_json(config)
    logger.info(f"preprocessing started... {game_dir.name}")
    start = time.time()
    frames = pd.read_json(game_dir / "frames.jsonl", lines=True, orient="records")
    reward_model = RewardModelBase.from_params(config["reward_model"])
    if state == "PVS":
        events = frames2events_pvs(
            frames,
            league=league,
            origin_pos=config["origin_pos"],
            reward_model=reward_model,
            absolute_coordinates=config["absolute_coordinates"],
            min_frame_len_threshold=config["min_frame_len_threshold"],
            max_frame_len_threshold=config["max_frame_len_threshold"],
        )
    elif state == "EDMS":
        events = frames2events_edms(
            frames,
            league=league,
            origin_pos=config["origin_pos"],
            reward_model=reward_model,
            absolute_coordinates=config["absolute_coordinates"],
            min_frame_len_threshold=config["min_frame_len_threshold"],
            max_frame_len_threshold=config["max_frame_len_threshold"],
        )
    save_as_jsonlines([event.to_dict() for event in events], save_dir / game_dir.name / "events.jsonl")
    logger.info(f"preprocessing finished... game_id: {game_dir.name} ({time.time() - start:.2f} sec)")

def preprocess_single_game(game_dir: str, league: str, save_dir: str, config: dict, match_id: str) -> None:
    save_dir = Path(save_dir)
    game_dir = Path(game_dir) / match_id
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

def generate_qmix_state(df: pd.DataFrame, frame_id: int) -> np.ndarray:
    from constant import compute_observations
    from preprocessing.sports.SAR_data.soccer.soccer_SAR_processing import extract_agent_features
    frame_df = df[df["frame_id"] == frame_id]
    last_positions = {}
    agents, ball = extract_agent_features(frame_df, last_positions)
    obs, _ = compute_observations(agents, ball)
    return np.concatenate(obs)


def process_frame_data(df: pd.DataFrame) -> list:
    """
    Convert cleaned RoboCup2D dataframe into a list of QMix-compatible episodes.
    Each episode is a dict with keys: obs, state, actions, roles, reward, terminated.
    """
    from preprocessing.sports.SAR_data.soccer.state_preprocess.preprocess_frame import extract_agent_features, compute_observations
    episodes = []
    # Group by frame to build time-series
    grouped = df.groupby('frame_id')
    episode = {"obs": [], "state": [], "actions": [], "roles": [], "reward": [], "terminated": []}
    last_positions = {}

    for frame_id, frame_df in grouped:
        # extract agent features (pos, velocity, stamina, last_action)
        agents, ball = extract_agent_features(frame_df, last_positions)
        obs, roles = compute_observations(agents, ball)
        state = np.concatenate(obs)

        episode["obs"].append(obs)
        episode["state"].append(state)
        episode["actions"].append([0] * len(agents))  # placeholder actions
        episode["roles"].append(roles)
        episode["reward"].append(0.0)  # placeholder rewards
        episode["terminated"].append(False)

    # Mark last timestep as terminated
    if episode["terminated"]:
        episode["terminated"][-1] = True
    episodes.append(episode)
    return episodes

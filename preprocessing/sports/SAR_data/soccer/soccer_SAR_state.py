import argparse
import logging
import re
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from preprocessing.sports.SAR_data.soccer.soccer_SAR_processing import extract_agent_features
from preprocessing.sports.SAR_data.soccer.state_preprocess.preprocess_frame import frames2events
from preprocessing.sports.SAR_data.soccer.state_preprocess.reward_model import RewardModelBase
from preprocessing.sports.SAR_data.soccer.utils.file_utils import load_json, save_as_jsonlines, save_formatted_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

def assign_roles_softmax(agents, ball_pos):
    dists = np.array([np.linalg.norm(agent["pos"] - ball_pos) for agent in agents.values()])
    logits = -dists
    probs = softmax(logits)
    role_ids = {}

    sorted_roles = np.argsort(probs)[::-1]
    for i, (pid, _) in zip(sorted_roles, agents.items()):
        role_ids[pid] = i % len(ROLES) # type: ignore
    return role_ids

def compute_observations(agents, ball_pos):
    obs = []
    roles = assign_roles_softmax(agents, ball_pos)

    for pid, agent in agents.items():
        dist = np.linalg.norm(agent["pos"] - ball_pos)
        angle = np.arctan2(*(ball_pos - agent["pos"]))
        vel = agent["velocity"]
        obs_vec = np.array([
            dist,
            angle,
            *vel,
            agent["stamina"],
            agent["last_action"],
            roles[pid]
        ])
        obs.append(obs_vec)
    return obs, roles

def generate_qmix_state(df, frame_id):
    frame_df = df[df["frame_id"] == frame_id]
    last_positions = {}  # Could load from cache if needed
    agents, ball = extract_agent_features(frame_df, last_positions)
    obs, _ = compute_observations(agents, ball)
    return np.concatenate(obs)

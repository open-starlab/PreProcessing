import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

from preprocessing.sports.SAR_data.soccer.state_preprocess.preprocess_frame import (
    frames2events_pvs,
    frames2events_edms,
)
from preprocessing.sports.SAR_data.soccer.state_preprocess.reward_model import RewardModelBase
from preprocessing.sports.SAR_data.soccer.utils.file_utils import load_json, save_as_jsonlines

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_single_game(
    game_dir: str,
    state: str,
    league: str,
    save_dir: str,
    config_path: str,
    match_id: str,
) -> None:
    """
    Preprocess one game's frames into events.jsonl for either PVS or EDMS.
    """
    save_dir = Path(save_dir)
    game_dir = Path(game_dir) / match_id
    config = load_json(config_path)

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
    else:
        raise ValueError(f"Unsupported state: {state}")

    out_path = save_dir / game_dir.name / "events.jsonl"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    save_as_jsonlines([e.to_dict() for e in events], out_path)

    logger.info(f"preprocessing finished... game_id: {game_dir.name} ({time.time() - start:.2f} sec)")


def generate_qmix_state(df: pd.DataFrame, frame_id: int) -> np.ndarray:
    """
    Produce a single global state vector for QMix at given frame_id.
    """
    from preprocessing.sports.SAR_data.soccer.soccer_SAR_processing import extract_agent_features
    from preprocessing.sports.SAR_data.soccer.soccer_SAR_state import compute_observations

    frame_df = df[df["frame_id"] == frame_id]
    last_positions = {}
    agents, ball = extract_agent_features(frame_df, last_positions)
    obs, _ = compute_observations(agents, ball)
    return np.concatenate(obs)


def process_frame_data(df: pd.DataFrame) -> list:
    """
    Convert cleaned RoboCup2D DataFrame into QMix‐compatible episodes.
    """
    from preprocessing.sports.SAR_data.soccer.soccer_SAR_processing import extract_agent_features
    from preprocessing.sports.SAR_data.soccer.soccer_SAR_state import compute_observations

    episodes = []
    last_positions = {}
    episode = {"obs": [], "state": [], "actions": [], "roles": [], "reward": [], "terminated": []}

    for frame_id, frame_df in df.groupby("frame_id"):
        agents, ball = extract_agent_features(frame_df, last_positions)
        obs, roles = compute_observations(agents, ball)
        state = np.concatenate(obs)

        episode["obs"].append(obs)
        episode["state"].append(state)
        episode["actions"].append([0] * len(agents))  # placeholder
        episode["roles"].append(roles)
        episode["reward"].append(0.0)
        episode["terminated"].append(False)

    # mark last timestep as terminal
    episode["terminated"][-1] = True
    episodes.append(episode)
    return episodes

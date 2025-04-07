import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.sports.SAR_data.soccer.utils.file_utils import load_json

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper / utility functions (can live in utils/data_utils.py)
# -----------------------------------------------------------------------------

def vector_velocity(curr: np.ndarray, prev: np.ndarray, fps: int) -> np.ndarray:
    """Return velocity vector given two XY points."""
    return (curr - prev) * fps


def vector_acceleration(curr_v: np.ndarray, prev_v: np.ndarray, fps: int) -> np.ndarray:
    """Return acceleration vector given two velocity vectors."""
    return (curr_v - prev_v) * fps


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec)) if vec is not None else 0.0


def convert_name(name: str) -> str:
    """Lower‑case, remove double‑spaces and accents."""
    import unicodedata
    if not isinstance(name, str):
        return ""
    name = " ".join(name.split())  # collapse whitespace
    name = unicodedata.normalize("NFKD", name.lower()).encode("ascii", "ignore").decode()
    return name


# -----------------------------------------------------------------------------
# Tracking processor
# -----------------------------------------------------------------------------

class TrackingProcessor:
    """Process SkillCorner tracking into DataStadium‑like CSVs."""

    def __init__(
        self,
        events_df: pd.DataFrame,
        players_df: pd.DataFrame,
        tracking_df: pd.DataFrame,
        metadata: pd.DataFrame,
        config: Dict[str, Any],
        save_dir: Path,
        fps: int = 10,
    ) -> None:
        self.events = events_df.copy()
        self.players = players_df.copy()
        self.tracking = tracking_df.copy()
        self.meta = metadata
        self.cfg = config
        self.fps = fps
        self.save_dir = Path(save_dir)
        self.game_id = str(self.events.loc[0, "match_id"])
        self.home_team_id = int(self.meta["home_team_id"].iloc[0])

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("[Tracking] start %s", self.game_id)
        self._split_ball_player()
        self._add_velocity_acceleration()
        self._export()
        logger.info("[Tracking] finished %s", self.game_id)

    # ------------------------------------------------------------------
    # Step 1: split raw SkillCorner JSON rows into tidy columns
    # ------------------------------------------------------------------

    def _split_ball_player(self) -> None:
        processed = []
        for _, row in self.tracking.iterrows():
            players = [e for e in row["data"] if e["track_id"] != 55]
            ball = next((e for e in row["data"] if e["track_id"] == 55), {"x": None, "y": None, "z": None})
            processed.append(
                {
                    "frame": row["frame"],
                    "period": row["period"],
                    "timestamp": row["timestamp"],
                    "player_data": players,
                    "ball_data": ball,
                }
            )
        self.tracking = pd.DataFrame(processed)
        self.tracking["GameID"] = self.game_id

    # ------------------------------------------------------------------
    # Step 2: add velocity / acceleration (vectorised where possible)
    # ------------------------------------------------------------------

    def _add_velocity_acceleration(self) -> None:
        prev_pos: Dict[int, np.ndarray] = {}
        prev_vel: Dict[int, np.ndarray] = {}
        prev_ball_pos: np.ndarray | None = None
        prev_ball_vel: np.ndarray | None = None

        for idx, row in self.tracking.iterrows():
            # players
            for p in row["player_data"]:
                pid = p["trackable_object"]
                curr = np.array([p["x"], p["y"]], dtype=float)
                vel_vec = np.zeros(2)
                acc_vec = np.zeros(2)
                if pid in prev_pos:
                    vel_vec = vector_velocity(curr, prev_pos[pid], self.fps)
                    if pid in prev_vel:
                        acc_vec = vector_acceleration(vel_vec, prev_vel[pid], self.fps)
                p["velocity"] = norm(vel_vec)
                p["acceleration"] = norm(acc_vec)
                prev_pos[pid] = curr
                prev_vel[pid] = vel_vec

            # ball
            ball = row["ball_data"]
            if ball["x"] is not None and ball["y"] is not None:
                curr_ball = np.array([ball["x"], ball["y"]], dtype=float)
                vel_vec = np.zeros(2)
                acc_vec = np.zeros(2)
                if prev_ball_pos is not None:
                    vel_vec = vector_velocity(curr_ball, prev_ball_pos, self.fps)
                    if prev_ball_vel is not None:
                        acc_vec = vector_acceleration(vel_vec, prev_ball_vel, self.fps)
                ball["velocity"] = norm(vel_vec)
                ball["acceleration"] = norm(acc_vec)
                prev_ball_pos = curr_ball
                prev_ball_vel = vel_vec
            else:
                ball["velocity"] = 0.0
                ball["acceleration"] = 0.0

    # ------------------------------------------------------------------
    # Step 3: export three CSVs (ball / player / players)
    # ------------------------------------------------------------------

    def _export(self) -> None:
        out_dir = self.save_dir / self.game_id
        out_dir.mkdir(parents=True, exist_ok=True)
        # ball.csv
        ball_df = self._make_ball_df()
        ball_df.to_csv(out_dir / "ball.csv", index=False)
        # player.csv
        player_df = self._make_player_df()
        player_df.to_csv(out_dir / "player.csv", index=False)
        # players.csv
        players_df = self._make_players_df()
        players_df.to_csv(out_dir / "players.csv", index=False)

    # ------------------------------------------------------------------
    # Helper – create each CSV
    # ------------------------------------------------------------------

    def _make_ball_df(self) -> pd.DataFrame:
        df = self.tracking[["GameID", "frame", "ball_data"]].copy()
        df.rename(columns={"frame": "Frame"}, inplace=True)
        df["HA"] = 0
        df["SysTarget"] = 0
        df["No"] = 0
        df["X"] = df["ball_data"].apply(lambda b: b.get("x"))
        df["Y"] = df["ball_data"].apply(lambda b: b.get("y"))
        df["Speed"] = df["ball_data"].apply(lambda b: b.get("velocity"))
        df["Acceleration"] = df["ball_data"].apply(lambda b: b.get("acceleration"))
        df.drop(columns="ball_data", inplace=True)
        return df

    def _make_player_df(self) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        jersey_lookup = {
            p["trackable_object"]: (1 if p["team_id"] == self.home_team_id else 2, p["jersey_number"])
            for _, p in self.players.iterrows()
            if not pd.isna(p["trackable_object"])
        }
        for _, row in tqdm(self.tracking.iterrows(), total=len(self.tracking)):
            for p in row["player_data"]:
                ha, no = jersey_lookup.get(p["trackable_object"], (0, 0))
                records.append(
                    {
                        "GameID": row["GameID"],
                        "Frame": row["frame"],
                        "HA": ha,
                        "SysTarget": None,
                        "No": no,
                        "x": p["x"],
                        "y": p["y"],
                        "Speed": p["velocity"],
                        "Acceleration": p["acceleration"],
                    }
                )
        return pd.DataFrame(records)

    def _make_players_df(self) -> pd.DataFrame:
        df_players = pd.DataFrame()
        df_players["試合ID"] = [self.game_id] * len(self.players)
        df_players["ホームアウェイF"] = (self.players["team_id"] == self.home_team_id).map({True: 1, False: 2})
        df_players["チームID"] = self.players["team_id"].fillna(0).astype(int)
        df_players["チーム名"] = self.players["team"].fillna("")
        df_players["試合ポジションID"] = self.players["position_group"].map(self.cfg["position_role_id"]).fillna(0).astype(int)
        df_players["背番号"] = self.players["jersey_number"].fillna(0).astype(int)
        df_players["選手ID"] = self.players["player_id"].fillna(0).astype(int)
        df_players["選手名"] = self.players["name"].apply(convert_name)
        df_players["出場"] = (~self.players["start_time"].isna()).astype(int)
        df_players["スタメン"] = (self.players["start_time"] == "00:00:00").astype(int)
        df_players["出場時間"] = self.players.apply(lambda r: 0 if pd.isna(r["start_time"]) else 90, axis=1)
        df_players["実出場時間"] = df_players["出場時間"] * 60
        df_players["身長"] = self.players["height"].fillna(0).astype(int)
        return df_players


# -----------------------------------------------------------------------------
# Entry function called by external script / CLI
# -----------------------------------------------------------------------------

def process_single_file(
    events_df: pd.DataFrame,
    players_df: pd.DataFrame,
    tracking_dir: Path | str,
    metadata_df: pd.DataFrame,
    config_path: Path | str,
    match_id: str,
    save_dir: Path | str,
):
    """Wrapper used by multiprocessing or CLI."""
    cfg = load_json(config_path)
    tracking_path = Path(tracking_dir) / f"{match_id}.json"
    with open(tracking_path, "r", encoding="utf-8") as fp:
        tracking_df = pd.DataFrame(json.load(fp))

    TrackingProcessor(
        events_df,
        players_df,
        tracking_df,
        metadata_df,
        cfg,
        save_dir,
    ).run()

    logger.info("Finished processing %s", match_id)

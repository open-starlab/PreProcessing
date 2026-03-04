from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple
import numpy as np

# Keep this module lightweight: avoid importing the full SAR preprocessing stack (and itsoptional dependencies) just to read precomputed `events.jsonl` outputs.
FIELD_LENGTH = 105.0  # meters
FIELD_WIDTH = 68.0  # meters

logger = logging.getLogger(__name__)


SplitName = Literal["train", "val", "test"]
SplitBy = Literal["game_id", "sequence_id"]


DEFAULT_ACTION_TO_ID: Dict[str, int] = {
    # Off-ball movement (direction discretization).
    "idle": 0,
    "right": 1,
    "up_right": 2,
    "up": 3,
    "up_left": 4,
    "left": 5,
    "down_left": 6,
    "down": 7,
    "down_right": 8,
    # On-ball.
    "pass": 9,
    "cross": 10,
    "dribble": 11,
    "shot": 12,
    "through_pass": 13,
    # Defensive / other (single bucket for compatibility with existing 0-14 action space).
    "pressure": 14,
    "ball_recovery": 14,
    "interception": 14,
    "clearance": 14,
    # Aliases.
    "goal": 12,
}

DEFAULT_PAD_ACTION_ID = 15
DEFAULT_VOCAB_SIZE = 16

_DIRECTION_ACTIONS = {
    "idle",
    "right",
    "up_right",
    "up",
    "up_left",
    "left",
    "down_left",
    "down",
    "down_right",
}

_ONBALL_ACTIONS = {"pass", "cross", "dribble", "shot", "through_pass", "goal"}


def _safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _load_jsonlines(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _save_jsonlines(items: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _save_formatted_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def _stable_uniform_01(key: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    # Use 64 bits for stable mapping.
    v = int(h[:16], 16)
    return v / float(2**64 - 1)


def _assign_split(key: str, seed: int, ratios: Tuple[float, float, float]) -> SplitName:
    train_r, val_r, test_r = ratios
    if train_r < 0 or val_r < 0 or test_r < 0 or not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")
    u = _stable_uniform_01(key, seed=seed)
    if u < train_r:
        return "train"
    if u < train_r + val_r:
        return "val"
    return "test"


def _extract_raw_state(event_state: Mapping[str, Any]) -> Mapping[str, Any]:
    # PVS: state == raw state; EDMS: state["raw_state"] contains player/ball.
    if "raw_state" in event_state and isinstance(event_state["raw_state"], Mapping):
        return event_state["raw_state"]  # type: ignore[return-value]
    return event_state


def _is_goalkeeper(player: Mapping[str, Any]) -> bool:
    role = str(player.get("player_role", "") or "").strip().upper()
    return role in {"GK", "GOALKEEPER"}


def _get_attack_players(raw_state: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    players = raw_state.get("attack_players")
    if not isinstance(players, list):
        raise KeyError("state.attack_players (or state.raw_state.attack_players) is missing or not a list")
    return [p for p in players if isinstance(p, Mapping)]


def _get_ball(raw_state: Mapping[str, Any]) -> Mapping[str, Any]:
    ball = raw_state.get("ball")
    if not isinstance(ball, Mapping):
        raise KeyError("state.ball (or state.raw_state.ball) is missing or not a mapping")
    return ball


def _get_pos_xy(entity: Mapping[str, Any]) -> Tuple[float, float]:
    pos = entity.get("position", {})
    if not isinstance(pos, Mapping):
        pos = {}
    return _safe_float(pos.get("x")), _safe_float(pos.get("y"))


def _get_vel_xy(entity: Mapping[str, Any]) -> Tuple[float, float]:
    vel = entity.get("velocity", {})
    if not isinstance(vel, Mapping):
        vel = {}
    return _safe_float(vel.get("x")), _safe_float(vel.get("y"))


def _action_to_id(action: Any, action_to_id: Mapping[str, int], default_id: int) -> int:
    a = str(action or "").strip()
    if not a:
        return default_id
    a = a.lower()
    return int(action_to_id.get(a, default_id))


def _select_agent_ids(
    attack_players_t0: Sequence[Mapping[str, Any]],
    n_agents: int,
    *,
    strict: bool,
) -> List[int]:
    outfield = [p for p in attack_players_t0 if not _is_goalkeeper(p)]
    ids = []
    for p in outfield:
        pid = p.get("player_id")
        if pid is None:
            continue
        try:
            pid_i = int(pid)
        except (TypeError, ValueError):
            continue
        if pid_i < 0:
            continue
        ids.append(pid_i)

    ids = sorted(set(ids))
    if strict and len(ids) != n_agents:
        raise ValueError(f"Expected exactly {n_agents} outfield attackers, got {len(ids)} (ids={ids})")
    if len(ids) < n_agents:
        raise ValueError(f"Not enough attackers to build {n_agents} agents (got {len(ids)})")
    return ids[:n_agents]


def _infer_onball_mask(
    attackers_by_id: Mapping[int, Mapping[str, Any]],
    agent_ids: Sequence[int],
    *,
    ball_xy: Tuple[float, float],
    onball_distance_threshold: float,
) -> np.ndarray:
    """
    Returns (N,) bool mask.

    Heuristic:
    1) If exactly one agent has an explicit on-ball action (pass/cross/dribble/shot/through_pass/goal), use it.
    2) Else, use closest-to-ball agent if within threshold.
    3) Else, all False (ball in transit / unknown).
    """
    n_agents = len(agent_ids)
    explicit = np.zeros((n_agents,), dtype=np.bool_)
    for i, pid in enumerate(agent_ids):
        a = str(attackers_by_id.get(pid, {}).get("action", "") or "").lower()
        if a in _ONBALL_ACTIONS:
            explicit[i] = True

    if explicit.sum() == 1:
        return explicit

    bx, by = ball_xy
    if not np.isfinite(bx) or not np.isfinite(by):
        return np.zeros((n_agents,), dtype=np.bool_)

    dists = np.full((n_agents,), np.inf, dtype=np.float32)
    for i, pid in enumerate(agent_ids):
        p = attackers_by_id.get(pid)
        if p is None:
            continue
        px, py = _get_pos_xy(p)
        dists[i] = float(math.hypot(bx - px, by - py))

    if not np.isfinite(dists).any():
        return np.zeros((n_agents,), dtype=np.bool_)

    j = int(np.nanargmin(dists))
    if float(dists[j]) <= float(onball_distance_threshold):
        m = np.zeros((n_agents,), dtype=np.bool_)
        m[j] = True
        return m
    return np.zeros((n_agents,), dtype=np.bool_)


def _build_obs_raw_pvs_basic(
    player: Mapping[str, Any],
    ball: Mapping[str, Any],
    *,
    onball: bool,
    vel_scale: float = 10.0,
) -> np.ndarray:
    px, py = _get_pos_xy(player)
    pvx, pvy = _get_vel_xy(player)
    bx, by = _get_pos_xy(ball)
    bvx, bvy = _get_vel_xy(ball)

    # Normalize positions to roughly [-1, 1] under origin_pos="center" conventions.
    px_n = px / (FIELD_LENGTH / 2)
    py_n = py / (FIELD_WIDTH / 2)
    bx_n = bx / (FIELD_LENGTH / 2)
    by_n = by / (FIELD_WIDTH / 2)

    pvx_n = pvx / vel_scale
    pvy_n = pvy / vel_scale
    bvx_n = bvx / vel_scale
    bvy_n = bvy / vel_scale

    dx = bx - px
    dy = by - py
    dx_n = dx / (FIELD_LENGTH / 2)
    dy_n = dy / (FIELD_WIDTH / 2)
    dist = float(math.hypot(dx, dy))
    dist_n = dist / FIELD_LENGTH

    ang = float(math.atan2(dy, dx)) if dist > 0 else 0.0
    sin_a = float(math.sin(ang))
    cos_a = float(math.cos(ang))

    return np.asarray(
        [
            px_n,
            py_n,
            pvx_n,
            pvy_n,
            bx_n,
            by_n,
            bvx_n,
            bvy_n,
            dx_n,
            dy_n,
            dist_n,
            sin_a,
            cos_a,
            1.0 if onball else 0.0,
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class SoccerRLDatasetConfig:
    n_agents: int = 10
    max_seq_len: int = 600
    min_seq_len: int = 2
    pad_action_id: int = DEFAULT_PAD_ACTION_ID
    vocab_size: int = DEFAULT_VOCAB_SIZE
    action_to_id: Mapping[str, int] = None  # type: ignore[assignment]
    unknown_action_id: int = 0
    strict_agents: bool = True
    onball_distance_threshold: float = 2.0
    split_by: SplitBy = "game_id"
    split_seed: int = 1337
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    truncate: Literal["tail", "head"] = "tail"

    def __post_init__(self) -> None:
        if self.action_to_id is None:
            object.__setattr__(self, "action_to_id", DEFAULT_ACTION_TO_ID)
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if self.min_seq_len <= 0:
            raise ValueError("min_seq_len must be > 0")
        if not (0 <= self.pad_action_id < self.vocab_size):
            raise ValueError("pad_action_id must be in [0, vocab_size)")


def _iter_event_files(sar_preprocessed_dir: Path) -> Iterable[Path]:
    if sar_preprocessed_dir.is_file():
        yield sar_preprocessed_dir
        return
    yield from sorted(sar_preprocessed_dir.glob("**/events.jsonl"))


def _convert_sequence(
    seq: Mapping[str, Any],
    *,
    config: SoccerRLDatasetConfig,
) -> Optional[Tuple[SplitName, Dict[str, Any]]]:
    events = seq.get("events")
    if not isinstance(events, list) or not events:
        return None

    game_id = str(seq.get("game_id", "") or "")
    half = str(seq.get("half", "") or "")
    sequence_id = str(seq.get("sequence_id", "") or "")
    seq_uid = f"{game_id}:{half}:{sequence_id}"

    # Determine split group key.
    if config.split_by == "game_id":
        split_key = game_id
    else:
        split_key = seq_uid

    split = _assign_split(split_key, seed=config.split_seed, ratios=config.split_ratios)

    # Truncate / filter.
    if len(events) < config.min_seq_len:
        return None
    if len(events) > config.max_seq_len:
        if config.truncate == "tail":
            events = events[-config.max_seq_len :]
        else:
            events = events[: config.max_seq_len]

    t_steps = len(events)
    # Agent IDs derived from first timestep.
    state0 = events[0].get("state")
    if not isinstance(state0, Mapping):
        return None
    raw0 = _extract_raw_state(state0)
    attack_players0 = _get_attack_players(raw0)
    try:
        agent_ids = _select_agent_ids(attack_players0, config.n_agents, strict=config.strict_agents)
    except ValueError as e:
        logger.debug(f"Skipping sequence {seq_uid}: {e}")
        return None

    obs_dim = 14  # _build_obs_raw_pvs_basic
    obs = np.zeros((config.max_seq_len, config.n_agents, obs_dim), dtype=np.float32)
    actions = np.full((config.max_seq_len, config.n_agents), config.pad_action_id, dtype=np.int64)
    onball_mask = np.zeros((config.max_seq_len, config.n_agents), dtype=np.bool_)
    rewards = np.zeros((config.max_seq_len,), dtype=np.float32)
    dones = np.zeros((config.max_seq_len,), dtype=np.float32)
    mask = np.zeros((config.max_seq_len,), dtype=np.float32)

    for t in range(t_steps):
        ev = events[t]
        if not isinstance(ev, Mapping):
            return None
        st = ev.get("state")
        if not isinstance(st, Mapping):
            return None
        raw = _extract_raw_state(st)

        ball = _get_ball(raw)
        bx, by = _get_pos_xy(ball)

        attack_players = _get_attack_players(raw)
        attackers_by_id: Dict[int, Mapping[str, Any]] = {}
        for p in attack_players:
            if _is_goalkeeper(p):
                continue
            pid = p.get("player_id")
            try:
                pid_i = int(pid)
            except (TypeError, ValueError):
                continue
            attackers_by_id[pid_i] = p

        if config.strict_agents and any(pid not in attackers_by_id for pid in agent_ids):
            logger.debug(f"Skipping sequence {seq_uid}: missing agent at t={t}")
            return None

        onball_t = _infer_onball_mask(
            attackers_by_id,
            agent_ids,
            ball_xy=(bx, by),
            onball_distance_threshold=config.onball_distance_threshold,
        )
        onball_mask[t, :] = onball_t

        for j, pid in enumerate(agent_ids):
            p = attackers_by_id.get(pid)
            if p is None:
                continue
            a_id = _action_to_id(p.get("action"), config.action_to_id, default_id=config.unknown_action_id)
            actions[t, j] = a_id
            obs[t, j, :] = _build_obs_raw_pvs_basic(p, ball, onball=bool(onball_t[j]))

        rewards[t] = _safe_float(ev.get("reward"), default=0.0)
        mask[t] = 1.0

    dones[t_steps - 1] = 1.0

    sample: Dict[str, Any] = {
        "id": seq_uid,
        "game_id": game_id,
        "half": half,
        "sequence_id": sequence_id,
        "seq_len": int(t_steps),
        "agent_ids": agent_ids,
        "observation": obs,
        "action": actions,
        "reward": rewards,
        "done": dones,
        "mask": mask,
        "onball_mask": onball_mask,
    }
    return split, sample


def build_rl_datasets_from_sar_events(
    sar_preprocessed_dir: str | Path,
    output_dir: str | Path,
    *,
    config: Optional[SoccerRLDatasetConfig] = None,
) -> None:
    """
    Builds a single shared, multi-agent (N=10 attackers) dataset that can be consumed by
    both DQN (by flattening N into the batch dimension) and QMIX (as-is).

    Input:
      - A directory produced by SAR preprocessing containing one or more `events.jsonl` files,
        where each line is a sequence (attack segment) with an `events` list.

    Output (written under output_dir):
      - meta.json
      - train.npz / val.npz / test.npz
      - train_manifest.jsonl / val_manifest.jsonl / test_manifest.jsonl
    """
    cfg = config or SoccerRLDatasetConfig()
    sar_dir = Path(sar_preprocessed_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: Dict[SplitName, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    n_sequences_total = 0
    n_sequences_kept = 0
    for events_path in _iter_event_files(sar_dir):
        sequences = _load_jsonlines(events_path)
        for seq in sequences:
            n_sequences_total += 1
            if not isinstance(seq, Mapping):
                continue
            converted = _convert_sequence(seq, config=cfg)
            if converted is None:
                continue
            split, sample = converted
            splits[split].append(sample)
            n_sequences_kept += 1

    if n_sequences_kept == 0:
        raise RuntimeError(f"No sequences were converted from {sar_dir}")

    meta = {
        "source_dir": str(sar_dir),
        "n_sequences_total": n_sequences_total,
        "n_sequences_kept": n_sequences_kept,
        "config": {
            "n_agents": cfg.n_agents,
            "max_seq_len": cfg.max_seq_len,
            "min_seq_len": cfg.min_seq_len,
            "pad_action_id": cfg.pad_action_id,
            "vocab_size": cfg.vocab_size,
            "unknown_action_id": cfg.unknown_action_id,
            "strict_agents": cfg.strict_agents,
            "onball_distance_threshold": cfg.onball_distance_threshold,
            "split_by": cfg.split_by,
            "split_seed": cfg.split_seed,
            "split_ratios": cfg.split_ratios,
            "truncate": cfg.truncate,
        },
        "action_to_id": dict(cfg.action_to_id),
        "observation_schema": {
            "name": "raw_pvs_basic",
            "dim": 14,
            "features": [
                "player_x_norm",
                "player_y_norm",
                "player_vx_norm",
                "player_vy_norm",
                "ball_x_norm",
                "ball_y_norm",
                "ball_vx_norm",
                "ball_vy_norm",
                "rel_ball_x_norm",
                "rel_ball_y_norm",
                "dist_ball_norm",
                "angle_ball_sin",
                "angle_ball_cos",
                "onball",
            ],
        },
    }
    _save_formatted_json(meta, out_dir / "meta.json")

    for split_name, samples in splits.items():
        if not samples:
            logger.warning(f"No samples for split={split_name}")
            continue

        obs = np.stack([s["observation"] for s in samples], axis=0)
        actions = np.stack([s["action"] for s in samples], axis=0)
        rewards = np.stack([s["reward"] for s in samples], axis=0)
        dones = np.stack([s["done"] for s in samples], axis=0)
        mask = np.stack([s["mask"] for s in samples], axis=0)
        onball_mask = np.stack([s["onball_mask"] for s in samples], axis=0)
        seq_len = np.asarray([s["seq_len"] for s in samples], dtype=np.int32)

        np.savez_compressed(
            out_dir / f"{split_name}.npz",
            observation=obs,
            action=actions,
            reward=rewards,
            done=dones,
            mask=mask,
            onball_mask=onball_mask,
            seq_len=seq_len,
        )

        manifest = []
        for i, s in enumerate(samples):
            manifest.append(
                {
                    "index": i,
                    "id": s["id"],
                    "game_id": s["game_id"],
                    "half": s["half"],
                    "sequence_id": s["sequence_id"],
                    "seq_len": s["seq_len"],
                    "agent_ids": s["agent_ids"],
                }
            )
        _save_jsonlines(manifest, out_dir / f"{split_name}_manifest.jsonl")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build DQN/QMIX datasets from SAR `events.jsonl` outputs.")
    p.add_argument("--sar-preprocessed-dir", type=str, required=True, help="Directory containing `**/events.jsonl`.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for .npz + meta.json.")
    p.add_argument("--max-seq-len", type=int, default=600)
    p.add_argument("--min-seq-len", type=int, default=2)
    p.add_argument("--n-agents", type=int, default=10)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--pad-action-id", type=int, default=DEFAULT_PAD_ACTION_ID)
    p.add_argument("--unknown-action-id", type=int, default=0)
    p.add_argument(
        "--action-map-json",
        type=str,
        default=None,
        help="Optional JSON file mapping action strings to ids (keys are matched case-insensitively).",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--split-by", type=str, choices=["game_id", "sequence_id"], default="game_id")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--onball-threshold", type=float, default=2.0)
    p.add_argument("--strict-agents", dest="strict_agents", action="store_true", help="Require exactly N attackers.")
    p.add_argument("--no-strict-agents", dest="strict_agents", action="store_false", help="Allow missing attackers.")
    p.set_defaults(strict_agents=True)
    p.add_argument("--truncate", type=str, choices=["tail", "head"], default="tail")
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    action_map: Optional[Dict[str, int]] = None
    if args.action_map_json:
        with open(args.action_map_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("--action-map-json must contain a JSON object of {str: int}")
        action_map = {}
        for k, v in raw.items():
            if k is None:
                continue
            try:
                action_map[str(k).strip().lower()] = int(v)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid action map entry: {k} -> {v}") from None

    cfg = SoccerRLDatasetConfig(
        n_agents=int(args.n_agents),
        max_seq_len=int(args.max_seq_len),
        min_seq_len=int(args.min_seq_len),
        pad_action_id=int(args.pad_action_id),
        vocab_size=int(args.vocab_size),
        action_to_id=action_map or DEFAULT_ACTION_TO_ID,
        unknown_action_id=int(args.unknown_action_id),
        split_by=args.split_by,  # type: ignore[arg-type]
        split_seed=int(args.seed),
        split_ratios=(float(args.train_ratio), float(args.val_ratio), float(args.test_ratio)),
        onball_distance_threshold=float(args.onball_threshold),
        strict_agents=bool(args.strict_agents),
        truncate=args.truncate,  # type: ignore[arg-type]
    )
    build_rl_datasets_from_sar_events(args.sar_preprocessed_dir, args.output_dir, config=cfg)


if __name__ == "__main__":
    main()

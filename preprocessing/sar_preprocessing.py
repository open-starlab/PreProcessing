"""
SAR (State-Action-Reward) Preprocessing Module
===============================================

Converts raw tracking/event data into explicit (State, Action, Reward) tuples
for Inverse Reinforcement Learning and Behavioral Cloning.

State (21-dim):   [8 defenders × 2 coords, ball × 2 coords, attack direction × 3]
Action (1):       {0:backward, 1:forward, 2:compress, 3:expand}
Reward (4-dim):   [space_score, stretch_index, pressure_index, line_height_rel]

Author: Defense Line Analysis Pipeline
Date: 2025
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any
from scipy.spatial import ConvexHull


# ============================================================================
# ATTACK DIRECTION COMPUTATION
# ============================================================================

def compute_attack_direction(state: Dict) -> Tuple[float, int, int]:
    """
    Compute attacking team direction vector.
    
    Args:
        state: Match state dictionary with 'players' and 'ball' keys
    
    Returns:
        Tuple of:
            - direction_angle (0-360°): angle of attack centroid
            - attack_intensity (int): number of attackers in final third
            - attack_zone (int): 0=left, 1=center, 2=right
    
    Example:
        >>> state = {"players": [...], "ball": {"position": {"x": 50, "y": 50}}}
        >>> angle, intensity, zone = compute_attack_direction(state)
        >>> print(f"Attack at {angle:.1f}° with {intensity} attackers in zone {zone}")
    """
    from feature_computation import _is_atk
    
    players = state.get("players", [])
    attackers = [p for p in players if _is_atk(p)]
    
    if not attackers:
        return 0.0, 0, 1  # No attack - default to center
    
    # Extract attacker positions
    att_positions = []
    for a in attackers:
        pos = a.get("position", {})
        if isinstance(pos, dict):
            try:
                x = float(pos.get("x", 0))
                y = float(pos.get("y", 0))
                att_positions.append((x, y))
            except:
                pass
    
    if not att_positions:
        return 0.0, 0, 1
    
    att_x = np.array([p[0] for p in att_positions])
    att_y = np.array([p[1] for p in att_positions])
    
    # Attack centroid (assumes 100m × 100m field)
    centroid_x = np.mean(att_x)
    centroid_y = np.mean(att_y)
    
    # Direction vector: from center field (50, 50) to centroid
    dx = centroid_x - 50.0
    dy = centroid_y - 50.0
    
    # Angle in degrees (0°=right, 90°=down, 180°=left, 270°=up)
    direction_angle = np.degrees(np.arctan2(dy, dx))
    if direction_angle < 0:
        direction_angle += 360.0
    
    # Attack intensity: count attackers in final third (x > 66)
    attack_intensity = int(np.sum(att_x > 66))
    
    # Attack zone based on y-position (left/center/right)
    if centroid_y < 33:
        attack_zone = 0  # Left wing
    elif centroid_y > 67:
        attack_zone = 2  # Right wing
    else:
        attack_zone = 1  # Central corridor
    
    return float(direction_angle), attack_intensity, attack_zone


# ============================================================================
# STATE FEATURE CONSTRUCTION (21-DIM)
# ============================================================================

def build_full_state_features(state: Dict) -> np.ndarray:
    """
    Build 21-dimensional state feature vector.
    
    Structure:
        [x1, y1, ..., x8, y8,  # 8 defenders × 2 coords (16-dim)
         ball_x, ball_y,        # ball position (2-dim)
         attack_dir,            # attack direction angle (1-dim)
         attack_intensity,      # attackers in final third (1-dim)
         attack_zone]           # left/center/right (1-dim)
    
    Args:
        state: Match state dictionary
    
    Returns:
        numpy array of shape (21,)
    """
    from feature_computation import build_step_features
    
    # Part 1: Defender positions + ball (18-dim from existing function)
    base_feats = build_step_features(state)
    
    # Part 2: Attack direction (3-dim)
    att_dir, att_int, att_zone = compute_attack_direction(state)
    attack_feats = [att_dir, float(att_int), float(att_zone)]
    
    # Combine: [defenders×2 + ball×2 + attack×3] = 21-dim
    full_state = np.array(base_feats + attack_feats, dtype=np.float32)
    
    assert len(full_state) == 21, f"Expected 21-dim state, got {len(full_state)}"
    return full_state


# ============================================================================
# ACTION EXTRACTION (MAJORITY VOTE)
# ============================================================================

ACTION_MAP = {"backward": 0, "forward": 1, "compress": 2, "expand": 3}
ACTION_NAMES = {0: "backward", 1: "forward", 2: "compress", 3: "expand"}


def extract_majority_action(defender_actions: List[Dict]) -> Tuple[int, str]:
    """
    Extract majority-vote action from individual defender actions.
    
    Args:
        defender_actions: List of dicts with 'action' key
    
    Returns:
        Tuple of (action_index, action_label)
        
    Example:
        >>> defender_actions = [{"action": "forward"}, {"action": "forward"}, {"action": "backward"}]
        >>> idx, label = extract_majority_action(defender_actions)
        >>> print(idx, label)
        1 forward
    """
    if not defender_actions:
        return ACTION_MAP["backward"], "backward"
    
    action_labels = [a.get("action", "backward") for a in defender_actions]
    maj_action = Counter(action_labels).most_common(1)[0][0]
    action_idx = ACTION_MAP.get(maj_action, ACTION_MAP["backward"])
    
    return action_idx, maj_action


# ============================================================================
# REWARD FEATURE EXTRACTION (4-DIM)
# ============================================================================

def extract_reward_features(state: Dict) -> np.ndarray:
    """
    Extract 4-dimensional reward features from state.
    
    Features:
        1. space_score: Zone-weighted attacker space dominance
        2. stretch_index: Convex hull area of defensive line
        3. pressure_index: Count of pressured attackers
        4. line_height_rel_ball: Relative positioning to ball
    
    Args:
        state: Match state dictionary
    
    Returns:
        numpy array of shape (4,) with reward features
    """
    from feature_computation import (
        _is_def,
        _is_atk,
        _extract_xy,
        get_ball_x,
        compute_space_score,
        compute_stretch_index,
        compute_pressure_index,
        compute_line_height_features
    )
    
    players = state.get("players", [])
    defenders_players = [p for p in players if _is_def(p)]
    attackers_players = [p for p in players if _is_atk(p)]

    defenders = []
    for p in defenders_players:
        x, y = _extract_xy(p)
        if x is not None and y is not None:
            defenders.append((float(x), float(y)))

    attackers = []
    for p in attackers_players:
        x, y = _extract_xy(p)
        if x is not None and y is not None:
            attackers.append((float(x), float(y)))

    ball_x = get_ball_x(state)
    ball_pos = state.get("ball", {}).get("position", {})
    ball_tuple = None
    if isinstance(ball_pos, dict):
        ball_tuple = (float(ball_pos.get("x", 0.0)), float(ball_pos.get("y", 0.0)))

    space_score = compute_space_score(defenders, attackers, ball_tuple)
    stretch_index = compute_stretch_index(defenders_players)
    pressure_index = compute_pressure_index(defenders, attackers, radius=3.0)
    line_height_rel_ball = compute_line_height_features(defenders, ball_tuple).get("line_height_relative", 0.0)
    
    reward = np.array(
        [space_score, stretch_index, float(pressure_index), line_height_rel_ball],
        dtype=np.float32
    )
    
    assert len(reward) == 4, f"Expected 4-dim reward, got {len(reward)}"
    return reward


# ============================================================================
# SAR TUPLE CREATION
# ============================================================================

class SARTuple:
    """Single (State, Action, Reward) transition."""
    
    def __init__(
        self,
        state: np.ndarray,
        action: int,
        action_label: str,
        reward: np.ndarray,
        metadata: Dict = None
    ):
        """
        Args:
            state: 21-dim state vector
            action: action index (0-3)
            action_label: action name (backward/forward/compress/expand)
            reward: 4-dim reward vector
            metadata: additional info (ball_pos, attack_dir, etc.)
        """
        self.state = state
        self.action = action
        self.action_label = action_label
        self.reward = reward
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/analysis."""
        return {
            "state": self.state.tolist(),
            "action": self.action,
            "action_label": self.action_label,
            "reward": self.reward.tolist(),
            **self.metadata
        }
    
    def __repr__(self):
        return (
            f"SARTuple(action={self.action_label}, "
            f"space_score={self.reward[0]:.3f}, "
            f"stretch={self.reward[1]:.3f})"
        )


# ============================================================================
# SEQUENCE BUILDING (SAR TRAJECTORIES)
# ============================================================================

def create_sar_sequences(
    events_list: List[Dict],
    sequence_length: int = 10,
    skip_initial: int = 1,
    overlap: int = 0
) -> List[Dict]:
    """
    Build sequences as explicit (State, Action, Reward) tuples.
    
    Creates non-overlapping sequences of defensive states with corresponding
    actions and reward features. Each sequence is labeled as success/failure
    based on defensive outcome.
    
    Args:
        events_list: List of event dictionaries with 'state' key
        sequence_length: Length of sequence (default: 10 frames)
        skip_initial: Skip this many initial events (default: 1)
        overlap: Frame overlap between sequences (default: 0 = no overlap)
    
    Returns:
        List of sequence dictionaries with keys:
            - start_idx: Starting event index
            - sar_tuples: List of SARTuple objects
            - outcome: 'success' or 'failure'
            - length: Number of valid SAR tuples
    
    Example:
        >>> sequences = create_sar_sequences(events_list, sequence_length=10)
        >>> for seq in sequences:
        >>>     print(f"Sequence: {seq['outcome']} with {seq['length']} steps")
        >>>     for sar in seq['sar_tuples']:
        >>>         print(f"  {sar.action_label}: reward={sar.reward}")
    """
    from feature_computation import classify_defensive_action, _is_def
    from reward_features import get_ball_x, get_ball_y
    
    sequences = []
    step = sequence_length - overlap
    
    for start in range(skip_initial, len(events_list) - sequence_length + 1, step):
        prev_state = None
        sar_tuples = []
        window_events = events_list[start:start+sequence_length]
        
        for ev in window_events:
            st = ev.get("state", {})
            if not isinstance(st, dict):
                continue
            
            # === BUILD STATE (21-dim) ===
            try:
                full_state = build_full_state_features(st)
            except Exception as e:
                print(f"[WARNING] Failed to build state features: {e}")
                continue
            
            # === CLASSIFY ACTIONS ===
            defenders = [p for p in st.get("players", []) if _is_def(p)]
            defender_actions = []
            
            for d in defenders:
                action_label = classify_defensive_action(d, st, prev_state)
                defender_actions.append({
                    "player_id": d.get("player_id") or d.get("id"),
                    "action": action_label,
                    "position": d.get("position", {})
                })
            
            action_idx, action_label = extract_majority_action(defender_actions)
            
            # === EXTRACT REWARD (4-dim) ===
            try:
                reward = extract_reward_features(st)
            except Exception as e:
                print(f"[WARNING] Failed to compute reward features: {e}")
                continue
            
            # === CREATE SAR TUPLE ===
            att_dir, att_int, att_zone = compute_attack_direction(st)
            metadata = {
                "ball_position": {"x": get_ball_x(st), "y": get_ball_y(st)},
                "attack_direction": att_dir,
                "attack_intensity": att_int,
                "attack_zone": att_zone,
                "num_defenders": len(defenders)
            }
            
            sar = SARTuple(
                state=full_state,
                action=action_idx,
                action_label=action_label,
                reward=reward,
                metadata=metadata
            )
            sar_tuples.append(sar)
            prev_state = st
        
        # === DETERMINE OUTCOME ===
        outcome = "failure" if any(e.get("label", 1) == 0 for e in window_events) else "success"
        
        if sar_tuples:  # Only add non-empty sequences
            first_event = window_events[0] if window_events else {}
            last_event = window_events[-1] if window_events else {}
            default_label = 0 if outcome == "failure" else 1
            sequences.append({
                "sequence_id": len(sequences),
                "start_idx": start,
                "end_idx": start + sequence_length,
                "start_frame": first_event.get("frame_id"),
                "end_frame": last_event.get("frame_id"),
                "match_id": first_event.get("match_id"),
                "team_id": first_event.get("team_id"),
                "home_team": first_event.get("home_team"),
                "away_team": first_event.get("away_team"),
                "label": first_event.get("label", default_label),
                "sar_tuples": sar_tuples,
                "outcome": outcome,
                "length": len(sar_tuples),
                "description": f"Seq[{start}:{start+sequence_length}] {outcome} ({len(sar_tuples)} steps)"
            })
    
    return sequences


# ============================================================================
# DATASET EXTRACTION FOR ML
# ============================================================================

def extract_ml_arrays(sequences: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract state, action, reward arrays from SAR sequences (FLATTENED).
    
    Args:
        sequences: List of sequences from create_sar_sequences()
    
    Returns:
        Tuple of:
            - X: State array, shape (N_transitions, 21)
            - A: Action array, shape (N_transitions,)
            - R: Reward array, shape (N_transitions, 4)
    
    Example:
        >>> X, A, R = extract_ml_arrays(sequences)
        >>> print(f"Dataset: {len(X)} transitions")
        >>> X_train, A_train, R_train = X[:800], A[:800], R[:800]
    """
    X, A, R = [], [], []
    
    for seq in sequences:
        for sar in seq["sar_tuples"]:
            X.append(sar.state)
            A.append(sar.action)
            R.append(sar.reward)
    
    return (
        np.array(X, dtype=np.float32),
        np.array(A, dtype=np.int64),
        np.array(R, dtype=np.float32)
    )


def extract_ml_sequences(sequences: List[Dict], seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Extract state, action, reward arrays from SAR sequences (STRUCTURED).
    
    Preserves sequence structure for SAR dataset output:
    states  → (N_sequences, seq_len, 21)
    actions → (N_sequences, seq_len)
    rewards → (N_sequences, seq_len, 4)
    
    Args:
        sequences: List of sequences from create_sar_sequences()
        seq_len: Expected sequence length (default 10)
    
    Returns:
        Tuple of:
            - X: State array, shape (N_sequences, seq_len, 21)
            - A: Action array, shape (N_sequences, seq_len)
            - R: Reward array, shape (N_sequences, seq_len, 4)
            - metadata: Dict with summary stats
    
    Example:
        >>> sequences = create_sar_sequences(events, sequence_length=10)
        >>> X, A, R, meta = extract_ml_sequences(sequences, seq_len=10)
        >>> print(f"SAR dataset: {X.shape[0]} sequences of length {X.shape[1]}")
    """
    X_seqs, A_seqs, R_seqs = [], [], []
    outcomes = []
    sequence_metadata = []
    
    for seq in sequences:
        sar_tuples = seq.get("sar_tuples", [])
        outcome = seq.get("outcome", "unknown")
        
        # Pad or truncate to seq_len
        n_steps = len(sar_tuples)
        
        X_seq = np.zeros((seq_len, 21), dtype=np.float32)
        A_seq = np.zeros(seq_len, dtype=np.int64)
        R_seq = np.zeros((seq_len, 4), dtype=np.float32)
        
        for t in range(min(n_steps, seq_len)):
            sar = sar_tuples[t]
            X_seq[t] = sar.state
            A_seq[t] = sar.action
            R_seq[t] = sar.reward
        
        X_seqs.append(X_seq)
        A_seqs.append(A_seq)
        R_seqs.append(R_seq)
        outcomes.append(outcome)
        sequence_metadata.append({
            "sequence_id": seq.get("sequence_id"),
            "match_id": seq.get("match_id"),
            "team_id": seq.get("team_id"),
            "home_team": seq.get("home_team"),
            "away_team": seq.get("away_team"),
            "label": seq.get("label"),
            "start_frame": seq.get("start_frame"),
            "end_frame": seq.get("end_frame")
        })
    
    # Stack sequences
    X_all = np.array(X_seqs, dtype=np.float32)  # (N_sequences, seq_len, 21)
    A_all = np.array(A_seqs, dtype=np.int64)    # (N_sequences, seq_len)
    R_all = np.array(R_seqs, dtype=np.float32)  # (N_sequences, seq_len, 4)
    
    # Metadata
    metadata = {
        "n_sequences": len(sequences),
        "seq_length": seq_len,
        "total_transitions": sum(len(seq.get("sar_tuples", [])) for seq in sequences),
        "n_success": outcomes.count("success"),
        "n_failure": outcomes.count("failure"),
        "success_rate": outcomes.count("success") / len(outcomes) if outcomes else 0.0,
        "sequence_metadata": sequence_metadata,
        "action_distribution": {
            ACTION_NAMES[i]: int((A_all == i).sum())
            for i in range(4)
        }
    }
    
    return X_all, A_all, R_all, metadata


# ============================================================================
# SUMMARY & DIAGNOSTICS
# ============================================================================

def summarize_sar_sequences(sequences: List[Dict]) -> Dict:
    """
    Generate summary statistics of SAR sequences.
    
    Args:
        sequences: List of sequences
    
    Returns:
        Dictionary with summary info
    """
    if not sequences:
        return {"error": "No sequences"}
    
    total_steps = sum(seq["length"] for seq in sequences)
    successes = sum(1 for seq in sequences if seq["outcome"] == "success")
    
    X, A, R = extract_ml_arrays(sequences)
    
    return {
        "num_sequences": len(sequences),
        "total_steps": total_steps,
        "avg_steps_per_seq": total_steps / len(sequences) if sequences else 0,
        "success_rate": successes / len(sequences) if sequences else 0,
        "X_shape": X.shape,
        "A_shape": A.shape,
        "R_shape": R.shape,
        "action_distribution": {
            ACTION_NAMES[i]: int((A == i).sum())
            for i in range(4)
        },
        "reward_stats": {
            "space_score": {"mean": float(R[:, 0].mean()), "std": float(R[:, 0].std())},
            "stretch_index": {"mean": float(R[:, 1].mean()), "std": float(R[:, 1].std())},
            "pressure_index": {"mean": float(R[:, 2].mean()), "std": float(R[:, 2].std())},
            "line_height_rel": {"mean": float(R[:, 3].mean()), "std": float(R[:, 3].std())}
        }
    }


if __name__ == "__main__":
    print("[SAR-PREPROCESSING] Module loaded")
    print(f"  Actions: {ACTION_NAMES}")
    print(f"  State dim: 21 (defenders×8×2 + ball×2 + attack×3)")
    print(f"  Reward dim: 4 (space, stretch, pressure, line_height)")

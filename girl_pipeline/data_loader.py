"""
Data loader for SAR (State-Action-Reward) datasets.

Loads pickle files from preprocessing/output/sar/ and returns tensors
ready for GIRL inverse reinforcement learning.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_sar_dataset(
    sar_dir: str = "./preprocessing/output/sar",
    config_suffix: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load SAR dataset from pickle files.
    
    Args:
        sar_dir: Directory containing SAR pickle files
        config_suffix: Optional suffix for specific configuration
                      (e.g., "all_matches_all_players_all_defensive_4_features")
    
    Returns:
        Tuple of:
        - states: np.ndarray of shape (N, 10, 21)
        - actions: np.ndarray of shape (N, 10)
        - rewards: np.ndarray of shape (N, 10, 4)
        - metadata: Dict with dataset information
    
    Raises:
        FileNotFoundError: If SAR files not found
        ValueError: If dataset shapes are inconsistent
    """
    sar_path = Path(sar_dir)
    
    if not sar_path.exists():
        raise FileNotFoundError(f"SAR directory not found: {sar_dir}")
    
    # Find pickle files
    if config_suffix:
        states_file = sar_path / f"states_{config_suffix}.pkl"
        actions_file = sar_path / f"actions_{config_suffix}.pkl"
        rewards_file = sar_path / f"rewards_{config_suffix}.pkl"
        metadata_file = sar_path / f"metadata_{config_suffix}.pkl"
    else:
        # Find first matching files
        states_files = list(sar_path.glob("states_*.pkl"))
        if not states_files:
            raise FileNotFoundError(f"No SAR dataset found in {sar_dir}")
        
        # Use the first dataset found
        states_file = states_files[0]
        suffix = states_file.stem.replace("states_", "")
        actions_file = sar_path / f"actions_{suffix}.pkl"
        rewards_file = sar_path / f"rewards_{suffix}.pkl"
        metadata_file = sar_path / f"metadata_{suffix}.pkl"
    
    # Load pickle files
    print(f"Loading SAR dataset from {sar_path}")
    print(f"  States: {states_file.name}")
    print(f"  Actions: {actions_file.name}")
    print(f"  Rewards: {rewards_file.name}")
    
    with open(states_file, 'rb') as f:
        states = pickle.load(f)
    
    with open(actions_file, 'rb') as f:
        actions = pickle.load(f)
    
    with open(rewards_file, 'rb') as f:
        rewards = pickle.load(f)
    
    # Load metadata if exists
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
    
    # Convert to numpy arrays
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    rewards = np.array(rewards, dtype=np.float32)
    
    # Validate shapes
    n_sequences = states.shape[0]
    
    if actions.shape[0] != n_sequences:
        raise ValueError(
            f"Mismatch: states has {n_sequences} sequences, "
            f"actions has {actions.shape[0]}"
        )
    
    if rewards.shape[0] != n_sequences:
        raise ValueError(
            f"Mismatch: states has {n_sequences} sequences, "
            f"rewards has {rewards.shape[0]}"
        )
    
    # Validate expected shapes
    assert states.ndim == 3, f"Expected states.ndim=3, got {states.ndim}"
    assert actions.ndim == 2, f"Expected actions.ndim=2, got {actions.ndim}"
    assert rewards.ndim == 3, f"Expected rewards.ndim=3, got {rewards.ndim}"
    
    seq_length = states.shape[1]
    state_dim = states.shape[2]
    n_reward_features = rewards.shape[2]
    
    # Detailed shape assertions
    assert seq_length == 10, f"Expected sequence length 10, got {seq_length}"
    assert state_dim == 21, f"Expected state dimension 21, got {state_dim}"
    assert n_reward_features == 4, f"Expected 4 reward features, got {n_reward_features}"
    assert actions.shape[1] == seq_length, \
        f"Actions sequence length {actions.shape[1]} != states {seq_length}"
    assert rewards.shape[1] == seq_length, \
        f"Rewards sequence length {rewards.shape[1]} != states {seq_length}"
    
    print(f"\n{'='*60}")
    print("SAR Dataset Validation")
    print(f"{'='*60}")
    print(f"✓ Tensor shapes valid")
    print(f"  States shape:   {states.shape} (expected: (N, 10, 21))")
    print(f"  Actions shape:  {actions.shape} (expected: (N, 10))")
    print(f"  Rewards shape:  {rewards.shape} (expected: (N, 10, 4))")
    print(f"✓ {n_sequences} sequences loaded")
    print(f"✓ Sequence length: {seq_length} timesteps")
    print(f"✓ State dimension: {state_dim} features")
    print(f"✓ Reward features: {n_reward_features}")
    print(f"✓ Action space: {metadata.get('action_space', [0, 1, 2, 3])}")
    print(f"{'='*60}\n")
    
    return states, actions, rewards, metadata


def get_action_names() -> Dict[int, str]:
    """
    Get mapping from action indices to action names.
    
    Returns:
        Dict mapping action index to action name
    """
    return {
        0: "backward",
        1: "forward",
        2: "compress",
        3: "expand"
    }


def validate_sar_dataset(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray
) -> bool:
    """
    Validate SAR dataset integrity.
    
    Args:
        states: State sequences
        actions: Action sequences
        rewards: Reward sequences
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check for NaN or Inf
    if np.any(np.isnan(states)):
        raise ValueError("States contain NaN values")
    if np.any(np.isinf(states)):
        raise ValueError("States contain Inf values")
    
    if np.any(np.isnan(rewards)):
        raise ValueError("Rewards contain NaN values")
    if np.any(np.isinf(rewards)):
        raise ValueError("Rewards contain Inf values")
    
    # Check action validity
    assert np.all(actions >= 0), f"Actions have negative values: min={actions.min()}"
    assert np.all(actions <= 3), f"Actions exceed max value 3: max={actions.max()}"
    
    # Check consistency
    n_seq = states.shape[0]
    if actions.shape[0] != n_seq or rewards.shape[0] != n_seq:
        raise ValueError("Number of sequences must match across states/actions/rewards")
    
    if states.shape[1] != actions.shape[1] or actions.shape[1] != rewards.shape[1]:
        raise ValueError("Sequence lengths must match across states/actions/rewards")
    
    # Print action distribution
    print(f"\n{'='*60}")
    print("Action Encoding Validation")
    print(f"{'='*60}")
    action_names = get_action_names()
    for action_idx in range(4):
        count = np.sum(actions == action_idx)
        percentage = 100 * count / actions.size if actions.size > 0 else 0
        action_name = action_names.get(action_idx, f"action_{action_idx}")
        print(f"✓ {action_name:12s}: {count:6d} ({percentage:5.1f}%)")
    print(f"✓ Action range: [{actions.min()}, {actions.max()}]")
    print(f"✓ Dataset validation passed")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    # Test data loader
    try:
        states, actions, rewards, metadata = load_sar_dataset()
        validate_sar_dataset(states, actions, rewards)
        
        print("\nAction distribution:")
        action_names = get_action_names()
        for action_idx, action_name in action_names.items():
            count = np.sum(actions == action_idx)
            percentage = 100 * count / actions.size
            print(f"  {action_name}: {count} ({percentage:.1f}%)")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run preprocessing pipeline first to generate SAR dataset")

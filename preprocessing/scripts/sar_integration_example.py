"""
SAR Preprocessing Integration Example
======================================

Shows how to use sar_preprocessing.py with the existing Defense Line pipeline
to create ML-ready datasets in State-Action-Reward format.

Author: Defense Line Analysis Pipeline
Date: 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add cleaned directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from preprocessing import preprocess_match
from sar_preprocessing import (
    create_sar_sequences,
    extract_ml_arrays,
    summarize_sar_sequences,
    SARTuple
)


def create_sar_dataset_from_match(
    match_dir: Path,
    config: PipelineConfig,
    sequence_length: int = 10,
    verbose: bool = True
) -> dict:
    """
    Create SAR dataset from a single match.
    
    Args:
        match_dir: Path to match directory with events.csv, etc.
        config: PipelineConfig
        sequence_length: Length of sequences (default 10)
        verbose: Print progress
    
    Returns:
        Dictionary with:
            - sequences: List of SAR sequences
            - X: State array (N, 21)
            - A: Action array (N,)
            - R: Reward array (N, 4)
            - summary: Summary statistics
    """
    if verbose:
        print(f"\n[STEP 1] Loading match: {match_dir.name}")
    
    # Preprocess match
    match_data = preprocess_match(match_dir, config)
    
    if not match_data:
        print(f"[ERROR] Failed to preprocess {match_dir}")
        return None
    
    events = match_data.get("events", [])
    if verbose:
        print(f"[STEP 2] Loaded {len(events)} events")
        print(f"         Home: {match_data['home_team']} vs Away: {match_data['away_team']}")
    
    # Convert to event list format expected by SAR functions
    # Wrap each event with preprocessing data
    events_with_state = []
    for event in events:
        events_with_state.append({"state": event})
    
    if verbose:
        print(f"[STEP 3] Creating SAR sequences (length={sequence_length})...")
    
    # Create SAR sequences
    sequences = create_sar_sequences(
        events_with_state,
        sequence_length=sequence_length,
        skip_initial=1,
        overlap=0
    )
    
    if verbose:
        print(f"         Generated {len(sequences)} sequences")
    
    # Extract ML arrays
    if sequences:
        X, A, R = extract_ml_arrays(sequences)
        summary = summarize_sar_sequences(sequences)
        
        if verbose:
            print(f"\n[DATASET SUMMARY]")
            print(f"  State shape:       {X.shape}")
            print(f"  Action shape:      {A.shape}")
            print(f"  Reward shape:      {R.shape}")
            print(f"  Success rate:      {summary['success_rate']:.1%}")
            print(f"  Action distribution:")
            for action, count in summary['action_distribution'].items():
                print(f"    {action:12}: {count:5d}")
        
        return {
            "match_id": match_dir.name,
            "sequences": sequences,
            "X": X,
            "A": A,
            "R": R,
            "summary": summary
        }
    else:
        print("[ERROR] No valid sequences created")
        return None


def create_sar_dataset_multi_matches(
    match_dirs: list,
    config: PipelineConfig,
    sequence_length: int = 10,
    verbose: bool = True
) -> dict:
    """
    Create SAR dataset from multiple matches.
    
    Args:
        match_dirs: List of match directory paths
        config: PipelineConfig
        sequence_length: Sequence length
        verbose: Print progress
    
    Returns:
        Dictionary with aggregated datasets and per-match results
    """
    all_sequences = []
    all_X, all_A, all_R = [], [], []
    match_results = {}
    
    for i, match_dir in enumerate(match_dirs, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(match_dirs)}] Processing {match_dir.name}")
        print(f"{'='*70}")
        
        result = create_sar_dataset_from_match(
            match_dir,
            config,
            sequence_length=sequence_length,
            verbose=verbose
        )
        
        if result:
            match_results[match_dir.name] = result
            all_sequences.extend(result['sequences'])
            all_X.append(result['X'])
            all_A.append(result['A'])
            all_R.append(result['R'])
        else:
            print(f"[SKIP] Failed to process {match_dir.name}")
    
    # Combine all arrays
    if all_X:
        X_combined = np.vstack(all_X)
        A_combined = np.concatenate(all_A)
        R_combined = np.vstack(all_R)
        
        print(f"\n{'='*70}")
        print("[AGGREGATED DATASET]")
        print(f"{'='*70}")
        print(f"Total matches:       {len(match_results)}")
        print(f"Total sequences:     {len(all_sequences)}")
        print(f"Total transitions:   {len(X_combined)}")
        print(f"State shape:         {X_combined.shape}")
        print(f"Action shape:        {A_combined.shape}")
        print(f"Reward shape:        {R_combined.shape}")
        
        return {
            "num_matches": len(match_results),
            "sequences": all_sequences,
            "X": X_combined,
            "A": A_combined,
            "R": R_combined,
            "match_results": match_results
        }
    else:
        print("[ERROR] No valid datasets created")
        return None


def save_sar_dataset(dataset: dict, output_dir: Path) -> None:
    """
    Save SAR dataset to disk as numpy files.
    
    Args:
        dataset: Dataset dictionary from create_sar_dataset_*()
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "X_states.npy", dataset["X"])
    np.save(output_dir / "A_actions.npy", dataset["A"])
    np.save(output_dir / "R_rewards.npy", dataset["R"])
    
    print(f"\n[SAVED] Dataset to {output_dir}")
    print(f"  - X_states.npy:   {dataset['X'].shape}")
    print(f"  - A_actions.npy:  {dataset['A'].shape}")
    print(f"  - R_rewards.npy:  {dataset['R'].shape}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Create SAR dataset from a single match
    """
    
    # Configuration
    config = PipelineConfig(
        data_match='all_matches',
        back_four='all_players',
        sequence_type='all_defensive',
        reward_features='5_features',
        method='feature_computation',
        data_dir='/home/s_dash/workspace6/Defense_line/Laliga2023/24'
    )
    
    # Single match example
    match_dir = Path('/home/s_dash/workspace6/Defense_line/Laliga2023/24/1018887')
    
    print("[SAR PREPROCESSING EXAMPLE]")
    print("="*70)
    
    dataset = create_sar_dataset_from_match(match_dir, config, verbose=True)
    
    if dataset:
        print(f"\n[SUCCESS] Created SAR dataset")
        print(f"  Sequences: {len(dataset['sequences'])}")
        print(f"  X shape:   {dataset['X'].shape}")
        print(f"  A shape:   {dataset['A'].shape}")
        print(f"  R shape:   {dataset['R'].shape}")
        
        # Display sample transitions
        print(f"\n[SAMPLE TRANSITIONS]")
        for i in range(min(3, len(dataset['sequences'][0]['sar_tuples']))):
            sar = dataset['sequences'][0]['sar_tuples'][i]
            print(f"  Step {i}: {sar}")
        
        # Save to disk
        # save_sar_dataset(dataset, Path('./sar_dataset'))
    else:
        print("[ERROR] Failed to create dataset")

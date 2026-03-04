"""
Main entry point for the Defense Line Analysis pipeline.

This module orchestrates the complete pipeline:
1. Data preprocessing (without reward features or back four detection)
2. Feature engineering with configurable flags
3. Output generation and optional visualization

Usage:
    # Via command line
    python main.py --data_match barcelona_madrid --method girl

    # Via Python API
    from preprocessing.config import PipelineConfig
    from main import run_pipeline
    
    config = PipelineConfig(
        data_match='all_matches',
        back_four='all_players',
        sequence_type='all_defensive',
        reward_features='5_features',
        method='feature_computation'
    )
    df_features = run_pipeline(config)
"""

import argparse
import gzip
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

from preprocessing.config import PipelineConfig
from preprocessing.preprocessing import preprocess_all_matches
from preprocessing.feature_engineering import compute_match_features


# ============================================================================
# MAIN PIPELINE ORCHESTRATION
# ============================================================================

def run_pipeline(config: PipelineConfig) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Run the complete analysis pipeline.
    
    Supports two modes:
    - method="feature_computation": outputs feature DataFrames
    - method="girl": outputs SAR dataset (pickles)
    
    Args:
        config: Pipeline configuration with flags
    
    Returns:
        Tuple of (features_df, sar_paths) where:
        - features_df: DataFrame if method="feature_computation", empty if "girl"
        - sar_paths: Dict with SAR pickle paths if method="girl", None otherwise
    """
    print("="*70)
    print("Defense Line Analysis Pipeline")
    print("="*70)
    print(f"Configuration:\n{config}\n")
    print(f"Method: {config.method}")
    print(f"Output mode: {'SAR Dataset' if config.method == 'girl' else 'Feature Computation'}\n")
    
    # Step 1: Preprocess all matches
    print("STEP 1: Preprocessing matches...")
    print("-"*70)
    all_matches = preprocess_all_matches(config)
    
    if len(all_matches) == 0:
        print("❌ No matches found after filtering!")
        return pd.DataFrame(), None

    # Save cleaned/preprocessed data as compressed pickle
    preprocess_path = save_preprocessed_data(all_matches, Path(config.output_dir), config)
    print(f"\n✓ Preprocessed data saved to: {preprocess_path}")
    
    # =========================================================================
    # BRANCH 1: GIRL Method → SAR Dataset
    # =========================================================================
    if config.method == "girl":
        print("\nSTEP 2: Building SAR sequences for GIRL...")
        print("-"*70)
        
        sar_states_list = []
        sar_actions_list = []
        sar_rewards_list = []
        sar_sequence_metadata_list = []
        
        for match_data in tqdm(all_matches, desc="Building SAR sequences"):
            sar_output = compute_match_features(match_data, config)
            
            if isinstance(sar_output, dict) and sar_output.get('is_sar'):
                sar_data = sar_output.get('sar_data')
                if sar_data is not None:
                    sar_states_list.append(sar_data['states'])
                    sar_actions_list.append(sar_data['actions'])
                    sar_rewards_list.append(sar_data['rewards'])

                    sequence_meta = sar_data.get('metadata', {}).get('sequence_metadata', [])
                    sar_sequence_metadata_list.extend(sequence_meta)
        
        if not sar_states_list:
            print("❌ No SAR sequences created!")
            return pd.DataFrame(), None
        
        # Step 3: Save SAR Dataset
        print("\nSTEP 3: Saving SAR dataset...")
        print("-"*70)
        sar_paths = save_sar_dataset(
            sar_states_list,
            sar_actions_list,
            sar_rewards_list,
            sar_sequence_metadata_list,
            Path(config.output_dir),
            config
        )
        
        # Print SAR dataset summary
        import numpy as np
        total_states = np.concatenate(sar_states_list, axis=0)
        total_actions = np.concatenate(sar_actions_list, axis=0)
        total_rewards = np.concatenate(sar_rewards_list, axis=0)
        
        print(f"\nSAR Dataset Summary:")
        print(f"  Total sequences: {total_states.shape[0]}")
        print(f"  Sequence length: {total_states.shape[1]}")
        print(f"  State dimension: {total_states.shape[2]}")
        print(f"  Action space: {{0:backward, 1:forward, 2:compress, 3:expand}}")
        print(f"  Reward features: {config.reward_features}")
        print(f"\n  Tensor shapes:")
        print(f"    states:  {total_states.shape}")
        print(f"    actions: {total_actions.shape}")
        print(f"    rewards: {total_rewards.shape}")
        
        return pd.DataFrame(), sar_paths
    
    # =========================================================================
    # BRANCH 2: Feature Computation Method → Feature DataFrames
    # =========================================================================
    else:
        print("\nSTEP 2: Computing features...")
        print("-"*70)
        all_features = []
        
        for match_data in tqdm(all_matches, desc="Computing features"):
            features_df = compute_match_features(match_data, config)
            if isinstance(features_df, pd.DataFrame) and not features_df.empty:
                all_features.append(features_df)
        
        if len(all_features) == 0:
            print("❌ No features computed!")
            return pd.DataFrame(), None
        
        # Combine all features
        result_df = pd.concat(all_features, ignore_index=True)
        
        # Step 3: Generate summary statistics
        print("\nSTEP 3: Generating summary statistics...")
        print("-"*70)
        
        print(f"\nTotal sequences analyzed: {len(result_df)}")
        print(f"Unique matches: {result_df['match_id'].nunique()}")
        print(f"\nFeature statistics:")
        print(result_df[['space_score', 'pressure_index', 'stretch_index', 
                         'line_height_relative']].describe())
        
        if config.reward_features == '5_features':
            print(f"\nline_height_absolute statistics:")
            print(result_df['line_height_absolute'].describe())

        # Transition-style processing log (similar to reference notebook script)
        log_paths = save_transition_processing_log(
            all_matches,
            result_df,
            Path(config.output_dir),
            config
        )
        print(f"\n✓ Processing log saved to: {log_paths['csv']}")
        print(f"✓ Processing log (json) saved to: {log_paths['json']}")

        # Team-based and overall summaries (mean for feature columns)
        overall_summary, team_summary = generate_summary_statistics(result_df)
        summary_paths = save_summary_statistics(
            overall_summary,
            team_summary,
            Path(config.output_dir),
            config
        )

        print("\nOverall summary:")
        print(overall_summary.to_string(index=False))
        print("\nTeam summary (first 10 rows):")
        print(team_summary.head(10).to_string(index=False))
        print(f"\n✓ Summary saved to: {summary_paths['overall']}")
        print(f"✓ Team summary saved to: {summary_paths['team']}")

        preview_cols = [
            'match_id', 'transition_idx', 'label', 'home_team', 'away_team',
            'team_lost_possession', 'team_gained_possession', 'period',
            'player_lost_possession'
        ]
        mean_feature_cols = [c for c in result_df.columns if c.endswith('_mean')]
        show_cols = [c for c in preview_cols + mean_feature_cols if c in result_df.columns]
        if show_cols:
            print("\nPreview table (5 rows):")
            print(result_df[show_cols].head(5).to_string(index=False))
        
        return result_df, None


def save_transition_processing_log(
    all_matches: list,
    features_df: pd.DataFrame,
    output_dir: Path,
    config: PipelineConfig
) -> dict:
    """Create and save per-match + overall processing log similar to notebook flow."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for match_data in all_matches:
        report = match_data.get('missing_data_report', {}) or {}
        match_id = match_data.get('match_id')
        match_features = features_df[features_df['match_id'] == match_id] if 'match_id' in features_df.columns else pd.DataFrame()

        labels = match_features['label'] if 'label' in match_features.columns else pd.Series(dtype=float)
        success_count = int((labels == 1).sum()) if not labels.empty else 0
        failure_count = int((labels == 0).sum()) if not labels.empty else 0
        total_labeled = success_count + failure_count

        rows.append({
            'match_id': match_id,
            'home_team': match_data.get('home_team'),
            'away_team': match_data.get('away_team'),
            'events_raw': int(report.get('events_raw', 0)),
            'events_synced': int(report.get('events_synced', 0)),
            'events_dropped_sync_or_time': int(report.get('events_dropped_sync_or_time', 0)),
            'possession_events': int(report.get('possession_events', 0)),
            'restart_events_removed': int(report.get('restart_events_removed', 0)),
            'turnover_events': int(report.get('turnover_events', 0)),
            'transitions_extracted': int(report.get('transitions_extracted', 0)),
            'sequences_extracted': int(report.get('sequences_extracted', 0)),
            'feature_rows': int(len(match_features)),
            'defense_success_count': success_count,
            'defense_failure_count': failure_count,
            'defense_success_rate': float(success_count / total_labeled) if total_labeled > 0 else np.nan
        })

    log_df = pd.DataFrame(rows)

    if not log_df.empty:
        numeric_cols = [
            c for c in log_df.columns
            if c not in {'match_id', 'home_team', 'away_team'}
        ]
        overall = {col: float(log_df[col].sum()) for col in numeric_cols if col != 'defense_success_rate'}
        if log_df['defense_success_rate'].notna().any():
            overall['defense_success_rate'] = float(log_df['defense_success_rate'].mean())
        else:
            overall['defense_success_rate'] = np.nan

        overall_row = {
            'match_id': 'OVERALL',
            'home_team': 'ALL',
            'away_team': 'ALL',
            **overall
        }
        log_df = pd.concat([log_df, pd.DataFrame([overall_row])], ignore_index=True)

    suffix = f"{config.data_match}_{config.back_four}_{config.sequence_type}_{config.reward_features}_{config.method}"
    csv_path = output_dir / f"processing_log_{suffix}.csv"
    json_path = output_dir / f"processing_log_{suffix}.json"

    log_df.to_csv(csv_path, index=False)
    log_df.to_json(json_path, orient='records', indent=2)

    print("\n===== Processing Log Preview =====")
    if not log_df.empty:
        print(log_df.head(10).to_string(index=False))

    return {'csv': csv_path, 'json': json_path}


def save_preprocessed_data(
    preprocessed_matches: list,
    output_dir: Path,
    config: PipelineConfig
) -> Path:
    """Save cleaned/preprocessed match data as compressed pickle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"preprocessed_"
        f"{config.data_match}_"
        f"{config.back_four}_"
        f"{config.sequence_type}.pkl.gz"
    )
    output_path = output_dir / filename

    with gzip.open(output_path, 'wb') as f:
        pickle.dump(preprocessed_matches, f, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path


def generate_summary_statistics(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate overall and team-based summary stats using mean feature values."""
    if features_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    mean_feature_cols = [c for c in features_df.columns if c.endswith('_mean')]

    overall = {
        'n_rows': int(len(features_df)),
        'n_matches': int(features_df['match_id'].nunique()) if 'match_id' in features_df.columns else 0,
        'label_success_rate': float((features_df['label'] == 0).mean()) if 'label' in features_df.columns else np.nan,
        'label_failure_rate': float((features_df['label'] == 1).mean()) if 'label' in features_df.columns else np.nan
    }
    for col in mean_feature_cols:
        overall[col] = float(features_df[col].mean())

    overall_df = pd.DataFrame([overall])

    group_keys = [c for c in ['team_lost_possession', 'home_team', 'away_team'] if c in features_df.columns]
    if not group_keys:
        team_df = pd.DataFrame()
    else:
        agg_dict = {col: 'mean' for col in mean_feature_cols}
        if 'label' in features_df.columns:
            agg_dict['label'] = 'mean'

        team_df = (
            features_df
            .groupby(group_keys, dropna=False)
            .agg(agg_dict)
            .reset_index()
            .rename(columns={'label': 'avg_label'})
        )

        counts_df = (
            features_df
            .groupby(group_keys, dropna=False)
            .size()
            .reset_index(name='n_sequences')
        )
        team_df = team_df.merge(counts_df, on=group_keys, how='left')

    return overall_df, team_df


def save_summary_statistics(
    overall_df: pd.DataFrame,
    team_df: pd.DataFrame,
    output_dir: Path,
    config: PipelineConfig
) -> dict:
    """Save overall and team summary tables to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{config.data_match}_{config.back_four}_{config.sequence_type}_{config.reward_features}_{config.method}"
    overall_path = output_dir / f"summary_overall_{suffix}.csv"
    team_path = output_dir / f"summary_team_{suffix}.csv"

    overall_df.to_csv(overall_path, index=False)
    team_df.to_csv(team_path, index=False)

    return {'overall': overall_path, 'team': team_path}


def save_sar_dataset(
    sar_states_list: list,
    sar_actions_list: list,
    sar_rewards_list: list,
    sar_sequence_metadata_list: list,
    output_dir: Path,
    config: PipelineConfig
) -> dict:
    """
    Save SAR dataset tensors to pickle files.
    
    Args:
        sar_states_list: List of (N_seq, 10, 21) state arrays per match
        sar_actions_list: List of (N_seq, 10) action arrays per match
        sar_rewards_list: List of (N_seq, 10, 4) reward arrays per match
        sar_sequence_metadata_list: List of per-sequence metadata dicts
        output_dir: Output directory path
        config: Pipeline configuration
    
    Returns:
        Dict with paths to saved pickle files
    """
    import numpy as np
    
    output_dir = Path(output_dir) / "sar"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Concatenate all matches if we have multiple
    if sar_states_list:
        states_combined = np.concatenate(sar_states_list, axis=0)
        actions_combined = np.concatenate(sar_actions_list, axis=0)
        rewards_combined = np.concatenate(sar_rewards_list, axis=0)
    else:
        states_combined = np.zeros((0, 10, 21), dtype=np.float32)
        actions_combined = np.zeros((0, 10), dtype=np.int64)
        rewards_combined = np.zeros((0, 10, 4), dtype=np.float32)
    
    # Save pickle files
    suffix = f"{config.data_match}_{config.back_four}_{config.sequence_type}_{config.reward_features}"
    
    states_path = output_dir / f"states_{suffix}.pkl"
    actions_path = output_dir / f"actions_{suffix}.pkl"
    rewards_path = output_dir / f"rewards_{suffix}.pkl"
    metadata_path = output_dir / f"metadata_{suffix}.pkl"
    
    with open(states_path, 'wb') as f:
        pickle.dump(states_combined, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(actions_path, 'wb') as f:
        pickle.dump(actions_combined, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(rewards_path, 'wb') as f:
        pickle.dump(rewards_combined, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Combined metadata
    combined_metadata = {
        "n_sequences": states_combined.shape[0],
        "seq_length": 10,
        "state_dim": 21,
        "action_space": [0, 1, 2, 3],
        "action_names": {0: "backward", 1: "forward", 2: "compress", 3: "expand"},
        "reward_features": config.reward_features,
        "sequence_metadata": sar_sequence_metadata_list,
        "per_match_metadata": sar_sequence_metadata_list,
        "total_transitions": int(actions_combined.sum() > 0) if len(actions_combined) > 0 else 0,
        "config": config.to_dict() if hasattr(config, 'to_dict') else {}
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(combined_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n✓ SAR Dataset saved:")
    print(f"  States:   {states_path} shape={states_combined.shape}")
    print(f"  Actions:  {actions_path} shape={actions_combined.shape}")
    print(f"  Rewards:  {rewards_path} shape={rewards_combined.shape}")
    print(f"  Metadata: {metadata_path}")
    
    return {
        'states': states_path,
        'actions': actions_path,
        'rewards': rewards_path,
        'metadata': metadata_path
    }


def save_features(
    features_df: pd.DataFrame,
    output_dir: Path,
    config: PipelineConfig
) -> Path:
    """
    Save computed features to CSV.
    
    Args:
        features_df: DataFrame with computed features
        output_dir: Output directory
        config: Pipeline configuration (used for filename generation)
    
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename based on configuration
    filename = (
        f"features_"
        f"{config.data_match}_"
        f"{config.back_four}_"
        f"{config.sequence_type}_"
        f"{config.reward_features}_"
        f"{config.method}.csv"
    )
    
    output_path = output_dir / filename
    features_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Features saved to: {output_path}")
    return output_path


def generate_comparison_report(
    features_list: dict,
    output_dir: Path
) -> None:
    """
    Generate comparison report for multiple configurations.
    
    Args:
        features_list: Dictionary of {config_name: features_df}
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison summary
    comparison = {}
    
    for config_name, df in features_list.items():
        if df.empty:
            continue
        
        comparison[config_name] = {
            'n_sequences': len(df),
            'n_matches': df['match_id'].nunique(),
            'space_score_mean': df['space_score'].mean(),
            'space_score_std': df['space_score'].std(),
            'pressure_index_mean': df['pressure_index'].mean(),
            'pressure_index_std': df['pressure_index'].std(),
            'stretch_index_mean': df['stretch_index'].mean(),
            'stretch_index_std': df['stretch_index'].std(),
            'line_height_relative_mean': df['line_height_relative'].mean(),
            'line_height_relative_std': df['line_height_relative'].std(),
        }
    
    comparison_df = pd.DataFrame(comparison).T
    comparison_path = output_dir / 'feature_comparison.csv'
    comparison_df.to_csv(comparison_path)
    
    print(f"\n✓ Comparison report saved to: {comparison_path}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Defense Line Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Run with Barcelona vs Real Madrid only
  python main.py --data_match barcelona_madrid
  
  # Run with back four detection and GIRL method
  python main.py --data_match all_matches --back_four back_four --method girl
  
  # Run all combinations and generate comparison
  python main.py --compare
        """
    )
    
    # Flag 1: Data match selection
    parser.add_argument(
        '--data_match',
        choices=['barcelona_madrid', 'all_matches'],
        default='all_matches',
        help='Dataset selection (default: all_matches)'
    )
    
    # Flag 2: Back four detection
    parser.add_argument(
        '--back_four',
        choices=['back_four', 'all_players'],
        default='all_players',
        help='Defender selection (default: all_players)'
    )
    
    # Flag 3: Sequence type
    parser.add_argument(
        '--sequence_type',
        choices=['negative_transition', 'all_defensive'],
        default='all_defensive',
        help='Defensive sequence type (default: all_defensive)'
    )
    
    # Flag 4: Reward features
    parser.add_argument(
        '--reward_features',
        choices=['4_features', '5_features'],
        default='5_features',
        help='Feature set (default: 5_features)'
    )
    
    # Flag 5: Method
    parser.add_argument(
        '--method',
        choices=['girl', 'feature_computation'],
        default='feature_computation',
        help='Feature computation method (default: feature_computation)'
    )
    
    # I/O options
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/Laliga2023/24',
        help='Input data directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save features to CSV'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run all flag combinations and generate comparison'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        run_comparison(args)
    else:
        run_single(args)


def run_single(args):
    """Run pipeline with single configuration."""
    config = PipelineConfig(
        data_match=args.data_match,
        back_four=args.back_four,
        sequence_type=args.sequence_type,
        reward_features=args.reward_features,
        method=args.method,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    features_df, _ = run_pipeline(config)

    if args.save and not features_df.empty:
        save_features(features_df, Path(args.output_dir), config)

    return features_df


def run_comparison(args):
    """Run pipeline with multiple configurations for comparison."""
    print("\n" + "="*70)
    print("COMPARISON MODE: Running multiple configurations")
    print("="*70 + "\n")
    
    # Generate all combinations
    data_matches = ['barcelona_madrid', 'all_matches']
    back_fours = ['back_four', 'all_players']
    sequence_types = ['negative_transition', 'all_defensive']
    reward_features = ['4_features', '5_features']
    methods = ['girl', 'feature_computation']
    
    features_by_config = {}
    
    for dm in data_matches:
        for bf in back_fours:
            for st in sequence_types:
                for rf in reward_features:
                    for m in methods:
                        config_name = f"{dm}|{bf}|{st}|{rf}|{m}"
                        print(f"\n[{config_name}]")
                        
                        config = PipelineConfig(
                            data_match=dm,
                            back_four=bf,
                            sequence_type=st,
                            reward_features=rf,
                            method=m,
                            data_dir=args.data_dir,
                            output_dir=args.output_dir
                        )
                        
                        try:
                            features_df, _ = run_pipeline(config)
                            features_by_config[config_name] = features_df
                            
                            if args.save and not features_df.empty:
                                save_features(features_df, Path(args.output_dir), config)
                        
                        except Exception as e:
                            print(f"❌ Error: {str(e)}")
                            features_by_config[config_name] = pd.DataFrame()
    
    # Generate comparison report
    generate_comparison_report(features_by_config, Path(args.output_dir))


if __name__ == "__main__":
    main()

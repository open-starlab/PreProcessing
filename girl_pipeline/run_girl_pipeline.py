#!/usr/bin/env python3
"""
GIRL Pipeline Runner

Main entry point for running the complete GIRL inverse reinforcement learning pipeline.

Workflow:
1. Load SAR dataset from preprocessing/output/sar/
2. Perform k-fold cross-validation
3. Train BC model on each fold
4. Compute policy gradients
5. Solve GIRL for reward weights
6. Report mean and std of recovered weights

Usage:
    python girl_pipeline/run_girl_pipeline.py [--n_folds 5] [--device cpu]
"""

import argparse
import sys
import numpy as np
import torch
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from girl_pipeline.data_loader import load_sar_dataset, get_action_names, validate_sar_dataset
from girl_pipeline.utils.cross_validation import cross_validate_girl, simple_train_test_split
from girl_pipeline.utils.plot_weights import plot_reward_weights


def main():
    """Main GIRL pipeline runner."""
    parser = argparse.ArgumentParser(
        description='GIRL Pipeline - Inverse Reinforcement Learning for Football Defense',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--sar_dir',
        type=str,
        default='./preprocessing/output/sar',
        help='Directory containing SAR dataset'
    )
    parser.add_argument(
        '--config_suffix',
        type=str,
        default=None,
        help='Optional config suffix for specific SAR dataset'
    )
    
    # Training arguments
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (use 0 for train/test split)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test size for train/test split (only used if n_folds=0)'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=64,
        help='LSTM hidden size for BC model'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of training epochs for BC'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    
    # GIRL arguments
    parser.add_argument(
        '--solver_method',
        type=str,
        default='quadprog',
        choices=['quadprog', 'cvxopt', 'analytical'],
        help='QP solver method for GIRL'
    )
    
    # Misc arguments
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode - use only first 20 sequences for quick testing'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./girl_pipeline/output',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    verbose = not args.quiet
    
    # Print header
    if verbose:
        print()
        print("="*80)
        print("GIRL PIPELINE - Goal-based Inverse Reinforcement Learning")
        print("="*80)
        print()
        print("Recovering reward weights from expert defensive demonstrations")
        print()
    
    # Set random seeds
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if device == 'cuda':
        torch.cuda.manual_seed(args.random_state)
    
    try:
        # Step 1: Load SAR dataset
        if verbose:
            print("STEP 1: Loading SAR Dataset")
            print("-"*80)
        
        states, actions, rewards, metadata = load_sar_dataset(
            sar_dir=args.sar_dir,
            config_suffix=args.config_suffix
        )
        
        # Validate dataset
        validate_sar_dataset(states, actions, rewards)
        
        # Debug mode - use subset of data
        if args.debug:
            if verbose:
                print("\n[DEBUG MODE] Using only first 20 sequences")
            states = states[:20]
            actions = actions[:20]
            rewards = rewards[:20]
        
        if verbose:
            print()
            action_names = get_action_names()
            print("Action distribution:")
            for action_idx, action_name in action_names.items():
                count = np.sum(actions == action_idx)
                percentage = 100 * count / actions.size
                print(f"  {action_name:12s}: {count:6d} ({percentage:5.1f}%)")
            print()
        
        # Step 2: Run GIRL pipeline
        if args.n_folds > 0:
            # Cross-validation
            if verbose:
                print("STEP 2: Running GIRL with Cross-Validation")
                print("-"*80)
            
            results = cross_validate_girl(
                states=states,
                actions=actions,
                rewards=rewards,
                n_splits=args.n_folds,
                random_state=args.random_state,
                hidden_size=args.hidden_size,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                device=device,
                solver_method=args.solver_method,
                verbose=verbose
            )
            
            # Extract results
            mean_weights = results['mean_weights']
            std_weights = results['std_weights']
            
        else:
            # Simple train/test split
            if verbose:
                print("STEP 2: Running GIRL with Train/Test Split")
                print("-"*80)
            
            mean_weights, results = simple_train_test_split(
                states=states,
                actions=actions,
                rewards=rewards,
                test_size=args.test_size,
                random_state=args.random_state,
                hidden_size=args.hidden_size,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                device=device,
                solver_method=args.solver_method,
                verbose=verbose
            )
            
            std_weights = np.zeros_like(mean_weights)  # No std for single split
        
        # Step 3: Display final results
        if verbose:
            print()
            print("="*80)
            print("FINAL RESULTS")
            print("="*80)
        
        print()
        print("Recovered Reward Weights:")
        print("-"*80)
        
        feature_names = ['stretch_index', 'pressure_index', 'space_score', 'line_height_rel']
        n_features = len(mean_weights)
        
        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            if args.n_folds > 0:
                print(f"  {name:20s}: {mean_weights[i]:.4f} ± {std_weights[i]:.4f}")
            else:
                print(f"  {name:20s}: {mean_weights[i]:.4f}")
        
        print()
        print(f"Sum of weights: {mean_weights.sum():.6f}")
        print()
        
        # Interpretation
        if verbose:
            print("Interpretation:")
            print("-"*80)
            max_idx = np.argmax(mean_weights)
            max_feature = feature_names[max_idx] if max_idx < len(feature_names) else f"feature_{max_idx}"
            print(f"Most important feature: {max_feature} (weight: {mean_weights[max_idx]:.4f})")
            print()
            
            print("Feature importance ranking:")
            sorted_indices = np.argsort(mean_weights)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                print(f"  {rank}. {name:20s}: {mean_weights[idx]:.4f}")
            print()
        
        # Step 4: Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save reward weights
        weights_df = pd.DataFrame({
            'feature': feature_names[:len(mean_weights)],
            'weight': mean_weights,
            'std': std_weights if args.n_folds > 0 else [0] * len(mean_weights)
        })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_file = output_dir / f"reward_weights_{timestamp}.csv"
        weights_df.to_csv(weights_file, index=False)
        
        if verbose:
            print(f"Saved reward weights to: {weights_file}")
        
        # Save cross-validation results if available
        if args.n_folds > 0 and 'fold_results' in results:
            cv_results = []
            for fold_result in results['fold_results']:
                for feat_idx, feat_name in enumerate(feature_names[:len(mean_weights)]):
                    cv_results.append({
                        'fold': fold_result['fold'],
                        'feature': feat_name,
                        'weight': fold_result['reward_weights'][feat_idx]
                    })
            
            cv_df = pd.DataFrame(cv_results)
            cv_file = output_dir / f"cross_validation_results_{timestamp}.csv"
            cv_df.to_csv(cv_file, index=False)
            
            if verbose:
                print(f"Saved cross-validation results to: {cv_file}")
        
        # Save experiment metadata
        experiment_metadata = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(states),
            'sequence_length': states.shape[1],
            'state_dimension': states.shape[2],
            'reward_dimension': rewards.shape[2],
            'action_count': 4,
            'cross_validation_folds': args.n_folds if args.n_folds > 0 else 1,
            'random_seed': args.random_state,
            'hidden_size': args.hidden_size,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'device': device,
            'solver_method': args.solver_method,
            'debug_mode': args.debug,
            'feature_names': feature_names[:len(mean_weights)],
            'recovered_weights': {
                feature_names[i]: float(mean_weights[i])
                for i in range(len(mean_weights))
            },
            'weight_std': {
                feature_names[i]: float(std_weights[i])
                for i in range(len(std_weights))
            } if args.n_folds > 0 else {}
        }
        
        metadata_file = output_dir / f"experiment_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        if verbose:
            print(f"Saved experiment metadata to: {metadata_file}")
        
        # Generate visualization
        if verbose:
            print("\nGenerating visualization...")
        
        plot_file = plot_reward_weights(
            weights=mean_weights,
            std_weights=std_weights if args.n_folds > 0 else None,
            feature_names=feature_names[:len(mean_weights)],
            title="Recovered Reward Weights (GIRL)",
            output_path=str(output_dir / "reward_weights.png"),
            show=False
        )
        
        if verbose:
            print()
            print("="*80)
            print("GIRL TRAINING COMPLETE")
            print("="*80)
            print()
            print("Pipeline Summary:")
            print("-"*80)
            print(f"Sequences used:           {len(states)}")
            print(f"Sequence length:          {states.shape[1]} timesteps")
            print(f"State dimension:          {states.shape[2]} features")
            print(f"Reward features:          {rewards.shape[2]}")
            print(f"Cross-validation folds:   {args.n_folds if args.n_folds > 0 else 'train/test split'}")
            print(f"Random seed:              {args.random_state}")
            print(f"Hidden size:              {args.hidden_size}")
            print(f"Training epochs:          {args.num_epochs}")
            print(f"Device:                   {device}")
            print()
            
            print("Recovered Reward Weights:")
            print("-"*80)
            for i, name in enumerate(feature_names[:len(mean_weights)]):
                if args.n_folds > 0:
                    print(f"  {name:20s}: {mean_weights[i]:.4f} ± {std_weights[i]:.4f}")
                else:
                    print(f"  {name:20s}: {mean_weights[i]:.4f}")
            print()
            
            print("Results saved to:")
            print(f"  {output_dir / f'reward_weights_{timestamp}.csv'}")
            if args.n_folds > 0:
                print(f"  {output_dir / f'cross_validation_results_{timestamp}.csv'}")
            print(f"  {metadata_file}")
            print(f"  {plot_file}")
            print()
            print("="*80)
            print()
        
        return mean_weights, std_weights
    
    except FileNotFoundError as e:
        print()
        print("ERROR: SAR dataset not found")
        print("-"*80)
        print(str(e))
        print()
        print("Please run the preprocessing pipeline first to generate SAR dataset:")
        print("  python main.py --method girl --save")
        print()
        sys.exit(1)
    
    except Exception as e:
        print()
        print("ERROR: Pipeline failed")
        print("-"*80)
        print(str(e))
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Cross-validation for GIRL pipeline.

Performs k-fold cross-validation for robust reward weight estimation.
"""

import numpy as np
import torch
from sklearn.model_selection import KFold
from typing import Dict, Tuple, Optional

from ..bc.train_bc import train_bc_model
from ..gradients.compute_gradients import compute_policy_gradients, compute_mean_gradients
from ..irl.girl_solver import solve_girl


def cross_validate_girl(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    hidden_size: int = 64,
    num_epochs: int = 50,
    batch_size: int = 32,
    device: str = 'cpu',
    solver_method: str = 'quadprog',
    verbose: bool = True
) -> Dict:
    """
    Perform k-fold cross-validation for GIRL.
    
    For each fold:
    1. Split data into train/val
    2. Train BC model on train set
    3. Compute gradients on validation set
    4. Solve GIRL for reward weights
    
    Args:
        states: State sequences of shape (N, seq_len, state_dim)
        actions: Action sequences of shape (N, seq_len)
        rewards: Reward sequences of shape (N, seq_len, n_reward_features)
        n_splits: Number of folds
        random_state: Random seed
        hidden_size: BC model hidden size
        num_epochs: BC training epochs
        batch_size: Training batch size
        device: Device to use
        solver_method: GIRL solver method
        verbose: Print progress
    
    Returns:
        Dict with cross-validation results:
        - mean_weights: Mean reward weights across folds
        - std_weights: Standard deviation of weights
        - all_weights: List of weights from each fold
        - fold_results: List of per-fold results
    """
    if verbose:
        print("="*70)
        print("GIRL Cross-Validation")
        print("="*70)
        print(f"Dataset size: {len(states)}")
        print(f"Number of folds: {n_splits}")
        print(f"Random state: {random_state}")
        print(f"Device: {device}")
        print()
    
    # Set random seeds
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # Initialize cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Storage for results
    all_weights = []
    fold_results = []
    
    # Perform cross-validation
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(states)):
        if verbose:
            print("="*70)
            print(f"Fold {fold_idx + 1}/{n_splits}")
            print("="*70)
            print(f"Train size: {len(train_indices)}")
            print(f"Val size: {len(val_indices)}")
            print()
        
        # Split data
        train_states = states[train_indices]
        train_actions = actions[train_indices]
        val_states = states[val_indices]
        val_actions = actions[val_indices]
        val_rewards = rewards[val_indices]
        
        # Step 1: Train BC model
        if verbose:
            print(f"\n[Fold {fold_idx + 1}] Training BC model...")
        
        model, train_history = train_bc_model(
            states=train_states,
            actions=train_actions,
            state_dim=train_states.shape[2],
            hidden_size=hidden_size,
            batch_size=batch_size,
            num_epochs=num_epochs,
            device=device,
            verbose=verbose
        )
        
        # Step 2: Compute gradients on validation set
        if verbose:
            print(f"\n[Fold {fold_idx + 1}] Computing gradients...")
        
        gradients, feature_expectations = compute_policy_gradients(
            model=model,
            states=val_states,
            actions=val_actions,
            rewards=val_rewards,
            device=device,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Step 3: Compute mean gradients
        mean_gradients, mean_features = compute_mean_gradients(
            gradients, feature_expectations
        )
        
        # Step 4: Solve GIRL
        if verbose:
            print(f"\n[Fold {fold_idx + 1}] Solving GIRL...")
        
        reward_weights, solver_info = solve_girl(
            mean_gradients=mean_gradients,
            method=solver_method,
            verbose=verbose
        )
        
        # Store results
        all_weights.append(reward_weights)
        fold_results.append({
            'fold': fold_idx + 1,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'reward_weights': reward_weights,
            'mean_features': mean_features,
            'train_accuracy': train_history['train_accuracy'][-1],
            'train_loss': train_history['best_loss'],
            'solver_info': solver_info
        })
        
        if verbose:
            print(f"\n[Fold {fold_idx + 1}] Complete")
            print("Reward weights for this fold:")
            feature_names = ['stretch_index', 'pressure_index', 'space_score', 'line_height_rel']
            for i, (name, weight) in enumerate(zip(feature_names[:len(reward_weights)], reward_weights)):
                print(f"  {name:20s}: {weight:.4f}")
            print()
    
    # Compute statistics across folds
    all_weights_array = np.array(all_weights)
    mean_weights = all_weights_array.mean(axis=0)
    std_weights = all_weights_array.std(axis=0)
    
    # Summary results
    results = {
        'mean_weights': mean_weights,
        'std_weights': std_weights,
        'all_weights': all_weights,
        'fold_results': fold_results,
        'n_splits': n_splits,
        'random_state': random_state
    }
    
    # Print summary
    if verbose:
        print()
        print("="*70)
        print("Cross-Validation Summary")
        print("="*70)
        print(f"Number of folds: {n_splits}")
        print()
        print("Mean reward weights (±std):")
        feature_names = ['stretch_index', 'pressure_index', 'space_score', 'line_height_rel']
        for i, name in enumerate(feature_names[:len(mean_weights)]):
            print(f"  {name:20s}: {mean_weights[i]:.4f} ± {std_weights[i]:.4f}")
        print()
        
        # Print per-fold training metrics
        print("Per-fold training accuracy:")
        for fold_result in fold_results:
            print(f"  Fold {fold_result['fold']}: {fold_result['train_accuracy']:.4f}")
        print()
    
    return results


def simple_train_test_split(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    hidden_size: int = 64,
    num_epochs: int = 50,
    batch_size: int = 32,
    device: str = 'cpu',
    solver_method: str = 'quadprog',
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Simple train/test split for GIRL (faster than cross-validation).
    
    Args:
        states: State sequences
        actions: Action sequences
        rewards: Reward sequences
        test_size: Fraction for test set
        random_state: Random seed
        hidden_size: BC model hidden size
        num_epochs: Training epochs
        batch_size: Batch size
        device: Device
        solver_method: GIRL solver method
        verbose: Print progress
    
    Returns:
        Tuple of (reward_weights, results_dict)
    """
    if verbose:
        print("="*70)
        print("GIRL Train/Test Split")
        print("="*70)
    
    # Set random seeds
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # Split data
    n_samples = len(states)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[n_test:]
    test_indices = indices[:n_test]
    
    train_states = states[train_indices]
    train_actions = actions[train_indices]
    test_states = states[test_indices]
    test_actions = actions[test_indices]
    test_rewards = rewards[test_indices]
    
    if verbose:
        print(f"Train size: {len(train_indices)}")
        print(f"Test size: {len(test_indices)}")
        print()
    
    # Train BC model
    model, train_history = train_bc_model(
        states=train_states,
        actions=train_actions,
        state_dim=train_states.shape[2],
        hidden_size=hidden_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        verbose=verbose
    )
    
    # Compute gradients
    gradients, feature_expectations = compute_policy_gradients(
        model=model,
        states=test_states,
        actions=test_actions,
        rewards=test_rewards,
        device=device,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Mean gradients
    mean_gradients, mean_features = compute_mean_gradients(
        gradients, feature_expectations
    )
    
    # Solve GIRL
    reward_weights, solver_info = solve_girl(
        mean_gradients=mean_gradients,
        method=solver_method,
        verbose=verbose
    )
    
    results = {
        'reward_weights': reward_weights,
        'mean_features': mean_features,
        'train_accuracy': train_history['train_accuracy'][-1],
        'train_loss': train_history['best_loss'],
        'solver_info': solver_info
    }
    
    return reward_weights, results


if __name__ == "__main__":
    # Test cross-validation
    print("Testing cross-validation...")
    
    # Create synthetic data
    n_sequences = 100
    seq_len = 10
    state_dim = 21
    n_reward_features = 4
    
    np.random.seed(42)
    states = np.random.randn(n_sequences, seq_len, state_dim).astype(np.float32)
    actions = np.random.randint(0, 4, (n_sequences, seq_len))
    rewards = np.random.rand(n_sequences, seq_len, n_reward_features).astype(np.float32)
    
    # Run cross-validation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = cross_validate_girl(
        states=states,
        actions=actions,
        rewards=rewards,
        n_splits=3,
        hidden_size=32,
        num_epochs=10,
        batch_size=16,
        device=device,
        verbose=True
    )
    
    print(f"Mean weights: {results['mean_weights']}")
    print(f"Std weights: {results['std_weights']}")
    
    print("✓ Cross-validation test passed")

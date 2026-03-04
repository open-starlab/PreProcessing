"""
Compute policy gradients for GIRL (Goal-based Inverse Reinforcement Learning).

Computes ∂ log π(a|s) / ∂θ for each trajectory and reward feature.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset


def compute_policy_gradients(
    model,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    device: str = 'cpu',
    batch_size: int = 32,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute policy gradients ∂ log π(a|s) / ∂θ for GIRL.
    
    For each trajectory and reward feature, computes the gradient of
    log probability of the expert action with respect to model parameters.
    
    Args:
        model: Trained behavior cloning model
        states: State sequences of shape (N, seq_len, state_dim)
        actions: Action sequences of shape (N, seq_len)
        rewards: Reward features of shape (N, seq_len, n_reward_features)
        device: Device to compute on
        batch_size: Batch size for gradient computation
        verbose: Print progress
    
    Returns:
        Tuple of:
        - gradients: Shape (N, n_params, n_reward_features)
          Gradients for each sequence, parameter, and reward feature
        - feature_expectations: Shape (N, n_reward_features)
          Sum of rewards per trajectory per feature
    """
    if verbose:
        print("="*60)
        print("Computing Policy Gradients")
        print("="*60)
    
    model.eval()
    
    n_sequences, seq_len, state_dim = states.shape
    n_reward_features = rewards.shape[2]
    
    # Get number of parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        print(f"Sequences: {n_sequences}")
        print(f"Sequence length: {seq_len}")
        print(f"State dim: {state_dim}")
        print(f"Reward features: {n_reward_features}")
        print(f"Model parameters: {n_params}")
        print()
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)
    rewards_tensor = torch.FloatTensor(rewards).to(device)
    
    # Create dataloader
    dataset = TensorDataset(states_tensor, actions_tensor, rewards_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Storage for gradients
    all_gradients = []
    all_feature_expectations = []
    
    # Process each batch
    for batch_idx, (batch_states, batch_actions, batch_rewards) in enumerate(dataloader):
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)
        batch_rewards = batch_rewards.to(device)
        
        current_batch_size = batch_states.shape[0]
        
        # Compute feature expectations (sum of rewards per trajectory)
        feature_expectations = batch_rewards.sum(dim=1)  # (batch, n_reward_features)
        all_feature_expectations.append(feature_expectations.cpu().numpy())
        
        # Compute gradients for each sequence and reward feature
        batch_gradients = np.zeros((current_batch_size, n_params, n_reward_features))
        
        for seq_idx in range(current_batch_size):
            # Get single sequence
            seq_states = batch_states[seq_idx:seq_idx+1]  # (1, seq_len, state_dim)
            seq_actions = batch_actions[seq_idx:seq_idx+1]  # (1, seq_len)
            seq_rewards = batch_rewards[seq_idx]  # (seq_len, n_reward_features)
            
            # For each reward feature
            for reward_idx in range(n_reward_features):
                # Zero gradients
                model.zero_grad()
                
                # Forward pass and compute log probabilities
                log_probs = model.get_log_probs(seq_states, seq_actions)  # (1, seq_len)
                
                # Weight by rewards for this feature
                reward_weights = seq_rewards[:, reward_idx]  # (seq_len,)
                weighted_log_probs = log_probs[0] * reward_weights  # (seq_len,)
                
                # Compute gradient of weighted log probability
                loss = -weighted_log_probs.sum()  # Negative for gradient ascent
                loss.backward()
                
                # Extract gradients
                grad_vector = []
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_vector.append(param.grad.view(-1).detach().cpu().numpy())
                
                if grad_vector:
                    grad_vector = np.concatenate(grad_vector)
                    batch_gradients[seq_idx, :, reward_idx] = grad_vector
        
        all_gradients.append(batch_gradients)
        
        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * batch_size}/{n_sequences} sequences")
    
    # Concatenate all batches
    gradients = np.concatenate(all_gradients, axis=0)
    feature_expectations = np.concatenate(all_feature_expectations, axis=0)
    
    # Verify gradient tensor shape
    assert gradients.ndim == 3, f"Expected gradients.ndim=3, got {gradients.ndim}"
    assert gradients.shape[0] == n_sequences, \
        f"Gradient sequences {gradients.shape[0]} != input {n_sequences}"
    assert gradients.shape[1] == n_params, \
        f"Gradient params {gradients.shape[1]} != model params {n_params}"
    assert gradients.shape[2] == n_reward_features, \
        f"Gradient features {gradients.shape[2]} != reward features {n_reward_features}"
    
    if verbose:
        print()
        print("="*60)
        print("Gradient Tensor Verification")
        print("="*60)
        print(f"✓ Gradient tensor shape: {gradients.shape}")
        print(f"  Sequences:        {gradients.shape[0]}")
        print(f"  Model parameters: {gradients.shape[1]}")
        print(f"  Reward features:  {gradients.shape[2]}")
        print(f"✓ Feature expectations shape: {feature_expectations.shape}")
        print()
        print("="*60)
        print("Gradient Computation Complete")
        print("="*60)
        print(f"Gradients shape: {gradients.shape}")
        print(f"Feature expectations shape: {feature_expectations.shape}")
        print()
        
        # Print gradient statistics
        print("Gradient statistics per reward feature:")
        for i in range(n_reward_features):
            grad_norm = np.linalg.norm(gradients[:, :, i], axis=1).mean()
            print(f"  Feature {i}: mean gradient norm = {grad_norm:.6f}")
        print()
    
    return gradients, feature_expectations


def compute_mean_gradients(
    gradients: np.ndarray,
    feature_expectations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean gradients across trajectories.
    
    Args:
        gradients: Shape (N, n_params, n_reward_features)
        feature_expectations: Shape (N, n_reward_features)
    
    Returns:
        Tuple of:
        - mean_gradients: Shape (n_params, n_reward_features)
        - mean_features: Shape (n_reward_features,)
    """
    mean_gradients = gradients.mean(axis=0)
    mean_features = feature_expectations.mean(axis=0)
    
    return mean_gradients, mean_features


if __name__ == "__main__":
    # Test gradient computation
    print("Testing gradient computation...")
    
    from girl_pipeline.bc.model import create_defense_bc_model
    
    # Create synthetic data
    n_sequences = 20
    seq_len = 10
    state_dim = 21
    n_reward_features = 4
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    states = np.random.randn(n_sequences, seq_len, state_dim).astype(np.float32)
    actions = np.random.randint(0, 4, (n_sequences, seq_len))
    rewards = np.random.rand(n_sequences, seq_len, n_reward_features).astype(np.float32)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_defense_bc_model(
        state_dim=state_dim,
        hidden_size=32,
        device=device
    )
    
    # Compute gradients
    gradients, feature_expectations = compute_policy_gradients(
        model=model,
        states=states,
        actions=actions,
        rewards=rewards,
        device=device,
        batch_size=8,
        verbose=True
    )
    
    print(f"Gradients shape: {gradients.shape}")
    print(f"Feature expectations shape: {feature_expectations.shape}")
    
    # Compute mean
    mean_grads, mean_feats = compute_mean_gradients(gradients, feature_expectations)
    print(f"Mean gradients shape: {mean_grads.shape}")
    print(f"Mean features: {mean_feats}")
    
    print("✓ Gradient computation test passed")

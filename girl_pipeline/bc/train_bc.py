"""
Training script for Behavior Cloning model.

Trains DefenseBC model on expert demonstrations from SAR dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path

from .model import DefenseBC, create_defense_bc_model


class SARDataset(Dataset):
    """PyTorch Dataset for SAR data."""
    
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            states: State sequences of shape (N, seq_len, state_dim)
            actions: Action sequences of shape (N, seq_len)
        """
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def train_bc_model(
    states: np.ndarray,
    actions: np.ndarray,
    state_dim: int = 21,
    hidden_size: int = 64,
    num_layers: int = 1,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = 'cpu',
    verbose: bool = True,
    early_stopping_patience: int = 10
) -> Tuple[DefenseBC, Dict]:
    """
    Train behavior cloning model on SAR dataset.
    
    Args:
        states: State sequences of shape (N, seq_len, state_dim)
        actions: Action sequences of shape (N, seq_len)
        state_dim: State dimension
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: L2 regularization weight
        device: Device to train on ('cpu' or 'cuda')
        verbose: Print training progress
        early_stopping_patience: Patience for early stopping
    
    Returns:
        Tuple of:
        - trained_model: Trained DefenseBC model
        - training_history: Dict with loss history
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if verbose:
        print("="*60)
        print("Training Behavior Cloning Model")
        print("="*60)
        print(f"Dataset size: {len(states)}")
        print(f"State dim: {state_dim}")
        print(f"Hidden size: {hidden_size}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {device}")
        print()
    
    # Create dataset and dataloader
    dataset = SARDataset(states, actions)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    model = create_defense_bc_model(
        state_dim=state_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=verbose
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'best_loss': float('inf'),
        'best_epoch': 0
    }
    
    # Early stopping
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            # Forward pass
            action_logits, _ = model(batch_states)
            
            # Reshape for loss computation
            batch_size, seq_len, num_actions = action_logits.shape
            action_logits_flat = action_logits.view(-1, num_actions)
            actions_flat = batch_actions.view(-1)
            
            # Compute loss
            loss = criterion(action_logits_flat, actions_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            
            # Compute accuracy
            predictions = torch.argmax(action_logits_flat, dim=1)
            correct = (predictions == actions_flat).sum().item()
            epoch_correct += correct
            epoch_total += actions_flat.numel()
        
        # Average loss and accuracy
        avg_loss = epoch_loss / len(dataloader)
        avg_accuracy = epoch_correct / epoch_total
        
        history['train_loss'].append(avg_loss)
        history['train_accuracy'].append(avg_accuracy)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Check for best model
        if avg_loss < history['best_loss']:
            history['best_loss'] = avg_loss
            history['best_epoch'] = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Accuracy: {avg_accuracy:.4f} "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if verbose:
        print()
        print("="*60)
        print("Training Complete")
        print("="*60)
        print(f"Best epoch: {history['best_epoch'] + 1}")
        print(f"Best loss: {history['best_loss']:.4f}")
        print(f"Final accuracy: {history['train_accuracy'][-1]:.4f}")
        print()
    
    return model, history


def evaluate_bc_model(
    model: DefenseBC,
    states: np.ndarray,
    actions: np.ndarray,
    device: str = 'cpu',
    batch_size: int = 32
) -> Dict:
    """
    Evaluate BC model on dataset.
    
    Args:
        model: Trained DefenseBC model
        states: State sequences
        actions: Action sequences
        device: Device
        batch_size: Batch size for evaluation
    
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    
    dataset = SARDataset(states, actions)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            # Forward pass
            action_logits, _ = model(batch_states)
            
            # Reshape
            batch_size, seq_len, num_actions = action_logits.shape
            action_logits_flat = action_logits.view(-1, num_actions)
            actions_flat = batch_actions.view(-1)
            
            # Compute loss
            loss = criterion(action_logits_flat, actions_flat)
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = torch.argmax(action_logits_flat, dim=1)
            correct = (predictions == actions_flat).sum().item()
            total_correct += correct
            total_samples += actions_flat.numel()
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'total_correct': total_correct,
        'total_samples': total_samples
    }


if __name__ == "__main__":
    # Test BC training
    print("Testing BC training...")
    
    # Create synthetic data
    n_sequences = 100
    seq_len = 10
    state_dim = 21
    
    np.random.seed(42)
    states = np.random.randn(n_sequences, seq_len, state_dim).astype(np.float32)
    actions = np.random.randint(0, 4, (n_sequences, seq_len))
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, history = train_bc_model(
        states=states,
        actions=actions,
        batch_size=16,
        num_epochs=20,
        device=device,
        verbose=True
    )
    
    # Evaluate
    metrics = evaluate_bc_model(model, states, actions, device=device)
    print(f"\nEvaluation metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Loss: {metrics['loss']:.4f}")
    
    print("✓ BC training test passed")

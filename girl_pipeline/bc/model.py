"""
Behavior Cloning Model for Defensive Line Actions.

Uses LSTM architecture to learn policy from expert demonstrations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DefenseBC(nn.Module):
    """
    Behavior Cloning model for defensive line actions.
    
    Architecture:
        - LSTM to process state sequences
        - Fully connected layers for action prediction
        - Softmax output over 4 actions
    
    Input:
        State sequence of shape (batch, seq_len, state_dim)
        where state_dim = 21 (defender + ball features)
    
    Output:
        Action probabilities of shape (batch, seq_len, 4)
    """
    
    def __init__(
        self,
        state_dim: int = 21,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_actions: int = 4,
        dropout: float = 0.2
    ):
        """
        Initialize DefenseBC model.
        
        Args:
            state_dim: Dimension of state features (default: 21)
            hidden_size: LSTM hidden size (default: 64)
            num_layers: Number of LSTM layers (default: 1)
            num_actions: Number of discrete actions (default: 4)
            dropout: Dropout rate (default: 0.2)
        """
        super(DefenseBC, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_actions = num_actions
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        states: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            states: State sequences of shape (batch, seq_len, state_dim)
            hidden: Optional hidden state tuple (h, c)
        
        Returns:
            Tuple of:
            - action_logits: Shape (batch, seq_len, num_actions)
            - hidden: Updated hidden state tuple
        """
        batch_size, seq_len, _ = states.shape
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(states, hidden)  # (batch, seq_len, hidden)
        
        # Apply layer norm
        lstm_out = self.layer_norm(lstm_out)
        
        # Fully connected layers (applied to each timestep)
        x = F.relu(self.fc1(lstm_out))  # (batch, seq_len, hidden)
        x = self.dropout(x)
        action_logits = self.fc2(x)  # (batch, seq_len, num_actions)
        
        return action_logits, hidden
    
    def predict_actions(
        self,
        states: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Predict action probabilities.
        
        Args:
            states: State sequences of shape (batch, seq_len, state_dim)
            temperature: Temperature for softmax (default: 1.0)
        
        Returns:
            Action probabilities of shape (batch, seq_len, num_actions)
        """
        action_logits, _ = self.forward(states)
        action_probs = F.softmax(action_logits / temperature, dim=-1)
        return action_probs
    
    def sample_actions(
        self,
        states: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample actions from policy.
        
        Args:
            states: State sequences of shape (batch, seq_len, state_dim)
            deterministic: If True, return argmax actions
        
        Returns:
            Sampled actions of shape (batch, seq_len)
        """
        action_probs = self.predict_actions(states)
        
        if deterministic:
            actions = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from categorical distribution
            actions = torch.distributions.Categorical(action_probs).sample()
        
        return actions
    
    def get_log_probs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Get log probabilities of taken actions.
        
        Args:
            states: State sequences of shape (batch, seq_len, state_dim)
            actions: Taken actions of shape (batch, seq_len)
        
        Returns:
            Log probabilities of shape (batch, seq_len)
        """
        action_logits, _ = self.forward(states)
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        # Gather log probs for taken actions
        batch_size, seq_len, _ = action_logits.shape
        actions_flat = actions.view(-1)
        log_probs_flat = log_probs.view(-1, self.num_actions)
        
        selected_log_probs = log_probs_flat.gather(1, actions_flat.unsqueeze(1))
        selected_log_probs = selected_log_probs.view(batch_size, seq_len)
        
        return selected_log_probs
    
    def init_hidden(self, batch_size: int, device: str = 'cpu'):
        """
        Initialize hidden state for LSTM.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
        
        Returns:
            Tuple of (h0, c0) hidden states
        """
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size
        ).to(device)
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size
        ).to(device)
        return (h0, c0)


def create_defense_bc_model(
    state_dim: int = 21,
    hidden_size: int = 64,
    num_layers: int = 1,
    device: str = 'cpu'
) -> DefenseBC:
    """
    Create and initialize a DefenseBC model.
    
    Args:
        state_dim: State dimension
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        device: Device to place model on
    
    Returns:
        Initialized DefenseBC model
    """
    model = DefenseBC(
        state_dim=state_dim,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    model = model.to(device)
    
    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'lstm' in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.kaiming_normal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing DefenseBC model...")
    
    # Set random seed
    torch.manual_seed(42)
    
    # Create model
    model = create_defense_bc_model(
        state_dim=21,
        hidden_size=64,
        num_layers=1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    state_dim = 21
    
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randint(0, 4, (batch_size, seq_len))
    
    # Forward pass
    action_logits, _ = model(states)
    print(f"Action logits shape: {action_logits.shape}")
    
    # Predict actions
    action_probs = model.predict_actions(states)
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Action prob sum: {action_probs[0, 0].sum().item():.4f}")
    
    # Get log probs
    log_probs = model.get_log_probs(states, actions)
    print(f"Log probs shape: {log_probs.shape}")
    
    print("✓ Model test passed")

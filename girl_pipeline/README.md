# GIRL Pipeline - Goal-based Inverse Reinforcement Learning

Complete inverse reinforcement learning pipeline for recovering reward weights from expert defensive demonstrations in football tracking data.

## Overview

The GIRL (Goal-based Inverse Reinforcement Learning) pipeline implements a complete IRL system that:

1. Loads SAR (State-Action-Reward) datasets from the preprocessing pipeline
2. Trains a Behavior Cloning (BC) model using LSTM to imitate expert demonstrations
3. Computes policy gradients ∂log π(a|s)/∂θ for each trajectory
4. Solves a Quadratic Programming (QP) problem to recover reward weights

## Architecture

### Components

- **data_loader.py**: Load and validate SAR dataset from pickles
- **bc/model.py**: DefenseBC LSTM model for behavior cloning (21 state features → 4 actions)
- **bc/train_bc.py**: BC training with CrossEntropyLoss, early stopping, learning rate scheduling
- **gradients/compute_gradients.py**: Compute policy gradients weighted by reward features
- **irl/girl_solver.py**: Solve QP to recover reward weights (quadprog, cvxopt, analytical methods)
- **utils/cross_validation.py**: K-fold cross-validation for robust weight estimation
- **run_girl_pipeline.py**: Main entry point orchestrating the entire pipeline

### Data Format

**Input (from preprocessing/output/sar/):**
- `states.npy`: Shape (N, 10, 21) - N sequences, 10 timesteps, 21 state features per defender
- `actions.npy`: Shape (N, 10) - Actions encoded as {0: backward, 1: forward, 2: compress, 3: expand}
- `rewards.npy`: Shape (N, 10, 4) - Four reward features per timestep
- `metadata.pickle`: Dict with action_names, feature_names, config, etc.

**Output:**
- Recovered reward weights for each feature (sums to 1, all non-negative)
- Mean ± std across cross-validation folds

## Usage

### Basic Usage

```bash
# Run with default settings (5-fold CV)
python girl_pipeline/run_girl_pipeline.py

# Specify custom parameters
python girl_pipeline/run_girl_pipeline.py \
    --n_folds 10 \
    --hidden_size 128 \
    --num_epochs 100 \
    --device cuda \
    --solver_method quadprog
```

### CLI Arguments

**Data:**
- `--sar_dir`: Directory containing SAR dataset (default: `./preprocessing/output/sar`)
- `--config_suffix`: Optional config suffix for specific SAR dataset

**Training:**
- `--n_folds`: Number of folds for cross-validation (default: 5, use 0 for train/test split)
- `--test_size`: Test size for train/test split (default: 0.2, only used if n_folds=0)
- `--hidden_size`: LSTM hidden size for BC model (default: 64)
- `--num_epochs`: Number of training epochs for BC (default: 50)
- `--batch_size`: Training batch size (default: 32)
- `--device`: Device to use (`auto`, `cpu`, `cuda`, default: auto)

**GIRL:**
- `--solver_method`: QP solver method (`quadprog`, `cvxopt`, `analytical`, default: quadprog)

**Misc:**
- `--random_state`: Random seed for reproducibility (default: 42)
- `--quiet`: Suppress verbose output

### Example Output

```
================================================================================
GIRL PIPELINE - Goal-based Inverse Reinforcement Learning
================================================================================

Recovering reward weights from expert defensive demonstrations

STEP 1: Loading SAR Dataset
--------------------------------------------------------------------------------
Loaded SAR dataset successfully!
  States shape:  (500, 10, 21)
  Actions shape: (500, 10)
  Rewards shape: (500, 10, 4)

Action distribution:
  backward    :   1234 ( 24.7%)
  forward     :   1876 ( 37.5%)
  compress    :   1012 ( 20.2%)
  expand      :    878 ( 17.6%)

STEP 2: Running GIRL with Cross-Validation
--------------------------------------------------------------------------------
Fold 1/5:
  Training BC model... BC Accuracy: 72.5%
  Computing gradients... Done (400 sequences)
  Solving GIRL... Done

Fold 2/5:
  [...]

================================================================================
FINAL RESULTS
================================================================================

Recovered Reward Weights:
--------------------------------------------------------------------------------
  stretch_index       : 0.2843 ± 0.0142
  pressure_index      : 0.3217 ± 0.0178
  space_score         : 0.2456 ± 0.0156
  line_height_rel     : 0.1484 ± 0.0089

Sum of weights: 1.000000

Interpretation:
--------------------------------------------------------------------------------
Most important feature: pressure_index (weight: 0.3217)

Feature importance ranking:
  1. pressure_index       : 0.3217
  2. stretch_index        : 0.2843
  3. space_score          : 0.2456
  4. line_height_rel      : 0.1484

================================================================================
GIRL PIPELINE COMPLETE
================================================================================
```

## Programmatic API

### Load and Run Pipeline

```python
from girl_pipeline import (
    load_sar_dataset,
    cross_validate_girl,
)

# Load data
states, actions, rewards, metadata = load_sar_dataset()

# Run cross-validation
results = cross_validate_girl(
    states=states,
    actions=actions,
    rewards=rewards,
    n_splits=5,
    hidden_size=64,
    num_epochs=50,
    device='cpu',
    solver_method='quadprog',
    verbose=True
)

# Extract weights
mean_weights = results['mean_weights']
std_weights = results['std_weights']

print(f"Weights: {mean_weights} ± {std_weights}")
```

### Individual Components

```python
from girl_pipeline import (
    DefenseBC,
    train_bc_model,
    compute_policy_gradients,
    solve_girl,
)

# 1. Train BC model
model, history = train_bc_model(
    states=states,
    actions=actions,
    hidden_size=64,
    num_epochs=50,
    device='cpu'
)

# 2. Compute gradients
gradients, feature_expectations = compute_policy_gradients(
    model=model,
    states=states,
    actions=actions,
    rewards=rewards,
    device='cpu'
)

# 3. Solve GIRL
weights, status = solve_girl(
    gradients=gradients,
    solver_method='quadprog'
)

print(f"Recovered weights: {weights}")
```

## Dependencies

**Required:**
- numpy
- torch
- scipy
- scikit-learn

**Optional (for QP solvers):**
- quadprog (recommended)
- cvxopt

Install using:
```bash
pip install numpy torch scipy scikit-learn quadprog
```

## Algorithm Details

### GIRL (Goal-based IRL)

The GIRL algorithm recovers reward weights θ by solving:

```
minimize:    θᵀ G G θ
subject to:  θ ≥ 0
             Σ θᵢ = 1
```

Where:
- G is the matrix of policy gradients ∂log π(a|s)/∂θ
- Each row corresponds to a trajectory
- Each column corresponds to a reward feature

### Behavior Cloning

The BC model uses an LSTM architecture:
- Input: 21 state features (defender positions, velocities, ball state, etc.)
- LSTM: hidden_size=64 (default)
- Output: Softmax over 4 actions

Loss: CrossEntropyLoss with expert actions as targets

### Policy Gradients

For each trajectory:
1. Forward pass through BC model to get log probabilities
2. Weight log probs by reward features: ∇ = ∂log π(a|s)/∂θ · R
3. Aggregate across time steps and trajectories

## Troubleshooting

**SAR dataset not found:**
```bash
# Run preprocessing pipeline first with --method girl
python main.py --method girl --save
```

**Out of memory:**
```bash
# Reduce batch size and hidden size
python girl_pipeline/run_girl_pipeline.py --batch_size 16 --hidden_size 32
```

**QP solver fails:**
```bash
# Try different solver method
python girl_pipeline/run_girl_pipeline.py --solver_method analytical
```

**Poor BC accuracy (<60%):**
- Increase training epochs: `--num_epochs 100`
- Increase model capacity: `--hidden_size 128`
- Check data quality and class balance

## References

1. Ratliff, N., Bagnell, J. A., & Zinkevich, M. (2006). Maximum margin planning.
2. Abbeel, P., & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning.

## License

See LICENSE file in root directory.

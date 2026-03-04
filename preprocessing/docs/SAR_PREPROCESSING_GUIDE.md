# SAR Preprocessing Module Documentation

## Overview

The **SAR (State-Action-Reward) Preprocessing Module** (`sar_preprocessing.py`) converts raw match tracking data into explicit state-action-reward tuples suitable for:

- **Inverse Reinforcement Learning (IRL)**
- **Behavioral Cloning (BC)**
- **Deep Reinforcement Learning (DRL)**
- Machine learning-based defensive analysis

## Key Features

### 1. **Attack Direction Computation** ✓
- Computes attacking team direction vector
- Returns: direction angle (0-360°), attack intensity, attack zone
- Identifies left wing, center, or right wing attacks

### 2. **21-Dimensional State Representation**
```
State = [Defenders (16-dim) + Ball (2-dim) + Attack (3-dim)]

Defenders (16-dim):
  - 8 defenders × 2 coordinates (x, y)
  - Sorted by x-coordinate (left to right)
  
Ball (2-dim):
  - x, y coordinates
  
Attack Direction (3-dim):
  - direction_angle: angle of attack centroid (0-360°)
  - attack_intensity: number of attackers in final third
  - attack_zone: 0=left, 1=center, 2=right
```

### 3. **Action Labels** (4 discrete actions)
```
0: backward   - Retreat toward goal
1: forward    - Advance toward attacking team
2: compress   - Reduce defensive line width
3: expand     - Increase defensive line width
```

### 4. **Reward Features** (4-dimensional)
```
Reward = [space_score, stretch_index, pressure_index, line_height_rel]

space_score:
  - Zone-weighted attacker space dominance (0 = high compression)
  - Exponential decay based on distance to nearest defender
  
stretch_index:
  - Convex hull area of defensive line
  - Measures lateral dispersion of defenders
  
pressure_index:
  - Count of attackers with nearby defenders (<3m)
  - Proxy for defensive pressure effectiveness
  
line_height_rel_ball:
  - Relative positioning: ball_x - average_defender_x
  - Positive = defending above ball; negative = defending below
```

## File Structure

### Main Module: `sar_preprocessing.py`

#### Core Functions:

**1. `compute_attack_direction(state) → (angle, intensity, zone)`**
```python
# Compute attacking team direction
angle, intensity, zone = compute_attack_direction(state)
print(f"Attack at {angle:.1f}° with {intensity} attackers in zone {zone}")
```

**2. `build_full_state_features(state) → np.ndarray(21,)`**
```python
# Create 21-dim state vector
state_vector = build_full_state_features(state)
print(f"State shape: {state_vector.shape}")  # (21,)
```

**3. `extract_reward_features(state) → np.ndarray(4,)`**
```python
# Extract 4-dim reward features
reward = extract_reward_features(state)
print(f"Space score: {reward[0]:.3f}, Stretch: {reward[1]:.3f}")
```

**4. `create_sar_sequences(events_list, sequence_length=10) → List[Dict]`**
```python
# Create SAR sequences from event list
sequences = create_sar_sequences(events_list, sequence_length=10)

for seq in sequences:
    print(f"{seq['outcome']}: {seq['length']} steps")
    for sar in seq['sar_tuples']:
        print(f"  Action: {sar.action_label}, Reward: {sar.reward}")
```

**5. `extract_ml_arrays(sequences) → (X, A, R)`**
```python
# Convert sequences to numpy arrays for ML
X, A, R = extract_ml_arrays(sequences)

print(f"States:  {X.shape}")    # (N, 21)
print(f"Actions: {A.shape}")    # (N,)
print(f"Rewards: {R.shape}")    # (N, 4)
```

#### Data Classes:

**SARTuple** - Single (S, A, R) transition
```python
sar = SARTuple(
    state=state_vector,            # 21-dim
    action=1,                       # 0-3
    action_label="forward",         # "backward", "forward", "compress", "expand"
    reward=reward_vector,           # 4-dim
    metadata={                      # Optional context
        "ball_position": {"x": 50, "y": 50},
        "attack_direction": 45.0,
        "attack_intensity": 2,
        "attack_zone": 1
    }
)

print(sar)  # SARTuple(action=forward, space_score=0.730, stretch=0.450)
```

### Integration Example: `sar_integration_example.py`

Shows how to use SAR preprocessing with the pipeline:

```python
from sar_integration_example import create_sar_dataset_from_match
from config import PipelineConfig
from pathlib import Path

# Configure pipeline
config = PipelineConfig(
    data_match='all_matches',
    back_four='all_players',
    sequence_type='all_defensive',
    reward_features='5_features'
)

# Process single match
match_dir = Path('./Defense_line/Laliga2023/24/1018887')
dataset = create_sar_dataset_from_match(match_dir, config)

# Access components
print(f"States:  {dataset['X'].shape}")    # (N, 21)
print(f"Actions: {dataset['A'].shape}")    # (N,)
print(f"Rewards: {dataset['R'].shape}")    # (N, 4)
```

## Usage Workflow

### Step 1: Load & Configure
```python
from config import PipelineConfig
from preprocessing import preprocess_match

config = PipelineConfig(...)
match_data = preprocess_match(match_dir, config)
```

### Step 2: Create SAR Sequences
```python
from sar_preprocessing import create_sar_sequences, extract_ml_arrays

# Get events from match_data
events_with_state = [{"state": event} for event in match_data["events"]]

# Create SAR sequences
sequences = create_sar_sequences(events_with_state, sequence_length=10)

# Extract ML-ready arrays
X, A, R = extract_ml_arrays(sequences)
```

### Step 3: Use for ML
```python
# For behavioral cloning
model = train_bc(X, A)

# For IRL
irl_weights = invert_rewards(X, A, R)

# For RL
policy = train_rl(X, A, R)
```

## Data Dimensionality

| Component | Dimension | Description |
|-----------|-----------|-------------|
| **State (S)** | 21 | Defenders(16) + Ball(2) + Attack(3) |
| **Action (A)** | 1 | Discrete: {0,1,2,3} |
| **Reward (R)** | 4 | space_score, stretch_idx, pressure_idx, line_height |
| **Sequence** | 10 frames | Configured via `sequence_length` |
| **Dataset** | Variable | Depends on number of matches |

## Example Output

### Single Match Dataset
```
Match: 1018887 (Almería vs Rayo Vallecano)
Generated sequences: 607
State shape:   (6070, 21)
Action shape:  (6070,)
Reward shape:  (6070, 4)

Success rate: 85.3%
Action distribution:
  backward : 1823
  forward  :  945
  compress :  980
  expand   : 1322
```

### Command Line Usage
```bash
python sar_integration_example.py
```

## Integration with Existing Pipeline

The SAR module **seamlessly integrates** with existing components:

```
Raw Data → preprocessing.py → sar_preprocessing.py → ML Models
              ↓
          feature_engineering.py (for reward computation)
```

## API Reference

### Attack Direction
- `compute_attack_direction(state) → (float, int, int)`
  - Returns: (angle_0_360, attackers_in_final_third, zone_0_1_2)

### State Features
- `build_full_state_features(state) → np.ndarray(21,)`
  - Input: state dict
  - Output: 21-dim feature vector

### Reward Features
- `extract_reward_features(state) → np.ndarray(4,)`
  - Input: state dict
  - Output: 4-dim reward vector

### Sequences
- `create_sar_sequences(events, length=10, skip=1, overlap=0) → List[Dict]`
  - Input: event list, sequence parameters
  - Output: sequences with SAR tuples

### ML Arrays
- `extract_ml_arrays(sequences) → (X, A, R)`
  - Input: sequences
  - Output: (states, actions, rewards) as numpy arrays

## Notes

1. **Attack Direction** is NEW - identifies where attacking team is positioned
2. **State includes attack direction** - provides context for defensive decisions
3. **Reward features unchanged** - space_score, stretch_index, pressure_index, line_height
4. **Actions are majority-voted** from individual defender actions
5. **Sequences labeled** as success/failure based on defensive outcome

---

For questions or extensions, see `sar_preprocessing.py` source code.

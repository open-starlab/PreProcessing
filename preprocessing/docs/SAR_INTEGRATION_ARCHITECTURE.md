# SAR Dataset Integration Architecture

## Overview

The preprocessing pipeline now supports **two execution modes** controlled by the `method` flag in `config.py`:

1. **Feature Computation** (`method="feature_computation"`)
   - Produces statistical/ML features (space_score, pressure_index, etc.)
   - Outputs: CSV feature tables, summaries, processing logs
   - Use case: Feature analysis, statistical testing, ML baseline models

2. **GIRL IRL** (`method="girl"`)
   - Produces SAR (State-Action-Reward) dataset
   - Outputs: Pickle tensors (states, actions, rewards, metadata)
   - Use case: Inverse Reinforcement Learning, Behavioral Cloning

---

## High-Level Architecture

```
Raw LaLiga Data (events.csv, tracking.json, match.json)
        │
        ▼
    main.py (orchestrator)
        │
        ▼
    preprocessing.py (unified pipeline)
        │
        ▼
    feature_engineering.py (routing layer)
        │
        ├─[method="feature_computation"]──→ feature_computation.py
        │   └─→ CSV outputs (features, summaries, logs)
        │
        └─[method="girl"]────────────────→ reward_features.py
            │                              ↓
            └─────────────────────→ sar_preprocessing.py
                                   └─→ Pickle tensors (SAR dataset)
```

---

## Flag-Based Routing

### Configuration in `config.py`

```python
@dataclass
class PipelineConfig:
    data_match: str                      # "barcelona_madrid" or "all_matches"
    back_four: str                       # "back_four" or "all_players"
    sequence_type: str                   # "negative_transition" or "all_defensive"
    reward_features: str                 # "4_features" or "5_features"
    method: str                          # "feature_computation" or "girl" ← KEY FLAG
```

### Execution Examples

**Feature Computation Mode:**
```bash
python main.py \
  --data_match barcelona_madrid \
  --back_four back_four \
  --sequence_type negative_transition \
  --reward_features 4_features \
  --method feature_computation
```

**GIRL Mode:**
```bash
python main.py \
  --data_match all \
  --back_four all_players \
  --sequence_type all_defensive \
  --reward_features 4_features \
  --method girl
```

---

## Feature Computation Branch (`method="feature_computation"`)

### Pipeline Flow
```
preprocessing.py 
  ↓
feature_engineering.py 
  ↓
feature_computation.py (extract spatial/temporal features)
  ↓
CSV outputs
```

### Output Location
```
output/
  feature_computation/
    transition_features_{flags}.csv      # One row per sequence
    action_rates_{flags}.csv             # Per-sequence action distributions
    sequence_summary_{flags}.csv         # Aggregate statistics
```

### Output Columns
- **Metadata**: `match_id`, `home_team`, `away_team`, `transition_idx`, `label`
- **Features**: `space_score_mean`, `pressure_index_mean`, `stretch_index_mean`, etc.
- **Actions**: `action_backward_rate`, `action_forward_rate`, `action_compress_rate`, `action_expand_rate`
- **State**: `state_dim` (18), `state_steps`

---

## GIRL IRL Branch (`method="girl"`)

### Pipeline Flow
```
preprocessing.py 
  ↓
feature_engineering.py → _build_state_from_transition_frame()
  ↓
sar_preprocessing.py
  ├─ create_sar_sequences() [builds (s, a, r) tuples]
  ├─ extract_ml_sequences() [structures to (N, T, D) tensors]
  └─→ Pickle save
```

### Output Location
```
output/
  sar/
    states_{flags}.pkl                  # Shape (N, 10, 21)
    actions_{flags}.pkl                 # Shape (N, 10)
    rewards_{flags}.pkl                 # Shape (N, 10, 4)
    metadata_{flags}.pkl                # Summary dict
```

### Tensor Specifications

**States**: `(N_sequences, T=10, D=21)`
```
[x₁, y₁, ..., x₈, y₈,        # 8 defenders (16-dim), sorted by x
 ball_x, ball_y,              # Ball position (2-dim)
 attack_angle,                # Attack direction (1-dim)
 attack_intensity,            # Attackers in final third (1-dim)
 attack_zone]                 # Attack side: left/center/right (1-dim)
```

**Actions**: `(N_sequences, T=10)` with values in {0, 1, 2, 3}
```
0 = backward    (retreat)
1 = forward     (advance)
2 = compress    (tighten formation)
3 = expand      (widen formation)
```

**Rewards**: `(N_sequences, T=10, 4)` for GIRL reward features
```
[stretch_index,      # Convex hull area (line dispersion)
 pressure_index,     # Count of pressured attackers
 space_score,        # Zone-weighted space control
 line_height_rel]    # Relative positioning to ball
```

**Metadata**: Dict with per-match and aggregated statistics
```python
{
    "n_sequences": 1250,
    "seq_length": 10,
    "state_dim": 21,
    "action_names": {0: "backward", 1: "forward", 2: "compress", 3: "expand"},
    "reward_features": "4_features",
    "success_rate": 0.62,
    "per_match_metadata": [...]
}
```

---

## Key Implementation Details

### 1. State Building (`feature_engineering.py`)

New function: `_build_state_from_transition_frame()`

```python
def _build_state_from_transition_frame(
    frame_row: pd.Series,
    defending_role: str,
    attacking_role: str
) -> Dict:
    """Converts tracking frame row to state dict for SAR generation."""
    state = {
        'players': [
            {'position': {'x': x_i, 'y': y_i}, 'team': role, ...}
            for each player in frame
        ],
        'ball': {'position': {'x': ball_x, 'y': ball_y}}
    }
    return state
```

### 2. SAR Sequence Creation (`sar_preprocessing.py`)

For each transition (10-frame window):
1. Extract per-frame state using `build_full_state_features()`
2. Classify defender actions using majority voting
3. Compute reward features (4-dim)
4. Stack into (T, 21), (T,), (T, 4) arrays
5. Return SARTuple list

New function: `extract_ml_sequences()`
- Takes list of SAR sequences
- Preserves sequence structure (N, T, D)
- Returns stacked numpy arrays + metadata

### 3. SAR Dataset Save (`main.py`)

New function: `save_sar_dataset()`
- Concatenates all match sequences
- Saves four pickle files: states, actions, rewards, metadata
- Creates `output/sar/` directory

### 4. Pipeline Branching (`main.py`)

Modified `run_pipeline()`:
```python
if config.method == "girl":
    # GIRL branch: build SAR sequences
    sar_sequences = create_sar_sequences(events_with_state, seq_length=10)
    X, A, R, metadata = extract_ml_sequences(sar_sequences)
    sar_paths = save_sar_dataset(...)
    return pd.DataFrame(), sar_paths
else:
    # Feature Computation branch: standard feature extraction
    compute_match_features() → DataFrame
    save_summary_statistics()
    return result_df, None
```

---

## Workflow for GIRL Training

1. **Generate SAR Dataset** (preprocessing repository)
   ```bash
   cd cleaned/
   python main.py --method girl --data_match all --reward_features 4_features
   ```
   Produces: `output/sar/states_*.pkl`, `actions_*.pkl`, `rewards_*.pkl`

2. **Load SAR in GIRL Repository** (separate project)
   ```python
   import pickle
   with open('output/sar/states.pkl', 'rb') as f:
       states = pickle.load(f)  # Shape: (N, 10, 21)
   with open('output/sar/actions.pkl', 'rb') as f:
       actions = pickle.load(f)  # Shape: (N, 10)
   with open('output/sar/rewards.pkl', 'rb') as f:
       rewards = pickle.load(f)  # Shape: (N, 10, 4)
   
   # Train GIRL or BC model
   model.train(states, actions, rewards)
   ```

3. **Reproducibility**
   - SAR dataset is deterministic (same input → same tensor)
   - Metadata dict captures configuration for traceability
   - Can regenerate from raw data anytime

---

## Backward Compatibility

- **Old code**: Still works with `method="feature_computation"`
- **CSV outputs**: Unchanged for feature computation mode
- **Preprocessing**: Shared by both branches (no duplication)

---

## Design Constraints (Preserved)

✅ **Action classifier**: No changes  
✅ **State vector definition**: No changes  
✅ **Sequence length T=10**: Fixed  
✅ **Defender count**: Variable per frame, fixed in output (8 + padding)  

---

## Testing

Run validation:
```bash
python validate_pipeline.py
```

Expected results:
- Config: ✓ Loads correctly
- State/Action functions: ✓ Accessible
- SAR extraction: ✓ Produces correct shapes
- Output saving: ✓ Pickles created

---

## Summary

The SAR dataset is now produced **inside preprocessing** and ready for immediate use in GIRL or other IRL/BC algorithms. No changes needed in downstream code—just load the pickles and train.

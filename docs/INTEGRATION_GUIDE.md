# Feature Computation Integration Guide

## Overview

The pipeline now supports **two feature computation methods** with **automatic routing** based on configuration flags.

---

## 🎯 Flag-Based Routing

### **Flag 3: `sequence_type`** (Controls preprocessing approach)

#### Option 1: `sequence_type="negative_transition"`
- **Preprocessing**: Uses `preprocess_with_synchronization()`
- **Features from**: Transitions after possession loss
- **Includes**: Event-tracking sync, back-four detection, restart handling

#### Option 2: `sequence_type="all_defensive"`  
- **Preprocessing**: Uses `preprocess_match()`
- **Features from**: 10-frame defensive sequences (anytime)
- **Basic**: Standard defensive analysis

---

### **Flag 5: `method`** (Controls feature computation)

#### Option 1: `method="girl"`
- **Module**: `reward_features.py`
- **Features**: GIRL reward features
  - 4 features: `[space_score, stretch_index, pressure_index, line_height_relative]`
  - 5 features: adds `line_height_absolute`
- **Controlled by**: `reward_features` flag (Flag 4)
- **Use case**: State-based reward learning

#### Option 2: `method="feature_computation"`
- **Module**: `feature_computation.py`
- **Features**: Advanced tactical features (20+)
  - Convex hull area, PDA, compactness
  - Space score (4 weighted zones)
  - Pressure index (within radius)
  - Line height (3 variants with aggregations)
  - Line velocity, stability, defense-attack correlation
- **Controlled by**: `back_four` flag (Flag 2)
- **Use case**: Detailed tactical analysis

---

## 📋 Complete Flag Decision Tree

```
START
  │
  ├─ sequence_type?
  │   ├─ "negative_transition" → preprocess_with_synchronization()
  │   │                           ├─ Event-tracking sync
  │   │                           ├─ Defender detection from events
  │   │                           ├─ Transition extraction (turnovers)
  │   │                           └─ Back-four identification
  │   │
  │   └─ "all_defensive" → preprocess_match()
  │                        └─ Standard 10-frame defensive sequences
  │
  ├─ method?
  │   ├─ "girl" → reward_features.py
  │   │            ├─ reward_features?
  │   │            │   ├─ "4_features" → [space, stretch, pressure, line_rel]
  │   │            │   └─ "5_features" → + line_height_absolute
  │   │            └─ Uses all defenders (excluding GK)
  │   │
  │   └─ "feature_computation" → feature_computation.py
  │                               ├─ back_four?
  │                               │   ├─ "back_four" → Use 4 deepest defenders
  │                               │   └─ "all_players" → Use all defenders
  │                               └─ Computes 20+ advanced features
  │
  └─ OUTPUT: features_df with computed features
```

---

## 🚀 Usage Examples

### Example 1: GIRL with Transitions (4 features)
```python
from preprocessing.config import PipelineConfig
from preprocessing.main import run_pipeline

config = PipelineConfig(
    data_match="barcelona_madrid",
    back_four="all_players",           # Not used by GIRL
    sequence_type="negative_transition", # Use transitions
    reward_features="4_features",       # 4 GIRL features
    method="girl"                       # Use reward_features.py
)

features_df, sar_paths = run_pipeline(config)
# Output: space_score, stretch_index, pressure_index, line_height_relative
```

### Example 2: Advanced Features with Back-Four
```python
config = PipelineConfig(
    data_match="all_matches",
    back_four="back_four",              # Use 4 deepest defenders
    sequence_type="negative_transition", # Use transitions
    reward_features="5_features",       # Not used by feature_computation
    method="feature_computation"        # Use feature_computation.py
)

features_df, sar_paths = run_pipeline(config)
# Output: 20+ features (stretch_index_mean, stretch_index_std, pressure_index_mean, etc.)
```

### Example 3: GIRL with All Defensive Sequences
```python
config = PipelineConfig(
    data_match="barcelona_madrid",
    back_four="all_players",
    sequence_type="all_defensive",      # Use any defensive frames
    reward_features="5_features",       # 5 GIRL features
    method="girl"
)

features_df, sar_paths = run_pipeline(config)
# Output: 5 GIRL features from any defensive sequence
```

### Example 4: Advanced Features with All Defenders
```python
config = PipelineConfig(
    data_match="all_matches",
    back_four="all_players",            # Use all defenders
    sequence_type="negative_transition",
    reward_features="4_features",
    method="feature_computation"        # Advanced features
)

features_df, sar_paths = run_pipeline(config)
# Output: 20+ features computed from all defenders
```

---

## 🔍 Module Responsibilities

### `preprocessing/preprocessing.py`
- **Function**: `preprocess_all_matches(config)`
- **Routes to**:
  - `preprocess_match()` if `sequence_type="all_defensive"`
  - `preprocess_with_synchronization()` if `sequence_type="negative_transition"`

### `preprocessing/feature_engineering.py`
- **Function**: `compute_match_features(match_data, config)`
- **Routes to**:
  - `compute_transition_features()` if transitions exist
  - Legacy sequence-based approach otherwise
- **Function**: `compute_transition_features(match_data, config)`
- **Routes to**:
  - `reward_features.py` if `method="girl"`
  - `feature_computation.py` if `method="feature_computation"`

### `reward_features.py` (GIRL method)
- **Main function**: `compute_reward_features_from_config(state, reward_flag)`
- **Features**:
  - `space_score`: Zone-weighted attacker space
  - `stretch_index`: Convex hull area
  - `pressure_index`: Attackers under pressure
  - `line_height_relative`: Ball x - line x
  - `line_height_absolute`: Mean line x (if 5_features)

### `feature_computation.py` (Advanced method)
- **Main function**: `compute_features_for_all_sequences(sequences, tracking, home_team, back_four_flag)`
- **Features** (20+):
  - `stretch_index_*`: Compactness (mean, std, min, max)
  - `pressure_index_*`: Pressure metrics
  - `space_score_*`: Space control (4 zones)
  - `line_height_*`: Line positioning (3 variants)
  - `line_velocity_*`, `line_stability`, `defense_attack_correlation`

### `transition_analysis.py`
- **Function**: `extract_event_sequences_with_restart_handling()`
  - Filters sequences interrupted by restarts
  - Overrides labels based on dangerous events
- **Function**: `main_sequence_preparation()`
  - Complete orchestration: extraction → features → training data
- **Restart events**: Throw In, Goal Kick, Corner, Offside, Foul, etc.
- **Dangerous events**: Shot, Goal, Penalty Won

---

## ⚙️ Command Line Examples

### GIRL with 4 features
```bash
python main.py --data_match barcelona_madrid --sequence_type negative_transition --method girl --reward_features 4_features
```

### Advanced features with back four
```bash
python main.py --data_match all_matches --sequence_type negative_transition --method feature_computation --back_four back_four
```

### All defensive sequences with GIRL
```bash
python main.py --sequence_type all_defensive --method girl --reward_features 5_features
```

---

## 📊 Output DataFrame Columns

### GIRL Method Output
```
Columns:
- transition_idx: Transition identifier
- defending_team: Team defending
- attacking_team: Team attacking
- period: Match period
- label: Defense outcome (0=failure, 1=success)
- space_score: Zone-weighted attacker space
- stretch_index: Defensive line spread
- pressure_index: Attackers under pressure
- line_height_relative: Ball x - line x
- line_height_absolute: Mean line x (if 5_features)
- match_id: Match identifier
- home_team: Home team name
- away_team: Away team name
```

### Feature Computation Output
```
Columns:
- transition_idx: Transition identifier
- label: Defense outcome
- team_lost_possession: Defending team
- team_gained_possession: Attacking team
- period: Match period
- player_lost_possession: Player who lost ball
- stretch_index_mean, stretch_index_std, stretch_index_min, stretch_index_max
- pressure_index_mean, pressure_index_std, pressure_index_min, pressure_index_max
- space_score_mean, space_score_std, space_score_min, space_score_max
- line_height_absolute_mean, line_height_absolute_std
- line_height_relative_mean, line_height_relative_std
- line_height_norm_mean, line_height_norm_std
- line_velocity_mean, line_velocity_std
- line_stability
- defense_attack_correlation
- match_id, home_team, away_team
```

---

## 🐛 Troubleshooting

### Issue: No transitions extracted
**Cause**: `sequence_type="all_defensive"` doesn't create transitions  
**Solution**: Use `sequence_type="negative_transition"`

### Issue: Back-four flag not affecting GIRL features
**Cause**: GIRL uses all defenders regardless of back_four flag  
**Solution**: Use `method="feature_computation"` to control via back_four

### Issue: Missing reward_features columns
**Cause**: Using `method="feature_computation"` which ignores reward_features flag  
**Solution**: Use `method="girl"` to get reward feature set

### Issue: Event sequences too short
**Cause**: Restart events interrupting sequences  
**Solution**: Already handled by `extract_event_sequences_with_restart_handling()` - sequences are dropped if < 10 events

---

## ✅ Integration Validation

To test the integration:

```bash
# Test 1: GIRL method with transitions
python main.py --sequence_type negative_transition --method girl --reward_features 4_features

# Test 2: Advanced features with back four
python main.py --sequence_type negative_transition --method feature_computation --back_four back_four

# Test 3: All defensive with GIRL
python main.py --sequence_type all_defensive --method girl --reward_features 5_features
```

Check outputs for:
- ✅ Correct number of features
- ✅ Appropriate feature names
- ✅ No NaN values (unless expected)
- ✅ Sensible feature ranges

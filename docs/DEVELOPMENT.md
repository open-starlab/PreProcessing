# Development Guide

This guide is for developers who want to extend or modify the Defense Line Analysis pipeline.

## Architecture Overview

```
├── main.py                    # Root shim entry point
└── preprocessing/
    ├── config.py              # Configuration classes and constants
    ├── preprocessing.py       # Data loading and basic processing (without features)
    ├── feature_engineering.py # Feature computation logic
    ├── main.py                # Pipeline orchestration and CLI
    └── data_reorganization.py # Data format standardization
```

### Design Principles

1. **Modularity**: Each step (preprocessing, feature engineering) is independent
2. **Reproducibility**: All configurations and random seeds should be deterministic
3. **Configurability**: Use flags to control behavior instead of code modifications
4. **Testability**: Functions should be pure and testable

## Adding New Features

### Step 1: Update Config

Add feature name to `preprocessing/config.py`:

```python
# Add to FEATURE_SET_5
FEATURE_SET_5 = FEATURE_SET_4 + ['new_feature_name']
```

### Step 2: Implement Computer Function

Add computation function to `preprocessing/feature_engineering.py`:

```python
@staticmethod
def calculate_new_feature(
    defender_positions: np.ndarray,
    ball_position: Tuple[float, float]
) -> float:
    """
    Calculate new feature.
    
    Args:
        defender_positions: Array of shape (n, 2)
        ball_position: (x, y) tuple
    
    Returns:
        Feature value (0-1 normalized)
    """
    # Implementation
    return value
```

### Step 3: Add to Feature Extraction

Update `extract_features_from_sequence()` to call new function:

```python
# Aggregate features across frames
new_features_list = []

for _, frame in sequence_frames.iterrows():
    new_feature = DefensiveFeatureComputer.calculate_new_feature(
        defender_positions, ball_pos
    )
    new_features_list.append(new_feature)

# In result dict
result['new_feature_name'] = float(np.mean(new_features_list))
```

## Adding New Methods

### GIRL Method Implementation

To implement the GIRL (Goal-based Interpretable Reward Learning) method:

1. Create new file: `girl_method.py`

```python
def compute_girl_features(match_data, config):
    """GIRL feature computation."""
    # Implementation using goal outcomes or reinforcement learning
    pass
```

2. Update `preprocessing/feature_engineering.py` to dispatch based on method:

```python
if config.method == 'girl':
    from girl_method import compute_girl_features
    features = compute_girl_features(match_data, config)
else:
    features = extract_features_from_sequence(...)
```

## Adding New Flags

### To Add a 6th Flag

1. Create new Enum in `preprocessing/config.py`:

```python
class NewFlagEnum(str, Enum):
    """Options for new flag."""
    OPTION_A = "option_a"
    OPTION_B = "option_b"
```

2. Update `PipelineConfig`:

```python
@dataclass
class PipelineConfig:
    new_flag: Literal["option_a", "option_b"] = "option_a"
```

3. Add validation in `__post_init__`

4. Update CLI in `preprocessing/main.py`:

```python
parser.add_argument(
    '--new_flag',
    choices=['option_a', 'option_b'],
    default='option_a'
)
```

## Testing

### Unit Tests

Create `tests/test_feature_engineering.py`:

```python
import numpy as np
from preprocessing.feature_engineering import DefensiveFeatureComputer

def test_space_score():
    defenders = np.array([[10, 34], [15, 34], [20, 34]])
    ball = np.array([25, 34])
    
    score = DefensiveFeatureComputer.calculate_space_score(
        defenders, ball
    )
    
    assert 0 <= score <= 1
    assert isinstance(score, float)
```

### Integration Tests

```python
from preprocessing.config import PipelineConfig
from preprocessing.main import run_pipeline

def test_pipeline_barcelona_madrid():
    config = PipelineConfig(
        data_match='barcelona_madrid',
        data_dir='./test_data'
    )
    
    df, _ = run_pipeline(config)
    assert not df.empty
    assert 'space_score' in df.columns
```

## Performance Optimization

### Vectorization

Always use NumPy operations instead of Python loops:

```python
# Slow (Python loop)
distances = [np.linalg.norm(pos - ball) for pos in defenders]

# Fast (vectorized)
distances = np.linalg.norm(defenders - ball, axis=1)
```

### Caching

For expensive operations, use caching:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(key):
    # Implementation
    pass
```

## Code Style

- Follow PEP 8
- Use type hints for all functions
- Include docstrings with Args, Returns, Raises
- Keep functions under 30 lines when possible
- Use descriptive variable names

Example:

```python
def calculate_feature(
    positions: np.ndarray,
    ball: Tuple[float, float],
    radius: float = 3.0
) -> float:
    """
    Brief description of what this calculates.
    
    Args:
        positions: Shape (n, 2) array of (x, y) coordinates
        ball: (x, y) coordinate tuple
        radius: Search radius in meters
    
    Returns:
        Normalized feature value between 0 and 1
    
    Raises:
        ValueError: If positions array is empty
    """
    if len(positions) == 0:
        raise ValueError("Positions array cannot be empty")
    # Implementation
    return result
```

## Data Processing Pipeline

Understanding the data flow:

```
Raw Data
  ↓
MatchDataLoader.load_all()
  ├─ events.csv
  ├─ tracking.json
  └─ match.json
  ↓
TrackingProcessor.process_frames()
  ↓
Standardized tracking_df
  ├─ frame, period, timestamp
  ├─ possession_team
  ├─ home_x, home_y, away_x, away_y
  └─ ball_x, ball_y, ball_z
  ↓
PossessionAnalyzer.identify_defensive_sequences()
  ↓
List of (start_frame, end_frame) sequences
  ↓
extract_features_from_sequence()
  ↓
Feature DataFrame
```

## Debugging

### Print match information

```python
match_data = all_matches[0]
print(f"Match: {match_data['home_team']} vs {match_data['away_team']}")
print(f"Tracking frames: {len(match_data['tracking'])}")
print(f"Sequences: {len(match_data['sequences'])}")
```

### Inspect a single frame

```python
tracking_df = match_data['tracking']
frame = tracking_df.iloc[100]

print(f"Ball position: ({frame['ball_x']}, {frame['ball_y']})")
print(f"Home players: {[x for x in frame['home_x'] if x]}")
```

### Validate feature outputs

```python
features_df = compute_match_features(match_data, config)

# Check for NaN values
print(features_df.isnull().sum())

# Check ranges
print(features_df[['space_score', 'pressure_index']].describe())
```

## Contributing

1. Create a feature branch: `git checkout -b feature/new-feature`
2. Make changes with tests
3. Ensure all tests pass
4. Submit pull request with description

## Common Issues

### Issue: No matches found after filtering
**Cause**: `data_match='barcelona_madrid'` but no Barca vs Madrid matches in data
**Solution**: Use `data_match='all_matches'` first to verify data exists

### Issue: NaN values in features
**Cause**: Missing tracking data or invalid player positions
**Solution**: Check if ball_x, ball_y are valid before computing features

### Issue: Out of memory
**Cause**: Processing too many matches at once
**Solution**: Process matches in batches or optimize data structures

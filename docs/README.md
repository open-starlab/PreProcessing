# Defense Line Analysis - Reproducible Pipeline

A Python pipeline for analyzing defensive sequences in La Liga 2023/24 data, with configurable preprocessing and feature engineering flags.

## Directory Structure

```
cleaned/
├── main.py                      # Main pipeline entry point
├── preprocessing/
│   ├── data_reorganization.py   # Organize raw data into standard format
│   ├── config.py                # Configuration and constants
│   ├── preprocessing.py         # Data loading and preprocessing
│   └── feature_engineering.py   # Feature computation methods
├── requirements.txt             # Python dependencies
├── data/                        # Data directory (not included in repo)
│   ├── Laliga2023/24/           # Organized match data
│   │   ├── {match_id}/
│   │   │   ├── events.csv
│   │   │   ├── match.json
│   │   │   └── tracking.json
│   │   └── metadata/
│   │       ├── matches.json
│   │       ├── players.json
│   │       └── teams.json
│   └── output/                  # Generated outputs
└── notebooks/                   # Experimental notebooks

## Configuration Flags

The pipeline supports 5 configuration flags for flexible experimentation:

### Flag 1: Data Match Selection
- `barcelona_madrid`: Matches containing Barcelona OR Real Madrid (any team they play against)
- `all_matches`: All available matches from La Liga 2023/24

### Flag 2: Defender Selection (excluding goalkeeper)
- `back_four`: Only center-backs and full-backs (back four structure)
- `all_players`: All defenders except goalkeeper (including defensive midfielders)

### Flag 3: Defensive Sequence Type
- `negative_transition`: 10-frame sequences starting AFTER possession loss
- `all_defensive`: 10-frame defensive sequences regardless of possession loss

### Flag 4: Reward Features
- `4_features`: space_score, pressure_index, stretch_index, line_height_relative
- `5_features`: Above 4 + line_height_absolute

### Flag 5: Feature Computation Method
- `girl`: Goal-based Interpretable Reward Learning approach
- `feature_computation`: Direct feature engineering approach

## Quick Start

### 1. Organize Data

```python
python preprocessing/data_reorganization.py
```

This reorganizes your raw data from scattered directories into:
```
data/Laliga2023/24/{match_id}/
├── events.csv
├── match.json
└── tracking.json
```

### 2. Run Pipeline

```python
python main.py \
    --data_match all_matches \
    --back_four all_players \
    --sequence_type all_defensive \
    --reward_features 5_features \
    --method feature_computation
```

Or use configuration objects in Python:

```python
from preprocessing.config import PipelineConfig
from preprocessing.main import run_pipeline

config = PipelineConfig(
    data_match='barcelona_madrid',
    back_four='back_four',
    sequence_type='negative_transition',
    reward_features='4_features',
    method='girl'
)

df_features, sar_paths = run_pipeline(config)
```

## Feature Output

The pipeline produces a DataFrame with:
- **Basic info**: match_id, team, period, frame numbers
- **Sequence info**: sequence_type, sequence_duration
- **Defensive features**:
  - space_score: Compression efficiency
  - pressure_index: Pressure application
  - stretch_index: Line stretch
  - line_height_relative: Relative defensive line height
  - line_height_absolute: Absolute defensive line height (if 5_features)

## Requirements

- Python 3.8+
- pandas
- numpy
- scipy
- scikit-learn
- tqdm

## Data Format

Raw data should be organized as:
```
Laliga2023/24/
├── {match_id}/
│   ├── events.csv          # Event data (StatsBomb format)
│   ├── match.json          # Match metadata
│   └── tracking.json       # Skillcorner tracking data
└── metadata/
    ├── matches.json        # Match information
    ├── players.json        # Player information
    └── teams.json          # Team information
```

## License

[Add your license here]

## Citation

[Add citation information if applicable]

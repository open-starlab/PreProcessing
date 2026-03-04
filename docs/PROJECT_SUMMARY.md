# Reproducible Defense Line Analysis Pipeline
## Project Setup Summary

This document summarizes what has been created for reproducible, GitHub-publication-ready code.

## Directory Structure

```
cleaned/
├── Core Modules
│   ├── preprocessing/config.py                     # Configuration flags and constants
│   ├── preprocessing/preprocessing.py              # Data loading (no feature computation)
│   ├── preprocessing/feature_engineering.py        # Feature computation methods
│   ├── preprocessing/main.py                       # Pipeline orchestration & CLI
│   └── preprocessing/data_reorganization.py        # Data format standardization
│
├── Documentation
│   ├── README.md                     # Quick start and overview
│   ├── DEVELOPMENT.md                # Extended development guide
│   └── example_notebook.ipynb        # Runnable examples with all flags
│
├── Configuration
│   ├── requirements.txt               # Python dependencies
│   ├── setup.py                      # Package setup for pip install
│   ├── setup.sh                      # Automated setup script
│   ├── pytest.ini                    # Testing configuration
│   ├── .gitignore                    # Git ignore patterns
│   └── LICENSE                       # MIT License
│
└── Tests
    └── tests/
        ├── test_config.py            # Config validation tests
        ├── test_feature_engineering.py # Feature computation tests
        └── __init__.py
```

## The 5 Configuration Flags

### Flag 1: Data Match
- **barcelona_madrid**: Matches containing Barcelona OR Real Madrid (any opponent)
- **all_matches**: All La Liga 2023/24 matches

### Flag 2: Back Four Detection (Excluding Goalkeeper)  
- **back_four**: Only center-backs and full-backs
- **all_players**: All defenders except goalkeeper

### Flag 3: Defensive Sequences
- **negative_transition**: 10-frame sequences AFTER possession loss
- **all_defensive**: 10-frame sequences any time

### Flag 4: Reward Features
- **4_features**: space_score, pressure_index, stretch_index, line_height_relative
- **5_features**: Above 4 + line_height_absolute

### Flag 5: Feature Computation Method
- **feature_computation**: Direct feature engineering approach
- **girl**: Goal-based Interpretable Reward Learning (framework prepared)

**Total combinations: 2 × 2 × 2 × 2 × 2 = 32 possible configurations**

## Key Features

### ✅ Preprocessing (Without Feature Computation)
- MatchDataLoader: Load events, tracking, and match metadata
- TrackingProcessor: Convert raw tracking to standardized format
- PossessionAnalyzer: Identify defensive sequences and transitions
- **NOTE**: Reward features and back-four detection are NOT in preprocessing

### ✅ Feature Engineering (Modular)
- DefensiveFeatureComputer: Compute all defensive metrics
  - space_score: Compression around ball
  - pressure_index: Defenders in pressure radius
  - stretch_index: Line left-right coverage
  - line_height_relative: Distance from own goal (normalized)
  - line_height_absolute: Actual x-coordinate

### ✅ Configurable Pipeline
- PipelineConfig dataclass: Type-safe configuration
- Five independent flags for flexible experimentation
- Automatic validation of flag values
- CLI interface via argparse
- Python API for programmatic usage

### ✅ Reproducibility
- Fixed random seeds where applicable
- Deterministic feature computation
- Configuration-based parameterization
- No hard-coded paths or magic numbers

### ✅ Documentation
- README with quick start guide
- DEVELOPMENT guide for extending code
- Example notebook with 6+ usage patterns
- Docstrings for all functions
- Type hints throughout

### ✅ Testing
- Unit tests for feature computations
- Configuration validation tests
- pytest configured and ready
- Can be extended with integration tests

## Quick Start

### 1. Install Dependencies
```bash
cd cleaned
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or use the setup script:
```bash
bash setup.sh
```

### 2. Organize Data
```bash
python preprocessing/data_reorganization.py \
    --raw_dir /path/to/raw/Laliga2023/24 \
    --output_dir ./data/Laliga2023/24
```

### 3. Run Pipeline

**Default configuration:**
```bash
python main.py
```

**Specific configuration:**
```bash
python main.py \
    --data_match barcelona_madrid \
    --back_four back_four \
    --sequence_type negative_transition \
    --reward_features 4_features \
    --method feature_computation \
    --save
```

**Python API:**
```python
from preprocessing.config import PipelineConfig
from preprocessing.main import run_pipeline

config = PipelineConfig(
    data_match='all_matches',
    back_four='all_players',
    sequence_type='all_defensive',
    reward_features='5_features',
    method='feature_computation'
)

features_df, sar_paths = run_pipeline(config)
```

## For GitHub Publication

### Ready to Go ✅
- ✅ Clear README with installation instructions
- ✅ Reproducible pipeline with no magic numbers
- ✅ Configurable flags for experimentation
- ✅ Type hints and docstrings
- ✅ MIT License included
- ✅ .gitignore for clean repository
- ✅ Example notebook with six usage patterns
- ✅ Test suite foundation
- ✅ Development guide for contributors

### Next Steps for Publication
1. Update author info in:
   - setup.py (author, author_email)
   - LICENSE (copyright holder)
   
2. Add data sources:
   - Document where to obtain La Liga 2023/24 data
   - Add data download instructions to README
   
3. Add citation (optional):
   - If based on published work, add citation to README
   
4. Create GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Defense Line Analysis pipeline"
   git remote add origin https://github.com/username/defense-line-analysis
   git push -u origin main
   ```

## Code Quality Standards

- ✅ PEP 8 compliant
- ✅ Type hints on all functions
- ✅ Comprehensive docstrings
- ✅ Vectorized NumPy operations
- ✅ No hard-coded paths
- ✅ Modular functions < 30 lines
- ✅ Descriptive variable names
- ✅ Error handling in data loading

## Extensibility

### Adding Features
1. Add to config: FEATURE_SET_5
2. Implement computation in DefensiveFeatureComputer
3. Integrate into extract_features_from_sequence()

### Adding Methods
1. Create new module (e.g., girl_method.py)
2. Update preprocessing/feature_engineering.py to dispatch
3. Test with all flag combinations

### Adding Flags
1. Create new Enum in preprocessing/config.py
2. Update PipelineConfig dataclass
3. Add CLI argument in preprocessing/main.py
4. Update comparison logic

## File Statistics

```
Core Modules:        ~2500 lines
Documentation:       ~500 lines
Tests:              ~400 lines
Config/Setup:        ~200 lines
────────────────────────────
Total:              ~3600 lines
```

## Modularity Score

Each component is independent:
- ✅ preprocessing/config.py: No dependencies on other modules
- ✅ preprocessing/preprocessing.py: Only depends on config
- ✅ preprocessing/feature_engineering.py: Only depends on config
- ✅ preprocessing/main.py: Orchestrates the above
- ✅ preprocessing/data_reorganization.py: Standalone utility

**High cohesion, low coupling** ✓

---

**Status**: ✅ Ready for GitHub publication
**Last Updated**: 2025-02-22

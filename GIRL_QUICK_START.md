# Quick Start Guide - GIRL Pipeline

## Overview

The GIRL (Goal-based Inverse Reinforcement Learning) pipeline recovers reward weights from expert defensive demonstrations in football tracking data.

## Prerequisites

```bash
# Install dependencies
pip install numpy torch pandas scikit-learn scipy matplotlib seaborn quadprog
```

## Full Pipeline (Recommended)

```bash
# Step 1: Generate SAR dataset (if not already done)
cd /home/s_dash/workspace6/cleaned
python preprocessing/main.py --method girl --save

# Step 2: Run GIRL pipeline
python girl_pipeline/run_girl_pipeline.py
```

**Output**: Results saved to `girl_pipeline/output/`

## Quick Test (Debug Mode)

```bash
# Use only 20 sequences for quick testing
python girl_pipeline/run_girl_pipeline.py --debug --num_epochs 10
```

**Runtime**: ~30 seconds
**Output**: Same as full pipeline, but with subset

## Advanced Options

### Custom Parameters

```bash
python girl_pipeline/run_girl_pipeline.py \
    --n_folds 10 \              # Increase cross-validation folds
    --hidden_size 128 \         # Larger LSTM model
    --num_epochs 100 \          # More training epochs
    --batch_size 16 \           # Smaller batches
    --device cuda \             # Use GPU
    --solver_method quadprog    # Choose QP solver
```

### Train/Test Split (Instead of CV)

```bash
python girl_pipeline/run_girl_pipeline.py \
    --n_folds 0 \               # Disable cross-validation
    --test_size 0.2             # 80% train, 20% test
```

### Specific SAR Dataset

```bash
python girl_pipeline/run_girl_pipeline.py \
    --config_suffix "all_matches_all_players_all_defensive_4_features"
```

### Custom Output Directory

```bash
python girl_pipeline/run_girl_pipeline.py \
    --output_dir "./my_results/"
```

### Suppress Output (Quiet Mode)

```bash
python girl_pipeline/run_girl_pipeline.py --quiet
```

## Command Reference

```
Usage: python girl_pipeline/run_girl_pipeline.py [OPTIONS]

Dataset options:
  --sar_dir PATH              SAR dataset directory (default: ./preprocessing/output/sar)
  --config_suffix STR         Config suffix for specific SAR dataset

Cross-validation options:
  --n_folds INT               Number of CV folds (default: 5, use 0 for train/test)
  --test_size FLOAT           Test size for split (default: 0.2)

Model options:
  --hidden_size INT           LSTM hidden size (default: 64)
  --num_epochs INT            Training epochs (default: 50)
  --batch_size INT            Batch size (default: 32)
  --device STR                Device: auto, cpu, cuda (default: auto)

GIRL options:
  --solver_method STR         QP solver: quadprog, cvxopt, analytical (default: quadprog)

Misc options:
  --random_state INT          Random seed (default: 42)
  --output_dir PATH           Results directory (default: ./girl_pipeline/output)
  --debug                     Use only 20 sequences for testing
  --quiet                     Suppress verbose output
```

## Output Files

**Location**: `girl_pipeline/output/`

1. **reward_weights_TIMESTAMP.csv**
   - Recovered weights with standard deviations
   - One feature per row

2. **cross_validation_results_TIMESTAMP.csv**
   - Per-fold weight estimates
   - Stack to analyze cv variability

3. **experiment_metadata.json**
   - Complete pipeline metadata
   - Dataset characteristics
   - Hyperparameters
   - Recovered weights

4. **reward_weights.png**
   - Bar chart visualization
   - Error bars from CV
   - Publication-ready quality

## Expected Output

### Console Output Example

```
Recovered Reward Weights:
  stretch_index       : 0.2843 ± 0.0142
  pressure_index      : 0.3218 ± 0.0178
  space_score         : 0.2456 ± 0.0156
  line_height_rel     : 0.1483 ± 0.0089

Results saved to:
  girl_pipeline/output/reward_weights_20260305_143022.csv
  girl_pipeline/output/cross_validation_results_20260305_143022.csv
  girl_pipeline/output/experiment_metadata.json
  girl_pipeline/output/reward_weights.png
```

### CSV Output Example

`reward_weights_*.csv`:
```csv
feature,weight,std
stretch_index,0.284315,0.014234
pressure_index,0.321756,0.017812
space_score,0.245632,0.015621
line_height_rel,0.148297,0.008945
```

## Troubleshooting

### Issue: SAR dataset not found

**Solution**: Generate SAR dataset first
```bash
python preprocessing/main.py --method girl --save
```

### Issue: Out of memory

**Solution**: Reduce model size
```bash
python girl_pipeline/run_girl_pipeline.py --batch_size 16 --hidden_size 32
```

### Issue: Low BC accuracy (<60%)

**Solution**: Train longer with larger model
```bash
python girl_pipeline/run_girl_pipeline.py --num_epochs 100 --hidden_size 128
```

### Issue: QP solver fails

**Solution**: Try different solver
```bash
python girl_pipeline/run_girl_pipeline.py --solver_method analytical
```

### Issue: Slow on CPU

**Solution**: Use GPU if available
```bash
python girl_pipeline/run_girl_pipeline.py --device cuda
```

## Development/Testing

### Run tests on placeholder data

```bash
# Test data loader
python girl_pipeline/data_loader.py

# Test plotting utilities
python girl_pipeline/utils/plot_weights.py
```

### Interactive Python usage

```python
from girl_pipeline import (
    load_sar_dataset,
    cross_validate_girl,
)

# Load data
states, actions, rewards, metadata = load_sar_dataset()

# Run pipeline
results = cross_validate_girl(
    states=states[:20],  # Use subset for testing
    actions=actions[:20],
    rewards=rewards[:20],
    n_splits=3,
    num_epochs=10,
    device='cpu'
)

# Access results
print(results['mean_weights'])
```

## Performance Benchmarks

**Full Pipeline with Default Settings**:
- Dataset: ~500 sequences
- Device: CPU
- Runtime: ~5-10 minutes
- Memory: ~2 GB

**Debug Mode**:
- Dataset: 20 sequences
- Runtime: ~30 seconds
- Memory: ~500 MB

**With GPU (CUDA)**:
- Runtime: ~1-2 minutes
- Memory: ~1.5 GB VRAM

## File Structure

```
girl_pipeline/
├── run_girl_pipeline.py      ← Main entry point (RUN THIS)
├── data_loader.py            ← Load SAR datasets
├── bc/
│   ├── model.py              ← DefenseBC LSTM model
│   └── train_bc.py           ← BC training
├── gradients/
│   └── compute_gradients.py  ← Policy gradient computation
├── irl/
│   └── girl_solver.py        ← QP solver for IRL
├── utils/
│   ├── cross_validation.py   ← K-fold CV utilities
│   └── plot_weights.py       ← Visualization
└── output/                   ← Results directory
    ├── reward_weights_*.csv
    ├── cross_validation_results_*.csv
    ├── experiment_metadata.json
    └── reward_weights.png
```

## References

- Original GIRL paper: Ratliff et al. (2006) "Maximum Margin Planning"
- IRL overview: Abbeel & Ng (2004) "Apprenticeship Learning via IRL"

## Support

For issues or questions, see `GIRL_INTEGRATION.md` or `GIRL_VERIFICATION_REPORT.md`

---

**Happy learning! 🎯**

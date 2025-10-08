# Ultimate Frisbee Tracking Data Preprocessing

This module provides preprocessing functionality for Ultimate Frisbee tracking data from multiple data providers, converting them into a standardized Metrica format for analysis.

## Overview

The Ultimate tracking data preprocessing system supports two main data providers:
- **UFA (Ultimate Frisbee Analytics)**: Professional Ultimate game data
- **Ultimate Track**: Research-grade tracking data with detailed motion features

## Data Providers

### UFA Data Provider
- **Input Format**: CSV/TXT files with player and disc positions
- **Features**: Player positions, velocities, disc tracking, holder identification
- **Output**: Metrica format with home/away team separation

### Ultimate Track Data Provider  
- **Input Format**: CSV files with raw tracking data
- **Features**: Enhanced motion analysis with velocity/acceleration calculations
- **Output**: Metrica format with calculated motion features

## Architecture

```
ultimate/
├── ultimate_tracking_class.py     # Main interface class
├── ufa_preprocessing/             # UFA data processing
│   ├── preprocessing.py           # UFA-specific preprocessing functions
│   ├── preprocess_config.py       # UFA configuration constants
│   └── README.md                  # UFA module documentation
└── ultimatetrack_preprocessing/   # Ultimate Track data processing
    ├── preprocessing.py           # Ultimate Track preprocessing functions  
    ├── preprocess_config.py       # Ultimate Track configuration constants
    └── __init__.py
```

## Usage

```python
from preprocessing.sports.tracking_data.ultimate import Ultimate_tracking_data

# For UFA data
ufa_tracker = Ultimate_tracking_data("UFA", "/path/to/ufa_data.csv")
home_df, away_df, events_df = ufa_tracker.preprocessing()

# For Ultimate Track data
ut_tracker = Ultimate_tracking_data("UltimateTrack", "/path/to/ut_data.csv")
home_df, away_df, events_df = ut_tracker.preprocessing()
```

## Output Format

All data providers output data in Metrica format with three DataFrames:

### Home/Away DataFrames
- **MultiIndex columns**: Team, Player ID, Coordinate (X/Y)
- **Data**: Player positions over time with disc tracking
- **Frequency**: Configurable based on data provider (10Hz for UFA, 15Hz for Ultimate Track)

### Events DataFrame
- **Columns**: Team, Type, Subtype, Period, Start Frame, Start Time, End Frame, End Time, From, To, Start X, Start Y, End X, End Y
- **Data**: Disc possession events and position data per frame

## Configuration

Each data provider has its own configuration file defining:
- Field dimensions and specifications
- Players per team (7 for Ultimate Frisbee)
- Tracking frequency and coordinate scaling
- Column mappings and processing parameters

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- Standard Python libraries (os, argparse for CLI usage)
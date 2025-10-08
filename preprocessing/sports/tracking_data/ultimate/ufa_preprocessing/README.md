# UFA (Ultimate Frisbee Analytics) Data Preprocessing

This module handles preprocessing of UFA tracking data, converting it from the native UFA format to standardized Metrica format for analysis.

## Overview

UFA provides professional Ultimate Frisbee game tracking data with player positions, velocities, and disc tracking information. This preprocessing module standardizes the data format while preserving all essential game information.

## Data Format

### Input (UFA Format)
The UFA data contains the following columns:
- `frame`: Frame number (10 Hz frequency)
- `id`: Player/disc unique identifier  
- `x`, `y`: Position coordinates (in meters)
- `vx`, `vy`: Velocity components
- `ax`, `ay`: Acceleration components
- `class`: Entity type ("offense", "defense", "disc")
- `holder`: Boolean indicating disc possession
- `closest`: ID of closest defender to each offensive player
- Additional motion features (magnitudes, angles, etc.)

### Output (Metrica Format)
- **Home DataFrame**: Offensive team tracking data with MultiIndex columns
- **Away DataFrame**: Defensive team tracking data with MultiIndex columns  
- **Events DataFrame**: Frame-by-frame disc position and possession data

## Processing Pipeline

1. **Data Loading**: Read UFA CSV/TXT files
2. **Column Filtering**: Remove unnecessary columns (selected, prev_holder, def_selected)
3. **Coordinate Scaling**: Apply scaling factor (default: 1.0 for meter-based data)
4. **Format Conversion**: Transform to Metrica format with proper team assignment

## Key Functions

### `preprocessing_for_ufa(data_path)`
Main preprocessing function that orchestrates the entire conversion process.

### `convert_to_metrica_format(intermediate_df)`
Converts UFA intermediate data to Metrica format, creating separate DataFrames for home team, away team, and events.

### `create_events_metrica(df)`
Creates events DataFrame with disc position and holder information for each frame.

### `create_tracking_metrica(df, team)`
Creates tracking DataFrames for home (offense) and away (defense) teams with MultiIndex column structure.

## Configuration

Key configuration parameters in `preprocess_config.py`:
- **Field Dimensions**: 109.73m × 48.77m (UFA standard)
- **Players Per Team**: 7 (Ultimate Frisbee standard)
- **Tracking Frequency**: 10 Hz
- **Coordinate Scale**: 1.0 (data already in meters)

## Usage Example

```python
from preprocessing.sports.tracking_data.ultimate.ufa_preprocessing import preprocessing_for_ufa

# Process UFA data file
data_path = "/path/to/ufa_game_data.csv"
home_df, away_df, events_df = preprocessing_for_ufa(data_path)

# Access tracking data
print(f"Home team shape: {home_df.shape}")
print(f"Away team shape: {away_df.shape}") 
print(f"Events shape: {events_df.shape}")
```

## Data Quality Notes

- UFA data provides high-quality professional game tracking
- Disc possession is explicitly marked with `holder` flag
- Player-defender relationships are pre-calculated in `closest` field
- Coordinate system uses standard Ultimate field dimensions
- Missing data is handled gracefully with NaN values
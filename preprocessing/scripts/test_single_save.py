#!/usr/bin/env python
"""
Quick test: process a single match and save features.
"""

from pathlib import Path
from preprocessing import preprocess_match
from feature_engineering import compute_match_features
from config import PipelineConfig
import pandas as pd

def main():
    print("Testing data processing on a single match...\n")
    
    # Process one match
    match_dir = Path('/home/s_dash/workspace6/Defense_line/Laliga2023/24/1018887')
    config = PipelineConfig(
        back_four='all_players',
        sequence_type='all_defensive',
        reward_features='5_features'
    )
    
    print(f"Processing: {match_dir.name}")
    match_data = preprocess_match(match_dir, config)
    
    if match_data:
        print(f"✓ Match loaded: {match_data['home_team']} vs {match_data['away_team']}")
        print(f"  Defensive sequences: {len(match_data['sequences'])}")
        
        # Compute features
        features_df = compute_match_features(match_data, config)
        print(f"✓ Features computed: {len(features_df)} rows")
        
        # Save to CSV
        output_dir = Path('/home/s_dash/workspace6/processed_features')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"test_match_{match_dir.name}.csv"
        features_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved to: {output_file}")
        print(f"\nFirst 10 rows:")
        print(features_df.head(10))
        
        print(f"\nColumns: {list(features_df.columns)}")
    else:
        print("✗ Match processing failed")

if __name__ == '__main__':
    main()

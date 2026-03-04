#!/usr/bin/env python
"""
Quick script to process La Liga data and save features.
"""

from pathlib import Path
from preprocessing.config import PipelineConfig
from preprocessing.main import run_pipeline, save_features
import sys

def main():
    print("Processing La Liga 2023/24 data...\n")
    
    # Configuration
    config = PipelineConfig(
        data_match='all_matches',
        back_four='all_players',
        sequence_type='all_defensive',
        reward_features='5_features',
        method='feature_computation',
        data_dir='/home/s_dash/workspace6/Defense_line/Laliga2023/24'
    )
    
    # Run pipeline
    print(f"Configuration:\n{config}\n")
    features_df = run_pipeline(config)
    
    # Save results
    if not features_df.empty:
        output_dir = Path('/home/s_dash/workspace6/processed_features')
        save_features(features_df, output_dir, config)
        
        # Show sample output
        print(f"\n{'='*70}")
        print("Sample of processed data:")
        print(f"{'='*70}")
        print(features_df.head(10))
        
        print(f"\n{'='*70}")
        print("Output saved successfully!")
        print(f"{'='*70}")
    else:
        print("❌ No features were computed!")
        sys.exit(1)

if __name__ == '__main__':
    main()

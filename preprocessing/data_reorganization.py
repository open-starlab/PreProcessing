#!/usr/bin/env python3
"""
Reorganize Laliga 2023/24 data into the standard structure.

This script reorganizes raw data from scattered directories into:

New structure:
data/Laliga2023/24/{ID}/
├── events.csv
├── match.json
└── tracking.json

Metadata in:
data/Laliga2023/24/metadata/
├── matches.json
├── players.json
└── teams.json

Usage:
    python data_reorganization.py \
        --raw_dir /path/to/raw/data \
        --output_dir ./data/Laliga2023/24
"""

import os
import shutil
import argparse
from pathlib import Path


def reorganize_data(raw_dir, output_dir):
    """
    Reorganize La Liga data into standard structure.
    
    Args:
        raw_dir: Directory containing Event/, Tracking/ subdirectories
        output_dir: Target output directory (Laliga2023/24)
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    
    EVENT_DIR = raw_dir / "Event"
    TRACKING_DIR = raw_dir / "Tracking" / "skillcorner"
    METADATA_DIR = output_dir / "metadata"
    
    # Create metadata directory
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created metadata directory: {METADATA_DIR}")
    
    # Copy metadata files
    metadata_files = ['matches.json', 'players.json', 'teams.json']
    for meta_file in metadata_files:
        src = TRACKING_DIR / "metadata" / meta_file
        dst = METADATA_DIR / meta_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✓ Copied {meta_file}")
        else:
            print(f"⚠ Warning: {meta_file} not found at {src}")
    
    # Get all match IDs
    if not EVENT_DIR.exists():
        print(f"✗ Error: Event directory not found at {EVENT_DIR}")
        return
    
    match_ids = sorted([d.name for d in EVENT_DIR.iterdir() if d.is_dir()])
    print(f"\nProcessing {len(match_ids)} matches...\n")
    
    success_count = 0
    partial_count = 0
    missing_tracking = []
    missing_match = []
    
    for idx, match_id in enumerate(match_ids):
        try:
            # Create new match directory
            new_match_dir = output_dir / match_id
            new_match_dir.mkdir(parents=True, exist_ok=True)
            
            # Source paths
            events_src = EVENT_DIR / match_id / "events.csv"
            match_json_src = TRACKING_DIR / "match" / f"{match_id}.json"
            tracking_json_src = TRACKING_DIR / "tracking" / f"{match_id}.json"
            
            # Destination paths
            events_dst = new_match_dir / "events.csv"
            match_json_dst = new_match_dir / "match.json"
            tracking_json_dst = new_match_dir / "tracking.json"
            
            # Copy files
            files_copied = 0
            all_files_present = True
            
            if events_src.exists():
                if not events_dst.exists():
                    shutil.copy2(events_src, events_dst)
                files_copied += 1
            else:
                all_files_present = False
            
            if match_json_src.exists():
                if not match_json_dst.exists():
                    shutil.copy2(match_json_src, match_json_dst)
                files_copied += 1
            else:
                all_files_present = False
                missing_match.append(match_id)
            
            if tracking_json_src.exists():
                if not tracking_json_dst.exists():
                    shutil.copy2(tracking_json_src, tracking_json_dst)
                files_copied += 1
            else:
                all_files_present = False
                missing_tracking.append(match_id)
            
            if all_files_present:
                success_count += 1
            elif files_copied > 0:
                partial_count += 1
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed: {idx + 1}/{len(match_ids)}")
            
        except Exception as e:
            print(f"✗ Error processing {match_id}: {str(e)}")
    
    print(f"\n{'='*70}")
    print(f"Reorganization complete!")
    print(f"{'='*70}")
    print(f"✓ Complete matches (all 3 files): {success_count}")
    if partial_count > 0:
        print(f"⚠ Partial matches (some files): {partial_count}")
    print(f"{'='*70}")
    
    if missing_tracking:
        print(f"\n⚠ Matches missing tracking.json ({len(missing_tracking)})")
    
    if missing_match:
        print(f"⚠ Matches missing match.json ({len(missing_match)})")


def main():
    parser = argparse.ArgumentParser(
        description='Reorganize La Liga 2023/24 data into standard format'
    )
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='/home/s_dash/workspace6/Defense_line/Laliga2023/24',
        help='Raw data directory containing Event/ and Tracking/ subdirectories'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/Laliga2023/24',
        help='Output directory for reorganized data'
    )
    
    args = parser.parse_args()
    reorganize_data(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()

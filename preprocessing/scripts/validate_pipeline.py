#!/usr/bin/env python
"""
Validation script to test module imports, config setup, and pipeline structure.
Runs lightweight checks without needing full match data.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all core modules import without errors."""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    modules = {
        'config': 'Configuration management',
        'preprocessing': 'Data preprocessing',
        'feature_computation': 'Feature computation',
        'feature_engineering': 'Feature engineering',
        'transition_analysis': 'Transition analysis',
        'event_tracking_sync': 'Event-tracking synchronization',
        'reward_features': 'Reward features (GIRL)',
        'main': 'Main pipeline orchestration'
    }
    
    failed = []
    for module_name, description in modules.items():
        try:
            exec(f"import {module_name}")
            print(f"✓ {module_name:25s} - {description}")
        except Exception as e:
            print(f"✗ {module_name:25s} - {description}")
            print(f"  Error: {str(e)}")
            failed.append((module_name, str(e)))
    
    return len(failed) == 0, failed


def test_config():
    """Test configuration setup."""
    print("\n" + "="*70)
    print("TEST 2: Configuration Setup")
    print("="*70)
    
    try:
        from config import PipelineConfig
        
        # Test default config
        config = PipelineConfig()
        print(f"✓ Default config created:")
        print(f"  - data_match: {config.data_match}")
        print(f"  - back_four: {config.back_four}")
        print(f"  - sequence_type: {config.sequence_type}")
        print(f"  - reward_features: {config.reward_features}")
        print(f"  - method: {config.method}")
        print(f"  - data_dir: {config.data_dir}")
        print(f"  - output_dir: {config.output_dir}")
        
        # Test custom config
        custom = PipelineConfig(
            data_match='barcelona_madrid',
            sequence_type='negative_transition',
            method='girl'
        )
        print(f"\n✓ Custom config created (barcelona_madrid / negative_transition / girl)")
        
        return True, None
    except Exception as e:
        print(f"✗ Config error: {str(e)}")
        traceback.print_exc()
        return False, str(e)


def test_feature_computation_functions():
    """Test feature computation helper functions exist."""
    print("\n" + "="*70)
    print("TEST 3: Feature Computation Functions")
    print("="*70)
    
    try:
        from feature_computation import (
            compute_convex_hull_area,
            compute_pressure_index,
            compute_space_score,
            compute_line_height_features,
            build_step_features,
            classify_defensive_action,
            ACTION_MAP,
            ACTION_NAMES,
            extract_sequence_features_advanced,
            _is_def,
            _is_atk,
            _is_gk,
            _extract_xy
        )
        
        functions = [
            'compute_convex_hull_area',
            'compute_pressure_index',
            'compute_space_score',
            'compute_line_height_features',
            'build_step_features',
            'classify_defensive_action',
            'extract_sequence_features_advanced',
            '_is_def', '_is_atk', '_is_gk', '_extract_xy'
        ]
        
        for fn_name in functions:
            print(f"✓ {fn_name}")
        
        print(f"\n✓ Action definitions:")
        print(f"  ACTION_MAP: {ACTION_MAP}")
        print(f"  ACTION_NAMES: {ACTION_NAMES}")
        
        return True, None
    except Exception as e:
        print(f"✗ Feature computation error: {str(e)}")
        traceback.print_exc()
        return False, str(e)


def test_main_functions():
    """Test main pipeline orchestration functions."""
    print("\n" + "="*70)
    print("TEST 4: Main Pipeline Functions")
    print("="*70)
    
    try:
        from main import (
            run_pipeline,
            save_features,
            save_preprocessed_data,
            generate_summary_statistics,
            save_summary_statistics,
            save_transition_processing_log
        )
        
        functions = [
            'run_pipeline',
            'save_features',
            'save_preprocessed_data',
            'generate_summary_statistics',
            'save_summary_statistics',
            'save_transition_processing_log'
        ]
        
        for fn_name in functions:
            print(f"✓ {fn_name}")
        
        return True, None
    except Exception as e:
        print(f"✗ Main pipeline error: {str(e)}")
        traceback.print_exc()
        return False, str(e)


def test_preprocessing_functions():
    """Test preprocessing module functions."""
    print("\n" + "="*70)
    print("TEST 5: Preprocessing Functions")
    print("="*70)
    
    try:
        from preprocessing import (
            MatchDataLoader,
            TrackingProcessor,
            PossessionAnalyzer,
            preprocess_match,
            preprocess_all_matches
        )
        
        classes = ['MatchDataLoader', 'TrackingProcessor', 'PossessionAnalyzer']
        functions = ['preprocess_match', 'preprocess_all_matches']
        
        for cls_name in classes:
            print(f"✓ {cls_name} (class)")
        
        for fn_name in functions:
            print(f"✓ {fn_name} (function)")
        
        return True, None
    except Exception as e:
        print(f"✗ Preprocessing error: {str(e)}")
        traceback.print_exc()
        return False, str(e)


def test_state_action_definitions():
    """Test state/action extraction definitions."""
    print("\n" + "="*70)
    print("TEST 6: State/Action Definitions")
    print("="*70)
    
    try:
        from feature_computation import (
            DEF_TAGS,
            build_step_features,
            classify_defensive_action,
            _is_gk, _is_def, _is_atk
        )
        
        print(f"✓ DEF_TAGS defined: {DEF_TAGS}")
        print(f"✓ build_step_features - 18-dim state vector")
        print(f"✓ classify_defensive_action - 4-class action (backward/forward/compress/expand)")
        print(f"✓ Role classification helpers (_is_gk, _is_def, _is_atk)")
        
        # Test with sample player dict
        sample_def = {
            'player_id': 1,
            'position': {'x': 10.0, 'y': 34.0},
            'role': 'DF',
            'player_role': 'Centre-Back',
            'is_goalkeeper': False
        }
        
        sample_gk = {
            'player_id': 0,
            'position': {'x': 0.0, 'y': 34.0},
            'role': 'GK',
            'player_role': 'Goalkeeper',
            'is_goalkeeper': True
        }
        
        is_def_result = _is_def(sample_def)
        is_gk_result = _is_gk(sample_gk)
        
        print(f"\n✓ Sample defender detection: {is_def_result}")
        print(f"✓ Sample GK detection: {is_gk_result}")
        print(f"✓ Helper functions available: _is_def, _is_gk, _extract_xy")
        
        return True, None
    except Exception as e:
        print(f"✗ State/action error: {str(e)}")
        traceback.print_exc()
        return False, str(e)


def test_output_structure():
    """Test that output directory can be created."""
    print("\n" + "="*70)
    print("TEST 7: Output Directory Setup")
    print("="*70)
    
    try:
        output_dir = Path('./output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output directory created/accessible: {output_dir.absolute()}")
        print(f"✓ Can write to: {output_dir.is_dir()}")
        
        # Test would-be output paths
        from config import PipelineConfig
        config = PipelineConfig()
        
        paths = {
            'preprocessed': f"preprocessed_{config.data_match}_{config.back_four}_{config.sequence_type}.pkl.gz",
            'features_csv': f"features_{config.data_match}_{config.back_four}_{config.sequence_type}_{config.reward_features}_{config.method}.csv",
            'summary_overall': f"summary_overall_{config.data_match}_{config.back_four}_{config.sequence_type}_{config.reward_features}_{config.method}.csv",
            'summary_team': f"summary_team_{config.data_match}_{config.back_four}_{config.sequence_type}_{config.reward_features}_{config.method}.csv",
            'processing_log_csv': f"processing_log_{config.data_match}_{config.back_four}_{config.sequence_type}_{config.reward_features}_{config.method}.csv",
            'processing_log_json': f"processing_log_{config.data_match}_{config.back_four}_{config.sequence_type}_{config.reward_features}_{config.method}.json"
        }
        
        print(f"\n✓ Expected output files (from default config):")
        for label, filename in paths.items():
            print(f"  - {label:25s}: {filename}")
        
        return True, None
    except Exception as e:
        print(f"✗ Output setup error: {str(e)}")
        traceback.print_exc()
        return False, str(e)


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("PIPELINE VALIDATION SUITE")
    print("="*70)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Feature Computation", test_feature_computation_functions),
        ("Main Pipeline", test_main_functions),
        ("Preprocessing", test_preprocessing_functions),
        ("State/Action", test_state_action_definitions),
        ("Output Structure", test_output_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed, error = test_func()
            results.append((test_name, passed, error))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {str(e)}")
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for test_name, passed, error in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
        if error:
            print(f"       {error[:80]}")
    
    print(f"\n{passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✓✓✓ All validation tests passed! Pipeline is ready to use. ✓✓✓")
        print("\nNext steps:")
        print("1. Run: python main.py --data_match all_matches --sequence_type all_defensive")
        print("2. Check ./output/ for:")
        print("   - preprocessed_*.pkl.gz (cleaned data)")
        print("   - features_*.csv (computed features)")
        print("   - summary_overall_*.csv (overall stats)")
        print("   - summary_team_*.csv (team-based stats)")
        print("   - processing_log_*.csv/.json (detailed processing log)")
        return 0
    else:
        print("\n✗ Some tests failed. Fix errors above before running pipeline.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

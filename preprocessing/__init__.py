"""
Preprocessing Pipeline Module

Orchestrates data loading, preprocessing, and feature extraction for La Liga football tracking data.

Main entry point:
    from preprocessing.config import PipelineConfig
    from preprocessing.preprocessing import preprocess_all_matches
    
    config = PipelineConfig(data_match='barcelona_madrid')
    matches = preprocess_all_matches(config)
"""

__version__ = "1.0.0"

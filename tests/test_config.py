"""Tests for configuration module."""

import pytest
from preprocessing.config import PipelineConfig, DataMatchEnum, BackFourEnum


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.data_match == 'all_matches'
        assert config.back_four == 'all_players'
        assert config.sequence_type == 'all_defensive'
        assert config.reward_features == '5_features'
        assert config.method == 'feature_computation'
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            data_match='barcelona_madrid',
            back_four='back_four',
            sequence_type='negative_transition',
            reward_features='4_features',
            method='girl'
        )
        
        assert config.data_match == 'barcelona_madrid'
        assert config.back_four == 'back_four'
        assert config.sequence_type == 'negative_transition'
        assert config.reward_features == '4_features'
        assert config.method == 'girl'
    
    def test_invalid_data_match(self):
        """Test invalid data_match value."""
        with pytest.raises(AssertionError):
            PipelineConfig(data_match='invalid_option')
    
    def test_invalid_back_four(self):
        """Test invalid back_four value."""
        with pytest.raises(AssertionError):
            PipelineConfig(back_four='invalid_option')
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = PipelineConfig(
            data_match='barcelona_madrid',
            reward_features='4_features'
        )
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['data_match'] == 'barcelona_madrid'
        assert config_dict['reward_features'] == '4_features'
    
    def test_config_string_representation(self):
        """Test string representation."""
        config = PipelineConfig()
        config_str = str(config)
        
        assert 'PipelineConfig' in config_str
        assert 'all_matches' in config_str

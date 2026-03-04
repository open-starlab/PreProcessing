"""Tests for feature engineering module."""

import numpy as np
import pytest
from preprocessing.feature_engineering import DefensiveFeatureComputer


class TestSpaceScore:
    """Test space_score calculation."""
    
    def test_space_score_single_defender(self):
        """Test space score with single defender."""
        defenders = np.array([[10, 34]])
        ball = np.array([15, 34])
        
        score = DefensiveFeatureComputer.calculate_space_score(defenders, ball)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    def test_space_score_multiple_defenders(self):
        """Test space score with multiple defenders."""
        defenders = np.array([
            [10, 34],
            [15, 34],
            [20, 34]
        ])
        ball = np.array([25, 34])
        
        score = DefensiveFeatureComputer.calculate_space_score(defenders, ball)
        
        assert 0 <= score <= 1
    
    def test_space_score_empty_defenders(self):
        """Test space score with no defenders."""
        defenders = np.array([]).reshape(0, 2)
        ball = np.array([52.5, 34])
        
        score = DefensiveFeatureComputer.calculate_space_score(defenders, ball)
        
        assert score == 0.0


class TestPressureIndex:
    """Test pressure_index calculation."""
    
    def test_pressure_index_in_radius(self):
        """Test pressure index with defenders in radius."""
        defenders = np.array([
            [20, 30],
            [20, 35],
            [20, 40],
            [50, 50]  # Far away
        ])
        ball = np.array([20, 34])
        
        pressure = DefensiveFeatureComputer.calculate_pressure_index(
            defenders, ball, radius=5.0
        )
        
        assert 0 <= pressure <= 1
        assert pressure > 0  # Some defenders in radius
    
    def test_pressure_index_no_defenders(self):
        """Test pressure index with no defenders."""
        defenders = np.array([]).reshape(0, 2)
        ball = np.array([52.5, 34])
        
        pressure = DefensiveFeatureComputer.calculate_pressure_index(
            defenders, ball
        )
        
        assert pressure == 0.0


class TestStretchIndex:
    """Test stretch_index calculation."""
    
    def test_stretch_index_compact_line(self):
        """Test stretch index for compact defensive line."""
        defenders = np.array([
            [30, 32],
            [30, 34],
            [30, 36]
        ])
        
        stretch = DefensiveFeatureComputer.calculate_stretch_index(defenders)
        
        assert 0 <= stretch <= 1
        assert stretch < 0.3  # Compact line
    
    def test_stretch_index_stretched_line(self):
        """Test stretch index for stretched defensive line."""
        defenders = np.array([
            [10, 10],
            [50, 34],
            [90, 58]
        ])
        
        stretch = DefensiveFeatureComputer.calculate_stretch_index(defenders)
        
        assert 0 <= stretch <= 1
        assert stretch > 0.5  # Stretched line
    
    def test_stretch_index_single_defender(self):
        """Test stretch index with single defender."""
        defenders = np.array([[50, 34]])
        
        stretch = DefensiveFeatureComputer.calculate_stretch_index(defenders)
        
        assert stretch == 0.0


class TestLineHeight:
    """Test line height calculations."""
    
    def test_line_height_left_goal(self):
        """Test line height for team defending left goal."""
        defenders = np.array([
            [20, 30],
            [20, 40],
            [25, 35]
        ])
        
        rel_height, abs_height = DefensiveFeatureComputer.calculate_line_height(
            defenders, defending_side='left'
        )
        
        assert 0 <= rel_height <= 1
        assert abs_height > 0
    
    def test_line_height_right_goal(self):
        """Test line height for team defending right goal."""
        defenders = np.array([
            [80, 30],
            [80, 40],
            [75, 35]
        ])
        
        rel_height, abs_height = DefensiveFeatureComputer.calculate_line_height(
            defenders, defending_side='right'
        )
        
        assert 0 <= rel_height <= 1
        assert abs_height > 0
    
    def test_line_height_no_defenders(self):
        """Test line height with no defenders."""
        defenders = np.array([]).reshape(0, 2)
        
        rel_height, abs_height = DefensiveFeatureComputer.calculate_line_height(
            defenders
        )
        
        assert rel_height == 0.0
        assert abs_height == 0.0


class TestCompactness:
    """Test compactness calculation."""
    
    def test_compactness_tight_group(self):
        """Test compactness for tightly grouped defenders."""
        defenders = np.array([
            [40, 32],
            [40, 34],
            [40, 36]
        ])
        
        compactness = DefensiveFeatureComputer.calculate_compactness(defenders)
        
        assert 0 <= compactness <= 1
        assert compactness > 0.5  # Tight group = high compactness
    
    def test_compactness_spread_out(self):
        """Test compactness for spread out defenders."""
        defenders = np.array([
            [0, 0],
            [105, 68],
            [50, 34]
        ])
        
        compactness = DefensiveFeatureComputer.calculate_compactness(defenders)
        
        assert 0 <= compactness <= 1
        # Spread out = lower compactness values

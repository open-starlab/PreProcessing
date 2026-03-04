"""
Configuration and constants for the Defense Line Analysis pipeline.

This module defines:
- Field and game constants
- Feature engineering constants
- Pipeline configuration flags
"""

from enum import Enum
from dataclasses import dataclass
from typing import Literal


# ============================================================================
# FIELD AND GAME CONSTANTS
# ============================================================================

FIELD_LENGTH = 105.0  # meters
FIELD_WIDTH = 68.0    # meters
GOAL_WIDTH = 7.32     # meters
PENALTY_X = FIELD_LENGTH - 16.5  # penalty box distance from baseline
PENALTY_Y_MIN = (FIELD_WIDTH - 40.32) / 2
PENALTY_Y_MAX = PENALTY_Y_MIN + 40.32

# ============================================================================
# FEATURE ENGINEERING CONSTANTS
# ============================================================================

# Compactness weight
COMPACTNESS_WEIGHT = 0.5

# Pressure analysis
PRESSURE_RADIUS = 3.0  # meters

# Space calculation
SPACE_EPSILON = 1e-5

# Sequence analysis
SEQUENCE_LENGTH = 10  # frames

# ============================================================================
# TEAM MAPPING
# ============================================================================

TEAM_NAME_MAPPING = {
    'UD Almería': 'Almería',
    'Real Sociedad': 'Real Sociedad',
    'Athletic Club de Bilbao': 'Athletic Club',
    'Villarreal CF': 'Villarreal',
    'RC Celta de Vigo': 'Celta Vigo',
    'Getafe CF': 'Getafe',
    'UD Las Palmas': 'Las Palmas',
    'Sevilla FC': 'Sevilla',
    'Cadiz CF': 'Cádiz',
    'Atlético Madrid': 'Atlético Madrid',
    'RCD Mallorca': 'Mallorca',
    'Valencia CF': 'Valencia',
    'CA Osasuna': 'Osasuna',
    'Girona FC': 'Girona',
    'Real Betis Balompié': 'Real Betis',
    'FC Barcelona': 'Barcelona',
    'Deportivo Alavés': 'Deportivo Alavés',
    'Granada CF': 'Granada',
    'Rayo Vallecano': 'Rayo Vallecano',
    'Real Madrid CF': 'Real Madrid'
}

# ============================================================================
# ENUMERATION CLASSES
# ============================================================================

class DataMatchEnum(str, Enum):
    """Options for data match selection."""
    BARCELONA_MADRID = "barcelona_madrid"  # Matches containing Barcelona OR Real Madrid
    ALL_MATCHES = "all_matches"             # All La Liga matches


class BackFourEnum(str, Enum):
    """Options for defender selection (excluding goalkeeper)."""
    BACK_FOUR = "back_four"              # Only back four defenders (CBs, FBs)
    ALL_PLAYERS = "all_players"          # All defenders except goalkeeper


class SequenceTypeEnum(str, Enum):
    """Options for defensive sequence type."""
    NEGATIVE_TRANSITION = "negative_transition"  # 10-frame sequences AFTER possession loss
    ALL_DEFENSIVE = "all_defensive"              # 10-frame defensive sequences (any time)


class RewardFeaturesEnum(str, Enum):
    """Options for reward features."""
    FOUR_FEATURES = "4_features"
    FIVE_FEATURES = "5_features"


class MethodEnum(str, Enum):
    """Options for feature computation method."""
    GIRL = "girl"
    FEATURE_COMPUTATION = "feature_computation"


# ============================================================================
# PIPELINE CONFIGURATION DATACLASS
# ============================================================================

@dataclass
class PipelineConfig:
    """
    Configuration object for the complete pipeline.
    
    Attributes:
        data_match: Dataset to use (barcelona_madrid or all_matches)
        back_four: Player selection (back_four or all_players)
        sequence_type: Type of sequences (negative_transition or all_defensive)
        reward_features: Feature set (4_features or 5_features)
        method: Feature computation method (girl or feature_computation)
        data_dir: Path to organized data directory
        output_dir: Path to output directory
    """
    
    # Flag 1: Data selection (Barcelona/RM matches or all)
    data_match: Literal["barcelona_madrid", "all_matches"] = "all_matches"
    # Flag 2: Defender selection (back four or all defenders, excluding GK)
    back_four: Literal["back_four", "all_players"] = "all_players"
    # Flag 3: Sequence timing (after possession loss or any defensive time)
    sequence_type: Literal["negative_transition", "all_defensive"] = "all_defensive"
    # Flag 4: Feature set (4 features or all 5 features)
    reward_features: Literal["4_features", "5_features"] = "5_features"
    # Flag 5: Computation method (direct feature computation or GIRL-based)
    method: Literal["girl", "feature_computation"] = "feature_computation"
    
    # Directory paths
    data_dir: str = "./data/Laliga2023/24"
    output_dir: str = "./output"
    
    def __post_init__(self):
        """Validate configuration values."""
        valid_matches = {e.value for e in DataMatchEnum}
        valid_back_four = {e.value for e in BackFourEnum}
        valid_sequences = {e.value for e in SequenceTypeEnum}
        valid_features = {e.value for e in RewardFeaturesEnum}
        valid_methods = {e.value for e in MethodEnum}
        
        assert self.data_match in valid_matches, \
            f"Invalid data_match: {self.data_match}"
        assert self.back_four in valid_back_four, \
            f"Invalid back_four: {self.back_four}"
        assert self.sequence_type in valid_sequences, \
            f"Invalid sequence_type: {self.sequence_type}"
        assert self.reward_features in valid_features, \
            f"Invalid reward_features: {self.reward_features}"
        assert self.method in valid_methods, \
            f"Invalid method: {self.method}"
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            'data_match': self.data_match,
            'back_four': self.back_four,
            'sequence_type': self.sequence_type,
            'reward_features': self.reward_features,
            'method': self.method,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir
        }
    
    def __str__(self):
        """String representation of configuration."""
        return (
            f"PipelineConfig(\n"
            f"  data_match={self.data_match}\n"
            f"  back_four={self.back_four}\n"
            f"  sequence_type={self.sequence_type}\n"
            f"  reward_features={self.reward_features}\n"
            f"  method={self.method}\n"
            f")"
        )


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Feature sets based on reward_features flag
FEATURE_SET_4 = [
    'space_score',
    'pressure_index',
    'stretch_index',
    'line_height_relative'
]

FEATURE_SET_5 = FEATURE_SET_4 + ['line_height_absolute']

# Mapping
FEATURE_SETS = {
    '4_features': FEATURE_SET_4,
    '5_features': FEATURE_SET_5
}

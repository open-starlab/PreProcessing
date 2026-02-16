# -*- coding: utf-8 -*-
"""Configuration for UFA data processing.

Attributes
----------
field_length : float
    The length of a Ultimate field (109.73 meters).
field_width : float
    The width of a Ultimate field (37 meters).
players_per_team : int
    Number of players per team in Ultimate (7).
tracking_herz : int
    Frequency of tracking data (10 frames per second).
coordinate_scale : float
    Scale factor for coordinate conversion.
"""

# Ultimate field specifications (in meters for UFA data)
FIELD_LENGTH: float = 109.73  # 109.73 meters total field length
FIELD_WIDTH: float = 48.77  # 48.77 meters width
PLAYING_FIELD_LENGTH: float = 73.15  # 73.15 meters playing field (without end zones)
END_ZONE_LENGTH: float = 18.29  # 18.29 meters each end zone

# Player configuration
PLAYERS_PER_TEAM: int = 7  # Standard Ultimate has 7 players per team
MAX_SUBSTITUTIONS: int = 0  # Unlimited substitutions in Ultimate

# Data processing configuration
TRACKING_HERZ: int = 10  # UFA data frame rate (10 fps)
COORDINATE_SCALE: float = 1.0  # UFA data is in meters

# Team identifiers
OFFENSE_TEAM: str = "offense"
DEFENSE_TEAM: str = "defense"
DISC_ENTITY: str = "disc"

# Data columns mapping for UFA
UFA_COLUMNS = {
    "frame": "frame",
    "id": "id",
    "x": "x",
    "y": "y",
    "vx": "vx",
    "vy": "vy",
    "ax": "ax",
    "ay": "ay",
    "v_mag": "v_mag",
    "a_mag": "a_mag",
    "v_angle": "v_angle",
    "a_angle": "a_angle",
    "diff_v_a_angle": "diff_v_a_angle",
    "diff_v_angle": "diff_v_angle",
    "diff_a_angle": "diff_a_angle",
    "class": "class",
    "holder": "holder",
    "closest": "closest",
    "selected": "selected",
    "prev_holder": "prev_holder",
    "def_selected": "def_selected",
}

# Columns to remove from UFA data
COLUMNS_TO_REMOVE = ["selected", "prev_holder", "def_selected"]

# File name patterns
DEFAULT_FILE_PATTERN = r"(\d+)_(\d+)\.txt"  # quarter_possession.txt

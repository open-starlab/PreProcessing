# -*- coding: utf-8 -*-
"""Configuration for Ultimate Track data processing.

Attributes
----------
field_length : float
    The length of an Ultimate field (94 meters).
field_width : float
    The width of an Ultimate field (37 meters).
players_per_team : int
    Number of players per team in Ultimate (7).
max_substitutions : int
    Maximum substitutions allowed.
tracking_herz : int
    Frequency of tracking data (15 frames per second).
coordinate_scale : float
    Scale factor for coordinate conversion.
"""

# Ultimate Frisbee field specifications (in meters for Ultimate Track data)
FIELD_LENGTH: float = 94.0  # 94 meters total field length
FIELD_WIDTH: float = 37.0  # 37 meters width
PLAYING_FIELD_LENGTH: float = 58.0  # 58 meters playing field (without end zones)
END_ZONE_LENGTH: float = 18.0  # 18 meters each end zone

# Player configuration
ULTIMATE_PLAYERS_PER_TEAM: int = 7  # Standard Ultimate has 7 players per team
MAX_SUBSTITUTIONS: int = 0  # Unlimited substitutions in Ultimate

# Data processing configuration
TRACKING_HERZ: int = 15  # Ultimate Track data frame rate (15 fps)
COORDINATE_SCALE: float = 1.0  # Ultimate Track data is in meters

# Team identifiers
OFFENSE_TEAM: str = "offense"
DEFENSE_TEAM: str = "defense"
DISC_ENTITY: str = "disc"

# Data columns mapping
ULTIMATE_TRACK_COLUMNS = {
    "frame": "frame",
    "id": "id",
    "class": "class",
    "x": "x",
    "y": "y",
    "vx": "vx",
    "vy": "vy",
    "ax": "ax",
    "ay": "ay",
    "closest": "closest",
    "holder": "holder",
}


# Output column templates
def get_tracking_columns(team_name: str) -> list:
    """Generate column names for tracking data output."""
    base_columns = ["Period", "Time [s]"]
    player_columns = []

    for i in range(1, ULTIMATE_PLAYERS_PER_TEAM + 1):
        player_columns.extend([f"{team_name}_{i}_x", f"{team_name}_{i}_y"])

    return base_columns + player_columns + ["disc_x", "disc_y"]


# File name patterns
DEFAULT_FILE_PATTERN = r"(\d+)_(\d+)_(\d+)\.csv"  # game_half_point.csv

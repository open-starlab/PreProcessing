from bidict import bidict

FIELD_LENGTH = 105.0  # unit: meters
FIELD_WIDTH = 68.0  # unit: meters
STOP_THRESHOLD = 0.1  # unit: m/s
VERY_SLOW_THRESHOLD = 0.05  # unit: m/s (almost stationary threshold)
MAX_REALISTIC_TIME = 30.0  # unit: seconds (realistic maximum time for calculations)
FPS = 25

HOME_AWAY_MAP = {
    0: "BALL",
    1: "HOME",
    2: "AWAY",
}

PLAYER_ROLE_MAP = bidict(
    {
        0: "Substitute",
        1: "GK",
        2: "DF",
        3: "MF",
        4: "FW",
        -1: "Unknown",
    }
)

INPUT_EVENT_COLUMNS_LALIGA = [
    "game_id",
    "frame_id",
    "absolute_time",
    "match_status_id",
    "home_away",
    "event_x",
    "event_y",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "jersey_number",
    "player_role_id",
    "event_id",
    "event_name",
    "ball_x",
    "ball_y",
    "attack_history_num",
    "attack_direction",
    "series_num",
    "ball_touch",
    "success",
    "history_num",
    "attack_start_history_num",
    "attack_end_history_num",
    "is_goal",
    "is_shot",
    "is_pass",
    "is_dribble",
    "is_pressure",
    "is_ball_recovery",
    "is_block",
    "is_interception",
    "is_clearance",
    "formation",
]

INPUT_EVENT_COLUMNS_JLEAGUE = [
    "game_id",
    "frame_id",
    "absolute_time",
    "match_status_id",
    "home_away",
    "event_x",
    "event_y",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "jersey_number",
    "player_role_id",
    "event_id",
    "event_name",
    "ball_x",
    "ball_y",
    "attack_history_num",
    "attack_direction",
    "series_num",
    "ball_touch",
    "success",
    "history_num",
    "attack_start_history_num",
    "attack_end_history_num",
    "is_goal",
    "is_shot",
    "is_pass",
    "is_dribble",
    "is_ball_recovery",
    "is_block",
    "is_interception",
    "is_clearance",
    "is_cross",
    "is_through_pass",
]


INPUT_TRACKING_COLUMNS = [
    "game_id",
    "frame_id",
    "home_away",
    "jersey_number",
    "x",
    "y",
]

INPUT_PLAYER_COLUMNS_PVS = [
    "home_away",
    "team_id",
    "player_id",
    "player_name",
    "player_role",
    "jersey_number",
    "starting_member",
    "on_pitch",
]

INPUT_PLAYER_COLUMNS_EDMS = [
    "home_away",
    "team_id",
    "player_id",
    "player_name",
    "player_role",
    "jersey_number",
    "starting_member",
    "on_pitch",
    "height",
]


laliga_player_name_map = {
    "enrique barja afonso": "enrique barja alfonso",
    "robert navarro munoz": "robert navarro sanchez",
    "lamine yamal nasraoui ebana": "lamine yamal nasroui ebana",
    "andre gomes magalhaes de almeida": "domingos andre ribeiro almeida",
    "unai gomez etxebarria": "unai gomez echevarria",
    "moriba kourouma kourouma": "moriba ilaix",
    "jon magunacelaya argoitia": "jon magunazelaia argoitia",
    "fabricio angileri": "fabrizio german angileri",
}

ACTION_MAP = {
    "idle": 0,
    "shot": 1,
    "dribble": 2,
    "cross": 3,
    "pass": 4,
    "up_right": 5,
    "up": 6,
    "up_left": 7,
    "left": 8,
    "down_left": 9,
    "down": 10,
    "down_right": 11,
    "right": 12,
}

LALIGA_VALID_EVENTS = [
    "Pass",
    "Shot",
    "Interception",
    "Dribble",
    "Foul Won",
    "Miscontrol",
    "Ball Receipt*",
    "Ball Recovery",
    "Pressure",
    "Block",
    "Carry",
    "Clearance",
]

JLEAGUE_VALID_EVENTS = [
    "パス",
    "インターセプト",
    "クロス",
    "シュート",
    "スルーパス",
    "タッチ",
    "タックル",
    "クリア",
    "ボールゲイン",
    "トラップ",
    "ドリブル",
    "ファウル受ける",
    "フリックオン",
    "Goal Keeper",
]

EDMF_ACTION_MAP = {
    "idle": 0,
    "right": 1,
    "up_right": 2,
    "up": 3,
    "up_left": 4,
    "left": 5,
    "down_left": 6,
    "down": 7,
    "down_right": 8,
    "shot": 9,
    "dribble": 10,
    "through_pass": 11,
    "pass": 12,
    "cross": 13,
    "defensive_action": 14,
}

EDMF_ONBALL_ACTION_IDS = [9, 10, 11, 12, 13]

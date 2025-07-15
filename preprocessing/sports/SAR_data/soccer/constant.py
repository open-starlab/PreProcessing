from bidict import bidict
from scipy.special import softmax

FIELD_LENGTH = 105.0  # unit: meters
FIELD_WIDTH = 68.0  # unit: meters
STOP_THRESHOLD = 0.1  # unit: m/s

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
    'game_id',
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
    'game_id',
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
    "is_dribble"# preprocessing/sports/SAR_data/soccer/constant.py
]

import numpy as np
from scipy.special import softmax
from bidict import bidict

# --- field & basic maps ---
FIELD_LENGTH = 105.0  # meters
FIELD_WIDTH =  68.0   # meters
STOP_THRESHOLD = 0.1  # m/s

HOME_AWAY_MAP = {
    0: "BALL",
    1: "HOME",
    2: "AWAY",
}

PLAYER_ROLE_MAP = bidict({
    0:  "Substitute",
    1:  "GK",
    2:  "DF",
    3:  "MF",
    4:  "FW",
    -1: "Unknown",
})

# --- input schemas ---
INPUT_EVENT_COLUMNS_LALIGA = [
    "game_id","frame_id","absolute_time","match_status_id","home_away",
    "event_x","event_y","team_id","team_name","player_id","player_name",
    "jersey_number","player_role_id","event_id","event_name","ball_x","ball_y",
    "attack_history_num","attack_direction","series_num","ball_touch","success",
    "history_num","attack_start_history_num","attack_end_history_num",
    "is_goal","is_shot","is_pass","is_dribble","is_pressure","is_ball_recovery",
    "is_block","is_interception","is_clearance","formation",
]

INPUT_EVENT_COLUMNS_JLEAGUE = [
    *INPUT_EVENT_COLUMNS_LALIGA[:-1],  # everything except 'formation'
    "is_cross","is_through_pass",
]

INPUT_TRACKING_COLUMNS = [
    "game_id","frame_id","home_away","jersey_number","x","y",
]

INPUT_PLAYER_COLUMNS_JLEAGUE = [
    "home_away","team_id","player_id","player_name","player_role",
    "jersey_number","starting_member","on_pitch",
]

INPUT_PLAYER_COLUMNS_LALIGA = [
    *INPUT_PLAYER_COLUMNS_JLEAGUE, "height",
]

laliga_player_name_map = {
    "enrique barja afonso":            "enrique barja alfonso",
    "robert navarro munoz":            "robert navarro sanchez",
    "lamine yamal nasraoui ebana":     "lamine yamal nasroui ebana",
    "andre gomes magalhaes de almeida":"domingos andre ribeiro almeida",
    "unai gomez etxebarria":           "unai gomez echevarria",
    "moriba kourouma kourouma":        "moriba ilaix",
    "jon magunacelaya argoitia":       "jon magunazelaia argoitia",
    "fabricio angileri":               "fabrizio german angileri",
}

# --- QMix helpers ---
ROLES = ["striker", "midfielder", "defender", "goalkeeper"]

def assign_roles_softmax(agents: dict, ball_pos: np.ndarray) -> dict:
    """
    Assign each agent a role index based on softmax over negative distances to the ball.
    """
    dists = np.array([np.linalg.norm(agent["pos"] - ball_pos)
                      for agent in agents.values()])
    probs = softmax(-dists)
    role_ids = {}
    # highest prob → role 0, next → role 1, ...
    sorted_idxs = np.argsort(probs)[::-1]
    for idx, (pid, _) in zip(sorted_idxs, agents.items()):
        role_ids[pid] = int(idx) % len(ROLES)
    return role_ids

def compute_observations(agents, ball_pos):
    obs = []
    roles = assign_roles_softmax(agents, ball_pos)

    for pid, agent in agents.items():
        dist = np.linalg.norm(agent["pos"] - ball_pos)
        angle = np.arctan2(*(ball_pos - agent["pos"]))
        vel = agent["velocity"]
        obs_vec = np.array([
            dist,
            angle,
            *vel,
            agent["stamina"],
            agent["last_action"],
            roles[pid]
        ])
        obs.append(obs_vec)
    return obs, roles


INPUT_TRACKING_COLUMNS = [
    "game_id",
    "frame_id",
    "home_away",
    "jersey_number",
    "x",
    "y",
]

INPUT_PLAYER_COLUMNS_JLEAGUE = [
    "home_away",
    "team_id",
    "player_id",
    "player_name",
    "player_role",
    "jersey_number",
    "starting_member",
    "on_pitch",
]

INPUT_PLAYER_COLUMNS_LALIGA = [
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


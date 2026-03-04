"""
Feature Computation Module

Advanced defensive features for sequence analysis:
- Convex hull area
- Pitch Dominant Area (PDA)
- Compactness
- Pressure index
- Space score (4 weighted zones)
- Line height (absolute, relative to ball, normalized)
- Line velocity and stability
- Defense-attack correlation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial import ConvexHull
from tqdm import tqdm

from .config import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    PENALTY_X,
    PENALTY_Y_MIN,
    PENALTY_Y_MAX
)


# ============================================================================
# CONSTANTS
# ============================================================================

# Compactness weight (balance between area and distance)
LAMBDA = 0.5

# Pressure radius (meters)
PRESSURE_RADIUS = 3.0

# Space score epsilon (avoid division by zero)
SPACE_EPSILON = 1e-6


# Action labels
ACTION_MAP = {"backward": 0, "forward": 1, "compress": 2, "expand": 3}
ACTION_NAMES = {0: "backward", 1: "forward", 2: "compress", 3: "expand"}


# Role tags used for state/action extraction
DEF_TAGS = {
    'DF', 'ＤＦ', 'DEF', 'CB', 'RB', 'LB', 'RWB', 'LWB',
    'CENTERBACK', 'センターバック', 'CBK', 'BACK'
}


# ============================================================================
# SPATIAL FEATURES
# ============================================================================

def compute_convex_hull_area(points: List[Tuple[float, float]]) -> float:
    """
    Calculate convex hull area of player positions.
    
    Args:
        points: List of (x, y) coordinates
    
    Returns:
        Area in square meters, or 0 if insufficient points
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        return float(hull.area)
    except Exception:
        return 0.0


def compute_pda(
    defenders: List[Tuple[float, float]],
    attackers: List[Tuple[float, float]]
) -> float:
    """
    Compute Pitch Dominant Area: average minimum distance from
    3 most advanced attackers to closest defenders.
    
    Args:
        defenders: List of defender (x, y) positions
        attackers: List of attacker (x, y) positions
    
    Returns:
        PDA value, or NaN if insufficient data
    """
    if len(attackers) < 3 or len(defenders) == 0:
        return np.nan
    
    # Get 3 most advanced attackers (highest x)
    closest_attackers = sorted(attackers, key=lambda a: a[0], reverse=True)[:3]
    
    # For each attacker, find minimum distance to any defender
    distances = []
    for attacker in closest_attackers:
        min_dist = min(
            np.linalg.norm(np.array(attacker) - np.array(defender))
            for defender in defenders
        )
        distances.append(min_dist)
    
    return float(np.mean(distances))


def compute_compactness(
    defenders: List[Tuple[float, float]],
    attackers: List[Tuple[float, float]],
    lambda_weight: float = LAMBDA
) -> float:
    """
    Compute defensive compactness: weighted combination of
    convex hull area and pitch dominant area.
    
    Compactness = λ * convex_hull_area + (1 - λ) * PDA
    
    Args:
        defenders: List of defender positions
        attackers: List of attacker positions
        lambda_weight: Weight for convex hull (0-1)
    
    Returns:
        Compactness value, or NaN if insufficient data
    """
    hull_area = compute_convex_hull_area(defenders)
    pda = compute_pda(defenders, attackers)
    
    if np.isnan(pda):
        return np.nan
    
    compactness = lambda_weight * hull_area + (1 - lambda_weight) * pda
    return float(compactness)


# ============================================================================
# PRESSURE FEATURES
# ============================================================================

def compute_pressure_index(
    defenders: List[Tuple[float, float]],
    attackers: List[Tuple[float, float]],
    radius: float = PRESSURE_RADIUS
) -> int:
    """
    Count how many attackers are under pressure (within radius of any defender).
    
    Args:
        defenders: List of defender positions
        attackers: List of attacker positions
        radius: Pressure radius in meters
    
    Returns:
        Number of pressured attackers (0-3+)
    """
    count = 0
    for attacker in attackers:
        # Check if any defender is within radius
        for defender in defenders:
            distance = np.linalg.norm(np.array(attacker) - np.array(defender))
            if distance < radius:
                count += 1
                break  # Count each attacker only once
    
    return count


# ============================================================================
# SPACE CONTROL FEATURES
# ============================================================================

def in_central_final_third(pos: Tuple[float, float]) -> bool:
    """Check if position is in central final third (Zone 1: 35% weight)."""
    x, y = pos
    return x >= 70 and 37.5 <= y <= 67.5


def in_penalty_buffer(pos: Tuple[float, float]) -> bool:
    """Check if position is in penalty box buffer zone (Zone 2: 30% weight)."""
    x, y = pos
    return (PENALTY_X - 5 <= x <= FIELD_LENGTH and 
            PENALTY_Y_MIN - 5 <= y <= PENALTY_Y_MAX + 5)


def in_wing_pockets(pos: Tuple[float, float]) -> bool:
    """Check if position is in wing pockets (Zone 3: 20% weight)."""
    x, y = pos
    return x >= 80 and (y <= 10 or y >= (FIELD_WIDTH - 10))


def in_ball_radius(pos: Tuple[float, float], ball_pos: Optional[Tuple[float, float]]) -> bool:
    """Check if position is within 3m of ball (Zone 4: 15% weight)."""
    if ball_pos is None:
        return False
    return np.linalg.norm(np.array(pos) - np.array(ball_pos)) <= 3.0


def compute_space_score(
    defenders: List[Tuple[float, float]],
    attackers: List[Tuple[float, float]],
    ball_pos: Optional[Tuple[float, float]]
) -> float:
    """
    Compute weighted space control score across 4 critical zones.
    
    Space score = Σ(weight_i * (defenders_i - attackers_i) / (defenders_i + attackers_i))
    
    Zones:
        1. Central final third: 35%
        2. Penalty box buffer: 30%
        3. Wing pockets: 20%
        4. Ball radius (3m): 15%
    
    Args:
        defenders: List of defender positions
        attackers: List of attacker positions
        ball_pos: Ball position (x, y) or None
    
    Returns:
        Space score (-1 to +1, positive favors defense)
    """
    zones = [
        ('central_final_third', in_central_final_third, 0.35),
        ('penalty_buffer', in_penalty_buffer, 0.30),
        ('wing_pockets', in_wing_pockets, 0.20),
        ('ball_radius', lambda p: in_ball_radius(p, ball_pos), 0.15)
    ]
    
    total_score = 0.0
    
    for zone_name, zone_fn, weight in zones:
        defenders_in_zone = sum(1 for d in defenders if zone_fn(d))
        attackers_in_zone = sum(1 for a in attackers if zone_fn(a))
        
        # Compute zone score
        denominator = defenders_in_zone + attackers_in_zone + SPACE_EPSILON
        zone_score = weight * (defenders_in_zone - attackers_in_zone) / denominator
        total_score += zone_score
    
    return float(total_score)


# ============================================================================
# DEFENSIVE LINE FEATURES
# ============================================================================

def compute_line_height_features(
    defenders: List[Tuple[float, float]],
    ball_pos: Optional[Tuple[float, float]],
    use_back_four: bool = False
) -> Dict[str, float]:
    """
    Compute defensive line height in 3 variants:
    1. Absolute: Mean x-coordinate of line
    2. Relative to ball: Ball x - line x
    3. Normalized: Line x / field length (0-1)
    
    Args:
        defenders: List of defender positions
        ball_pos: Ball position (x, y) or None
        use_back_four: If True, use 4 deepest defenders; else all
    
    Returns:
        Dictionary with line_height_abs, line_height_rel_ball, line_height_norm
    """
    if len(defenders) == 0:
        return {
            'line_height_abs': np.nan,
            'line_height_rel_ball': np.nan,
            'line_height_norm': np.nan
        }
    
    # Select defenders for line calculation
    if use_back_four and len(defenders) >= 4:
        # Take 4 deepest (lowest x)
        line_defenders = sorted(defenders, key=lambda d: d[0])[:4]
    else:
        line_defenders = defenders
    
    # Mean x-coordinate
    line_x_mean = np.mean([d[0] for d in line_defenders])
    
    # 1. Absolute
    line_height_abs = float(line_x_mean)
    
    # 2. Relative to ball
    if ball_pos is not None and pd.notna(ball_pos[0]):
        line_height_rel_ball = float(ball_pos[0] - line_x_mean)
    else:
        line_height_rel_ball = np.nan
    
    # 3. Normalized
    line_height_norm = float(line_x_mean / FIELD_LENGTH)
    
    return {
        'line_height_abs': line_height_abs,
        'line_height_rel_ball': line_height_rel_ball,
        'line_height_norm': line_height_norm
    }


def compute_line_velocity_stability(
    line_depths: List[float],
    attacker_depths: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute defensive line movement patterns:
    - Line velocity: Rate of line movement
    - Line stability: Inverse of depth variation
    - Defense-attack correlation: How line reacts to attackers
    
    Args:
        line_depths: List of line x-coordinates over time
        attacker_depths: Optional list of attacker depths
    
    Returns:
        Dictionary with velocity, stability, correlation metrics
    """
    features = {}
    
    if len(line_depths) < 2:
        return {
            'line_velocity_mean': np.nan,
            'line_velocity_std': np.nan,
            'line_stability': np.nan,
            'defense_attack_correlation': np.nan
        }
    
    # Velocity (rate of change)
    line_velocities = np.diff(line_depths)
    features['line_velocity_mean'] = float(np.mean(line_velocities))
    features['line_velocity_std'] = float(np.std(line_velocities))
    
    # Stability (inverse of variation)
    line_std = np.std(line_depths)
    features['line_stability'] = float(1.0 / (1.0 + line_std))
    
    # Correlation with attackers
    if attacker_depths is not None and len(attacker_depths) == len(line_depths):
        correlation = np.corrcoef(line_depths, attacker_depths)[0, 1]
        features['defense_attack_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
    else:
        features['defense_attack_correlation'] = np.nan
    
    return features


# ============================================================================
# STATE & ACTION EXTRACTION
# ============================================================================

def _extract_xy(player: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Extract (x, y) coordinates from a player dictionary."""
    pos = player.get('position', {})
    if isinstance(pos, dict):
        return pos.get('x'), pos.get('y')
    return None, None


def _is_gk(player: Dict) -> bool:
    """Check if player is goalkeeper."""
    if any(player.get(k) is True for k in ('is_goalkeeper', 'is_gk')):
        return True
    role = str(player.get('player_role', '') or player.get('role', '') or player.get('position', '')).upper()
    return 'GK' in role or 'GOALKEEP' in role


def _is_def(player: Dict) -> bool:
    """Check if player is defender (excluding goalkeeper)."""
    role_raw = (player.get('player_role') or player.get('role') or player.get('position') or '')
    role = str(role_raw).upper()
    return any(tag in role for tag in DEF_TAGS) and not _is_gk(player)


def _is_atk(player: Dict) -> bool:
    """Check if player is attacker/midfielder."""
    role_raw = (player.get('player_role') or player.get('role') or player.get('position') or '')
    role = str(role_raw).upper()
    return any(x in role for x in ('FW', 'ＦＷ', 'ATT', 'STRIK', 'CF', 'AM', 'LW', 'RW', 'MF', 'ＭＦ'))


def get_ball_x(state: Dict) -> float:
    """Extract ball x-coordinate."""
    pos = state.get("ball", {}).get("position", {})
    return float(pos.get("x", 0.0)) if isinstance(pos, dict) else 0.0


def compute_stretch_index(players: List[Dict]) -> float:
    """Compute convex hull area for defender positions."""
    defenders = [p for p in players if _is_def(p)]
    if len(defenders) < 3:
        return 0.0

    points = []
    for player in defenders:
        x, y = _extract_xy(player)
        if x is None or y is None:
            continue
        try:
            points.append([float(x), float(y)])
        except Exception:
            continue

    if len(points) < 3:
        return 0.0

    try:
        hull = ConvexHull(np.array(points))
        return float(hull.area)
    except Exception:
        return 0.0


def build_step_features(state_dict: Dict) -> List[float]:
    """
    Build 18-dim state feature vector:
    [x1, y1, ..., x8, y8, ball_x, ball_y]
    Defenders sorted by x-coordinate, zero-padded to 8.
    """
    players = state_dict.get('players', [])
    defenders = [p for p in players if _is_def(p)]

    xs, ys = [], []
    for player in defenders:
        x, y = _extract_xy(player)
        if x is None or y is None:
            continue
        try:
            xs.append(float(x))
            ys.append(float(y))
        except Exception:
            continue

    if xs:
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]

    features = []
    for i in range(8):
        if i < len(xs):
            features.extend([xs[i], ys[i]])
        else:
            features.extend([0.0, 0.0])

    ball_pos = state_dict.get('ball', {}).get('position', {})
    bx = float(ball_pos.get('x', 0.0)) if isinstance(ball_pos, dict) else 0.0
    by = float(ball_pos.get('y', 0.0)) if isinstance(ball_pos, dict) else 0.0
    features.extend([bx, by])

    return features


def classify_defensive_action(
    player: Dict,
    state_dict: Dict,
    prev_state_dict: Optional[Dict] = None
) -> str:
    """Classify defender action into backward/forward/compress/expand."""
    if prev_state_dict is None:
        return "backward"

    curr_pos = player.get('position', {})
    if not isinstance(curr_pos, dict):
        return "backward"

    curr_x = float(curr_pos.get('x', 0.0))
    curr_y = float(curr_pos.get('y', 0.0))

    player_id = player.get('player_id') or player.get('id')
    prev_player = next(
        (
            p for p in prev_state_dict.get('players', [])
            if (p.get('player_id') or p.get('id')) == player_id
        ),
        None
    )
    if prev_player is None:
        return "backward"

    prev_pos = prev_player.get('position', {})
    if not isinstance(prev_pos, dict):
        return "backward"

    prev_x = float(prev_pos.get('x', 0.0))
    prev_y = float(prev_pos.get('y', 0.0))

    dx = curr_x - prev_x
    dy = curr_y - prev_y
    move_mag = np.hypot(dx, dy)

    hold_thresh = 0.25
    min_dir = 0.5
    min_stretch = 0.3

    defs_now = [p for p in state_dict.get('players', []) if _is_def(p)]
    defs_prev = [p for p in prev_state_dict.get('players', []) if _is_def(p)]

    stretch_curr = compute_stretch_index(defs_now)
    stretch_prev = compute_stretch_index(defs_prev)
    stretch_change = stretch_curr - stretch_prev

    if abs(stretch_change) >= min_stretch:
        return "expand" if stretch_change > 0 else "compress"

    ball_x = get_ball_x(state_dict)
    if abs(dx) >= min_dir and abs(dx) >= abs(dy):
        if dx > 0:
            return "forward" if ball_x > curr_x else "backward"
        return "forward" if ball_x < curr_x else "backward"

    if move_mag < hold_thresh:
        return "forward" if ball_x > curr_x else "backward"

    return "forward" if dx >= 0 else "backward"


def _build_state_from_frame(
    frame_row: pd.Series,
    defending_role: str,
    attacking_role: str
) -> Dict:
    """Build a state dict with players and ball from one tracking frame."""
    state = {
        'players': [],
        'ball': {
            'position': {
                'x': float(frame_row.get('ball_x', 0.0)) if pd.notna(frame_row.get('ball_x')) else 0.0,
                'y': float(frame_row.get('ball_y', 0.0)) if pd.notna(frame_row.get('ball_y')) else 0.0
            }
        }
    }

    prefix_def = 'h' if defending_role == 'home' else 'a'
    prefix_att = 'a' if defending_role == 'home' else 'h'

    defenders = []
    for i in range(1, 24):
        x_col = f'{prefix_def}{i}_x'
        y_col = f'{prefix_def}{i}_y'
        x_val = frame_row.get(x_col)
        y_val = frame_row.get(y_col)
        if pd.notna(x_val) and pd.notna(y_val):
            defenders.append({
                'player_id': f'{defending_role}_{i}',
                'position': {'x': float(x_val), 'y': float(y_val)},
                'role': 'DF',
                'player_role': 'DF',
                'team': defending_role,
                'is_goalkeeper': False
            })

    # Heuristic GK exclusion: remove deepest defender by x (normalized orientation)
    if len(defenders) >= 5:
        deepest_idx = int(np.argmin([d['position']['x'] for d in defenders]))
        defenders[deepest_idx]['is_goalkeeper'] = True
        defenders[deepest_idx]['role'] = 'GK'
        defenders[deepest_idx]['player_role'] = 'GK'

    attackers = []
    for i in range(1, 24):
        x_col = f'{prefix_att}{i}_x'
        y_col = f'{prefix_att}{i}_y'
        x_val = frame_row.get(x_col)
        y_val = frame_row.get(y_col)
        if pd.notna(x_val) and pd.notna(y_val):
            attackers.append({
                'player_id': f'{attacking_role}_{i}',
                'position': {'x': float(x_val), 'y': float(y_val)},
                'role': 'FW',
                'player_role': 'FW',
                'team': attacking_role
            })

    state['players'] = defenders + attackers
    return state


# ============================================================================
# SEQUENCE FEATURE EXTRACTION
# ============================================================================

def aggregate_stats(values: List[float]) -> Dict[str, float]:
    """
    Compute mean for a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with mean only
    """
    if len(values) == 0:
        return {'mean': np.nan}
    
    return {'mean': float(np.nanmean(values))}


def extract_sequence_features_advanced(
    event_sequence: pd.DataFrame,
    df_tracking: pd.DataFrame,
    home_team_name: str,
    defending_team: str,
    back_four_flag: bool = False
) -> Dict[str, float]:
    """
    Extract advanced defensive features from event sequence.
    
    Uses the back_four flag to determine defender selection:
    - back_four_flag=True: Use 4 deepest defenders
    - back_four_flag=False: Use all defenders
    
    Args:
        event_sequence: DataFrame of events in sequence
        df_tracking: Complete tracking DataFrame
        home_team_name: Name of home team
        defending_team: Name of team defending
        back_four_flag: Whether to use back four or all defenders
    
    Returns:
        Dictionary of computed features
    """
    # Determine team roles
    defending_role = 'home' if defending_team == home_team_name else 'away'
    attacking_role = 'away' if defending_role == 'home' else 'home'
    prefix_def = 'h' if defending_role == 'home' else 'a'
    prefix_att = 'a' if defending_role == 'home' else 'h'
    
    # Storage for frame-level features
    compactness_vals = []
    pressure_vals = []
    space_vals = []
    line_height_abs_vals = []
    line_height_rel_ball_vals = []
    line_height_norm_vals = []
    line_depths = []
    attacker_depths = []
    
    # Process each event frame
    for _, event in event_sequence.iterrows():
        frame_id = event.get('skillcorner_frame')
        if pd.isna(frame_id):
            continue
        
        # Get tracking frame
        row = df_tracking[df_tracking['frame'] == int(frame_id)]
        if row.empty:
            continue
        row = row.iloc[0]
        
        # Extract defender positions
        defenders = []
        for i in range(1, 24):
            x_col = f'{prefix_def}{i}_x'
            y_col = f'{prefix_def}{i}_y'
            if pd.notna(row.get(x_col)) and pd.notna(row.get(y_col)):
                defenders.append((row[x_col], row[y_col]))
        
        # Extract attacker positions
        attackers = []
        for i in range(1, 24):
            x_col = f'{prefix_att}{i}_x'
            y_col = f'{prefix_att}{i}_y'
            if pd.notna(row.get(x_col)) and pd.notna(row.get(y_col)):
                attackers.append((row[x_col], row[y_col]))
        
        # Ball position
        ball_x = row.get('ball_x')
        ball_y = row.get('ball_y')
        ball_pos = (ball_x, ball_y) if pd.notna(ball_x) and pd.notna(ball_y) else None
        
        # Skip if insufficient data
        if len(defenders) < 3:
            continue
        
        # Determine which defenders to use for features
        if back_four_flag and len(defenders) >= 4:
            # Use 4 deepest defenders
            feature_defenders = sorted(defenders, key=lambda d: d[0])[:4]
        else:
            # Use all defenders
            feature_defenders = defenders
        
        # Compute features
        compactness_vals.append(compute_compactness(feature_defenders, attackers))
        pressure_vals.append(compute_pressure_index(feature_defenders, attackers))
        space_vals.append(compute_space_score(feature_defenders, attackers, ball_pos))
        
        # Line height features
        line_features = compute_line_height_features(defenders, ball_pos, use_back_four=back_four_flag)
        line_height_abs_vals.append(line_features['line_height_abs'])
        line_height_rel_ball_vals.append(line_features['line_height_rel_ball'])
        line_height_norm_vals.append(line_features['line_height_norm'])
        
        # Track line depths for velocity/stability
        if not np.isnan(line_features['line_height_abs']):
            line_depths.append(line_features['line_height_abs'])
            
            # Track attacker depth (most advanced attacker)
            if len(attackers) > 0:
                attacker_depths.append(max(a[0] for a in attackers))
    
    # Aggregate features (mean only)
    compactness_stats = aggregate_stats(compactness_vals)
    pressure_stats = aggregate_stats(pressure_vals)
    space_stats = aggregate_stats(space_vals)
    line_abs_stats = aggregate_stats(line_height_abs_vals)
    line_rel_ball_stats = aggregate_stats(line_height_rel_ball_vals)
    line_norm_stats = aggregate_stats(line_height_norm_vals)
    
    # Line velocity/stability
    velocity_features = compute_line_velocity_stability(line_depths, attacker_depths)
    
    # Compile all features (mean only)
    features = {
        # Compactness (stretch_index in paper)
        'stretch_index_mean': compactness_stats['mean'],
        
        # Pressure index (clamp to 0-3)
        'pressure_index_mean': round(min(3, max(0, pressure_stats['mean']))) if not np.isnan(pressure_stats['mean']) else 0,
        
        # Space score
        'space_score_mean': space_stats['mean'],
        
        # Line height (absolute)
        'line_height_absolute_mean': line_abs_stats['mean'],
        
        # Line height (relative to ball)
        'line_height_relative_mean': line_rel_ball_stats['mean'],
        
        # Line height (normalized)
        'line_height_norm_mean': line_norm_stats['mean'],
        
        # Line dynamics
        'line_velocity_mean': velocity_features['line_velocity_mean'],
        'line_stability': velocity_features['line_stability'],
        'defense_attack_correlation': velocity_features['defense_attack_correlation']
    }
    
    return features


def compute_features_for_all_sequences(
    event_sequences: List[Dict],
    df_tracking: pd.DataFrame,
    home_team_name: str,
    back_four_flag: bool = False
) -> pd.DataFrame:
    """
    Compute features for all event sequences with progress bar.
    
    Args:
        event_sequences: List of sequence dictionaries
        df_tracking: Complete tracking DataFrame
        home_team_name: Name of home team
        back_four_flag: Whether to use back four (True) or all defenders (False)
    
    Returns:
        DataFrame with features for each sequence
    """
    feature_data = []
    
    for instance in tqdm(event_sequences, desc="Computing features"):
        defending_team = instance['team_lost_possession']
        attacking_team = instance['team_gained_possession']
        period = instance['period']
        transition_idx = instance['transition_idx']
        label = instance['label']

        defending_role = 'home' if defending_team == home_team_name else 'away'
        attacking_role = 'away' if defending_role == 'home' else 'home'
        
        # Get player who lost possession if available
        player_lost_possession = None
        if 'events' in instance and not instance['events'].empty:
            player_lost_possession = instance['events'].iloc[0].get('player', None)
        
        # Extract features
        features = extract_sequence_features_advanced(
            instance['events'],
            df_tracking,
            home_team_name,
            defending_team,
            back_four_flag
        )

        # Build state/action signals across sequence frames
        state_vectors = []
        action_counts = {name: 0 for name in ACTION_MAP}
        prev_state = None

        for _, event_row in instance['events'].iterrows():
            frame_id = event_row.get('skillcorner_frame')
            if pd.isna(frame_id):
                continue

            track_row = df_tracking[df_tracking['frame'] == int(frame_id)]
            if track_row.empty:
                continue
            track_row = track_row.iloc[0]

            state = _build_state_from_frame(track_row, defending_role, attacking_role)
            state_vectors.append(build_step_features(state))

            defenders = [p for p in state.get('players', []) if _is_def(p)]
            for defender in defenders:
                action = classify_defensive_action(defender, state, prev_state)
                action_counts[action] += 1

            prev_state = state

        total_actions = sum(action_counts.values())
        if total_actions > 0:
            action_rates = {
                f"action_{name}_rate": action_counts[name] / total_actions
                for name in ACTION_MAP
            }
            dominant_action = max(action_counts, key=action_counts.get)
        else:
            action_rates = {
                f"action_{name}_rate": 0.0
                for name in ACTION_MAP
            }
            dominant_action = 'backward'
        
        # Add metadata
        features.update({
            'transition_idx': transition_idx,
            'label': label,
            'team_lost_possession': defending_team,
            'team_gained_possession': attacking_team,
            'period': period,
            'player_lost_possession': player_lost_possession,
            'state_dim': 18,
            'state_steps': len(state_vectors),
            'dominant_action': dominant_action,
            'action_backward': ACTION_MAP['backward'],
            'action_forward': ACTION_MAP['forward'],
            'action_compress': ACTION_MAP['compress'],
            'action_expand': ACTION_MAP['expand'],
            **action_rates
        })
        
        feature_data.append(features)
    
    return pd.DataFrame(feature_data)

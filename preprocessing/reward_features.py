"""
Reward Features Module (GIRL Method)

Implements Goal-based Interpretable Reward Learning feature computation.
Used when method="girl" flag is set.

Reward features (controlled by reward_features flag):
- 4_features: [space_score, stretch_index, pressure_index, line_height_relative]
- 5_features: [space_score, stretch_index, pressure_index, line_height_relative, line_height_absolute]
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.spatial import ConvexHull

from .config import FIELD_LENGTH, FIELD_WIDTH, PRESSURE_RADIUS


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _is_def(player: Dict) -> bool:
    """Check if player is a defender (excluding goalkeeper)."""
    position = player.get("position", "").lower()
    return position in ['centre-back', 'center back', 'fullback', 'right-back', 'left-back', 'defender']


def _is_atk(player: Dict) -> bool:
    """Check if player is an attacker."""
    position = player.get("position", "").lower()
    return position in ['forward', 'striker', 'winger', 'attacking midfielder', 'midfielder']


def get_ball_x(state: Dict) -> float:
    """Extract ball x-coordinate from state."""
    pos = state.get("ball", {}).get("position", {})
    return float(pos.get("x", 0.0)) if isinstance(pos, dict) else 0.0


def get_ball_y(state: Dict) -> float:
    """Extract ball y-coordinate from state."""
    pos = state.get("ball", {}).get("position", {})
    return float(pos.get("y", 0.0)) if isinstance(pos, dict) else 0.0


def zone_weight(x: float, y: float) -> float:
    """
    Return tactical importance weight based on pitch zone.
    
    Zones:
    - Final third (x >= 66) + central corridor (30 <= y <= 70): 2.0
    - Final third + half-spaces: 1.5
    - Middle third (33 <= x < 66): 1.0
    - Defensive third (x < 33): 0.5
    
    Args:
        x: X-coordinate (0-105)
        y: Y-coordinate (0-100, normalized width)
    
    Returns:
        Tactical weight (0.5 to 2.0)
    """
    if x >= 66:  # Final third
        if 30 <= y <= 70:  # Central corridor
            return 2.0
        return 1.5  # Half-spaces
    if x >= 33:  # Middle third
        return 1.0
    return 0.5  # Defensive third


# ============================================================================
# REWARD FEATURE COMPUTATIONS
# ============================================================================

def compute_space_score(
    state: Dict,
    defenders: List[Dict],
    attackers: List[Dict],
    decay: float = 0.5
) -> float:
    """
    Zone-weighted attacker space dominance score.
    Measures how much space attackers have relative to defenders.
    
    For each attacker:
    1. Find minimum distance to any defender
    2. Apply exponential decay: exp(-decay * min_dist)
    3. Weight by zone importance
    4. Sum across all attackers
    
    Higher score = attackers have more space (worse for defense)
    
    Args:
        state: Game state dictionary
        defenders: List of defender player dictionaries
        attackers: List of attacker player dictionaries
        decay: Decay rate for distance weighting (default 0.5)
    
    Returns:
        Space score (higher = more attacker space)
    """
    score = 0.0
    
    for attacker in attackers:
        pos_a = attacker.get("position", {})
        if not isinstance(pos_a, dict):
            continue
        
        xa = float(pos_a.get("x", 0))
        ya = float(pos_a.get("y", 0))
        
        # Find minimum distance to any defender
        distances = []
        for defender in defenders:
            pos_d = defender.get("position", {})
            if not isinstance(pos_d, dict):
                continue
            
            xd = float(pos_d.get("x", 0))
            yd = float(pos_d.get("y", 0))
            distances.append(np.hypot(xa - xd, ya - yd))
        
        if not distances:
            continue
        
        min_dist = min(distances)
        z_w = zone_weight(xa, ya)
        score += z_w * np.exp(-decay * min_dist)
    
    return float(score)


def compute_stretch_index(defenders: List[Dict]) -> float:
    """
    Convex hull area of defensive line.
    Measures lateral/depth dispersion of defenders.
    
    Larger area = more stretched defense (potentially vulnerable)
    Smaller area = more compact defense
    
    Args:
        defenders: List of defender player dictionaries
    
    Returns:
        Convex hull area in square meters (0 if insufficient defenders)
    """
    if not defenders or len(defenders) < 3:
        return 0.0
    
    # Extract positions
    points = []
    for defender in defenders:
        pos = defender.get("position", {})
        if isinstance(pos, dict):
            try:
                x = float(pos.get("x", 0))
                y = float(pos.get("y", 0))
                points.append([x, y])
            except (ValueError, TypeError):
                pass
    
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(np.array(points))
        return float(hull.area)
    except Exception:
        return 0.0


def compute_pressure_index(
    defenders: List[Dict],
    attackers: List[Dict],
    radius: float = PRESSURE_RADIUS
) -> int:
    """
    Count attackers with at least one nearby defender.
    Measures defensive pressure on attacking players.
    
    An attacker is "under pressure" if ANY defender is within radius.
    
    Args:
        defenders: List of defender player dictionaries
        attackers: List of attacker player dictionaries
        radius: Pressure radius in meters (default from config)
    
    Returns:
        Number of attackers under pressure (0 to len(attackers))
    """
    count = 0
    
    for attacker in attackers:
        pos_a = attacker.get("position", {})
        if not isinstance(pos_a, dict):
            continue
        
        xa = float(pos_a.get("x", 0))
        ya = float(pos_a.get("y", 0))
        
        # Check if any defender is within radius
        for defender in defenders:
            pos_d = defender.get("position", {})
            if not isinstance(pos_d, dict):
                continue
            
            xd = float(pos_d.get("x", 0))
            yd = float(pos_d.get("y", 0))
            
            if np.hypot(xa - xd, ya - yd) <= radius:
                count += 1
                break  # Count each attacker only once
    
    return int(count)


def compute_line_height_rel_ball(defenders: List[Dict], ball_x: float) -> float:
    """
    Relative advance/retreat positioning of defensive line.
    
    Line height = mean x-coordinate of all defenders
    Relative height = ball_x - line_height_abs
    
    Positive value = defenders ahead of ball (high press)
    Negative value = defenders behind ball (deep block)
    
    Args:
        defenders: List of defender player dictionaries
        ball_x: Ball x-coordinate
    
    Returns:
        Relative line height (ball_x - line_x)
    """
    if not defenders:
        return 0.0
    
    # Extract x-coordinates
    xs = []
    for defender in defenders:
        pos = defender.get("position", {})
        if isinstance(pos, dict):
            try:
                xs.append(float(pos.get("x", 0)))
            except (ValueError, TypeError):
                pass
    
    if not xs:
        return 0.0
    
    line_height_abs = float(np.mean(xs))
    return ball_x - line_height_abs


def compute_line_height_absolute(defenders: List[Dict]) -> float:
    """
    Absolute defensive line height (mean x-coordinate).
    
    Measures how high up the pitch the defensive line is positioned.
    Higher value = more advanced line
    Lower value = deeper line
    
    Args:
        defenders: List of defender player dictionaries
    
    Returns:
        Mean x-coordinate of defensive line (0-105)
    """
    if not defenders:
        return 0.0
    
    # Extract x-coordinates
    xs = []
    for defender in defenders:
        pos = defender.get("position", {})
        if isinstance(pos, dict):
            try:
                xs.append(float(pos.get("x", 0)))
            except (ValueError, TypeError):
                pass
    
    if not xs:
        return 0.0
    
    return float(np.mean(xs))


# ============================================================================
# MAIN REWARD FEATURE FUNCTION
# ============================================================================

def compute_reward_features(
    state: Dict,
    include_absolute_height: bool = False
) -> List[float]:
    """
    Compute reward feature vector for GIRL method.
    
    4 features (default):
        [space_score, stretch_index, pressure_index, line_height_relative]
    
    5 features (if include_absolute_height=True):
        [space_score, stretch_index, pressure_index, line_height_relative, line_height_absolute]
    
    Args:
        state: Game state dictionary with "players" and "ball" keys
        include_absolute_height: If True, return 5 features; else 4
    
    Returns:
        List of feature values
    """
    # Extract players
    players = state.get("players", [])
    defenders = [p for p in players if _is_def(p)]
    attackers = [p for p in players if _is_atk(p)]
    
    # Get ball position
    ball_x = get_ball_x(state)
    
    # Compute 4 core features
    space_score = compute_space_score(state, defenders, attackers, decay=0.5)
    stretch_index = compute_stretch_index(defenders)
    pressure_index = compute_pressure_index(defenders, attackers, radius=PRESSURE_RADIUS)
    line_height_rel_ball = compute_line_height_rel_ball(defenders, ball_x)
    
    # Base 4 features
    features = [
        space_score,
        stretch_index,
        pressure_index,
        line_height_rel_ball
    ]
    
    # Add 5th feature if requested
    if include_absolute_height:
        line_height_abs = compute_line_height_absolute(defenders)
        features.append(line_height_abs)
    
    return features


def compute_reward_features_from_config(state: Dict, reward_flag: str) -> List[float]:
    """
    Compute reward features based on config flag.
    
    Args:
        state: Game state dictionary
        reward_flag: "4_features" or "5_features"
    
    Returns:
        Feature vector of length 4 or 5
    """
    include_absolute = (reward_flag == "5_features")
    return compute_reward_features(state, include_absolute_height=include_absolute)


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def compute_reward_features_batch(
    states: List[Dict],
    reward_flag: str = "5_features"
) -> np.ndarray:
    """
    Compute reward features for multiple states.
    
    Args:
        states: List of game state dictionaries
        reward_flag: "4_features" or "5_features"
    
    Returns:
        Array of shape (n_states, n_features)
    """
    features_list = []
    
    for state in states:
        features = compute_reward_features_from_config(state, reward_flag)
        features_list.append(features)
    
    return np.array(features_list)


# ============================================================================
# FEATURE NAMES
# ============================================================================

def get_feature_names(reward_flag: str = "5_features") -> List[str]:
    """
    Get feature names for the given reward flag.
    
    Args:
        reward_flag: "4_features" or "5_features"
    
    Returns:
        List of feature names
    """
    base_features = [
        'space_score',
        'stretch_index',
        'pressure_index',
        'line_height_relative'
    ]
    
    if reward_flag == "5_features":
        return base_features + ['line_height_absolute']
    else:
        return base_features


# ============================================================================
# LOGGING
# ============================================================================

def print_feature_info(reward_flag: str = "5_features"):
    """Print information about computed reward features."""
    print("[REWARD-FEATS] GIRL reward features computed:")
    print("  - space_score: zone-weighted attacker space dominance")
    print("  - stretch_index: convex hull area of defensive line")
    print("  - pressure_index: count of pressured attackers")
    print("  - line_height_relative: ball_x - line_x (relative positioning)")
    
    if reward_flag == "5_features":
        print("  - line_height_absolute: mean x-coordinate of defensive line")
    
    print(f"\nTotal features: {len(get_feature_names(reward_flag))}")

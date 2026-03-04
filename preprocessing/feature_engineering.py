"""
Feature engineering module for defensive analysis.

This module computes features for defensive sequences:
- space_score: Compression efficiency
- pressure_index: Pressure application
- stretch_index: Line stretch
- line_height_relative: Relative defensive line height
- line_height_absolute: Absolute defensive line height

Methods:
- girl: GIRL reward features (4 or 5 features via reward_features.py)
- feature_computation: Advanced feature computation (via feature_computation.py)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.spatial import ConvexHull, distance_matrix

from .config import (
    FIELD_LENGTH, FIELD_WIDTH, COMPACTNESS_WEIGHT,
    PRESSURE_RADIUS, SPACE_EPSILON, PipelineConfig
)


# ============================================================================
# FEATURE COMPUTATION UTILITIES
# ============================================================================

class DefensiveFeatureComputer:
    """Compute defensive features from tracking data."""
    
    @staticmethod
    def get_defenders(
        match_data: Dict,
        defending_team: str,
        back_four_only: bool = False
    ) -> List[Tuple[float, float]]:
        """
        Get defender positions for a given frame.
        Goalkeeper is always excluded. If back_four_only, further restrict
        to center-backs and full-backs.
        
        Args:
            match_data: Match data dictionary
            defending_team: Name of defending team
            back_four_only: If True, only return back four (CBs, FBs).
                           If False, return all defenders except goalkeeper.
        
        Returns:
            List of (x, y) positions of defenders
        """
        trackable_objects = match_data['trackable_objects']
        
        # Identify defenders by position
        defender_positions = []
        back_four_positions = []
        
        for track_id, info in trackable_objects.items():
            if info['team'] != defending_team or info['role'] == 'ball':
                continue
            
            position = info['position'].lower()
            
            # Back four identification
            if position in ['centre-back', 'center back', 'fullback', 'right-back', 'left-back']:
                back_four_positions.append(info['id'])
            
            if position not in ['goalkeeper', 'goal keeper']:
                defender_positions.append(info['id'])
        
        # Use back four if requested
        if back_four_only:
            defender_positions = back_four_positions
        
        return defender_positions
    
    @staticmethod
    def calculate_space_score(
        defender_positions: np.ndarray,
        ball_position: Tuple[float, float],
        opponent_positions: np.ndarray = None
    ) -> float:
        """
        Calculate space score: measure of compression around the ball.
        
        Space score = 1 - (average_distance_to_ball / max_possible_distance)
        Higher values indicate better compression.
        
        Args:
            defender_positions: Array of shape (n, 2) with defender positions
            ball_position: (x, y) tuple of ball position
            opponent_positions: Optional opponent positions (unused in basic version)
        
        Returns:
            Space score between 0 and 1
        """
        if len(defender_positions) == 0:
            return 0.0
        
        # Calculate distances from defenders to ball
        distances = np.linalg.norm(defender_positions - ball_position, axis=1)
        avg_distance = np.mean(distances)
        
        # Normalize by maximum possible distance
        max_distance = np.sqrt(FIELD_LENGTH**2 + FIELD_WIDTH**2)
        
        # Space score (inverted so higher = better compression)
        space_score = 1.0 - (avg_distance / max_distance)
        return max(0.0, min(1.0, space_score))
    
    @staticmethod
    def calculate_pressure_index(
        defender_positions: np.ndarray,
        ball_position: Tuple[float, float],
        radius: float = PRESSURE_RADIUS
    ) -> float:
        """
        Calculate pressure index: number of defenders within pressure_radius of ball.
        
        Args:
            defender_positions: Array of shape (n, 2) with defender positions
            ball_position: (x, y) tuple of ball position
            radius: Pressure radius in meters
        
        Returns:
            Pressure index (0-1 normalized by max defenders)
        """
        if len(defender_positions) == 0:
            return 0.0
        
        distances = np.linalg.norm(defender_positions - ball_position, axis=1)
        defenders_in_radius = np.sum(distances <= radius)
        
        # Normalize by number of defenders (max 3-4 for realistic pressure)
        max_pressurers = 4
        pressure_index = defenders_in_radius / max_pressurers
        return min(1.0, pressure_index)
    
    @staticmethod
    def calculate_stretch_index(
        defender_positions: np.ndarray
    ) -> float:
        """
        Calculate stretch index: measure of defensive line stretching.
        
        Lower values = compact line, Higher values = stretched line
        
        Args:
            defender_positions: Array of shape (n, 2) with defender positions
        
        Returns:
            Stretch index (0-1)
        """
        if len(defender_positions) < 2:
            return 0.0
        
        # Calculate convex hull span
        try:
            hull = ConvexHull(defender_positions)
            hull_points = defender_positions[hull.vertices]
        except:
            hull_points = defender_positions
        
        # X-axis stretch (left-right coverage)
        x_stretch = np.max(hull_points[:, 0]) - np.min(hull_points[:, 0])
        
        # Normalize by field width
        stretch_index = min(1.0, x_stretch / FIELD_WIDTH)
        return stretch_index
    
    @staticmethod
    def calculate_line_height(
        defender_positions: np.ndarray,
        defending_side: str = 'left'
    ) -> Tuple[float, float]:
        """
        Calculate defensive line height metrics.
        
        Args:
            defender_positions: Array of shape (n, 2) with defender positions
            defending_side: 'left' or 'right' (which goal they defend)
        
        Returns:
            Tuple of (relative_height, absolute_height)
            - relative_height: distance from own goal / half-field length (0-1)
            - absolute_height: actual x-coordinate (0-FIELD_LENGTH)
        """
        if len(defender_positions) == 0:
            return 0.0, 0.0
        
        # Average x-coordinate (height) of defensive line
        avg_x = np.mean(defender_positions[:, 0])
        
        if defending_side == 'left':
            # Defending left goal (x near 0)
            relative_height = avg_x / (FIELD_LENGTH / 2)
            absolute_height = avg_x
        else:
            # Defending right goal (x near FIELD_LENGTH)
            relative_height = (FIELD_LENGTH - avg_x) / (FIELD_LENGTH / 2)
            absolute_height = avg_x
        
        # Clamp to [0, 1] for relative
        relative_height = max(0.0, min(1.0, relative_height))
        
        return relative_height, absolute_height
    
    @staticmethod
    def calculate_compactness(
        defender_positions: np.ndarray
    ) -> float:
        """
        Calculate defensive compactness: measure of how close defenders are.
        
        Uses convex hull area as measure.
        
        Args:
            defender_positions: Array of shape (n, 2)
        
        Returns:
            Compactness score (0-1, higher = more compact)
        """
        if len(defender_positions) < 3:
            return 0.0
        
        try:
            hull = ConvexHull(defender_positions)
            hull_area = hull.volume  # In 2D, volume is actually area
        except:
            return 0.0
        
        # Normalize by field area
        max_area = FIELD_LENGTH * FIELD_WIDTH
        normalized_area = hull_area / max_area
        
        # Inverse relationship: smaller area = more compact = higher score
        compactness = 1.0 - min(1.0, normalized_area)
        return compactness


# ============================================================================
# FEATURE EXTRACTION FROM SEQUENCES
# ============================================================================

def extract_features_from_sequence(
    match_data: Dict,
    start_frame: int,
    end_frame: int,
    config: PipelineConfig
) -> Dict:
    """
    Extract features for a defensive sequence.
    
    Args:
        match_data: Preprocessed match data
        start_frame: Start frame number
        end_frame: End frame number
        config: Pipeline configuration
    
    Returns:
        Dictionary with computed features
    """
    tracking_df = match_data['tracking']
    match_info = match_data['match_info']
    processor = match_data['processor']
    
    # Get frames in range
    sequence_frames = tracking_df[
        (tracking_df['frame'] >= start_frame) &
        (tracking_df['frame'] <= end_frame)
    ]
    
    if len(sequence_frames) == 0:
        return None
    
    # Determine defending team (opposite of possession)
    possession_team = sequence_frames.iloc[0]['possession_team']
    if possession_team == match_data['home_team']:
        defending_team = match_data['away_team']
    else:
        defending_team = match_data['home_team']
    
    # Get defender positions for this sequence
    back_four_only = config.back_four == 'back_four'
    defender_ids = DefensiveFeatureComputer.get_defenders(
        match_data,
        defending_team,
        back_four_only
    )
    
    # Aggregate features across frames
    space_scores = []
    pressure_indices = []
    stretch_indices = []
    line_heights_rel = []
    line_heights_abs = []
    compactness_scores = []
    
    for _, frame in sequence_frames.iterrows():
        # Get positions
        ball_pos = np.array([frame['ball_x'], frame['ball_y']])
        
        # Skip if ball position is invalid
        if np.isnan(ball_pos).any():
            continue
        
        # Extract defender coordinates
        if defending_team == match_data['home_team']:
            home_x = [x for x in frame['home_x'] if x is not None]
            home_y = [y for y in frame['home_y'] if y is not None]
            defender_positions = np.array(list(zip(home_x, home_y)))
        else:
            away_x = [x for x in frame['away_x'] if x is not None]
            away_y = [y for y in frame['away_y'] if y is not None]
            defender_positions = np.array(list(zip(away_x, away_y)))
        
        # Compute features
        space_score = DefensiveFeatureComputer.calculate_space_score(
            defender_positions, ball_pos
        )
        space_scores.append(space_score)
        
        pressure_idx = DefensiveFeatureComputer.calculate_pressure_index(
            defender_positions, ball_pos
        )
        pressure_indices.append(pressure_idx)
        
        stretch_idx = DefensiveFeatureComputer.calculate_stretch_index(
            defender_positions
        )
        stretch_indices.append(stretch_idx)
        
        rel_height, abs_height = DefensiveFeatureComputer.calculate_line_height(
            defender_positions,
            processor.home_side
        )
        line_heights_rel.append(rel_height)
        line_heights_abs.append(abs_height)
        
        compactness = DefensiveFeatureComputer.calculate_compactness(
            defender_positions
        )
        compactness_scores.append(compactness)
    
    if len(space_scores) == 0:
        return None
    
    # Aggregate: average across frames
    result = {
        'start_frame': start_frame,
        'end_frame': end_frame,
        'num_frames': len(space_scores),
        'space_score': float(np.mean(space_scores)),
        'pressure_index': float(np.mean(pressure_indices)),
        'stretch_index': float(np.mean(stretch_indices)),
        'line_height_relative': float(np.mean(line_heights_rel)),
        'compactness': float(np.mean(compactness_scores))
    }
    
    # Add line_height_absolute if 5_features flag is set
    if config.reward_features == '5_features':
        result['line_height_absolute'] = float(np.mean(line_heights_abs))
    
    return result


def compute_match_features(
    match_data: Dict,
    config: PipelineConfig
) -> pd.DataFrame:
    """
    Compute all features for a match based on method flag.
    
    Method routing:
    - method="girl": Use GIRL reward features (reward_features.py)
    - method="feature_computation": Use advanced features (feature_computation.py if transitions exist)
    
    Args:
        match_data: Preprocessed match data
        config: Pipeline configuration
    
    Returns:
        DataFrame with features for each sequence
    """
    # Check if we have transition-based data (from preprocess_with_synchronization)
    if 'transitions' in match_data and config.sequence_type == 'negative_transition':
        return compute_transition_features(match_data, config)
    
    # Otherwise use legacy sequence-based approach
    sequences = match_data.get('sequences', [])
    features_list = []
    
    for start_frame, end_frame in sequences:
        features = extract_features_from_sequence(
            match_data,
            start_frame,
            end_frame,
            config
        )
        
        if features is not None:
            features['match_id'] = match_data['match_id']
            features['home_team'] = match_data['home_team']
            features['away_team'] = match_data['away_team']
            features_list.append(features)
    
    if len(features_list) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(features_list)


def _build_state_from_transition_frame(
    frame_row: pd.Series,
    defending_role: str,
    attacking_role: str
) -> Dict:
    """
    Build state dict from a tracking frame for SAR generation.
    
    Args:
        frame_row: One row from transition_frames DataFrame
        defending_role: 'home' or 'away'
        attacking_role: 'away' or 'home'
    
    Returns:
        State dict with players and ball position
    """
    state = {
        'players': [],
        'ball': {
            'position': {
                'x': float(frame_row.get('ball_x', 0.0)) if pd.notna(frame_row.get('ball_x')) else 0.0,
                'y': float(frame_row.get('ball_y', 0.0)) if pd.notna(frame_row.get('ball_y')) else 0.0
            }
        }
    }
    
    # Add defenders
    prefix_def = 'h' if defending_role == 'home' else 'a'
    for i in range(1, 24):
        x_col = f'{prefix_def}{i}_x'
        y_col = f'{prefix_def}{i}_y'
        if x_col in frame_row.index and pd.notna(frame_row[x_col]) and pd.notna(frame_row[y_col]):
            state['players'].append({
                'player_id': f'{defending_role}_{i}',
                'position': {'x': float(frame_row[x_col]), 'y': float(frame_row[y_col])},
                'team': defending_role,
                'role': 'DF',
                'is_goalkeeper': False
            })
    
    # Add attackers
    prefix_att = 'a' if defending_role == 'home' else 'h'
    for i in range(1, 24):
        x_col = f'{prefix_att}{i}_x'
        y_col = f'{prefix_att}{i}_y'
        if x_col in frame_row.index and pd.notna(frame_row[x_col]) and pd.notna(frame_row[y_col]):
            state['players'].append({
                'player_id': f'{attacking_role}_{i}',
                'position': {'x': float(frame_row[x_col]), 'y': float(frame_row[y_col])},
                'team': attacking_role,
                'role': 'ATK',
                'is_goalkeeper': False
            })
    
    return state


def compute_transition_features(
    match_data: Dict,
    config: PipelineConfig
) -> dict:
    """
    Compute features for transitions using new modules.
    
    Routes to appropriate feature computation based on method flag:
    - method="girl": Use reward_features.py
    - method="feature_computation": Use feature_computation.py
    
    Args:
        match_data: Match data with transitions (from preprocess_with_synchronization)
        config: Pipeline configuration
    
    Returns:
        DataFrame with computed features
    """
    if config.method == "girl":
        # GIRL method: produce SAR dataset (via sar_preprocessing)
        # This path stores SAR data instead of computing features
        # SAR data will be saved in main.py as pickle files
        
        from .sar_preprocessing import create_sar_sequences, extract_ml_sequences
        
        # Convert transitions to event sequence format
        events_with_state = []
        
        for transition in match_data.get('transitions', []):
            transition_frames = transition.get('transition_frames', pd.DataFrame())
            
            if transition_frames.empty:
                continue
            
            defending_role = transition.get('defending_team_role', 'home')
            attacking_role = 'away' if defending_role == 'home' else 'home'
            
            # For each frame in transition, build a state dict
            for _, frame_row in transition_frames.iterrows():
                state = _build_state_from_transition_frame(
                    frame_row, defending_role, attacking_role
                )
                
                events_with_state.append({
                    "state": state,
                    "frame_id": frame_row.get('frame', None),
                    "period": transition.get('period', 1),
                    "transition_idx": transition.get('event_index', 0),
                    "match_id": match_data.get('match_id'),
                    "home_team": match_data.get('home_team'),
                    "away_team": match_data.get('away_team'),
                    "team_id": transition.get('defending_team_id', transition.get('defending_team')),
                    "label": transition.get('defense_label', 1)
                })
        
        if not events_with_state:
            # Return empty marker for SAR (will be handled in main.py)
            return {"sar_data": None, "match_id": match_data['match_id']}
        
        # Create SAR sequences
        sar_sequences = create_sar_sequences(events_with_state, sequence_length=10)
        
        if not sar_sequences:
            return {"sar_data": None, "match_id": match_data['match_id']}
        
        # Extract structured arrays (N, T, D) format
        X, A, R, metadata = extract_ml_sequences(sar_sequences, seq_len=10)
        
        # Return SAR data (not DataFrame) with match info
        return {
            "sar_data": {
                "states": X,          # (N_seq, 10, 21)
                "actions": A,         # (N_seq, 10)
                "rewards": R,         # (N_seq, 10, 4)
                "metadata": metadata
            },
            "match_id": match_data['match_id'],
            "home_team": match_data['home_team'],
            "away_team": match_data['away_team'],
            "is_sar": True
        }
    
    elif config.method == "feature_computation":
        # Use advanced feature computation
        from .feature_computation import compute_features_for_all_sequences
        from .transition_analysis import extract_event_sequences_with_restart_handling
        
        # Extract event sequences with restart handling
        print(f"Extracting event sequences for {match_data['match_id']}...")
        event_sequences, dropped = extract_event_sequences_with_restart_handling(
            match_data['transitions'],
            match_data['events_synced'],
            sequence_length=10,
            keep_partial=False
        )
        
        print(f"  ✅ {len(event_sequences)} sequences extracted")
        print(f"  🚫 {len(dropped)} sequences dropped")
        
        if len(event_sequences) == 0:
            return pd.DataFrame()
        
        # Compute features
        back_four_flag = (config.back_four == "back_four")
        features_df = compute_features_for_all_sequences(
            event_sequences,
            match_data['tracking'],
            match_data['home_team'],
            back_four_flag
        )
        
        # Add match info
        features_df['match_id'] = match_data['match_id']
        features_df['home_team'] = match_data['home_team']
        features_df['away_team'] = match_data['away_team']
        
        return features_df
    
    else:
        raise ValueError(f"Unknown method: {config.method}")

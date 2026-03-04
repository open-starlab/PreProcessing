"""
Transition Analysis Module

Handles:
- Extracting defensive transitions from possession changes
- Identifying back-four defenders
- Labeling transition outcomes
- Extracting features for sequence analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .config import FIELD_LENGTH, FIELD_WIDTH


# ============================================================================
# CONSTANTS
# ============================================================================

FRAME_RATE = 25  # SkillCorner data is 25 fps
PENALTY_BOX_X = FIELD_LENGTH - 16.5
DEFENSIVE_THIRD_X = FIELD_LENGTH * 2 / 3


# ============================================================================
# COORDINATE NORMALIZATION
# ============================================================================

def get_attack_direction(
    defending_team_role: str,
    period: int,
    home_side: str
) -> int:
    """
    Determine attack direction for coordinate normalization.
    
    Returns 1 if attacking left-to-right, -1 if attacking right-to-left.
    
    Args:
        defending_team_role: 'home' or 'away'
        period: Match period (1 or 2)
        home_side: 'left' or 'right' (which goal home team defends initially)
    
    Returns:
        Direction multiplier: 1 or -1
    """
    if period == 1:
        if (defending_team_role == 'home' and home_side == 'left') or \
           (defending_team_role == 'away' and home_side == 'right'):
            return 1
        else:
            return -1
    elif period == 2:
        if (defending_team_role == 'home' and home_side == 'right') or \
           (defending_team_role == 'away' and home_side == 'left'):
            return 1
        else:
            return -1
    else:
        return 1


def normalize_coordinates(
    df: pd.DataFrame,
    direction: int
) -> pd.DataFrame:
    """
    Normalize coordinates so attack is always left-to-right.
    
    Args:
        df: DataFrame with coordinate columns (*_x, *_y)
        direction: 1 for no change, -1 to flip
    
    Returns:
        DataFrame with normalized coordinates
    """
    if direction == -1:
        df = df.copy()
        
        # Flip x-coordinates
        x_cols = [col for col in df.columns if col.endswith('_x')]
        for col in x_cols:
            df[col] = FIELD_LENGTH - df[col]
        
        # Flip y-coordinates
        y_cols = [col for col in df.columns if col.endswith('_y')]
        for col in y_cols:
            df[col] = FIELD_WIDTH - df[col]
    
    return df


# ============================================================================
# TRANSITION EXTRACTION
# ============================================================================

def extract_transition_frames(
    df_tracking: pd.DataFrame,
    turnover_idx: int,
    buffer_s: float = 1.0
) -> pd.DataFrame:
    """
    Extract frames within buffer around possession change.
    
    Args:
        df_tracking: Tracking DataFrame
        turnover_idx: Index of turnover in tracking data
        buffer_s: Time buffer (seconds) before/after
    
    Returns:
        DataFrame of frames around transition
    """
    buffer_frames = int(buffer_s * FRAME_RATE)
    start_idx = max(0, turnover_idx - buffer_frames)
    end_idx = min(len(df_tracking) - 1, turnover_idx + buffer_frames)
    
    return df_tracking.iloc[start_idx:end_idx + 1].copy()


def extract_all_transitions(
    turnover_events: pd.DataFrame,
    df_tracking: pd.DataFrame,
    home_team_name: str,
    home_side: str,
    buffer_s: float = 1.0
) -> List[Dict]:
    """
    Extract all defensive transitions from turnovers.
    
    Args:
        turnover_events: DataFrame of turnover events
        df_tracking: Tracking DataFrame
        home_team_name: Name of home team
        home_side: Which goal home team defends
        buffer_s: Time buffer (seconds)
    
    Returns:
        List of transition dictionaries
    """
    transitions = []
    
    for _, row in turnover_events.iterrows():
        idx = int(row.get('tracking_idx', -1))
        
        # Validate index
        if idx < 0 or idx >= len(df_tracking):
            continue
        
        # Extract frames around transition
        transition_frames = extract_transition_frames(df_tracking, idx, buffer_s)
        
        if len(transition_frames) == 0:
            continue
        
        period = row['period']
        defending_team_name = row['prev_possession_team']
        attacking_team_name = row['possession_team']
        defending_team_role = 'home' if defending_team_name == home_team_name else 'away'
        
        # Get attack direction
        direction = get_attack_direction(defending_team_role, period, home_side)
        
        # Normalize coordinates
        transition_frames_norm = normalize_coordinates(transition_frames.copy(), direction)
        
        # Check if transition is in defensive third
        ball_x = transition_frames_norm.iloc[0].get('ball_x')
        if pd.isna(ball_x):
            continue
        
        in_defensive_third = ball_x >= DEFENSIVE_THIRD_X
        
        # Check defender presence
        if defending_team_role == 'home':
            defender_cols = [col for col in df_tracking.columns 
                           if col.startswith('h') and col.endswith('_x')]
        else:
            defender_cols = [col for col in df_tracking.columns 
                           if col.startswith('a') and col.endswith('_x')]
        
        defenders_present = transition_frames_norm.iloc[0][defender_cols].notna().sum()
        
        # Only include transitions in defensive third with 4+ defenders
        if in_defensive_third and defenders_present >= 4:
            transitions.append({
                'event_index': row.get('index'),
                'tracking_idx': idx,
                'defending_team': defending_team_name,
                'attacking_team': attacking_team_name,
                'defending_team_role': defending_team_role,
                'period': period,
                'seconds': transition_frames_norm.iloc[0].get('seconds'),
                'transition_frames': transition_frames_norm,
                'direction': direction
            })
    
    return transitions


# ============================================================================
# BACK-FOUR IDENTIFICATION
# ============================================================================

def identify_back_four_defenders(
    transition_frames: pd.DataFrame,
    defending_team_role: str,
    goalkeeper_ids: Dict[str, Optional[int]]
) -> List[Dict]:
    """
    Identify the 4 outfield defenders closest to own goal.
    
    Args:
        transition_frames: DataFrame of transition frames
        defending_team_role: 'home' or 'away'
        goalkeeper_ids: Dict with 'home' and 'away' goalkeeper IDs
    
    Returns:
        List of back-four data per frame
    """
    back_four_sequences = []
    goalkeeper_id = goalkeeper_ids.get(defending_team_role)
    prefix = 'h' if defending_team_role == 'home' else 'a'
    
    for _, frame in transition_frames.iterrows():
        player_positions = []
        
        # Get all player positions (excluding goalkeeper)
        for i in range(23):
            if i == goalkeeper_id:
                continue
            
            x_col = f'{prefix}{i+1}_x'
            y_col = f'{prefix}{i+1}_y'
            
            if x_col in frame.index and y_col in frame.index:
                if pd.notna(frame[x_col]) and pd.notna(frame[y_col]):
                    player_positions.append({
                        'player_id': i,
                        'x': frame[x_col],
                        'y': frame[y_col],
                        'position': (frame[x_col], frame[y_col])
                    })
        
        # Sort by x (closest to own goal first)
        player_positions.sort(key=lambda x: x['x'])
        
        # Take 4 closest to own goal
        back_four = player_positions[:4]
        back_four_positions = [p['position'] for p in back_four]
        
        back_four_sequences.append({
            'frame_seconds': frame.get('seconds'),
            'back_four_players': back_four,
            'back_four_positions': back_four_positions
        })
    
    return back_four_sequences


def identify_closest_attackers(
    transition_frames: pd.DataFrame,
    attacking_team_role: str,
    n_attackers: int = 3
) -> List[Dict]:
    """
    Identify n closest attackers to opponent goal.
    
    Args:
        transition_frames: DataFrame of transition frames
        attacking_team_role: 'home' or 'away'
        n_attackers: Number of closest attackers to extract
    
    Returns:
        List of attacking positions per frame
    """
    closest_attackers_sequences = []
    prefix = 'h' if attacking_team_role == 'home' else 'a'
    
    for _, frame in transition_frames.iterrows():
        attacker_positions = []
        
        # Get all attacker positions
        for i in range(23):
            x_col = f'{prefix}{i+1}_x'
            y_col = f'{prefix}{i+1}_y'
            
            if x_col in frame.index and y_col in frame.index:
                if pd.notna(frame[x_col]) and pd.notna(frame[y_col]):
                    attacker_positions.append({
                        'player_id': i,
                        'x': frame[x_col],
                        'y': frame[y_col],
                        'position': (frame[x_col], frame[y_col])
                    })
        
        # Sort by x (closest to goal first)
        attacker_positions.sort(key=lambda x: x['x'], reverse=True)
        
        # Take n closest to goal
        closest = attacker_positions[:n_attackers]
        closest_positions = [p['position'] for p in closest]
        
        closest_attackers_sequences.append({
            'frame_seconds': frame.get('seconds'),
            'closest_attackers': closest,
            'closest_attacker_positions': closest_positions
        })
    
    return closest_attackers_sequences


# ============================================================================
# TRANSITION LABELING
# ============================================================================

def invaded_penalty_area_by_tracking(
    transition_df: pd.DataFrame
) -> bool:
    """Check if ball entered penalty area during transition."""
    if transition_df[['ball_x', 'ball_y']].isnull().any().any():
        return False
    return transition_df['ball_x'].max() >= PENALTY_BOX_X


def enhanced_label_transition(
    event_row: pd.DataFrame,
    transition_df: pd.DataFrame,
    events: pd.DataFrame,
    period: int
) -> int:
    """
    Label transition as success (1) or failure (0).
    
    Failure if: penalty area invaded, shot attempted, or goal scored.
    
    Args:
        event_row: The turnover event row
        transition_df: Frames in transition
        events: All events DataFrame
        period: Match period
    
    Returns:
        0 for failure, 1 for success
    """
    start_time = transition_df['seconds'].min()
    end_time = transition_df['seconds'].max()
    
    # Get events in sequence
    sequence_events = events[
        (events['period'] == period) &
        (events['seconds'] >= start_time) &
        (events['seconds'] <= end_time)
    ]
    
    # Check for dangerous outcomes
    if invaded_penalty_area_by_tracking(transition_df):
        return 0
    
    if len(sequence_events[sequence_events['event_type'] == 'Shot']) > 0:
        return 0
    
    if len(sequence_events[sequence_events['event_type'] == 'Goal']) > 0:
        return 0
    
    return 1


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_sequence_features(
    event_sequence: pd.DataFrame,
    tracking_frames: pd.DataFrame,
    back_four_data: List[Dict],
    closest_attackers_data: List[Dict]
) -> Dict:
    """
    Extract comprehensive features from sequence.
    
    Args:
        event_sequence: Events in sequence
        tracking_frames: Tracking frames in sequence
        back_four_data: Back-four positions per frame
        closest_attackers_data: Attacker positions per frame
    
    Returns:
        Dictionary of computed features
    """
    features = {}
    
    # ===== EVENT-BASED FEATURES =====
    event_types = event_sequence['event_type'].value_counts()
    for event_type in ['Pass', 'Shot', 'Dribble', 'Tackle', 'Interception', 'Clearance']:
        features[f'event_count_{event_type.lower()}'] = int(event_types.get(event_type, 0))
    
    features['possession_changes'] = int(
        (event_sequence['possession_team'] != event_sequence['possession_team'].shift()).sum()
    )
    features['sequence_length'] = len(event_sequence)
    
    # Pass completion
    pass_events = event_sequence[event_sequence['event_type'] == 'Pass']
    if len(pass_events) > 0:
        features['pass_completion_rate'] = float(pass_events['pass_outcome'].isna().sum() / len(pass_events))
    else:
        features['pass_completion_rate'] = 0.0
    
    # ===== DEFENSIVE LINE FEATURES =====
    if back_four_data:
        defensive_metrics = []
        
        for frame_data in back_four_data:
            if len(frame_data['back_four_positions']) >= 4:
                positions = frame_data['back_four_positions']
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]
                
                metrics = {
                    'line_depth': float(np.mean(x_coords)),
                    'line_width': float(max(y_coords) - min(y_coords)),
                    'line_compactness': float(np.std(x_coords)) if len(x_coords) > 1 else 0.0,
                    'deepest_defender': float(min(x_coords)),
                    'highest_defender': float(max(x_coords))
                }
                defensive_metrics.append(metrics)
        
        # Aggregate metrics
        if defensive_metrics:
            for metric_name in defensive_metrics[0].keys():
                values = [m[metric_name] for m in defensive_metrics]
                features[f'defense_{metric_name}_mean'] = float(np.mean(values))
                features[f'defense_{metric_name}_std'] = float(np.std(values)) if len(values) > 1 else 0.0
                features[f'defense_{metric_name}_min'] = float(np.min(values))
                features[f'defense_{metric_name}_max'] = float(np.max(values))
    
    # ===== ATTACKING THREAT FEATURES =====
    if closest_attackers_data:
        attacking_metrics = []
        
        for frame_data in closest_attackers_data:
            if len(frame_data['closest_attacker_positions']) >= 3:
                positions = frame_data['closest_attacker_positions']
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]
                
                metrics = {
                    'attack_depth': float(np.mean(x_coords)),
                    'attack_width': float(max(y_coords) - min(y_coords)),
                    'attack_compactness': float(np.std(x_coords)) if len(x_coords) > 1 else 0.0,
                    'most_advanced': float(max(x_coords))
                }
                attacking_metrics.append(metrics)
        
        # Aggregate
        if attacking_metrics:
            for metric_name in attacking_metrics[0].keys():
                values = [m[metric_name] for m in attacking_metrics]
                features[f'attack_{metric_name}_mean'] = float(np.mean(values))
                features[f'attack_{metric_name}_std'] = float(np.std(values)) if len(values) > 1 else 0.0
                features[f'attack_{metric_name}_max'] = float(np.max(values))
    
    # ===== DEFENSE-ATTACK SPATIAL RELATIONSHIP =====
    defensive_depths = [m['deepest_defender'] for m in defensive_metrics] if defensive_metrics else []
    attacking_depths = [m['most_advanced'] for m in attacking_metrics] if attacking_metrics else []
    
    if defensive_depths and attacking_depths:
        gaps = [a - d for a, d in zip(attacking_depths, defensive_depths)]
        features['defense_attack_gap_mean'] = float(np.mean(gaps))
        features['defense_attack_gap_std'] = float(np.std(gaps)) if len(gaps) > 1 else 0.0
        features['defense_attack_gap_min'] = float(np.min(gaps))
    
    # ===== BALL TRAJECTORY FEATURES =====
    ball_positions = tracking_frames[['ball_x', 'ball_y']].dropna()
    if len(ball_positions) > 1:
        features['ball_movement_x'] = float(ball_positions['ball_x'].max() - ball_positions['ball_x'].min())
        features['ball_movement_y'] = float(ball_positions['ball_y'].max() - ball_positions['ball_y'].min())
        
        diffs_x = np.diff(ball_positions['ball_x'].values)
        diffs_y = np.diff(ball_positions['ball_y'].values)
        distances = np.sqrt(diffs_x**2 + diffs_y**2)
        features['ball_trajectory_length'] = float(np.sum(distances))
    
    # ===== TEMPORAL FEATURES =====
    if len(event_sequence) > 0:
        duration = event_sequence['seconds'].max() - event_sequence['seconds'].min()
        features['sequence_duration'] = float(duration)
        features['avg_time_between_events'] = float(duration / len(event_sequence)) if len(event_sequence) > 0 else 0.0
    
    return features


# ============================================================================
# ML-READY DATA PREPARATION
# ============================================================================

def create_feature_matrix(
    sequence_data: List[Dict]
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Create feature matrix from sequences.
    
    Args:
        sequence_data: List of sequence dictionaries with features
    
    Returns:
        Tuple of (feature_df, labels, metadata_df)
    """
    feature_dicts = [seq['features'] for seq in sequence_data]
    feature_df = pd.DataFrame(feature_dicts).fillna(0)
    
    labels = np.array([seq['label'] for seq in sequence_data])
    
    metadata = pd.DataFrame([
        {
            'sequence_id': f"seq_{i}",
            'defending_team': seq.get('defending_team'),
            'attacking_team': seq.get('attacking_team'),
            'period': seq.get('period')
        }
        for i, seq in enumerate(sequence_data)
    ])
    
    return feature_df, labels, metadata


def prepare_training_data(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Prepare training data with scaling and split.
    
    Args:
        feature_df: Feature DataFrame
        labels: Label array
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        Dictionary with train/test splits and scaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_df.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_df.columns.tolist()
    }


# ============================================================================
# RESTART EVENT HANDLING & SEQUENCE EXTRACTION
# ============================================================================

# Restart events that end sequences
RESTART_EVENTS = {
    'Throw In', 'Goal Kick', 'Corner', 'Kick Off', 'Substitution',
    'Offside', 'Foul Committed', 'Free Kick Won', 'Penalty Conceded',
    'Half End', 'End of Half', 'End of Match'
}

# Dangerous events that indicate defensive failure
DANGEROUS_EVENTS = {'Shot', 'Goal', 'Penalty Won'}


def extract_event_sequences_with_restart_handling(
    transitions: List[Dict],
    events: pd.DataFrame,
    sequence_length: int = 10,
    keep_partial: bool = False
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Extract event sequences with restart logic and label override.
    
    Logic:
    1. Extract up to sequence_length events after each transition
    2. Stop if restart event encountered
    3. Drop sequences shorter than sequence_length (unless keep_partial=True)
    4. Override label:
       - Label = 0 (failure) if Shot/Goal/Penalty occurs
       - Label = 1 (success) if restart interrupts sequence
       - Otherwise keep original transition label
    
    Args:
        transitions: List of transition dictionaries
        events: DataFrame of all events with 'seconds' column
        sequence_length: Number of events to extract
        keep_partial: Whether to keep sequences < sequence_length
    
    Returns:
        Tuple of (sequence_list, dropped_sequences_df)
    """
    # Ensure seconds column exists
    if 'seconds' not in events.columns:
        events = events.copy()
        events['seconds'] = events.apply(
            lambda x: x['minute'] * 60 + x['second'], 
            axis=1
        )
    
    sequence_list = []
    dropped_log = []
    
    for t in transitions:
        transition_time = t.get('seconds')
        period = t.get('period')
        transition_idx = t.get('event_index', t.get('transition_idx', 0))
        label = t.get('defense_label', t.get('label', 1))
        team_lost = t.get('team_lost_possession', t.get('defending_team'))
        team_gained = t.get('team_gained_possession', t.get('attacking_team'))
        
        # Get candidate events after transition
        candidate_events = events[
            (events['period'] == period) &
            (events['seconds'] > transition_time)
        ].sort_values('seconds')
        
        clean_sequence = []
        early_end_reason = None
        
        # Extract events until sequence_length or restart
        for _, row in candidate_events.iterrows():
            if row['event_type'] in RESTART_EVENTS:
                early_end_reason = f"Restart: {row['event_type']}"
                break
            
            clean_sequence.append(row)
            
            if len(clean_sequence) == sequence_length:
                break
        
        # Create DataFrame from sequence
        final_events = pd.DataFrame(clean_sequence)
        
        # Drop short sequences unless keep_partial is True
        if len(final_events) < sequence_length and not keep_partial:
            dropped_log.append({
                'transition_idx': transition_idx,
                'team': team_lost,
                'reason': early_end_reason or "Too short",
                'length': len(final_events)
            })
            continue
        
        # Override label based on events in sequence
        if len(final_events) > 0:
            if any(evt in DANGEROUS_EVENTS for evt in final_events['event_type'].values):
                final_label = 0  # Defensive failure
            elif early_end_reason is not None:
                final_label = 1  # Success due to interruption
            else:
                final_label = label  # Keep original label
        else:
            final_label = label
        
        sequence_list.append({
            'transition_idx': transition_idx,
            'team_lost_possession': team_lost,
            'team_gained_possession': team_gained,
            'period': period,
            'start_time': transition_time,
            'label': final_label,
            'events': final_events.reset_index(drop=True),
            'early_end_reason': early_end_reason
        })
    
    # Create dropped sequences DataFrame
    dropped_df = pd.DataFrame(dropped_log)
    
    return sequence_list, dropped_df


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main_sequence_preparation(
    transitions: List[Dict],
    events: pd.DataFrame,
    df_tracking: pd.DataFrame,
    home_team_name: str,
    back_four_flag: bool = False,
    sequence_length: int = 10,
    keep_partial: bool = False
) -> Dict:
    """
    Main function to prepare the complete dataset for training.
    
    Pipeline:
    1. Extract event sequences with restart handling
    2. Compute features using feature_computation module
    3. Prepare training data with train/test split
    4. Print summary statistics
    
    Args:
        transitions: List of transition dictionaries
        events: DataFrame of all events
        df_tracking: Complete tracking DataFrame
        home_team_name: Name of home team
        back_four_flag: Whether to use back four (True) or all defenders (False)
        sequence_length: Number of events per sequence
        keep_partial: Keep sequences shorter than sequence_length
    
    Returns:
        Dictionary with sequence_data, features, labels, metadata, training_data
    """
    from feature_computation import compute_features_for_all_sequences
    
    print("=" * 50)
    print("PREPARING TRANSITION SEQUENCE DATASET")
    print("=" * 50)
    
    # Step 1: Extract sequences with restart handling
    print(f"\nExtracting transition sequences (length={sequence_length})...")
    event_sequences, dropped_sequences = extract_event_sequences_with_restart_handling(
        transitions, 
        events, 
        sequence_length=sequence_length,
        keep_partial=keep_partial
    )
    print(f"✅ Extracted {len(event_sequences)} valid sequences.")
    print(f"🚫 Dropped {len(dropped_sequences)} sequences due to restart or short length.")
    
    if len(dropped_sequences) > 0:
        print("\nDropped sequences summary:")
        print(dropped_sequences['reason'].value_counts())
    
    # Step 2: Compute features
    print(f"\n{'Using back four defenders' if back_four_flag else 'Using all defenders'}")
    print("Computing features...")
    features_df = compute_features_for_all_sequences(
        event_sequences,
        df_tracking,
        home_team_name,
        back_four_flag
    )
    
    print(f"✅ Feature matrix shape: {features_df.shape}")
    print(f"   Number of features: {features_df.shape[1] - 6}")  # Exclude metadata columns
    
    # Step 3: Prepare training data
    print("\nPreparing training data...")
    
    # Extract labels and feature columns
    labels = features_df['label'].values
    metadata_cols = ['transition_idx', 'label', 'team_lost_possession', 
                      'team_gained_possession', 'period', 'player_lost_possession']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    feature_matrix = features_df[feature_cols]
    metadata = features_df[metadata_cols]
    
    # Split and scale
    training_data = prepare_training_data(feature_matrix, labels)
    
    # Step 4: Print summary
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total samples: {len(labels)}")
    print(f"Successful defenses: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"Failed defenses: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")
    print(f"Training samples: {len(training_data['X_train'])}")
    print(f"Test samples: {len(training_data['X_test'])}")
    
    # Show sample features
    print("\n" + "=" * 50)
    print("SAMPLE FEATURES (Top 10)")
    print("=" * 50)
    for i, feature in enumerate(feature_cols[:10]):
        feature_values = feature_matrix[feature]
        print(f"  {feature:30s}: {feature_values.mean():.3f} ± {feature_values.std():.3f}")
    
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more features")
    
    return {
        'event_sequences': event_sequences,
        'dropped_sequences': dropped_sequences,
        'features_df': features_df,
        'feature_matrix': feature_matrix,
        'labels': labels,
        'metadata': metadata,
        'training_data': training_data
    }

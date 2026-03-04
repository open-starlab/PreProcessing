"""
Event-to-Tracking Synchronization Module

Handles:
- Synchronizing events with tracking frames
- Detecting defenders from events.csv
- Position classification and role identification
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ============================================================================
# POSITION-BASED DEFENDER CLASSIFICATION
# ============================================================================

DEFENDER_POSITIONS = {
    'goalkeeper': ['Goalkeeper', 'Goal Keeper'],
    'center_back': ['Centre-Back', 'Center Back', 'Right Center Back', 'Left Center Back'],
    'full_back': ['Right Back', 'Left Back', 'Right-Back', 'Left-Back', 'Fullback'],
    'defensive_midfielder': ['Center Defensive Midfield', 'Right Defensive Midfield', 
                            'Left Defensive Midfield']
}

BACK_FOUR_POSITIONS = {
    'center_back': ['Centre-Back', 'Center Back', 'Right Center Back', 'Left Center Back'],
    'full_back': ['Right Back', 'Left Back', 'Right-Back', 'Left-Back', 'Fullback']
}


def is_defender(position: str) -> bool:
    """
    Check if a position is a defender (excluding goalkeeper).
    
    Args:
        position: Position name from events.csv
    
    Returns:
        True if position is a defender
    """
    if pd.isna(position):
        return False
    
    for defender_type, positions in DEFENDER_POSITIONS.items():
        if defender_type == 'goalkeeper':
            continue
        if any(pos.lower() in position.lower() for pos in positions):
            return True
    
    return False


def is_goalkeeper(position: str) -> bool:
    """Check if position is goalkeeper."""
    if pd.isna(position):
        return False
    
    return any(pos.lower() in position.lower() for pos in DEFENDER_POSITIONS['goalkeeper'])


def is_back_four(position: str) -> bool:
    """
    Check if position is part of back four (center-back or full-back).
    
    Args:
        position: Position name from events.csv
    
    Returns:
        True if position is CB or FB
    """
    if pd.isna(position):
        return False
    
    for back_four_type, positions in BACK_FOUR_POSITIONS.items():
        if any(pos.lower() in position.lower() for pos in positions):
            return True
    
    return False


# ============================================================================
# EVENT-TRACKING SYNCHRONIZATION
# ============================================================================

def synchronize_events_with_tracking(
    events: pd.DataFrame,
    tracking_df: pd.DataFrame,
    frame_column: str = 'skillcorner_frame'
) -> pd.DataFrame:
    """
    Synchronize events with tracking frames.
    
    Args:
        events: Events DataFrame with skillcorner_frame column
        tracking_df: Tracking DataFrame with frame, seconds, period columns
        frame_column: Name of frame column in events
    
    Returns:
        Events DataFrame with tracking synchronization columns added
    """
    events = events.copy()
    
    # Ensure skillcorner_frame exists and is integer
    if frame_column not in events.columns:
        raise ValueError(f"Frame column '{frame_column}' not found in events")
    
    events = events.dropna(subset=[frame_column])
    events[frame_column] = events[frame_column].astype(int)
    
    # Prepare tracking data
    tracking_sync = tracking_df[['frame', 'seconds', 'period']].copy()
    tracking_sync['frame'] = tracking_sync['frame'].astype(int)
    
    # Merge events with tracking
    events = events.merge(
        tracking_sync,
        left_on=frame_column,
        right_on='frame',
        how='left',
        suffixes=('', '_tracking')
    )
    
    # Handle period column conflicts
    if 'period_tracking' in events.columns:
        events['period'] = events['period_tracking']
        events = events.drop(columns=['period_tracking'])
    
    # Rename frame column for backward compatibility
    events['tracking_idx'] = events[frame_column]
    
    # Drop rows without synchronization
    events = events.dropna(subset=['seconds'])
    
    return events


# ============================================================================
# DEFENDER EXTRACTION FROM EVENTS
# ============================================================================

def extract_starting_lineup_defenders(
    events: pd.DataFrame,
    team_name: str
) -> Dict[int, Dict]:
    """
    Extract defender information from Starting XI event.
    
    Args:
        events: Events DataFrame
        team_name: Team name to extract defenders for
    
    Returns:
        Dictionary mapping player_id to defender info
    """
    starting_xi = events[
        (events['event_type'] == 'Starting XI') & 
        (events['team'] == team_name)
    ]
    
    if len(starting_xi) == 0:
        return {}
    
    # Parse lineup from first Starting XI event for this team
    lineup_str = starting_xi.iloc[0].get('tactics_lineup', '{}')
    
    try:
        lineup_data = json.loads(lineup_str)
        lineup = lineup_data.get('lineup', [])
    except:
        return {}
    
    defenders = {}
    
    for player_entry in lineup:
        player_info = player_entry.get('player', {})
        position_info = player_entry.get('position', {})
        
        player_id = player_info.get('id')
        player_name = player_info.get('name')
        position = position_info.get('name')
        jersey_number = player_entry.get('jersey_number')
        
        if player_id and is_defender(position):
            defenders[player_id] = {
                'name': player_name,
                'position': position,
                'jersey_number': jersey_number,
                'is_back_four': is_back_four(position),
                'is_goalkeeper': is_goalkeeper(position)
            }
    
    return defenders


def extract_all_defenders(
    events: pd.DataFrame
) -> Dict[str, Dict[int, Dict]]:
    """
    Extract defenders for both teams from events.
    
    Args:
        events: Events DataFrame
    
    Returns:
        Dictionary mapping team_name -> player_id -> defender_info
    """
    defenders_by_team = {}
    
    for team_name in events['team'].dropna().unique():
        defenders = extract_starting_lineup_defenders(events, team_name)
        if defenders:
            defenders_by_team[team_name] = defenders
    
    return defenders_by_team


# ============================================================================
# POSSESSION AND EVENT FILTERING
# ============================================================================

def filter_possession_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Filter events to possession-relevant events only.
    
    Excludes: Starting XI, Half Start/End, Formation, Substitution, etc.
    
    Args:
        events: Events DataFrame
    
    Returns:
        Filtered events DataFrame
    """
    excluded_event_types = {
        'Starting XI', 'Half Start', 'Half End', '50/50', 'Formation', 
        'Substitution', 'Kick Off', 'End of Half', 'End of Match', 
        'End of Game', 'Goalkeeper', 'Offside', 'Foul Committed', 
        'Foul Won', 'Yellow Card', 'Red Card', 'Penalty Won',
        'Penalty Missed', 'Penalty Scored', 'Throw In', 'Corner', 
        'Goal Kick'
    }
    
    return events[~events['event_type'].isin(excluded_event_types)].copy()


def detect_possession_changes(
    events: pd.DataFrame
) -> pd.DataFrame:
    """
    Detect possession changes in events.
    
    Args:
        events: Possession-filtered events DataFrame
    
    Returns:
        Events DataFrame with possession_change column added
    """
    events = events.copy()
    events['prev_possession_team'] = events['possession_team'].shift(1)
    events['possession_change'] = (
        events['possession_team'] != events['prev_possession_team']
    )
    
    return events


def extract_turnover_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Extract turnover events (possession changes with valid team info).
    
    Args:
        events: Events DataFrame with possession_change column
    
    Returns:
        DataFrame of turnover events
    """
    turnovers = events[
        events['possession_change'] & 
        events['possession_team'].notna() & 
        events['prev_possession_team'].notna()
    ].copy()
    
    return turnovers


# ============================================================================
# VALIDATION AND QUALITY CHECKS
# ============================================================================

def validate_synchronization(
    events_synced: pd.DataFrame,
    min_synchronized_pct: float = 0.8
) -> bool:
    """
    Validate synchronization quality.
    
    Args:
        events_synced: Synchronized events DataFrame
        min_synchronized_pct: Minimum percentage of events that should have tracking sync
    
    Returns:
        True if synchronization quality is acceptable
    """
    if events_synced.empty:
        return False
    
    synced_count = events_synced['seconds'].notna().sum()
    total_count = len(events_synced)
    sync_pct = synced_count / total_count if total_count > 0 else 0
    
    return sync_pct >= min_synchronized_pct


def print_synchronization_report(
    events_original: pd.DataFrame,
    events_synced: pd.DataFrame,
    defenders_by_team: Dict[str, Dict]
) -> None:
    """Print synchronization report."""
    print("\n" + "="*70)
    print("EVENT-TRACKING SYNCHRONIZATION REPORT")
    print("="*70)
    
    print(f"\nOriginal events: {len(events_original)}")
    print(f"Synchronized events: {len(events_synced)}")
    
    synced_with_tracking = events_synced['seconds'].notna().sum()
    sync_pct = (synced_with_tracking / len(events_synced) * 100) if len(events_synced) > 0 else 0
    print(f"Events with tracking sync: {synced_with_tracking} ({sync_pct:.1f}%)")
    
    print(f"\nDefenders detected by team:")
    for team_name, defenders in defenders_by_team.items():
        back_four_count = sum(1 for d in defenders.values() if d['is_back_four'])
        total_defenders = sum(1 for d in defenders.values() if not d['is_goalkeeper'])
        print(f"  {team_name}: {total_defenders} defenders ({back_four_count} back-four)")
    
    print("="*70 + "\n")

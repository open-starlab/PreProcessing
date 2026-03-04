"""
Data preprocessing module for La Liga tracking and event data.

This module handles:
- Loading match data (events, tracking, metadata)
- Converting coordinate systems
- Creating tracking dataframes
- Building possession sequences
- Identifying defensive sequences/transitions 

For advanced analysis:
- Event-to-tracking synchronization: see event_tracking_sync.py
- Transition analysis & labeling: see transition_analysis.py
- Back-four detection: see transition_analysis.py

NOTE: Reward features and back four detection are left for separate
feature engineering steps to maintain modularity.
"""

import json
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .config import (
    FIELD_LENGTH, FIELD_WIDTH, TEAM_NAME_MAPPING,
    PipelineConfig
)

warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING
# ============================================================================

class MatchDataLoader:
    """Load and preprocess match data from files."""
    
    def __init__(self, match_dir: Path):
        """
        Initialize loader for a specific match.
        
        Args:
            match_dir: Path to directory containing match files
        """
        self.match_dir = Path(match_dir)
        self.events_path = self.match_dir / "events.csv"
        self.tracking_path = self.match_dir / "tracking.json"
        self.match_info_path = self.match_dir / "match.json"
    
    def load_all(self) -> Tuple[pd.DataFrame, list, dict]:
        """
        Load all match data.
        
        Returns:
            Tuple of (events_df, tracking_list, match_info_dict)
        """
        events = self._load_events()
        tracking = self._load_tracking()
        match_info = self._load_match_info()
        
        return events, tracking, match_info
    
    def _load_events(self) -> pd.DataFrame:
        """Load and return events CSV."""
        if not self.events_path.exists():
            raise FileNotFoundError(f"Events file not found: {self.events_path}")
        return pd.read_csv(self.events_path)
    
    def _load_tracking(self) -> list:
        """Load and return tracking JSON."""
        if not self.tracking_path.exists():
            raise FileNotFoundError(f"Tracking file not found: {self.tracking_path}")
        with open(self.tracking_path) as f:
            return json.load(f)
    
    def _load_match_info(self) -> dict:
        """Load and return match info JSON."""
        if not self.match_info_path.exists():
            raise FileNotFoundError(f"Match info file not found: {self.match_info_path}")
        with open(self.match_info_path, encoding='utf-8') as f:
            return json.load(f)


# ============================================================================
# TRACKING DATA PROCESSING
# ============================================================================

class TrackingProcessor:
    """Convert raw tracking data into standardized format."""
    
    def __init__(self, match_info: dict):
        """
        Initialize processor with match metadata.
        
        Args:
            match_info: Match information dictionary
        """
        self.match_info = match_info
        self.trackable_objects = self._build_trackable_objects()
        self.home_side = None
    
    def _build_trackable_objects(self) -> Dict:
        """
        Build mapping of trackable object IDs to player information.
        
        Returns:
            Dictionary mapping trackable_object ID to player info
        """
        trackable_objects = {}
        
        # Get team information
        home_team = self.match_info['home_team']
        away_team = self.match_info['away_team']
        
        team_dict = {
            home_team['id']: {
                'role': 'home',
                'name': TEAM_NAME_MAPPING.get(home_team['name'], home_team['name'])
            },
            away_team['id']: {
                'role': 'away',
                'name': TEAM_NAME_MAPPING.get(away_team['name'], away_team['name'])
            }
        }
        
        # Track player counts
        home_count = 0
        away_count = 0
        
        # Process all players
        for player in self.match_info['players']:
            role = team_dict[player['team_id']]['role']
            position = player['player_role']['name']
            
            if role == 'home':
                player_id = home_count
                home_count += 1
            else:
                player_id = away_count
                away_count += 1
            
            trackable_objects[player['trackable_object']] = {
                'name': f"{player['first_name']} {player['last_name']}".strip(),
                'team': team_dict[player['team_id']]['name'],
                'team_id': player['team_id'],
                'role': role,
                'id': player_id,
                'position': position,
                'shirt_number': player.get('shirt_number')
            }
        
        # Add ball
        trackable_objects[self.match_info['ball']['trackable_object']] = {
            'name': 'ball',
            'team': 'ball',
            'team_id': None,
            'role': 'ball',
            'position': 'ball'
        }
        
        return trackable_objects
    
    def process_frames(self, tracking_frames: list) -> pd.DataFrame:
        """
        Convert raw tracking frames to standardized DataFrame.
        
        Args:
            tracking_frames: List of frame dictionaries from JSON
        
        Returns:
            DataFrame with standardized tracking data (flattened columns)
        """
        frames_data = []
        home_gk_x = away_gk_x = None
        
        for frame in tracking_frames:
            frame_data = self._process_frame(frame)
            
            # Detect home side from goalkeeper positions
            if home_gk_x is None or away_gk_x is None:
                for obj in frame.get("data", []):
                    track_obj = self.trackable_objects.get(obj['trackable_object'])
                    if track_obj and track_obj['position'] == "Goalkeeper":
                        x_coord = obj['x']
                        if track_obj['role'] == 'home' and home_gk_x is None:
                            home_gk_x = x_coord
                        elif track_obj['role'] == 'away' and away_gk_x is None:
                            away_gk_x = x_coord
                
                if home_gk_x is not None and away_gk_x is not None:
                    self.home_side = 'left' if home_gk_x < away_gk_x else 'right'
            
            frame_data['home_side'] = self.home_side
            frames_data.append(frame_data)
        
        # Build column names for all 23 players + ball
        home_tracking_columns = []
        away_tracking_columns = []
        home_trackable_columns = []
        away_trackable_columns = []
        
        for i in range(1, 24):
            home_tracking_columns.extend([f"h{i}_x", f"h{i}_y"])
            away_tracking_columns.extend([f"a{i}_x", f"a{i}_y"])
            home_trackable_columns.append(f"h{i}_trackable_id")
            away_trackable_columns.append(f"a{i}_trackable_id")
        
        columns = (
            ["seconds", "period", "possession_team"] +
            home_tracking_columns +
            away_tracking_columns +
            home_trackable_columns +
            away_trackable_columns +
            ["home_side", "ball_x", "ball_y", "ball_z", "frame"]
        )
        
        df = pd.DataFrame(frames_data, columns=columns)
        df["seconds"] = df["seconds"].round(1)
        return df
    
    def _process_frame(self, frame: dict) -> dict:
        """
        Process a single frame.
        
        Args:
            frame: Single frame dictionary
        
        Returns:
            Processed frame data as dictionary with flattened columns
        """
        # Initialize coordinate arrays for 23 players per team
        home_coords = [None] * 23 * 2  # h1_x, h1_y, h2_x, h2_y, ...
        away_coords = [None] * 23 * 2  # a1_x, a1_y, a2_x, a2_y, ...
        home_trackable_ids = [None] * 23
        away_trackable_ids = [None] * 23
        
        ball_x = ball_y = ball_z = None
        
        # Parse timestamp
        timestamp = frame.get('timestamp', '')
        try:
            if timestamp:
                time_parts = timestamp.split(':')
                frame_second = (float(time_parts[0]) * 3600 + 
                              float(time_parts[1]) * 60 + 
                              float(time_parts[2]))
            else:
                frame_second = 0
        except (ValueError, IndexError):
            frame_second = 0
        
        # Process tracked objects
        for obj in frame.get('data', []):
            track_obj = self.trackable_objects.get(obj['trackable_object'])
            if not track_obj:
                continue
            
            # Convert coordinates (field coordinates -> normalized field)
            x_norm = obj['x'] + FIELD_LENGTH / 2
            y_norm = -obj['y'] + FIELD_WIDTH / 2
            
            if track_obj['role'] == 'home':
                idx = track_obj['id']
                home_coords[2 * idx] = round(x_norm, 2)
                home_coords[2 * idx + 1] = round(y_norm, 2)
                home_trackable_ids[idx] = obj['trackable_object']
            
            elif track_obj['role'] == 'away':
                idx = track_obj['id']
                away_coords[2 * idx] = round(x_norm, 2)
                away_coords[2 * idx + 1] = round(y_norm, 2)
                away_trackable_ids[idx] = obj['trackable_object']
            
            elif track_obj['role'] == 'ball':
                ball_x = round(x_norm, 2)
                ball_y = round(y_norm, 2)
                ball_z = round(obj.get('z', 0), 2)
        
        # Build flat dictionary with all columns in order
        frame_data = {
            'seconds': frame_second,
            'period': frame.get('period'),
            'possession_team': frame.get('possession', {}).get('group'),
        }
        
        # Add all home player coordinates (h1_x, h1_y, h2_x, h2_y, ...)
        for i in range(23):
            frame_data[f'h{i+1}_x'] = home_coords[2 * i]
            frame_data[f'h{i+1}_y'] = home_coords[2 * i + 1]
        
        # Add all away player coordinates (a1_x, a1_y, a2_x, a2_y, ...)
        for i in range(23):
            frame_data[f'a{i+1}_x'] = away_coords[2 * i]
            frame_data[f'a{i+1}_y'] = away_coords[2 * i + 1]
        
        # Add trackable IDs for home players
        for i in range(23):
            frame_data[f'h{i+1}_trackable_id'] = home_trackable_ids[i]
        
        # Add trackable IDs for away players
        for i in range(23):
            frame_data[f'a{i+1}_trackable_id'] = away_trackable_ids[i]
        
        # Add ball and frame information
        frame_data['home_side'] = None  # Will be set by process_frames
        frame_data['ball_x'] = ball_x
        frame_data['ball_y'] = ball_y
        frame_data['ball_z'] = ball_z
        frame_data['frame'] = frame.get('frame')
        
        return frame_data


# ============================================================================
# POSSESSION AND TRANSITION DETECTION
# ============================================================================

class PossessionAnalyzer:
    """Analyze possession and detect defensive phases."""
    
    @staticmethod
    def identify_transitions(events_df: pd.DataFrame) -> List[Dict]:
        """
        Identify possession transitions (positive and negative).
        
        Args:
            events_df: Events DataFrame
        
        Returns:
            List of transition dictionaries with timing information
        """
        transitions = []
        
        # Look for events that indicate possession change
        # Type 50: Dispossessed, Type 51: Error, Type 3: Block
        transition_types = [50, 51, 3, 14]  # Dispossessed, Error, Block, Miscontrol
        
        for idx, row in events_df.iterrows():
            if row.get('type', {}).get('id') in transition_types:
                transitions.append({
                    'frame': row.get('frame'),
                    'period': row.get('period'),
                    'timestamp': row.get('timestamp'),
                    'type': row.get('type', {}).get('name'),
                    'team': row.get('team', {}).get('name'),
                    'player': row.get('player', {}).get('name')
                })
        
        return transitions
    
    @staticmethod
    def identify_defensive_sequences(
        events_df: pd.DataFrame,
        tracking_df: pd.DataFrame,
        sequence_type: str
    ) -> List[Tuple[int, int]]:
        """
        Identify 10-frame defensive sequences based on possession analysis.
        
        Args:
            events_df: Events DataFrame
            tracking_df: Tracking DataFrame
            sequence_type: 'negative_transition' (after possession loss) or 
                          'all_defensive' (any defensive time)
        
        Returns:
            List of (start_frame, end_frame) tuples for 10-frame sequences
        """
        sequences = []
        
        if sequence_type == 'negative_transition':
            # 10-frame sequences starting AFTER possession loss
            transitions = PossessionAnalyzer.identify_transitions(events_df)
            for trans in transitions:
                frame = trans.get('frame')
                if frame and frame in tracking_df['frame'].values:
                    # Get 10-frame sequence starting after possession loss
                    start_idx = tracking_df[tracking_df['frame'] == frame].index[0]
                    end_idx = min(start_idx + 10, len(tracking_df) - 1)
                    sequences.append((tracking_df.loc[start_idx, 'frame'], 
                                    tracking_df.loc[end_idx, 'frame']))
        else:  # all_defensive
            # 10-frame defensive sequences regardless of possession loss
            tracking_df = tracking_df.sort_values('frame')
            frames = tracking_df['frame'].values
            
            # Split all frames into 10-frame sequences
            for i in range(0, len(frames) - 10, 10):
                sequences.append((frames[i], frames[i + 10]))
        
        return sequences


# ============================================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================================

def preprocess_match(
    match_dir: Path,
    config: PipelineConfig,
    save_tracking_csv: bool = True
) -> Dict:
    """
    Main preprocessing function for a single match.
    
    For all sequence_type values, applies:
    1. Event-to-tracking synchronization
    2. Defender extraction from Starting XI events
    3. Possession change detection
    4. Restart event filtering
    5. Coordinate normalization
    
    For negative_transition:
    - Also extracts transitions (turnovers)
    - Identifies back-four defenders
    - Labels transitions (success/failure)
    
    Args:
        match_dir: Path to match directory
        config: Pipeline configuration
        save_tracking_csv: Whether to save tracking output CSV
    
    Returns:
        Dictionary containing processed data
    """
    from .event_tracking_sync import (
        synchronize_events_with_tracking,
        extract_all_defenders,
        filter_possession_events,
        detect_possession_changes,
        extract_turnover_events,
        validate_synchronization,
        print_synchronization_report
    )
    from .transition_analysis import (
        extract_all_transitions,
        identify_back_four_defenders,
        identify_closest_attackers,
        enhanced_label_transition,
        normalize_coordinates,
        get_attack_direction,
        extract_transition_frames,
        RESTART_EVENTS
    )
    
    match_id = match_dir.name
    
    try:
        # Load data
        loader = MatchDataLoader(match_dir)
        events, tracking, match_info = loader.load_all()
        
        # Process tracking data
        processor = TrackingProcessor(match_info)
        tracking_df = processor.process_frames(tracking)
        
        # Save tracking output CSV if requested
        if save_tracking_csv:
            tracking_output_path = match_dir / "tracking_output.csv"
            tracking_df.to_csv(tracking_output_path, index=False)
            print(f"Tracking data saved for match {match_id}")
        
        # Filter for Barcelona vs Real Madrid if needed
        home_team = match_info['home_team']['name']
        away_team = match_info['away_team']['name']
        home_team_mapped = TEAM_NAME_MAPPING.get(home_team, home_team)
        away_team_mapped = TEAM_NAME_MAPPING.get(away_team, away_team)
        
        if config.data_match == 'barcelona_madrid':
            teams = {home_team_mapped, away_team_mapped}
            if not {'Barcelona', 'Real Madrid'}.issubset(teams):
                return None
        
        # ===== SYNCHRONIZATION & PREPARATION (for all sequence types) =====
        print(f"\n[Step 1] Synchronizing events with tracking for {match_id}...")
        events_synced = synchronize_events_with_tracking(
            events,
            tracking_df,
            frame_column='skillcorner_frame'
        )
        
        if not validate_synchronization(events_synced):
            print(f"  ⚠ Warning: Low synchronization quality for {match_id}")
        
        # Extract defenders
        print(f"[Step 2] Extracting defenders from events...")
        defenders_by_team = extract_all_defenders(events_synced)
        
        # Detect possession changes
        print(f"[Step 3] Detecting possession changes...")
        possession_events = filter_possession_events(events_synced)
        possession_events = detect_possession_changes(possession_events)
        
        # Remove restart events
        print(f"[Step 4] Filtering restart events...")
        non_restart_events = possession_events[
            ~possession_events['event_type'].isin(RESTART_EVENTS)
        ].reset_index(drop=True)
        
        missing_data_report = {
            'match_id': match_id,
            'events_raw': int(len(events)),
            'events_synced': int(len(events_synced)),
            'events_dropped_sync_or_time': int(len(events) - len(events_synced)),
            'possession_events': int(len(possession_events)),
            'restart_events_removed': int(len(possession_events) - len(non_restart_events)),
            'non_restart_events': int(len(non_restart_events)),
            'tracking_frames': int(len(tracking_df)),
            'sequence_type': config.sequence_type
        }
        
        # ===== HANDLE SEQUENCE TYPE =====
        if config.sequence_type == "negative_transition":
            # Extract turnovers and transitions
            print(f"[Step 5] Extracting turnovers...")
            turnover_events = extract_turnover_events(non_restart_events)
            missing_data_report['turnover_events'] = int(len(turnover_events))
            
            print(f"[Step 6] Extracting defensive transitions...")
            transitions = extract_all_transitions(
                turnover_events,
                tracking_df,
                home_team_mapped,
                processor.home_side,
                buffer_s=1.0
            )
            missing_data_report['transitions_extracted'] = int(len(transitions))
            
            print(f"[Step 7] Identifying back-four defenders and labeling...")
            goalkeeper_ids = {
                'home': processor.trackable_objects.get(
                    next((k for k, v in processor.trackable_objects.items() 
                          if v.get('role') == 'home' and v.get('position') == 'Goalkeeper'),
                    None)
                ),
                'away': processor.trackable_objects.get(
                    next((k for k, v in processor.trackable_objects.items() 
                          if v.get('role') == 'away' and v.get('position') == 'Goalkeeper'),
                    None)
                )
            }
            
            for transition in transitions:
                # Normalize coordinates
                transition_frames = transition['transition_frames'].copy()
                direction = transition['direction']
                transition_frames = normalize_coordinates(transition_frames, direction)
                
                # Identify back-four
                back_four_data = identify_back_four_defenders(
                    transition_frames,
                    transition['defending_team_role'],
                    goalkeeper_ids
                )
                transition['back_four_data'] = back_four_data
                
                # Identify closest attackers
                attacking_role = 'away' if transition['defending_team_role'] == 'home' else 'home'
                closest_attackers_data = identify_closest_attackers(
                    transition_frames,
                    attacking_role,
                    n_attackers=3
                )
                transition['closest_attackers_data'] = closest_attackers_data
                
                # Label transition
                event_row = events_synced[events_synced['index'] == transition['event_index']].iloc[0] \
                    if len(events_synced[events_synced['index'] == transition['event_index']]) > 0 else None
                
                if event_row is not None:
                    label = enhanced_label_transition(
                        event_row,
                        transition_frames,
                        events_synced,
                        transition['period']
                    )
                    transition['defense_label'] = label
                else:
                    transition['defense_label'] = 1
                
                # Store normalized frames
                transition['transition_frames'] = transition_frames
            
            print_synchronization_report(events, events_synced, defenders_by_team)
            print(f"Identified {len(transitions)} defensive transitions\n")
            print(f"[Missing Data Report] {match_id}: "
                f"raw={missing_data_report['events_raw']}, "
                f"synced={missing_data_report['events_synced']}, "
                f"dropped_sync={missing_data_report['events_dropped_sync_or_time']}, "
                f"restart_removed={missing_data_report['restart_events_removed']}, "
                f"turnovers={missing_data_report['turnover_events']}, "
                f"transitions={missing_data_report['transitions_extracted']}")
            
            return {
                'match_id': match_id,
                'home_team': home_team_mapped,
                'away_team': away_team_mapped,
                'events': events,
                'tracking': tracking_df,
                'match_info': match_info,
                'processor': processor,
                'events_synced': events_synced,
                'defenders_by_team': defenders_by_team,
                'transitions': transitions,
                'missing_data_report': missing_data_report,
                'trackable_objects': processor.trackable_objects
            }
        
        else:  # sequence_type = "all_defensive"
            # Extract 10-frame defensive sequences from non-restart events
            print(f"[Step 5] Extracting 10-frame defensive sequences...")
            sequences = []
            skipped_missing_frame = 0
            skipped_frame_not_found = 0
            skipped_too_short = 0
            
            for idx, (event_idx, event_row) in enumerate(non_restart_events.iterrows()):
                frame_id = event_row.get('skillcorner_frame')
                if pd.isna(frame_id):
                    skipped_missing_frame += 1
                    continue
                
                frame_id = int(frame_id)
                
                # Get 10 frames starting from this event
                tracking_idx = tracking_df[tracking_df['frame'] == frame_id].index
                if len(tracking_idx) == 0:
                    skipped_frame_not_found += 1
                    continue
                
                start_idx = tracking_idx[0]
                end_idx = min(start_idx + 10, len(tracking_df) - 1)
                
                if end_idx - start_idx < 10:
                    skipped_too_short += 1
                    continue  # Skip if < 10 frames
                
                # Extract frames
                sequence_frames = tracking_df.iloc[start_idx:end_idx + 1].copy()
                
                # Determine defending team (from possession)
                defending_team = event_row.get('possession_team', home_team_mapped)
                attacking_team = away_team_mapped if defending_team == home_team_mapped else home_team_mapped
                defending_role = 'home' if defending_team == home_team_mapped else 'away'
                
                # Get attack direction and normalize coordinates
                direction = get_attack_direction(defending_role, event_row.get('period', 1), processor.home_side)
                sequence_frames = normalize_coordinates(sequence_frames, direction)
                
                sequences.append({
                    'start_frame': frame_id,
                    'end_frame': int(sequence_frames.iloc[-1]['frame']),
                    'defending_team': defending_team,
                    'attacking_team': attacking_team,
                    'period': event_row.get('period', 1),
                    'frames': sequence_frames,
                    'direction': direction
                })
            
            print(f"Extracted {len(sequences)} 10-frame defensive sequences\n")
            missing_data_report.update({
                'events_skipped_missing_frame': int(skipped_missing_frame),
                'events_skipped_frame_not_found': int(skipped_frame_not_found),
                'sequences_skipped_too_short': int(skipped_too_short),
                'sequences_extracted': int(len(sequences))
            })
            print(f"[Missing Data Report] {match_id}: "
                  f"raw={missing_data_report['events_raw']}, "
                  f"synced={missing_data_report['events_synced']}, "
                  f"dropped_sync={missing_data_report['events_dropped_sync_or_time']}, "
                  f"restart_removed={missing_data_report['restart_events_removed']}, "
                  f"seq_missing_frame={missing_data_report['events_skipped_missing_frame']}, "
                  f"seq_frame_not_found={missing_data_report['events_skipped_frame_not_found']}, "
                  f"seq_too_short={missing_data_report['sequences_skipped_too_short']}, "
                  f"sequences={missing_data_report['sequences_extracted']}")
            
            return {
                'match_id': match_id,
                'home_team': home_team_mapped,
                'away_team': away_team_mapped,
                'events': events,
                'tracking': tracking_df,
                'match_info': match_info,
                'processor': processor,
                'events_synced': events_synced,
                'defenders_by_team': defenders_by_team,
                'sequences': sequences,
                'missing_data_report': missing_data_report,
                'trackable_objects': processor.trackable_objects
            }
    
    except Exception as e:
        print(f"Error processing match {match_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def preprocess_all_matches(config: PipelineConfig) -> List[Dict]:
    """
    Preprocess all matches in data directory.
    
    Uses unified preprocessing with synchronization for all sequence_type values.
    
    Args:
        config: Pipeline configuration
    
    Returns:
        List of preprocessed match data dictionaries
    """
    data_dir = Path(config.data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    match_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    preprocess_func = preprocess_match
    print("Using unified preprocessing with synchronization, restart filtering, and normalization")
    
    all_matches = []
    reports = []
    for match_dir in tqdm(match_dirs, desc="Preprocessing matches"):
        match_data = preprocess_func(match_dir, config)
        if match_data is not None:
            all_matches.append(match_data)
            if 'missing_data_report' in match_data:
                reports.append(match_data['missing_data_report'])
    
    print(f"\n✓ Processed {len(all_matches)} matches")
    
    if reports:
        reports_df = pd.DataFrame(reports)
        numeric_cols = [c for c in reports_df.columns if c not in ['match_id', 'sequence_type']]
        totals = reports_df[numeric_cols].sum(numeric_only=True)
        print("\n===== Missing Data Summary (All Matches) =====")
        print(f"Events raw: {int(totals.get('events_raw', 0))}")
        print(f"Events synced: {int(totals.get('events_synced', 0))}")
        print(f"Events dropped (sync/time): {int(totals.get('events_dropped_sync_or_time', 0))}")
        print(f"Restart events removed: {int(totals.get('restart_events_removed', 0))}")
        if config.sequence_type == 'negative_transition':
            print(f"Turnover events: {int(totals.get('turnover_events', 0))}")
            print(f"Transitions extracted: {int(totals.get('transitions_extracted', 0))}")
        else:
            print(f"Skipped (missing frame): {int(totals.get('events_skipped_missing_frame', 0))}")
            print(f"Skipped (frame not found): {int(totals.get('events_skipped_frame_not_found', 0))}")
            print(f"Skipped (too short): {int(totals.get('sequences_skipped_too_short', 0))}")
            print(f"Sequences extracted: {int(totals.get('sequences_extracted', 0))}")
    return all_matches


# ============================================================================
# ADVANCED ANALYSIS: SYNCHRONIZATION AND TRANSITIONS
# ============================================================================

def preprocess_with_synchronization(
    match_dir: Path,
    config: PipelineConfig
) -> Dict:
    """
    Full preprocessing with event-to-tracking synchronization.
    
    This function orchestrates:
    1. Standard preprocessing (preprocessing.preprocess_match)
    2. Event-to-tracking synchronization (event_tracking_sync)
    3. Defender identification from events (event_tracking_sync)
    4. Defensive transition extraction (transition_analysis)
    
    Args:
        match_dir: Path to match directory
        config: Pipeline configuration
    
    Returns:
        Dictionary with match data including synchronized events and transitions
    
    Example:
        >>> config = PipelineConfig(data_match='barcelona_madrid')
        >>> match_data = preprocess_with_synchronization(
        ...     Path('data/Laliga2023/24/123456'),
        ...     config
        ... )
        >>> print(f"Synchronized {len(match_data['events_synced'])} events")
        >>> print(f"Identified {len(match_data['transitions'])} transitions")
    
    Note:
        Requires event_tracking_sync and transition_analysis modules.
        See those modules for detailed function documentation.
    """
    from .event_tracking_sync import (
        synchronize_events_with_tracking,
        extract_all_defenders,
        filter_possession_events,
        detect_possession_changes,
        extract_turnover_events,
        validate_synchronization,
        print_synchronization_report
    )
    from .transition_analysis import (
        extract_all_transitions,
        identify_back_four_defenders,
        identify_closest_attackers,
        enhanced_label_transition
    )
    
    match_id = match_dir.name
    
    try:
        # Step 1: Standard preprocessing
        print(f"\n[Step 1] Standard preprocessing for {match_id}...")
        match_data = preprocess_match(match_dir, config)
        
        if match_data is None:
            return None
        
        events = match_data['events']
        tracking_df = match_data['tracking']
        match_info = match_data['match_info']
        processor = match_data['processor']
        home_side = processor.home_side
        
        # Step 2: Event-to-tracking synchronization
        print(f"[Step 2] Synchronizing events with tracking...")
        events_synced = synchronize_events_with_tracking(
            events,
            tracking_df,
            frame_column='skillcorner_frame'
        )
        
        # Validate synchronization
        if not validate_synchronization(events_synced):
            print(f"  ⚠ Warning: Low synchronization quality for {match_id}")
        
        # Step 3: Defender extraction
        print(f"[Step 3] Extracting defenders from events...")
        defenders_by_team = extract_all_defenders(events_synced)
        
        # Step 4: Possession change detection
        print(f"[Step 4] Detecting possession changes...")
        possession_events = filter_possession_events(events_synced)
        possession_events = detect_possession_changes(possession_events)
        turnover_events = extract_turnover_events(possession_events)
        
        # Step 5: Transition extraction
        print(f"[Step 5] Extracting defensive transitions...")
        home_team_name = match_data['home_team']
        transitions = extract_all_transitions(
            turnover_events,
            tracking_df,
            home_team_name,
            home_side,
            buffer_s=1.0
        )
        
        # Step 6: Back-four identification and labeling
        print(f"[Step 6] Identifying back-four defenders and labeling...")
        goalkeeper_ids = {
            'home': processor.trackable_objects.get(
                next((k for k, v in processor.trackable_objects.items() 
                      if v.get('role') == 'home' and v.get('position') == 'Goalkeeper'),
                None)
            ),
            'away': processor.trackable_objects.get(
                next((k for k, v in processor.trackable_objects.items() 
                      if v.get('role') == 'away' and v.get('position') == 'Goalkeeper'),
                None)
            )
        }
        
        for transition in transitions:
            # Identify back-four
            back_four_data = identify_back_four_defenders(
                transition['transition_frames'],
                transition['defending_team_role'],
                goalkeeper_ids
            )
            transition['back_four_data'] = back_four_data
            
            # Identify closest attackers
            attacking_role = 'away' if transition['defending_team_role'] == 'home' else 'home'
            closest_attackers_data = identify_closest_attackers(
                transition['transition_frames'],
                attacking_role,
                n_attackers=3
            )
            transition['closest_attackers_data'] = closest_attackers_data
            
            # Label transition
            event_row = events_synced[events_synced['index'] == transition['event_index']].iloc[0]
            label = enhanced_label_transition(
                event_row,
                transition['transition_frames'],
                events_synced,
                transition['period']
            )
            transition['defense_label'] = label
        
        # Print report
        print_synchronization_report(events, events_synced, defenders_by_team)
        print(f"Identified {len(transitions)} defensive transitions\n")
        
        # Extend original match_data with synchronization results
        match_data['events_synced'] = events_synced
        match_data['defenders_by_team'] = defenders_by_team
        match_data['transitions'] = transitions
        
        return match_data
    
    except Exception as e:
        print(f"Error preprocessing match {match_id} with synchronization: {str(e)}")
        return None

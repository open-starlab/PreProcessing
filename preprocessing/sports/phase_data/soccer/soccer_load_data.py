import json
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import xml.etree.ElementTree as ET
import os
import pickle
from typing import List, Dict, Any

def load_bepro(tracking_xml_path: str, tracking_json_paths: list, event_path: str, meta_data_path: str) -> pd.DataFrame:
    """
    Loads and processes event and tracking data from soccer match recordings.

    This function combines event data with tracking data by merging based on event time. It also adds 
    additional features extracted from metadata, such as player information, and converts position 
    coordinates to the correct scale for analysis.

    Args:
        tracking_path (str): Path to the XML or JSON file containing tracking data.
        event_path (str): Path to the CSV file containing event data.
        meta_path (str): Path to the XML file containing match metadata (pitch, teams, players, etc.).

    Returns:
        pd.DataFrame: A DataFrame containing the merged and processed event and tracking data, 
                      with additional features including player positions, speeds, ball position, 
                      and metadata (e.g., player names, shirt numbers, positions).
    """

    def extract_tracking_data_from_xml(xml_path: str) -> List[Dict[str, Any]]:
        """
        Parse the XML file and extract tracking data for players and the ball.

        Args:
            xml_path (str): Path to the XML file.
        Returns:
            list of dict: A list containing tracking information for each player and the ball in each frame.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tracking_data = []

        for frame in root.findall("frame"):
            frame_number = int(frame.get("frameNumber"))
            match_time = int(frame.get("matchTime"))
            
            for element in frame.findall("*"): 
                if element.tag == "player":
                    player_id = element.get("playerId")
                    loc = element.get("loc")
                elif element.tag == "ball":
                    player_id = "ball"
                    loc = element.get("loc")
                else:
                    continue
                if loc is None:
                    continue

                try:
                    x, y = map(float, loc.strip("[]").split(","))
                    
                    tracking_data.append({
                        "frame": frame_number,
                        "match_time": match_time,
                        "player_id": player_id,
                        "x": "{:.2f}".format(x * 105 - 52.5), 
                        "y": "{:.2f}".format(y * 68 - 34.0)
                    })
                except ValueError:
                    raise ValueError(f"Invalid location format for player {player_id} in frame {frame_number}")

        return tracking_data

    def extract_tracking_data_from_json(json_path: str, period: str) -> List[Dict[str, Any]]:
        """
        Parse the JSON file and extract tracking data.

        Args:
            json_path (str): Path to the JSON file.
        Returns:
            list of dict: A list containing tracking information for each player in each frame.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        tracking_data = []
        for frame_number, players in data.items():
            for player in players:
                try:
                    tracking_data.append({
                        "period": period,
                        "frame": int(frame_number),
                        "match_time": int(player.get("match_time", 0)),
                        "player_id": "ball" if player.get("player_id") == None else player.get("player_id"),
                        "x": "{:.2f}".format(float(player.get("x", 0) - 52.5)),
                        "y": "{:.2f}".format(float(player.get("y", 0) - 34.0))
                    })
                except ValueError:
                    raise ValueError(f"Invalid data format in frame {frame_number}")

        return tracking_data
    
    def devide_by_period(tracking_data_list: List[dict]) -> List[pd.DataFrame]:
        """Splits tracking data into multiple DataFrames based on period resets detected via frame number decreases.

        Args:
            tracking_data_list (List[dict]): A list of dictionaries containing raw tracking data.

        Returns:
            List[pd.DataFrame]: A list of DataFrames, each representing a single period with an added 'period' column.
        """
        if not tracking_data_list:
            return []

        df = pd.DataFrame(tracking_data_list)
        
        first_occurrence_of_frame = df.drop_duplicates(subset=['frame', 'match_time'], keep='first')
        frame_diff = first_occurrence_of_frame['frame'].diff().fillna(0)
        period_reset_indices = frame_diff[frame_diff < 0].index
        
        split_indices = [0]
        
        for reset_idx in period_reset_indices:
            if reset_idx > 0:
                split_indices.append(reset_idx) 
        
        split_indices.append(len(df))
        split_indices = sorted(list(set(split_indices)))
        period_df_list = []
        
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i+1]
            current_period = i + 1
            period_df = df.iloc[start_idx:end_idx].copy()
            period_df.loc[:, 'period'] = current_period
            period_df_list.append(period_df.reset_index(drop=True))

        return period_df_list
    
    def extract_meta_info_from_xml(xml_path: str) -> dict:
        """
        Extract team information (ID, name, side) from an XML metadata file.

        Args:
            xml_path (str): Path to the XML metadata file.
        Returns:
            dict: Dictionary in the format: {player_id: {'position': str, 'team_id': str, 'side': str}}.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        team_info = {}
        player_info = {}

        teams_element = root.find("teams")
        if teams_element is not None:
            for team in teams_element.findall("team"):
                team_id = team.get("id")
                team_name = team.get("name")
                side = team.get("side")
                
                if team_id:
                    team_info[team_id] = {
                        "team_name": team_name,
                        "side": side
                    }
        players_element = root.find("players")
        if players_element is not None:
            for player in players_element.findall("player"):
                player_id = player.get("id")
                player_name = player.get("name")
                team_id = player.get("teamId")
                position = player.get("position")
                
                if player_id:
                    side = team_info.get(team_id, {}).get("side")
                    team_name = team_info.get(team_id, {}).get("team_name")
                    
                    player_info[player_id] = {
                        "team_id": team_id,
                        "team_name": team_name,
                        "side": side,
                        "player_name": player_name,
                        "position": position,
                    }
        return player_info

    def extract_meta_info_from_json(json_path: str) -> dict:
        """
        Extract team information (ID, name, side) from an JSON metadata file.

        Args:
            xml_path (str): Path to the XML metadata file.
        Returns:
            dict: Dictionary in the format: {player_id: {'position': str, 'team_id': str, 'side': str}}.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        player_info = {}
        
        teams = {
            "home": data.get("home_team", {}),
            "away": data.get("away_team", {})
        }
        
        for side, team_data in teams.items():
            if team_data:
                team_id = str(team_data.get("team_id"))
                team_name = str(team_data.get("team_name"))
                
                if "players" in team_data:
                    for player in team_data["players"]:
                        player_id = str(player.get("player_id"))
                        player_name = str(player.get("full_name"))
                        position = player.get("initial_position_name")
                        
                        if player_id:
                            player_info[player_id] = {
                                "team_id": team_id,
                                "team_name": team_name,
                                "side": side,
                                "player_name": player_name,
                                "position": position,
                            }
                            
        return player_info

    def get_inplay_start_time(event_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'inplay_num' column to event_df.
        If the first word in filtered_event_types matches the specified event type,
        it is considered the start of a new in-play event, and inplay_num is incremented.

        Args:
            event_df (pd.DataFrame): A DataFrame containing soccer event data.
        Returns:
            pd.DataFrame: A DataFrame with the 'inplay_num' column added.
        """
        
        event_df = event_df.copy()
        START_EVENT_STRINGS = ['goalKick', 'throwIn', 'cornerKick', 'freeKick', 'goalAgainst']

        event_df.loc[:, 'first_event_type'] = event_df['filtered_event_types'].fillna('').str.split(' ').str[0]

        is_start_frame = pd.Series(False, index=event_df.index)
        is_start_frame.iloc[0] = True
        is_restart_event = event_df['first_event_type'].isin(START_EVENT_STRINGS)

        is_normal_restart = is_restart_event & (event_df['first_event_type'] != 'goalAgainst')
        is_start_frame = is_start_frame | is_normal_restart
        is_goal_against = event_df['first_event_type'] == 'goalAgainst'
        shifted_goal_against = is_goal_against.shift(periods=-1)
        filled_shifted = shifted_goal_against.fillna(False).astype(bool)
        is_start_frame = is_start_frame.astype(bool)
        is_start_frame = is_start_frame | filled_shifted

        event_df.loc[:, 'inplay_num'] = is_start_frame.cumsum().astype(int)

        event_df = event_df.drop(columns=['first_event_type'], errors='ignore')

        return event_df
    
    def get_tracking(tracking_df: pd.DataFrame, event_df: pd.DataFrame, player_info_df: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """
        Aggregates tracking data per frame into a wide-format DataFrame sorted by team side and position, and assigns in-play IDs.

        Args:
            tracking_df (pd.DataFrame): Processed tracking data containing frame, period, coordinates, and player IDs.
            event_df (pd.DataFrame): Processed event data containing match_time, period, and inplay_num.
            player_info_df (Dict[str, Dict[str, str]]): Metadata mapping player IDs to positions, team IDs, and team sides.

        Returns:
            pd.DataFrame: A wide-format tracking DataFrame structured frame-by-frame with normalized team orientations.
        """
        POSITION_ORDER = ['GK', 'CB', 'RWB', 'RB', 'LWB', 'LB', 'CDM', 'RM', 'CM', 'LM', 'CAM', 'RW', 'LW', 'CF']
        
        # -----------------------------------------------
        # 0. Player Information Merging and Preprocessing
        # -----------------------------------------------
        event_df = event_df.copy()
        player_map_df = pd.DataFrame.from_dict(player_info_df, orient='index').reset_index().rename(
            columns={'index': 'player_id', 'side': 'team_side', 'team_name': 'team_name'}
        )
        
        tracking_df['player_id'] = tracking_df['player_id'].astype(str)
        
        tracking_df = pd.merge(tracking_df, player_map_df, on='player_id', how='left')

        tracking_df.loc[tracking_df['player_id'] == 'ball', ['team_id', 'team_name', 'team_side', 'position', 'player_name']] = \
            ['ball', 'ball', 'ball', 'ball', 'ball']

        # -----------------------------------------------
        # 1. Determine Team Side (left/right) based on initial frame
        # -----------------------------------------------
        target_frame = tracking_df['frame'].min() + 10
        gk_data_initial = tracking_df[(tracking_df['position'] == 'GK') & (tracking_df['frame'] == target_frame)]
        
        left_team_id = gk_data_initial.loc[gk_data_initial['x'].idxmin(), 'team_id']
        
        team_meta = {}
        unique_teams = tracking_df[tracking_df['team_id'] != 'ball'][['team_id', 'team_name', 'team_side']].drop_duplicates()
        
        for _, row in unique_teams.iterrows():
            current_side = 'left' if row['team_id'] == left_team_id else 'right'
            
            team_meta[f'{current_side}_team_id'] = row['team_id']
            team_meta[f'{current_side}_team_name'] = row['team_name']
            team_meta[f'{current_side}_team_side'] = row['team_side']

        # -----------------------------------------------
        # 2. Assign In-Play Numbers (inplay_num)
        # -----------------------------------------------
        tracking_df['inplay_num'] = np.nan

        inplay_times = event_df[['inplay_num', 'event_time']].drop_duplicates().sort_values('inplay_num')
        inplay_periods = inplay_times.groupby('inplay_num')['event_time'].agg(['min', 'max']).reset_index()
        inplay_periods.columns = ['inplay_num', 'start_time', 'end_time']

        for period in tracking_df['period'].unique():
            p_inplay_periods = inplay_periods.copy()

            for _, row in p_inplay_periods.iterrows():
                current_inplay_num = row['inplay_num']
                start_time = row['start_time']
                end_time = row['end_time']
                
                mask_index = tracking_df[
                    (tracking_df['period'] == period) & 
                    (tracking_df['match_time'] >= start_time) & 
                    (tracking_df['match_time'] <= end_time)
                ].index
                
                tracking_df.loc[mask_index, 'inplay_num'] = current_inplay_num

        final_tracking_df = tracking_df.copy()

        # -----------------------------------------------
        # 3. Determine Player Ordering and Join Keys
        # -----------------------------------------------
        is_player = (final_tracking_df['player_id'] != 'ball')
        side_calculated = np.where(
            final_tracking_df['team_id'] == left_team_id,
            'left',
            'right'
        )
        side_series = pd.Series(side_calculated, index=final_tracking_df.index)
        if 'side' not in final_tracking_df.columns:
            final_tracking_df.loc[:, 'side'] = np.nan
        final_tracking_df['side'] = final_tracking_df['side'].astype(object)
        final_tracking_df.loc[is_player, 'side'] = side_series.loc[is_player]
        final_tracking_df.loc[final_tracking_df['player_id'] == 'ball', 'side'] = 'ball'

        pos_map = {pos: order for order, pos in enumerate(POSITION_ORDER, 1)}
        player_df = final_tracking_df[final_tracking_df['player_id'] != 'ball'].copy()
        player_df.loc[:, 'pos_order'] = player_df['position'].map(pos_map)
        player_df.loc[:, 'pos_rank'] = player_df.groupby(['frame', 'side'])['pos_order'].rank(method='first').astype(int)
        player_df.loc[:, 'variable'] = player_df['side'] + '_' + player_df['pos_rank'].astype(str)
        
        # -----------------------------------------------
        # 4. Transform Player Data to Wide Format (Pivot)
        # -----------------------------------------------
        value_cols = ['x', 'y', 'player_id', 'player_name', 'position']
        
        wide_data_list = []
        
        for col in value_cols:
            pivot_df = player_df.pivot_table(
                index=['frame', 'match_time', 'period', 'inplay_num'], 
                columns='variable', 
                values=col,
                aggfunc='first'
            ).add_suffix(f'_{col.replace("player_id", "id").replace("player_name", "name")}')

            wide_data_list.append(pivot_df)

        wide_player_df = wide_data_list[0].join(wide_data_list[1:])
        
        # -----------------------------------------------
        # 5. Extract and Merge Ball Data and Team Metadata
        # -----------------------------------------------
        ball_df = final_tracking_df[final_tracking_df['player_id'] == 'ball'][['frame', 'x', 'y', 'match_time', 'period', 'inplay_num']].rename(
            columns={'x': 'ball_x', 'y': 'ball_y'}
        ).set_index(['frame', 'match_time', 'period', 'inplay_num'])
        final_tracking_df = wide_player_df.join(ball_df).reset_index()
        
        for col, value in team_meta.items():
            final_tracking_df[col] = value

        # -----------------------------------------------
        # 6. Final Column Formatting and Reordering
        # -----------------------------------------------
        ordered_player_cols = []
        for side in ['left', 'right']:
            for i in range(1, 12):
                prefix = f'{side}_{i}_'
                
                ordered_player_cols.append(prefix + 'id')
                ordered_player_cols.append(prefix + 'name')
                ordered_player_cols.append(prefix + 'position')
                ordered_player_cols.append(prefix + 'x')
                ordered_player_cols.append(prefix + 'y')

        base_cols = ['period', 'inplay_num', 'frame', 'match_time', 'ball_x', 'ball_y']
        
        team_cols = []
        for side in ['left', 'right']:
            team_cols.extend([f'{side}_team_id', f'{side}_team_name', f'{side}_team_side'])
            
        final_cols = base_cols + team_cols + ordered_player_cols
        final_tracking_df = final_tracking_df.reindex(columns=final_cols)
        
        return final_tracking_df
    
    # Load the event data
    event_df = pd.read_csv(event_path)
    # devide by period
    grouped_events = event_df.groupby('event_period')
    PERIOD_ORDER = ['FIRST_HALF', 'SECOND_HALF', 'EXTRA_FIRST_HALF', 'EXTRA_SECOND_HALF']
    # check if the format is the latest version
    if tracking_xml_path is None:
        list_of_tracking_data = []
        for i in range(len(tracking_json_paths)):
            tracking_data = extract_tracking_data_from_json(tracking_json_paths[i], period=str(i+1))
            list_of_tracking_data.append(tracking_data)
        player_info_df = extract_meta_info_from_json(meta_data_path)
    else:
        tracking_data = extract_tracking_data_from_xml(tracking_xml_path)
        # add period
        list_of_tracking_data = devide_by_period(tracking_data)
        player_info_df = extract_meta_info_from_xml(meta_data_path)
    
    final_tracking_df_list = []
    for i in range(len(list_of_tracking_data)):
        event_df = grouped_events.get_group(PERIOD_ORDER[i])
        tracking_df = pd.DataFrame(list_of_tracking_data[i])
        # Get additional features
        event_df = get_inplay_start_time(event_df)
        # Get tracking features
        processed_tracking_df = get_tracking(tracking_df, event_df, player_info_df)
        final_tracking_df_list.append(processed_tracking_df)
    
    final_tracking_df = pd.concat(final_tracking_df_list, ignore_index=True)
    return final_tracking_df

def load_statsbomb_skillcorner(sb_event_path: str, sc_tracking_path: str, sc_match_path: str, sc_players_path: str) -> pd.DataFrame:
    """
    Load and merge StatsBomb event data with SkillCorner tracking data.

    Args:
        statsbomb_event_path (str): File path for StatsBomb event data.
        skillcorner_tracking_path (str): File path for SkillCorner tracking data.
        skillcorner_match_path (str): File path for SkillCorner match data.
        skillcorner_players_path (str): File path for SkillCorner players data.

    Returns:
        pd.DataFrame: Combined DataFrame with event and tracking data.
    """
    
    def extract_meta_info_from_match(sc_match: dict, sc_players: list) -> dict:
        """
        Extract team and player information (ID, name, side) from a json match data file.

        Args:
            sc_match (dict): Dataframe of match data file.
            sc_players (dict): List of players data file.
        Returns:
            dict: Dictionary in the format: {team_id: {'team_name': str, 'team_side': str}}, {player_id: {'position': str, 'team_id': str, 'side': str}}.
        """
        team_meta_df = {}
        player_meta_df = {}

        player_trackable_map = {p['id']: p.get('trackable_object') for p in sc_players}

        home_id = sc_match['home_team']['id']
        team_meta_df[home_id] = {
            'team_name': sc_match['home_team']['name'],
            'team_side': 'home'
        }

        away_id = sc_match['away_team']['id']
        team_meta_df[away_id] = {
            'team_name': sc_match['away_team']['name'],
            'team_side': 'away'
        }

        for p in sc_match['players']:
            player_id = p['id']
            trackable_id = player_trackable_map.get(player_id)
            player_meta_df[trackable_id] = {
                'team_id': p['team_id'],
                'player_name': p['short_name'],
                'position_name': p['player_role']['name'],
                'position_acronym': p['player_role']['acronym']
            }

        return team_meta_df, player_meta_df

    def get_left_team_id(sc_tracking, team_meta_df, player_meta_df):
        """
        Identifies which team ID is attacking the left side of the pitch based on the goalkeeper's position.
        
        Args:
            sc_tracking (list): Raw tracking data containing frame-by-frame object positions.
            team_meta_df (dict/pd.DataFrame): Metadata for teams including names and IDs.
            player_meta_df (dict/pd.DataFrame): Metadata for players including their team IDs and positions.
            
        Returns:
            int or None: The team ID assigned to the left side (x < 0), or None if not found.
        """
        all_team_ids = list(team_meta_df.keys())
        for frame_data in sc_tracking:
            if frame_data['data']==None:
                continue
            for obj in frame_data['data']:
                if 'z' in obj:
                    continue
                p_id = obj['trackable_object']
                p_info = player_meta_df[p_id]
                if p_info['position_acronym'] == 'GK':
                    if obj['x'] < 0.0:
                        left_team_id = p_info['team_id']
                    else:
                        left_team_id = [tid for tid in all_team_ids if tid != p_info['team_id']][0] 
                    return left_team_id
        return None

    def process_all_tracking(sc_tracking, team_meta_df, player_meta_df, left_team_id):
        """
        Iterates through all frames to return a flattened DataFrame sorted by tactical positions.
        
        Args:
            sc_tracking (list): Raw tracking data list.
            team_meta_df (dict): Metadata containing team details.
            player_meta_df (dict): Metadata containing player details.
            left_team_id (int): The ID of the team currently on the left side.
            
        Returns:
            pd.DataFrame: Processed tracking data with fixed columns for ball and 22 players (sorted by position).
        """
        POSITION_ORDER = ['GK', 'CB', 'RCB', 'LCB', 'RWB', 'RB', 'LWB', 'LB', 'CDM', 'RDM', 'LDM', 'RM', 'CM', 'LM', 'CAM', 'RW', 'LW', 'CF']
        pos_priority = {pos: i for i, pos in enumerate(POSITION_ORDER)}

        all_team_ids = list(team_meta_df.keys())
        right_team_id = [tid for tid in all_team_ids if tid != left_team_id][0]
        
        all_frames_processed = []

        for frame_data in sc_tracking:
            res = {
                'period': int(frame_data['period']) if pd.notna(frame_data['period']) else None,
                'inplay_num': None,
                'frame': frame_data['frame'],
                'match_time': frame_data['timestamp'],
                'ball_x': None,
                'ball_y': None,
                'left_team_id': left_team_id,
                'left_team_name': team_meta_df[left_team_id]['team_name'],
                'left_team_side': team_meta_df[left_team_id]['team_side'],
                'right_team_id': right_team_id,
                'right_team_name': team_meta_df[right_team_id]['team_name'],
                'right_team_side': team_meta_df[right_team_id]['team_side']
            }

            left_players_in_frame = []
            right_players_in_frame = []
            
            for obj in frame_data['data']:
                if 'z' in obj:
                    res['ball_x'] = obj['x']
                    res['ball_y'] = obj['y']
                    continue
                
                p_id = obj['track_id']
                if p_id in player_meta_df:
                    p_info = player_meta_df[p_id]
                    player_data = {
                        'id': p_id,
                        'name': p_info['player_name'],
                        'pos': p_info['position_acronym'],
                        'x': obj['x'],
                        'y': obj['y'],
                        'priority': pos_priority.get(p_info['position_acronym'], 99)
                    }
                    
                    if p_info['team_id'] == left_team_id:
                        left_players_in_frame.append(player_data)
                    else:
                        right_players_in_frame.append(player_data)

            left_players_sorted = sorted(left_players_in_frame, key=lambda x: (x['priority'], x['id']))
            right_players_sorted = sorted(right_players_in_frame, key=lambda x: (x['priority'], x['id']))

            for i in range(11):
                idx = i + 1
                if i < len(left_players_sorted):
                    p = left_players_sorted[i]
                    res[f"left_{idx}_id"] = p['id']
                    res[f"left_{idx}_name"] = p['name']
                    res[f"left_{idx}_position"] = p['pos']
                    res[f"left_{idx}_x"] = p['x']
                    res[f"left_{idx}_y"] = p['y']
                else:
                    res[f"left_{idx}_id"] = None
                    res[f"left_{idx}_name"] = None
                    res[f"left_{idx}_position"] = None
                    res[f"left_{idx}_x"] = None
                    res[f"left_{idx}_y"] = None

            for i in range(11):
                idx = i + 1
                if i < len(right_players_sorted):
                    p = right_players_sorted[i]
                    res[f"right_{idx}_id"] = p['id']
                    res[f"right_{idx}_name"] = p['name']
                    res[f"right_{idx}_position"] = p['pos']
                    res[f"right_{idx}_x"] = p['x']
                    res[f"right_{idx}_y"] = p['y']
                else:
                    res[f"right_{idx}_id"] = None
                    res[f"right_{idx}_name"] = None
                    res[f"right_{idx}_position"] = None
                    res[f"right_{idx}_x"] = None
                    res[f"right_{idx}_y"] = None

            all_frames_processed.append(res)

        return pd.DataFrame(all_frames_processed)
    
    def get_inplay_start_time(event_df: pd.DataFrame):
        """
        Assigns in-play sequence numbers and identifies start times for each sequence from event data.
        
        Args:
            event_df (pd.DataFrame): Dataframe of match events (passes, play patterns, etc.).
            
        Returns:
            list: A list of dictionaries, each containing 'inplay_num', 'period', and 'timestamp' for sequence starts.
        """
        df = event_df.copy()
        inplay_info_list = []
        current_inplay = 0

        continuing_patterns = ['Regular Play', 'From Counter', 'From Keeper']
        restart_types = ['Throw-in', 'Corner', 'Goal Kick', 'Free Kick']

        for i in range(len(df) - 1):
            curr_ev = df.iloc[i]
            next_ev = df.iloc[i + 1]

            if pd.isna(next_ev['pass_type']):
                continue
            
            is_new_inplay = False

            next_ts = pd.Timestamp(next_ev['timestamp']).round('100ms')
            curr_ts = pd.Timestamp(curr_ev['timestamp']).round('100ms')
            if next_ts < curr_ts:
                is_new_inplay = True

            elif curr_ev['play_pattern'] != next_ev['play_pattern']:
                if next_ev['play_pattern'] not in continuing_patterns:
                    is_new_inplay = True

            elif next_ev['pass_type'] in restart_types:
                is_new_inplay = True

            if is_new_inplay:
                current_inplay += 1
                inplay_info_list.append({
                    'inplay_num': current_inplay,
                    'period': int(next_ev['period']),
                    'timestamp': next_ts
                })

        return inplay_info_list
    
    def get_inplay_tracking(tracking_df: pd.DataFrame, inplay_info_list: List) -> pd.DataFrame:
        """
        Filters tracking data to include only in-play periods and assigns sequence numbers.
        
        Args:
            tracking_df (pd.DataFrame): Processed tracking data.
            inplay_info_list (list): List of dictionaries defining start times of in-play sequences.
            
        Returns:
            pd.DataFrame: Tracking data filtered for in-play time, sampled at 5fps (200ms intervals).
        """
        df = tracking_df.copy()

        df['tmp_timestamp'] = pd.to_datetime(
            df['match_time'], format='%H:%M:%S.%f', errors='coerce'
        ).map(lambda x: x.replace(year=1900, month=1, day=1) if pd.notna(x) else x)

        def normalize_period_time(group):
            period_start = group['tmp_timestamp'].min()
            base = pd.Timestamp('1900-01-01 00:00:00')
            group['tmp_timestamp'] = base + (group['tmp_timestamp'] - period_start)
            return group

        def normalize_time(ts):
            if isinstance(ts, pd.Timestamp):
                return ts.replace(year=1900, month=1, day=1)
            return ts
        
        df = df.groupby('period', group_keys=False).apply(normalize_period_time)

        for i in range(len(inplay_info_list)):
            current_info = inplay_info_list[i]
            
            start_time = normalize_time(current_info['timestamp'])
            period = current_info['period']
            num = current_info['inplay_num']

            period_mask = (df['period'] == period)

            next_event_in_same_period = None
            for j in range(i + 1, len(inplay_info_list)):
                if int(inplay_info_list[j]['period']) == period:
                    next_event_in_same_period = normalize_time(inplay_info_list[j]['timestamp'])
                    break
            
            if next_event_in_same_period is not None:
                time_mask = (df['tmp_timestamp'] >= start_time) & (df['tmp_timestamp'] < next_event_in_same_period)
            else:
                time_mask = (df['tmp_timestamp'] >= start_time)

            final_mask = period_mask & time_mask
            df.loc[final_mask, 'inplay_num'] = num

        df = df.dropna(subset=['inplay_num'])

        base_time = pd.Timestamp('1900-01-01 00:00:00')
        df['match_time'] = (df['tmp_timestamp'] - base_time).dt.total_seconds() * 1000
        df = df[df['match_time'] % 200 == 0]
        df['match_time'] = df['match_time'].astype(int)
        df = df.drop(columns=['tmp_timestamp'])

        df['period'] = df['period'].astype(int)
        df['inplay_num'] = df['inplay_num'].astype(int)

        return df.reset_index(drop=True)
    
    # Load the event data
    with open(sb_event_path, 'rb') as f:
        sb_event = pickle.load(f)
    with open(sc_tracking_path, 'r', encoding='utf-8') as f:
        sc_tracking = json.load(f)
    with open(sc_match_path, 'r', encoding='utf-8') as f:
        sc_match = json.load(f)
    with open(sc_players_path, 'r', encoding='utf-8') as f:
        sc_players = json.load(f)

    team_meta_df, player_meta_df = extract_meta_info_from_match(sc_match, sc_players)

    left_team_id = get_left_team_id(sc_tracking, team_meta_df, player_meta_df)

    tracking_df = process_all_tracking(sc_tracking, team_meta_df, player_meta_df, left_team_id)

    inplay_info_list = get_inplay_start_time(sb_event)

    processed_tracking_df = get_inplay_tracking(tracking_df, inplay_info_list)

    return processed_tracking_df
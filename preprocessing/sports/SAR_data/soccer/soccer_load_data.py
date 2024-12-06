#Target data provider [Statsbomb_Skillcorner, Datastadium]

import json
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from statsbombpy import sb
from tqdm import tqdm
from datetime import datetime
import os
import pdb


def load_statsbomb_skillcorner(statsbomb_event_dir: str, statsbomb_lineup_dir, skillcorner_tracking_dir: str, skillcorner_match_dir: str, statsbomb_match_id: str, skillcorner_match_id: str) -> pd.DataFrame:
    """
    Load and merge StatsBomb event data with SkillCorner tracking data.

    Args:
        statsbomb_event_dir (str): Directory path for StatsBomb event data.
        skillcorner_tracking_dir (str): Directory path for SkillCorner tracking data.
        skillcorner_match_dir (str): Directory path for SkillCorner match data.
        statsbomb_match_id (str): Match ID for StatsBomb data.
        skillcorner_match_id (str): Match ID for SkillCorner data.

    Returns:
        pd.DataFrame: Combined DataFrame with event and tracking data.
    """
    
    # File paths
    statsbomb_event_path = f"{statsbomb_event_dir}/{statsbomb_match_id}.csv"
    statsbomb_lineup_path = f"{statsbomb_lineup_dir}/{statsbomb_match_id}.json"
    skillcorner_tracking_path = f"{skillcorner_tracking_dir}/{skillcorner_match_id}.json"
    skillcorner_match_path = f"{skillcorner_match_dir}/{skillcorner_match_id}.json"

    # Load StatsBomb events
    events = pd.read_csv(statsbomb_event_path)

    # Load StatsBomb lineup data
    with open(statsbomb_lineup_path) as f:
        lineup = json.load(f)
    
    # Load SkillCorner tracking and match data
    with open(skillcorner_tracking_path) as f:
        tracking = json.load(f)
        
    with open(skillcorner_match_path) as f:
        match = json.load(f)

    #check if the file exists
    if not os.path.exists(statsbomb_event_path):
        print(f"Statsbomb event file not found: {statsbomb_event_path}")
        return None
    if not os.path.exists(skillcorner_tracking_path):
        print(f"Skillcorner tracking file not found: {skillcorner_tracking_path}")
        return None
    if not os.path.exists(skillcorner_match_path):
        print(f"Skillcorner match file not found: {skillcorner_match_path}")
        return None

    # Team name mapping
    team_name_dict = {
        'UD Almería': 'Almería', 'Real Sociedad': 'Real Sociedad', 'Athletic Club de Bilbao': 'Athletic Club', 
        'Villarreal CF': 'Villarreal', 'RC Celta de Vigo': 'Celta Vigo', 'Getafe CF': 'Getafe', 
        'UD Las Palmas': 'Las Palmas', 'Sevilla FC': 'Sevilla', 'Cadiz CF': 'Cádiz', 
        'Atlético Madrid': 'Atlético Madrid', 'RCD Mallorca': 'Mallorca', 'Valencia CF': 'Valencia', 
        'CA Osasuna': 'Osasuna', 'Girona FC': 'Girona', 'Real Betis Balompié': 'Real Betis', 
        'FC Barcelona': 'Barcelona', 'Deportivo Alavés': 'Deportivo Alavés', 'Granada CF': 'Granada', 
        'Rayo Vallecano': 'Rayo Vallecano', 'Real Madrid CF': 'Real Madrid'
    }
    
    home_team_name = team_name_dict[match['home_team']['name']]
    away_team_name = team_name_dict[match['away_team']['name']]
    
    team_dict = {
        match['home_team']['id']: {'role': 'home', 'name': home_team_name},
        match['away_team']['id']: {'role': 'away', 'name': away_team_name}
    }

    metadata_df = pd.DataFrame(
        columns=['match_id', 'home_team', 'away_team', 'home_team_id', 'away_team_id'],
    )

    # Prepare metadata
    metadata_df.loc[0] = [skillcorner_match_id, home_team_name, away_team_name, match['home_team']['id'], match['away_team']['id']]

    # Convert the trackable object dict
    trackable_objects = {}
    home_count = away_count = 0


    def extract_height(lineup, team, number):
        team_lineup = lineup[team]
        for player in team_lineup:
            if player['jersey_number'] == number:
                return player['player_height']
        return None

    
    for player in match['players']:
        role = team_dict[player['team_id']]['role']
        position = player['player_role']['name']
        position_group = player['player_role']['position_group']
        team_name = team_dict[player['team_id']]['name']
        height = extract_height(lineup, team_name, player['number'])
        if role == 'home':
            trackable_objects[player['trackable_object']] = {
                'name': f"{player['first_name']} {player['last_name']}",
                'team': team_dict[player['team_id']]['name'],
                'team_id': player['team_id'],
                'role': role,
                'id': home_count,
                'position': position,
                'position_group': position_group,
                'start_time': player['start_time'],
                'end_time': player['end_time'],
                'jersey_number': player['number'],
                'player_id': player['id'],
                'height': height
            }
            home_count += 1
        elif role == 'away':
            trackable_objects[player['trackable_object']] = {
                'name': f"{player['first_name']} {player['last_name']}",
                'team': team_dict[player['team_id']]['name'],
                'team_id': player['team_id'],
                'role': role,
                'id': away_count,
                'position': position,
                'position_group': position_group,
                'start_time': player['start_time'],
                'end_time': player['end_time'],
                'jersey_number': player['number'],
                'player_id': player['id'],
                'height': height
            }
            away_count += 1


    trackable_objects[match['ball']['trackable_object']] = {'name': 'ball', 'team': 'ball', 'role': 'ball', 'position': 'ball'}
    ball_id = match['ball']['trackable_object']

    ##sync the tracking data with the events based on the ball velocity
    #get the first 5s of the match
    ball_velocity_period_1 = []
    ball_velocity_period_2 = []

    for frame in tracking:
        time = frame['timestamp']
        period = frame['period']
        data = frame['data']
        time_components = time.split(':') if time else None
        seconds = float(time_components[0]) * 3600 + float(time_components[1]) * 60 + float(time_components[2]) if time else 0
        if time and period==1 and seconds<=5:
            for obj in data:
                if obj['trackable_object']==ball_id:
                    ball_velocity_period_1.append([time, obj['x'], obj['y'],obj['z']])

        if time and period==2 and seconds <= 45*60+5:
            for obj in data:
                if obj['trackable_object']==ball_id:
                    ball_velocity_period_2.append([time, obj['x'], obj['y'],obj['z']])
            
    if not ball_velocity_period_1 == [] or not ball_velocity_period_2 == []:
        try:
            max_velocity_timestamp1, max_velocity1 = calculate_velocity_and_max_timestamp(ball_velocity_period_1)
            max_velocity_seconds1 = max_velocity_timestamp1.split(':')
            max_velocity_seconds1 = float(max_velocity_seconds1[0]) * 3600 + float(max_velocity_seconds1[1]) * 60 + float(max_velocity_seconds1[2])
        except:
            max_velocity_seconds1 = -1
        
        try:
            max_velocity_timestamp2, max_velocity2 = calculate_velocity_and_max_timestamp(ball_velocity_period_2)
            max_velocity_seconds2 = max_velocity_timestamp2.split(':')
            max_velocity_seconds2 = float(max_velocity_seconds2[0]) * 3600 + float(max_velocity_seconds2[1]) * 60 + float(max_velocity_seconds2[2])
            max_velocity_seconds2 = max_velocity_seconds2 - 45*60
        except:
            max_velocity_seconds2 = -1
        
        if max_velocity_seconds1 == -1 and max_velocity_seconds2 != -1:
            max_velocity_seconds1 = max_velocity_seconds2
        elif max_velocity_seconds1 != -1 and max_velocity_seconds2 == -1:
            max_velocity_seconds2 = max_velocity_seconds1
        elif max_velocity_seconds1 == -1 and max_velocity_seconds2 == -1:
            max_velocity_seconds1 = max_velocity_seconds2 = 0
    
    # Process tracking data
    tracking_dict = {}
    for frame in tracking:
        time = frame['timestamp']
        if time:
            time_components = time.split(':')
            seconds = float(time_components[0]) * 3600 + float(time_components[1]) * 60 + float(time_components[2])
            period = frame['period']
            if period == 1:
                seconds = seconds - max_velocity_seconds1
            elif period == 2:
                seconds = seconds - max_velocity_seconds2
            seconds = round(seconds, 1)
            uid = f"{period}_{seconds}"
            tracking_dict[uid] = frame['data']
    
    # Prepare data for DataFrame
    df_list = []
    for _, event in events.iterrows():
        event_id = event['id']
        match_id = skillcorner_match_id
        period = event['period']
        time = event['timestamp']
        minute = event['minute']
        second = event['second']
        event_type = event['type']
        event_type_2 = None
        index = event['index']
        possession = event['possession']
        tactics = event['tactics']
        tactics = ast.literal_eval(tactics) if isinstance(tactics, str) else tactics
        tactics = tactics.to_dict() if hasattr(tactics, 'to_dict') else tactics
        formation = tactics['formation'] if not pd.isna(tactics) else np.nan

        end_x = end_y = None
        outcome = None
        if event_type == "Pass":
            end_location=event.get('pass_end_location')
            #check if end_location is a string
            if isinstance(end_location, (str)):
                end_location = [float(x) for x in end_location[1:-1].split(",")]
                end_x = end_location[0]
                end_y = end_location[1]
            cross=event.get('pass_cross')
            pass_height=event.get('pass_height')
            pass_type=event.get('pass_type')
            if pass_type:
                if pass_type != "Wayward":
                    event_type_2=pass_type
            elif cross and not np.isnan(cross):
                event_type_2="Cross"
            elif pass_height:
                event_type_2=pass_height
            outcome = event.get('pass_outcome')
        elif event_type=="Shot":
            outcome = event.get('shot_outcome')
        elif event_type == "Dribble":
            outcome = event.get('dribble_outcome')
        team = event['team']
        home_team = 1 if team == home_team_name else 0
        player = event['player']
        player_id = event['player_id']
        jersey_number = np.nan
        for ply_info in lineup[team]:
            if ply_info['player_id'] == player_id:
                jersey_number = ply_info['jersey_number']
                break
        location = event['location']

        if isinstance(location, str):
            location = [float(x) for x in location[1:-1].split(",")]
            start_x, start_y = location[0], location[1]
        else:
            start_x = start_y = None

        time_components = time.split(':')
        seconds = round(float(time_components[0]) * 3600 + float(time_components[1]) * 60 + float(time_components[2]), 4)
        if period == 2:
            seconds += 45 * 60
        elif period == 3:
            seconds += 90 * 60
        elif period == 4:
            seconds += (90 + 15) * 60

        seconds_rounded = round(seconds, 1)
        uid = f"{period}_{seconds_rounded}"
        tracking_data = tracking_dict.get(uid)
        home_tracking = [None] * 2 * 23
        away_tracking = [None] * 2 * 23
        ball_tracking = [None] * 2
        home_side = [None]
        
        if tracking_data:
            for obj in tracking_data:
                track_obj = trackable_objects[obj['trackable_object']]
                if track_obj['role'] == 'home':
                    home_tracking[2 * track_obj['id']] = obj['x']
                    home_tracking[2 * track_obj['id'] + 1] = obj['y']
                elif track_obj['role'] == 'away':
                    away_tracking[2 * track_obj['id']] = obj['x']
                    away_tracking[2 * track_obj['id'] + 1] = obj['y']
                elif track_obj['role'] == 'ball':
                    ball_tracking[0] = obj['x']
                    ball_tracking[1] = obj['y']

                if track_obj['position'] == "Goalkeeper":
                    if track_obj['role'] == 'home':
                        home_gk_x = obj['x']
                    elif track_obj['role'] == 'away':
                        away_gk_x = obj['x']

                
            # Determine the side of the home team based on the goalkeeper's position
            if home_gk_x < away_gk_x:
                home_side = 'left'
            else:
                home_side = 'right'
            
            home_side = [home_side]

        df_list.append([match_id, period, time, minute, second, seconds, event_type, event_type_2, outcome, team, home_team, player, jersey_number, start_x, start_y, end_x, end_y, index, possession, formation, *home_tracking, *away_tracking, *ball_tracking, *home_side])
    # Define DataFrame columns
    home_tracking_columns = []
    away_tracking_columns = []
    player_idxs = []
    ball_tracking = ["ball_x", "ball_y"]
    for i in range(1, 24):
        home_tracking_columns.extend([f"h{i}_x", f"h{i}_y"])
        away_tracking_columns.extend([f"a{i}_x", f"a{i}_y"])
        player_idxs.extend([f"h{i}", f"a{i}"])
    columns = ["match_id", "period", "time", "minute", "second", 'seconds', "event_type", "event_type_2", "outcome", "team", "home_team", "player", "jersey_number", "start_x", "start_y","end_x","end_y" ,"index", "possession", "formation",] + home_tracking_columns + away_tracking_columns + ball_tracking + ["home_side"]

    # Convert the event list to a DataFrame
    df = pd.DataFrame(df_list, columns=columns)
    players_columns = ["name", "team", "team_id", "trackable_object", "role", "id", "position", "position_group", "start_time", "end_time", "jersey_number", "player_id", "height"]    
    df_players = pd.DataFrame(columns=players_columns, index=player_idxs)
    for trackable_object, player in trackable_objects.items():
        if player['role'] != 'ball':
            id = player['id'] + 1
            player_data = [trackable_object if col == 'trackable_object' else player[col] for col in players_columns]
            if player['role'] == 'home':
                df_players.loc[f"h{id}"] = player_data
            elif player['role'] == 'away':
                df_players.loc[f"a{id}"] = player_data

    # change index name of df_players to player_index
    df_players.index.name = "player_index"

    #Sort the DataFrame by 'period' then 'seconds'
    df = df.sort_values(by=["period", "seconds"]).reset_index(drop=True)

    return df, df_players, metadata_df

def calculate_velocity_and_max_timestamp(data):
    """
    Calculate the velocity for each time interval and find the timestamp with the highest velocity.

    Parameters:
    data (list): List of lists, where each sublist contains [timestamp, x, y, z].

    Returns:
    tuple: (max_velocity_timestamp, max_velocity)
        - max_velocity_timestamp: The timestamp with the highest velocity.
        - max_velocity: The highest velocity value.
    """
    # Extract timestamps, x, y, z coordinates
    timestamps = [entry[0] for entry in data]
    x = np.array([entry[1] for entry in data])
    y = np.array([entry[2] for entry in data])
    z = np.array([entry[3] for entry in data])

    # Convert timestamps to seconds
    time_seconds = np.array([
        (datetime.strptime(ts, "%H:%M:%S.%f") - datetime.strptime(timestamps[0], "%H:%M:%S.%f")).total_seconds()
        for ts in timestamps
    ])

    # Calculate differences
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    delta_z = np.diff(z)
    delta_t = np.diff(time_seconds)

    # Calculate velocity components and magnitude
    vx = delta_x / delta_t
    vy = delta_y / delta_t
    vz = delta_z / delta_t
    velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

    # Find the index of the maximum velocity
    max_velocity_index = np.argmax(velocity_magnitude)
    max_velocity = velocity_magnitude[max_velocity_index]
    max_velocity_timestamp = timestamps[max_velocity_index + 1]  # Use +1 to get the ending timestamp of the interval

    return max_velocity_timestamp, max_velocity


if __name__ == "__main__":
    import pdb
    import os
    #cd to ../PreProcessing

    statsbomb_skillcorner_event_path="/data_pool_1/laliga_23/statsbomb/events"
    statsbomb_skillcorner_lineup_path="/data_pool_1/laliga_23/statsbomb/lineups"
    statsbomb_skillcorner_tracking_path="/data_pool_1/laliga_23/skillcorner/tracking"
    statsbomb_skillcorner_match_path="/data_pool_1/laliga_23/skillcorner/match"
    match_id_dict=os.getcwd()+"/SAR_data/match_id_dict.json"
    
    datastadium_event_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/play.csv"
    datastadium_home_tracking_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/home_tracking.csv"
    datastadium_away_tracking_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/away_tracking.csv"

    # load match_id_dict
    with open(match_id_dict) as f:
        match_id_dict = json.load(f)

    for dir in os.scandir(statsbomb_skillcorner_tracking_path):
        dir_name = dir.name[:7]
        skillcorner_match_id = dir_name
        statsbomb_match_id = match_id_dict[skillcorner_match_id]
        skillcorner_match_id = int(skillcorner_match_id)
        statsbomb_match_id = int(statsbomb_match_id)

        #test load_statsbomb_skillcorner
        statsbomb_skillcorner_df, players_df, metadata=load_statsbomb_skillcorner(statsbomb_skillcorner_event_path,statsbomb_skillcorner_lineup_path,statsbomb_skillcorner_tracking_path,
                                                            statsbomb_skillcorner_match_path,statsbomb_match_id,skillcorner_match_id)
        output_dir=os.getcwd()+f"/test/sports/event_data/data/statsbomb_skillcorner/{skillcorner_match_id}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        statsbomb_skillcorner_df.to_csv(os.path.join(output_dir, "event_360_data.csv"),index=False)
        players_df.to_csv(os.path.join(output_dir, "players_info.csv"),index=True)
        metadata.to_csv(os.path.join(output_dir, "metadata.csv"),index=False)



    print("----------------done-----------------")
    pdb.set_trace()
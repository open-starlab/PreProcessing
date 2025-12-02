#Target data provider [Metrica,Robocup 2D simulation,Statsbomb,Wyscout,Opta data,DataFactory,sportec]

import json
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from statsbombpy import sb
from tqdm import tqdm
from datetime import datetime
import os
import pdb
import csv

def load_bepro(tracking_xml_path: str, tracking_json_paths: list, event_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Loads and processes event and tracking data from soccer match recordings.

    This function combines event data with tracking data by merging based on event time. It also adds 
    additional features extracted from metadata, such as player information, and converts position 
    coordinates to the correct scale for analysis.

    Args:
        event_path (str): Path to the CSV file containing event data.
        tracking_path (str): Path to the XML file containing tracking data.
        meta_path (str): Path to the XML file containing match metadata (pitch, teams, players, etc.).
        verbose (bool, optional): If True, prints additional information about the merging process and 
                                  feature extraction. Default is False.

    Returns:
        pd.DataFrame: A DataFrame containing the merged and processed event and tracking data, 
                      with additional features including player positions, speeds, ball position, 
                      and metadata (e.g., player names, shirt numbers, positions).
    """
    def extract_tracking_data_from_xml(xml_path):
        """
        Parse the XML file and extract tracking data.

        Args:
            xml_path (str): Path to the XML file.
        Returns:
            list of dict: A list containing tracking information for each player in each frame.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tracking_data = []

        for frame in root.findall("frame"):
            frame_number = int(frame.get("frameNumber"))
            match_time = int(frame.get("matchTime"))
            
            for player in frame:
                player_id = player.get("playerId")
                loc = player.get("loc")
                # Convert loc string to float coordinates
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
        tracking_df = add_period_column(tracking_data)

        return tracking_df

    def extract_tracking_data_from_json(json_path):
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
                        "frame": int(frame_number),
                        "match_time": int(player.get("match_time", 0)),
                        "player_id": "ball" if player.get("player_id") == None else player.get("player_id"),
                        "x": "{:.2f}".format(float(player.get("x", 0) - 52.5)),
                        "y": "{:.2f}".format(float(player.get("y", 0) - 34.0))
                    })
                except ValueError:
                    raise ValueError(f"Invalid data format in frame {frame_number}")
        tracking_df = add_period_column(tracking_data)

        return tracking_df
    
    def add_period_column(tracking_data_list):
        """
        Add a 'period' column to the tracking_data list.

        Increment the period each time the frame number significantly decreases (resets).
        Args:
            tracking_data_list (list of dict): A list containing tracking_data.
        Returns:
            pandas.DataFrame: A DataFrame with the 'period' column added.
        """

        df = pd.DataFrame(tracking_data_list)
        first_occurrence_of_frame = df.drop_duplicates(subset=['frame'], keep='first')
        frame_diff = first_occurrence_of_frame['frame'].diff().fillna(0)
        period_reset = (frame_diff < 0)
        period_values = period_reset.cumsum() + 1
        period_map = pd.Series(period_values.values, index=first_occurrence_of_frame['frame']).to_dict()
        df['period'] = df['frame'].map(period_map)
        cols = ['period'] + [col for col in df.columns if col != 'period']
        df = df[cols]

        return df

    def get_additional_features(event_df, meta_data):
        #player info: id name nameEN shirtNumber position
        # create features period, seconds, event_type, event_type_2, outcome, home_team, x_unscaled, y_unscaled,
        period_dict = {"FIRST_HALF": 1, "SECOND_HALF": 2, "EXTRA_FIRST_HALF": 3, "EXTRA_SECOND_HALF": 4}
        event_df["period"] = event_df["event_period"].map(period_dict)
        event_df["seconds"] = event_df["event_time"]/1000

        event_type_list = []
        for i in range(len(event_df)):
            event_i = event_df.iloc[i].event_types
            # print(event_i)
            if not isinstance(event_i, str):
                event_type_list.append(None)
            else:
                event_i = event_i.split(" ")[0]
                event_type_list.append(event_i)
        event_df["event_type"] = event_type_list

        home_team_dict = {int(team_info["id"]):team_info["side"] for team_info in meta_data["team_info"]}
        event_df["home_team"] = event_df["team_id"].map(home_team_dict)
        #convert "home" to 1 and "away" to 0 for home_team
        event_df["home_team"] = event_df["home_team"].map({"home":1,"away":0})

        #x and y coordinates of the field (height,width) for the event data (inverse of the tracking data)
        event_df["x_unscaled"] = event_df["y"]*int(meta_data["pitch_info"]["width"])
        event_df["y_unscaled"] = event_df["x"]*int(meta_data["pitch_info"]["height"])
        return event_df
    
    def calculate_sync_bias(event_df, tracking_data, period=1, verbose=False):
        # 'FIRST_HALF' "SECOND_HALF"
        # Calculate the bias between event time and tracking time
        limit = 5.0 #seconds
        time_list = [key for key in tracking_data.keys()]
        #split the time_list into two halves
        if period == 1:
            time_list = [time for time in time_list if tracking_data[time]['eventPeriod'] == 'FIRST_HALF']
            first_event_time = event_df[event_df["event_period"]=="FIRST_HALF"].iloc[0].event_time if "FIRST_HALF" in event_df["event_period"].values else 0
        elif period == 2:
            time_list = [time for time in time_list if tracking_data[time]['eventPeriod'] == 'SECOND_HALF']
            first_event_time = event_df[event_df["event_period"]=="SECOND_HALF"].iloc[0].event_time if "SECOND_HALF" in event_df["event_period"].values else 0
        elif period == 3:
            time_list = [time for time in time_list if tracking_data[time]['eventPeriod'] == 'EXTRA_FIRST_HALF']
            first_event_time = event_df[event_df["event_period"]=="EXTRA_FIRST_HALF"].iloc[0].event_time if "EXTRA_FIRST_HALF" in event_df["event_period"].values else 0
        
        if time_list == []:
            return 0

        time_list.sort()
        start_time = max(time_list[0],0)
        #round to the nearest 1000
        start_time = round(start_time/1000)*1000
        print("start_time:",start_time) if verbose else None
        #drop the time that exceeds the limit of the event time
        time_list = [time for time in time_list if time <= start_time+limit*1000]
        #order the time_list in ascending order
        time_list.sort()
        
        ball_coordinates = []
        for time_i in time_list:
            tracking_data_i = tracking_data[time_i]
            ball_data_i = tracking_data_i['ball']['loc']
            ball_coordinates.append(ball_data_i)
        #find the time with the highest acceleration
        ball_coordinates = np.array(ball_coordinates)
        ball_speed = np.linalg.norm(np.diff(ball_coordinates,axis=0),axis=1)
        max_speed_index = np.argmax(ball_speed)
        max_speed_time = time_list[max_speed_index]
        bias = max_speed_time - first_event_time

        return bias

    def get_tracking_features(event_df, tracking_data, meta_data, verbose=True):
        # combine the event data with the tracking data via event_time and matchTime
        #get the player info
        time_list = [key for key in tracking_data.keys()]
        time_diff_list = []
        player_dict = {}
        home_team_player_count = 0
        away_team_player_count = 0
        home_team_dict = {int(team_info["id"]):team_info["side"] for team_info in meta_data["team_info"]}
        for player_i in meta_data["player_info"]:
            player_dict[player_i["id"]] = player_i
            team_id = int(player_dict[player_i["id"]]['teamId'])
            if home_team_dict[team_id] == 'home':
                player_dict[player_i["id"]]["player_num"] = home_team_player_count+1
                home_team_player_count += 1
            elif home_team_dict[team_id] == 'away':
                player_dict[player_i["id"]]["player_num"] = away_team_player_count+1
                away_team_player_count += 1
            else:
                print("team_id not found")
                pdb.set_trace()
        
        #create the additional features
        tracking_features=["player_id","x","y","speed"]
        meta_features=["name","nameEn","shirtNumber","position"]
        ball_features = ["ball_x","ball_y","ball_speed"]
        additional_features = tracking_features+meta_features
        additional_featurs_dict = {}
        for i in range(home_team_player_count):
            for j in range(len(additional_features)):
                additional_featurs_dict[f"home_{additional_features[j]}_{i+1}"] = []
        for i in range(away_team_player_count):
            for j in range(len(additional_features)):
                additional_featurs_dict[f"away_{additional_features[j]}_{i+1}"] = []
        for j in range(len(ball_features)):
            additional_featurs_dict[ball_features[j]] = []
        
        additional_featurs_dict["tracking_time"] = []
        additaional_features_dict_key_list = [key for key in additional_featurs_dict.keys()]

        #get the sync bias for the event and tracking data
        bias_1 = calculate_sync_bias(event_df, tracking_data, period=1, verbose=verbose) #FIRST_HALF
        bias_2 = calculate_sync_bias(event_df, tracking_data, period=2, verbose=verbose) #SECOND_HALF
        bias_3 = calculate_sync_bias(event_df, tracking_data, period=3, verbose=verbose) #EXTRA_FIRST_HALF

        print("bias_1:",bias_1,"bias_2:",bias_2,"bias_3:",bias_3) if verbose else None

        if verbose:
            iterable = tqdm(range(len(event_df)))
        else:
            iterable = range(len(event_df))
        for i in iterable:
            updated_features = []
            event_time = event_df.iloc[i].event_time
            period = event_df.iloc[i].event_period
            if period == 'FIRST_HALF':
                event_time += bias_1
            elif period == 'SECOND_HALF':
                event_time += bias_2
            elif period == 'EXTRA_FIRST_HALF':
                event_time += bias_3
            else:
                print("period not included")
            #find the nearest time in the tracking data
            nearest_time = min(time_list, key=lambda x:abs(x-event_time))
            try:
                additional_featurs_dict["tracking_time"].append(nearest_time)
                updated_features+=["tracking_time"]
            except:
                pass
            time_diff_list.append(nearest_time-event_time)
            #get the tracking data
            tracking_data_i = tracking_data[nearest_time]
            for player_track_j in tracking_data_i['players']:
                player_j_id = player_track_j['playerId']
                player_j_num = player_dict[player_j_id]["player_num"]
                player_j_team = player_dict[player_j_id]["teamId"]
                player_j_home = home_team_dict[int(player_j_team)]
                # append the tracking data and meta data to the additional features
                additional_featurs_dict[f"{player_j_home}_player_id_{player_j_num}"].append(player_track_j['playerId'])
                additional_featurs_dict[f"{player_j_home}_x_{player_j_num}"].append(round(player_track_j['loc'][0]*int(meta_data["pitch_info"]["width"]),2))
                additional_featurs_dict[f"{player_j_home}_y_{player_j_num}"].append(round(player_track_j['loc'][1]*int(meta_data["pitch_info"]["height"]),2))
                additional_featurs_dict[f"{player_j_home}_speed_{player_j_num}"].append(player_track_j['speed'])
                additional_featurs_dict[f"{player_j_home}_name_{player_j_num}"].append(player_dict[player_j_id]["name"])
                additional_featurs_dict[f"{player_j_home}_nameEn_{player_j_num}"].append(player_dict[player_j_id]["nameEn"])
                additional_featurs_dict[f"{player_j_home}_shirtNumber_{player_j_num}"].append(player_dict[player_j_id]["shirtNumber"])
                additional_featurs_dict[f"{player_j_home}_position_{player_j_num}"].append(player_dict[player_j_id]["position"])
                updated_features+=[f"{player_j_home}_player_id_{player_j_num}",f"{player_j_home}_x_{player_j_num}",f"{player_j_home}_y_{player_j_num}",f"{player_j_home}_speed_{player_j_num}",f"{player_j_home}_name_{player_j_num}",f"{player_j_home}_nameEn_{player_j_num}",f"{player_j_home}_shirtNumber_{player_j_num}",f"{player_j_home}_position_{player_j_num}"]
            ball_track = tracking_data_i['ball']
            additional_featurs_dict[f"ball_x"].append(round(ball_track['loc'][0]*int(meta_data["pitch_info"]["width"]),2))
            additional_featurs_dict[f"ball_y"].append(round(ball_track['loc'][1]*int(meta_data["pitch_info"]["height"]),2))
            if ball_track['speed'] == 'NA':
                additional_featurs_dict[f"ball_speed"].append(None)
            else:
                additional_featurs_dict[f"ball_speed"].append(ball_track['speed'])
            updated_features+=["ball_x","ball_y","ball_speed"]
            # for features in additaional_features_dict_key_list but not in updated_features, append None
            for key in additaional_features_dict_key_list:
                if key not in updated_features:
                    additional_featurs_dict[key].append(None)
        
        #add the additional features to the event_df
        out_event_df = event_df.copy()
        if verbose:
            for key in additional_featurs_dict.keys():
                print(key,len(additional_featurs_dict[key]))

        # Create a DataFrame from the additional features dictionary
        additional_features_df = pd.DataFrame(additional_featurs_dict)

        # Concatenate the original event_df with the additional features DataFrame
        out_event_df = pd.concat([event_df, additional_features_df], axis=1)

        #print the mean and std of the time_diff_list
        if verbose:
            print("mean time difference:",round(np.mean(time_diff_list),4))
            print("std time difference:",round(np.std(time_diff_list),4))
            print("max time difference:",round(np.max(time_diff_list),4))
            print("min time difference:",round(np.min(time_diff_list),4))
        return out_event_df
    
    # check if the format is the latest version
    if tracking_xml_path is None:
        list_of_tracking_dfs = []
        for i in range(len(tracking_json_paths)):
            input_json = tracking_json_paths[i]
            tracking_df = extract_tracking_data_from_json(input_json)
            list_of_tracking_dfs.append(tracking_df)
        tracking_df = pd.concat(list_of_tracking_dfs, ignore_index=True)
    else:
        input_json = tracking_xml_path
        tracking_df = extract_tracking_data_from_xml(input_json)
    # Load the event data
    event_df = pd.read_csv(event_path)
    # Get additional features
    event_df = get_additional_features(event_df)
    # Get tracking features
    event_df = get_tracking_features(event_df, tracking_df, verbose=verbose)

    return event_df

def load_statsbomb_skillcorner(statsbomb_event_dir: str, skillcorner_tracking_dir: str, skillcorner_match_dir: str, statsbomb_match_id: str, skillcorner_match_id: str) -> pd.DataFrame:
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
    skillcorner_tracking_path = f"{skillcorner_tracking_dir}/{skillcorner_match_id}.json"
    skillcorner_match_path = f"{skillcorner_match_dir}/{skillcorner_match_id}.json"

    # Load StatsBomb events
    events = pd.read_csv(statsbomb_event_path)
    
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

    # Convert the trackable object dict
    trackable_objects = {}
    home_count = away_count = 0
    
    for player in match['players']:
        role = team_dict[player['team_id']]['role']
        position = player['player_role']['name']
        if role == 'home':
            trackable_objects[player['trackable_object']] = {
                'name': f"{player['first_name']} {player['last_name']}",
                'team': team_dict[player['team_id']]['name'],
                'role': role,
                'id': home_count,
                'position': position
            }
            home_count += 1
        elif role == 'away':
            trackable_objects[player['trackable_object']] = {
                'name': f"{player['first_name']} {player['last_name']}",
                'team': team_dict[player['team_id']]['name'],
                'role': role,
                'id': away_count,
                'position': position
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
        match_id = statsbomb_match_id
        period = event['period']
        time = event['timestamp']
        minute = event['minute']
        second = event['second']
        event_type = event['type']
        event_type_2 = None
        end_x = end_y = None
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
            if pass_type=="Corner":
                event_type_2="Corner"
            elif cross and not np.isnan(cross):
                event_type_2="Cross"
            elif pass_height:
                event_type_2=pass_height
        elif event_type=="Shot":
            event_type_2=event.get('shot_outcome')

        team = event['team']
        home_team = 1 if team == home_team_name else 0
        player = event['player']
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

        df_list.append([match_id, period, time, minute, second, seconds, event_type, event_type_2, team, home_team, player, start_x, start_y, end_x, end_y, *home_tracking, *away_tracking, *home_side])
    
    # Define DataFrame columns
    home_tracking_columns = []
    away_tracking_columns = []
    for i in range(1, 24):
        home_tracking_columns.extend([f"h{i}_x", f"h{i}_y"])
        away_tracking_columns.extend([f"a{i}_x", f"a{i}_y"])
    columns = ["match_id", "period", "time", "minute", "second", 'seconds', "event_type", "event_type_2", "team", "home_team", "player", "start_x", "start_y","end_x","end_y"] + home_tracking_columns + away_tracking_columns + ["home_side"]

    # Convert the event list to a DataFrame
    df = pd.DataFrame(df_list, columns=columns)

    #Sort the DataFrame by 'period' then 'seconds'
    df = df.sort_values(by=["period", "seconds"]).reset_index(drop=True)

    return df

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

def load_pff2metrica(event_path:str, match_id:str = None) -> pd.DataFrame:
    """
    Convert PFF-style event data to Metrica format.

    Parameters
    ----------
    event_df : pd.DataFrame
        Event data from PFF dataset with columns like:
        - gameEvents_period
        - gameEvents_playerName
        - possessionEvents_receiverPlayerName
        - possessionEvents_possessionEventType
        - startTime, endTime, duration
        - gameEvents_homeTeam
        - various outcome types for success/failure
    match_id : str, optional
        Match identifier to add as a column, by default None

    Returns
    -------
    Metrica_df : pd.DataFrame
        DataFrame in Metrica format with columns:
        ['Team', 'Type', 'Subtype', 'Period', 'Start Frame', 'Start Time [s]',
         'End Frame', 'End Time [s]', 'From', 'To', 'Start X', 'Start Y', 'End X', 'End Y']
    """
    with open(event_path, 'r') as f:
        event_data = json.load(f)
        event_df = pd.json_normalize(event_data, sep='_')
    
    def type_id2name(x):
        """
        Map event type codes to descriptive names.

        Parameters
        ----------
        x : str | int | float | None
            Event type code (e.g., 'PA', 'SH', 'FO', etc.)

        Returns
        -------
        str | None
            Descriptive event type name, or None if not mapped.
        """
        import math
        if x in ['PA']:
            x = "pass"
        elif x in ['CR']:
            x = "cross"
        # elif x == 2:
        #     x = "throw_in"
        # elif x == 5:
        #     x = "corner_crossed"
        # elif x == 7:
        #     x = "take_on"
        elif x in ['FO']:
            x = "foul"
        elif x in ['CH']:
            x = "tackle"
        # elif x == 10:
        #     x = "interception"
        elif x in ['SH']:
            x = "shot"
        elif x in ['CL']:
            x = "clearance"
        elif x in ['BC']:
            x = "dribble"
        # elif x == 22:
        #     x = "goalkick"
        elif x in ['IT', 'RE', 'TC']:
            x = "other"
        elif x is None or (isinstance(x, (float, int)) and math.isnan(x)):
            x = None
        else:
            print(f"Unmapped event type: {x}")
        return x
    def extract_player_xy(row):
        """
        Extracts the (x, y) coordinates of the player involved in a game event.

        Parameters
        ----------
        row : pd.Series
            A row from a DataFrame containing game event and player information. 
            Expected keys:
                - "gameEvents_homeTeam" (bool): True if home team, False if away team.
                - "homePlayers" (list|str): List or stringified list of home team players.
                - "awayPlayers" (list|str): List or stringified list of away team players.
                - "gameEvents_playerId" (int): ID of the player involved in the event.

        Returns
        -------
        pd.Series
            A Series with coordinates:
            - "start_x"
            - "start_y"
            - "end_x"
            - "end_y"
            If the player is not found, all values are None.
        """
        # choose player list
        if row["gameEvents_homeTeam"] is True:
            player_dict = row["homePlayers"]
        elif row["gameEvents_homeTeam"] is False:
            player_dict = row["awayPlayers"]
        else:
            return pd.Series([None, None, None, None], index=["start_x", "start_y", "end_x", "end_y"])
        
        # find target player
        player_dict = ast.literal_eval(player_dict) if type(player_dict) == str else player_dict
        target_player = next((d for d in player_dict if d["playerId"] == row["gameEvents_playerId"]), None)

        if target_player:
            return pd.Series(
                [target_player["x"], target_player["y"], target_player["x"], target_player["y"]],
                index=["start_x", "start_y", "end_x", "end_y"]
            )
        else:
            return pd.Series([None, None, None, None], index=["start_x", "start_y", "end_x", "end_y"])

    # drop row where gameEvents_startGameClock is NaN
    event_df = event_df.dropna(subset=['gameEvents_startGameClock']).reset_index(drop=True)

    # set column name
    column_name = ['Team', 
          'Type',
          'Subtype',
          'Period',
          'Start Frame',
          'Start Time [s]',
          'End Frame',
          'End Time [s]',
          'From',
          'To',
          'Start X',
          'Start Y',
          'End X',
          'End Y']
    Metrica_df = pd.DataFrame(columns=column_name)
    Metrica_df['Period'] = event_df['gameEvents_period']
    event_df[["start_x", "start_y", "end_x", "end_y"]] = event_df.apply(extract_player_xy, axis=1)
    Metrica_df['Start X'] = event_df['start_x'] #- 52.5
    Metrica_df['Start Y'] = event_df['start_y'] #- 34
    Metrica_df['End X'] = event_df['end_x'] #- 52.5
    Metrica_df['End Y'] = event_df['end_y'] #- 34
    Metrica_df['From'] = event_df['gameEvents_playerName']
    Metrica_df['To'] = event_df['possessionEvents_receiverPlayerName']
    Metrica_df['Type'] = event_df['possessionEvents_possessionEventType']
    Metrica_df['Type'] = Metrica_df['Type'].apply(type_id2name)

    idx = event_df.index

    def col(name):
        """Safe getter: returns Series aligned to df (all NaN if col missing)."""
        return event_df[name] if name in event_df.columns else pd.Series(pd.NA, index=idx)

    # Raw outcome columns
    pass_out   = col('possessionEvents_passOutcomeType')       
    cross_out  = col('possessionEvents_crossOutcomeType')       
    shot_out   = col('possessionEvents_shotOutcomeType')        
    clr_out    = col('possessionEvents_clearanceOutcomeType')  
    tkl_out    = col('possessionEvents_challengeOutcomeType')   
    carry_out  = col('possessionEvents_ballCarryOutcome')       
    touch_out  = col('possessionEvents_touchOutcomeType')       

    # Per-action success masks (nullable booleans)
    event_df['pass_success']      = pass_out.isin(['C'])
    event_df['cross_success']     = cross_out.isin(['C'])
    event_df['shot_success']      = shot_out.isin(['G'])
    event_df['clearance_success'] = ~clr_out.isin(['B','D']) & clr_out.notna()
    event_df['tackle_success']    = tkl_out.isin(['B','C','M'])
    event_df['dribble_success']   = carry_out.isin(['R'])
    event_df['touch_success']     = touch_out.isin(['R'])

    # Where each action is *present* (not NaN), assign Subtype based on its success
    event_df['Subtype'] = np.nan

    def apply_subtype(success_col, present_series):
        """Set Subtype for rows where this action is present."""
        is_present = present_series.notna()
        success    = event_df[success_col] == True
        fail       = event_df[success_col] == False
        event_df.loc[is_present & success, 'Subtype'] = 'success'
        event_df.loc[is_present & fail,    'Subtype'] = 'fail'

    apply_subtype('pass_success',      pass_out)
    apply_subtype('cross_success',     cross_out)
    apply_subtype('shot_success',      shot_out)
    apply_subtype('clearance_success', clr_out)
    apply_subtype('tackle_success',    tkl_out)
    apply_subtype('dribble_success',   carry_out)
    apply_subtype('touch_success',     touch_out)
    Metrica_df['Subtype'] = event_df['Subtype']

    fps = 29.97

    Metrica_df['Start Time [s]'] = (event_df['gameEvents_startGameClock']).round().astype(int)
    Metrica_df['End Time [s]'] = (event_df['duration'] + event_df['gameEvents_startGameClock']).round().astype(int)

    Metrica_df['Start Frame'] = ((event_df['startTime'] - event_df['startTime'][0]) * fps).round().astype(int)
    end_frame = ((event_df['endTime'] - event_df['startTime'][0]) * fps).round()
    Metrica_df['End Frame'] = end_frame.fillna(Metrica_df['Start Frame']).astype(int)
    Metrica_df['Team'] = np.where(event_df['gameEvents_homeTeam'] == True, 'Home',
                      np.where(event_df['gameEvents_homeTeam'] == False, 'Away', None))

    #drop rows where start_x or start_y is NaN
    Metrica_df = Metrica_df.dropna(subset=['Start X', 'Start Y'])
    Metrica_df = Metrica_df.reset_index(drop=True)

    if match_id is not None:
        Metrica_df['match_id'] = match_id
        cols = Metrica_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        Metrica_df = Metrica_df[cols]

    return Metrica_df

def load_datastadium(
    datastadium_event_path: str,
    datastadium_home_tracking_path: str,
    datastadium_away_tracking_path: str
    ) -> pd.DataFrame:
    """
    Loads and processes event and tracking data from stadium event recordings.
    
    Args:
        datastadium_event_path (str): Path to the CSV file containing event data.
        datastadium_home_tracking_path (str): Path to the CSV file containing home team tracking data.
        datastadium_away_tracking_path (str): Path to the CSV file containing away team tracking data.
        
    Returns:
        pd.DataFrame: A DataFrame containing the merged and processed event and tracking data.
    """
    # Load data
    event = pd.read_csv(datastadium_event_path, encoding='shift_jis')
    home_tracking = pd.read_csv(datastadium_home_tracking_path)
    away_tracking = pd.read_csv(datastadium_away_tracking_path)

    # Define required columns and flags
    required_columns = [
        "試合ID", "ホームアウェイF", "チーム名", "選手名", "アクション名", "F_成功",
        "位置座標X", "位置座標Y", "敵陣F", "点差", "自スコア", "相手スコア",
        "F_ゴール", "F_セーブ", "F_シュートGK以外", "F_ミスヒット", "ゴール角度", 
        "ゴール距離", "F_パス", "F_クロス", "F_ドリブル", "F_クリア", 
        "F_ハンドクリア", "F_ゴールキック", "F_コーナーキック", "F_直接フリーキック",
        "F_間接フリーキック", "絶対時間秒数", "フレーム番号","距離"
    ]
    flags = [
        "F_ゴール", "F_セーブ", "F_シュートGK以外", "F_ミスヒット", "F_パス", 
        "F_クロス", "F_ドリブル", "F_クリア", "F_ハンドクリア", "F_ゴールキック", 
        "F_コーナーキック", "F_直接フリーキック", "F_間接フリーキック"
    ]
    event_type_dict = {
        "前半開始": "First Half Start", "前半終了": "First Half End", "後半開始": "Second Half Start", 
        "後半終了": "Second Half End", "延長前半開始": "Overtime First Half Start", 
        "延長前半終了": "Overtime First Half End", "延長後半開始": "Overtime Second Half Start",
        "延長後半終了": "Overtime Second Half End", "再延長前半開始": "Second Overtime First Half Start",
        "再延長前半終了": "Second Overtime First Half End", "再延長後半開始": "Second Overtime Second Start",
        "再延長後半終了": "Second Overtime Second End", "PK戦開始": "PK Start", "PK戦終了": "PK End",
        "シュート": "Shoot", "GK": "GK", "直接FK": "Direct FK", "キャッチ": "Catch", 
        "警告(イエロー)": "YellowCard", "PK": "PK", "CK": "CK", "間接FK": "Indirect FK", 
        "オフサイド": "Offside", "退場(レッド)": "RedCard", "交代": "Change", "キックオフ": "KickOff", 
        "ファウルする": "Foul", "オウンゴール": "OwnGoal", "ホームパス": "HomePass", 
        "アウェイパス": "AwayPass", "PKパス": "PKPass", "ポジション変更": "Position Change", 
        "中断": "Suspension", "ドリブル": "Dribble", "スルーパス": "Through Pass", 
        "ハンドクリア": "Hand Clear", "ファウル受ける": "Foul", "ドロップボール": "Drop Ball", 
        "ボールアウト": "Ball Out", "インターセプト": "Intercept", "クリア": "Clear", 
        "ブロック": "Block", "スローイン": "ThrowIn", "クロス": "Cross", "トラップ": "Trap", 
        "PK合戦": "PK Battle", "試合再開": "Resume", "フィード": "Feed", "タッチ": "Touch", 
        "タックル": "Tackle", "フリックオン": "FrickOn", "試合中断": "Suspension", 
        "ポスト/バー": "Post Bar", "試合中断(試合中)": "Suspension(InGame)", 
        "試合再開(試合中)": "Resume(InGame)"
    }
    flag_dict = {
        "F_ゴール": "Goal", "F_セーブ": "Save", "F_シュートGK以外": "Shot(not_GK)", 
        "F_ミスヒット": "MissHit", "F_パス": "Pass", "F_クロス": "Cross", "F_ドリブル": "Dribble",
        "F_クリア": "Clear", "F_ハンドクリア": "HandClear", "F_ゴールキック": "GoalKick", 
        "F_コーナーキック": "CornerKick", "F_直接フリーキック": "DirectFreeKick",
        "F_間接フリーキック": "IndirectFreeKick"
    }
    
    # Filter columns and preprocess data
    event = event[required_columns].copy()
    event["絶対時間秒数"] = event["絶対時間秒数"].astype(float)
    event = event.sort_values(by="絶対時間秒数")

    # Create event_type_2 column based on flags
    def get_event_type_2(row):
        event_types = [flag_dict[f] for f in flags if row[f] == 1]
        return "/".join(event_types) if event_types else None

    event["event_type_2"] = event.apply(get_event_type_2, axis=1)
    event = event.drop(columns=flags)

    # Rename columns
    event.columns = [
        "match_id", "home", "team", "player", "event_type", "success", 
        "start_x", "start_y", "opp_field", "point_diff", "self_score", 
        "opp_score", "angle2goal", "dist2goal", "absolute_time", 
        "frame", "dist", "event_type_2"
    ]
    
    # Reorder columns
    event = event[[
        "match_id", "team", "home", "player", "frame", "absolute_time",
        "event_type", "event_type_2", "success", "start_x", "start_y","dist",
        "opp_field", "point_diff", "self_score", "opp_score", "angle2goal",
        "dist2goal"
    ]]

    # Convert event_type to English
    event["event_type"] = event["event_type"].map(event_type_dict).fillna(event["event_type"])

    # Calculate period, minute, and second
    def calculate_time(row, half_start, period_flag):
        time_elapsed = float(row["absolute_time"]) - half_start
        return int(time_elapsed / 60), round(time_elapsed % 60, 4)

    period, minute, second = [], [], []
    half_start = float(event.iloc[0]["absolute_time"])
    period_flag = 1

    for _, row in event.iterrows():
        if row["event_type"] == "Second Half Start":
            period_flag = 2
            half_start = float(row["absolute_time"])

        period.append(period_flag)
        m, s = calculate_time(row, half_start, period_flag)
        minute.append(m)
        second.append(s)

    event["Period"] = period
    event["Minute"] = minute
    event["Second"] = second

    # Reorder columns
    event = event[[
        "match_id", "Period", "Minute", "Second", "frame", "absolute_time",
        "team", "home", "player", "event_type", "event_type_2", "success",
        "start_x", "start_y", "dist", "opp_field", "point_diff", "self_score",
        "opp_score", "angle2goal", "dist2goal"
    ]]

    #reset the index
    event.reset_index(drop=True, inplace=True)

    # get the tracking start time for 2nd half
    tracking_start_time_2 = home_tracking[home_tracking["Period"] == 2].iloc[0]["Time [s]"]

    #sort both tracking data
    home_tracking = home_tracking.sort_values(by="Time [s]").reset_index(drop=True)
    away_tracking = away_tracking.sort_values(by="Time [s]").reset_index(drop=True)

    home_tracking_time = home_tracking["Time [s]"].round(2).values
    tracking_col_home = [f"Home_{i}_x" for i in range(1, 15)] + [f"Home_{i}_y" for i in range(1, 15)]
    tracking_col_away = [f"Away_{i}_x" for i in range(1, 15)] + [f"Away_{i}_y" for i in range(1, 15)]

    # Calculate event times vectorized
    event_time = event["Minute"] * 60 + event["Second"] + tracking_start_time_2 * (event["Period"] == 2)

    # Find nearest indices using numpy
    nearest_indices = np.searchsorted(home_tracking_time, event_time,side='left')
    nearest_indices = np.clip(nearest_indices, 0, len(home_tracking_time) - 1)

    # Get the corresponding tracking data
    home_tracking_data = home_tracking.iloc[nearest_indices][tracking_col_home].values
    away_tracking_data = away_tracking.iloc[nearest_indices][tracking_col_away].values

    # pdb.set_trace()

    # Combine the results
    new_df = pd.concat([event, pd.DataFrame(home_tracking_data, columns=tracking_col_home),
                        pd.DataFrame(away_tracking_data, columns=tracking_col_away)], axis=1)


    # Create final DataFrame
    columns = [
        "match_id", "absolute_time", "Period", "Minute", "Second", "team", "home", "player", 
        "event_type", "event_type_2", "success", "start_x", "start_y", "dist",
        "opp_field", "point_diff", "self_score", "opp_score", 
        "angle2goal", "dist2goal"] + tracking_col_home + tracking_col_away
    
    final_df = pd.DataFrame(new_df, columns=columns)

    return final_df

def load_robocup_2d(event_path: str, match_id: str = None, tracking_path: str = None) -> pd.DataFrame:
    """
    Load event data from CSV file and optionally merge with tracking data.

    Args:
        event_path (str): Path to the CSV file containing event data.
        match_id (str, optional): Identifier for the match. Defaults to None.
        tracking_path (str, optional): Path to the CSV file containing tracking data. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing event and tracking data.
    """
    # Load event data from CSV file
    event_df = pd.read_csv(event_path)
    
    # Load tracking data if provided
    if tracking_path:
        tracking_df = pd.read_csv(tracking_path)
    
    # Define columns for the DataFrame
    columns = ["match_id", "seconds", "event_type", "outcome", "team", "player", "start_x", "start_y", "end_x", "end_y"]
    if tracking_path:
        columns.extend([" l_score", " r_score", " b_x", " b_y"])
        for i in range(1, 12):
            columns.extend([f" l{i}_x", f" l{i}_y"])
        for i in range(1, 12):
            columns.extend([f" r{i}_x", f" r{i}_y"])
        
    
    # Initialize an empty list to store event details
    event_list = []
    
    # Iterate through event records
    for index, record in event_df.iterrows():
        seconds = record.get('Time1', None)
        event_type = record.get('Type', None)
        outcome = record.get('Success', None)
        team = record.get('Side1', None)
        player = record.get('Unum1', None)
        start_x = record.get('X1', None)
        start_y = record.get('Y1', None)
        end_x = record.get('X2', None)
        end_y = record.get('Y2', None)
        
        # If tracking data is provided, merge with event details
        if tracking_path:
            if seconds in tracking_df[' cycle'].values:
                tracking_record = tracking_df[tracking_df[' cycle'] == seconds]
                if tracking_record.shape[0] != 1:
                    print(f"Error: Tracking record {index} has more than one row")
                    continue
                
                # Extract tracking data
                tracking_values = tracking_record.iloc[0].to_dict()

                # tracking_values.pop(' cycle')  # Remove the cycle column
                tracking_values = {key: value for key, value in tracking_values.items() if key in columns}
                # Append event and tracking details to the list
                event_list.append([match_id, seconds, event_type, outcome, team, player, start_x, start_y, end_x, end_y, *tracking_values.values()])
        else:
            # Append only event details
            event_list.append([match_id, seconds, event_type, outcome, team, player, start_x, start_y, end_x, end_y])
    
    # Convert the event list to a DataFrame
    df = pd.DataFrame(event_list, columns=columns)
    
    # Sort the DataFrame by 'seconds'
    df = df.sort_values(by="seconds").reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    import pdb
    import os
    #cd to ../PreProcessing
    datafactory_path=os.getcwd()+"/test/sports/event_data/data/datafactory/datafactory_events.json"
    metrica_event_json_path=os.getcwd()+"/test/sports/event_data/data/metrica/metrica_events.json"
    metrica_event_csv_path=os.getcwd()+"/test/sports/event_data/data/metrica/Sample_Game_1/Sample_Game_1_RawEventsData.csv"
    metrica_tracking_home_path=os.getcwd()+"/test/sports/event_data/data/metrica/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
    metrica_tracking_away_path=os.getcwd()+"/test/sports/event_data/data/metrica/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"
    opta_f7_path=os.getcwd()+"/test/sports/event_data/data/opta/opta_f7.xml"
    opta_f24_path=os.getcwd()+"/test/sports/event_data/data/opta/opta_f24.xml"
    robocup_2d_event_path=os.getcwd()+"/test/sports/event_data/data/robocup_2d/202307091024-HELIOS2023_1-vs-CYRUS_0-pass.csv"
    robocup_2d_tracking_path=os.getcwd()+"/test/sports/event_data/data/robocup_2d/202307091024-HELIOS2023_1-vs-CYRUS_0.csv"
    sportec_event_path=os.getcwd()+"/test/sports/event_data/data/sportec/sportec_events.xml"
    sportec_tracking_path=os.getcwd()+"/test/sports/event_data/data/sportec/sportec_positional.xml"
    sportec_meta_path=os.getcwd()+"/test/sports/event_data/data/sportec/sportec_meta.xml"
    statsbomb_event_path=os.getcwd()+"/test/sports/event_data/data/statsbomb/events/3805010.json"
    statsbomb_360_path=os.getcwd()+"/test/sports/event_data/data/statsbomb/three-sixty/3805010.json"
    statsbomb_api_path=os.getcwd()+"/test/sports/event_data/data/statsbomb/api.json"
    statsbomb_skillcorner_event_path="/data_pool_1/laliga_23/statsbomb/events"
    statsbomb_skillcorner_tracking_path="/data_pool_1/laliga_23/skillcorner/tracking"
    statsbomb_skillcorner_match_path="/data_pool_1/laliga_23/skillcorner/match"
    wyscout_event_path=os.getcwd()+"/test/sports/event_data/data/wyscout/events_England.json"
    wyscout_matches_path=os.getcwd()+"/test/sports/event_data/data/wyscout/matches_England.json"
    datastadium_event_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/play.csv"
    datastadium_home_tracking_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/home_tracking.csv"
    datastadium_away_tracking_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/away_tracking.csv"

    #test load_datafactory
    # datafactory_df=load_datafactory(datafactory_path)
    # datafactory_df.to_csv(os.getcwd()+"/test/sports/event_data/data/datafactory/test_data.csv",index=False)

    #test load_metrica
    # metrica_df=load_metrica(metrica_event_json_path,1,metrica_tracking_home_path,metrica_tracking_away_path)
    # metrica_df.to_csv(os.getcwd()+"/test/sports/event_data/data/metrica/test_data_json.csv",index=False)
    # metrica_df=load_metrica(metrica_event_csv_path,1,metrica_tracking_home_path,metrica_tracking_away_path)
    # metrica_df.to_csv(os.getcwd()+"/test/sports/event_data/data/metrica/test_data_csv.csv",index=False)
    
    #test load_opta_xml
    # opta_df=load_opta_xml(opta_f24_path,1)
    # opta_df.to_csv(os.getcwd()+"/test/sports/event_data/data/opta/test_data.csv",index=False)

    #test load_robocup_2d
    # robocup_2d_df=load_robocup_2d(robocup_2d_event_path,1,robocup_2d_tracking_path)
    # robocup_2d_df.to_csv(os.getcwd()+"/test/sports/event_data/data/robocup_2d/test_data.csv",index=False)

    #test load_sportec
    # sportec_df=load_sportec(sportec_event_path,sportec_tracking_path,sportec_meta_path)
    # sportec_df.to_csv(os.getcwd()+"/test/sports/event_data/data/sportec/test_data.csv",index=False)
    
    #test load_statsbomb with json file
    # statsbomb_df=load_statsbomb(statsbomb_event_path,statsbomb_360_path)
    # statsbomb_df.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_data.csv",index=False)

    # test load_statsbomb with api data
    # statsbomb_df=load_statsbomb(match_id=3795108)
    # statsbomb_df.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_api_data.csv",index=False)

    #test load_statsbomb_skillcorner
    # statsbomb_skillcorner_df=load_statsbomb_skillcorner(statsbomb_skillcorner_event_path,statsbomb_skillcorner_tracking_path,
    #                                                     statsbomb_skillcorner_match_path,3894907,1553748)
    # statsbomb_skillcorner_df.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb_skillcorner/test_data.csv",index=False)

    #test load_wyscout
    # wyscout_df=load_wyscout(wyscout_event_path,wyscout_matches_path)
    # wyscout_df.head(1000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_data.csv",index=False)


    #test load_datastadium
    # event=load_datastadium(datastadium_event_path,datastadium_home_tracking_path,datastadium_away_tracking_path)
    # event.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/load.csv",index=False)

    #test load_soccertrack
    soccer_track_event_path="/data_pool_1/soccertrackv2/2023-11-18/Event/event.csv"
    soccer_track_tracking_path="/data_pool_1/soccertrackv2/2023-11-18/Tracking/tracking.xml"
    soccer_track_meta_path="/data_pool_1/soccertrackv2/2023-11-18/Tracking/meta.xml"
    df_soccertrack=load_bepro(soccer_track_event_path,soccer_track_tracking_path,soccer_track_meta_path,True)
    df_soccertrack.head(1000).to_csv(os.getcwd()+"/test/sports/event_data/data/soccertrack/test_load_function_sync.csv",index=False)

    print("----------------done-----------------")
    # pdb.set_trace()


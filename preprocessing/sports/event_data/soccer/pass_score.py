import pandas as pd
import json
import os
import numpy as np
import math
from tqdm import tqdm

# File paths
statsbomb_event_dir = "/home/z_chen/workspace3/test"
skillcorner_tracking_dir = '/home/z_chen/workspace3/laliga/laliga_23/skillcorner_v2/tracking'
skillcorner_match_dir = '/home/z_chen/workspace3/laliga/laliga_23/skillcorner_v2/match'
match_id_df = '/home/z_chen/workspace3/PreProcessing/example/id_matching.csv'

statsbomb_match_id = 'data_processed'
skillcorner_match_id = '1553748'

statsbomb_event_path = f"{statsbomb_event_dir}/{statsbomb_match_id}.csv"
skillcorner_tracking_path = f"{skillcorner_tracking_dir}/{skillcorner_match_id}.json"
skillcorner_match_path = f"{skillcorner_match_dir}/{skillcorner_match_id}.json"

# Check if files exist
if not os.path.exists(statsbomb_event_path):
    raise FileNotFoundError(f"Statsbomb event file not found: {statsbomb_event_path}")
if not os.path.exists(skillcorner_tracking_path):
    raise FileNotFoundError(f"Skillcorner tracking file not found: {skillcorner_tracking_path}")
if not os.path.exists(skillcorner_match_path):
    raise FileNotFoundError(f"Skillcorner match file not found: {skillcorner_match_path}")

# Load StatsBomb events
events = pd.read_csv(statsbomb_event_path)

# Load SkillCorner tracking and match data
with open(skillcorner_tracking_path) as f:
    tracking = json.load(f)

with open(skillcorner_match_path, encoding='utf-8') as f:
    match = json.load(f)

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

# Filter pass events
pass_events = events[events['event_type'] == 'Pass']  # Adjust based on your events file's column name
pass_events_seconds = pass_events["seconds"].tolist()
 
# Get window of frames around an action
def get_window_of_frames_around(action_second, tracking_frames, ta):
    """
    Gets a window of frames around the given action_second.
    """
    window = []
    for frame in tracking_frames:
        timestamp = frame.get('timestamp')
        if timestamp:
            try:
                time_components = list(map(float, timestamp.split(':')))
                frame_seconds = time_components[0] * 3600 + time_components[1] * 60 + time_components[2]
                if action_second - ta <= frame_seconds <= action_second + ta:
                    window.append(frame)
            except (ValueError, IndexError):
                print(f"Invalid timestamp format: {timestamp}")
                continue
    return window

# Process windows
ta = 5
windows = []
for action_second in tqdm(pass_events_seconds, desc="Processing windows"):
    window = get_window_of_frames_around(action_second, tracking, ta)
    if not window:
        print(f"No frames found around action_second={action_second}")
    windows.append(window)

# save documents

# test_save_path = "/home/z_chen/workspace3/test/windows_test.json"
# with open(test_save_path, 'w', encoding='utf-8') as f:
#     json.dump(windows, f, ensure_ascii=False, indent=4)  # 使用 indent=4 格式化 JSON 输出
# print(f"Windows saved to {test_save_path}")

# try:
#     with open(test_save_path,"r",encoding="utf-8") as f:
#         datas = json.load(f)

# except FileNotFoundError:
#     print(f"文件 {test_save_path} 不存在，请检查路径！")
# except json.JSONDecodeError as e:
#     print(f"JSON 文件解码失败: {e}")

# windows = []
# for data in datas:
#     windows.append(data)


# Sync the tracking data with the events based on the highest ball acceleration and the kick-off player

pass_events = events[events['event_type'] == 'Pass']
pass_player = pass_events["player"]
pass_player = pass_player.str.strip()
player_periods = {}

ball_velocity_periods = []
ball_id = match['ball']['trackable_object']


for player in tqdm(match["players"], desc="Processing players"):
    full_name = f"{player['first_name']} {player['last_name']}".strip()
    obj_id = player["trackable_object"]

    if full_name in pass_player:
        for index, window in enumerate(windows):
            key = index + 1
            if key not in player_periods:
                player_periods[key] = []

            for frame in window:
                time = frame.get('timestamp')
                period = frame.get('period')
                data = frame.get('data', [])
                possession = frame.get("possession", {})  
                team_group = possession.get("group")  
                if not team_group:
                    continue

                if time:
                    try:
                        time_components = time.split(':')
                        seconds = float(time_components[0]) * 3600 + float(time_components[1]) * 60 + float(time_components[2])
                    except (ValueError, IndexError):
                        continue

                    for obj in data:
                        if obj.get('trackable_object') == ball_id:
                            ball_velocity_periods.append([seconds, obj['x'], obj['y'], obj['z']])
                        elif obj.get('trackable_object') == obj_id:
                            player_periods[key].append([seconds, obj['x'], obj['y'], obj["trackable_object"],team_group])

    else:

        for index, window in enumerate(windows):
            key = index + 1
            if key not in player_periods:
                player_periods[key] = [] 

            for frame in window:
                possession = frame.get("possession", {})  
                team_group = possession.get("group")  # may be empty
           
                if not team_group:
                    continue

                time = frame.get('timestamp')
                if time:
                    try:
                        time_components = time.split(':')
                        seconds = float(time_components[0]) * 3600 + float(time_components[1]) * 60 + float(time_components[2])
                    except (ValueError, IndexError):
                        print(f"Invalid timestamp format: {time}. Skipping frame.")
                        continue
                period = frame.get('period')
                data = frame.get('data', [])

       
                for obj in data:
                    trackable_object = obj.get('trackable_object')

  
                    if trackable_object == ball_id:
                        ball_velocity_periods.append([seconds, obj['x'], obj['y'], obj.get('z', 0)])
            
                    elif trackable_object == obj_id:
                        player_periods[key].append([seconds, obj['x'], obj['y'], trackable_object,team_group])




def find_min_distance(results):
    min_distance = float("inf")
    min_distance_second = None
    for item in results:
        if len(item) == 2:
            second, distance = item
            if distance < min_distance:
                min_distance = distance
                min_distance_second = second
        else:
            print(f"Unexpected item in results: {item}")
    return min_distance, min_distance_second


if not isinstance(pass_events, pd.DataFrame):
    raise ValueError("pass_events is not a Pandas DataFrame. Check data loading or filtering.")


if "seconds" not in pass_events.columns:
    raise KeyError("'seconds' column is missing in pass_events DataFrame.")


pass_events_periods = []
for _, row in pass_events.iterrows():
    pass_second = row["seconds"]
    pass_x = row["start_x"]
    pass_y = row["start_y"]
    pass_event_team = row["home_team"]
    if pass_event_team == 1:
        pass_events_periods.append([pass_second,pass_x,pass_y,'home team'])
    else:
        pass_events_periods.append([pass_second,pass_x,pass_y,'away team'])



def distance_score(pass_event, player_period, ball_velocity_period):

    event_second = pass_event[0]    
    event_player_x = pass_event[1]
    event_player_y = pass_event[2]
    event_player_side = pass_event[3]

    ball_data_dict = {
        "second": ball_velocity_period[0], 
        "x": ball_velocity_period[1], 
        "y": ball_velocity_period[2]   
    }

    tracking_event_frame = [] # save the distances and time of all tracking players in this second
    ball_distances_frame = [] # all distances between tracking players and ball in this second

    tracking_second = player_period[0]    
    tracking_player_x = player_period[1]
    tracking_player_y = player_period[2]
    tracking_player_id = player_period[3]

    tracking_event_distance = ((event_player_x - tracking_player_x) ** 2 + (event_player_y - tracking_player_y) ** 2) ** 0.5
    tracking_event_frame.append((tracking_second, tracking_event_distance, tracking_player_id))

    ball_distance = ((ball_data_dict["x"] - tracking_player_x) ** 2 + (ball_data_dict["y"] - tracking_player_y) ** 2) ** 0.5
    ball_distances_frame.append((tracking_second, ball_distance, tracking_player_id))


    # calculate weighted scores
    scores = []
    weight_event = 0.5
    weight_ball = 0.5

    for i in range(len(tracking_event_frame)):
        tracking_time_event, event_distance, player_id_event = tracking_event_frame[i]
        tracking_time_ball, ball_distance, player_id_ball = ball_distances_frame[i]

        if tracking_time_event != tracking_time_ball or player_id_event != player_id_ball:
             continue

        combined_score = weight_event * (1 / (event_distance + 1e-6)) + weight_ball * (1 / (ball_distance + 1e-6))
        scores.append((tracking_time_event, player_id_event, combined_score))

        if scores:
            best_score = max(scores, key=lambda x: x[2])
            return best_score




for index, (pass_period, ball_velocity_period) in enumerate(zip(pass_events_periods, ball_velocity_periods)):
    pass_event_second = pass_period[0]
    pass_event_team = pass_period[3]
    ta = 5
    max_score = float('-inf') 
    max_score_info = [] 

    if index + 1 < len(player_periods):
        for i, player_period in enumerate(player_periods[index+1]):
            player_second = player_period[0]
            player_id = player_period[3]
            player_team = player_period[4]
            score = distance_score(pass_period, player_period, ball_velocity_period)[2]

            if score > max_score and pass_event_team == player_team:
                max_score = score
                max_score_info = [pass_event_second, player_second, max_score, player_id]

        if max_score_info:
            print(f"Latest max_score_info: {max_score_info}")




def distance_acceleration_score(ball_velocity_periods, player_periods):
    results = []



    for ball_velocity_period, player in zip(ball_velocity_periods, player_periods):
        distances_frame = []
        
        ball_data_dict = {ball_velocity_period[0]: (ball_velocity_period[1], ball_velocity_period[2], ball_velocity_period[3])}
        player_data_dict = {player[0]: (player[1], player[2])}

        for time, (ball_x, ball_y, ball_z) in ball_data_dict.items():
            if isinstance(time, str):
                time_components = time.split(':')
                seconds = float(time_components[0]) * 3600 + float(time_components[1]) * 60 + float(time_components[2])
            else:
                seconds = time

                player_x, player_y = player_data_dict[time]      
                distance = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
                distances_frame.append((seconds, distance))

    return results



 
def acceleration(data, data2):
    """
    Calculate the acceleration for each time interval and find the timestamp with the highest acceleration,
    under the condition that the ball is within 2 meters of the player position.

    Parameters:
    data (list): List of lists, where each sublist contains [timestamp, x, y, z].
    player_position (tuple): Tuple of player's x, y, z coordinates.

    Returns:
    tuple: (max_acceleration_timestamp, max_acceleration)
        - max_acceleration_timestamp: The timestamp with the highest acceleration while meeting the distance condition.
        - max_acceleration: The highest acceleration value.
    """
    # Extract timestamps, x, y, z coordinates

    accelerations = []

    seconds = [entry[0] for entry in data2]
    x = np.array([entry[1] for entry in data])
    y = np.array([entry[2] for entry in data])
    z = np.array([entry[3] for entry in data])


    # Calculate differences
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    delta_z = np.diff(z)
    delta_t = np.diff(seconds)

    # Calculate velocity components and magnitude
    vx = delta_x / delta_t
    vy = delta_y / delta_t
    vz = delta_z / delta_t
    velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

    # Calculate acceleration
    delta_vx = np.diff(vx) / delta_t[1:]  # Acceleration in x direction
    delta_vy = np.diff(vy) / delta_t[1:]  # Acceleration in y direction
    delta_vz = np.diff(vz) / delta_t[1:]  # Acceleration in z direction
    acceleration_magnitude = np.sqrt(delta_vx**2 + delta_vy**2 + delta_vz**2)

    accelerations = [[t, a] for t, a in zip(seconds, acceleration_magnitude)]

    return accelerations












#Target data provider [Metrica,Robocup 2D simulation,Statsbomb,Wyscout,Opta data,DataFactory,sportec]

import json
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from statsbombpy import sb
import pdb
# import logging

# logger = logging.getLogger('preprocessing.sports.event_data.load_data')

def load_datafactory(event_path: str) -> pd.DataFrame:
    """
    Load the datafactory event data from a JSON file and convert it into a DataFrame.

    Parameters:
        event_path (str): Path to the JSON file containing event data.

    Returns:
        pd.DataFrame: DataFrame containing event data with additional columns for analysis.
    """
    # Load data from the JSON file
    with open(event_path) as f:
        data = json.load(f)

    # Extract match id and team names
    match_id = data["match"]["matchId"]
    team_dict = {data['match']["homeTeamId"]: data['match']["homeTeamName"],
                 data['match']["awayTeamId"]: data['match']["awayTeamName"]}

    # Extract event details
    event_list = []
    for event_type, events in data["incidences"].items():  # Iterate over different types of events
        for event_id, record in events.items():  # Iterate over individual events
            period=record.get("t", {}).get("half", None)
            minute = record.get("t", {}).get("m", None)
            second = record.get("t", {}).get("s", None)
            event_type_2 = record.get("type", None)
            team = team_dict.get(record.get("team", None), None)
            player = record.get("plyrId", None)

            # Extract start and end coordinates if available
            start_coord = record.get("coord", {}).get("1", {})
            end_coord = record.get("coord", {}).get("2", {})
            start_x, start_y, start_z = start_coord.get("x", None), start_coord.get("y", None), start_coord.get("z", None)
            end_x, end_y, end_z = end_coord.get("x", None), end_coord.get("y", None), end_coord.get("z", None)

            #convert minute and second to float
            minute=float(minute)
            second=float(second)

            # Append event details to the list
            event_list.append([match_id, period, minute, second, event_type, event_type_2, team, player, start_x, start_y, start_z, end_x, end_y, end_z])

    # Convert the event list to a DataFrame
    columns = ["match_id", "period", "minute", "second", "event_type", "event_type_2", "team", "player", "start_x", "start_y", "start_z", "end_x", "end_y", "end_z"]
    df = pd.DataFrame(event_list, columns=columns)

    # Sort the DataFrame by 'minute' and 'second'
    df["seconds"] = df["minute"] * 60 + df["second"]

    # Reorder columns and sort the DataFrame by 'seconds'
    df = df[["match_id", "period",  "minute", "second", "seconds", "event_type", "event_type_2", "team", "player", "start_x", "start_y", "start_z", "end_x", "end_y", "end_z"]]
    df = df.sort_values(by="seconds").reset_index(drop=True)

    return df

def load_metrica(event_path: str, match_id: str = None, tracking_home_path: str = None, tracking_away_path: str = None) -> pd.DataFrame:
    """
    Loads event data and tracking data from specified file paths and returns a DataFrame containing the merged information.
    
    Parameters:
    -----------
    event_path : str
        Path to the JSON or CSV file containing event data.
    match_id : str, optional
        ID of the match. Default is None.
    tracking_home_path : str, optional
        Path to the CSV file containing home team tracking data. Default is None.
    tracking_away_path : str, optional
        Path to the CSV file containing away team tracking data. Default is None.
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the event data along with the extracted details, optionally including tracking data.
    """
    # Load event data
    if event_path.endswith('.json'):
        with open(event_path, 'r') as f:
            event_data = json.load(f)['data']
    elif event_path.endswith('.csv'):
        event_data = pd.read_csv(event_path)

    # Load tracking data if paths are provided
    tracking_home_df = pd.read_csv(tracking_home_path, skiprows=2) if tracking_home_path else None
    tracking_away_df = pd.read_csv(tracking_away_path, skiprows=2) if tracking_away_path else None

    # Function to process tracking data
    def process_tracking_data(tracking_df, frame):
        if tracking_df is not None and frame in tracking_df['Frame'].values:
            tracking = tracking_df[tracking_df['Frame'] == frame].copy()
            tracking.drop(tracking.columns[[0, 1, 2, -1, -2]], axis=1, inplace=True)
            tracking = tracking.values.tolist()[0]
            tracking.extend([None] * (2 * 16 - len(tracking)))
            return tracking
        return [None] * (2 * 16)

    # Extract event details and combine with tracking data
    event_list = []
    if isinstance(event_data, list):  # JSON data
        for record in event_data:
            # extract the subtypes if available
            event_type_2_dict = record.get('subtypes', None)
            if event_type_2_dict is not None:
                if isinstance(event_type_2_dict, list):
                    event_type_2 = "_".join(event.get('name', '') for event in event_type_2_dict)
                else:
                    event_type_2 = event_type_2_dict.get('name', None)
            else:
                event_type_2 = None
            # Extract event details
            event_details = [
                match_id,
                record.get('period', None),
                record.get('start', {}).get('time', None),
                frame := record.get('start', {}).get('frame', None),
                record.get('type', {}).get('name', None),
                event_type_2,
                record.get('team', {}).get('name', None),
                record.get('from', {}).get('name', None),
                record.get('start', {}).get('x', None),
                record.get('start', {}).get('y', None),
                record.get('end', {}).get('x', None),
                record.get('end', {}).get('y', None),
                *process_tracking_data(tracking_home_df, frame),
                *process_tracking_data(tracking_away_df, frame)
            ]
            event_list.append(event_details)
    else:  # CSV data
        for _, record in event_data.iterrows():
            event_details = [
                match_id,
                record.get('Period', None),
                record.get('Start Time [s]', None),
                frame := record.get('Start Frame', None),
                record.get('Type', None),
                record.get('Subtype', None),
                record.get('Team', None),
                record.get('From', None),
                record.get('Start X', None),
                record.get('Start Y', None),
                record.get('End X', None),
                record.get('End Y', None),
                *process_tracking_data(tracking_home_df, frame),
                *process_tracking_data(tracking_away_df, frame)
            ]
            event_list.append(event_details)

    # Define columns based on the availability of tracking data
    columns = ["match_id", "period", "seconds", "frame", "event_type", "event_type_2", "team", "player", "start_x", "start_y", "end_x", "end_y"]
    if tracking_home_df is not None:
        for i in range(1, 17):
            columns.extend([f"h{i}_x", f"h{i}_y"])
    if tracking_away_df is not None:
        for i in range(1, 17):
            columns.extend([f"a{i}_x", f"a{i}_y"])

    # Convert the event list to a DataFrame and sort by 'seconds'
    df = pd.DataFrame(event_list, columns=columns)
    df = df.sort_values(by="seconds").reset_index(drop=True)

    return df

def load_opta(f24_path: str, match_id: str = None) -> pd.DataFrame:
    """
    Loads Opta XML data and converts it into a pandas DataFrame.

    Args:
        f24_path (str): The file path to the Opta f24 XML file.
        match_id (str, optional): The ID of the match. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing event data.
    """
    # Parse the XML file
    tree = ET.parse(f24_path)
    root = tree.getroot()

    # Initialize an empty list to store event details
    event_list = []

    # Iterate through each Event in the Game
    for game in root.findall('Game'):
        for event in game.findall('Event'):
            # Extract event details
            period = event.attrib.get('period_id', None)
            minute = float(event.attrib.get('min', 0))
            second = float(event.attrib.get('sec', 0))
            event_type = event.attrib.get('event_id', None)
            event_type2 = event.attrib.get('type_id', None)
            outcome = event.attrib.get('outcome', None)
            team = event.attrib.get('team_id', None)
            start_x = event.attrib.get('x', None)
            start_y = event.attrib.get('y', None)

            # Append event details to the list
            event_list.append([match_id, period, minute, second, event_type, event_type2, outcome, team, start_x, start_y])

    # Convert the event list to a DataFrame
    columns = ["match_id", "period", "minute", "second", "event_type", "event_type_2", "outcome", "team", "start_x", "start_y"]
    df = pd.DataFrame(event_list, columns=columns)

    # Convert minute and second to total seconds for sorting
    df["total_seconds"] = df["minute"] * 60 + df["second"]

    # Sort the DataFrame by 'total_seconds'
    df = df.sort_values(by="total_seconds").reset_index(drop=True)

    # Reorder columns
    df = df[["match_id", "period", "minute", "second", "total_seconds", "event_type", "event_type_2", "outcome", "team", "start_x", "start_y"]]

    return df

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
        
def load_sportec(event_path: str, tracking_path: str = None, meta_path: str = None) -> pd.DataFrame:
    """
    Load Sportec event, tracking, and meta data from XML files into a pandas DataFrame. 

    Args:
    - event_path (str): Path to the event XML file.
    - tracking_path (str, optional): Path to the tracking XML file. Defaults to None.
    - meta_path (str, optional): Path to the meta XML file. Defaults to None.

    Returns:
    - pd.DataFrame: DataFrame containing the combined event, tracking, and meta data.
    """
    # Load the XML files
    tree_event = ET.parse(event_path)
    root_event = tree_event.getroot()

    root_tracking = root_meta = None
    if tracking_path:
        tree_tracking = ET.parse(tracking_path)
        root_tracking = tree_tracking.getroot()

    if meta_path:
        tree_meta = ET.parse(meta_path)
        root_meta = tree_meta.getroot()

    # Create dictionaries for team and player information
    team_dict = {}
    if root_meta:
        for team in root_meta.find('.//Teams'):
            team_dict[team.attrib.get('TeamId')] = {
                "name": team.attrib.get('TeamName'),
                "role": team.attrib.get('role')
            }

    player_dict = {}
    if root_meta:
        for team in root_meta.find('.//Teams'):
            players = team.find('Players')
            for player in players.findall('Player'):
                player_dict[player.attrib.get('PersonId')] = player.attrib.get('FirstName') + ' ' + player.attrib.get('LastName')

    # Initialize an empty list to store event details
    event_list = []

    # Iterate through each event in the game
    for record in root_event.findall('Event'):
        match_id = record.attrib.get('MatchId', None)
        time = record.attrib.get('EventTime', None)
        seconds = None

        if tracking_path:
            time_components = time.split('T')[1].split('+')[0].split(':')
            seconds = float(time_components[0]) * 3600 + float(time_components[1]) * 60 + float(time_components[2])

        start_x = record.attrib.get('X-Position', None)
        start_y = record.attrib.get('Y-Position', None)
        end_x = record.attrib.get('X-Source-Position', None)
        end_y = record.attrib.get('Y-Source-Position', None)

        event_type = None
        event_type2 = None
        team = None
        outcome = None
        player = None

        # Process event children and subchildren to extract event details
        for child in record:
            team = child.attrib.get('Team', team)
            outcome = child.attrib.get('Evaluation', outcome)
            player = child.attrib.get('Player', player)
            event_type = child.tag

            for subchild in child:
                team = subchild.attrib.get('Team', team)
                outcome = subchild.attrib.get('Evaluation', outcome)
                player = subchild.attrib.get('Player', player)
                event_type2 = subchild.tag if event_type2 is None else f"{event_type2}_{subchild.tag}"

                for subsubchild in subchild:
                    team = subsubchild.attrib.get('Team', team)
                    outcome = subsubchild.attrib.get('Evaluation', outcome)
                    player = subsubchild.attrib.get('Player', player)
                    event_type2 = f"{event_type2}_{subsubchild.tag}"

        # Replace team and player IDs with their names
        team_name = team_dict.get(team, {}).get('name')
        player_name = player_dict.get(player)

        home_team = away_team = []

        # Add the tracking data if available
        if tracking_path:
            for tracklet in root_tracking.findall('.//FrameSet'):
                if tracklet.attrib.get('TeamId') not in [None, "BALL"]:
                    for frame in tracklet:
                        frame_time = frame.attrib.get("T")
                        frame_seconds = sum(float(t) * f for t, f in zip(frame_time.split('T')[1].split('+')[0].split(':'), [3600, 60, 1]))

                        if abs(frame_seconds - seconds) <= 0.04:
                            coords = [frame.attrib.get('X'), frame.attrib.get('Y'), frame.attrib.get('Z')]
                            if team_dict.get(tracklet.attrib.get('TeamId'))['name'] == "home":
                                home_team.extend(coords)
                            else:
                                away_team.extend(coords)

            # Ensure both home and away teams have the correct number of coordinates
            home_team.extend([None] * (3 * 16 - len(home_team)))
            away_team.extend([None] * (3 * 16 - len(away_team)))

        # Append event details to the list
        event_list.append([
            match_id, time, event_type, event_type2, outcome, team_name, player_name,
            start_x, start_y, end_x, end_y, *home_team, *away_team
        ] if tracking_path else [
            match_id, time, event_type, event_type2, outcome, team_name, player_name,
            start_x, start_y, end_x, end_y
        ])

    # Define columns for the DataFrame
    columns = [
        "match_id", "time", "event_type", "event_type_2", "outcome", "team", "player",
        "start_x", "start_y", "end_x", "end_y"
    ]

    if tracking_path:
        columns += [f"h{i}_{axis}" for i in range(1, 17) for axis in ["x", "y", "z"]]
        columns += [f"a{i}_{axis}" for i in range(1, 17) for axis in ["x", "y", "z"]]

    # Convert the event list to a DataFrame
    df = pd.DataFrame(event_list, columns=columns)

    return df

def load_statsbomb(event_path: str = None, sb360_path: str = None, match_id: str = None, *statsbomb_api_args) -> pd.DataFrame:
    """
    Load StatsBomb event and 360 data from files or API. (currently only the api freezeframe data is not implemented)

    This function reads StatsBomb event data and optionally 360 freeze frame data from specified file paths or API.
    It returns a pandas DataFrame containing the consolidated information.

    Parameters:
    - event_path (str): Path to the JSON file containing the event data.
    - sb360_path (str): Path to the JSON file containing the 360 data.
    - match_id (str): Match ID to fetch the event data from the StatsBomb API.
    - *statsbomb_api_args: Additional arguments to pass to the StatsBomb API.

    Returns:
    - pd.DataFrame: DataFrame containing the consolidated event and freeze frame data.
    """
    # Read event data from file or API
    if event_path:
        with open(event_path) as f:
            event_data = json.load(f)
    elif match_id:
        event_api_data = sb.events(match_id=match_id, *statsbomb_api_args)

    # Read 360 data from file
    if sb360_path:
        with open(sb360_path) as f:
            sb360_data = json.load(f)

    # Convert 360 data into a DataFrame
    if sb360_path:
        sb360_list_teammate = []
        sb360_list_opponent = []
        for event in sb360_data:
            sb360_id = event.get('event_uuid')
            sb360_freeze_frame = event.get('freeze_frame')
            if sb360_freeze_frame:
                temp_list_teammate = []
                temp_list_opponent = []
                for frame in sb360_freeze_frame:
                    sb360_teammate = frame.get('teammate')
                    sb360_actor = frame.get('actor')
                    sb360_keeper = frame.get('keeper')
                    sb360_location = frame.get('location')
                    sb360_x = sb360_location[0] if sb360_location else None
                    sb360_y = sb360_location[1] if sb360_location else None
                    if sb360_teammate:
                        temp_list_teammate.append([sb360_teammate, sb360_actor, sb360_keeper, sb360_x, sb360_y])
                    else:
                        temp_list_opponent.append([sb360_teammate, sb360_actor, sb360_keeper, sb360_x, sb360_y])
                # Padding the list to have 11 players
                temp_list_teammate.extend([[None]*5]*(11-len(temp_list_teammate)))
                temp_list_opponent.extend([[None]*5]*(11-len(temp_list_opponent)))
                # Flatten the list, add in the sb360_id, and append to the main list
                sb360_list_teammate.append([sb360_id] + [item for sublist in temp_list_teammate for item in sublist])
                sb360_list_opponent.append([sb360_id] + [item for sublist in temp_list_opponent for item in sublist])

        # Define columns for the DataFrame
        columns_sb360 = ["sb360_id"] + ["player"+str(i)+"_"+j for i in range(1, 12) for j in ["teammate", "actor", "keeper", "x", "y"]]
        # Convert the event list to a DataFrame
        df_sb360_teammate = pd.DataFrame(sb360_list_teammate, columns=columns_sb360)
        df_sb360_opponent = pd.DataFrame(sb360_list_opponent, columns=columns_sb360)

    # Iterate through the event data and extract the required information
    if event_path:
        team_dict = {"home": event_data[0].get('team', {}).get('name'), "away": event_data[1].get('team', {}).get('name')}
        event_list = []
        for event in event_data:
            event_id = event.get('id')
            match_id = event_path.split('/')[-1].split('.')[0]
            period = event.get('period')
            time = event.get('timestamp')
            minute = event.get('minute')
            second = event.get('second')
            event_type = event.get('type', {}).get('name')
            event_type_2 = None
            end_x = end_y = None
            if event_type == "Pass":
                cross=event.get('pass', {}).get("cross")
                corner=event.get('play_pattern', {}).get('name')
                pass_type=event.get('pass', {}).get('pass_cluster_label')
                end_loc = event.get('pass', {}).get('end_location', {})
                if end_loc:
                    end_x = end_loc[0]
                    end_y = end_loc[1]
                if corner=="From Corner":
                    event_type_2="Corner"
                elif cross:
                    event_type_2="Cross"
                elif pass_type:
                    pass_type=pass_type.split("-")[-2:]
                    event_type_2="_".join(pass_type)
                    #remove space
                    event_type_2=event_type_2.replace(" ","")
            elif event_type=="Shot":
                event_type_2=event.get('shot', {}).get('outcome', {}).get('name')
            
            team = event.get('team', {}).get('name')
            player = event.get('player', {}).get('name')
            location = event.get('location')
            start_x = location[0] if location else None
            start_y = location[1] if location else None

            # Match the event id with 360 data
            if sb360_path:
                sb360_teammate = df_sb360_teammate[df_sb360_teammate["sb360_id"] == event_id]
                sb360_opponent = df_sb360_opponent[df_sb360_opponent["sb360_id"] == event_id]
                if sb360_teammate.empty or sb360_opponent.empty:
                    continue
                else:
                    if team == team_dict["home"]:
                        sb360_teammate_team = sb360_teammate
                        sb360_opponent_team = sb360_opponent
                    else:
                        sb360_teammate_team = sb360_opponent
                        sb360_opponent_team = sb360_teammate

            if sb360_path:       
                event_list.append([match_id, period, time, minute, second, event_type, event_type_2, team, player, start_x, start_y, end_x, end_y, *sb360_teammate_team.values[0][1:], *sb360_opponent_team.values[0][1:]])
                sb360_columns = ["h"+str(i)+"_"+j for i in range(1, 12) for j in ["teammate", "actor", "keeper", "x", "y"]] + ["a"+str(i)+"_"+j for i in range(1, 12) for j in ["teammate", "actor", "keeper", "x", "y"]]
                columns = ["match_id", "period", "time", "minute", "second", "event_type", "event_type_2", "team", "player", "start_x", "start_y", "end_x", "end_y"] + sb360_columns
            else:
                event_list.append([match_id, period, time, minute, second, event_type, event_type_2, team, player, start_x, start_y, end_x, end_y])
                columns = ["match_id", "period", "time", "minute", "second", "event_type", "event_type_2", "team", "player", "start_x", "start_y", "end_x", "end_y"]

        # Convert the event list to a DataFrame
        df = pd.DataFrame(event_list, columns=columns)

    elif match_id:
        event_id = event_api_data['id'].tolist()
        match_id = event_api_data['match_id'].tolist()
        period = event_api_data['period'].tolist()
        time = event_api_data['timestamp'].tolist()
        minute = event_api_data['minute'].tolist()
        second = event_api_data['second'].tolist()
        event_type = event_api_data['type'].tolist()
        team = event_api_data['team'].tolist()
        player = event_api_data['player'].tolist()
        location = event_api_data['location'].tolist()

        #pass_cross,pass_height,pass_type
        event_type_2 = [None]*len(event_type)
        end_x = [None]*len(event_type)
        end_y = [None]*len(event_type)
        for i in range(len(event_type)):
            event_type_2[i]=None
            end_x[i]=None
            end_y[i]=None
            if event_type[i] == "Pass":
                end_location=event_api_data.loc[i,'pass_end_location']
                #check if end_location is a list
                if isinstance(end_location, (list, np.ndarray)):
                    end_x[i]=end_location[0]
                    end_y[i]=end_location[1]
                pass_height=event_api_data.loc[i,'pass_height']
                cross=event_api_data.loc[i,'pass_cross']
                pass_type=event_api_data.loc[i,'pass_type']
                if pass_type=="Corner":
                    event_type_2[i]="Corner"
                #check if cross in nan
                elif cross and not np.isnan(cross):
                    event_type_2[i]="Cross"
                elif pass_height:
                    event_type_2[i]=pass_height
            elif event_type[i]=="Shot":
                event_type_2[i]=event_api_data.loc[i,'shot_outcome']

        start_x = [loc[0] if isinstance(loc, (list, np.ndarray)) else None for loc in location]
        start_y = [loc[1] if isinstance(loc, (list, np.ndarray)) else None for loc in location]
        
        columns = ["match_id", "period", "time", "minute", "second", "event_type", "event_type_2", "team", "player", "start_x", "start_y", "end_x", "end_y"]
        df = pd.DataFrame(list(zip(match_id, period, time, minute, second, event_type, event_type_2, team, player, start_x, start_y, end_x, end_y)), columns=columns)
        # Order the df by period, minute and second
        df = df.sort_values(by=["period", "minute", "second"]).reset_index(drop=True)

    return df

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
        if role == 'home':
            trackable_objects[player['trackable_object']] = {
                'name': f"{player['first_name']} {player['last_name']}",
                'team': team_dict[player['team_id']]['name'],
                'role': role,
                'id': home_count
            }
            home_count += 1
        elif role == 'away':
            trackable_objects[player['trackable_object']] = {
                'name': f"{player['first_name']} {player['last_name']}",
                'team': team_dict[player['team_id']]['name'],
                'role': role,
                'id': away_count
            }
            away_count += 1

    trackable_objects[match['ball']['trackable_object']] = {'name': 'ball', 'team': 'ball', 'role': 'ball'}

    # Process tracking data
    tracking_dict = {}
    for frame in tracking:
        time = frame['timestamp']
        if time:
            time_components = time.split(':')
            seconds = float(time_components[0]) * 3600 + float(time_components[1]) * 60 + float(time_components[2])
            period = frame['period']
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
        player = event['player']
        location = event['location']

        if isinstance(location, str):
            location = [float(x) for x in location[1:-1].split(",")]
            start_x, start_y = location[0], location[1]
        else:
            start_x = start_y = None

        time_components = time.split(':')
        seconds = float(time_components[0]) * 3600 + float(time_components[1]) * 60 + round(float(time_components[2]), 1)
        if period == 2:
            seconds += 45 * 60
        elif period == 3:
            seconds += 90 * 60
        elif period == 4:
            seconds += (90 + 15) * 60

        uid = f"{period}_{seconds}"
        tracking_data = tracking_dict.get(uid)
        home_tracking = [None] * 2 * 23
        away_tracking = [None] * 2 * 23
        
        if tracking_data:
            for obj in tracking_data:
                track_obj = trackable_objects[obj['trackable_object']]
                if track_obj['role'] == 'home':
                    home_tracking[2 * track_obj['id']] = obj['x']
                    home_tracking[2 * track_obj['id'] + 1] = obj['y']
                elif track_obj['role'] == 'away':
                    away_tracking[2 * track_obj['id']] = obj['x']
                    away_tracking[2 * track_obj['id'] + 1] = obj['y']

        df_list.append([match_id, period, time, minute, second, seconds, event_type, event_type_2, team, player, start_x, start_y, end_x, end_y, *home_tracking, *away_tracking])
    
    # Define DataFrame columns
    home_tracking_columns = []
    away_tracking_columns = []
    for i in range(1, 24):
        home_tracking_columns.extend([f"h{i}_x", f"h{i}_y"])
        away_tracking_columns.extend([f"a{i}_x", f"a{i}_y"])
    columns = ["match_id", "period", "time", "minute", "second", 'seconds', "event_type", "event_type_2", "team", "player", "start_x", "start_y","end_x","end_y"] + home_tracking_columns + away_tracking_columns

    # Sort the DataFrame by 'period', 'minute', and 'second'
    df_list = sorted(df_list, key=lambda x: (x[1], x[3], x[4]))

    # Convert the event list to a DataFrame
    df = pd.DataFrame(df_list, columns=columns)

    return df
    
def load_wyscout(event_path: str) -> pd.DataFrame:
    # Load the JSON file
    with open(event_path) as f:
        event = json.load(f)
    event_list = []
    for i in range(len(event)):
        data=event[i]
        match_id = data.get('matchId', None)
        period = data.get('matchPeriod', None)
        seconds = data.get('eventSec', None)
        event_type = data.get('eventName', None)
        tag=data.get('tags', None)
        accurate=False
        for id_i in tag:
            if id_i['id']==1801:
                accurate=True
                break
        if event_type == "Shot":
            tag=data.get('tags', None)
            for id_i in tag:
                if id_i["id"]==102:
                    event_type_2="Own_goal"
                    break
                elif id_i["id"]==101:
                    event_type_2="Goal"
                    break
                else:
                    event_type_2 = data.get('subEventName', None)
        elif event_type == "Duel":
            tag=data.get('tags', None)
            for id_i in tag:
                if id_i["id"] in [501,502,503,504]:
                    event_type_2="Ground attacking duel_off dribble"
                    break
                else:
                    event_type_2 = data.get('subEventName', None)
        elif event_type == "Others on the ball":
            event_type_2 = data.get('subEventName', None)
            if event_type_2 == "Touch":
                #check if tags is an empty list
                tag=data.get('tags', None)
                if len(tag)==0:
                    event_type_2="Touch_good" 
            elif event_type_2 == "own-goal":
                event_type_2="Own_goal"   
        elif event_type == "Pass":
            event_type_2 = data.get('subEventName', None)
            if event_type_2 == "own-goal":
                event_type_2="Own_goal"
        else:
            event_type_2 = data.get('subEventName', None)
        team = data.get('teamId', None)
        player = data.get('playerId', None)
        start_pos = data.get('positions')
        if start_pos and len(start_pos) > 1:
            start_x = start_pos[0].get('x', None)
            start_y = start_pos[0].get('y', None)
            end_x = start_pos[1].get('x', None)
            end_y = start_pos[1].get('y', None)
        elif start_pos and len(start_pos) == 1:
            start_x = start_pos[0].get('x', None)
            start_y = start_pos[0].get('y', None)
            end_x = None
            end_y = None
        else:
            start_x = None
            start_y = None
            end_x = None
            end_y = None

        # Append event details to the list
        event_list.append([match_id, period, seconds, event_type, event_type_2, accurate, team, player, start_x, start_y, end_x, end_y])
    # Convert the event list to a DataFrame
    columns = ["match_id", "period", "seconds", "event_type", "event_type_2","accurate", "team", "player", "start_x", "start_y", "end_x", "end_y"]
    df = pd.DataFrame(event_list, columns=columns)

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

    #test load_statsbomb with api data
    # statsbomb_df=load_statsbomb(match_id=3795108)
    # statsbomb_df.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_api_data.csv",index=False)

    #test load_statsbomb_skillcorner
    # statsbomb_skillcorner_df=load_statsbomb_skillcorner(statsbomb_skillcorner_event_path,statsbomb_skillcorner_tracking_path,
    #                                                     statsbomb_skillcorner_match_path,3894907,1553748)
    # statsbomb_skillcorner_df.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb_skillcorner/test_data.csv",index=False)

    #test load_wyscout
    wyscout_df=load_wyscout(wyscout_event_path)
    wyscout_df.head(1000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_data.csv",index=False)


    print("----------------done-----------------")
    pdb.set_trace()
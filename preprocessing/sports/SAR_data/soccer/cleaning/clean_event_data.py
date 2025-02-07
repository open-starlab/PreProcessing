import warnings
from typing import Dict, List, Tuple

import pandas as pd

from preprocessing.sports.SAR_data.soccer.constant import FIELD_LENGTH, FIELD_WIDTH, PLAYER_ROLE_MAP

warnings.filterwarnings('ignore')


def get_changed_player_list(event_data: pd.DataFrame, league: str) -> Tuple[List[int], List[int]]:
    """
    This function takes in a DataFrame of event data and returns two lists of players who have changed.
    The first list contains the players who have changed in the home team
    and the second list contains the players who have changed in the away team.

    Parameters:
    event_data (pd.DataFrame): DataFrame containing event data

    Returns:
    Tuple[List[int], List[int]]:
        Tuple containing two lists of players who have changed in the home and away teams respectively
    """
    if league == 'jleague':
        changed_player_list_in_home = list(
            event_data.query("event_name == '交代' and home_away == 'HOME'")["jersey_number"].values.astype(int)
        )
        changed_player_list_in_away = list(
            event_data.query("event_name == '交代' and home_away == 'AWAY'")["jersey_number"].values.astype(int)
        )
    elif league == 'laliga':
        changed_player_list_in_home = list(
            event_data.query("event_name == 'Substitution' and home_away == 'HOME'")["jersey_number"].values.astype(int)
        )
        changed_player_list_in_away = list(
            event_data.query("event_name == 'Substitution' and home_away == 'AWAY'")["jersey_number"].values.astype(int)
        )
    return changed_player_list_in_home, changed_player_list_in_away



def get_timestamp(event_data: pd.DataFrame, league: str) -> Dict[str, int]:
    """
    This function takes in a DataFrame of event data and returns a dictionary of timestamps.
    The dictionary contains the start and end frames of the first and second halves of the game.

    Parameters:
    event_data (pd.DataFrame): DataFrame containing event data

    Returns:
    Dict[str, int]: Dictionary containing the start and end frames of the first and second halves of the game
    """
    
    if league == 'jleague':
        timestamp_dict = {
            "first_start_frame": event_data.loc[event_data["event_name"] == "前半開始", "frame_id"].values[0],
            "first_end_frame": event_data.loc[event_data["event_name"] == "前半終了", "frame_id"].values[0],
            "second_start_frame": event_data.loc[event_data["event_name"] == "後半開始", "frame_id"].values[0],
            "second_end_frame": event_data.loc[event_data["event_name"] == "後半終了", "frame_id"].values[0],
        }
    elif league == 'laliga':
        try:
            timestamp_dict = {
                "first_start_frame": event_data.loc[event_data["event_name"] == "Half Start 1", "frame_id"].values[0],
                "first_end_frame": event_data.loc[event_data["event_name"] == "Half End 1", "frame_id"].values[0],
                "second_start_frame": event_data.loc[event_data["event_name"] == "Half Start 2", "frame_id"].values[0],
                "second_end_frame": event_data.loc[event_data["event_name"] == "Half End 2", "frame_id"].values[0],
            }
        except:
            timestamp_dict = {
                "first_start_frame": event_data.loc[event_data["event_name"] == "Half Start 1", "frame_id"].values[0],
                "first_end_frame": event_data.loc[event_data["event_name"] == "Half End 1", "frame_id"].values[0],
                "second_start_frame": event_data.loc[event_data["event_name"] == "Half Start 2", "frame_id"].values[0],
                "second_end_frame": event_data.iloc[-1]['frame_id']+1,
            }
    return timestamp_dict


def clean_event_data(
    event_data: pd.DataFrame,
    event_priority: List[str],
    first_start_frame: int,
    first_end_frame: int,
    second_start_frame: int,
    second_end_frame: int,
    original_sampling_rate: int = 25,
) -> pd.DataFrame:
    """
    This function cleans the event data by removing unnecessary events, adding new columns,
    and renaming existing columns.
    It also rounds off the time from the start of each half to the nearest tenth of a second.
    Finally, duplicate events are removed so that only one event exists per frame and time.
    The event that is kept is the one with the highest priority.

    Parameters:
    event_data (pd.DataFrame): DataFrame containing event data
    event_priority (List[str]): List of events in order of priority
    first_start_frame (int): Frame number at the start of the first half
    first_end_frame (int): Frame number at the end of the first half
    second_start_frame (int): Frame number at the start of the second half
    second_end_frame (int): Frame number at the end of the second half

    Returns:
    pd.DataFrame: Cleaned event data
    """

    # remove unnecessary events
    event_data = event_data.query("event_name not in ['Half Start 1', 'Half Start 2', 'Half End 1', 'Half End 2']")
    event_data = event_data.query(
        f'''
        {first_start_frame} <= frame_id <= {first_end_frame} \
        or {second_start_frame} <= frame_id <= {second_end_frame}
        '''
    )

    # time
    event_data['time_from_half_start'] = 0
    event_data.loc[event_data['half'] == "first", 'time_from_half_start'] = (
        event_data.loc[event_data['half'] == "first", 'frame_id'] - first_start_frame
    ) / original_sampling_rate
    event_data.loc[event_data['half'] == "second", 'time_from_half_start'] = (
        event_data.loc[event_data['half'] == "second", 'frame_id'] - second_start_frame
    ) / original_sampling_rate
    event_data['time_from_half_start'] = event_data['time_from_half_start'].round(1)

    # player role
    event_data['player_role'] = event_data['player_role_id'].map(
        lambda x: PLAYER_ROLE_MAP[x] if x in PLAYER_ROLE_MAP else 'nan'
    )
    event_data.drop(columns=['player_role_id'], inplace=True)

    # remove some events so the only one event exists per frame and time
    event_data = event_data.dropna(subset=['event_name'])
    event_data['event_name'] = pd.Categorical(event_data['event_name'], categories=event_priority, ordered=True)
    event_data = (
        event_data.sort_values('event_name')
        .groupby("time_from_half_start", as_index=False)
        .first()
        .sort_values('time_from_half_start')
        .reset_index(drop=True)
    )
    event_data = (
        event_data.sort_values('event_name')
        .groupby("frame_id", as_index=False)
        .first()
        .sort_values('frame_id')
        .reset_index(drop=True)
    )
    assert event_data['frame_id'].nunique() == len(
        event_data
    ), f" {event_data['frame_id'].nunique()} != {len(event_data)}"
    assert event_data['time_from_half_start'].nunique() == len(
        event_data
    ), f" {event_data['time_from_half_start'].nunique()} != {len(event_data)}"
    return event_data


def preprocess_coordinates_in_event_data(
    event_data: pd.DataFrame, origin_pos: str = 'center', absolute_coordinates: bool = True, league: str = 'jleague'
) -> pd.DataFrame:
    """
    This function preprocesses the coordinates in the event data.
    It converts the coordinates to meters and adjusts them based on the origin position
    and whether absolute coordinates are used.

    Parameters:
    event_data (pd.DataFrame): DataFrame containing event data
    origin_pos (str, optional): The origin position for the coordinates. Defaults to 'center'.
    absolute_coordinates (bool, optional): Whether to use absolute coordinates. Defaults to True.

    Returns:
    pd.DataFrame: DataFrame with preprocessed coordinates
    """

    # common: convert coordinates to meters
    if league == 'jleague':
        event_data[['ball_x', 'ball_y']] = event_data[['ball_x', 'ball_y']] / 3
    elif league == 'laliga':
        event_data[['ball_x', 'ball_y']] = event_data[['ball_x', 'ball_y']]
    else:
        raise ValueError("league must be 'jleague' or 'laliga'")

    if origin_pos == "center":
        event_data['event_x'] = event_data['event_x'] - FIELD_LENGTH / 2
        event_data['event_y'] = event_data['event_y'] - FIELD_WIDTH / 2
    elif origin_pos == "top_left":
        event_data['ball_x'] = event_data['ball_x'] + FIELD_LENGTH / 2
        event_data['ball_y'] = event_data['ball_y'] + FIELD_WIDTH / 2
    else:
        raise ValueError("origin_pos must be 'center' or 'top_left'")

    if absolute_coordinates:
        event_data.loc[event_data['attack_direction'] == 2, 'event_x'] = -event_data.loc[
            event_data['attack_direction'] == 2, 'event_x'
        ]
        event_data.loc[event_data['attack_direction'] == 2, 'event_y'] = -event_data.loc[
            event_data['attack_direction'] == 2, 'event_y'
        ]
    else:
        event_data.loc[event_data['attack_direction'] == 2, 'ball_x'] = -event_data.loc[
            event_data['attack_direction'] == 2, 'ball_x'
        ]
        event_data.loc[event_data['attack_direction'] == 2, 'ball_y'] = -event_data.loc[
            event_data['attack_direction'] == 2, 'ball_y'
        ]
    return event_data

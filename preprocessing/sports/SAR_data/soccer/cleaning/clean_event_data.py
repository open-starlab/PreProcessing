import warnings
from typing import Dict, List, Tuple

import pandas as pd

from preprocessing.sports.SAR_data.soccer.constant import FIELD_LENGTH, FIELD_WIDTH, PLAYER_ROLE_MAP

warnings.filterwarnings("ignore")


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
    # Fill missing jersey_number for substitution events using player_name within same team.
    sub_event_name = {"jleague": "交代", "fifawc": "交代", "laliga": "Substitution"}.get(league)
    assert sub_event_name is not None, f"Unsupported league: {league}"

    sub_mask = event_data["event_name"].eq(sub_event_name)
    missing_mask = (
        sub_mask & event_data["jersey_number"].isna() & event_data["player_name"].notna() & event_data["home_away"].notna()
    )
    if missing_mask.any():
        known = event_data.loc[
            event_data["jersey_number"].notna() & event_data["player_name"].notna() & event_data["home_away"].notna(),
            ["home_away", "player_name", "jersey_number"],
        ]
        if not known.empty:
            jersey_map = known.groupby(["home_away", "player_name"], sort=False)["jersey_number"].agg(
                lambda s: s.value_counts().idxmax()
            )
            keys = pd.MultiIndex.from_frame(event_data.loc[missing_mask, ["home_away", "player_name"]])
            event_data.loc[missing_mask, "jersey_number"] = keys.map(jersey_map).to_numpy()

    sub_df = event_data.loc[sub_mask, ["home_away", "jersey_number"]]
    changed_player_list_in_home = sub_df.loc[sub_df["home_away"].eq("HOME"), "jersey_number"].dropna().astype(int).tolist()
    changed_player_list_in_away = sub_df.loc[sub_df["home_away"].eq("AWAY"), "jersey_number"].dropna().astype(int).tolist()
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

    if league == "jleague" or league == "fifawc":
        timestamp_dict = {
            "first_start_frame": event_data.loc[event_data["event_name"] == "前半開始", "frame_id"].values[0],
            "first_end_frame": event_data.loc[event_data["event_name"] == "前半終了", "frame_id"].values[0],
            "second_start_frame": event_data.loc[event_data["event_name"] == "後半開始", "frame_id"].values[0],
            "second_end_frame": event_data.loc[event_data["event_name"] == "後半終了", "frame_id"].values[0],
        }
    elif league == "laliga":
        try:
            timestamp_dict = {
                "first_start_frame": event_data.loc[event_data["event_name"] == "Half Start 1", "frame_id"].values[0],
                "first_end_frame": event_data.loc[event_data["event_name"] == "Half End 1", "frame_id"].values[0],
                "second_start_frame": event_data.loc[event_data["event_name"] == "Half Start 2", "frame_id"].values[0],
                "second_end_frame": event_data.loc[event_data["event_name"] == "Half End 2", "frame_id"].values[0],
            }
        except IndexError:
            timestamp_dict = {
                "first_start_frame": event_data.loc[event_data["event_name"] == "Half Start 1", "frame_id"].values[0],
                "first_end_frame": event_data.loc[event_data["event_name"] == "Half End 1", "frame_id"].values[0],
                "second_start_frame": event_data.loc[event_data["event_name"] == "Half Start 2", "frame_id"].values[0],
                "second_end_frame": event_data.iloc[-1]["frame_id"] + 1,
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
        f"""
        {first_start_frame} <= frame_id <= {first_end_frame} \
        or {second_start_frame} <= frame_id <= {second_end_frame}
        """
    )

    # time
    event_data["time_from_half_start"] = 0
    event_data.loc[event_data["half"] == "first", "time_from_half_start"] = (
        event_data.loc[event_data["half"] == "first", "frame_id"] - first_start_frame
    ) / original_sampling_rate
    event_data.loc[event_data["half"] == "second", "time_from_half_start"] = (
        event_data.loc[event_data["half"] == "second", "frame_id"] - second_start_frame
    ) / original_sampling_rate
    event_data["time_from_half_start"] = event_data["time_from_half_start"].round(1)

    # player role
    event_data["player_role"] = event_data["player_role_id"].map(
        lambda x: PLAYER_ROLE_MAP[x] if x in PLAYER_ROLE_MAP else "nan"
    )
    event_data.drop(columns=["player_role_id"], inplace=True)

    # remove some events so the only one event exists per frame and time
    event_data = event_data.dropna(subset=["event_name"])
    event_data["event_name"] = pd.Categorical(event_data["event_name"], categories=event_priority, ordered=True)
    event_data = (
        event_data.sort_values("event_name")
        .groupby("time_from_half_start", as_index=False)
        .first()
        .sort_values("time_from_half_start")
        .reset_index(drop=True)
    )
    event_data = (
        event_data.sort_values("event_name")
        .groupby("frame_id", as_index=False)
        .first()
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    assert event_data["frame_id"].nunique() == len(event_data), f" {event_data['frame_id'].nunique()} != {len(event_data)}"
    assert event_data["time_from_half_start"].nunique() == len(event_data), (
        f" {event_data['time_from_half_start'].nunique()} != {len(event_data)}"
    )
    return event_data


def preprocess_coordinates_in_event_data(
    event_data: pd.DataFrame, origin_pos: str = "center", absolute_coordinates: bool = True, league: str = "jleague"
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
    if league == "jleague":
        event_data[["ball_x", "ball_y"]] = event_data[["ball_x", "ball_y"]] / 3
    elif league == "laliga" or league == "fifawc":
        event_data[["ball_x", "ball_y"]] = event_data[["ball_x", "ball_y"]]
    else:
        raise ValueError("league must be 'jleague' or 'laliga'")

    if origin_pos == "center":
        event_data["event_x"] = event_data["event_x"] - FIELD_LENGTH / 2
        event_data["event_y"] = event_data["event_y"] - FIELD_WIDTH / 2
    elif origin_pos == "top_left":
        event_data["ball_x"] = event_data["ball_x"] + FIELD_LENGTH / 2
        event_data["ball_y"] = event_data["ball_y"] + FIELD_WIDTH / 2
    else:
        raise ValueError("origin_pos must be 'center' or 'top_left'")

    if absolute_coordinates:
        event_data.loc[event_data["attack_direction"] == 2, "event_x"] = -event_data.loc[
            event_data["attack_direction"] == 2, "event_x"
        ]
        event_data.loc[event_data["attack_direction"] == 2, "event_y"] = -event_data.loc[
            event_data["attack_direction"] == 2, "event_y"
        ]
    else:
        event_data.loc[event_data["attack_direction"] == 2, "ball_x"] = -event_data.loc[
            event_data["attack_direction"] == 2, "ball_x"
        ]
        event_data.loc[event_data["attack_direction"] == 2, "ball_y"] = -event_data.loc[
            event_data["attack_direction"] == 2, "ball_y"
        ]
    return event_data


def apply_event_name_mapping(event_data: pd.DataFrame, event_mapping: dict) -> pd.DataFrame:
    """
    Apply event name mapping to event data based on configuration

    Parameters
    ----------
    event_data : pd.DataFrame
        Event data with 'event_name' column
    event_mapping : dict
        Event name mapping configuration

    Returns
    -------
    pd.DataFrame
        Event data with updated event names
    """
    event_data = event_data.copy()

    for _, mapping_config in event_mapping.items():
        target_name = mapping_config["target_name"]
        source_names = mapping_config["source_names"]

        # Apply mapping
        mask = event_data["event_name"].isin(source_names)
        event_data.loc[mask, "event_name"] = target_name

    return event_data

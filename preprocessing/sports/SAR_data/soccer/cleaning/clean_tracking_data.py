import json
import logging
from copy import deepcopy
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from preprocessing.sports.SAR_data.soccer.constant import FIELD_LENGTH, FIELD_WIDTH, HOME_AWAY_MAP

getcontext().prec = 28  # Set high precision for Decimal calculations
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def safe_json_parse(data):
    """
    Safely parse JSON data, handling both string and object types.

    Parameters:
    data: JSON string or already parsed object

    Returns:
    Parsed object or original data if already parsed
    """
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {data}")
            return data
    return data


def safe_get_player_info(player_dict, home_away, jersey_number, key, default=None):
    """
    Safely get player information from player dictionary.

    Parameters:
    player_dict: Dictionary containing player information
    home_away: 'HOME' or 'AWAY'
    jersey_number: Player's jersey number
    key: Information key to retrieve
    default: Default value if key not found

    Returns:
    Player information or default value
    """
    try:
        player_key = (home_away, jersey_number)
        if player_key in player_dict:
            return player_dict[player_key].get(key, default)
        else:
            logger.warning(f"Player ({home_away}, {jersey_number}) not found in player_dict")
            return default
    except (KeyError, TypeError, AttributeError) as e:
        logger.warning(f"Error accessing player info for ({home_away}, {jersey_number}): {e}")
        return default


def generate_time_sequence(start_time, end_time, sampling_rate):
    """
    Generate high-precision time sequence using Decimal arithmetic.

    Parameters:
    start_time (float): Start time in seconds
    end_time (float): End time in seconds
    sampling_rate (int): Sampling rate (frames per second)

    Returns:
    List[float]: List of time values with 0.1 second precision
    """
    step = Decimal("1") / Decimal(str(sampling_rate))
    current = Decimal(str(start_time)).quantize(Decimal("0.1"))
    end_decimal = Decimal(str(end_time)).quantize(Decimal("0.1"))

    times = []
    while current <= end_decimal:
        times.append(float(current))
        current = (current + step).quantize(Decimal("0.1"))

    return times


def remove_temporal_duplicates(df):
    """
    Remove temporal duplicates with enhanced validation.

    Parameters:
    df (pd.DataFrame): DataFrame with potential temporal duplicates

    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    initial_count = len(df)

    # Remove duplicates based on time_from_half_start and half, keeping the last occurrence
    df_clean = (
        df.drop_duplicates(subset=["time_from_half_start", "half"], keep="last")
        .sort_values(["half", "time_from_half_start"])
        .reset_index(drop=True)
    )

    final_count = len(df_clean)

    if initial_count > final_count:
        logger.info(f"Removed {initial_count - final_count} temporal duplicates from tracking data")

    return df_clean


def validate_temporal_consistency(df):
    """
    Validate temporal consistency of tracking data.

    Parameters:
    df (pd.DataFrame): DataFrame to validate

    Returns:
    bool: True if data is temporally consistent
    """
    # Check for duplicate timestamps
    duplicates = df.duplicated(["time_from_half_start", "half"]).sum()

    if duplicates > 0:
        logger.warning(f"Found {duplicates} temporal duplicates")
        return False

    # Check for temporal continuity within each half
    for half in df["half"].unique():
        half_data = df[df["half"] == half].sort_values("time_from_half_start")
        if len(half_data) > 1:
            time_diffs = half_data["time_from_half_start"].diff().dropna()
            expected_diff = 0.1  # Expected 0.1 second intervals for 10Hz

            # Allow larger tolerance for real-world data irregularities
            irregular_intervals = abs(time_diffs - expected_diff) > 0.05
            irregular_count = irregular_intervals.sum()

            if irregular_count > 0:
                # Only warn if irregularities are frequent (>1% of data)
                if irregular_count / len(time_diffs) > 0.01:
                    logger.warning(
                        f"Significant irregular time intervals found in {half} half: {irregular_count}/{len(time_diffs)} intervals"
                    )
                    return False
                else:
                    logger.info(
                        f"Minor irregular time intervals found in {half} half: {irregular_count}/{len(time_diffs)} intervals (acceptable)"
                    )

    return True


def update_player_dict_with_substitutions(player_dict, player_change_list):
    """
    Update player dictionary with substitution information.

    Parameters:
    player_dict: Dictionary containing player information
    player_change_list: List of player changes/substitutions

    Returns:
    Updated player dictionary
    """
    updated_dict = player_dict.copy()

    for change in player_change_list:
        if "player_change_info" in change:
            for change_info in change["player_change_info"]:
                try:
                    home_away = change_info["home_away"]
                    player_in = change_info["player_in"]
                    player_out = change_info["player_out"]

                    # Copy information from outgoing player to incoming player
                    if (home_away, player_out) in updated_dict:
                        player_info = updated_dict[(home_away, player_out)].copy()
                        player_info["jersey_number"] = player_in
                        updated_dict[(home_away, player_in)] = player_info
                        logger.info(f"Updated player_dict: {home_away} player {player_out} -> {player_in}")
                    else:
                        # Create basic info for incoming player if outgoing player not found
                        updated_dict[(home_away, player_in)] = {
                            "jersey_number": player_in,
                            "player_name": f"Player_{player_in}",
                            "player_id": player_in,
                            "player_role": "Field Player",
                            "height": None,
                        }
                        logger.warning(f"Created basic info for substitute player {home_away} #{player_in}")

                except (KeyError, TypeError) as e:
                    logger.warning(f"Error processing substitution: {e}")
                    continue

    return updated_dict


def complement_tracking_ball_with_event_data(
    tracking_ball: pd.DataFrame, event_data: pd.DataFrame, first_end_frame: int, league: str
) -> pd.DataFrame:
    """
    This function complements the tracking ball data with event data.
    It merges the two dataframes on the 'frame_id' column.

    Parameters:
    tracking_ball (pd.DataFrame): DataFrame containing tracking ball data
    event_data (pd.DataFrame): DataFrame containing event data

    Returns:
    pd.DataFrame: DataFrame with complemented tracking ball data
    """
    complemented_data = (
        pd.merge(tracking_ball, event_data[["frame_id", "ball_x", "ball_y"]], on="frame_id", how="outer")
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    complemented_data["game_id"] = event_data["game_id"].iloc[0]

    if league == "jleague" or league == "fifawc":
        complemented_data["half"] = complemented_data["half"].fillna(method="ffill").fillna(method="bfill").astype(str)
    elif league == "laliga" or league == "soccernet":
        complemented_data["half"] = complemented_data["frame_id"].apply(
            lambda x: "first" if x <= first_end_frame else "second"
        )

    complemented_data["home_away"] = "BALL"
    complemented_data["jersey_number"] = 0
    complemented_data["x"] = complemented_data["x"].fillna(complemented_data["ball_x"])
    complemented_data["y"] = complemented_data["y"].fillna(complemented_data["ball_y"])
    complemented_data = complemented_data.drop(columns=["ball_x", "ball_y"])

    return complemented_data


def interpolate_ball_tracking_data(
    tracking_ball: pd.DataFrame,
    event_data: pd.DataFrame,
    ignored_events: List[str] = [
        "交代",
        "警告(イエロー)",
        "退場(レッド)",
    ],
) -> pd.DataFrame:
    """
    This function interpolates the tracking ball data.
    It first gets the valid series boundaries, ignoring some events.
    Then, it interpolates the data for each valid series.
    Finally, it concatenates all the interpolated data and returns it.

    Parameters:
    tracking_ball (pd.DataFrame): DataFrame containing tracking ball data
    event_data (pd.DataFrame): DataFrame containing event data
    ignored_events (List[str], optional): List of events to be ignored. Defaults to ['交代', '警告(イエロー)', '退場(レッド)'].

    Returns:
    pd.DataFrame: DataFrame with interpolated tracking ball data
    """

    # get valid series boundaries, ignoring some events
    valid_series_num = list(  # noqa: F841
        event_data["series_num"].value_counts()[event_data["series_num"].value_counts() != 1].index
    )
    valid_series_boundaries = (
        event_data.query("event_name not in @ignored_events")
        .query("series_num in @valid_series_num")
        .groupby("series_num")
        .agg({"frame_id": ["min", "max"]})["frame_id"]
        .reset_index()
    )
    valid_series_boundaries.columns = ["series_num", "min_frame_id", "max_frame_id"]

    interpolated_data_list = []
    for _, item in valid_series_boundaries.iterrows():
        start_frame = item["min_frame_id"]
        end_frame = item["max_frame_id"]
        new_index = pd.DataFrame({"frame_id": range(int(start_frame), int(end_frame) + 1)})
        interpolated_data = pd.merge(new_index, tracking_ball, on="frame_id", how="left")
        interpolated_data[["x", "y"]] = interpolated_data[["x", "y"]].interpolate(method="linear", limit_direction="both")
        if interpolated_data["x"].isnull().sum() > 0:
            print(f"Skip {item['series_num']} for lack of tracking data")
            continue

        try:
            interpolated_data["game_id"] = tracking_ball.query("@start_frame <= frame_id <= @end_frame")["game_id"].iloc[0]
        except Exception as e:
            print(f"Skip {item['series_num']} for lack of tracking data")
            print(f"Error: {e}")
            continue
        interpolated_data["half"] = tracking_ball.query("@start_frame <= frame_id <= @end_frame")["half"].iloc[0]
        interpolated_data["home_away"] = "BALL"
        interpolated_data["jersey_number"] = 0
        interpolated_data_list.append(interpolated_data)

    interpolated_tracking_ball = (
        pd.concat(interpolated_data_list)
        .sort_values("frame_id")
        .reset_index(drop=True)[
            [
                "game_id",
                "frame_id",
                "half",
                "home_away",
                "jersey_number",
                "x",
                "y",
            ]
        ]
    )

    # If there are rows with the same frame_id
    interpolated_tracking_ball = interpolated_tracking_ball.drop_duplicates(subset=["frame_id"], keep="last")
    # interpolated_tracking_ball = interpolated_tracking_ball[~interpolated_tracking_ball.duplicated(subset=['frame_id', 'x', 'y'], keep='first')]

    if interpolated_tracking_ball["frame_id"].nunique() != len(interpolated_tracking_ball):
        print("interpolated_tracking_ball:", interpolated_tracking_ball)
        print("unique frame_ids:", interpolated_tracking_ball["frame_id"].nunique())
        print("length of interpolated_tracking_ball:", len(interpolated_tracking_ball))

        # Extract columns with duplicate frame_id
        duplicated_frame_id = interpolated_tracking_ball[
            interpolated_tracking_ball.duplicated(subset=["frame_id"], keep=False)
        ]
        print("duplicated_frame_id:", duplicated_frame_id)
        import pdb

        pdb.set_trace()

    assert interpolated_tracking_ball["frame_id"].nunique() == len(interpolated_tracking_ball)
    return interpolated_tracking_ball


def clean_tracking_data(tracking_data: pd.DataFrame, first_end_frame: int) -> pd.DataFrame:
    """
    This function renames the columns in the tracking data.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    first_end_frame (int): Frame number at the end of the first half

    Returns:
    pd.DataFrame: DataFrame with renamed columns
    """
    tracking_data["half"] = tracking_data["frame_id"].apply(lambda x: "first" if x <= first_end_frame else "second")
    tracking_data["home_away"] = tracking_data["home_away"].apply(lambda x: HOME_AWAY_MAP[x])
    tracking_data = tracking_data[["game_id", "frame_id", "half", "home_away", "jersey_number", "x", "y"]]
    return tracking_data


def merge_tracking_data(tracking_player: pd.DataFrame, tracking_ball: pd.DataFrame) -> pd.DataFrame:
    """
    This function merges the tracking player and tracking ball dataframes.

    Parameters:
    tracking_player (pd.DataFrame): DataFrame containing tracking player data
    tracking_ball (pd.DataFrame): DataFrame containing tracking ball data

    Returns:
    pd.DataFrame: DataFrame with merged tracking data
    """
    in_play_frame_num = tracking_ball["frame_id"].unique()  # noqa:  F841
    tracking_player = tracking_player.query("frame_id in @in_play_frame_num")
    assert tracking_ball[["x", "y"]].isnull().sum().sum() == 0

    tracking_data = (
        pd.concat([tracking_player, tracking_ball], axis=0)
        .sort_values(by=["frame_id", "home_away", "jersey_number"])
        .reset_index(drop=True)
    )[
        [
            "game_id",
            "frame_id",
            "half",
            "home_away",
            "jersey_number",
            "x",
            "y",
        ]
    ]
    return tracking_data


def cut_frames_out_of_game(
    tracking_data: pd.DataFrame,
    first_start_frame: int,
    first_end_frame: int,
    second_start_frame: int,
    second_end_frame: int,
) -> pd.DataFrame:
    """
    This function cuts out frames that are not in play.
    """
    tracking_data = tracking_data.query(
        "(@first_start_frame <= frame_id <= @first_end_frame) | (@second_start_frame <= frame_id <= @second_end_frame)"
    ).reset_index(drop=True)
    return tracking_data


def preprocess_coordinates_in_tracking_data(
    tracking_data: pd.DataFrame,
    event_data: pd.DataFrame,
    origin_pos: str = "center",
    absolute_coordinates: bool = True,
    league: str = "jleague",
) -> pd.DataFrame:
    """
    This function preprocesses the coordinates in the tracking data.
    It converts the coordinates to meters and adjusts them based on the origin position
    and whether absolute coordinates are used.
    Event data is used to determine the attack direction.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    event_data (pd.DataFrame): DataFrame containing event data
    origin_pos (str, optional): The origin position for the coordinates. Defaults to 'center'.
    absolute_coordinates (bool, optional): Whether to use absolute coordinates. Defaults to True.

    Returns:
    pd.DataFrame: DataFrame with preprocessed coordinates
    """

    def _convert_coordinate(
        tracking_data: pd.DataFrame, origin_pos: str, absolute_coordinates: bool, league: str
    ) -> pd.DataFrame:
        if league == "jleague":
            tracking_data.loc[:, "x"] = tracking_data["x"].map(lambda x: x / 100)
            tracking_data.loc[:, "y"] = tracking_data["y"].map(lambda x: -x / 100)
        elif league == "laliga" or league == "soccernet" or league == "fifawc":
            tracking_data.loc[:, "x"] = tracking_data["x"].map(lambda x: x)
            tracking_data.loc[:, "y"] = tracking_data["y"].map(lambda x: -x)

        if origin_pos == "top_left":
            tracking_data.loc[:, "x"] = tracking_data["x"] + 52.5
            tracking_data.loc[:, "y"] = tracking_data["y"] + 34.0
        elif origin_pos != "center":
            raise ValueError("origin_pos must be 'center' or 'bottom_left'")

        if absolute_coordinates is False:
            tracking_data.loc[tracking_data["attack_direction"] != 1, "x"] = -tracking_data.loc[
                tracking_data["attack_direction"] != 1, "x"
            ]
            tracking_data.loc[tracking_data["attack_direction"] != 1, "y"] = -tracking_data.loc[
                tracking_data["attack_direction"] != 1, "y"
            ]

        # fix padding
        if origin_pos == "center":
            tracking_data.loc[tracking_data["jersey_number"] <= -1, "y"] = 0.0
            if absolute_coordinates:
                tracking_data.loc[
                    (tracking_data["jersey_number"] <= -1) & (tracking_data["attack_direction"] == 1),
                    "x",
                ] = round(-FIELD_LENGTH / 2, 2)
                tracking_data.loc[
                    (tracking_data["jersey_number"] <= -1) & (tracking_data["attack_direction"] != 1),
                    "x",
                ] = round(FIELD_LENGTH / 2, 2)
            else:
                tracking_data.loc[tracking_data["jersey_number"] <= -1, "x"] = round(-FIELD_WIDTH / 2, 2)

        elif origin_pos == "top_left":
            tracking_data.loc[tracking_data["jersey_number"] <= -1, "y"] = round(FIELD_WIDTH / 2, 2)
            if absolute_coordinates:
                tracking_data.loc[
                    (tracking_data["jersey_number"] <= -1) & (tracking_data["attack_direction"] == 1),
                    "x",
                ] = 0.0
                tracking_data.loc[
                    (tracking_data["jersey_number"] <= -1) & (tracking_data["attack_direction"] != 1),
                    "x",
                ] = round(FIELD_LENGTH, 2)
            else:
                tracking_data.loc[tracking_data["jersey_number"] <= -1, "x"] = 0.0
        else:
            raise ValueError("origin_pos must be 'center' or 'top_left'")

        # clip (use np.clip ?)
        if origin_pos == "center":
            tracking_data.loc[:, "x"] = tracking_data["x"].clip(-FIELD_LENGTH / 2, FIELD_LENGTH / 2)
            tracking_data.loc[:, "y"] = tracking_data["y"].clip(-FIELD_WIDTH / 2, FIELD_WIDTH / 2)
        elif origin_pos == "top_left":
            tracking_data.loc[:, "x"] = tracking_data["x"].clip(0, FIELD_LENGTH)
            tracking_data.loc[:, "y"] = tracking_data["y"].clip(0, FIELD_WIDTH)
        else:
            raise ValueError("origin_pos must be 'center' or 'top_left'")

        return tracking_data

    tracking_data = pd.merge(
        tracking_data,
        event_data[
            [
                "half",
                "time_from_half_start",
                "attack_direction",
                "attack_start_history_num",
            ]
        ],
        on=["half", "time_from_half_start"],
        how="left",
    )
    # decide attack direction by the majority vote within the same attack_start_history_num
    tracking_data[["attack_direction", "attack_start_history_num"]] = (
        tracking_data[["attack_direction", "attack_start_history_num"]]
        .fillna(method="ffill")
        .fillna(method="bfill")
        .astype(int)
    )
    tracking_data.loc[:, "attack_direction"] = tracking_data.groupby(["half", "attack_start_history_num"])[
        "attack_direction"
    ].transform(lambda x: x.value_counts().index[0])
    tracking_data = _convert_coordinate(tracking_data, origin_pos, absolute_coordinates, league)
    return tracking_data.sort_values(by=["half", "time_from_half_start", "home_away", "jersey_number"])[
        [
            "game_id",
            "half",
            "series_num",
            "attack_direction",
            "time_from_half_start",
            "home_away",
            "jersey_number",
            "x",
            "y",
        ]
    ].reset_index(drop=True)


def get_player_change_log(
    tracking_data: pd.DataFrame,
    player_data: pd.DataFrame,
    changed_player_list_in_home: List[int],
    changed_player_list_in_away: List[int],
) -> List[Dict[str, Any]]:
    """
    Optimized function to get the player change log using vectorized operations.

    This function efficiently detects player substitutions by:
    1. Pre-computing frame-wise player compositions using vectorized pandas operations
    2. Detecting composition changes only when they actually occur
    3. Processing only frames where substitutions happen

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    player_data (pd.DataFrame): DataFrame containing player data
    changed_player_list_in_home (List[int]): List of players who have changed in the home team
    changed_player_list_in_away (List[int]): List of players who have changed in the away team

    Returns:
    List[Dict[str, Any]]: List of dictionaries containing the frame number and the players who have changed
    """
    logger.info("Starting optimized player change detection...")

    # Get initial starting players
    player_ever_on_pitch_home: Set[int] = set(
        player_data.query("home_away == 'HOME'").query("starting_member == 1")["jersey_number"].astype(int).values
    )
    player_ever_on_pitch_away: Set[int] = set(
        player_data.query("home_away == 'AWAY'").query("starting_member == 1")["jersey_number"].astype(int).values
    )

    # Phase 1: Vectorized frame-wise player composition extraction
    logger.info(f"Processing {tracking_data['frame_id'].nunique():,} unique frames...")

    # Filter out ball data and get only player data
    player_tracking = tracking_data[tracking_data["home_away"] != "BALL"].copy()

    # Create frame-wise player compositions using vectorized groupby
    frame_compositions = (
        player_tracking.groupby(["frame_id", "home_away"])["jersey_number"]
        .apply(lambda x: frozenset(x))
        .unstack(fill_value=frozenset())
    )

    # Handle missing columns safely
    if "HOME" in frame_compositions.columns:
        home_compositions = frame_compositions["HOME"]
    else:
        home_compositions = pd.Series(
            [frozenset() for _ in range(len(frame_compositions))], index=frame_compositions.index, dtype=object
        )

    if "AWAY" in frame_compositions.columns:
        away_compositions = frame_compositions["AWAY"]
    else:
        away_compositions = pd.Series(
            [frozenset() for _ in range(len(frame_compositions))], index=frame_compositions.index, dtype=object
        )

    logger.info("Frame-wise compositions computed successfully")

    # Phase 2: Detect changes efficiently
    player_change_list = []
    home_substitute_queue = changed_player_list_in_home.copy()  # Create copy to avoid modifying original
    away_substitute_queue = changed_player_list_in_away.copy()

    current_home_players = player_ever_on_pitch_home.copy()
    current_away_players = player_ever_on_pitch_away.copy()

    # Process frames in order (matching legacy logic exactly)
    sorted_frames = sorted(frame_compositions.index)

    # Handle halftime substitutions: Update initial player sets with early frame data
    # This prevents false substitution detection for halftime changes
    if sorted_frames:
        # Look at first several frames to establish baseline (handles FIFA WC data structure)
        frames_to_check = sorted_frames[: min(10, len(sorted_frames))]  # Check first 10 frames
        all_early_home_players = set()
        all_early_away_players = set()

        for frame in frames_to_check:
            if frame in home_compositions.index:
                all_early_home_players.update(home_compositions[frame])
            if frame in away_compositions.index:
                all_early_away_players.update(away_compositions[frame])

        # Include any players present in early frames as "initial" players
        # This is crucial for FIFA WC data where halftime subs might appear early
        current_home_players = current_home_players.union(all_early_home_players)
        current_away_players = current_away_players.union(all_early_away_players)

        logger.debug(f"Initial HOME players (including early frames): {sorted(current_home_players)}")
        logger.debug(f"Initial AWAY players (including early frames): {sorted(current_away_players)}")

    # Track previous compositions to detect changes
    prev_home_composition = frozenset(current_home_players)
    prev_away_composition = frozenset(current_away_players)

    for frame_id in sorted_frames:
        home_frame_players = set(home_compositions[frame_id]) if frame_id in home_compositions.index else set()
        away_frame_players = set(away_compositions[frame_id]) if frame_id in away_compositions.index else set()

        player_change_info = []

        # Check for HOME team changes
        new_players_home = home_frame_players - current_home_players
        if new_players_home:
            logger.debug(f"Frame {frame_id}: New HOME players detected: {new_players_home}")
            logger.debug(f"Frame {frame_id}: HOME substitute queue: {home_substitute_queue}")

            try:
                for new_player in new_players_home:
                    if home_substitute_queue:
                        player_out = home_substitute_queue.pop(0)
                        player_change_info.append(
                            {
                                "home_away": "HOME",
                                "player_in": new_player,
                                "player_out": player_out,
                            }
                        )
                        logger.debug(f"HOME substitution: {player_out} -> {new_player}")
                    else:
                        logger.warning(f"No substitute available for HOME team new player {new_player}")
                        logger.warning(f"Frame {frame_id}: Current HOME players: {current_home_players}")
                        logger.warning(f"Frame {frame_id}: Frame HOME players: {home_frame_players}")
            except Exception as e:
                logger.error(f"Error processing HOME team substitution at frame {frame_id}: {e}")
                logger.error(f"new_players_home: {new_players_home}")
                logger.error(f"home_substitute_queue: {home_substitute_queue}")
                logger.error(f"current_home_players: {current_home_players}")

        # Check for AWAY team changes
        new_players_away = away_frame_players - current_away_players
        if new_players_away:
            logger.debug(f"Frame {frame_id}: New AWAY players detected: {new_players_away}")
            logger.debug(f"Frame {frame_id}: AWAY substitute queue: {away_substitute_queue}")

            try:
                for new_player in new_players_away:
                    if away_substitute_queue:
                        player_out = away_substitute_queue.pop(0)
                        player_change_info.append(
                            {
                                "home_away": "AWAY",
                                "player_in": new_player,
                                "player_out": player_out,
                            }
                        )
                        logger.debug(f"AWAY substitution: {player_out} -> {new_player}")
                    else:
                        logger.warning(f"No substitute available for AWAY team new player {new_player}")
                        logger.warning(f"Frame {frame_id}: Current AWAY players: {current_away_players}")
                        logger.warning(f"Frame {frame_id}: Frame AWAY players: {away_frame_players}")
            except Exception as e:
                logger.error(f"Error processing AWAY team substitution at frame {frame_id}: {e}")
                logger.error(f"new_players_away: {new_players_away}")
                logger.error(f"away_substitute_queue: {away_substitute_queue}")
                logger.error(f"current_away_players: {current_away_players}")

        # Record changes if any occurred
        if player_change_info:
            player_change_list.append(
                {
                    "frame_id": frame_id,
                    "player_change_info": player_change_info,
                }
            )
            logger.info(f"Detected {len(player_change_info)} player changes at frame {frame_id}")

        # CRITICAL: Update current player sets to match legacy behavior exactly
        # Legacy version uses union() to accumulate all players ever seen
        current_home_players = current_home_players.union(home_frame_players)
        current_away_players = current_away_players.union(away_frame_players)

    logger.info(f"Player change detection completed. Found {len(player_change_list)} substitution events.")
    return player_change_list


def get_player_change_log_legacy(
    tracking_data: pd.DataFrame,
    player_data: pd.DataFrame,
    changed_player_list_in_home: List[int],
    changed_player_list_in_away: List[int],
) -> List[Dict[str, Any]]:
    """
    Legacy implementation of get_player_change_log for comparison/fallback.
    This is the original frame-by-frame processing version.
    """
    player_ever_on_pitch_home: Set[int] = set(
        player_data.query("home_away == 'HOME'").query("starting_member == 1")["jersey_number"].astype(int).values
    )
    player_ever_on_pitch_away: Set[int] = set(
        player_data.query("home_away == 'AWAY'").query("starting_member == 1")["jersey_number"].astype(int).values
    )
    player_change_list = []
    for _, group in tracking_data.groupby("frame_id"):
        players_in_frame_home = set(group.query("home_away == 'HOME'")["jersey_number"].values)
        players_in_frame_away = set(group.query("home_away == 'AWAY'")["jersey_number"].values)

        player_change_info = []
        if len(new_players_home := players_in_frame_home - player_ever_on_pitch_home) > 0:
            try:
                player_change_info.extend(
                    [
                        {
                            "home_away": "HOME",
                            "player_in": player,
                            "player_out": changed_player_list_in_home.pop(0),
                        }
                        for player in new_players_home
                    ]
                )
            except Exception as e:
                print(f"Error in processing player change for HOME team: {e}")
                print("new_players_home:", new_players_home)
                print("changed_player_list_in_home:", changed_player_list_in_home)
                print("player_ever_on_pitch_home:", player_ever_on_pitch_home)
                import pdb

                pdb.set_trace()

        if len(new_players_away := players_in_frame_away - player_ever_on_pitch_away) > 0:
            player_change_info.extend(
                [
                    {
                        "home_away": "AWAY",
                        "player_in": player,
                        "player_out": changed_player_list_in_away.pop(0),
                    }
                    for player in new_players_away
                ]
            )
        if len(player_change_info) > 0:
            player_change_list.append(
                {
                    "frame_id": group["frame_id"].values[0],
                    "player_change_info": player_change_info,
                }
            )

        player_ever_on_pitch_home = players_in_frame_home.union(player_ever_on_pitch_home)
        player_ever_on_pitch_away = players_in_frame_away.union(player_ever_on_pitch_away)
    return player_change_list


def pad_players_and_interpolate_tracking_data(
    tracking_data: pd.DataFrame,
    player_data: pd.DataFrame,
    event_data: pd.DataFrame,
    player_change_list: List[Dict[str, Any]],
    origin_pos: str = "center",
    absolute_coordinates: bool = True,
) -> pd.DataFrame:
    """
    This function pads the players and interpolates the tracking data.
    It first interpolates the tracking data for each series so that tracking data exists for every frame
    for every player on the pitch.
    Then, it pads the players who are not on the pitch for each frame.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    player_data (pd.DataFrame): DataFrame containing player data
    event_data (pd.DataFrame): DataFrame containing event data
    player_change_list (List[Dict[str, Any]]):
        List of dictionaries containing the frame number and the players who have changed
    origin_pos (str): The origin position for the coordinates. Defaults to 'center'.
    absolute_coordinates (bool): Whether to use absolute coordinates. Defaults to True.

    Returns:
    pd.DataFrame: DataFrame with padded players and interpolated tracking data

    """

    def __pad_coordinates(row: pd.Series, origin_pos: str, absolute_coordinates: bool) -> pd.Series:
        # padding coordinates are always center left (the center of the defensive goal)
        if origin_pos == "center":
            row["y"] = 0.0
            if absolute_coordinates:
                if row["attack_direction"] == 1:
                    row["x"] = round(-FIELD_LENGTH / 2 * 100, 2)
                else:
                    row["x"] = round(FIELD_LENGTH / 2 * 100, 2)
            else:
                row["x"] = round(-FIELD_LENGTH / 2 * 100, 2)

        elif origin_pos == "top_left":
            row["y"] = round(-FIELD_WIDTH / 2 * 100, 2)
            if absolute_coordinates:
                if row["attack_direction"] == 1:
                    row["x"] = 0.0
                else:
                    row["x"] = round(FIELD_LENGTH * 100, 2)
            else:
                row["x"] = 0.0
        else:
            raise ValueError("origin_pos must be 'center' or 'top_left'")
        return row

    def __add_player_padding(data: pd.DataFrame, origin_pos: str, absolute_coordinates: bool) -> pd.DataFrame:
        frames_to_be_padded = (
            data.query("home_away != 'BALL'")
            .groupby(["game_id", "frame_id", "half", "home_away", "series_num"])["jersey_number"]
            .nunique()[
                data.query("home_away != 'BALL'")
                .groupby(["game_id", "frame_id", "half", "home_away", "series_num"])["jersey_number"]
                .nunique()
                < 11
            ]
            .reset_index()
            .copy()
        )
        frames_to_be_padded["num_pad"] = frames_to_be_padded["jersey_number"].apply(lambda x: 11 - x)

        padding_list = []
        for _, item in frames_to_be_padded.iterrows():
            frame_id = item["frame_id"]
            attack_direction = data.query(f"frame_id == {frame_id}")["attack_direction"].iloc[0]
            padding_list.append(
                pd.DataFrame(
                    {
                        "game_id": [item["game_id"]] * item["num_pad"],
                        "frame_id": [item["frame_id"]] * item["num_pad"],
                        "half": [item["half"]] * item["num_pad"],
                        "home_away": [item["home_away"]] * item["num_pad"],
                        "series_num": [item["series_num"]] * item["num_pad"],
                        "attack_direction": [attack_direction] * item["num_pad"],
                        "jersey_number": [-1 * (i + 1) for i in range(item["num_pad"])],
                        "x": [0] * item["num_pad"],
                        "y": [0] * item["num_pad"],
                    }
                )
            )
        if len(padding_list) == 0:
            return data
        padding = pd.concat(padding_list).reset_index(drop=True)
        padding[["x", "y"]] = 0.0
        padding = padding.apply(__pad_coordinates, axis=1, args=(origin_pos, absolute_coordinates))
        return pd.concat([data, padding], ignore_index=True)

    def merge_ball_only_series(data):
        new_data_list = []
        previous_series = None

        for _, series in data.groupby("series_num"):
            if previous_series is None:
                previous_series = series
            else:
                if set(series["home_away"].unique()) == {"BALL"}:
                    # merge ball only series to next series
                    previous_series = pd.concat([previous_series, series]).sort_values("frame_id")
                    previous_series["series_num"] = previous_series["series_num"].max()
                else:
                    # if not ball only series, save current series and start new series
                    new_data_list.append(previous_series)
                    previous_series = series

        # add last series
        if previous_series is not None:
            new_data_list.append(previous_series)

        return pd.concat(new_data_list).reset_index(drop=True)

    # need series num and player change info for finer interpolation
    tracking_data = pd.merge(
        tracking_data,
        event_data[["frame_id", "series_num", "attack_direction"]],
        on="frame_id",
        how="left",
    )
    tracking_data[["series_num", "attack_direction"]] = (
        tracking_data[["series_num", "attack_direction"]].fillna(method="ffill").fillna(method="bfill").astype(int)
    )
    player_change_list = sorted(player_change_list, key=lambda x: x["frame_id"]) if len(player_change_list) > 0 else []

    new_data_list = []
    player_on_pitch_home = set(player_data.query("starting_member == 1 and home_away == 'HOME'")["jersey_number"].values)
    player_on_pitch_away = set(player_data.query("starting_member == 1 and home_away == 'AWAY'")["jersey_number"].values)
    for idx in range(len(player_change_list) + 1):
        start_frame = tracking_data["frame_id"].min() if idx == 0 else player_change_list[idx - 1]["frame_id"]
        end_frame = (
            player_change_list[idx]["frame_id"] - 1 if idx != len(player_change_list) else tracking_data["frame_id"].max()
        )

        data = tracking_data.query(f"{start_frame} <= frame_id <= {end_frame}")
        data = data.query(
            """
            (jersey_number in @player_on_pitch_home and home_away == 'HOME') \
            or (jersey_number in @player_on_pitch_away and home_away == 'AWAY') or (home_away == 'BALL')
            """
        )
        # merge ball only series to next series
        data = merge_ball_only_series(data)
        # interpolation and padding
        for _, series in data.groupby("series_num"):
            series_start_frame = series["frame_id"].min()
            series_end_frame = series["frame_id"].max()
            new_series_list = []
            for _, group in series.groupby(["home_away", "jersey_number"]):
                new_index = pd.DataFrame({"frame_id": range(series_start_frame, series_end_frame + 1)})
                new_group = pd.merge(new_index, group, on="frame_id", how="left")
                # Handle duplicates by keeping the last occurrence
                new_group = new_group.drop_duplicates(subset=["frame_id"], keep="last")
                new_group[["x", "y"]] = new_group[["x", "y"]].interpolate(method="linear", limit_direction="both")
                new_group["game_id"] = new_group["game_id"].fillna(method="ffill").fillna(method="bfill").astype(int)
                new_group["half"] = new_group["half"].fillna(method="ffill").fillna(method="bfill").astype(str)
                new_group[["series_num", "jersey_number", "attack_direction"]] = (
                    new_group[["series_num", "jersey_number", "attack_direction"]]
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                    .astype(int)
                )
                new_group["home_away"] = new_group["home_away"].fillna(method="ffill").fillna(method="bfill").astype(str)
                new_series_list.append(new_group)
                assert new_group[["x", "y"]].isnull().sum().sum() == 0
                if new_group["frame_id"].nunique() != len(new_group):
                    print(f"Error in series {new_group['series_num'].iloc[0]} in game {new_group['game_id'].iloc[0]}")
                    print(f"frame_id: {new_group['frame_id'].unique()}")
                    print(f"home_away: {new_group['home_away'].unique()}")
                    print(f"jersey_number: {new_group['jersey_number'].unique()}")
                    breakpoint()
                assert new_group["frame_id"].nunique() == len(new_group)
            new_series = pd.concat(new_series_list).sort_values("frame_id").reset_index(drop=True)
            # padding
            new_series = __add_player_padding(new_series, origin_pos, absolute_coordinates)
            new_data_list.append(new_series)
            assert (
                new_series.query("home_away != 'BALL'").groupby(["frame_id", "half", "home_away"])["jersey_number"].nunique()
                != 11
            ).sum() == 0, f"{new_series['game_id'].iloc[0]} {new_series['series_num'].iloc[0]}"

        if idx != len(player_change_list):
            player_change_info_list = player_change_list[idx]["player_change_info"]
            for player_change_info in player_change_info_list:
                if player_change_info["home_away"] == "HOME":
                    try:
                        player_on_pitch_home.remove(player_change_info["player_out"])
                        player_on_pitch_home.add(player_change_info["player_in"])
                    except Exception as e:
                        print(f"Error in processing player change for HOME team: {e}")
                        print(f"game_id: {tracking_data['game_id'].iloc[0]}")
                        print(f"player_change_info: {player_change_info}")
                        print(f"player_on_pitch_home: {player_on_pitch_home}")
                        import pdb

                        pdb.set_trace()
                else:
                    try:
                        player_on_pitch_away.remove(player_change_info["player_out"])
                        player_on_pitch_away.add(player_change_info["player_in"])
                    except Exception as e:
                        print(f"Error in processing player change for AWAY team: {e}")
                        print(f"game_id: {tracking_data['game_id'].iloc[0]}")
                        print(f"player_change_info: {player_change_info}")
                        print(f"player_on_pitch_away: {player_on_pitch_away}")
                        import pdb

                        pdb.set_trace()

    new_tracking_data = pd.concat(new_data_list)
    new_tracking_data = new_tracking_data.sort_values(by=["half", "frame_id", "home_away", "jersey_number"]).reset_index(
        drop=True
    )

    first_half_end_series_num = new_tracking_data.query("half == 'first'")["series_num"].max()
    second_half_start_series_num = new_tracking_data.query("half == 'second'")["series_num"].min()

    if first_half_end_series_num == second_half_start_series_num:
        new_tracking_data.loc[new_tracking_data["half"] == "second", "series_num"] += 1

    assert (
        new_tracking_data.query("home_away != 'BALL'").groupby(["frame_id", "half", "home_away"])["jersey_number"].nunique()
        != 11
    ).sum() == 0
    assert (new_tracking_data.query("home_away == 'BALL'").groupby("frame_id").value_counts() != 1).sum() == 0
    assert new_tracking_data[["x", "y"]].isna().sum().sum() == 0
    for series_num, series in new_tracking_data.groupby("series_num"):
        min_frame = series["frame_id"].min()
        max_frame = series["frame_id"].max()

        # invalid_frame_ids = series.groupby("frame_id").filter(lambda x: len(x) != 23)["frame_id"].unique()

        # delete invalid series num
        if len(series) != (max_frame - min_frame + 1) * 23:
            new_tracking_data = new_tracking_data[new_tracking_data["series_num"] != series_num]

        # assert (
        #     len(series) == (max_frame - min_frame + 1) * 23
        # ), f"{len(series)} != {(max_frame - min_frame + 1) * 23} in series {series['series_num'].iloc[0]}, game_id {series['game_id'].iloc[0]}, min_frame {min_frame}, max_frame {max_frame}"

    new_tracking_data = new_tracking_data.reset_index(drop=True)
    return new_tracking_data


def resample_tracking_data(
    tracking_data: pd.DataFrame,
    timestamp_dict: Dict[str, int],
    player_change_list: List[Dict[str, Any]],
    original_sampling_rate: int = 25,
    target_sampling_rate: int = 10,
) -> pd.DataFrame:
    """
    This function resamples the tracking data.
    It first resamples the tracking data to target_sampling_rate.
    We then carefully deal with the duplicated data in the same time_from_half_start.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    timestamp_dict (Dict[str, int]): Dictionary containing the start frame number for each half
    player_change_list (List[Dict[str, Any]]):
        List of dictionaries containing the frame number and the players who have changed
    original_sampling_rate (int, optional): Sampling rate of the original tracking data. Defaults to 25.
    target_sampling_rate (int, optional): Sampling rate of the resampled tracking data. Defaults to 10.

    Returns:
    pd.DataFrame: DataFrame with resampled tracking data
    """
    first_start_frame, second_start_frame = (
        timestamp_dict["first_start_frame"],
        timestamp_dict["second_start_frame"],
    )

    resampled_data_list = []
    for idx in range(len(player_change_list) + 1):
        start_frame = tracking_data["frame_id"].min() if idx == 0 else player_change_list[idx - 1]["frame_id"]
        end_frame = (
            player_change_list[idx]["frame_id"] - 1 if idx != len(player_change_list) else tracking_data["frame_id"].max()
        )

        data = tracking_data.query(f"{start_frame} <= frame_id <= {end_frame}")
        for _, group in data.groupby(["half", "series_num", "home_away", "jersey_number"]):
            start_frame = group["frame_id"].min()
            end_frame = group["frame_id"].max()
            half_start_frame = first_start_frame if group["half"].iloc[0] == "first" else second_start_frame
            assert group[["x", "y"]].isnull().sum().sum() == 0

            resampled_data = pd.DataFrame(
                signal.resample_poly(
                    group[["x", "y"]],
                    up=target_sampling_rate,
                    down=original_sampling_rate,
                    axis=0,
                    padtype="line",
                ),
                columns=["x", "y"],
            )

            # Check for NaN values after signal resampling
            resampled_nans = resampled_data[["x", "y"]].isna().sum().sum()
            if resampled_nans > 0:
                logger.warning(
                    f"Found {resampled_nans} NaN values after signal resampling for player/ball "
                    f"{group['home_away'].iloc[0]}/{group['jersey_number'].iloc[0]}, applying interpolation"
                )
                # Apply interpolation to fix NaN values from signal processing
                resampled_data[["x", "y"]] = resampled_data[["x", "y"]].interpolate(method="linear", limit_direction="both")

                # If interpolation fails, use original data mean
                remaining_nans = resampled_data[["x", "y"]].isna().sum().sum()
                if remaining_nans > 0:
                    group_mean_x = group["x"].mean()
                    group_mean_y = group["y"].mean()
                    logger.warning(
                        f"Interpolation failed, using group mean ({group_mean_x:.2f},{group_mean_y:.2f}) for player/ball "
                        f"{group['home_away'].iloc[0]}/{group['jersey_number'].iloc[0]}"
                    )
                    resampled_data["x"] = resampled_data["x"].fillna(group_mean_x)
                    resampled_data["y"] = resampled_data["y"].fillna(group_mean_y)
            # Use high-precision Decimal-based time generation
            start_time = round((start_frame - half_start_frame) / original_sampling_rate, 1)
            end_time = round((end_frame - half_start_frame) / original_sampling_rate, 1)

            # Generate precise time sequence using Decimal arithmetic
            time_sequence = generate_time_sequence(start_time, end_time, target_sampling_rate)

            # Adjust resampled data to match the precise time sequence length
            if len(time_sequence) != len(resampled_data):
                # Resample to exact length if needed
                target_length = len(time_sequence)
                current_length = len(resampled_data)

                if target_length > current_length:
                    # Interpolate to increase length
                    indices = np.linspace(0, current_length - 1, target_length)
                    resampled_x = np.interp(indices, range(current_length), resampled_data["x"])
                    resampled_y = np.interp(indices, range(current_length), resampled_data["y"])
                    resampled_data = pd.DataFrame({"x": resampled_x, "y": resampled_y})
                else:
                    # Truncate to match target length
                    resampled_data = resampled_data.iloc[:target_length].copy()

            resampled_data.loc[:, "time_from_half_start"] = time_sequence
            resampled_data = resampled_data.drop_duplicates(subset=["time_from_half_start"])

            # In some cases, there's missing 'time_from_half_start'.
            # Here, we make time_from_half_start complete using precise generation
            complete_time_sequence = generate_time_sequence(start_time, end_time, target_sampling_rate)
            interpolated_resampled_data = pd.DataFrame({"time_from_half_start": complete_time_sequence})
            interpolated_resampled_data = pd.merge(
                interpolated_resampled_data,
                resampled_data,
                on="time_from_half_start",
                how="left",
            )
            # Enhanced interpolation with NaN handling
            interpolated_resampled_data[["x", "y"]] = interpolated_resampled_data[["x", "y"]].interpolate(
                method="linear", limit_direction="both"
            )

            # Handle remaining NaN values after interpolation
            remaining_nans = interpolated_resampled_data[["x", "y"]].isna().sum().sum()
            if remaining_nans > 0:
                logger.warning(
                    f"Found {remaining_nans} remaining NaN values after interpolation for player/ball "
                    f"{group['home_away'].iloc[0]}/{group['jersey_number'].iloc[0]}, applying fallback methods"
                )

                # Strategy 1: Forward fill then backward fill
                interpolated_resampled_data[["x", "y"]] = (
                    interpolated_resampled_data[["x", "y"]].fillna(method="ffill").fillna(method="bfill")
                )

                # Strategy 2: If still NaN (all values were NaN), use group mean or zero
                remaining_nans_after_fill = interpolated_resampled_data[["x", "y"]].isna().sum().sum()
                if remaining_nans_after_fill > 0:
                    # Use group mean if available, otherwise use zero
                    group_mean_x = group["x"].mean()
                    group_mean_y = group["y"].mean()

                    if pd.isna(group_mean_x) or pd.isna(group_mean_y):
                        # If group mean is also NaN, use default field center position
                        fill_x = 0.0
                        fill_y = 0.0
                        logger.warning(
                            f"Using default center position (0,0) for player/ball "
                            f"{group['home_away'].iloc[0]}/{group['jersey_number'].iloc[0]}"
                        )
                    else:
                        fill_x = group_mean_x
                        fill_y = group_mean_y
                        logger.warning(
                            f"Using group mean position ({fill_x:.2f},{fill_y:.2f}) for player/ball "
                            f"{group['home_away'].iloc[0]}/{group['jersey_number'].iloc[0]}"
                        )

                    interpolated_resampled_data["x"] = interpolated_resampled_data["x"].fillna(fill_x)
                    interpolated_resampled_data["y"] = interpolated_resampled_data["y"].fillna(fill_y)

                # Final validation
                final_nans = interpolated_resampled_data[["x", "y"]].isna().sum().sum()
                if final_nans > 0:
                    logger.error(
                        f"Critical: Still have {final_nans} NaN values after all fallback methods "
                        f"for player/ball {group['home_away'].iloc[0]}/{group['jersey_number'].iloc[0]}"
                    )
                    # Emergency fallback: replace any remaining NaN with zero
                    interpolated_resampled_data[["x", "y"]] = interpolated_resampled_data[["x", "y"]].fillna(0.0)
            interpolated_resampled_data["game_id"] = group["game_id"].iloc[0]
            interpolated_resampled_data["home_away"] = group["home_away"].iloc[0]
            interpolated_resampled_data["jersey_number"] = group["jersey_number"].iloc[0]
            interpolated_resampled_data["half"] = group["half"].iloc[0]
            interpolated_resampled_data["series_num"] = group["series_num"].iloc[0]
            resampled_data_list.append(interpolated_resampled_data)

    resampled_tracking_data = pd.concat(resampled_data_list)

    # Enhanced duplicate removal with logging
    initial_count = len(resampled_tracking_data)
    resampled_tracking_data = resampled_tracking_data.drop_duplicates(
        subset=["time_from_half_start", "half", "home_away", "jersey_number"],
        keep="last",
    )
    final_count = len(resampled_tracking_data)

    if initial_count > final_count:
        logger.info(f"Removed {initial_count - final_count} duplicates during resampling")

    resampled_tracking_data = resampled_tracking_data.sort_values(
        by=["half", "time_from_half_start", "home_away", "jersey_number"]
    )[
        [
            "game_id",
            "half",
            "series_num",
            "time_from_half_start",
            "home_away",
            "jersey_number",
            "x",
            "y",
        ]
    ].reset_index(drop=True)

    # 選手数が11を超えた場合の修正処理（交代を考慮）
    def fix_player_count(df):
        player_data = df.query("home_away != 'BALL'").copy()
        player_counts = player_data.groupby(["time_from_half_start", "half", "home_away"])["jersey_number"].nunique()
        problematic_times = player_counts[player_counts != 11].index.tolist()

        if not problematic_times:
            return df

        logger.info(f"Found {len(problematic_times)} time points with incorrect player count")

        for time_point, half, team in problematic_times:
            current_mask = (df["time_from_half_start"] == time_point) & (df["half"] == half) & (df["home_away"] == team)
            current_players = df[current_mask]["jersey_number"].tolist()

            if len(current_players) > 11:
                # 前後の時刻で実際にプレーしている選手を確認
                time_values = sorted(
                    player_data[(player_data["half"] == half) & (player_data["home_away"] == team)][
                        "time_from_half_start"
                    ].unique()
                )

                current_idx = time_values.index(time_point)

                # 前後10フレームの選手を確認（交代の瞬間を考慮）
                # より高速なベクトル化されたアプローチ
                time_values = np.array(
                    sorted(
                        player_data[(player_data["half"] == half) & (player_data["home_away"] == team)][
                            "time_from_half_start"
                        ].unique()
                    )
                )

                current_idx = np.where(time_values == time_point)[0][0]

                # 前後10フレームのインデックス範囲を計算
                start_idx = max(0, current_idx - 10)
                end_idx = min(len(time_values), current_idx + 11)
                context_time_range = time_values[start_idx:end_idx]

                # 一度のクエリで該当する全フレームのデータを取得
                context_mask = (
                    player_data["time_from_half_start"].isin(context_time_range)
                    & (player_data["half"] == half)
                    & (player_data["home_away"] == team)
                )
                context_data = player_data[context_mask]

                # 正常なフレーム（11人）のみをフィルタリング
                valid_frames = context_data.groupby("time_from_half_start")["jersey_number"].nunique()
                valid_times = valid_frames[valid_frames == 11].index

                # 正常なフレームの選手を集計
                context_players = set(
                    context_data[context_data["time_from_half_start"].isin(valid_times)]["jersey_number"].tolist()
                )

                # パディング選手と実選手を分離
                padding_players = [p for p in current_players if p < 0]  # 負の背番号はパディング選手
                real_players = [p for p in current_players if p > 0]  # 正の背番号は実選手

                # 実選手が11人を超えている場合の処理
                if len(real_players) > 11:
                    # 前後のフレームに出現する実選手を優先
                    if len(context_players) >= 11:
                        context_real_players = [p for p in real_players if p in context_players]
                        other_real_players = [p for p in real_players if p not in context_players]

                        players_to_keep = context_real_players[:11]
                        if len(players_to_keep) < 11:
                            players_to_keep.extend(other_real_players[: 11 - len(players_to_keep)])
                    else:
                        # 前後のフレーム情報が不十分な場合、背番号順で選択
                        players_to_keep = sorted(real_players)[:11]
                elif len(real_players) <= 11:
                    # 実選手が11人以下の場合、すべての実選手を保持
                    players_to_keep = real_players.copy()
                    # 不足分をパディング選手で補う（最大11人まで）
                    needed_padding = 11 - len(players_to_keep)
                    if needed_padding > 0 and padding_players:
                        # パディング選手を背番号順でソートして必要数だけ追加
                        sorted_padding = sorted(padding_players, reverse=True)  # -1, -2, -3...の順
                        players_to_keep.extend(sorted_padding[:needed_padding])

                logger.info(
                    f"Player selection at time {time_point}: Real={len([p for p in players_to_keep if p > 0])}, Padding={len([p for p in players_to_keep if p < 0])}"
                )

                # 選択されなかった選手を除去
                players_to_remove = [p for p in current_players if p not in players_to_keep]
                remove_mask = current_mask & df["jersey_number"].isin(players_to_remove)
                df = df[~remove_mask].copy()

                logger.info(
                    f"Removed {len(players_to_remove)} excess players at time {time_point}, {half} half, {team} team. Kept: {players_to_keep}"
                )

        return df

    resampled_tracking_data = fix_player_count(resampled_tracking_data)

    assert (
        resampled_tracking_data.query("home_away != 'BALL'")
        .groupby(["time_from_half_start", "half", "home_away"])["jersey_number"]
        .nunique()
        != 11
    ).sum() == 0, f"is not 0, game_id: {resampled_tracking_data['game_id'].iloc[0]}"

    assert (
        resampled_tracking_data.query("home_away == 'BALL'").groupby("time_from_half_start").value_counts() != 1
    ).sum() == 0
    assert resampled_tracking_data[["x", "y"]].isna().sum().sum() == 0
    return resampled_tracking_data


def format_tracking_data(
    tracking_data: pd.DataFrame,
    home_team_name: str,
    away_team_name: str,
    player_dict: Dict[Tuple[str, str], Dict],
    state_def: str,
) -> pd.DataFrame:
    """
    This function formats the tracking data.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    home_team_name (str): Home team name
    away_team_name (str): Away team name
    player_dict (Dict[Tuple[str, str], Dict]): Dictionary containing player information

    Returns:
    pd.DataFrame: DataFrame with formatted tracking data
    """
    tracking_list = []
    for _, group in tracking_data.groupby(["half", "time_from_half_start"]):
        # Use Decimal for precise time handling
        precise_time = float(Decimal(str(group["time_from_half_start"].values[0])).quantize(Decimal("0.1")))
        frame_dict = {
            "time_from_half_start": precise_time,
            "half": group["half"].values[0],
            "ball": None,
            "players": [],
        }
        if state_def == "PVS":
            for _, d in group.iterrows():
                if d["jersey_number"] == 0:
                    frame_dict["ball"] = {"position": {"x": d["x"], "y": d["y"]}}
                else:
                    home_away_str = d["home_away"]
                    jersey_number = d["jersey_number"]
                    frame_dict["players"].append(
                        {
                            "team_name": home_team_name if home_away_str == "HOME" else away_team_name,
                            "player_name": safe_get_player_info(player_dict, home_away_str, jersey_number, "player_name", None)
                            if jersey_number > 0
                            else None,
                            "player_id": safe_get_player_info(
                                player_dict, home_away_str, jersey_number, "player_id", jersey_number
                            )
                            if jersey_number > 0
                            else jersey_number,
                            "player_role": safe_get_player_info(player_dict, home_away_str, jersey_number, "player_role", None)
                            if jersey_number > 0
                            else None,
                            "jersey_number": jersey_number,
                            "position": {"x": d["x"], "y": d["y"]},
                        }
                    )
        elif state_def == "EDMS":
            for _, d in group.iterrows():
                if d["jersey_number"] == 0:
                    frame_dict["ball"] = {"position": {"x": d["x"], "y": d["y"]}}
                elif d["jersey_number"] < 0:
                    home_away_str = d["home_away"]
                    frame_dict["players"].append(
                        {
                            "team_name": home_team_name if home_away_str == "HOME" else away_team_name,
                            "player_name": None,
                            "player_id": None,
                            "player_role": None,
                            "jersey_number": d["jersey_number"],
                            "height": None,
                            "position": {"x": d["x"], "y": d["y"]},
                        }
                    )
                else:
                    home_away_str = d["home_away"]
                    jersey_number = d["jersey_number"]
                    frame_dict["players"].append(
                        {
                            "team_name": home_team_name if home_away_str == "HOME" else away_team_name,
                            "player_name": safe_get_player_info(player_dict, home_away_str, jersey_number, "player_name", None)
                            if jersey_number > 0
                            else None,
                            "player_id": safe_get_player_info(
                                player_dict, home_away_str, jersey_number, "player_id", jersey_number
                            )
                            if jersey_number > 0
                            else jersey_number,
                            "player_role": safe_get_player_info(player_dict, home_away_str, jersey_number, "player_role", None)
                            if jersey_number > 0
                            else None,
                            "jersey_number": jersey_number if jersey_number > 0 else None,
                            "height": safe_get_player_info(player_dict, home_away_str, jersey_number, "height", None),
                            "position": {"x": d["x"], "y": d["y"]},
                        }
                    )
        tracking_list.append(frame_dict)

    result_df = pd.DataFrame(tracking_list)

    # Use enhanced temporal duplicate removal
    result_df = remove_temporal_duplicates(result_df)

    # Validate temporal consistency
    is_consistent = validate_temporal_consistency(result_df)
    if not is_consistent:
        logger.warning("Temporal consistency issues detected in formatted tracking data")

    return result_df


def format_tracking_data_laliga(
    tracking_data: pd.DataFrame,
    home_team_name: str,
    away_team_name: str,
    player_dict: Dict[Tuple[str, str], Dict],
) -> pd.DataFrame:
    """
    This function formats the tracking data.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    home_team_name (str): Home team name
    away_team_name (str): Away team name
    player_dict (Dict[Tuple[str, str], Dict]): Dictionary containing player information

    Returns:
    pd.DataFrame: DataFrame with formatted tracking data
    """
    tracking_list = []
    for _, group in tracking_data.groupby(["half", "time_from_half_start"]):
        frame_dict = {
            "time_from_half_start": round(group["time_from_half_start"].values[0], 1),
            "half": group["half"].values[0],
            "ball": None,
            "players": [],
        }
        for _, d in group.iterrows():
            if d["jersey_number"] == 0:
                frame_dict["ball"] = {"position": {"x": d["x"], "y": d["y"]}}
            elif d["jersey_number"] < 0:
                home_away_str = d["home_away"]
                frame_dict["players"].append(
                    {
                        "team_name": home_team_name if home_away_str == "HOME" else away_team_name,
                        "player_name": None,
                        "player_id": None,
                        "player_role": None,
                        "jersey_number": d["jersey_number"],
                        "height": None,
                        "position": {"x": d["x"], "y": d["y"]},
                    }
                )
            else:
                home_away_str = d["home_away"]
                jersey_number = d["jersey_number"]
                frame_dict["players"].append(
                    {
                        "team_name": home_team_name if home_away_str == "HOME" else away_team_name,
                        "player_name": player_dict[home_away_str, jersey_number]["player_name"] if jersey_number > 0 else None,
                        "player_id": player_dict[home_away_str, jersey_number]["player_id"]
                        if jersey_number > 0
                        else jersey_number,
                        "player_role": player_dict[home_away_str, jersey_number]["player_role"] if jersey_number > 0 else None,
                        "jersey_number": jersey_number if jersey_number > 0 else None,
                        "height": player_dict[home_away_str, jersey_number]["height"],
                        "position": {"x": d["x"], "y": d["y"]},
                    }
                )
        tracking_list.append(frame_dict)

    return pd.DataFrame(tracking_list)


def parse_tracking_data(x):
    if x is None:
        return None
    if isinstance(x, list):
        if len(x) == 0:
            return None
        return x
    if isinstance(x, dict):
        if len(x) == 0:
            return None
        # velocityキーが存在する辞書は空ではない
        if "velocity" in x or "acceleration" in x or "position" in x:
            return x
        return x
    try:
        parsed = json.loads(x)
        if isinstance(parsed, dict) and len(parsed) == 0:
            return None
        return parsed
    except json.JSONDecodeError:
        logger.warning(f"Warning: Unable to parse JSON: {x}")
        return None


def clean_empty_data(series):
    def convert_empty(x):
        if isinstance(x, dict):
            # 重要なキーが含まれている場合は空と判定しない
            if any(key in x for key in ["velocity", "acceleration", "position"]):
                return x
            if len(x) == 0:
                return None
        elif isinstance(x, list) and len(x) == 0:
            return None
        return x

    return series.apply(convert_empty)


def calculate_speed(tracking_data: pd.DataFrame, sampling_rate: int = 10) -> pd.DataFrame:
    """
    This function calculates the speed of each player and the ball.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    sampling_rate (int, optional): Sampling rate of the tracking data. Defaults to 10.

    Returns:
    pd.DataFrame: DataFrame with speed
    """

    def __get_player2pos(player_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        player2pos = {}
        for player in player_data:
            player2pos[player["player_name"]] = player["position"]
        return player2pos

    def __get_player2vel(player_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        player2vel = {}
        for player in player_data:
            player2vel[player["player_name"]] = player["velocity"]
        return player2vel

    tracking_data = tracking_data.copy()
    time_delta = 1 / sampling_rate

    for idx, data in tracking_data.iterrows():
        ball_pos = deepcopy(data["ball"])
        player_data = deepcopy(data["players"])
        current_time_from_half_start = deepcopy(data["time_from_half_start"])

        if ball_pos is None:
            continue  # or handle the None case appropriately

        if idx == len(tracking_data) - 1:
            # same as the previous frame
            prev_data = tracking_data.iloc[idx - 1]
            if (abs(current_time_from_half_start - prev_data["time_from_half_start"]) - time_delta) < 1e-5:
                ball_pos["velocity"] = deepcopy(json.loads(prev_data["ball"])["velocity"])
                prev_player2vel = __get_player2vel(json.loads(prev_data["players"]))
                for d in player_data:
                    d["velocity"] = deepcopy(prev_player2vel[d["player_name"]])
            else:
                # singleton? -> set velocity to 0
                ball_pos["velocity"] = {"x": 0, "y": 0}
                for d in player_data:
                    d["velocity"] = {"x": 0, "y": 0}
        else:
            next_data = tracking_data.iloc[idx + 1]
            next_time_from_half_start = next_data["time_from_half_start"]
            if (abs(current_time_from_half_start - next_time_from_half_start) - time_delta) < 1e-5:
                # calculate velocity using the next frame
                try:
                    ball_pos["velocity"] = {
                        "x": (next_data["ball"]["position"]["x"] - ball_pos["position"]["x"]) / time_delta,
                        "y": (next_data["ball"]["position"]["y"] - ball_pos["position"]["y"]) / time_delta,
                    }
                except Exception as e:
                    logger.warning(f"Error calculating ball velocity: {e}")
                    logger.warning("ball is not in next_data")
                    ball_pos["velocity"] = {"x": 0, "y": 0}
                next_player2pos = __get_player2pos(next_data["players"])
                for d in player_data:
                    if d["player_name"] in next_player2pos:
                        d["velocity"] = {
                            "x": (next_player2pos[d["player_name"]]["x"] - d["position"]["x"]) / time_delta,
                            "y": (next_player2pos[d["player_name"]]["y"] - d["position"]["y"]) / time_delta,
                        }
                    else:
                        logger.warning(f"{d['player_name']} is not in next_player2pos")
                        d["velocity"] = {"x": 0, "y": 0}
            else:
                prev_data = tracking_data.iloc[idx - 1]
                if (abs(current_time_from_half_start - prev_data["time_from_half_start"]) - time_delta) < 1e-5:
                    # same as the previous frame
                    ball_pos["velocity"] = deepcopy(json.loads(prev_data["ball"])["velocity"])
                    prev_player2vel = __get_player2vel(json.loads(prev_data["players"]))
                    for d in player_data:
                        try:
                            d["velocity"] = deepcopy(prev_player2vel[d["player_name"]])
                        except Exception as e:
                            logger.warning(f"Error getting player velocity: {e}")
                            print(f"prev_data: {prev_data}.")
                            d["velocity"] = {"x": 0, "y": 0}
                else:
                    # singleton? -> set velocity to 0
                    ball_pos["velocity"] = {"x": 0, "y": 0}
                    for d in player_data:
                        d["velocity"] = {"x": 0, "y": 0}
        tracking_data.loc[idx, "ball"] = json.dumps(ball_pos)
        tracking_data.loc[idx, "players"] = json.dumps(player_data)

    tracking_data["ball"] = tracking_data["ball"].apply(parse_tracking_data)
    tracking_data["players"] = tracking_data["players"].apply(parse_tracking_data)

    tracking_data["ball"] = clean_empty_data(tracking_data["ball"])
    tracking_data["players"] = clean_empty_data(tracking_data["players"])

    tracking_data["ball"] = tracking_data["ball"].fillna(method="ffill").fillna(method="bfill")
    tracking_data["players"] = tracking_data["players"].fillna(method="ffill").fillna(method="bfill")

    # Validate that all players have velocity fields
    tracking_data = validate_and_fix_velocity_fields(tracking_data)

    return tracking_data


def validate_and_fix_velocity_fields(tracking_data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that all players have velocity fields and add default velocity if missing.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data with calculated velocities

    Returns:
    pd.DataFrame: DataFrame with validated velocity fields for all players
    """
    tracking_data = tracking_data.copy()

    for idx, row in tracking_data.iterrows():
        # Check ball velocity
        ball_data = row["ball"]
        if ball_data and "velocity" not in ball_data:
            logger.warning(f"Missing velocity field for ball at frame {idx}, adding default velocity")
            ball_data["velocity"] = {"x": 0.0, "y": 0.0}
            tracking_data.at[idx, "ball"] = ball_data

        # Check player velocities
        player_data = row["players"]
        if player_data:
            players_modified = False
            for player in player_data:
                if "velocity" not in player:
                    logger.warning(
                        f"Missing velocity field for player {player.get('player_name', 'Unknown')} at frame {idx}, adding default velocity"
                    )
                    player["velocity"] = {"x": 0.0, "y": 0.0}
                    players_modified = True
                elif (
                    not isinstance(player["velocity"], dict) or "x" not in player["velocity"] or "y" not in player["velocity"]
                ):
                    logger.warning(
                        f"Invalid velocity format for player {player.get('player_name', 'Unknown')} at frame {idx}, fixing velocity"
                    )
                    player["velocity"] = {"x": 0.0, "y": 0.0}
                    players_modified = True

            if players_modified:
                tracking_data.at[idx, "players"] = player_data

    return tracking_data


def calculate_acceleration(tracking_data: pd.DataFrame, sampling_rate: int = 10) -> pd.DataFrame:
    """
    This function calculates the acceleration of each player and the ball.
    Enhanced to better handle FIFA World Cup data with frequent substitutions.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    sampling_rate (int, optional): Sampling rate of the tracking data. Defaults to 10.

    Returns:
    pd.DataFrame: DataFrame with acceleration
    """

    def __is_likely_substitution_scenario(
        player_name: str, jersey_number: int, tracking_data: pd.DataFrame, frame_idx: int
    ) -> bool:
        """
        Detect if a player not found scenario is likely due to substitution.

        Args:
            player_name: Name of the player
            jersey_number: Jersey number
            tracking_data: DataFrame with tracking data
            frame_idx: Current frame index

        Returns:
            bool: True if likely a substitution scenario
        """
        # For FIFA WC data, substitutions are common and expected
        # Check for common substitution indicators
        if not player_name or jersey_number <= 0:
            return True

        # Check if this is near halftime or end of match (common substitution times)
        current_time = tracking_data.iloc[frame_idx].get("time_from_half_start", 0)

        # Common substitution times (in seconds from half start)
        # Early in half (0-300s), mid-half (1200-1800s), late in half (2400-2700s)
        substitution_windows = [(0, 300), (1200, 1800), (2400, 2700)]

        for start, end in substitution_windows:
            if start <= current_time <= end:
                return True

        return False

    def __get_player2vel(player_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        player2vel = {}
        for player in player_data:
            player_name = player.get("player_name")
            jersey_number = player.get("jersey_number", 0)

            # パディング選手（負の背番号）をスキップ
            if jersey_number < 0:
                # パディング選手のvelocityはデフォルトで0に設定
                player2vel[f"padding_{jersey_number}"] = {"x": 0, "y": 0}
                continue

            if not player_name:
                logger.warning(f"Real player without name found: jersey_number={jersey_number}")
                continue

            if "velocity" not in player:
                logger.debug(f"Player {player_name} has no velocity data, setting to zero (likely substitution scenario)")
                player2vel[player_name] = {"x": 0, "y": 0}
            else:
                player2vel[player_name] = player["velocity"]
        return player2vel

    tracking_data = tracking_data.copy()
    time_delta = 1 / sampling_rate

    for idx, data in tracking_data.iterrows():
        ball_pos = deepcopy(data["ball"])
        player_data = deepcopy(data["players"])
        current_time_from_half_start = deepcopy(data["time_from_half_start"])

        if ball_pos is None:
            continue  # or handle the None case appropriately

        if idx == len(tracking_data) - 1:
            # Same as the previous frame
            prev_data = tracking_data.iloc[idx - 1]
            if (abs(current_time_from_half_start - prev_data["time_from_half_start"]) - time_delta) < 1e-5:
                prev_ball_data = safe_json_parse(prev_data["ball"])
                prev_players_data = safe_json_parse(prev_data["players"])
                ball_pos["acceleration"] = deepcopy(prev_ball_data.get("acceleration", {"x": 0, "y": 0}))
                prev_player2vel = __get_player2vel(prev_players_data)
                for d in player_data:
                    player_name = d.get("player_name")
                    jersey_number = d.get("jersey_number", 0)

                    # パディング選手の処理
                    if jersey_number < 0:
                        d["acceleration"] = {"x": 0, "y": 0}
                        continue

                    # 実選手の処理
                    if player_name and player_name in prev_player2vel:
                        d["acceleration"] = deepcopy(prev_player2vel[player_name])
                    elif jersey_number < 0 and f"padding_{jersey_number}" in prev_player2vel:
                        d["acceleration"] = {"x": 0, "y": 0}
                    else:
                        if player_name:  # 実選手のみログ出力
                            # Use substitution detection for better context
                            is_substitution = __is_likely_substitution_scenario(player_name, jersey_number, tracking_data, idx)
                            substitution_note = " (substitution detected)" if is_substitution else " (unexpected absence)"
                            logger.debug(
                                f"Player {player_name} (#{jersey_number}) not found in previous frame, setting acceleration to zero{substitution_note}"
                            )
                        d["acceleration"] = {"x": 0, "y": 0}
            else:
                # Singleton? -> set acceleration to 0
                ball_pos["acceleration"] = {"x": 0, "y": 0}
                for d in player_data:
                    d["acceleration"] = {"x": 0, "y": 0}
        else:
            next_data = tracking_data.iloc[idx + 1]
            next_time_from_half_start = next_data["time_from_half_start"]
            if (abs(current_time_from_half_start - next_time_from_half_start) - time_delta) < 1e-5:
                # Calculate acceleration using the next frame
                try:
                    ball_pos["acceleration"] = {
                        "x": (next_data["ball"]["velocity"]["x"] - ball_pos["velocity"]["x"]) / time_delta,
                        "y": (next_data["ball"]["velocity"]["y"] - ball_pos["velocity"]["y"]) / time_delta,
                    }
                except Exception as e:
                    logger.warning(f"Error calculating ball acceleration: {e}")
                    continue
                next_player2vel = __get_player2vel(next_data["players"])
                for d in player_data:
                    player_name = d.get("player_name")
                    jersey_number = d.get("jersey_number", 0)

                    # パディング選手の処理
                    if jersey_number < 0:
                        d["acceleration"] = {"x": 0, "y": 0}
                        continue

                    if not player_name:
                        logger.warning(f"Real player without name found: jersey_number={jersey_number}")
                        d["acceleration"] = {"x": 0, "y": 0}
                        continue

                    if player_name in next_player2vel:
                        if "velocity" not in d:
                            # Use substitution detection for better context
                            is_substitution = __is_likely_substitution_scenario(player_name, jersey_number, tracking_data, idx)
                            substitution_note = " (substitution detected)" if is_substitution else " (data missing)"
                            logger.debug(
                                f"Player {player_name} (#{jersey_number}) has no velocity data, setting acceleration to zero{substitution_note}"
                            )
                            d["acceleration"] = {"x": 0, "y": 0}
                        else:
                            try:
                                d["acceleration"] = {
                                    "x": (next_player2vel[player_name]["x"] - d["velocity"]["x"]) / time_delta,
                                    "y": (next_player2vel[player_name]["y"] - d["velocity"]["y"]) / time_delta,
                                }
                            except (KeyError, TypeError) as e:
                                logger.warning(f"Error calculating acceleration for {player_name}: {e}")
                                d["acceleration"] = {"x": 0, "y": 0}
                    else:
                        # Use substitution detection for better context
                        is_substitution = __is_likely_substitution_scenario(player_name, jersey_number, tracking_data, idx)
                        substitution_note = " (substitution detected)" if is_substitution else " (unexpected absence)"
                        logger.debug(
                            f"Player {player_name} (#{jersey_number}) not found in next frame, setting acceleration to zero{substitution_note}"
                        )
                        d["acceleration"] = {"x": 0, "y": 0}
            else:
                prev_data = tracking_data.iloc[idx - 1]
                if (abs(current_time_from_half_start - prev_data["time_from_half_start"]) - time_delta) < 1e-5:
                    try:
                        ball_pos["acceleration"] = deepcopy(json.loads(prev_data["ball"])["acceleration"])
                        prev_player2vel = __get_player2vel(json.loads(prev_data["players"]))
                    except Exception as e:
                        logger.warning(f"Error getting previous ball acceleration: {e}")
                        print(f"prev_data: {prev_data['ball']}")
                        continue

                    for d in player_data:
                        try:
                            d["acceleration"] = deepcopy(prev_player2vel[d["player_name"]])
                        except Exception as e:
                            logger.warning(f"Error getting player acceleration: {e}")
                            print(f"prev_data: {prev_data}.")
                            continue
                else:
                    # Singleton? -> set acceleration to 0
                    ball_pos["acceleration"] = {"x": 0, "y": 0}
                    for d in player_data:
                        d["acceleration"] = {"x": 0, "y": 0}

        tracking_data.loc[idx, "ball"] = json.dumps(ball_pos)
        tracking_data.loc[idx, "players"] = json.dumps(player_data)

    tracking_data["ball"] = tracking_data["ball"].apply(parse_tracking_data)
    tracking_data["players"] = tracking_data["players"].apply(parse_tracking_data)

    # Validate that all players have velocity and acceleration fields
    tracking_data = validate_and_fix_velocity_acceleration_fields(tracking_data)

    return tracking_data


def validate_and_fix_velocity_acceleration_fields(tracking_data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that all players have velocity and acceleration fields and add defaults if missing.
    This is called after calculate_acceleration to ensure all required fields exist.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data with calculated velocities and accelerations

    Returns:
    pd.DataFrame: DataFrame with validated velocity and acceleration fields for all players
    """
    tracking_data = tracking_data.copy()

    for idx, row in tracking_data.iterrows():
        # Check ball velocity and acceleration
        ball_data = row["ball"]
        if ball_data:
            if "velocity" not in ball_data:
                logger.warning(f"Missing velocity field for ball at frame {idx}, adding default velocity")
                ball_data["velocity"] = {"x": 0.0, "y": 0.0}
            if "acceleration" not in ball_data:
                logger.warning(f"Missing acceleration field for ball at frame {idx}, adding default acceleration")
                ball_data["acceleration"] = {"x": 0.0, "y": 0.0}
            tracking_data.at[idx, "ball"] = ball_data

        # Check player velocities and accelerations
        player_data = row["players"]
        if player_data:
            players_modified = False
            for player in player_data:
                # Validate velocity
                if "velocity" not in player:
                    logger.warning(
                        f"Missing velocity field for player {player.get('player_name', 'Unknown')} at frame {idx}, adding default velocity"
                    )
                    player["velocity"] = {"x": 0.0, "y": 0.0}
                    players_modified = True
                elif (
                    not isinstance(player["velocity"], dict) or "x" not in player["velocity"] or "y" not in player["velocity"]
                ):
                    logger.warning(
                        f"Invalid velocity format for player {player.get('player_name', 'Unknown')} at frame {idx}, fixing velocity"
                    )
                    player["velocity"] = {"x": 0.0, "y": 0.0}
                    players_modified = True

                # Validate acceleration
                if "acceleration" not in player:
                    logger.warning(
                        f"Missing acceleration field for player {player.get('player_name', 'Unknown')} at frame {idx}, adding default acceleration"
                    )
                    player["acceleration"] = {"x": 0.0, "y": 0.0}
                    players_modified = True
                elif (
                    not isinstance(player["acceleration"], dict)
                    or "x" not in player["acceleration"]
                    or "y" not in player["acceleration"]
                ):
                    logger.warning(
                        f"Invalid acceleration format for player {player.get('player_name', 'Unknown')} at frame {idx}, fixing acceleration"
                    )
                    player["acceleration"] = {"x": 0.0, "y": 0.0}
                    players_modified = True

            if players_modified:
                tracking_data.at[idx, "players"] = player_data

    return tracking_data

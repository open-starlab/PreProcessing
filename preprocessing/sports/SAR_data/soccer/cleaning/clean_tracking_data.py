import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from preprocessing.sports.SAR_data.soccer.constant import FIELD_LENGTH, FIELD_WIDTH, HOME_AWAY_MAP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def complement_tracking_ball_with_event_data(tracking_ball: pd.DataFrame, event_data: pd.DataFrame, first_end_frame: int, league: str) -> pd.DataFrame:
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
        pd.merge(tracking_ball, event_data[['frame_id', 'ball_x', 'ball_y']], on='frame_id', how='outer')
        .sort_values('frame_id')
        .reset_index(drop=True)
    )
    complemented_data['game_id'] = event_data['game_id'].iloc[0]

    if league == 'jleague':
        complemented_data['half'] = complemented_data['half'].fillna(method='ffill').fillna(method='bfill').astype(str)
    elif league == 'laliga':
        complemented_data['half'] = complemented_data['frame_id'].apply(lambda x: "first" if x <= first_end_frame else "second")

    complemented_data['home_away'] = 'BALL'
    complemented_data['jersey_number'] = 0
    complemented_data['x'] = complemented_data['x'].fillna(complemented_data['ball_x'])
    complemented_data['y'] = complemented_data['y'].fillna(complemented_data['ball_y'])
    complemented_data = complemented_data.drop(columns=['ball_x', 'ball_y'])

    return complemented_data


def complement_tracking_ball_with_event_data_laliga(tracking_ball: pd.DataFrame, event_data: pd.DataFrame, first_end_frame: int) -> pd.DataFrame:
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
        pd.merge(tracking_ball, event_data[['frame_id', 'ball_x', 'ball_y']], on='frame_id', how='outer')
        .sort_values('frame_id')
        .reset_index(drop=True)
    )
    complemented_data['game_id'] = event_data['game_id'].iloc[0]
    # complemented_data['half'] = complemented_data['half'].fillna(method='ffill').fillna(method='bfill').astype(str)
    complemented_data['half'] = complemented_data['frame_id'].apply(lambda x: "first" if x <= first_end_frame else "second")
    complemented_data['home_away'] = 'BALL'
    complemented_data['jersey_number'] = 0
    complemented_data['x'] = complemented_data['x'].fillna(complemented_data['ball_x'])
    complemented_data['y'] = complemented_data['y'].fillna(complemented_data['ball_y'])
    complemented_data = complemented_data.drop(columns=['ball_x', 'ball_y'])
    complemented_data = complemented_data.dropna(subset=['x', 'y'])

    return complemented_data


def interpolate_ball_tracking_data(
    tracking_ball: pd.DataFrame,
    event_data: pd.DataFrame,
    ignored_events: List[str] = [
        '交代',
        '警告(イエロー)',
        '退場(レッド)',
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
        event_data['series_num'].value_counts()[event_data['series_num'].value_counts() != 1].index
    )
    valid_series_boundaries = (
        event_data.query("event_name not in @ignored_events")
        .query("series_num in @valid_series_num")
        .groupby('series_num')
        .agg({'frame_id': ['min', 'max']})['frame_id']
        .reset_index()
    )
    valid_series_boundaries.columns = ['series_num', 'min_frame_id', 'max_frame_id']

    interpolated_data_list = []
    for _, item in valid_series_boundaries.iterrows():
        start_frame = item['min_frame_id']
        end_frame = item['max_frame_id']
        new_index = pd.DataFrame({'frame_id': range(int(start_frame), int(end_frame) + 1)})
        interpolated_data = pd.merge(new_index, tracking_ball, on='frame_id', how='left')
        interpolated_data[['x', 'y']] = interpolated_data[['x', 'y']].interpolate(
            method='linear', limit_direction='both'
        )
        if interpolated_data['x'].isnull().sum() > 0:
            print(f"Skip {item['series_num']} for lack of tracking data")
            continue
        
        try:
            interpolated_data['game_id'] = tracking_ball.query("@start_frame <= frame_id <= @end_frame")['game_id'].iloc[0]
        except:
            print(f'Skip {item["series_num"]} for lack of tracking data')
            continue
        interpolated_data['half'] = tracking_ball.query("@start_frame <= frame_id <= @end_frame")['half'].iloc[0]
        interpolated_data['home_away'] = 'BALL'
        interpolated_data['jersey_number'] = 0
        interpolated_data_list.append(interpolated_data)

    interpolated_tracking_ball = (
        pd.concat(interpolated_data_list)
        .sort_values('frame_id')
        .reset_index(drop=True)[
            [
                'game_id',
                'frame_id',
                "half",
                "home_away",
                "jersey_number",
                "x",
                "y",
            ]
        ]
    )

    # If there are rows with the same frame_id
    interpolated_tracking_ball = interpolated_tracking_ball.drop_duplicates(subset=['frame_id'], keep='last')
    # interpolated_tracking_ball = interpolated_tracking_ball[~interpolated_tracking_ball.duplicated(subset=['frame_id', 'x', 'y'], keep='first')]


    if interpolated_tracking_ball['frame_id'].nunique() != len(interpolated_tracking_ball):
        print("interpolated_tracking_ball:", interpolated_tracking_ball)
        print("unique frame_ids:", interpolated_tracking_ball['frame_id'].nunique())
        print("length of interpolated_tracking_ball:", len(interpolated_tracking_ball))

        # Extract columns with duplicate frame_id
        duplicated_frame_id = interpolated_tracking_ball[interpolated_tracking_ball.duplicated(subset=['frame_id'], keep=False)]
        print("duplicated_frame_id:", duplicated_frame_id)
        import pdb; pdb.set_trace()
    
    assert interpolated_tracking_ball['frame_id'].nunique() == len(interpolated_tracking_ball)
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
    tracking_data['half'] = tracking_data['frame_id'].apply(lambda x: "first" if x <= first_end_frame else "second")
    tracking_data['home_away'] = tracking_data['home_away'].apply(lambda x: HOME_AWAY_MAP[x])
    tracking_data = tracking_data[['game_id', "frame_id", "half", 'home_away', 'jersey_number', 'x', 'y']]
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
    in_play_frame_num = tracking_ball['frame_id'].unique()  # noqa:  F841
    tracking_player = tracking_player.query("frame_id in @in_play_frame_num")
    assert tracking_ball[['x', 'y']].isnull().sum().sum() == 0

    tracking_data = (
        pd.concat([tracking_player, tracking_ball], axis=0)
        .sort_values(by=['frame_id', 'home_away', 'jersey_number'])
        .reset_index(drop=True)
    )[
        [
            'game_id',
            'frame_id',
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
    origin_pos: str = 'center',
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

    def _convert_coordinate(tracking_data: pd.DataFrame, origin_pos: str, absolute_coordinates: bool, league: str) -> pd.DataFrame:
        if league == "jleague":
            tracking_data.loc[:, 'x'] = tracking_data['x'].map(lambda x: x / 100)
            tracking_data.loc[:, 'y'] = tracking_data['y'].map(lambda x: -x / 100)
        elif league == "laliga":
            tracking_data.loc[:, 'x'] = tracking_data['x'].map(lambda x: x)
            tracking_data.loc[:, 'y'] = tracking_data['y'].map(lambda x: -x)

        if origin_pos == "top_left":
            tracking_data.loc[:, 'x'] = tracking_data['x'] + 52.5
            tracking_data.loc[:, 'y'] = tracking_data['y'] + 34.0
        elif origin_pos != "center":
            raise ValueError("origin_pos must be 'center' or 'bottom_left'")

        if absolute_coordinates is False:
            tracking_data.loc[tracking_data['attack_direction'] != 1, 'x'] = -tracking_data.loc[
                tracking_data['attack_direction'] != 1, 'x'
            ]
            tracking_data.loc[tracking_data['attack_direction'] != 1, 'y'] = -tracking_data.loc[
                tracking_data['attack_direction'] != 1, 'y'
            ]

        # fix padding
        if origin_pos == "center":
            tracking_data.loc[tracking_data['jersey_number'] <= -1, 'y'] = 0.0
            if absolute_coordinates:
                tracking_data.loc[
                    (tracking_data['jersey_number'] <= -1) & (tracking_data['attack_direction'] == 1), 'x'
                ] = round(-FIELD_LENGTH / 2, 2)
                tracking_data.loc[
                    (tracking_data['jersey_number'] <= -1) & (tracking_data['attack_direction'] != 1), 'x'
                ] = round(FIELD_LENGTH / 2, 2)
            else:
                tracking_data.loc[tracking_data['jersey_number'] <= -1, 'x'] = round(-FIELD_WIDTH / 2, 2)

        elif origin_pos == 'top_left':
            tracking_data.loc[tracking_data['jersey_number'] <= -1, 'y'] = round(FIELD_WIDTH / 2, 2)
            if absolute_coordinates:
                tracking_data.loc[
                    (tracking_data['jersey_number'] <= -1) & (tracking_data['attack_direction'] == 1), 'x'
                ] = 0.0
                tracking_data.loc[
                    (tracking_data['jersey_number'] <= -1) & (tracking_data['attack_direction'] != 1), 'x'
                ] = round(FIELD_LENGTH, 2)
            else:
                tracking_data.loc[tracking_data['jersey_number'] <= -1, 'x'] = 0.0
        else:
            raise ValueError("origin_pos must be 'center' or 'top_left'")

        # clip (use np.clip ?)
        if origin_pos == "center":
            tracking_data.loc[:, 'x'] = tracking_data['x'].clip(-FIELD_LENGTH / 2, FIELD_LENGTH / 2)
            tracking_data.loc[:, 'y'] = tracking_data['y'].clip(-FIELD_WIDTH / 2, FIELD_WIDTH / 2)
        elif origin_pos == 'top_left':
            tracking_data.loc[:, 'x'] = tracking_data['x'].clip(0, FIELD_LENGTH)
            tracking_data.loc[:, 'y'] = tracking_data['y'].clip(0, FIELD_WIDTH)
        else:
            raise ValueError("origin_pos must be 'center' or 'top_left'")

        return tracking_data

    tracking_data = pd.merge(
        tracking_data,
        event_data[['half', 'time_from_half_start', 'attack_direction', 'attack_start_history_num']],
        on=['half', 'time_from_half_start'],
        how='left',
    )
    # decide attack direction by the majority vote within the same attack_start_history_num
    tracking_data[['attack_direction', 'attack_start_history_num']] = (
        tracking_data[['attack_direction', 'attack_start_history_num']]
        .fillna(method='ffill')
        .fillna(method='bfill')
        .astype(int)
    )
    tracking_data.loc[:, 'attack_direction'] = tracking_data.groupby(['half', 'attack_start_history_num'])[
        'attack_direction'
    ].transform(lambda x: x.value_counts().index[0])
    tracking_data = _convert_coordinate(tracking_data, origin_pos, absolute_coordinates, league="jleague")
    return tracking_data.sort_values(by=['half', 'time_from_half_start', 'home_away', 'jersey_number'])[
        [
            'game_id',
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
    This function gets the player change log.
    It returns a list of dictionaries containing the frame number and the players who have changed.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    player_data (pd.DataFrame): DataFrame containing player data
    changed_player_list_in_home (List[int]): List of players who have changed in the home team
    changed_player_list_in_away (List[int]): List of players who have changed in the away team

    Returns:
    List[Dict[str, Any]]: List of dictionaries containing the frame number and the players who have changed
    """

    player_ever_on_pitch_home: Set[int] = set(
        player_data.query("home_away == 'HOME'").query("starting_member == 1")['jersey_number'].astype(int).values
    )
    player_ever_on_pitch_away: Set[int] = set(
        player_data.query("home_away == 'AWAY'").query("starting_member == 1")['jersey_number'].astype(int).values
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
            except:
                print("new_players_home:", new_players_home)
                print("changed_player_list_in_home:", changed_player_list_in_home)
                print("player_ever_on_pitch_home:", player_ever_on_pitch_home)
                import pdb; pdb.set_trace()

        if len(new_players_away := players_in_frame_away - player_ever_on_pitch_away) > 0:
            try:
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
            except:
                print("new_players_home:", new_players_home)
                print("changed_player_list_in_home:", changed_player_list_in_home)
                print("player_ever_on_pitch_home:", player_ever_on_pitch_home)
                import pdb; pdb.set_trace()
        if len(player_change_info) > 0:
            player_change_list.append(
                {"frame_id": group["frame_id"].values[0], "player_change_info": player_change_info}
            )

        player_ever_on_pitch_home = players_in_frame_home.union(player_ever_on_pitch_home)
        player_ever_on_pitch_away = players_in_frame_away.union(player_ever_on_pitch_away)

    return player_change_list


def pad_players_and_interpolate_tracking_data(
    tracking_data: pd.DataFrame,
    player_data: pd.DataFrame,
    event_data: pd.DataFrame,
    player_change_list: List[Dict[str, Any]],
    origin_pos: str = 'center',
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
            row['y'] = 0.0
            if absolute_coordinates:
                if row['attack_direction'] == 1:
                    row['x'] = round(-FIELD_LENGTH / 2 * 100, 2)
                else:
                    row['x'] = round(FIELD_LENGTH / 2 * 100, 2)
            else:
                row['x'] = round(-FIELD_LENGTH / 2 * 100, 2)

        elif origin_pos == 'top_left':
            row['y'] = round(-FIELD_WIDTH / 2 * 100, 2)
            if absolute_coordinates:
                if row['attack_direction'] == 1:
                    row['x'] = 0.0
                else:
                    row['x'] = round(FIELD_LENGTH * 100, 2)
            else:
                row['x'] = 0.0
        else:
            raise ValueError("origin_pos must be 'center' or 'top_left'")
        return row

    def __add_player_padding(data: pd.DataFrame, origin_pos: str, absolute_coordinates: bool) -> pd.DataFrame:
        frames_to_be_padded = (
            data.query("home_away != 'BALL'")
            .groupby(['game_id', 'frame_id', 'half', 'home_away', 'series_num'])['jersey_number']
            .nunique()[
                data.query("home_away != 'BALL'")
                .groupby(['game_id', 'frame_id', 'half', 'home_away', 'series_num'])['jersey_number']
                .nunique()
                < 11
            ]
            .reset_index()
            .copy()
        )
        frames_to_be_padded['num_pad'] = frames_to_be_padded['jersey_number'].apply(lambda x: 11 - x)

        padding_list = []
        for _, item in frames_to_be_padded.iterrows():
            frame_id = item['frame_id']
            attack_direction = data.query(f"frame_id == {frame_id}")['attack_direction'].iloc[0]
            padding_list.append(
                pd.DataFrame(
                    {
                        "game_id": [item['game_id']] * item['num_pad'],
                        "frame_id": [item['frame_id']] * item['num_pad'],
                        "half": [item['half']] * item['num_pad'],
                        "home_away": [item['home_away']] * item['num_pad'],
                        "series_num": [item['series_num']] * item['num_pad'],
                        "attack_direction": [attack_direction] * item['num_pad'],
                        "jersey_number": [-1 * (i + 1) for i in range(item['num_pad'])],
                        "x": [0] * item['num_pad'],
                        "y": [0] * item['num_pad'],
                    }
                )
            )
        if len(padding_list) == 0:
            return data
        padding = pd.concat(padding_list).reset_index(drop=True)
        padding[['x', 'y']] = 0.0
        padding = padding.apply(__pad_coordinates, axis=1, args=(origin_pos, absolute_coordinates))
        return pd.concat([data, padding], ignore_index=True)
    
    def merge_ball_only_series(data):
        new_data_list = []
        previous_series = None

        for _, series in data.groupby('series_num'):
            if previous_series is None:
                previous_series = series
            else:
                if set(series['home_away'].unique()) == {'BALL'}:
                    # merge ball only series to next series
                    previous_series = pd.concat([previous_series, series]).sort_values('frame_id')
                    previous_series['series_num'] = previous_series['series_num'].max()
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
        tracking_data, event_data[['frame_id', 'series_num', 'attack_direction']], on='frame_id', how='left'
    )
    tracking_data[['series_num', 'attack_direction']] = (
        tracking_data[['series_num', 'attack_direction']].fillna(method='ffill').fillna(method='bfill').astype(int)
    )
    player_change_list = sorted(player_change_list, key=lambda x: x['frame_id'])

    new_data_list = []
    player_on_pitch_home = set(
        player_data.query("starting_member == 1 and home_away == 'HOME'")['jersey_number'].values
    )
    player_on_pitch_away = set(
        player_data.query("starting_member == 1 and home_away == 'AWAY'")['jersey_number'].values
    )
    for idx in range(len(player_change_list) + 1):
        start_frame = tracking_data['frame_id'].min() if idx == 0 else player_change_list[idx - 1]['frame_id']
        end_frame = (
            player_change_list[idx]['frame_id'] - 1
            if idx != len(player_change_list)
            else tracking_data['frame_id'].max()
        )

        data = tracking_data.query(f"{start_frame} <= frame_id <= {end_frame}")
        data = data.query(
            '''
            (jersey_number in @player_on_pitch_home and home_away == 'HOME') \
            or (jersey_number in @player_on_pitch_away and home_away == 'AWAY') or (home_away == 'BALL')
            '''
        )
        # merge ball only series to next series
        data = merge_ball_only_series(data)
        # interpolation and padding
        for _, series in data.groupby('series_num'):
            series_start_frame = series['frame_id'].min()
            series_end_frame = series['frame_id'].max()
            new_series_list = []
            for _, group in series.groupby(['home_away', 'jersey_number']):
                new_index = pd.DataFrame({'frame_id': range(series_start_frame, series_end_frame + 1)})
                new_group = pd.merge(new_index, group, on='frame_id', how='left')
                new_group[['x', 'y']] = new_group[['x', 'y']].interpolate(method='linear', limit_direction='both')
                new_group["game_id"] = new_group['game_id'].fillna(method='ffill').fillna(method='bfill').astype(int)
                new_group["half"] = new_group['half'].fillna(method='ffill').fillna(method='bfill').astype(str)
                new_group[["series_num", "jersey_number", "attack_direction"]] = (
                    new_group[["series_num", "jersey_number", "attack_direction"]]
                    .fillna(method='ffill')
                    .fillna(method='bfill')
                    .astype(int)
                )
                new_group["home_away"] = (
                    new_group['home_away'].fillna(method='ffill').fillna(method='bfill').astype(str)
                )
                new_series_list.append(new_group)
                assert new_group[['x', 'y']].isnull().sum().sum() == 0
                assert new_group['frame_id'].nunique() == len(new_group)
            new_series = pd.concat(new_series_list).sort_values('frame_id').reset_index(drop=True)
            # padding
            new_series = __add_player_padding(new_series, origin_pos, absolute_coordinates)
            new_data_list.append(new_series)
            assert (
                new_series.query("home_away != 'BALL'")
                .groupby(['frame_id', 'half', 'home_away'])['jersey_number']
                .nunique()
                != 11
            ).sum() == 0, f"{new_series['game_id'].iloc[0]} {new_series['series_num'].iloc[0]}"
        if idx != len(player_change_list):
            player_change_info_list = player_change_list[idx]['player_change_info']
            for player_change_info in player_change_info_list:
                if player_change_info['home_away'] == 'HOME':
                    try:
                        player_on_pitch_home.remove(player_change_info['player_out'])
                        player_on_pitch_home.add(player_change_info['player_in'])
                    except:
                        print(f"game_id: {tracking_data['game_id'].iloc[0]}")
                        print(f"player_change_info: {player_change_info}")
                        print(f"player_on_pitch_home: {player_on_pitch_home}")
                        continue
                else:
                    try:
                        player_on_pitch_away.remove(player_change_info['player_out'])
                        player_on_pitch_away.add(player_change_info['player_in'])
                    except:
                        print(f"game_id: {tracking_data['game_id'].iloc[0]}")
                        print(f"player_change_info: {player_change_info}")
                        print(f"player_on_pitch_away: {player_on_pitch_away}")
                        continue

    new_tracking_data = pd.concat(new_data_list)
    new_tracking_data = new_tracking_data.sort_values(
        by=['half', 'frame_id', 'home_away', 'jersey_number']
    ).reset_index(drop=True)

    first_half_end_series_num = new_tracking_data.query("half == 'first'")['series_num'].max()
    second_half_start_series_num = new_tracking_data.query("half == 'second'")['series_num'].min()

    if first_half_end_series_num == second_half_start_series_num:
        new_tracking_data.loc[
            new_tracking_data['half'] == 'second', 'series_num'
        ] += 1


    assert (
        new_tracking_data.query("home_away != 'BALL'")
        .groupby(['frame_id', 'half', 'home_away'])['jersey_number']
        .nunique()
        != 11
    ).sum() == 0
    assert (new_tracking_data.query("home_away == 'BALL'").groupby('frame_id').value_counts() != 1).sum() == 0
    assert new_tracking_data[['x', 'y']].isna().sum().sum() == 0
    for series_num, series in new_tracking_data.groupby("series_num"):
        min_frame = series['frame_id'].min()
        max_frame = series['frame_id'].max()
        
        invalid_frame_ids = series.groupby('frame_id').filter(lambda x: len(x) != 23)['frame_id'].unique()
        
        # delete invalid series num
        if len(series) != (max_frame - min_frame + 1) * 23:
            new_tracking_data = new_tracking_data[new_tracking_data['series_num'] != series_num]


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
        timestamp_dict['first_start_frame'],
        timestamp_dict['second_start_frame'],
    )

    resampled_data_list = []
    for idx in range(len(player_change_list) + 1):
        start_frame = tracking_data['frame_id'].min() if idx == 0 else player_change_list[idx - 1]['frame_id']
        end_frame = (
            player_change_list[idx]['frame_id'] - 1
            if idx != len(player_change_list)
            else tracking_data['frame_id'].max()
        )

        data = tracking_data.query(f"{start_frame} <= frame_id <= {end_frame}")
        for _, group in data.groupby(['half', 'series_num', "home_away", "jersey_number"]):
            start_frame = group['frame_id'].min()
            end_frame = group['frame_id'].max()
            half_start_frame = first_start_frame if group['half'].iloc[0] == "first" else second_start_frame
            assert group[['x', 'y']].isnull().sum().sum() == 0

            resampled_data = pd.DataFrame(
                signal.resample_poly(
                    group[["x", "y"]], up=target_sampling_rate, down=original_sampling_rate, axis=0, padtype='line'
                ),
                columns=["x", "y"],
            )
            resampled_data.loc[:, "time_from_half_start"] = np.linspace(
                round((start_frame - half_start_frame) / original_sampling_rate, 1),
                round((end_frame - half_start_frame) / original_sampling_rate, 1),
                len(resampled_data),
            ).round(1)
            resampled_data = resampled_data.drop_duplicates(subset=['time_from_half_start'])

            # In some cases, there's missing 'time_from_half_start'.
            # Here, we make time_from_half_start complete
            interpolated_resampled_data = pd.DataFrame(
                np.arange(
                    round((start_frame - half_start_frame) / original_sampling_rate, 1),
                    round((end_frame - half_start_frame) / original_sampling_rate, 1),
                    1 / target_sampling_rate,
                ).round(1),
                columns=['time_from_half_start'],
            )
            interpolated_resampled_data = pd.merge(
                interpolated_resampled_data, resampled_data, on='time_from_half_start', how='left'
            )
            interpolated_resampled_data[['x', 'y']] = interpolated_resampled_data[['x', 'y']].interpolate(
                method='linear', limit_direction='both'
            )
            interpolated_resampled_data["game_id"] = group['game_id'].iloc[0]
            interpolated_resampled_data["home_away"] = group['home_away'].iloc[0]
            interpolated_resampled_data["jersey_number"] = group['jersey_number'].iloc[0]
            interpolated_resampled_data["half"] = group['half'].iloc[0]
            interpolated_resampled_data["series_num"] = group['series_num'].iloc[0]
            resampled_data_list.append(interpolated_resampled_data)

    resampled_tracking_data = pd.concat(resampled_data_list)
    # there could be duplicated data in the same time_from_half_start, and we need to make it uniquem
    resampled_tracking_data = resampled_tracking_data.drop_duplicates(
        subset=['time_from_half_start', 'half', 'home_away', 'jersey_number'], keep='last'
    )

    resampled_tracking_data = resampled_tracking_data.sort_values(
        by=['half', 'time_from_half_start', 'home_away', 'jersey_number']
    )[
        [
            'game_id',
            "half",
            "series_num",
            "time_from_half_start",
            "home_away",
            "jersey_number",
            "x",
            "y",
        ]
    ].reset_index(
        drop=True
    )
    assert (
        resampled_tracking_data.query("home_away != 'BALL'")
        .groupby(['time_from_half_start', 'half', 'home_away'])['jersey_number']
        .nunique()
        != 11
    ).sum() == 0, f'''
    {(
        resampled_tracking_data.query("home_away != 'BALL'")
        .groupby(['time_from_half_start', 'half', 'home_away'])['jersey_number']
        .nunique() != 11
    ).sum()} is not 0, game_id: {resampled_tracking_data['game_id'].iloc[0]}
    '''
    assert (
        resampled_tracking_data.query("home_away == 'BALL'").groupby('time_from_half_start').value_counts() != 1
    ).sum() == 0
    assert resampled_tracking_data[['x', 'y']].isna().sum().sum() == 0
    return resampled_tracking_data


def format_tracking_data(
    tracking_data: pd.DataFrame,
    home_team_name: str,
    away_team_name: str,
    player_dict: Dict[Tuple[str, str], Dict]
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
    for _, group in tracking_data.groupby(['half', 'time_from_half_start']):
        frame_dict = {
            "time_from_half_start": round(group['time_from_half_start'].values[0], 1),
            "half": group['half'].values[0],
            "ball": None,
            "players": [],
        }
        for _, d in group.iterrows():
            if d['jersey_number'] == 0:
                frame_dict['ball'] = {"position": {"x": d['x'], "y": d['y']}}
            else:
                home_away_str = d['home_away']
                jersey_number = d['jersey_number']
                frame_dict['players'].append(
                    {
                        "team_name": home_team_name if home_away_str == 'HOME' else away_team_name,
                        "player_name": player_dict[home_away_str, jersey_number]['player_name']
                        if jersey_number > 0
                        else None,
                        "player_id": player_dict[home_away_str, jersey_number]['player_id']
                        if jersey_number > 0
                        else jersey_number,
                        "player_role": player_dict[home_away_str, jersey_number]['player_role']
                        if jersey_number > 0
                        else None,
                        "jersey_number": jersey_number,
                        "position": {"x": d['x'], "y": d['y']},
                    }
                )
        tracking_list.append(frame_dict)

    return pd.DataFrame(tracking_list)


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
    for _, group in tracking_data.groupby(['half', 'time_from_half_start']):
        frame_dict = {
            "time_from_half_start": round(group['time_from_half_start'].values[0], 1),
            "half": group['half'].values[0],
            "ball": None,
            "players": [],
        }
        for _, d in group.iterrows():
            if d['jersey_number'] == 0:
                frame_dict['ball'] = {"position": {"x": d['x'], "y": d['y']}}
            elif d['jersey_number'] < 0:
                home_away_str = d['home_away']
                frame_dict['players'].append(
                    {
                        "team_name": home_team_name if home_away_str == 'HOME' else away_team_name,
                        "player_name": None,
                        "player_id": None,
                        "player_role": None,
                        "jersey_number": d['jersey_number'],
                        "height": None,
                        "position": {"x": d['x'], "y": d['y']},
                    }
                )
            else:
                home_away_str = d['home_away']
                jersey_number = d['jersey_number']
                frame_dict['players'].append(
                    {
                        "team_name": home_team_name if home_away_str == 'HOME' else away_team_name,
                        "player_name": player_dict[home_away_str, jersey_number]['player_name']
                        if jersey_number > 0
                        else None,
                        "player_id": player_dict[home_away_str, jersey_number]['player_id']
                        if jersey_number > 0
                        else jersey_number,
                        "player_role": player_dict[home_away_str, jersey_number]['player_role']
                        if jersey_number > 0
                        else None,
                        "jersey_number": jersey_number
                        if jersey_number > 0
                        else None,
                        "height": player_dict[home_away_str, jersey_number]['height'],
                        "position": {"x": d['x'], "y": d['y']},
                    }
                )
        tracking_list.append(frame_dict)

    return pd.DataFrame(tracking_list)


def parse_tracking_data(x):
    if x is None:
        return {}
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        print(f"Warning: Unable to parse JSON: {x}")
        return {}


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
            player2pos[player['player_name']] = player['position']
        return player2pos

    def __get_player2vel(player_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        player2vel = {}
        for player in player_data:
            player2vel[player['player_name']] = player['velocity']
        return player2vel

    tracking_data = tracking_data.copy()
    time_delta = 1 / sampling_rate

    for idx, data in tracking_data.iterrows():
        ball_pos = deepcopy(data['ball'])
        player_data = deepcopy(data['players'])
        current_time_from_half_start = deepcopy(data['time_from_half_start'])

        if ball_pos is None:
            continue  # or handle the None case appropriately

        if idx == len(tracking_data) - 1:
            # same as the previous frame
            prev_data = tracking_data.iloc[idx - 1]
            if (abs(current_time_from_half_start - prev_data['time_from_half_start']) - time_delta) < 1e-5:
                ball_pos["velocity"] = deepcopy(json.loads(prev_data['ball'])['velocity'])
                prev_player2vel = __get_player2vel(json.loads(prev_data['players']))
                for d in player_data:
                    d["velocity"] = deepcopy(prev_player2vel[d['player_name']])
            else:
                # singleton? -> set velocity to 0
                ball_pos["velocity"] = {"x": 0, "y": 0}
                for d in player_data:
                    d["velocity"] = {"x": 0, "y": 0}
        else:
            next_data = tracking_data.iloc[idx + 1]
            next_time_from_half_start = next_data['time_from_half_start']
            if (abs(current_time_from_half_start - next_time_from_half_start) - time_delta) < 1e-5:
                # calculate velocity using the next frame
                try:
                    ball_pos["velocity"] = {
                        "x": (next_data['ball']['position']['x'] - ball_pos["position"]["x"]) / time_delta,
                        "y": (next_data['ball']['position']['y'] - ball_pos["position"]["y"]) / time_delta,
                    }
                except:
                    logger.warning(f"ball is not in next_data")
                    ball_pos["velocity"] = {"x": 0, "y": 0}
                next_player2pos = __get_player2pos(next_data['players'])
                for d in player_data:
                    if d['player_name'] in next_player2pos:
                        d["velocity"] = {
                            "x": (next_player2pos[d['player_name']]["x"] - d["position"]["x"]) / time_delta,
                            "y": (next_player2pos[d['player_name']]["y"] - d["position"]["y"]) / time_delta,
                        }
                    else:
                        logger.warning(f"{d['player_name']} is not in next_player2pos")
                        d["velocity"] = {"x": 0, "y": 0}
            else:
                prev_data = tracking_data.iloc[idx - 1]
                if (abs(current_time_from_half_start - prev_data['time_from_half_start']) - time_delta) < 1e-5:
                    # same as the previous frame
                    ball_pos["velocity"] = deepcopy(json.loads(prev_data['ball'])['velocity'])
                    prev_player2vel = __get_player2vel(json.loads(prev_data['players']))
                    for d in player_data:
                        try:
                            d["velocity"] = deepcopy(prev_player2vel[d['player_name']])
                        except:
                            print(f"prev_data: {prev_data}.") # add for pytest
                            continue
                else:
                    # singleton? -> set velocity to 0
                    ball_pos["velocity"] = {"x": 0, "y": 0}
                    for d in player_data:
                        d["velocity"] = {"x": 0, "y": 0}
        tracking_data.loc[idx, 'ball'] = json.dumps(ball_pos)
        tracking_data.loc[idx, 'players'] = json.dumps(player_data)

    tracking_data['ball'] = tracking_data['ball'].apply(parse_tracking_data)
    tracking_data['players'] = tracking_data['players'].apply(parse_tracking_data)
    return tracking_data


def calculate_acceleration(tracking_data: pd.DataFrame, sampling_rate: int = 10) -> pd.DataFrame:
    """
    This function calculates the acceleration of each player and the ball.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    sampling_rate (int, optional): Sampling rate of the tracking data. Defaults to 10.

    Returns:
    pd.DataFrame: DataFrame with acceleration
    """
    
    def __get_player2vel(player_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        player2vel = {}
        for player in player_data:
            if 'velocity' not in player:
                player2vel[player['player_name']] = {"x": 0, "y": 0}
            else:
                player2vel[player['player_name']] = player['velocity']
        return player2vel

    tracking_data = tracking_data.copy()
    time_delta = 1 / sampling_rate

    for idx, data in tracking_data.iterrows():
        ball_pos = deepcopy(data['ball'])
        player_data = deepcopy(data['players'])
        current_time_from_half_start = deepcopy(data['time_from_half_start'])

        if ball_pos is None:
            continue  # or handle the None case appropriately

        if idx == len(tracking_data) - 1:
            # Same as the previous frame
            prev_data = tracking_data.iloc[idx - 1]
            if (abs(current_time_from_half_start - prev_data['time_from_half_start']) - time_delta) < 1e-5:
                try:
                    ball_pos["acceleration"] = deepcopy(json.loads(prev_data['ball'])['acceleration'])
                except:
                    continue
                prev_player2vel = __get_player2vel(json.loads(prev_data['players']))
                for d in player_data:
                    d["acceleration"] = deepcopy(prev_player2vel[d['player_name']])
            else:
                # Singleton? -> set acceleration to 0
                ball_pos["acceleration"] = {"x": 0, "y": 0}
                for d in player_data:
                    d["acceleration"] = {"x": 0, "y": 0}
        else:
            next_data = tracking_data.iloc[idx + 1]
            next_time_from_half_start = next_data['time_from_half_start']
            if (abs(current_time_from_half_start - next_time_from_half_start) - time_delta) < 1e-5:
                # Calculate acceleration using the next frame
                try:
                    ball_pos["acceleration"] = {
                        "x": (next_data['ball']['velocity']['x'] - ball_pos["velocity"]["x"]) / time_delta,
                        "y": (next_data['ball']['velocity']['y'] - ball_pos["velocity"]["y"]) / time_delta,
                    }
                except:
                    continue # add for pytest
                
                next_player2vel = __get_player2vel(next_data['players'])
                for d in player_data:
                    if d['player_name'] in next_player2vel:
                        d["acceleration"] = {
                            "x": (next_player2vel[d['player_name']]["x"] - d["velocity"]["x"]) / time_delta,
                            "y": (next_player2vel[d['player_name']]["y"] - d["velocity"]["y"]) / time_delta,
                        }
                    else:
                        d["acceleration"] = {"x": 0, "y": 0}
            else:
                prev_data = tracking_data.iloc[idx - 1]
                if (abs(current_time_from_half_start - prev_data['time_from_half_start']) - time_delta) < 1e-5:
                    try:    
                        ball_pos["acceleration"] = deepcopy(json.loads(prev_data['ball'])['acceleration'])
                        prev_player2vel = __get_player2vel(json.loads(prev_data['players']))
                    except:
                        print(f"prev_data: {prev_data['ball']}")
                        continue
                
                    for d in player_data:
                        try:
                            d["acceleration"] = deepcopy(prev_player2vel[d['player_name']])
                        except:
                            print(f"prev_data: {prev_data}.")
                            continue
                else:
                    # Singleton? -> set acceleration to 0
                    ball_pos["acceleration"] = {"x": 0, "y": 0}
                    for d in player_data:
                        d["acceleration"] = {"x": 0, "y": 0}

        tracking_data.loc[idx, 'ball'] = json.dumps(ball_pos)
        tracking_data.loc[idx, 'players'] = json.dumps(player_data)

    tracking_data['ball'] = tracking_data['ball'].apply(parse_tracking_data)
    tracking_data['players'] = tracking_data['players'].apply(parse_tracking_data)
    return tracking_data

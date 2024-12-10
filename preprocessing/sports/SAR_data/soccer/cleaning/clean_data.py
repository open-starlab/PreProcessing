import logging
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from soccer.constant import HOME_AWAY_MAP, PLAYER_ROLE_MAP

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def clean_player_data(player_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the player data by filtering out players who did not participate in the game.
    It also maps the home and away teams to their respective identifiers.

    Parameters:
    player_data (pd.DataFrame): DataFrame containing player data

    Returns:
    pd.DataFrame: Cleaned player data
    """

    player_data = player_data.query("on_pitch == 1")
    player_data.loc[:, 'home_away'] = player_data['home_away'].apply(lambda x: HOME_AWAY_MAP[x])
    player_data.loc[:, 'player_role'] = player_data['player_role'].apply(lambda x: PLAYER_ROLE_MAP[x])
    player_data = player_data[
        ['team_id', 'home_away', 'player_id', 'player_name', 'player_role', 'jersey_number', 'height', 'starting_member']
    ]
    return player_data


def merge_tracking_and_event_data(tracking_data: pd.DataFrame, event_data: pd.DataFrame, state_def) -> List[Dict[str, Any]]:
    """
    This function merges the tracking data and event data.
    Merge operation is done on the half and time_from_half_start columns.

    Parameters:
    tracking_data (pd.DataFrame): DataFrame containing tracking data
    event_data (pd.DataFrame): DataFrame containing event data

    Returns:
    List[Dict[str, Any]]: List of dictionaries containing the merged tracking and event data
    """

    event_columns_CDS = [
        'game_id',
        'frame_id',
        'half',
        'time_from_half_start',
        'event_id',
        'event_name',
        'event_x',
        'event_y',
        'team_id',
        'team_name',
        'home_away',
        'player_id',
        'player_name',
        'jersey_number',
        'player_role',
        'attack_history_num',
        'attack_direction',
        'series_num',
        'ball_touch',
        'success',
        'history_num',
        'attack_start_history_num',
        'attack_end_history_num',
        'is_goal',
        'is_shot',
        'is_pass',
        'is_cross',
        'is_through_pass',
        'is_dribble',
        'ball',
        'players',
    ]
    
    event_columns_LDS = [
        'game_id',
        'frame_id',
        'half',
        'time_from_half_start',
        'event_id',
        'event_name',
        'event_x',
        'event_y',
        'team_id',
        'team_name',
        'home_away',
        'player_id',
        'player_name',
        'jersey_number',
        'player_role',
        'attack_history_num',
        'attack_direction',
        'series_num',
        'ball_touch',
        'success',
        'history_num',
        'attack_start_history_num',
        'attack_end_history_num',
        'is_goal',
        'is_shot',
        'is_pass',
        'is_dribble',
        "is_ball_recovery",
        "is_block",
        "is_interception",
        "is_clearance",
        'is_cross',
        'is_through_pass',
        'ball',
        'players',
    ]
    
    frame_df_columns_CDS = [
        'frame_id',
        'event_id',
        'team_id',
        'player_id',
        'jersey_number',
        'ball_touch',
        'success',
        'history_num',
        'is_goal',
        'is_shot',
        'is_pass',
        'is_cross',
        'is_through_pass',
        'is_dribble',
    ]
    
    frame_df_columns_LDS = [
        'frame_id',
        'event_id',
        'team_id',
        'player_id',
        'jersey_number',
        'ball_touch',
        'success',
        'history_num',
        'is_goal',
        'is_shot',
        'is_pass',
        'is_dribble',
        "is_ball_recovery",
        "is_block",
        "is_interception",
        "is_clearance",
        'is_cross',
        'is_through_pass',
    ]
    
    if state_def == 'CDS':
        frame_df = pd.merge(tracking_data, event_data, on=["half", 'time_from_half_start'], how='left')[
            event_columns_CDS
        ].reset_index(drop=True)
    
        frame_df[
            frame_df_columns_CDS
        ] = (
            frame_df[
                frame_df_columns_CDS
            ]
            .fillna(-1)
            .astype(int)
        )
    elif state_def == "LDS":
        frame_df = pd.merge(tracking_data, event_data, on=["half", 'time_from_half_start'], how='left')[
            event_columns_LDS
        ].reset_index(drop=True)
    
        frame_df[
            frame_df_columns_LDS
        ] = (
            frame_df[
                frame_df_columns_LDS
            ]
            .fillna(-1)
            .astype(int)
        )


    frame_df['half'] = frame_df['half'].fillna(method='ffill').fillna(method='bfill')
    frame_df['state'] = frame_df.apply(lambda x: {"ball": x['ball'], "players": x['players']}, axis=1)
    frame_df[
        [
            'game_id',
            'attack_history_num',
            'series_num',
            'attack_start_history_num',
            'attack_end_history_num',
            'attack_direction',
        ]
    ] = (
        frame_df[
            [
                'game_id',
                'attack_history_num',
                'series_num',
                'attack_start_history_num',
                'attack_end_history_num',
                'attack_direction',
            ]
        ]
        .fillna(method='ffill')
        .fillna(method='bfill')
        .astype(int)
    )
    frame_df.loc[:, 'attack_direction'] = frame_df.groupby(['half', 'attack_start_history_num'])[
        'attack_direction'
    ].transform(lambda x: x.value_counts().index[0])
    frame_df[['event_name', 'player_role']] = frame_df[['event_name', 'player_role']].astype(str).replace('nan', None)
    frame_df = frame_df.fillna(np.nan).replace(np.nan, None)
    frame_df = frame_df.drop(['ball', 'players'], axis=1)
    frame_df = frame_df.sort_values(by=["half", "time_from_half_start"]).reset_index(drop=True)
    return frame_df.to_dict(orient='records')

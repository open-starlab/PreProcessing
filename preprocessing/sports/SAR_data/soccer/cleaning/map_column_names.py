from typing import Dict

import pandas as pd

from ..constant import (
    INPUT_EVENT_COLUMNS, 
    INPUT_PLAYER_COLUMNS, 
    INPUT_TRACKING_COLUMNS, 
    INPUT_EVENT_COLUMNS_LALIGA
)


def check_and_rename_event_columns(event_data: pd.DataFrame, event_columns_mapping: Dict[str, str], league: str) -> pd.DataFrame:
    print("Actual columns:", event_data.columns.tolist())
    print("Expected columns:", list(event_columns_mapping.values()))

    # import pdb; pdb.set_trace()
    if league == "jleague":
        assert set(event_columns_mapping.keys()) == set(
            INPUT_EVENT_COLUMNS
        ), f"{set(event_columns_mapping.keys()).symmetric_difference(set(INPUT_EVENT_COLUMNS))}"
        event_data = event_data.rename(columns={v: k for k, v in event_columns_mapping.items()})[
            INPUT_EVENT_COLUMNS
        ]  # type: ignore
    elif league == "laliga":
        assert set(event_columns_mapping.keys()) == set(
            INPUT_EVENT_COLUMNS_LALIGA
        ), f"{set(event_columns_mapping.keys()).symmetric_difference(set(INPUT_EVENT_COLUMNS_LALIGA))}"
        event_data = event_data.rename(columns={v: k for k, v in event_columns_mapping.items()})[
            INPUT_EVENT_COLUMNS_LALIGA
        ]
    
    return event_data


def check_and_rename_tracking_columns(
    tracking_data: pd.DataFrame, tracking_columns_mapping: Dict[str, str]
) -> pd.DataFrame:
    assert set(tracking_columns_mapping.keys()) == set(
        INPUT_TRACKING_COLUMNS
    ), f"{set(tracking_columns_mapping.keys()).symmetric_difference(set(INPUT_TRACKING_COLUMNS))}"
    tracking_data = tracking_data.rename(columns={v: k for k, v in tracking_columns_mapping.items()})[
        INPUT_TRACKING_COLUMNS
    ]  # type: ignore
    return tracking_data


def check_and_rename_player_columns(player_data: pd.DataFrame, player_columns_mapping: Dict[str, str]) -> pd.DataFrame:    
    if str(player_data['試合ID'].iloc[0])[:4] == "2019" or str(player_data['試合ID'].iloc[0])[:4] == "2020":
        player_data['試合ポジションID'] = -1
    
    assert set(player_columns_mapping.keys()) == set(
        INPUT_PLAYER_COLUMNS
    ), f"{set(player_columns_mapping.keys()).symmetric_difference(set(INPUT_PLAYER_COLUMNS))}"
    player_data = player_data.rename(columns={v: k for k, v in player_columns_mapping.items()})[
        INPUT_PLAYER_COLUMNS
    ]  # type: ignore

    return player_data


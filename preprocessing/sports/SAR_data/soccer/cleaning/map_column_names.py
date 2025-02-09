from typing import Dict

import pandas as pd

from football_markov.constant import (
    INPUT_EVENT_COLUMNS, 
    INPUT_PLAYER_COLUMNS, 
    INPUT_TRACKING_COLUMNS, 
    INPUT_EVENT_COLUMNS_LALIGA, 
    INPUT_PLAYER_COLUMNS_LALIGA,
    INPUT_EVENT_COLUMNS_LDS,
    INPUT_PLAYER_COLUMNS_LDS
)


def check_and_rename_event_columns(event_data: pd.DataFrame, event_columns_mapping: Dict[str, str], state_def) -> pd.DataFrame:
    print("Actual columns:", event_data.columns.tolist())
    print("Expected columns:", list(event_columns_mapping.values()))

    # import pdb; pdb.set_trace()

    if state_def == 'CDS':        
        assert set(event_columns_mapping.keys()) == set(
            INPUT_EVENT_COLUMNS
        ), f"{set(event_columns_mapping.keys()).symmetric_difference(set(INPUT_EVENT_COLUMNS))}"
        event_data = event_data.rename(columns={v: k for k, v in event_columns_mapping.items()})[
            INPUT_EVENT_COLUMNS
        ]  # type: ignore
    elif state_def == "LDS":
        assert set(event_columns_mapping.keys()) == set(
            INPUT_EVENT_COLUMNS_LDS
        ), f"{set(event_columns_mapping.keys()).symmetric_difference(set(INPUT_EVENT_COLUMNS_LALIGA))}"
        event_data = event_data.rename(columns={v: k for k, v in event_columns_mapping.items()})[
            INPUT_EVENT_COLUMNS_LDS
        ]
    else:
        raise ValueError(f"Invalid state_def: {state_def}")
    
    return event_data

def check_and_rename_event_columns_laliga(event_data: pd.DataFrame, event_columns_mapping: Dict[str, str]) -> pd.DataFrame:
    print("Actual columns:", event_data.columns.tolist())
    print("Expected columns:", list(event_columns_mapping.values()))

    if 'フォーメーション' not in event_data.columns:
        print(f"Formation column not found in event data. match_id = {event_data['試合ID'].unique()}")
        raise KeyError(f"Formation column not found in event data. match_id = {event_data['試合ID'].unique()}")

    match_id = event_data['試合ID'].unique()
    assert set(event_columns_mapping.keys()) == set(
        INPUT_EVENT_COLUMNS_LALIGA
    ), f"Column mismatch for match_id {match_id}: {set(event_columns_mapping.keys()).symmetric_difference(set(INPUT_EVENT_COLUMNS_LALIGA))}"
    
    event_data = event_data.rename(columns={v: k for k, v in event_columns_mapping.items()})[
        INPUT_EVENT_COLUMNS_LALIGA
    ]  # type: ignore
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


def check_and_rename_player_columns(player_data: pd.DataFrame, player_columns_mapping: Dict[str, str], state_def) -> pd.DataFrame:
    if state_def == 'CDS':
        assert set(player_columns_mapping.keys()) == set(
            INPUT_PLAYER_COLUMNS
        ), f"{set(player_columns_mapping.keys()).symmetric_difference(set(INPUT_PLAYER_COLUMNS))}"
        player_data = player_data.rename(columns={v: k for k, v in player_columns_mapping.items()})[
            INPUT_PLAYER_COLUMNS
        ]  # type: ignore
    elif state_def == "LDS":
        assert set(player_columns_mapping.keys()) == set(
            INPUT_PLAYER_COLUMNS_LDS
        ), f"{set(player_columns_mapping.keys()).symmetric_difference(set(INPUT_PLAYER_COLUMNS))}"
        player_data = player_data.rename(columns={v: k for k, v in player_columns_mapping.items()})[
            INPUT_PLAYER_COLUMNS_LDS
        ]
    else:
        raise ValueError(f"Invalid state_def: {state_def}")

    return player_data

def check_and_rename_player_columns_laliga(player_data: pd.DataFrame, player_columns_mapping: Dict[str, str]) -> pd.DataFrame:
    assert set(player_columns_mapping.keys()) == set(
        INPUT_PLAYER_COLUMNS_LALIGA
    ), f"{set(player_columns_mapping.keys()).symmetric_difference(set(INPUT_PLAYER_COLUMNS))}"
    player_data = player_data.rename(columns={v: k for k, v in player_columns_mapping.items()})[
        INPUT_PLAYER_COLUMNS_LALIGA
    ]  # type: ignore
    return player_data

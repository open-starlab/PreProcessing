from typing import Dict, List
import pandas as pd

from preprocessing.sports.SAR_data.soccer.constant import (
    INPUT_EVENT_COLUMNS_JLEAGUE,
    INPUT_EVENT_COLUMNS_LALIGA,
    INPUT_PLAYER_COLUMNS_PVS,
    INPUT_PLAYER_COLUMNS_EDMS,
    INPUT_TRACKING_COLUMNS,
)


def _validate_columns(columns_mapping: Dict[str, str], expected_columns: List[str], context: str):
    missing = set(expected_columns) - set(columns_mapping.keys())
    extra = set(columns_mapping.keys()) - set(expected_columns)
    assert not missing and not extra, f"Column mismatch for {context}: Missing columns: {missing}, Unexpected columns: {extra}"


def check_and_rename_event_columns(
    event_data: pd.DataFrame, event_columns_mapping: Dict[str, str], league: str
) -> pd.DataFrame:
    league_columns_map = {
        "jleague": INPUT_EVENT_COLUMNS_JLEAGUE,
        "laliga": INPUT_EVENT_COLUMNS_LALIGA,
    }

    if league not in league_columns_map:
        raise ValueError(f"Unsupported league: {league}")

    expected_columns = league_columns_map[league]
    _validate_columns(event_columns_mapping, expected_columns, context=f"event data ({league})")

    event_data = event_data.rename(columns={v: k for k, v in event_columns_mapping.items()})
    event_data = event_data[expected_columns]

    return event_data


def check_and_rename_event_columns_laliga(event_data: pd.DataFrame, event_columns_mapping: Dict[str, str]) -> pd.DataFrame:
    print(f"Actual columns: {event_data.columns.tolist()}")
    print(f"Expected columns for laliga: {list(event_columns_mapping.values())}")
    _validate_columns(event_columns_mapping, INPUT_EVENT_COLUMNS_LALIGA, context="event data (laliga)")
    event_data = event_data.rename(columns={v: k for k, v in event_columns_mapping.items()})
    event_data = event_data[INPUT_EVENT_COLUMNS_LALIGA]
    return event_data


def check_and_rename_event_columns_jleague(event_data: pd.DataFrame, event_columns_mapping: Dict[str, str]) -> pd.DataFrame:
    print(f"Actual columns: {event_data.columns.tolist()}")
    print(f"Expected columns for jleague: {list(event_columns_mapping.values())}")
    _validate_columns(event_columns_mapping, INPUT_EVENT_COLUMNS_JLEAGUE, context="event data (jleague)")
    event_data = event_data.rename(columns={v: k for k, v in event_columns_mapping.items()})
    event_data = event_data[INPUT_EVENT_COLUMNS_JLEAGUE]
    return event_data


def check_and_rename_tracking_columns(tracking_data: pd.DataFrame, tracking_columns_mapping: Dict[str, str]) -> pd.DataFrame:
    _validate_columns(tracking_columns_mapping, INPUT_TRACKING_COLUMNS, context="tracking data")

    tracking_data = tracking_data.rename(columns={v: k for k, v in tracking_columns_mapping.items()})
    tracking_data = tracking_data[INPUT_TRACKING_COLUMNS]

    return tracking_data


def check_and_rename_player_columns(
    player_data: pd.DataFrame, player_columns_mapping: Dict[str, str], state: str
) -> pd.DataFrame:
    match_id_prefix = str(player_data["試合ID"].iloc[0])[:4]
    if match_id_prefix in ("2019", "2020"):
        player_data["試合ポジションID"] = -1

    state_columns_map = {
        "PVS": INPUT_PLAYER_COLUMNS_PVS,
        "EDMS": INPUT_PLAYER_COLUMNS_EDMS,
    }

    expected_columns = state_columns_map[state]
    _validate_columns(player_columns_mapping, expected_columns, context="player data")
    player_data = player_data.rename(columns={v: k for k, v in player_columns_mapping.items()})
    player_data = player_data[expected_columns]
    return player_data

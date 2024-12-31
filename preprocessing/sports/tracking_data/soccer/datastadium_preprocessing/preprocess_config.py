# -*- coding: utf-8 -*-
"""Configuration of the SPADL language.

Attributes
----------
field_length : float
    The length of a pitch (in meters).
field_width : float
    The width of a pitch (in meters).
bodyparts : list(str)
    The bodyparts used in the SPADL language.
results : list(str)
    The action results used in the SPADL language.
actiontypes : list(str)
    The action types used in the SPADL language.

"""
from typing import List

import pandas as pd  # type: ignore
import re

FOOTBALL_PLAYER_NUM = 11
SUBSTITUTION_NUM = 3

FIELD_LENGTH: float = 105.0  # unit: meters
FIELD_WIDTH: float = 68.0  # unit: meters

# The following isn't regarding spadl config but regarding J League data config.
TRACKING_HERZ = 25

bodyparts: List[str] = ["foot", "head", "other", "head/other"]
results: List[str] = [
    "fail",
    "success",
    "offside",
    "owngoal",
    "yellow_card",
    "red_card",
]
actiontypes: List[str] = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "take_on",
    "foul",
    "tackle",
    "interception",
    "shot",
    "shot_penalty",
    "shot_freekick",
    "keeper_save",
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
    "clearance",
    "bad_touch",
    "non_action",
    "dribble",
    "goalkick",
    # "recovery", # add
]
defense_actiontypes: List[str] = [
    "foul",  # adjustable
    "tackle",
    "interception",
    "keeper_save",
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
    "clearance",
    "block",
]


def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'type_id' and 'type_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])


def results_df() -> pd.DataFrame:
    """Return a dataframe with the result id and result name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'result_id' and 'result_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(results)), columns=["result_id", "result_name"])


def bodyparts_df() -> pd.DataFrame:
    """Return a dataframe with the bodypart id and bodypart name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'bodypart_id' and 'bodypart_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(bodyparts)), columns=["bodypart_id", "bodypart_name"])


def str2int(text: str) -> int:
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> list:
    return [str2int(c) for c in re.split(r"(\d+)", text)]
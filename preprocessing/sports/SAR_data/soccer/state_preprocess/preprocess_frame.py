import logging
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.sports.SAR_data.soccer.constant import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    STOP_THRESHOLD,
    LALIGA_VALID_EVENTS,
    JLEAGUE_VALID_EVENTS,
)
from preprocessing.sports.SAR_data.soccer.dataclass import (
    Ball,
    Event_PVS,
    Events_PVS,
    Player,
    Position,
    State_PVS,
    Event_EDMS,
    Events_EDMS,
    State_EDMS,
    RelativeState,
    RawState,
)
from preprocessing.sports.SAR_data.soccer.state_preprocess.state_edms import calc_absolute_state, calc_offball, calc_onball
from preprocessing.sports.SAR_data.soccer.state_preprocess.reward_model import RewardModelBase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def discretize_direction(velocity_x: float, velocity_y: float) -> str:
    """
    Discretize the direction of the ball/player into 8 directions
        - idle: 0 (when velocity is below STOP_THRESHOLD)
        - right: 1
        - up_right 2
        - up: 3
        - up_left: 4
        - left: 5
        - down_left: 6
        - down: 7
        - down_right: 8
    """

    # if velocity is below threshold, then idle
    if np.sqrt(velocity_x**2 + velocity_y**2) < STOP_THRESHOLD:
        return "idle"

    # calculate angle
    angle = np.arctan2(velocity_y, velocity_x)
    angle = np.rad2deg(angle)
    angle = (angle + 360) % 360

    # discretize angle into 8 directions
    if 22.5 <= angle < 67.5:
        direction = "up_right"
    elif 67.5 <= angle < 112.5:
        direction = "up"
    elif 112.5 <= angle < 157.5:
        direction = "up_left"
    elif 157.5 <= angle < 202.5:
        direction = "left"
    elif 202.5 <= angle < 247.5:
        direction = "down_left"
    elif 247.5 <= angle < 292.5:
        direction = "down"
    elif 292.5 <= angle < 337.5:
        direction = "down_right"
    else:
        direction = "right"

    return direction


def last_attack_event_in_frames(frames: pd.DataFrame, data_type: str) -> pd.Series | None:
    """
    Find the last attack event in frames

    Parameters
    ----------
    frames : pd.DataFrame
        Event frames
    data_type : str
        Type of data source (laliga or jleague)

    Returns
    -------
    pd.Series or None
        Last event or None if no valid events found
    """
    # Use pre-defined event lists
    valid_event_names = LALIGA_VALID_EVENTS if data_type == "laliga" else JLEAGUE_VALID_EVENTS

    # Use query for better performance with larger dataframes
    filtered_frames = frames[frames["event_name"].isin(valid_event_names)]

    return filtered_frames.iloc[-1] if not filtered_frames.empty else None


def find_goal_keeper_after_shot(frames: pd.DataFrame, shot_index: int, data_type: str) -> pd.Series | None:
    """
    Find Goal Keeper event after a shot event

    Parameters
    ----------
    frames : pd.DataFrame
        Event frames
    shot_index : int
        Index of the shot event
    data_type : str
        Type of data source (laliga or jleague)

    Returns
    -------
    pd.Series or None
        Goal Keeper event or None if not found
    """
    # Define Goal Keeper event names based on data type
    if data_type == "laliga":
        goal_keeper_events = ["Goal Keeper"]
    elif data_type == "jleague":
        goal_keeper_events = ["Goal Keeper"]
    else:
        goal_keeper_events = ["Goal Keeper"]

    # Search for Goal Keeper events after the shot
    frames_after_shot = frames.iloc[shot_index + 1 :]
    goal_keeper_frames = frames_after_shot[frames_after_shot["event_name"].isin(goal_keeper_events)]

    return goal_keeper_frames.iloc[0] if not goal_keeper_frames.empty else None


def get_action_from_event(frame: pd.Series, data_type: str) -> str | None:
    """
    Get action from event data

    Parameters
    ----------
    frame : pd.Series
        Frame data
    data_type : str
        Type of data source (laliga or jleague)

    Returns
    -------
    str or None
        Action string or None if no action found
    """
    try:
        if data_type == "laliga":
            if frame.get("is_goal", False):
                return "goal"
            elif frame.get("is_shot", False):
                return "shot"
            elif frame.get("is_dribble", False):
                return "dribble"
            elif frame.get("is_pressure", False):
                return "pressure"
            elif frame.get("is_ball_recovery", False):
                return "ball_recovery"
            elif frame.get("is_interception", False):
                return "interception"
            elif frame.get("is_clearance", False):
                return "clearance"
            elif frame.get("is_pass", False):
                return "pass"
            else:
                return None
        elif data_type == "jleague":
            if frame.get("is_goal", False):
                return "goal"
            elif frame.get("is_shot", False):
                return "shot"
            elif frame.get("is_dribble", False):
                return "dribble"
            elif frame.get("is_ball_recovery", False):
                return "ball_recovery"
            elif frame.get("is_interception", False):
                return "interception"
            elif frame.get("is_clearance", False):
                return "clearance"
            elif frame.get("is_cross", False):
                return "cross"
            elif frame.get("is_through_pass", False):
                return "through_pass"
            elif frame.get("is_pass", False):
                return "pass"
            else:
                return None
        else:
            logger.warning(f"Unknown data_type: {data_type}")
            return None
    except KeyError as e:
        logger.debug(f"KeyError in get_action_from_event: {e}")
        # Minimal fallback for known events when fields are missing
        if data_type == "jleague":
            if frame.get("is_goal", False):
                return "goal"
            elif frame.get("is_shot", False):
                return "shot"
            elif frame.get("is_dribble", False):
                return "dribble"
            elif frame.get("is_cross", False):
                return "cross"
            elif frame.get("is_through_pass", False):
                return "through_pass"
            elif frame.get("is_pass", False):
                return "pass"
        return None
    except Exception as e:
        logger.error(f"Error determining action: {e}")
        return None


def opponent_goal_position(origin_pos: str, absolute_coordinates: bool, attack_direction: int) -> Position:
    if absolute_coordinates:
        if attack_direction == 1:
            return Position(x=FIELD_LENGTH / 2, y=0) if origin_pos == "center" else Position(x=FIELD_LENGTH, y=FIELD_WIDTH / 2)
        else:
            return Position(x=-FIELD_LENGTH / 2, y=0) if origin_pos == "center" else Position(x=0, y=FIELD_WIDTH / 2)
    else:
        return Position(x=FIELD_LENGTH / 2, y=0) if origin_pos == "center" else Position(x=FIELD_LENGTH, y=FIELD_WIDTH / 2)


class InvalidPlayerIDException(Exception):
    pass


class InvalidBallException(Exception):
    """Exception raised for invalid ball data in the state."""

    pass


def _prepare_players(frame, team_name_attack, goal_position, league):
    """
    Common function to prepare player data for both state types

    Parameters
    ----------
    frame : pd.Series
        Frame data
    team_name_attack : str
        Name of attacking team
    goal_position : Position
        Position of the goal
    league : str
        League name

    Returns
    -------
    tuple
        Tuple of (players_with_action, attack_players, defense_players, player_onball)
    """
    state = frame["state"]

    # Create role priority mapping
    # role_priority = {"GK": 1, "DF": 2, "MF": 3, "FW": 4}

    # Pre-compute distances for better sorting performance
    for player in state["players"]:
        if "position" in player and "x" in player["position"] and "y" in player["position"]:
            player["_distance_to_goal"] = np.sqrt(
                (player["position"]["x"] - goal_position.x) ** 2 + (player["position"]["y"] - goal_position.y) ** 2
            )
        else:
            player["_distance_to_goal"] = float("inf")  # Place at the end if position is missing

    # Sort players using pre-computed distances
    players = sorted(
        state["players"],
        key=lambda player: (
            player["team_name"] != team_name_attack,  # attacking team first
            player.get("player_role", "") or "",  # then by role
            player["_distance_to_goal"],  # then by pre-computed distance
        ),
    )

    # Remove temp field to keep data clean
    for player in players:
        if "_distance_to_goal" in player:
            del player["_distance_to_goal"]

    players_with_action = []
    player_onball = None

    for idx, player in enumerate(players):
        # Add index
        player["index"] = idx

        # Determine action
        if frame["player_id"] == player.get("player_id"):
            player_onball = player
            player["action"] = get_action_from_event(frame, league) or discretize_direction(
                player.get("velocity", {}).get("x", 0), player.get("velocity", {}).get("y", 0)
            )
        else:
            player["action"] = discretize_direction(
                player.get("velocity", {}).get("x", 0), player.get("velocity", {}).get("y", 0)
            )

        # Handle None values
        player["player_name"] = player.get("player_name", "") or ""
        player["player_role"] = player.get("player_role", "") or ""

        # Handle missing player_id
        if "player_id" not in player or player["player_id"] is None:
            if player is player_onball:  # Critical error for player with the ball
                logger.error(f"game_id: {frame['game_id']}, frame_id: {frame['frame_id']}")
                logger.error(f"Invalid player_id in player with ball: {player}")
                raise InvalidPlayerIDException("Invalid player_id in player with ball")
            else:
                player["player_id"] = -1  # Assign placeholder ID for other players

        players_with_action.append(Player.from_dict(player))

    # Convert player_onball to Player object
    if player_onball is not None:
        player_onball = Player.from_dict(player_onball)

    # Split players by team
    attack_players = [p for p in players_with_action if p.team_name == team_name_attack]
    defense_players = [p for p in players_with_action if p.team_name != team_name_attack]

    return players_with_action, attack_players, defense_players, player_onball


def frame2state_pvs(
    frame: pd.Series,
    team_name_attack: str,
    origin_pos: str = "center",
    absolute_coordinates: bool = False,
    league: str = "jleague",
) -> State_PVS:
    state = frame["state"]

    ball = Ball.from_dict(state["ball"])
    goal_position = opponent_goal_position(
        origin_pos=origin_pos,
        absolute_coordinates=absolute_coordinates,
        attack_direction=frame["attack_direction"],
    )

    players_with_action, attack_players, defense_players, _ = _prepare_players(
        frame=frame, team_name_attack=team_name_attack, goal_position=goal_position, league=league
    )

    return State_PVS(ball=ball, players=players_with_action, attack_players=attack_players, defense_players=defense_players)


def frames2events_pvs(
    frames: pd.DataFrame,
    league: str,
    reward_model: RewardModelBase,
    origin_pos: str = "center",
    absolute_coordinates: bool = False,
    min_frame_len_threshold: int = 30,
    max_frame_len_threshold: int = 600,
) -> List[Events_PVS]:
    events_list: List[Events_PVS] = []
    attack_start_history_num_list = frames["attack_start_history_num"].unique()
    attack_start_history_num_list_len = len(attack_start_history_num_list)
    for idx in range(attack_start_history_num_list_len):
        current_attack_start_history_num = attack_start_history_num_list[idx]
        next_attack_start_history_num = (
            attack_start_history_num_list[idx + 1] if idx + 1 < attack_start_history_num_list_len else None
        )
        current_frames = frames.query(f"attack_start_history_num == {current_attack_start_history_num}").reset_index(drop=True)
        next_frames = (
            frames.query(f"attack_start_history_num == {next_attack_start_history_num}").reset_index(drop=True)
            if next_attack_start_history_num
            else None
        )
        if next_frames is not None and (current_frames.iloc[0]["half"] != next_frames.iloc[0]["half"]):
            next_frames = None

        last_attack_event = last_attack_event_in_frames(current_frames, league)
        if last_attack_event is None:
            continue
        current_frames = current_frames.iloc[: last_attack_event.name + 1]
        if len(current_frames) < min_frame_len_threshold:
            continue
        if len(current_frames) > max_frame_len_threshold:
            current_frames = current_frames.iloc[-max_frame_len_threshold:]

        team_names = list(set(list(player["team_name"] for player in current_frames.iloc[0]["state"]["players"])))
        team_name_attack = current_frames["team_name"].value_counts().index[0]
        team_name_defense = team_names[1] if team_names[0] == team_name_attack else team_names[0]

        states = current_frames.apply(
            frame2state_pvs, axis=1, args=(team_name_attack, origin_pos, absolute_coordinates, league)
        )

        if None in states.values:
            logger.warning(f"None in states: {states}")
            # remove None Values
            states = states[states.notnull()].reset_index(drop=True)

        rewards = reward_model.calculate_reward(league, current_frames, next_frames)
        events = [Event_PVS(state=state, reward=reward) for state, reward in zip(states, rewards)]

        events_list.append(
            Events_PVS(
                game_id=str(current_frames.iloc[0]["game_id"]),
                half=str(current_frames.iloc[0]["half"]),
                sequence_id=len(events_list),
                sequence_start_frame=str(current_frames.iloc[0]["time_from_half_start"]),
                sequence_end_frame=str(current_frames.iloc[-1]["time_from_half_start"]),
                team_name_attack=team_name_attack,
                team_name_defense=team_name_defense,
                events=events,
            )
        )

    return events_list


def frame2state_edms(
    frame: pd.Series,
    team_name_attack: str,
    origin_pos: str = "center",
    absolute_coordinates: bool = False,
    league: str = "laliga",
    prev_team: str = None,
) -> State_EDMS:
    state = frame["state"]
    if league == "laliga":
        formation = frame["formation"]
    try:
        ball = Ball.from_dict(state["ball"])
    except Exception as e:
        logger.error(f"game_id: {frame['game_id']}, frame_id: {frame['frame_id']}")
        logger.error(f"state: {state['ball']}")
        logger.error(f"Original error: {e}")
        raise InvalidBallException(f"Invalid ball data in state: {state['ball']}")

    goal_position = opponent_goal_position(
        origin_pos=origin_pos,
        absolute_coordinates=absolute_coordinates,
        attack_direction=frame["attack_direction"],
    )

    players_with_action, attack_players, defense_players, player_onball = _prepare_players(
        frame=frame, team_name_attack=team_name_attack, goal_position=goal_position, league=league
    )

    onball_team = prev_team if player_onball is None else player_onball.team_name
    onball, weighted_area, weighted_area_vel = calc_onball(
        players_with_action, attack_players, defense_players, player_onball, ball, goal_position, team_name_attack, onball_team
    )
    offball = calc_offball(
        players_with_action,
        attack_players,
        defense_players,
        player_onball,
        ball,
        weighted_area,
        weighted_area_vel,
        team_name_attack,
        goal_position,
    )
    relative_state = RelativeState(onball=onball, offball=offball)
    absolute_state = calc_absolute_state(
        players_with_action, ball, attack_players, defense_players, formation if league == "laliga" else None
    )
    raw_state = RawState(
        ball=ball, players=players_with_action, attack_players=attack_players, defense_players=defense_players
    )
    state = State_EDMS(relative_state=relative_state, absolute_state=absolute_state, raw_state=raw_state)
    return state, onball_team


def frames2events_edms(
    frames: pd.DataFrame,
    league: str,
    reward_model: RewardModelBase,
    origin_pos: str = "center",
    absolute_coordinates: bool = False,
    min_frame_len_threshold: int = 30,
    max_frame_len_threshold: int = 600,
) -> List[Events_EDMS]:
    events_list: List[Events_EDMS] = []
    attack_start_history_num_list = frames["attack_start_history_num"].unique()
    attack_start_history_num_list_len = len(attack_start_history_num_list)
    for idx in tqdm(range(attack_start_history_num_list_len)):
        current_attack_start_history_num = attack_start_history_num_list[idx]
        next_attack_start_history_num = (
            attack_start_history_num_list[idx + 1] if idx + 1 < attack_start_history_num_list_len else None
        )
        current_frames = frames.query(f"attack_start_history_num == {current_attack_start_history_num}").reset_index(drop=True)
        next_frames = (
            frames.query(f"attack_start_history_num == {next_attack_start_history_num}").reset_index(drop=True)
            if next_attack_start_history_num
            else None
        )
        if next_frames is not None and (current_frames.iloc[0]["half"] != next_frames.iloc[0]["half"]):
            next_frames = None

        last_attack_event = last_attack_event_in_frames(current_frames, league)
        if last_attack_event is None:
            continue

        # Check if the last attack event is a shot, and if so, look for Goal Keeper event
        end_frame_index = last_attack_event.name + 1
        if last_attack_event.get("is_shot", False):
            goal_keeper_event = find_goal_keeper_after_shot(current_frames, last_attack_event.name, league)
            if goal_keeper_event is not None:
                end_frame_index = goal_keeper_event.name + 1

        current_frames = current_frames.iloc[:end_frame_index]
        if len(current_frames) < min_frame_len_threshold:
            continue
        if len(current_frames) > max_frame_len_threshold:
            current_frames = current_frames.iloc[-max_frame_len_threshold:]

        team_names = list(set(list(player["team_name"] for player in current_frames.iloc[0]["state"]["players"])))
        team_name_attack = current_frames["team_name"].value_counts().index[0]
        try:
            team_name_defense = team_names[1] if team_names[0] == team_name_attack else team_names[0]
        except IndexError:
            print(f"Skipping attack sequence due to invalid team_names: {team_names}")
            continue

        try:
            previous_team = None
            states = []

            for idx, row in current_frames.iterrows():
                state, onball_team = frame2state_edms(
                    frame=row,
                    team_name_attack=team_name_attack,
                    origin_pos=origin_pos,
                    absolute_coordinates=absolute_coordinates,
                    league=league,
                    prev_team=previous_team,
                )
                states.append(state)
                previous_team = onball_team
            states = pd.Series(states)
        except InvalidPlayerIDException as e:
            print(f"Skipping attack sequence due to invalid player_id: {e}")
            continue
        rewards = reward_model.calculate_reward(league, current_frames, next_frames)
        events = [Event_EDMS(state=state, reward=reward) for state, reward in zip(states, rewards)]

        events_list.append(
            Events_EDMS(
                game_id=str(current_frames.iloc[0]["game_id"]),
                half=str(current_frames.iloc[0]["half"]),
                sequence_id=len(events_list),
                sequence_start_frame=str(current_frames.iloc[0]["time_from_half_start"]),
                sequence_end_frame=str(current_frames.iloc[-1]["time_from_half_start"]),
                team_name_attack=team_name_attack,
                team_name_defense=team_name_defense,
                events=events,
            )
        )

    return events_list

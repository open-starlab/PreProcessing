import logging
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from ..constant import FIELD_LENGTH, FIELD_WIDTH, STOP_THRESHOLD
from ..dataclass import Ball, Event, Events, Player, Position, RelativeState, State, RawState
from .reward_model import RewardModelBase, SimpleEPVReward
from .state import calc_absolute_state, calc_offball, calc_onball

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
    if data_type == "laliga":
        valid_event_names = [
            'Pass',
            'Shot',
            'Interception',
            'Dribble',
            'Foul Won',
            'Miscontrol',
            'Ball Receipt*',
            'Ball Recovery',
            'Pressure',
            'Block',
            'Carry',
            'Clearance',
        ]
    elif data_type == "jleague":
        valid_event_names = [
            'アウェイパス',
            'インターセプト',
            'クロス',
            'シュート',
            'スルーパス',
            'タッチ',
            'タックル',
            'ブロック',
            'クリア',
            'ボールゲイン',
            'トラップ',
            'ドリブル',
            'ファウル受ける',
            'フィード',
            'フリックオン',
            'ホームパス',
        ]
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    
    frames = frames[frames['event_name'].isin(valid_event_names)]
    if len(frames) > 0:
        return frames.iloc[-1]
    else:
        return None


def get_action_from_event(frame: pd.Series, data_type: str) -> str | None:
    if data_type == "laliga":
        if frame['is_goal']:
            return "goal"
        elif frame['is_shot']:
            return "shot"
        elif frame['is_dribble']:
            return "dribble"
        elif frame['is_pressure']:
            return "pressure"
        elif frame['is_ball_recovery']:
            return "ball_recovery"
        elif frame['is_interception']:
            return "interception"
        elif frame['is_clearance']:
            return "clearance"
        elif frame['is_pass']:
            return "pass"
        else:
            return None
    elif data_type == "jleague":
        if frame['is_goal']:
            return "goal"
        elif frame['is_shot']:
            return "shot"
        elif frame['is_dribble']:
            return "dribble"
        elif frame['is_ball_recovery']:
            return "ball_recovery"
        elif frame['is_interception']:
            return "interception"
        elif frame['is_clearance']:
            return "clearance"
        elif frame['is_cross']:
            return "cross"
        elif frame['is_through_pass']:
            return "through_pass"
        elif frame['is_pass']:
            return "pass"
        else:
            return None


def opponent_goal_position(origin_pos: str, absolute_coordinates: bool, attack_direction: int) -> Position:
    if absolute_coordinates:
        if attack_direction == 1:
            return (
                Position(x=FIELD_LENGTH / 2, y=0)
                if origin_pos == "center"
                else Position(x=FIELD_LENGTH, y=FIELD_WIDTH / 2)
            )
        else:
            return Position(x=-FIELD_LENGTH / 2, y=0) if origin_pos == "center" else Position(x=0, y=FIELD_WIDTH / 2)
    else:
        return (
            Position(x=FIELD_LENGTH / 2, y=0)
            if origin_pos == "center"
            else Position(x=FIELD_LENGTH, y=FIELD_WIDTH / 2)
        )
    
class InvalidPlayerIDException(Exception):
    pass

def frame2state(
    frame: pd.Series, team_name_attack: str, origin_pos: str = "center", absolute_coordinates: bool = False, data_type: str = "laliga"
) -> State:
    state = frame['state']
    if data_type == "laliga":
        formation = frame['formation']
    
    try:
        ball = Ball.from_dict(state['ball'])
    except:
        # show Error message
        print(f"game_id: {frame['game_id']}, frame_id: {frame['frame_id']}")
        print(f"state: {state['ball']}")
        raise InvalidPlayerIDException(f"Invalid ball in {state['ball']}")
    goal_position = opponent_goal_position(
        origin_pos=origin_pos,
        absolute_coordinates=absolute_coordinates,
        attack_direction=frame['attack_direction'],
    )
    # sort players by the distance to the goal
    players = sorted(
        state['players'],
        key=lambda player: np.sqrt(
            (player['position']['x'] - goal_position.x) ** 2 + (player['position']['y'] - goal_position.y) ** 2
        ),
    )
    players_with_action = []
    player_onball = None
    for idx, player in enumerate(players):
        player['index'] = idx
        if frame['player_id'] == player['player_id']:
            player_onball = player
            player['action'] = get_action_from_event(frame, data_type) or discretize_direction(
                player['velocity']['x'], player['velocity']['y']
            )
        else:
            player['action'] = discretize_direction(player['velocity']['x'], player['velocity']['y'])
        if player['player_name'] is None:
            player['player_name'] = ''
        if player['player_role'] is None:
            player['player_role'] = ''
        if 'player_id' not in player or player['player_id'] is None:
            print(f"game_id: {frame['game_id']}, frame_id: {frame['frame_id']}")
            print(f"Invalid player_id in {player}")
            raise InvalidPlayerIDException(f"Invalid player_id in {player}")
        else:
            players_with_action.append(Player.from_dict(player))
    attack_players = [player for player in players_with_action if player.team_name == team_name_attack]
    defense_players = [player for player in players_with_action if player.team_name != team_name_attack]
    if player_onball is not None:
        player_onball = Player.from_dict(player_onball)
    # onball_team = player_onball.team_name
    onball_team = "None"
    onball, weighted_area, weighted_area_vel = calc_onball(players_with_action, attack_players, defense_players, player_onball, ball, goal_position, team_name_attack, onball_team)
    offball = calc_offball(players_with_action, attack_players, defense_players, player_onball, ball, weighted_area, weighted_area_vel, team_name_attack)
    relative_state = RelativeState(onball=onball, offball=offball)
    absolute_state = calc_absolute_state(players_with_action, ball, attack_players, defense_players, formation if data_type == "laliga" else None)
    raw_state = RawState(ball=ball, players=players_with_action, attack_players=attack_players, defense_players=defense_players)
    state = State(relative_state=relative_state, absolute_state=absolute_state, raw_state=raw_state)
    # print(f"state: {state}")
    return state

# `frame2state` 関数の並列化処理用関数
def process_frame(row, team_name_attack, origin_pos, absolute_coordinates, data_type):
    return frame2state(row, team_name_attack, origin_pos, absolute_coordinates, data_type)

# 並列処理の適用
def parallel_frame2state(current_frames, team_name_attack, origin_pos, absolute_coordinates, data_type):
    current_frames = current_frames.reset_index(drop=True)
    results = [None] * len(current_frames) 
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(process_frame, row, team_name_attack, origin_pos, absolute_coordinates, data_type): idx
            for idx, row in current_frames.iterrows()
        }
        
        for future in concurrent.futures.as_completed(futures):
            try:
                idx = futures[future]
                result = future.result()
                results[idx] = result
            except Exception as e:
                print(f"Error: {e}")
                print(f"future: {future}")
                print(f"futures[future]: {futures[future]}")
                print(f"result: {future.result()}")
                import pdb; pdb.set_trace()
    
    return results


def frames2events(
    frames: pd.DataFrame,
    data_type: str,
    reward_model: RewardModelBase,
    origin_pos: str = "center",
    absolute_coordinates: bool = False,
    min_frame_len_threshold: int = 30,
    max_frame_len_threshold: int = 600,
) -> List[Events]:
    events_list: List[Events] = []
    attack_start_history_num_list = frames['attack_start_history_num'].unique()
    attack_start_history_num_list_len = len(attack_start_history_num_list)
    for idx in tqdm(range(attack_start_history_num_list_len)):
        current_attack_start_history_num = attack_start_history_num_list[idx]
        next_attack_start_history_num = (
            attack_start_history_num_list[idx + 1] if idx + 1 < attack_start_history_num_list_len else None
        )
        current_frames = frames.query(f"attack_start_history_num == {current_attack_start_history_num}").reset_index(
            drop=True
        )
        next_frames = (
            frames.query(f"attack_start_history_num == {next_attack_start_history_num}").reset_index(drop=True)
            if next_attack_start_history_num
            else None
        )
        if next_frames is not None and (current_frames.iloc[0]['half'] != next_frames.iloc[0]['half']):
            next_frames = None

        last_attack_event = last_attack_event_in_frames(current_frames, data_type)
        if last_attack_event is None:
            continue
        current_frames = current_frames.iloc[: last_attack_event.name + 1]
        if len(current_frames) < min_frame_len_threshold:
            continue
        if len(current_frames) > max_frame_len_threshold:
            current_frames = current_frames.iloc[-max_frame_len_threshold:]

        team_names = list(set(list(player['team_name'] for player in current_frames.iloc[0]['state']['players'])))
        team_name_attack = current_frames['team_name'].value_counts().index[0]
        try:
            team_name_defense = team_names[1] if team_names[0] == team_name_attack else team_names[0]
        except IndexError:
            print(f"Skipping attack sequence due to invalid team_names: {team_names}")
            print(f"")
            continue

        try:
            # states = parallel_frame2state(current_frames, team_name_attack, origin_pos, absolute_coordinates, data_type)
            states = current_frames.apply(frame2state, axis=1, args=(team_name_attack, origin_pos, absolute_coordinates, data_type))
        except InvalidPlayerIDException as e:
            print(f"Skipping attack sequence due to invalid player_id: {e}")
            continue
        rewards = reward_model.calculate_reward(data_type, current_frames, next_frames)
        events = [Event(state=state, reward=reward) for state, reward in zip(states, rewards)]
        # print(f"idx: {idx}, events: {events}")

        # for debubbging
        last_attack_event = last_attack_event_in_frames(current_frames, data_type)
        if current_frames.iloc[-1]['time_from_half_start'] - last_attack_event['time_from_half_start'] > 5:
            game_id = current_frames.iloc[0]['game_id']
            elapsed_time = last_attack_event['time_from_half_start'] - current_frames.iloc[0]['time_from_half_start']
            time_from_last_frame = (
                current_frames.iloc[-1]['time_from_half_start'] - last_attack_event['time_from_half_start']
            )
            logger.info(
                f"game_id: {game_id}, attack_start_history_num: {current_attack_start_history_num}\n"
                f"Last event: {last_attack_event['event_name']} (elapsed: {elapsed_time:.2f} sec, "
                f"{time_from_last_frame:.2f} sec from the last frame)"
            )

        events_list.append(
            Events(
                game_id=str(current_frames.iloc[0]['game_id']),
                half=str(current_frames.iloc[0]['half']),
                sequence_id=len(events_list),
                team_name_attack=team_name_attack,
                team_name_defense=team_name_defense,
                events=events,
            )
        )

    return events_list

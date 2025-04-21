import logging
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from preprocessing.sports.SAR_data.soccer.constant import FIELD_LENGTH, FIELD_WIDTH, STOP_THRESHOLD
from preprocessing.sports.SAR_data.soccer.dataclass import Ball, Event, Events, Player, Position, State
from preprocessing.sports.SAR_data.soccer.state_preprocess.reward_model import RewardModelBase, SimpleEPVReward

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


def last_attack_event_in_frames(frames: pd.DataFrame, league: str) -> pd.Series | None:
    if league == "laliga":
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
    elif league == "jleague":
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
        raise ValueError(f"Invalid league: {league}")
    
    frames = frames[frames['event_name'].isin(valid_event_names)]
    if len(frames) > 0:
        return frames.iloc[-1]
    else:
        return None


def get_action_from_event(frame: pd.Series, league: str) -> str | None:
    if league == "laliga":
        if 'is_goal' in frame and frame['is_goal']:
            return "goal"
        elif 'is_shot' in frame and frame['is_shot']:
            return "shot"
        elif 'is_dribble' in frame and frame['is_dribble']:
            return "dribble"
        elif 'is_pressure' in frame and frame['is_pressure']:
            return "pressure"
        elif 'is_ball_recovery' in frame and frame['is_ball_recovery']:
            return "ball_recovery"
        elif 'is_interception' in frame and frame['is_interception']:
            return "interception"
        elif 'is_clearance' in frame and frame['is_clearance']:
            return "clearance"
        elif 'is_pass' in frame and frame['is_pass']:
            return "pass"
        else:
            return None
    elif league == "jleague":
        if 'is_goal' in frame and frame['is_goal']:
            return "goal"
        elif 'is_shot' in frame and frame['is_shot']:
            return "shot"
        elif 'is_dribble' in frame and frame['is_dribble']:
            return "dribble"
        elif 'is_ball_recovery' in frame and frame['is_ball_recovery']:
            return "ball_recovery"
        elif 'is_interception' in frame and frame['is_interception']:
            return "interception"
        elif 'is_clearance' in frame and frame['is_clearance']:
            return "clearance"
        elif 'is_cross' in frame and frame['is_cross']:
            return "cross"
        elif 'is_through_pass' in frame and frame['is_through_pass']:
            return "through_pass"
        elif 'is_pass' in frame and frame['is_pass']:
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
    frame: pd.Series, team_name_attack: str, origin_pos: str = "center", absolute_coordinates: bool = False, league: str = "jleague"
) -> State:
    state = frame['state']
    ball = Ball.from_dict(state['ball'])
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
    for idx, player in enumerate(players):
        player['index'] = idx
        if frame['player_id'] == player['player_id']:
            player['action'] = get_action_from_event(frame, league) or discretize_direction(
                player['velocity']['x'], player['velocity']['y']
            )
        else:
            player['action'] = discretize_direction(player['velocity']['x'], player['velocity']['y'])
        if player['player_name'] is None:
            player['player_name'] = ''
        if player['player_role'] is None:
            player['player_role'] = ''
        if player['player_id'] is None:
            player["player_id"] = -1
        players_with_action.append(Player.from_dict(player))
    attack_players = [player for player in players_with_action if player.team_name == team_name_attack]
    defense_players = [player for player in players_with_action if player.team_name != team_name_attack]
    return State(
        ball=ball, players=players_with_action, attack_players=attack_players, defense_players=defense_players
    )

# `frame2state` 関数の並列化処理用関数
def process_frame(row, team_name_attack, origin_pos, absolute_coordinates, league):
    return frame2state(row, team_name_attack, origin_pos, absolute_coordinates, league)

# 並列処理の適用
def parallel_frame2state(current_frames, team_name_attack, origin_pos, absolute_coordinates, league):
    current_frames = current_frames.reset_index(drop=True)
    results = [None] * len(current_frames) 
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(process_frame, row, team_name_attack, origin_pos, absolute_coordinates, league): idx
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
    league: str,
    reward_model: RewardModelBase,
    origin_pos: str = "center",
    absolute_coordinates: bool = False,
    min_frame_len_threshold: int = 30,
    max_frame_len_threshold: int = 600,
) -> List[Events]:
    events_list: List[Events] = []
    attack_start_history_num_list = frames['attack_start_history_num'].unique()
    attack_start_history_num_list_len = len(attack_start_history_num_list)
    for idx in range(attack_start_history_num_list_len):
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

        last_attack_event = last_attack_event_in_frames(current_frames, league)
        if last_attack_event is None:
            continue
        current_frames = current_frames.iloc[: last_attack_event.name + 1]
        if len(current_frames) < min_frame_len_threshold:
            continue
        if len(current_frames) > max_frame_len_threshold:
            current_frames = current_frames.iloc[-max_frame_len_threshold:]

        team_names = list(set(list(player['team_name'] for player in current_frames.iloc[0]['state']['players'])))
        team_name_attack = current_frames['team_name'].value_counts().index[0]
        team_name_defense = team_names[1] if team_names[0] == team_name_attack else team_names[0]

        states = current_frames.apply(frame2state, axis=1, args=(team_name_attack, origin_pos, absolute_coordinates, league))
        rewards = reward_model.calculate_reward(league, current_frames, next_frames)
        events = [Event(state=state, reward=reward) for state, reward in zip(states, rewards)]

        # for debubbging
        last_attack_event = last_attack_event_in_frames(current_frames, league)
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
                sequence_start_frame=str(current_frames.iloc[0]['time_from_half_start']),
                sequence_end_frame=str(current_frames.iloc[-1]['time_from_half_start']),
                team_name_attack=team_name_attack,
                team_name_defense=team_name_defense,
                events=events,
            )
        )

    return events_list

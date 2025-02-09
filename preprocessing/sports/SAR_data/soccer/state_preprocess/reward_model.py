from pathlib import Path
from typing import List

import pandas as pd
from tango.common import Registrable

from preprocessing.sports.SAR_data.soccer.state_preprocess.epv import EPV

class RewardModelBase(Registrable):
    def __init__(self) -> None:
        pass

    def calculate_reward(self, current_frames: pd.DataFrame, next_frames: pd.DataFrame | None = None) -> List[float]:
        """
        Calculate the reward for a given frame and the next frame.
        :param current_frames: the current frames
        :param next_frames: the next frames
        :return: the reward
        """
        raise NotImplementedError


@RewardModelBase.register("simple_epv")
class SimpleEPVReward(RewardModelBase):
    def __init__(self, epv_path: Path | str | None = None):
        super().__init__()
        self.epv = EPV(epv_path)

    @staticmethod
    def last_attack_event_in_frames(frames: pd.DataFrame, data_type: str) -> pd.DataFrame:
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
        return frames[frames['event_name'].isin(valid_event_names)].iloc[-1]

    def calculate_reward(self, data_type: str, current_frames: pd.DataFrame, next_frames: pd.DataFrame | None = None) -> List[float]:
        """
        Calculate the reward for a given frame and the next frame.
        :param current_frames: the current frames
        :param next_frames: the next frames
        :return: the reward
        """

        current_attacking_team = current_frames['home_away'].value_counts().index[0]
        next_attacking_team = next_frames['home_away'].value_counts().index[0] if next_frames is not None else None

        if 1 in current_frames['is_goal'].values:  # if goal
            rewards = [0.0] * (len(current_frames) - 1) + [1.0]  # reward 1 for the last frame

        elif (
            next_frames is not None
            and 1 in next_frames['is_goal'].values
            and current_attacking_team != next_attacking_team
        ):  # concede goal
            rewards = [0.0] * (len(current_frames) - 1) + [-1.0]

        else:  # epv
            last_attack_event = self.last_attack_event_in_frames(current_frames, data_type)
            if last_attack_event['event_x'] is not None and last_attack_event['event_y'] is not None:
                epv = self.epv.calculate(
                    last_attack_event['event_x'], last_attack_event['event_y'], attack_direction=1
                )
            else:
                epv = self.epv.calculate(**last_attack_event['state']['ball']['position'], attack_direction=1)
            rewards = [0.0] * (len(current_frames) - 1) + [epv]

        assert len(rewards) == len(current_frames)
        return rewards

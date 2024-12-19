from typing import List
import numpy as np
import math

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Position(BaseModel):
    x: float
    y: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(x=d["x"], y=d["y"])

    def distance_to(self, other: "Position") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class Velocity(BaseModel):
    x: float
    y: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d: dict) -> "Velocity":
        return cls(x=d["x"], y=d["y"])


class Player(BaseModel):
    index: int
    team_name: str
    player_name: str
    player_id: int
    player_role: str
    position: Position
    velocity: Velocity
    action: str
    action_probs: List[float] | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "team_name": self.team_name,
            "player_name": self.player_name,
            "player_id": self.player_id,
            "player_role": self.player_role,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "action": self.action,
            "action_probs": self.action_probs or None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Player":
        return cls(
            index=d["index"],
            team_name=d["team_name"],
            player_name=d["player_name"],
            player_id=d["player_id"],
            player_role=d["player_role"],
            position=Position.from_dict(d["position"]),
            velocity=Velocity.from_dict(d["velocity"]),
            action=d["action"],
            action_probs=d["action_probs"] if "action_probs" in d else None,
        )


class Ball(BaseModel):
    position: Position
    velocity: Velocity

    def to_dict(self) -> dict:
        return {
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Ball":
        return cls(
            position=Position.from_dict(d["position"]),
            velocity=Velocity.from_dict(d["velocity"]),
        )
    
class OnBall(BaseModel):
    dist_ball_opponent: List[float]
    dribble_score: List[float]
    dribble_score_vel: List[float]
    dist_goal: List[float]
    angle_goal: List[float]
    ball_speed: float
    transition: List[float]
    shot_score: float
    long_ball_score: List[float]

    def to_dict(self) -> dict:
        return {
            "dist_ball_opponent": self.dist_ball_opponent,
            "dribble_score": self.dribble_score,
            "dribble_score_vel": self.dribble_score_vel,
            "dist_goal": self.dist_goal,
            "angle_goal": self.angle_goal,
            "ball_speed": self.ball_speed,
            "transition": self.transition,
            "shot_score": self.shot_score,
            "long_ball_score": self.long_ball_score,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "OnBall":
        return cls(
            dist_ball_opponent=d["dist_ball_opponent"],
            dribble_score=d["dribble_score"],
            dribble_score_vel=d["dribble_score_vel"],
            dist_goal=d["dist_goal"],
            angle_goal=d["angle_goal"],
            ball_speed=d["ball_speed"],
            transition=d["transition"],
            shot_score=d["shot_score"],
            long_ball_score=d["long_ball_score"],
        )

        

class OffBall(BaseModel):
    fast_space: List[float]
    fast_space_vel: List[float]
    dist_ball: List[float]
    time_to_player: List[float]
    time_to_passline: List[float]
    variation_space: List[List[float]]
    variation_space_vel: List[List[float]]
    defense_space: List[float]
    defense_space_vel: List[float]
    defense_dist_ball: List[float]

    def to_dict(self) -> dict:
        return {
            "fast_space": self.fast_space,
            "fast_space_vel": self.fast_space_vel,
            "dist_ball": self.dist_ball,
            "time_to_player": self.time_to_player,
            "time_to_passline": self.time_to_passline,
            "variation_space": self.variation_space,
            "variation_space_vel": self.variation_space_vel,
            "defense_space": self.defense_space,
            "defense_space_vel": self.defense_space_vel,
            "defense_dist_ball": self.defense_dist_ball,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "OffBall":
        return cls(
            fast_space=d["fast_space"],
            fast_space_vel=d["fast_space_vel"],
            dist_ball=d["dist_ball"],
            time_to_player=d["time_to_player"],
            time_to_passline=d["time_to_passline"],
            variation_space=d["variation_space"],
            variation_space_vel=d["variation_space_vel"],
            defense_space=d["defense_space"],
            defense_space_vel=d["defense_space_vel"],
            defense_dist_ball=d["defense_dist_ball"],
        )


class RelativeState(BaseModel):
    onball: OnBall
    offball: OffBall

    def to_dict(self) -> dict:
        return {
            "onball": self.onball.to_dict(),
            "offball": self.offball.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "OnBall":
        return cls(
            onball=OnBall.from_dict(d["onball"]),
            offball=OffBall.from_dict(d["offball"]),
        )
    

class AbsoluteState(BaseModel):
    dist_offside_line: List[float]
    formation: str
    attack_action: List[str]
    defense_action: List[str]

    def to_dict(self) -> dict:
        return {
            "dist_offside_line": self.dist_offside_line,
            "formation": self.formation,
            "attack_action": self.attack_action,
            "defense_action": self.defense_action,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "OnBall":
        return cls(
            dist_offside_line=d["dist_offside_line"],
            formation=d["formation"],
            attack_action=d["attack_action"],
            defense_action=d["defense_action"],
        )


class RawState(BaseModel):
    ball: Ball
    players: List[Player]
    attack_players: List[Player]
    defense_players: List[Player]

    @field_validator("attack_players", "defense_players")
    @classmethod
    def players_must_be_list_of_players(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("players must be a list")
        for player in v:
            if not isinstance(player, Player):
                raise TypeError("players must be a list of Player")
        return v

    def to_dict(self) -> dict:
        return {
            "ball": self.ball.to_dict(),
            "players": [player.to_dict() for player in self.players],
            "attack_players": [player.to_dict() for player in self.attack_players],
            "defense_players": [player.to_dict() for player in self.defense_players],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(
            ball=Ball.from_dict(d["ball"]),
            players=[Player.from_dict(player) for player in d["players"]],
            attack_players=[Player.from_dict(player) for player in d["attack_players"]],
            defense_players=[Player.from_dict(player) for player in d["defense_players"]],
        )


class State(BaseModel):
    relative_state: RelativeState
    absolute_state: AbsoluteState
    raw_state: RawState

    def to_dict(self) -> dict:
        return {
            "relative_state": self.relative_state.dict(),
            "absolute_state": self.absolute_state.dict(),
            "raw_state": self.raw_state.dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(
            relative_state=RelativeState(**d["relative_state"]),
            absolute_state=AbsoluteState(**d["absolute_state"]),
            raw_state=RawState(**d["raw_state"]),
        )
    

class Observation(BaseModel):
    ball: Ball
    players: List[Player]  # without ego_player
    ego_player: Player

    @classmethod
    def from_state(cls, state: State, ego_player: Player) -> "Observation":
        raise NotImplementedError

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @field_validator("players")
    @classmethod
    def players_must_be_list_of_players(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("players must be a list")
        for player in v:
            if not isinstance(player, Player):
                raise TypeError("players must be a list of Player")
        return v


class SimpleObservation(Observation):
    @classmethod
    def from_state(cls, state: State, ego_player: Player) -> "SimpleObservation":
        ego_position = ego_player.position
        ego_player_id = ego_player.player_id
        players = [
            Player(
                index=player.index,
                team_name=player.team_name,
                player_name=player.player_name,
                player_id=player.player_id,
                player_role=player.player_role,
                position=Position(x=player.position.x - ego_position.x, y=player.position.y - ego_position.y),
                velocity=player.velocity,
                action=player.action,
                action_probs=player.action_probs,
            )
            for player in state.raw_state.players
            if player.player_id != ego_player_id
        ]
        ball = Ball(
            position=state.raw_state.ball.position,
            velocity=state.raw_state.ball.velocity,
        )
        return cls(ball=ball, players=players, ego_player=ego_player)

    def to_tensor(self) -> torch.Tensor:
        data = []
        for player in self.players:
            data.extend([player.position.x, player.position.y, player.velocity.x, player.velocity.y])
        data.extend([self.ball.position.x, self.ball.position.y, self.ball.velocity.x, self.ball.velocity.y])
        data.extend([self.ego_player.position.x, self.ego_player.position.y])
        return torch.tensor(data)

    @property
    def dimension(self) -> int:
        return len(self.to_tensor())

    def to_dict(self):
        return {
            "ball": self.ball.to_dict(),
            "players": [player.to_dict() for player in self.players],
            "ego_player": self.ego_player.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            ball=Ball.from_dict(data["ball"]),
            players=[Player.from_dict(player) for player in data["players"]],
            ego_player=Player.from_dict(data["ego_player"]),
        )


class SimpleObservationAction(BaseModel):
    player: Player
    observation: SimpleObservation
    action: str
    reward: float

    def to_dict(self):
        return {
            "player": self.player.to_dict(),
            "observation": self.observation.to_dict(),
            "action": self.action,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            player=Player.from_dict(data["player"]),
            observation=SimpleObservation.from_dict(data["observation"]),
            action=data["action"],
            reward=data["reward"],
        )


class SimpleObservationActionSequence(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    team_name_attack: str
    team_name_defense: str
    sequence: List[SimpleObservationAction]

    def to_dict(self):
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "sequence": [obs_action.to_dict() for obs_action in self.sequence],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            game_id=data["game_id"],
            half=data["half"],
            sequence_id=data["sequence_id"],
            team_name_attack=data["team_name_attack"],
            team_name_defense=data["team_name_defense"],
            sequence=[SimpleObservationAction.from_dict(obs_action) for obs_action in data["sequence"]],
        )


class OnBall_RL(BaseModel):
    dist_ball_opponent: List[float]
    dribble_score: List[float]
    dist_goal: List[float]
    angle_goal: List[float]
    ball_speed: float
    transition: List[float]
    shot_score: float
    long_ball_score: List[float]

    def to_dict(self) -> dict:
        return {
            "dist_ball_opponent": self.dist_ball_opponent,
            "dribble_score_vel": self.dribble_score,
            "dist_goal": self.dist_goal,
            "angle_goal": self.angle_goal,
            "ball_speed": self.ball_speed,
            "transition": self.transition,
            "shot_score": self.shot_score,
            "long_ball_score": self.long_ball_score,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "OnBall_RL":
        return cls(
            dist_ball_opponent=d["dist_ball_opponent"],
            dribble_score=d["dribble_score_vel"],
            dist_goal=d["dist_goal"],
            angle_goal=d["angle_goal"],
            ball_speed=d["ball_speed"],
            transition=d["transition"],
            shot_score=d["shot_score"],
            long_ball_score=d["long_ball_score"],
        )


class OffBall_RL(BaseModel):
    fast_space: List[float]
    dist_ball: List[float]
    time_to_player: List[float]
    time_to_passline: List[float]
    variation_space: List[List[float]]
    pass_score: List[float]

    @field_validator("variation_space")
    @classmethod
    def validate_variation_space(cls, v):  # type: ignore
        if not all(isinstance(i, list) for i in v):
            raise TypeError("variation_space must be a list of lists")
        return v

    def to_dict(self) -> dict:
        return {
            "fast_space": self.fast_space,
            "dist_ball": self.dist_ball,
            "time_to_player": self.time_to_player,
            "time_to_passline": self.time_to_passline,
            "variation_space": self.variation_space,
            "pass_score": self.pass_score,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "OffBall_RL":
        return cls(
            fast_space=d["fast_space"],
            dist_ball=d["dist_ball"],
            time_to_player=d["time_to_player"],
            time_to_passline=d["time_to_passline"],
            variation_space=d["variation_space"],
            pass_score=d["pass_score"],
        )
    

class AbsoluteState_RL(BaseModel):
    dist_offside_line: List[float]
    formation: str

    def to_dict(self) -> dict:
        return {
            "dist_offside_line": self.dist_offside_line,
            "formation": self.formation,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "AbsoluteState_RL":
        return cls(
            dist_offside_line=d["dist_offside_line"],
            formation=d["formation"],
        )


class CommonState(BaseModel):
    onball_state: OnBall_RL
    offball_state: OffBall_RL
    absolute_state: AbsoluteState_RL

    @classmethod
    def from_dict(cls, data) -> "CommonState":
        return cls(
            onball_state=OnBall_RL.from_dict(data["onball_state"]),
            offball_state=OffBall_RL.from_dict(data["offball_state"]),
            absolute_state=AbsoluteState_RL.from_dict(data["absolute_state"]),
        )
    
    def to_dict(self) -> dict:
        return {
            "onball_state": self.onball_state.to_dict(),
            "offball_state": self.offball_state.to_dict(),
            "absolute_state": self.absolute_state.to_dict(),
        }


class LDSObservation(BaseModel):
    ego_player: Player
    common_state: CommonState

    @classmethod
    def from_state(cls, ego_player: Player, common_state: CommonState) -> "LDSObservation":
        raise NotImplementedError

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError


class LDSSimpleObservation(LDSObservation):
    @classmethod
    def adjust_onball_state(cls, onball_states: OnBall):
        # make input onball state the same size
        if not np.isnan(onball_states.shot_score):
            shot_score = onball_states.shot_score
            long_ball_score = [0,0,0]         
        elif not np.all(np.isnan(onball_states.long_ball_score)):
            shot_score = 0
            long_ball_score = onball_states.long_ball_score
        else:
            shot_score = 0
            long_ball_score = [0,0,0]
        
        transition = [0]*22
        ball_speed = 0
        dist_ball_opponent = [onball_states.dist_ball_opponent[0], 0]
        dist_goal = [onball_states.dist_goal[0], 0]
        angle_goal = [onball_states.angle_goal[0], 0]
        # dribble_score = onball_states.dribble_score
        dribble_score = [max(onball_states.dribble_score)]
        
        onball_state = OnBall_RL(
            dist_ball_opponent=dist_ball_opponent,
            dribble_score=dribble_score,
            dist_goal=dist_goal,
            angle_goal=angle_goal,
            ball_speed=ball_speed,
            transition=transition,
            shot_score=shot_score,
            long_ball_score=long_ball_score
        )

        return onball_state

    
    @classmethod
    def calc_pass_score(cls, offball_states: OffBall, num_offball_players: int):
        pass_score = []
        fast_space = offball_states.fast_space_vel
        dist_ball = offball_states.dist_ball
        time_to_player = offball_states.time_to_player
        time_to_passline = offball_states.time_to_passline

        k_fast_space = 0.3
        k_dist_ball = 0.5
        k_time_to_player = 0.2
        k_time_to_passline = 0.2

        # calculate pass score for each player
        pass_score = [
            k_fast_space * fs + k_dist_ball * db + k_time_to_player * tp + k_time_to_passline * tpl
            for fs, db, tp, tpl in zip(fast_space, dist_ball, time_to_player, time_to_passline)
        ]

        # get top num_offball_players index
        player_index = np.argsort(pass_score)[:num_offball_players]

        variation_space = [max(offball_states.variation_space_vel[i]) for i in player_index]
        # variation_space = [offball_states.variation_space_vel[i] for i in player_index]

        return OffBall_RL(
            fast_space=[fast_space[i] for i in player_index],
            dist_ball=[dist_ball[i] for i in player_index],
            time_to_player=[time_to_player[i] for i in player_index],
            time_to_passline=[time_to_passline[i] for i in player_index],
            variation_space=[list(v) if isinstance(v, list) else [v] for v in variation_space],
            pass_score=[pass_score[i] for i in player_index]
        )


    @classmethod
    def from_state(cls, state: State, ego_player: Player, num_offball_players: int, onball_list: list, index: int, onball_flag: int) -> "LDSSimpleObservation":
        attack_action = ["pass", "dribble", "shot", "through_pass", "cross"]
        
        onball_states = state.relative_state.onball
        offball_states = state.relative_state.offball
        absolute_states = state.absolute_state

        if ego_player.action in attack_action or (onball_list[index] == 1 and not any([player.action in attack_action for player in state.raw_state.players]) and onball_flag):
            onball_state = cls.adjust_onball_state(onball_states)
            offball_state = OffBall_RL(
                fast_space=[0]*num_offball_players,
                dist_ball=[0]*num_offball_players,
                time_to_player=[0]*num_offball_players,
                time_to_passline=[0]*num_offball_players,
                variation_space = [[0] for _ in range(num_offball_players)],
                # variation_space = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(num_offball_players)],
                pass_score=[0]*num_offball_players
            )

        else:
            onball_state = OnBall_RL(
                dist_ball_opponent = onball_states.dist_ball_opponent,
                dribble_score = [0], 
                # dribble_score = [0]*8,
                dist_goal = onball_states.dist_goal,
                angle_goal = onball_states.angle_goal,
                ball_speed = onball_states.ball_speed,
                transition = onball_states.transition,
                shot_score = 0,
                long_ball_score = [0,0,0],
            )
            offball_state = cls.calc_pass_score(offball_states, num_offball_players)

        absolute_state = AbsoluteState_RL(
            dist_offside_line=absolute_states.dist_offside_line,
            formation=absolute_states.formation
        )

        common_state = CommonState(
            onball_state=onball_state,
            offball_state=offball_state,
            absolute_state=absolute_state
        )

        onball_flag = 1 if ego_player.action in attack_action else 0

        return cls(
            ego_player=ego_player,
            common_state=common_state
        ), onball_flag


    def to_tensor(self, direction) -> torch.Tensor:
        data = []
        if self.common_state.absolute_state.formation != '':
            self.common_state.absolute_state.formation = self.common_state.absolute_state.formation.replace(" ", "_")
        else:
            self.common_state.absolute_state.formation = '0'
        
        self.common_state.absolute_state.formation = int(self.common_state.absolute_state.formation)

        ego_player = [self.ego_player.position.x, self.ego_player.position.y]
        ego_player = [0 if math.isnan(value) else value for value in ego_player]
        
        try:
            if direction == 1:
                    common_state = [
                    self.common_state.onball_state.dist_ball_opponent[0],
                    self.common_state.onball_state.dist_ball_opponent[1],
                    self.common_state.onball_state.dribble_score[0],
                    self.common_state.onball_state.dist_goal[0],
                    self.common_state.onball_state.dist_goal[1],
                    self.common_state.onball_state.angle_goal[0],
                    self.common_state.onball_state.angle_goal[1], 
                    self.common_state.onball_state.ball_speed,
                    self.common_state.onball_state.shot_score, 
                    self.common_state.onball_state.long_ball_score[0], 
                    self.common_state.onball_state.long_ball_score[1], 
                    self.common_state.onball_state.long_ball_score[2],
                    self.common_state.offball_state.fast_space[0],
                    self.common_state.offball_state.fast_space[1],
                    self.common_state.offball_state.fast_space[2],
                    self.common_state.offball_state.dist_ball[0],
                    self.common_state.offball_state.dist_ball[1],
                    self.common_state.offball_state.dist_ball[2],
                    self.common_state.offball_state.time_to_player[0],
                    self.common_state.offball_state.time_to_player[1],
                    self.common_state.offball_state.time_to_player[2],
                    self.common_state.offball_state.time_to_passline[0],
                    self.common_state.offball_state.time_to_passline[1],
                    self.common_state.offball_state.time_to_passline[2],
                    self.common_state.offball_state.variation_space[0][0],
                    self.common_state.offball_state.variation_space[1][0],
                    self.common_state.offball_state.variation_space[2][0],
                    self.common_state.offball_state.pass_score[0],
                    self.common_state.offball_state.pass_score[1],
                    self.common_state.offball_state.pass_score[2],
                    self.common_state.absolute_state.dist_offside_line[0],
                    self.common_state.absolute_state.dist_offside_line[1],
                    self.common_state.absolute_state.formation
                ]
            else:
                common_state = [
                    self.common_state.onball_state.dist_ball_opponent[0],
                    self.common_state.onball_state.dist_ball_opponent[1],
                    self.common_state.onball_state.dribble_score[0],
                    self.common_state.onball_state.dribble_score[1],
                    self.common_state.onball_state.dribble_score[2],
                    self.common_state.onball_state.dribble_score[3],
                    self.common_state.onball_state.dribble_score[4],
                    self.common_state.onball_state.dribble_score[5],
                    self.common_state.onball_state.dribble_score[6],
                    self.common_state.onball_state.dribble_score[7],
                    self.common_state.onball_state.dist_goal[0],
                    self.common_state.onball_state.dist_goal[1],
                    self.common_state.onball_state.angle_goal[0],
                    self.common_state.onball_state.angle_goal[1], 
                    self.common_state.onball_state.ball_speed,
                    self.common_state.onball_state.shot_score, 
                    self.common_state.onball_state.long_ball_score[0], 
                    self.common_state.onball_state.long_ball_score[1], 
                    self.common_state.onball_state.long_ball_score[2],
                    self.common_state.offball_state.fast_space[0],
                    self.common_state.offball_state.fast_space[1],
                    self.common_state.offball_state.fast_space[2],
                    self.common_state.offball_state.dist_ball[0],
                    self.common_state.offball_state.dist_ball[1],
                    self.common_state.offball_state.dist_ball[2],
                    self.common_state.offball_state.time_to_player[0],
                    self.common_state.offball_state.time_to_player[1],
                    self.common_state.offball_state.time_to_player[2],
                    self.common_state.offball_state.time_to_passline[0],
                    self.common_state.offball_state.time_to_passline[1],
                    self.common_state.offball_state.time_to_passline[2],
                    self.common_state.offball_state.variation_space[0][0],
                    self.common_state.offball_state.variation_space[0][1],
                    self.common_state.offball_state.variation_space[0][2],
                    self.common_state.offball_state.variation_space[0][3],
                    self.common_state.offball_state.variation_space[0][4],
                    self.common_state.offball_state.variation_space[0][5],
                    self.common_state.offball_state.variation_space[0][6],
                    self.common_state.offball_state.variation_space[0][7],
                    self.common_state.offball_state.variation_space[1][0],
                    self.common_state.offball_state.variation_space[1][1],
                    self.common_state.offball_state.variation_space[1][2],
                    self.common_state.offball_state.variation_space[1][3],
                    self.common_state.offball_state.variation_space[1][4],
                    self.common_state.offball_state.variation_space[1][5],
                    self.common_state.offball_state.variation_space[1][6],
                    self.common_state.offball_state.variation_space[1][7],
                    self.common_state.offball_state.variation_space[2][0],
                    self.common_state.offball_state.variation_space[2][1],
                    self.common_state.offball_state.variation_space[2][2],
                    self.common_state.offball_state.variation_space[2][3],
                    self.common_state.offball_state.variation_space[2][4],
                    self.common_state.offball_state.variation_space[2][5],
                    self.common_state.offball_state.variation_space[2][6],
                    self.common_state.offball_state.variation_space[2][7],
                    self.common_state.offball_state.pass_score[0],
                    self.common_state.offball_state.pass_score[1],
                    self.common_state.offball_state.pass_score[2],
                    self.common_state.absolute_state.dist_offside_line[0],
                    self.common_state.absolute_state.dist_offside_line[1],
                    self.common_state.absolute_state.formation
                ]
        except:
            print(f"dribble_score: {self.common_state.onball_state.dribble_score}")
            print(f"variation_space: {self.common_state.offball_state.variation_space}")
            AssertionError("Error in LDSObservation to_tensor")
        
        common_state = [0 if math.isnan(value) else value for value in common_state]
        common_state = [0 if math.isinf(value) else value for value in common_state]

        data.extend(ego_player)
        data.extend(common_state)

        return torch.tensor(data)

    @property
    def dimension(self) -> int:
        return len(self.to_tensor())

    def to_dict(self):

        return {
            "ego_player": self.ego_player.to_dict(),
            "common_state": self.common_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            ego_player=Player.from_dict(data["ego_player"]),
            common_state=CommonState.from_dict(data["common_state"]),
        )


class LDSSimpleObservationAction(BaseModel):
    player: Player
    observation: LDSSimpleObservation
    action: str
    reward: float

    def to_dict(self):
        return {
            "player": self.player.to_dict(),
            "observation": self.observation.to_dict(),
            "action": self.action,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            player=Player.from_dict(data["player"]),
            observation=LDSSimpleObservation.from_dict(data["observation"]),
            action=data["action"],
            reward=data["reward"],
        )


class LDSSimpleObservationActionSequence(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    team_name_attack: str
    team_name_defense: str
    sequence: List[LDSSimpleObservationAction]

    def to_dict(self):
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "sequence": [obs_action.to_dict() for obs_action in self.sequence],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            game_id=data["game_id"],
            half=data["half"],
            sequence_id=data["sequence_id"],
            team_name_attack=data["team_name_attack"],
            team_name_defense=data["team_name_defense"],
            sequence=[LDSSimpleObservationAction.from_dict(obs_action) for obs_action in data["sequence"]],
        )


class Event(BaseModel):
    state: State
    action: List[List[str]] | None = None
    reward: float

    @model_validator(mode="after")
    def set_and_validate_action(self) -> "Event":
        if self.action is None:
            self.action = [self.state.absolute_state.attack_action, self.state.absolute_state.defense_action]
        # for action in self.action:
        #     if not isinstance(action, str):
        #         raise TypeError("action must be a list of int")
        return self

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "action": self.action,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            state=State.from_dict(d["state"]),
            action=d["action"],
            reward=d["reward"],
        )


class Events(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    team_name_attack: str
    team_name_defense: str
    events: List[Event]

    @field_validator("events")
    @classmethod
    def events_must_be_list_of_events(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("events must be a list")
        for event in v:
            if not isinstance(event, Event):
                raise TypeError("events must be a list of Event")
        return v

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Events":
        return cls(
            game_id=d["game_id"],
            half=d["half"],
            sequence_id=d["sequence_id"],
            team_name_attack=d["team_name_attack"],
            team_name_defense=d["team_name_defense"],
            events=[Event.from_dict(event) for event in d["events"]],
        )


class ObservationactionInstance(BaseModel):
    observation: torch.Tensor  # (events_len, n_agents, obs_dim)
    action: torch.Tensor  # (events_len, n_agents)
    reward: torch.Tensor  # (events_len, n_agents)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ObservationactionBatch(BaseModel):
    observation: torch.Tensor  # (batch_size, max_events_len, n_agents, obs_dim)
    action: torch.Tensor  # (batch_size, max_events_len, n_agents)
    reward: torch.Tensor  # (batch_size, max_events_len, n_agents)
    mask: torch.Tensor  # (batch_size, max_events_len, n_agents)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Prediction(BaseModel):
    q_values: torch.Tensor  # (batch_size, max_events_len, n_agents, action_dim)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ObservationactionForLMInstance(BaseModel):
    sequence: List[int]
    action_mask: List[int]  # 1 if action, else 0


class ObservationActionForLMBatch(BaseModel):
    sequence: torch.Tensor  # (batch_size, max_seq_len )
    mask: torch.Tensor  # (batch_size, max_seq_len)
    action_mask: torch.Tensor  # (batch_size, max_seq_len)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ActionPrediction(BaseModel):
    q_values: torch.Tensor  # (batch_size, max_events_len, action_dim)
    model_config = ConfigDict(arbitrary_types_allowed=True)

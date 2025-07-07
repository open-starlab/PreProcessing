from typing import List
import math

from pydantic import BaseModel, field_validator, model_validator


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

    def angle_to(self, other: "Position") -> float:
        return math.atan2(other.y - self.y, other.x - self.x)


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


class State_PVS(BaseModel):
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
    def from_dict(cls, d: dict) -> "State_PVS":
        return cls(
            ball=Ball.from_dict(d["ball"]),
            players=[Player.from_dict(player) for player in d["players"]],
            attack_players=[Player.from_dict(player) for player in d["attack_players"]],
            defense_players=[Player.from_dict(player) for player in d["defense_players"]],
        )


class Event_PVS(BaseModel):
    state: State_PVS
    action: List[str] | None = None
    reward: float

    @model_validator(mode="after")
    def set_and_validate_action(self) -> "Event_PVS":
        if self.action is None:
            self.action = [player.action for player in self.state.players]
        for action in self.action:
            if not isinstance(action, str):
                raise TypeError("action must be a list of int")
        return self

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "action": self.action,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event_PVS":
        return cls(
            state=State_PVS.from_dict(d["state"]),
            action=d["action"],
            reward=d["reward"],
        )


class Events_PVS(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    sequence_start_frame: str
    sequence_end_frame: str
    team_name_attack: str
    team_name_defense: str
    events: List[Event_PVS]

    @field_validator("events")
    @classmethod
    def events_must_be_list_of_events(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("events must be a list")
        for event in v:
            if not isinstance(event, Event_PVS):
                raise TypeError("events must be a list of Event")
        return v

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "sequence_start_frame": self.sequence_start_frame,
            "sequence_end_frame": self.sequence_end_frame,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Events_PVS":
        return cls(
            game_id=d["game_id"],
            half=d["half"],
            sequence_id=d["sequence_id"],
            sequence_start_frame=d["sequence_start_frame"],
            sequence_end_frame=d["sequence_end_frame"],
            team_name_attack=d["team_name_attack"],
            team_name_defense=d["team_name_defense"],
            events=[Event_PVS.from_dict(event) for event in d["events"]],
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
    angle_ball: List[float]
    dist_goal: List[float]
    angle_goal: List[float]
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
            "angle_ball": self.angle_ball,
            "dist_goal": self.dist_goal,
            "angle_goal": self.angle_goal,
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
            angle_ball=d["angle_ball"],
            dist_goal=d["dist_goal"],
            angle_goal=d["angle_goal"],
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
    def from_dict(cls, d: dict) -> "RelativeState":
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
    def from_dict(cls, d: dict) -> "AbsoluteState":
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
    def from_dict(cls, d: dict) -> "RawState":
        return cls(
            ball=Ball.from_dict(d["ball"]),
            players=[Player.from_dict(player) for player in d["players"]],
            attack_players=[Player.from_dict(player) for player in d["attack_players"]],
            defense_players=[Player.from_dict(player) for player in d["defense_players"]],
        )


class State_EDMS(BaseModel):
    relative_state: RelativeState
    absolute_state: AbsoluteState
    raw_state: RawState

    def __repr__(self):
        return f"State(relative_state={self.relative_state}, absolute_state={self.absolute_state}, raw_state={self.raw_state})"

    def to_dict(self) -> dict:
        return {
            "relative_state": self.relative_state.to_dict(),
            "absolute_state": self.absolute_state.to_dict(),
            "raw_state": self.raw_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State_EDMS":
        return cls(
            relative_state=RelativeState(**d["relative_state"]),
            absolute_state=AbsoluteState(**d["absolute_state"]),
            raw_state=RawState(**d["raw_state"]),
        )


class Event_EDMS(BaseModel):
    state: State_EDMS
    action: List[List[str]] | None = None
    reward: float

    @model_validator(mode="after")
    def set_and_validate_action(self) -> "Event_EDMS":
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
    def from_dict(cls, d: dict) -> "Event_EDMS":
        return cls(
            state=State_EDMS.from_dict(d["state"]),
            action=d["action"],
            reward=d["reward"],
        )


class Events_EDMS(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    sequence_start_frame: str
    sequence_end_frame: str
    team_name_attack: str
    team_name_defense: str
    events: List[Event_EDMS]

    @field_validator("events")
    @classmethod
    def events_must_be_list_of_events(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("events must be a list")
        for event in v:
            if not isinstance(event, Event_EDMS):
                raise TypeError("events must be a list of Event")
        return v

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "sequence_start_frame": self.sequence_start_frame,
            "sequence_end_frame": self.sequence_end_frame,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Events_EDMS":
        return cls(
            game_id=d["game_id"],
            half=d["half"],
            sequence_id=d["sequence_id"],
            sequence_start_frame=d["sequence_start_frame"],
            sequence_end_frame=d["sequence_end_frame"],
            team_name_attack=d["team_name_attack"],
            team_name_defense=d["team_name_defense"],
            events=[Event_EDMS.from_dict(event) for event in d["events"]],
        )

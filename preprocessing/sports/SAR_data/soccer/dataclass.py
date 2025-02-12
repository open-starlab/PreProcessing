from typing import List
import numpy as np
import math

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

class State(BaseModel):
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

class Event(BaseModel):
    state: State
    action: List[str] | None = None
    reward: float

    @model_validator(mode="after")
    def set_and_validate_action(self) -> "Event":
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
    sequence_start_frame: str
    sequence_end_frame: str
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
            "sequence_start_frame": self.sequence_start_frame,
            "sequence_end_frame": self.sequence_end_frame,
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
            sequence_start_frame=d["sequence_start_frame"],
            sequence_end_frame=d["sequence_end_frame"],
            team_name_attack=d["team_name_attack"],
            team_name_defense=d["team_name_defense"],
            events=[Event.from_dict(event) for event in d["events"]],
        )

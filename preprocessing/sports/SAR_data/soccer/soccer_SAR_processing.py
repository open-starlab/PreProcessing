import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import json
import unicodedata
from tqdm import tqdm

# if __name__ == '__main__':
from preprocessing.sports.SAR_data.soccer.utils.file_utils import load_json
# else:
#     from .utils.file_utils import load_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HALF_START_COUNT = 0
HALF_END_COUNT = 0


class ProcessSkillcornerTrackingData:
    """
    Preprocess tracking data

    params:
    ----------
    df: pd.DataFrame
    tracking_df : pd.DataFrame
    players_df: pd.DataFrame
    metadata: dict
    config: dict
    save_dir: Path
    ----------
    """

    def __init__(
        self,
        df,
        players_df,
        tracking_df,
        metadata,
        config,
        save_dir,
        fps: int = 10,
    ):
        self.df = df
        self.players_df = players_df
        self.tracking_df = tracking_df
        self.metadata = metadata
        self.config = config
        self.fps = fps
        self.save_data_dir = save_dir

        self.game_name = self.df["match_id"].loc[0]

    def calc_vel_acc(self):
        """
        Calculate velocity and acceralation for each player and ball in the tracking data.

        return: self.tracking with velocity and acceralation added
        """

        previous_positions = {}
        previous_velocities = {}
        previous_ball_position = None
        previous_ball_velocity = None

        for index, frame in self.tracking_df.iterrows():
            current_positions = {}
            current_velocities = {}

            for player in frame["player_data"]:
                player_id = player["trackable_object"]
                x, y = player["x"], player["y"]

                # Calculate player velocity and acceralation
                if player_id in previous_positions:
                    prev_x, prev_y = previous_positions[player_id]
                    velocity_x = (x - prev_x) * self.fps
                    velocity_y = (y - prev_y) * self.fps
                    velocity = np.sqrt(velocity_x**2 + velocity_y**2)
                    current_velocities[player_id] = (velocity_x, velocity_y)

                    if player_id in previous_velocities:
                        prev_velocity_x, prev_velocity_y = previous_velocities[player_id]
                        acceralation_x = (velocity_x - prev_velocity_x) * self.fps
                        acceralation_y = (velocity_y - prev_velocity_y) * self.fps
                        acceralation = np.sqrt(acceralation_x**2 + acceralation_y**2)
                    else:
                        # if the previous velocity is not available, set acceralation to 0
                        acceralation = 0
                else:
                    # if the previous position is not available, set velocity and acceralation to 0
                    velocity = 0
                    acceralation = 0

                current_positions[player_id] = (x, y)
                player["velocity"] = velocity
                player["acceralation"] = acceralation

            ball = frame["ball_data"]
            ball_x, ball_y = ball["x"], ball["y"]
            # pdb.set_trace()

            # Calculate ball velocity and acceralation
            if ball_x is not None and ball_y is not None and previous_ball_position is not None:
                prev_ball_x, prev_ball_y = previous_ball_position
                if prev_ball_x is not None and prev_ball_y is not None:
                    ball_velocity_x = (ball_x - prev_ball_x) * self.fps
                    ball_velocity_y = (ball_y - prev_ball_y) * self.fps
                    ball_velocity = np.sqrt(ball_velocity_x**2 + ball_velocity_y**2)

                    if previous_ball_velocity is not None:
                        prev_ball_velocity_x, prev_ball_velocity_y = previous_ball_velocity
                        ball_acceralation_x = (ball_velocity_x - prev_ball_velocity_x) * self.fps
                        ball_acceralation_y = (ball_velocity_y - prev_ball_velocity_y) * self.fps
                        ball_acceralation = np.sqrt(ball_acceralation_x**2 + ball_acceralation_y**2)
                    else:
                        # if previous_ball_velocity is not available, set ball_acceralation to 0
                        ball_acceralation = 0
                else:
                    # if previous_ball_position is not available, set ball_velocity and ball_acceralation to 0
                    ball_velocity = 0
                    ball_acceralation = 0
            else:
                # if ball_x and ball_y are not available, set ball_velocity and ball_acceralation to 0
                ball_velocity = 0
                ball_acceralation = 0

            ball["velocity"] = ball_velocity
            ball["acceralation"] = ball_acceralation

            previous_positions = current_positions
            previous_velocities = current_velocities
            previous_ball_position = (ball_x, ball_y)
            previous_ball_velocity = (ball_velocity_x, ball_velocity_y) if ball_velocity != 0 else None

            self.tracking_df.at[index, "player_data"] = frame["player_data"]
            self.tracking_df.at[index, "ball_data"] = ball

    def extract_value(self, row, key):
        """
        Extract value from a dictionary
        """
        if row and isinstance(row, dict):
            return row.get(key)
        return None

    def expand_player_data(self, home_team_id):
        """
        Expand player data in the tracking data
        """
        # Initialize an empty list to collect all new rows
        all_new_rows = []
        tracking_id_set = set()
        tracking_id_info = {}
        data_match_players = self.players_df.to_dict("records")

        for _, row in tqdm(list(self.tracking_df.iterrows()), total=self.tracking_df.shape[0]):
            for player in row["player_data"]:
                player["is_visible"] = True

                # check if the player is already in the tracking_id_set
                trackable_object = player["trackable_object"]
                if trackable_object not in tracking_id_set:
                    tracking_id_set.add(trackable_object)
                    for match_player in data_match_players:
                        if pd.isna(match_player["trackable_object"]):
                            continue
                        elif match_player["trackable_object"] == trackable_object:
                            ha_value = 1 if match_player["team_id"] == home_team_id else 2
                            no_value = match_player["jersey_number"]
                            tracking_id_info[trackable_object] = (ha_value, no_value)
                            break
                else:
                    try:
                        ha_value, no_value = tracking_id_info[trackable_object]
                    except KeyError:
                        import pdb

                        pdb.set_trace()

                # Collect data for the new row
                all_new_rows.append(
                    {
                        "GameID": row["GameID"],
                        "Frame": row["frame"],
                        "HA": ha_value,
                        "SysTarget": None,
                        "No": no_value,
                        "track_id": player["track_id"],
                        "trackable_object": player["trackable_object"],
                        "is_visible": player["is_visible"],
                        "x": player["x"],
                        "y": player["y"],
                        "velocity": player.get("velocity", None),
                        "acceralation": player.get("acceralation", None),
                    }
                )

        # Convert all collected rows into a DataFrame at once
        rows = pd.DataFrame(all_new_rows)

        return rows

    def calculate_minutes(self, time):
        """
        Convert time string to minutes
        """
        # split time into hours, minutes
        time_parts = time.split(":")

        # Convert each part to an integer and confirm it is not None
        hours = int(time_parts[0]) if time_parts[0] else 0
        minutes = int(time_parts[1]) if time_parts[1] else 0
        seconds = int(time_parts[2]) if time_parts[2] else 0

        # convert hours and seconds to minutes and sum total time
        total_minutes = hours * 60 + minutes + seconds / 60
        return total_minutes

    def calculate_play_time(self, player):
        """
        Calculate play time for each player
        """
        if player["start_time"] is None or (isinstance(player["start_time"], float) and np.isnan(player["start_time"])):
            return 0
        elif player["end_time"] is None or (isinstance(player["end_time"], float) and np.isnan(player["end_time"])):
            return 90 - self.calculate_minutes(player["start_time"])
        else:
            return self.calculate_minutes(player["end_time"]) - self.calculate_minutes(player["start_time"])

    def ball_data(self):
        """
        Convert ball data from tracking data into df_ball.
        """

        df_ball = self.tracking_df[["GameID", "frame", "ball_data"]].copy()
        df_ball.rename(columns={"frame": "Frame"}, inplace=True)
        self.tracking_df.drop("ball_data", axis=1, inplace=True)
        df_ball[["HA", "SysTarget", "No", "X", "Y", "Speed", "Acceralation"]] = None
        df_ball[["HA", "SysTarget", "No"]] = 0
        df_ball["X"] = df_ball.ball_data.apply(lambda row: self.extract_value(row, "x"))
        df_ball["Y"] = df_ball.ball_data.apply(lambda row: self.extract_value(row, "y"))
        df_ball["Speed"] = df_ball.ball_data.apply(lambda row: self.extract_value(row, "velocity"))
        df_ball["Acceralation"] = df_ball.ball_data.apply(lambda row: self.extract_value(row, "acceralation"))
        df_ball.drop("ball_data", axis=1, inplace=True)
        return df_ball

    def convert_player_name(self, player):
        """
        Convert player name to the format used in the database
        """
        player_name = player["name"] if player["name"] is not None else ""
        if isinstance(player_name, str):
            if "  " in player_name:
                player_name = player_name.replace("  ", " ")

            # convert large letters to small letters
            player_name = player_name.lower()

            # remove accents
            player_name = unicodedata.normalize("NFKD", player_name).encode("ascii", "ignore").decode("utf-8")

        return player_name

    def player_data(self):
        """
        Convert player data from tracking data into df_player.
        """

        home_team_id = self.metadata["home_team_id"].iloc[0]
        df_player = self.expand_player_data(home_team_id)
        df_player = df_player.sort_index()
        df_player.drop(["track_id", "trackable_object", "is_visible"], axis=1, inplace=True)

        return df_player

    def players_data(self):
        """
        Convert players data from tracking data into df_players.
        """

        df_players = pd.DataFrame(index=range(len(self.players_df)))
        df_players["節"] = None
        df_players["試合ID"] = self.game_name

        home_team_id = self.metadata["home_team_id"].iloc[0]

        # Extract player data from tracking data
        for i, player in self.players_df.iterrows():
            df_players.loc[i, "ホームアウェイF"] = 1 if player["team_id"] == home_team_id else 2
            df_players.loc[i, "チームID"] = player["team_id"] if player["team_id"] is not None else 0
            df_players.loc[i, "チーム名"] = player["team"] if player["team"] is not None else ""
            if player["position"] == "Goalkeeper":
                player["position_group"] = "GK"
            df_players.loc[i, "試合ポジションID"] = (
                self.config["position_role_id"].get(player["position_group"], 0)
                if player["position_group"] is not np.nan
                else 0
            )
            df_players.loc[i, "背番号"] = player["jersey_number"] if player["jersey_number"] is not None else 0
            df_players.loc[i, "選手ID"] = player["player_id"] if player["player_id"] is not None else 0
            df_players.loc[i, "選手名"] = self.convert_player_name(player)
            df_players.loc[i, "出場"] = 1 if player["start_time"] is not None else 0
            df_players.loc[i, "スタメン"] = 1 if player["start_time"] == "00:00:00" else 0
            df_players.loc[i, "出場時間"] = self.calculate_play_time(player)
            df_players.loc[i, "実出場時間"] = df_players.loc[i, "出場時間"] * 60
            df_players.loc[i, "身長"] = player["height"] if player["height"] is not None else 0

        df_players["ホームアウェイF"] = df_players["ホームアウェイF"].fillna(0).astype(int)
        df_players["チームID"] = df_players["チームID"].fillna(0).astype(int)
        df_players["試合ポジションID"] = df_players["試合ポジションID"].fillna(0).astype(int)
        df_players["背番号"] = df_players["背番号"].fillna(0).astype(int)
        df_players["選手ID"] = df_players["選手ID"].fillna(0).astype(int)
        df_players["出場"] = df_players["出場"].fillna(0).astype(int)
        df_players["スタメン"] = df_players["スタメン"].fillna(0).astype(int)
        df_players["出場時間"] = df_players["出場時間"].fillna(0).astype(int)
        df_players["実出場時間"] = df_players["実出場時間"].fillna(0).astype(int)
        df_players["身長"] = df_players["身長"].fillna(0).astype(int)

        return df_players

    def convert_tracking_data(self, raw_data):
        """
        Original data format: not split into player and ball data
        Converted data format: split into player and ball data
        """

        print("Converting tracking data...")

        processed_data = [
            {
                "player_data": [entry for entry in frame[1].get("data", []) if entry["track_id"] != 55],
                "ball_data": next(
                    (
                        {"x": entry["x"], "y": entry["y"], "z": entry.get("z")}
                        for entry in frame[1].get("data", [])
                        if entry["track_id"] == 55
                    ),
                    {"x": None, "y": None, "z": None},
                ),
                "possession": frame[1].get("possession"),
                "frame": frame[1].get("frame"),
                "image_corners_projection": frame[1].get("image_corners_projection", []),
                "timestamp": frame[1].get("timestamp"),
                "period": frame[1].get("period"),
            }
            for frame in raw_data.iterrows()
        ]

        return pd.DataFrame(processed_data)

    def match_ball_tracking(self, df_ball):
        """
        Match ball tracking data with event data
        """
        start_ball_x = self.df["ball_x"].iloc[0]
        start_ball_y = self.df["ball_y"].iloc[0]
        start_frame = df_ball[(df_ball["X"] == start_ball_x) & (df_ball["Y"] == start_ball_y)]["Frame"]
        if len(start_frame) > 0:
            start_frame = start_frame.iloc[0]
        else:
            ArithmeticError("Start frame not found in ball tracking data")

        df_ball = df_ball[df_ball["Frame"] >= start_frame]
        df_ball = df_ball.reset_index(drop=True)

        return df_ball

    def match_player_tracking(self, df_player):
        """
        Match player tracking data with event data
        """
        frame = 0

        for i in range(0, len(df_player), 22):
            player_data = df_player.loc[i : i + 21]

            coordinates = []
            for j in range(1, 24):
                hx = self.df[f"h{j}_x"].iloc[0]
                hy = self.df[f"h{j}_y"].iloc[0]
                ax = self.df[f"a{j}_x"].iloc[0]
                ay = self.df[f"a{j}_y"].iloc[0]
                if ax is not None and ay is not None and hx is not None and hy is not None:
                    if not np.isnan(hx) and not np.isnan(hy):
                        coordinates.append((hx, hy))
                    if not np.isnan(ax) and not np.isnan(ay):
                        coordinates.append((ax, ay))

            for coord in coordinates:
                x, y = coord
                if (x == player_data["x"]).any():
                    y_ = player_data[player_data["x"] == x]["y"]
                    if y == y_.iloc[0]:
                        frame = player_data["Frame"].iloc[0]
                break

            if frame != 0:
                break

        df_player = df_player[df_player["Frame"] >= frame]

        return df_player

    def skillcorner_to_datastadium(self):
        """
        convert skillcorner into datastadium format
        """

        logger.info(f"preprocess tracking data started... {self.game_name}")
        start_time = time.time()

        # preprocess tracking data
        self.tracking_df = self.convert_tracking_data(self.tracking_df)
        self.calc_vel_acc()
        self.tracking_df["GameID"] = self.game_name

        # ball.csv
        df_ball = self.ball_data()

        # player.csv
        df_player = self.player_data()

        # players.csv
        df_players = self.players_data()

        # match tracking data and event data
        df_ball = self.match_ball_tracking(df_ball)
        df_player = self.match_player_tracking(df_player)

        # save
        output_dir = Path(self.save_data_dir) / str(self.game_name)
        output_dir.mkdir(exist_ok=True, parents=True)
        df_ball.to_csv(output_dir / "ball.csv", index=False)
        df_player.to_csv(output_dir / "player.csv", index=False)
        df_players.to_csv(output_dir / "players.csv", index=False)
        logger.info(
            f"""
            preprocess tracking data {self.game_name} finished in {time.time() - start_time:.2f} sec
            """
        )


class ProcessStatsbombEventData:
    """
    Preprocess event data

    params:
    ----------
    event_game_dirs: Path
        event game directory
    metadata: dict
        match metadata
    df_event: pd.DataFrame
        event data
    df_lineup: pd.DataFrame
        lineup data
    kickoff_frame: list
        kickoff frame
    tracking_match: pd.DataFrame
        tracking match data
    args: argparse.ArgumentParser
    config: dict
    ----------
    """

    def __init__(self, df, players_df, metadata, config, save_dir, fps: int = 10):
        self.df = df
        self.players_df = players_df
        self.metadata = metadata
        self.game_name = self.df["match_id"].loc[0]
        self.config = config
        self.save_data_dir = save_dir

        self.fps = fps

    def number_half_events(self, event_type):
        """
        get number of half events
        """
        global HALF_START_COUNT, HALF_END_COUNT
        if event_type == "Starting XI":
            HALF_START_COUNT = 0
            HALF_END_COUNT = 0
            return event_type
        elif event_type == "Half Start":
            HALF_START_COUNT += 1
            half_number = (HALF_START_COUNT + 1) // 2
            return f"Half Start {half_number}"
        elif event_type == "Half End":
            HALF_END_COUNT += 1
            half_number = (HALF_END_COUNT + 1) // 2
            return f"Half End {half_number}"
        else:
            return event_type

    def extract_location(self, loc, index, type):
        """
        Extract location data
        """
        if type == "p":
            if isinstance(loc, list) and len(loc) > 0:
                # convert player location data to datastadium format
                return loc[index] * 105 / 120 if index == 0 else loc[index] * 68 / 80
            return np.nan
        elif type == "b":
            if isinstance(loc, list) and len(loc) > 0:
                # convert ball location data to datastadium format
                return (loc[index] - 60) * 315 / 120 if index == 0 else (loc[index] - 40) * 204 / 80
            return np.nan

    def match_player_id(self, row):
        """
        Get the player id from the tracking data
        """
        if row["player"] is None:
            return None
        else:
            for _, player in self.players_df.iterrows():
                player_name = player["name"]
                if player_name == row["player"]:
                    return player["player_id"]
            return None

    def match_jursey_number(self, row):
        """
        Extract the jersey number of the player from the lineup data
        """
        if row["player"] is None:
            return None
        else:
            for _, player in self.players_df.iterrows():
                player_name = player["name"]
                if player_name == row["player"]:
                    return player["jersey_number"]
            return None

    def position_id(self, row):
        """
        Get the position id of the player from the lineup data
        """
        if isinstance(row["player"], str):
            team_player = self.players_df[self.players_df["team"] == row["team"]]
            position = team_player.loc[team_player["jersey_number"] == row["jersey_number"], "position"].iloc[0]
        else:
            return 0

        if position in self.config["GK_position"]:
            return 1
        elif position in self.config["DF_position"]:
            return 2
        elif position in self.config["MF_position"]:
            return 3
        elif position in self.config["FW_position"]:
            return 4
        else:
            AssertionError(f"Position {position} is not found")

    def series_number(self, row, series_count):
        """
        Get the series number of the player
        """

        if row["event_type"] in self.config["action_type"]:
            series_count += 1
        elif row["event_type"] == "Pass" and row["event_type_2"] in self.config["action_type"]:
            series_count += 1
        return series_count

    def calculate_attack_periods(self, df_p):
        """
        Calculate attack periods
        """
        df_p["攻撃変化"] = (df_p["攻撃履歴No"] != df_p["攻撃履歴No"].shift()) | (
            df_p["攻撃履歴No"].isnull() != df_p["攻撃履歴No"].shift().isnull()
        )

        # set attack start and end history number
        df_p["攻撃開始履歴No"] = df_p.loc[df_p["攻撃変化"], "履歴No"]
        df_p["攻撃開始履歴No"] = df_p["攻撃開始履歴No"].ffill()

        # set attack end history number
        df_p["攻撃終了履歴No"] = df_p.loc[df_p["攻撃変化"].shift(-1, fill_value=True), "履歴No"]
        df_p["攻撃終了履歴No"] = df_p["攻撃終了履歴No"].bfill()
        df_p = df_p.drop("攻撃変化", axis=1)

        # fill NaN values with 0
        df_p["攻撃開始履歴No"] = df_p["攻撃開始履歴No"].fillna(0).astype(int)
        df_p["攻撃終了履歴No"] = df_p["攻撃終了履歴No"].fillna(0).astype(int)

        return df_p

    def outcome_flag(self, row):
        """
        Get the outcome flag of the player
        """
        try:
            if row["アクション名"] == "Shot":
                return self.df.loc[row["履歴No"] - 1, "outcome"]
            elif row["アクション名"] == "Pass":
                return self.df.loc[row["履歴No"] - 1, "outcome"]
            elif row["アクション名"] == "Dribble":
                return self.df.loc[row["履歴No"] - 1, "outcome"]
        except KeyError:
            None
        return None

    def success_flag(self, row):
        """
        Get the success flag of the action
        """
        if row["outcome"] == "Goal" or row["outcome"] == "Complete":
            return 1
        elif row["アクション名"] == "Pass" and pd.isnull(row["outcome"]):
            return 1
        return 0

    def action_flag(self, row):
        """
        Get the action flag of the player action
        """

        if (
            row["アクション名"] == "Pass"
            or row["アクション名"] == "Kick Off"
            or row["アクション名"] == "Free Kick"
            or row["アクション名"] == "Throw-in"
            or row["アクション名"] == "Corner Kick"
        ):
            row["F_パス"] = 1
        if row["アクション名"] == "Shot":
            row["F_シュート"] = 1
        if row["アクション名"] == "Dribble" or row["アクション名"] == "Carry":
            row["F_ドリブル"] = 1
        if row["outcome"] == "Goal":
            row["F_ゴール"] = 1
        if row["アクション名"] == "Pressure":
            row["F_プレッシャー"] = 1
        if row["アクション名"] == "Ball Recovery":
            row["F_ボールリカバリ"] = 1
        if row["アクション名"] == "Block":
            row["F_ブロック"] = 1
        if row["アクション名"] == "Interception":
            row["F_インターセプト"] = 1
        if row["アクション名"] == "Clearance":
            row["F_クリア"] = 1
        if row["アクション名"] == "Cross":
            row["F_クロス"] = 1
        if row["アクション名"] == "Through Pass":
            row["F_スルーパス"] = 1
        return row

    def statsbomb_to_datastadium(self):
        """
        convert statsbomb into datastadium format
        """
        logger.info(f"preprocess event data started... {self.game_name}")
        start_time = time.time()
        series_count = 0

        self.df["event_type"] = self.df["event_type"].apply(self.number_half_events)

        home_team_name = self.metadata["home_team"].loc[0]

        df_play = pd.DataFrame(index=range(len(self.df)))

        # convert statsbomb event data into datastadium format
        df_play["試合ID"] = self.df["match_id"]
        df_play["フレーム番号"] = np.where(
            self.df["period"] == 1, self.df["seconds"] * self.fps, (self.df["seconds"] * self.fps) + 6000
        )
        df_play["絶対時間秒数"] = self.df["minute"] * 60 + self.df["second"]
        df_play["試合状態ID"] = self.df["period"]
        df_play["ホームアウェイF"] = self.df["home_team"].apply(lambda x: 2 if x == 0 else 1)
        df_play["アクションID"] = self.df.apply(
            lambda row: self.config["event_id"][row["event_type"]]
            if pd.isna(row["event_type_2"])
            else self.config["event_id"][row["event_type_2"]],
            axis=1,
        )
        df_play["アクション名"] = self.df.apply(
            lambda row: row["event_type"] if pd.isna(row["event_type_2"]) else row["event_type_2"], axis=1
        )
        df_play["位置座標X"] = self.df["start_x"] * 105 / 120
        df_play["位置座標Y"] = self.df["start_y"] * 68 / 80
        df_play["チームID"] = self.df.apply(
            lambda row: self.metadata["home_team_id"].iloc[0]
            if row["team"] == self.metadata["home_team"].iloc[0]
            else self.metadata["away_team_id"].iloc[0],
            axis=1,
        )
        df_play["チーム名"] = self.df["team"]
        df_play["選手ID"] = self.df.apply(lambda row: self.match_player_id(row), axis=1)
        df_play["選手名"] = np.replace(self.df["player"], "  ", " ") if "  " in self.df["player"] else self.df["player"]
        df_play["選手背番号"] = self.df.apply(lambda row: self.match_jursey_number(row), axis=1)
        df_play["ポジションID"] = self.df.apply(lambda row: self.position_id(row), axis=1)
        df_play["ボールＸ"] = (self.df["ball_x"] - 60) * 315 / 120
        df_play["ボールＹ"] = (self.df["ball_y"] - 40) * 204 / 80
        df_play["攻撃履歴No"] = self.df["possession"]
        df_play["攻撃方向"] = self.df.apply(
            lambda row: 1
            if (row["team"] == home_team_name and row["period"] == 1) or (row["team"] != home_team_name and row["period"] == 2)
            else 2,
            axis=1,
        )
        df_play["履歴No"] = self.df["index"]
        df_play["シリーズNo"] = self.df.apply(lambda row: self.series_number(row, series_count), axis=1).cumsum()

        df_play = self.calculate_attack_periods(df_play)

        df_play["F_ボールタッチ"] = df_play["アクションID"].isin(self.config["ball_touch"]).astype(int)
        df_play["outcome"] = df_play.apply(lambda row: self.outcome_flag(row), axis=1)
        df_play["F_成功"] = df_play.apply(lambda row: self.success_flag(row), axis=1)
        df_play[
            [
                "F_パス",
                "F_シュート",
                "F_ドリブル",
                "F_ゴール",
                "F_プレッシャー",
                "F_ボールリカバリー",
                "F_ブロック",
                "F_インターセプト",
                "F_クリア",
                "F_クロス",
                "F_スルーパス",
            ]
        ] = 0
        df_play["フォーメーション"] = self.df["formation"]
        df_play = df_play.apply(lambda row: self.action_flag(row), axis=1)
        df_play = df_play.drop(["outcome"], axis=1)

        # Sort the columns
        df_play = df_play[
            [
                "試合ID",
                "フレーム番号",
                "絶対時間秒数",
                "試合状態ID",
                "ホームアウェイF",
                "位置座標X",
                "位置座標Y",
                "チームID",
                "チーム名",
                "選手ID",
                "選手名",
                "選手背番号",
                "ポジションID",
                "アクションID",
                "アクション名",
                "ボールＸ",
                "ボールＹ",
                "攻撃履歴No",
                "攻撃方向",
                "シリーズNo",
                "F_ボールタッチ",
                "F_成功",
                "履歴No",
                "攻撃開始履歴No",
                "攻撃終了履歴No",
                "F_ゴール",
                "F_シュート",
                "F_パス",
                "F_ドリブル",
                "F_プレッシャー",
                "F_ボールリカバリー",
                "F_ブロック",
                "F_インターセプト",
                "F_クリア",
                "F_クロス",
                "F_スルーパス",
                "フォーメーション",
            ]
        ]

        # save
        output_dir = Path(self.save_data_dir) / str(self.game_name)
        output_dir.mkdir(exist_ok=True, parents=True)
        df_play.to_csv(output_dir / "play.csv", index=False)

        logger.info(
            f"""
            preprocess event data {self.game_name} finished in {time.time() - start_time:.2f} sec
            """
        )


class ProcessFIFAWCTrackingData:
    def __init__(self, tracking_list, match_id, save_dir):
        self.tracking_list = tracking_list
        self.match_id = match_id
        self.save_dir = save_dir
        self.frame_rate = 29.97
        self.frame_time_delta = 1 / self.frame_rate

    def interpolate_duplicate_coordinates(self, df):
        """
        Interpolate duplicate coordinates in the DataFrame.
        """
        initial_na_count = df[["X", "Y"]].isna().sum().sum()
        df[["X", "Y"]] = df[["X", "Y"]].interpolate(method="linear")

        df[["X", "Y"]] = df[["X", "Y"]].ffill().bfill()

        final_na_count = df[["X", "Y"]].isna().sum().sum()
        interpolated_count = initial_na_count - final_na_count

        coord_diff = df[["X", "Y"]].diff()
        duplicate_mask = (coord_diff == 0).all(axis=1)
        duplicate_count = duplicate_mask.sum()

        if duplicate_count > 0:
            np.random.seed(42)
            noise = np.random.normal(0, 0.05, (duplicate_count, 2))

            df.loc[duplicate_mask, ["X", "Y"]] += noise
            interpolated_count += duplicate_count

        return df

    def calculate_ball_speed_acceleration_(self, ball_data, match_id):
        """
        Calculate ball speed and acceleration from ball tracking data.
        """
        if len(ball_data) < 2:
            return ball_data

        df = pd.DataFrame(ball_data)
        df["GameID"] = df["GameID"].ffill()
        df["GameID"] = df["GameID"].fillna(match_id)

        df = df.sort_values(["GameID", "Frame"]).reset_index(drop=True)

        initial_len = len(df)
        df = df.drop_duplicates(subset=["GameID", "Frame"]).reset_index(drop=True)
        if len(df) < initial_len:
            print(f"    Removed {initial_len - len(df)} duplicate ball records")

        df = self.interpolate_duplicate_coordinates(df)
        df[["X", "Y"]] = df[["X", "Y"]].round(3)
        coord_diff = df[["X", "Y"]].diff()
        speed = np.sqrt((coord_diff**2).sum(axis=1)) / self.frame_time_delta
        speed = speed.fillna(0)
        acceleration = speed.diff() / self.frame_time_delta
        acceleration = acceleration.fillna(0)

        speed_values = speed.values[speed > 0]
        acceleration_values = acceleration.values[acceleration != 0]

        speed_99 = None
        acc_99 = None

        if len(speed_values) > 0:
            speed_99 = np.percentile(speed_values, 99)
            speed = np.clip(speed, 0, speed_99)
            print(f"  Applied ball speed cap: {speed_99:.1f} m/s")

        if len(acceleration_values) > 0:
            acc_99 = np.percentile(np.abs(acceleration_values), 99)
            acceleration = np.clip(acceleration, -acc_99, acc_99)
            print(f"  Applied ball acceleration cap: ±{acc_99:.1f} m/s²")

        if len(df) >= 3:
            speed = speed.rolling(window=3, center=True, min_periods=1).mean()
            acceleration = acceleration.rolling(window=3, center=True, min_periods=1).mean()

        df["Speed"] = speed.round(3)
        df["Acceleration"] = acceleration.round(3)

        return df.to_dict("records")

    def calculate_velocity_acceleration_(self, position_data, match_id):
        """
        Calculate velocity and acceleration from position data.
        """
        if len(position_data) < 2:
            return position_data

        df = pd.DataFrame(position_data)
        df["GameID"] = df["GameID"].ffill()
        df["GameID"] = df["GameID"].fillna(match_id)

        df = df.sort_values(["GameID", "Frame", "HA", "No"]).reset_index(drop=True)

        initial_len = len(df)
        df = df.drop_duplicates(subset=["GameID", "Frame", "HA", "No"]).reset_index(drop=True)
        if len(df) < initial_len:
            print(f"    Removed {initial_len - len(df)} duplicate records before velocity calculation")

        df["velocity"] = 0.0
        df["acceleration"] = 0.0

        all_velocities = []
        all_accelerations = []

        for (ha, player_no), group in df.groupby(["HA", "No"]):
            if pd.isna(player_no) or len(group) < 2:
                continue

            group = group.sort_values("Frame")
            indices = group.index

            x_diff = group["x"].diff()
            y_diff = group["y"].diff()

            velocity = np.sqrt(x_diff**2 + y_diff**2) / self.frame_time_delta
            velocity = velocity.fillna(0)

            acceleration = velocity.diff() / self.frame_time_delta
            acceleration = acceleration.fillna(0)

            valid_velocities = velocity[velocity > 0].values
            valid_accelerations = acceleration[acceleration != 0].values

            if len(valid_velocities) > 0:
                all_velocities.extend(valid_velocities)
            if len(valid_accelerations) > 0:
                all_accelerations.extend(valid_accelerations)

            df.loc[indices, "velocity"] = velocity.round(3)
            df.loc[indices, "acceleration"] = acceleration.round(3)

        velocity_99 = None
        acc_99 = None

        if all_velocities:
            velocity_99 = np.percentile(all_velocities, 99)
            df["velocity"] = np.clip(df["velocity"], 0, velocity_99)

        if all_accelerations:
            acc_99 = np.percentile(np.abs(all_accelerations), 99)
            df["acceleration"] = np.clip(df["acceleration"], -acc_99, acc_99)

        print(
            f"    Applied velocity cap: {velocity_99:.1f} m/s, acceleration cap: ±{acc_99:.1f} m/s²"
            if all_velocities
            else "    No velocity data processed"
        )

        return df.to_dict("records")

    def parse_tracking_records(self):
        """Extract player and ball tracking data from the loaded tracking_list."""
        player_data = []
        ball_data = []
        first_frame_num = None

        for record in self.tracking_list:
            game_id = record.get("gameRefId", self.match_id)
            original_frame_num = record.get("frameNum")

            if first_frame_num is None:
                first_frame_num = original_frame_num
            frame_num = original_frame_num - first_frame_num

            # Player data
            for team_key, ha in [("homePlayers", 1), ("awayPlayers", 2)]:
                if team_key in record:
                    for player in record[team_key]:
                        player_record = {
                            "GameID": game_id,
                            "Frame": frame_num,
                            "HA": ha,
                            "SysTarget": 0,
                            "No": player.get("jerseyNum"),
                            "x": round(player.get("x", 0), 3) if player.get("x") is not None else None,
                            "y": round(player.get("y", 0), 3) if player.get("y") is not None else None,
                            "velocity": 0,
                            "acceleration": 0,
                        }
                        player_data.append(player_record)

            # Ball data
            if "balls" in record and record["balls"]:
                ball = record["balls"][0]
                ball_record = {
                    "GameID": game_id,
                    "Frame": frame_num,
                    "HA": 0,
                    "SysTarget": 0,
                    "No": 0,
                    "X": round(ball.get("x", 0), 3) if ball.get("x") is not None else None,
                    "Y": round(ball.get("y", 0), 3) if ball.get("y") is not None else None,
                    "Speed": 0,
                    "Acceralation": 0,
                }
                ball_data.append(ball_record)
        return player_data, ball_data

    def process_tracking_data(self):
        player_data, ball_data = self.parse_tracking_records()

        if player_data:
            player_data = self.calculate_velocity_acceleration_(player_data, self.match_id)
            pd.DataFrame(player_data).to_csv(self.save_dir / "player.csv", index=False)
            print(f"  Saved player.csv with {len(player_data)} records (optimized processing)")
        if ball_data:
            ball_data = self.calculate_ball_speed_acceleration_(ball_data, self.match_id)
            pd.DataFrame(ball_data).to_csv(self.save_dir / "ball.csv", index=False)
            print(f"  Saved ball.csv with {len(ball_data)} records (optimized processing)")


class ProcessFIFAWCEventData:
    def __init__(self, event_df, match_id, save_dir, config):
        self.event_df = event_df
        self.match_id = match_id
        self.save_dir = save_dir
        self.frame_rate = 29.97
        self.config = config

    def get_detailed_action_name_(self, game_event_type, possession_event_type, setpiece_type, period, event_flags):
        """Determine the detailed action name"""
        candidates = []

        if game_event_type == "FIRSTKICKOFF" and period == 1:
            return "前半開始"
        elif game_event_type == "SECONDKICKOFF" and period == 2:
            return "後半開始"
        elif game_event_type in ["FIRSTKICKOFF", "SECONDKICKOFF"]:
            return "キックオフ"

        if game_event_type == "END":
            if period == 1:
                return "前半終了"
            elif period == 2:
                return "後半終了"
            else:
                return "終了"

        if game_event_type == "SUB":
            return "交代"

        if game_event_type and game_event_type in self.config["raw_event_mapping"]:
            candidates.append(self.config["raw_event_mapping"][game_event_type])

        if possession_event_type and possession_event_type in self.config["raw_event_mapping"]:
            candidates.append(self.config["raw_event_mapping"][possession_event_type])

        if setpiece_type and setpiece_type in self.config["raw_event_mapping"]:
            candidates.append(self.config["raw_event_mapping"][setpiece_type])

        if event_flags.get("F_スルーパス"):
            candidates.append("スルーパス")
        if event_flags.get("F_クロス"):
            candidates.append("クロス")
        if event_flags.get("F_ブロック"):
            candidates.append("ブロック")
        if event_flags.get("F_インターセプト"):
            candidates.append("インターセプト")
        if event_flags.get("F_クリア"):
            candidates.append("クリア")

        for priority_event in self.config["raw_event_priority"]:
            if priority_event in candidates:
                return priority_event

        return candidates[0] if candidates else "タッチ"

    def add_halftime_substitutions_(self, df, all_events):
        """
        Detect halftime substitutions and add them to the event data
        """
        print("  Checking for halftime substitutions...")

        second_half_start = df[df["アクション名"] == "後半開始"]
        if second_half_start.empty:
            print("    No second half start event found")
            return df

        halftime_subs = []

        if halftime_subs:
            new_subs_df = pd.DataFrame(halftime_subs)
            df = pd.concat([df, new_subs_df], ignore_index=True)
            df = df.sort_values(["フレーム番号", "アクション名"]).reset_index(drop=True)
        else:
            print("No halftime substitutions detected or all substitutions already recorded")

        return df

    def determine_attack_sequences_(self, events_df):
        """
        Determine attack sequences, series numbers, and history numbers.
        """
        events_df = events_df.sort_values(["絶対時間"]).reset_index(drop=True)
        events_df["履歴No"] = range(1, len(events_df) + 1)

        series_no = 1
        attack_no = 1
        current_attack_start = 1

        series_numbers = []
        attack_numbers = []
        attack_starts = []
        attack_ends = []

        prev_action = None
        prev_team = None
        consecutive_opponent_actions = 0

        for i, row in events_df.iterrows():
            action_name = row["アクション名"]
            team = row["ホームアウェイF"]

            if i == 0 and action_name == "前半開始":
                pass
            else:
                period_change_actions = ["前半開始", "後半開始", "前半終了", "後半終了"]
                if (
                    action_name in ["ボールアウト", "ファウルする", "ファウル受ける", "オフサイド", "キックオフ"]
                    or action_name in period_change_actions
                ):
                    if i > 0:
                        series_no += 1
                        consecutive_opponent_actions = 0

                team_changed = prev_team is not None and team != prev_team
                if team_changed:
                    consecutive_opponent_actions += 1
                else:
                    consecutive_opponent_actions = 0

                attack_should_end = False

                if (
                    action_name in ["ボールアウト", "キックオフ", "CK", "スローイン", "PK", "直接FK", "間接FK"]
                    or action_name in period_change_actions
                ):
                    attack_should_end = True

                elif team_changed and i > 0:
                    attack_should_end = True

                elif consecutive_opponent_actions >= 2:
                    attack_should_end = True

                elif prev_action and prev_action in ["インターセプト", "タックル", "ボールゲイン"]:
                    attack_should_end = True

                if attack_should_end:
                    if i > 0:
                        for j in range(current_attack_start - 1, i):
                            if j < len(attack_ends):
                                attack_ends[j] = i
                    attack_no += 1
                    current_attack_start = i + 1
                    consecutive_opponent_actions = 0

            series_numbers.append(series_no)
            attack_numbers.append(attack_no)
            attack_starts.append(current_attack_start)
            attack_ends.append(len(events_df))

            prev_action = action_name
            prev_team = team

        if attack_ends:
            for j in range(current_attack_start - 1, len(attack_ends)):
                attack_ends[j] = len(events_df)

        events_df["シリーズNo"] = series_numbers
        events_df["攻撃履歴No"] = attack_numbers
        events_df["攻撃開始履歴No"] = attack_starts
        events_df["攻撃終了履歴No"] = attack_ends

        return events_df

    def set_event_flags_(self, event_record, possession_events):
        """Set event flags based on possession events."""
        pe = possession_events

        flags = {
            "F_ゴール": 0,
            "F_シュート": 0,
            "F_パス": 0,
            "F_ドリブル": 0,
            "F_ボールゲイン": 0,
            "F_ブロック": 0,
            "F_インターセプト": 0,
            "F_クリア": 0,
            "F_クロス": 0,
            "F_スルーパス": 0,
            "F_成功": 0,
        }

        possession_type = pe.get("possessionEventType", "")
        pass_type = pe.get("passType", "")
        pass_outcome = pe.get("passOutcomeType", "")
        cross_outcome = pe.get("crossOutcomeType", "")
        shot_outcome = pe.get("shotOutcomeType", "")

        if possession_type == "PA":
            flags["F_パス"] = 1
        elif possession_type == "SH":
            flags["F_シュート"] = 1
        elif possession_type == "BC":
            flags["F_ドリブル"] = 1
        elif possession_type == "RE":
            flags["F_ボールゲイン"] = 1
        elif possession_type == "CL":
            flags["F_クリア"] = 1
        elif possession_type == "CR":
            flags["F_クロス"] = 1

        if pass_type == "T":
            flags["F_スルーパス"] = 1

        if pass_outcome == "B" or cross_outcome == "B" or shot_outcome == "B":
            flags["F_ブロック"] = 1

        if pass_outcome == "D" or cross_outcome == "D":
            flags["F_インターセプト"] = 1

        if shot_outcome == "G":
            flags["F_ゴール"] = 1

        if (
            pass_outcome in ["C", "G"]
            or cross_outcome in ["C", "G"]
            or shot_outcome in ["C", "G"]
            or possession_type in ["RE", "BC"]
        ):
            flags["F_成功"] = 1

        event_record.update(flags)
        return event_record, flags

    def calculate_frame_from_firstkickoff(self, events, absolute_time, game_event_id):
        """
        Define the frame number based on the FIRSTKICKOFF event.
        """
        firstkickoff_time = None
        for event in events:
            ge = event.get("gameEvents", {})
            if ge.get("gameEventType") == "FIRSTKICKOFF":
                firstkickoff_time = event.get("eventTime")
                break

        if firstkickoff_time is not None and absolute_time is not None:
            time_diff = absolute_time - firstkickoff_time
            frame_diff = int(time_diff * self.frame_rate)
            return max(0, frame_diff)
        else:
            return 0

    def process_event_data(self):
        """
        Process FIFA World Cup event data.
        """
        events = []
        all_events = self.event_df
        for event in self.event_df:
            game_id = event.get("gameId")
            absolute_time = event.get("eventTime")

            # Calculate firstkickoff frame
            frame_id = self.calculate_frame_from_firstkickoff(all_events, absolute_time, event.get("gameEventId"))
            ge = event.get("gameEvents", {})
            pe = event.get("possessionEvents", {})

            is_home_team = ge.get("homeTeam", True)
            period = ge.get("period", 1)

            temp_record = {
                "試合ID": game_id,
                "フレーム番号": frame_id,
                "絶対時間": absolute_time,
                "試合状態ID": period,
                "ホームアウェイF": 1 if is_home_team else 2,
                "位置座標X": None,
                "位置座標Y": None,
                "チームID": ge.get("teamId"),
                "チーム名": ge.get("teamName"),
                "選手ID": ge.get("playerId"),
                "選手名": ge.get("playerName"),
                "選手背番号": None,
                "ポジションID": None,
                "アクションID": frame_id,
                "ボールX": None,
                "ボールY": None,
                "攻撃履歴No": None,
                "攻撃方向": 1 if (is_home_team and period == 1) or (not is_home_team and period == 2) else 2,
                "シリーズNo": None,
                "F_ボールタッチ": 1 if (ge.get("touches") or 0) > 0 else 0,
                "F_成功": None,
                "履歴No": None,
                "攻撃開始履歴No": None,
                "攻撃終了履歴No": None,
                "フォーメーション": None,
            }

            event_record, event_flags = self.set_event_flags_(temp_record, pe)

            if ge.get("gameEventType") == "SUB":
                temp_record["選手ID"] = ge.get("playerOffId")
                temp_record["選手名"] = ge.get("playerOffName")
                player_off_id = ge.get("playerOffId")
                if player_off_id:
                    for players_key in ["homePlayers", "awayPlayers"]:
                        if players_key in event and event[players_key]:
                            for player in event[players_key]:
                                if str(player.get("playerId")) == str(player_off_id):
                                    temp_record["選手背番号"] = player.get("jerseyNum")
                                    temp_record["ポジションID"] = player.get("positionGroupType")
                                    break
                            if temp_record["選手背番号"] is not None:
                                break

            action_name = self.get_detailed_action_name_(
                ge.get("gameEventType"), pe.get("possessionEventType"), ge.get("setpieceType"), period, event_flags
            )
            event_record["アクション名"] = action_name

            player_id = event_record["選手ID"]
            if player_id:
                for players_key in ["homePlayers", "awayPlayers"]:
                    if players_key in event and event[players_key]:
                        for player in event[players_key]:
                            if str(player.get("playerId")) == str(player_id):
                                event_record["位置座標X"] = player.get("x")
                                event_record["位置座標Y"] = player.get("y")
                                event_record["選手背番号"] = player.get("jerseyNum")
                                event_record["ポジションID"] = player.get("positionGroupType")
                                break
                        if event_record["位置座標X"] is not None:
                            break

            if "ball" in event and event["ball"] and len(event["ball"]) > 0:
                ball = event["ball"][0]
                event_record["ボールX"] = ball.get("x")
                event_record["ボールY"] = ball.get("y")

            events.append(event_record)

        df = pd.DataFrame(events)

        if "交代" in df["アクション名"].values:
            valid_data = df.dropna(subset=["選手ID", "チーム名", "チームID", "ホームアウェイF"])
            player_team_map = dict(zip(valid_data["選手ID"].astype(str), valid_data["チーム名"]))
            player_team_id_map = dict(zip(valid_data["選手ID"].astype(str), valid_data["チームID"]))
            player_home_away_map = dict(zip(valid_data["選手ID"].astype(str), valid_data["ホームアウェイF"]))

            sub_mask = df["アクション名"] == "交代"
            sub_indices = df[sub_mask].index

            for idx in sub_indices:
                player_id_str = str(df.at[idx, "選手ID"])
                df.at[idx, "チーム名"] = player_team_map.get(player_id_str)
                df.at[idx, "チームID"] = player_team_id_map.get(player_id_str)
                df.at[idx, "ホームアウェイF"] = player_home_away_map.get(player_id_str)

        initial_count = len(df)

        protected_events = ["前半開始", "後半開始", "前半終了", "後半終了", "交代"]

        missing_mask = (
            (df["選手ID"].isna())
            | (df["位置座標X"].isna())
            | (df["位置座標Y"].isna())
            | (df["チーム名"].isna())
            | (df["選手名"].isna())
        )

        protected_mask = df["アクション名"].isin(protected_events)

        delete_mask = missing_mask & ~protected_mask

        if delete_mask.sum() > 0:
            print(f"  Removing {delete_mask.sum()} events with missing player/position data")
            print(f"  Protected {protected_mask.sum()} critical events from deletion")
            df = df[~delete_mask].reset_index(drop=True)

        df = self.add_halftime_substitutions_(df, all_events)

        df = self.determine_attack_sequences_(df)

        df.to_csv(self.save_dir / "play.csv", index=False)
        print(f"  Saved play.csv with {len(df)} events (removed {initial_count - len(df)} incomplete events)")


class ProcessFIFAWCMetaDataANDRosters:
    def __init__(self, rosters_df, metadata_df, players_df, match_id, save_dir, config):
        self.rosters_df = rosters_df
        self.metadata_df = metadata_df
        self.players_df = players_df
        self.match_id = match_id
        self.save_dir = save_dir
        self.config = config

    def get_player_height_from_original_data(self):
        """Get player height data from raw_data"""
        height_data = {}

        if "height" in self.players_df.columns and "player.id" in self.players_df.columns:
            for _, row in self.players_df.iterrows():
                player_id = row.get("player.id")
                height = row.get("height")
                if pd.notna(player_id) and pd.notna(height):
                    height_data[str(player_id)] = height
        return height_data

    def process_metadata_roster(self):
        height_data = self.get_player_height_from_original_data()
        players = []
        for player_info in self.rosters_df:
            team_name = player_info["team"]["name"]
            home_team = self.metadata_df.get("homeTeam", {}).get("name")
            home_away = 1 if team_name == home_team else 2

            player_id = str(player_info["player"]["id"])

            player_role = self.config["raw_player_role_mapping"].get(player_info["positionGroupType"], None)

            player_record = {
                "節": None,
                "試合ID": self.metadata_df.get("id"),
                "ホームアウェイF": home_away,
                "チームID": player_info["team"]["id"],
                "チーム名": player_info["team"]["name"],
                "試合ポジションID": player_role,
                "背番号": player_info["shirtNumber"],
                "選手ID": player_info["player"]["id"],
                "選手名": player_info["player"]["nickname"],
                "出場": 1,
                "スタメン": 1 if player_info["started"] else 0,
                "出場時間": None,
                "実出場時間": None,
                "身長": height_data.get(player_id),
            }
            players.append(player_record)

        if players:
            pd.DataFrame(players).to_csv(self.save_dir / "players.csv", index=False)


def process_statsbomb_skillcorner_single_file(data_df, player_df, tracking_path, metadata_df, config, match_id, save_dir):
    """
    Preprocess single file
    """
    config = load_json(config)
    tracking_path = Path(tracking_path) / f"{match_id}.json"
    tracking_data = json.load(open(tracking_path))
    tracking_df = pd.DataFrame(tracking_data)

    ProcessSkillcornerTrackingData(data_df, player_df, tracking_df, metadata_df, config, save_dir).skillcorner_to_datastadium()
    ProcessStatsbombEventData(data_df, player_df, metadata_df, config, save_dir).statsbomb_to_datastadium()

    print(f"Processing {match_id} finished")


def process_fifawc_single_file(
    event_df, tracking_list, metadata_df, rosters_df, players_df, match_id, save_preprocess_dir, config
):
    """
    Process FIFAWC single file
    """
    config = load_json(config)
    save_preprocess_dir = Path(save_preprocess_dir) / str(match_id)
    save_preprocess_dir.mkdir(exist_ok=True, parents=True)
    ProcessFIFAWCEventData(event_df, match_id, save_preprocess_dir, config).process_event_data()
    ProcessFIFAWCTrackingData(tracking_list, match_id, save_preprocess_dir).process_tracking_data()
    ProcessFIFAWCMetaDataANDRosters(
        rosters_df, metadata_df, players_df, match_id, save_preprocess_dir, config
    ).process_metadata_roster()
    print(f"Processing {match_id} finished")

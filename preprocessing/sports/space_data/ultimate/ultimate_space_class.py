import os

import pandas as pd
from tqdm import tqdm


class Ultimate_space_data:
    def __init__(
        self,
        data_provider,
        tracking_data_path,
        out_path=None,
        testing_mode=False,
    ):
        self.data_provider = data_provider
        self.tracking_path = tracking_data_path
        self.testing_mode = testing_mode
        self.out_path = out_path
        if self.data_provider == "UltimateTrack":
            self.tracking_herz = 15
        elif self.data_provider == "UFATrack":
            self.tracking_herz = 10

    def get_files(self):
        if os.path.isdir(self.tracking_path):
            data_files = [
                os.path.join(self.tracking_path, f)
                for f in os.listdir(self.tracking_path)
                if f.endswith(".csv")
            ]
        elif os.path.isfile(self.tracking_path) and self.tracking_path.endswith(".csv"):
            data_files = [self.tracking_path]
        else:
            raise ValueError(f"Invalid data path: {self.tracking_path}")
        return data_files

    def preprocessing(self):
        tracking_files = self.get_files()
        if self.testing_mode:
            tracking_files = tracking_files[:2]
            print("Running in testing mode. Limited files will be processed.")

        from .ultimate_space_preprocessing import (
            convert_to_metrica_format,
            create_intermediate_file,
            format_tracking_headers,
        )

        home_tracking_dict = {}
        away_tracking_dict = {}
        event_data_dict = {}
        for tracking_path_i in tqdm(
            tracking_files, total=len(tracking_files), desc="Processing tracking files"
        ):
            match_i = os.path.splitext(
                os.path.splitext(os.path.basename(tracking_path_i))[0]
            )[0]
            match_tracking_df = pd.read_csv(tracking_path_i)

            # Create intermediate DataFrame with all required columns
            intermidiate_df = create_intermediate_file(match_tracking_df)

            # Convert to Metrica format
            home_df, away_df, events_df = convert_to_metrica_format(
                intermidiate_df, self.tracking_herz
            )

            home_df = format_tracking_headers(home_df, team_prefix="Home")
            away_df = format_tracking_headers(away_df, team_prefix="Away")

            home_tracking_dict[match_i] = home_df
            away_tracking_dict[match_i] = away_df
            event_data_dict[match_i] = events_df

        if self.out_path:
            # create output directory if not exists
            os.makedirs(self.out_path + "/event", exist_ok=True)
            os.makedirs(self.out_path + "/home_tracking", exist_ok=True)
            os.makedirs(self.out_path + "/away_tracking", exist_ok=True)

            for match_id, df in event_data_dict.items():
                df.to_csv(
                    os.path.join(self.out_path, "event", f"{match_id}.csv"),
                    index=False,
                )
            for match_id, df in home_tracking_dict.items():
                df.to_csv(
                    os.path.join(self.out_path, "home_tracking", f"{match_id}.csv"),
                    index=False,
                )
            for match_id, df in away_tracking_dict.items():
                df.to_csv(
                    os.path.join(self.out_path, "away_tracking", f"{match_id}.csv"),
                    index=False,
                )

        return event_data_dict, home_tracking_dict, away_tracking_dict

    def detect_initiations(
        self,
        velocity_threshold=3.0,
        acceleration_threshold=4.0,
        distance_threshold=5.0,
        player_threshold=2,
    ):
        """
        Detect play initiations in tracking data.

        This method processes raw tracking data (2_1.csv or 1_1_1.csv format),
        calculates required motion features, detects play initiations, and optionally
        extracts individual play segments.

        Args:
            velocity_threshold (float): Threshold for velocity (default: 3.0).
            acceleration_threshold (float): Threshold for acceleration (default: 4.0).
            distance_threshold (float): Threshold for distance (default: 5.0).
            player_threshold (int): Threshold for player proximity (default: 2).
            output_detected_dir (str): Optional directory to save detected play data.
            output_extracted_dir (str): Optional directory to save extracted play segments.

        Returns:
            tuple: (detected_plays_dict, extracted_plays_dict)
                - detected_plays_dict: Dictionary mapping match_id to detected play DataFrame
                - extracted_plays_dict: Dictionary mapping (match_id, offense_id, play_num) to extracted play DataFrame
        """
        tracking_files = self.get_files()
        if self.testing_mode:
            tracking_files = tracking_files[:2]
            print("Running in testing mode. Limited files will be processed.")

        from .ultimate_space_detect_initiation import detect_play, extract_play
        from .ultimate_space_preprocessing import (
            convert_to_metrica_format,
            create_intermediate_file,
            format_tracking_headers,
        )

        detected_plays_dict = {}
        for tracking_path_i in tqdm(
            tracking_files, total=len(tracking_files), desc="Processing tracking files"
        ):
            match_i = os.path.splitext(
                os.path.splitext(os.path.basename(tracking_path_i))[0]
            )[0]

            # Read raw tracking data
            match_tracking_df = pd.read_csv(tracking_path_i)

            # Prepare data with required columns
            intermediate_df = create_intermediate_file(match_tracking_df)

            # Detect play initiations
            detected_df = detect_play(
                intermediate_df,
                velocity_threshold=velocity_threshold,
                acceleration_threshold=acceleration_threshold,
                distance_threshold=distance_threshold,
                player_threshold=player_threshold,
            )

            # Extract individual play segments
            plays = extract_play(detected_df)

            home_tracking_dict = {}
            away_tracking_dict = {}
            event_data_dict = {}
            for play_key, play_df in plays.items():
                # Convert to Metrica format
                home_df, away_df, events_df = convert_to_metrica_format(
                    play_df, self.tracking_herz
                )

                home_df = format_tracking_headers(home_df, team_prefix="Home")
                away_df = format_tracking_headers(away_df, team_prefix="Away")

                home_tracking_dict[play_key] = home_df
                away_tracking_dict[play_key] = away_df
                event_data_dict[play_key] = events_df

                detected_plays_dict[len(detected_plays_dict)] = (
                    f"{match_i}-id{play_key[1]}-play{play_key[2]}"
                )

            if self.out_path:
                # create output directory if not exists
                os.makedirs(self.out_path + "/initiation/event", exist_ok=True)
                os.makedirs(self.out_path + "/initiation/home_tracking", exist_ok=True)
                os.makedirs(self.out_path + "/initiation/away_tracking", exist_ok=True)

                for play_key, df in event_data_dict.items():
                    df.to_csv(
                        os.path.join(
                            self.out_path,
                            "initiation/event",
                            f"{match_i}-id{play_key[1]}-play{play_key[2]}.csv",
                        ),
                        index=False,
                    )
                for play_key, df in home_tracking_dict.items():
                    df.to_csv(
                        os.path.join(
                            self.out_path,
                            "initiation/home_tracking",
                            f"{match_i}-id{play_key[1]}-play{play_key[2]}.csv",
                        ),
                        index=False,
                    )
                for play_key, df in away_tracking_dict.items():
                    df.to_csv(
                        os.path.join(
                            self.out_path,
                            "initiation/away_tracking",
                            f"{match_i}-id{play_key[1]}-play{play_key[2]}.csv",
                        ),
                        index=False,
                    )
        return detected_plays_dict

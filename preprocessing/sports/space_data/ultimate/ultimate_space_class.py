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
        elif self.data_provider == "UFA":
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

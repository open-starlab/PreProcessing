# Target data provider [Metrica, Robocup 2D simulation, Statsbomb, Wyscout, Opta data, DataFactory, sportec]
"""
format of the data source:
 Metrica: csv and json
 Robocup 2D simulation: csv and gz
 Statsbomb: json
 Wyscout: json
 Opta data: xml
 DataFactory: json
 sportec: xml
 DataStadium: csv
 soccertrack: csv and xml
"""

import os
import pickle
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pdb

from preprocessing.sports.SAR_data.soccer.soccer_load_data import clean_robocup_data
from preprocessing.sports.SAR_data.soccer.soccer_SAR_processing import process_frame_data
from preprocessing.sports.SAR_data.soccer.soccer_SAR_state import generate_qmix_state

from . import soccer_load_data
from . import soccer_SAR_processing
from . import soccer_SAR_cleaning
from . import soccer_SAR_state


class RoboCup2D_QMix_Processor:
    """Process RoboCup 2D simulation data into QMix episodes."""
    def __init__(self, data_path, match_id, config_path):
        self.data_path = data_path
        self.match_id = match_id
        self.config_path = config_path
        self.raw_df = None
        self.episodes = []

    def load_data(self):
        # clean and load raw CSV
        self.raw_df = clean_robocup_data(self.data_path, self.match_id)
        # split into QMix episodes
        self.episodes = process_frame_data(self.raw_df)

    def get_episodes(self):
        return self.episodes

    def get_state_at_frame(self, frame_id):
        return generate_qmix_state(self.raw_df, frame_id)


class Soccer_SAR_data:
    """
    Wrapper for all other soccer SAR data providers:
      - datastadium (single or multiple folders)
      - statsbomb_skillcorner (single-file and bulk)
    QMix (robocup_2d) is handled by RoboCup2D_QMix_Processor above.
    """
    def __init__(
        self,
        data_provider,
        state_def,
        data_path,
        match_id=None,
        config_path=None,
        statsbomb_skillcorner_match_id=None,
        max_workers=4,
        preprocess_method=None,
    ):
        self.data_provider = data_provider
        self.state_def = state_def
        self.data_path = data_path
        self.match_id = match_id
        self.config_path = config_path
        self.statsbomb_skillcorner_match_id = statsbomb_skillcorner_match_id
        self.max_workers = max_workers
        self.preprocess_method = preprocess_method

        if data_provider == "statsbomb_skillcorner":
            self.skillcorner_data_dir = os.path.join(self.data_path, "skillcorner", "tracking")

    def load_data_single_file(self, match_id=None):
        """Load one match’s raw data, then clean & preprocess into SAR frames."""
        if match_id:
            self.match_id = match_id

        if self.data_provider == "statsbomb_skillcorner":
            save_dir = os.path.join(os.getcwd(), "data", "stb_skc", "sar_data")
            df, df_players, df_meta = soccer_load_data.load_single_statsbomb_skillcorner(
                self.data_path,
                self.statsbomb_skillcorner_match_id,
                self.match_id,
            )
            soccer_SAR_processing.process_single_file(
                df,
                df_players,
                self.skillcorner_data_dir,
                df_meta,
                self.config_path,
                self.match_id,
                save_dir=save_dir,
            )
            soccer_SAR_cleaning.clean_single_data(
                save_dir,
                self.match_id,
                self.config_path,
                league="laliga",
                state=self.state_def,
                save_dir=os.path.join(os.getcwd(), "data", "stb_skc", "clean_data"),
            )

        elif self.data_provider == "datastadium":
            soccer_SAR_cleaning.clean_single_data(
                self.data_path,
                self.match_id,
                self.config_path,
                league="jleague",
                state=self.state_def,
                save_dir=os.path.join(os.getcwd(), "data", "dss", "clean_data"),
            )

        else:
            raise ValueError(f"Data provider '{self.data_provider}' not supported for single file loading.")

    def load_data(self):
        """Dispatch load either single file or directory batch."""
        print(f"Loading data from {self.data_provider}")

        # Single-file usage
        if (
            (self.data_provider in ("datastadium", "statsbomb_skillcorner"))
            and self.match_id is not None
        ):
            self.load_data_single_file(self.match_id)
            return

        # Bulk directory usage for DataStadium
        if self.data_provider == "datastadium" and os.path.isdir(self.data_path):
            folders = ["Data_20200508", "Data_20210127", "Data_20210208", "Data_20220308"]
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = []
                for folder in folders:
                    folder_path = os.path.join(self.data_path, folder)
                    if not os.path.isdir(folder_path):
                        continue
                    for d in os.listdir(folder_path):
                        futures.append(ex.submit(self.load_data_single_file, d[:10]))
                for _ in tqdm(as_completed(futures), total=len(futures)):
                    pass
            return

        # Bulk directory usage for SkillCorner
        if self.data_provider == "statsbomb_skillcorner" and os.path.isdir(self.skillcorner_data_dir):
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = [
                    ex.submit(self.load_data_single_file, d[:7])
                    for d in os.listdir(self.skillcorner_data_dir)
                ]
                for _ in tqdm(as_completed(futures), total=len(futures)):
                    pass
            return

        raise ValueError(f"Invalid path or parameters for provider '{self.data_provider}'.")

    def preprocess_single_data(self, cleaning_dir, preprocessed_dir):
        """SAR‑style preprocessing for a single match folder."""
        if self.preprocess_method != "SAR":
            raise ValueError(f"Unsupported preprocess method '{self.preprocess_method}'.")

        # delegate to common state‑preprocessing function
        soccer_SAR_state.preprocess_single_game(
            game_dir=cleaning_dir,
            state=self.state_def,
            league="laliga" if self.data_provider == "statsbomb_skillcorner" else "jleague",
            save_dir=preprocessed_dir,
            config=self.config_path,
            match_id=self.match_id,
        )

    def preprocess_data(self, cleaning_dir, preprocessed_dir):
        """SAR‑style preprocessing for all cleaned games in a folder."""
        if self.preprocess_method != "SAR":
            raise ValueError(f"Unsupported preprocess method '{self.preprocess_method}'.")

        # Datastadium bulk
        if self.data_provider == "datastadium":
            match_ids = [d[:10] for d in os.listdir(cleaning_dir)]
        else:  # statsbomb_skillcorner bulk
            match_ids = [d[:7] for d in os.listdir(cleaning_dir)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [
                ex.submit(
                    soccer_SAR_state.preprocess_single_game,
                    cleaning_dir,
                    self.state_def,
                    "laliga" if self.data_provider == "statsbomb_skillcorner" else "jleague",
                    preprocessed_dir,
                    self.config_path,
                    mid,
                )
                for mid in match_ids
            ]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass


if __name__ == "__main__":
    # quick smoke test
    datastadium_dir = "/path/to/Data_20200508/"
    Soccer_SAR_data(
        data_provider="datastadium",
        state_def="PVS",
        data_path=datastadium_dir,
        match_id="2019091416",
        config_path="data/dss/config/preprocessing_dssports2020.json",
        preprocess_method="SAR",
    ).load_data()
    print("done")

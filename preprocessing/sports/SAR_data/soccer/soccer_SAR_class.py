# Target data provider [Metrica,Robocup 2D simulation,Statsbomb,Wyscout,Opta data,DataFactory,sportec]

"""
format of the data source
Metrica:csv and json (tracking data will be included in the future due to lack of matching data)
Robocup 2D simulation:csv and gz
Statsbomb: json
Wyscout: json
Opta data:xml
DataFactory:json
sportec:xml
DataStadium:csv
soccertrack:csv and xml
"""

import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import soccer_load_data
from . import soccer_SAR_processing
from . import soccer_SAR_cleaning
from . import soccer_SAR_state


class Soccer_SAR_data:
    def __init__(
        self,
        data_provider,
        state_def,
        data_path=None,
        match_id=None,
        config_path=None,
        statsbomb_skillcorner_match_id=None,
        max_workers=1,
        preprocess_method="SAR",
    ):
        self.data_provider = data_provider
        self.state_def = state_def
        self.data_path = data_path
        self.match_id = match_id
        self.config_path = config_path
        self.max_workers = max_workers
        self.statsbomb_skillcorner_match_id = statsbomb_skillcorner_match_id
        if self.data_provider == "statsbomb_skillcorner":
            self.skillcorner_data_dir = self.data_path + "/skillcorner/tracking"
        self.preprocess_method = preprocess_method

    def load_data_single_file(self, match_id=None):
        # based on the data provider, load the dataloading function from load_data.py (single file)
        if match_id is not None:
            self.match_id = match_id
        if self.data_provider == "statsbomb_skillcorner":
            save_preprocess_dir = os.getcwd() + "/data/stb_skc/sar_data/"
            df, df_players, df_metadata = soccer_load_data.load_single_statsbomb_skillcorner(
                self.data_path, self.statsbomb_skillcorner_match_id, self.match_id
            )
            soccer_SAR_processing.process_statsbomb_skillcorner_single_file(
                df,
                df_players,
                self.skillcorner_data_dir,
                df_metadata,
                self.config_path,
                self.match_id,
                save_dir=save_preprocess_dir,
            )
            soccer_SAR_cleaning.clean_single_data(
                save_preprocess_dir,
                self.match_id,
                self.config_path,
                "laliga",
                self.state_def,
                save_dir=os.getcwd() + "/data/stb_skc/clean_data",
            )
        elif self.data_provider == "fifawc":
            save_preprocess_dir = os.getcwd() + "/data/fifawc/sar_data/"
            event_df, tracking_list, metadata_df, rosters_df, players_df = soccer_load_data.load_single_fifawc(
                self.data_path, self.match_id
            )
            soccer_SAR_processing.process_fifawc_single_file(
                event_df,
                tracking_list,
                metadata_df,
                rosters_df,
                players_df,
                self.match_id,
                save_preprocess_dir,
                self.config_path,
            )
            soccer_SAR_cleaning.clean_single_data(
                save_preprocess_dir,
                self.match_id,
                self.config_path,
                "fifawc",
                self.state_def,
                save_dir=os.getcwd() + "/data/fifawc/clean_data",
            )
        elif self.data_provider == "datastadium":
            soccer_SAR_cleaning.clean_single_data(
                self.data_path,
                self.match_id,
                self.config_path,
                "jleague",
                self.state_def,
                save_dir=os.getcwd() + "/data/dss/clean_data",
            )
        else:
            raise ValueError("Data provider not supported or not found")

    def load_data(self):
        print(f"Loading data from {self.data_provider}")
        # check if processing single file or multiple files
        if (
            (self.data_provider == "datastadium" and self.match_id is not None)
            or (self.data_provider == "statsbomb_skillcorner" and self.match_id is not None)
            or (self.data_provider == "fifawc" and self.match_id is not None)
        ):
            # Load single file - call load_data_single_file
            self.load_data_single_file()
        # load data from multiple files with parallel processing
        elif (self.data_path is not None and os.path.isdir(self.data_path)) or (
            self.data_provider == "statsbomb_skillcorner" and self.match_id is None
        ):
            # statsbomb_skillcorner
            if self.data_provider == "statsbomb_skillcorner":
                match_id_list = [d[:7] for d in os.listdir(self.skillcorner_data_dir)]
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for match_id in match_id_list:
                        # Create temporary instance for each match
                        temp_instance = Soccer_SAR_data(
                            data_provider=self.data_provider,
                            state_def=self.state_def,
                            data_path=self.data_path,
                            match_id=match_id,
                            config_path=self.config_path,
                            statsbomb_skillcorner_match_id=self.statsbomb_skillcorner_match_id,
                            max_workers=1,  # Set to 1 to avoid nested parallelization
                            preprocess_method=self.preprocess_method,
                        )
                        futures.append(executor.submit(temp_instance.load_data_single_file))
                    # Collect results as they are completed
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        future.result()
            elif self.data_provider == "fifawc":
                event_dir = self.data_path / "Event Data"
                match_id_list = sorted([f.stem for f in event_dir.glob("*.json")])
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for match_id in match_id_list:
                        # Create temporary instance for each match
                        temp_instance = Soccer_SAR_data(
                            data_provider=self.data_provider,
                            state_def=self.state_def,
                            data_path=self.data_path,
                            match_id=match_id,
                            config_path=self.config_path,
                            max_workers=1,  # Set to 1 to avoid nested parallelization
                            preprocess_method=self.preprocess_method,
                        )
                        futures.append(executor.submit(temp_instance.load_data_single_file))
                    # Collect results as they are completed
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        future.result()
            elif self.data_provider == "datastadium":
                folder_name_list = ["Data_20200508/", "Data_20210127/", "Data_20210208/", "Data_20220308/"]
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for folder_name in folder_name_list:
                        data_path = os.path.join(self.data_path, folder_name)
                        match_id_list = [d[:10] for d in os.listdir(data_path)]
                        for match_id in match_id_list:
                            # Create temporary instance for each match
                            temp_instance = Soccer_SAR_data(
                                data_provider=self.data_provider,
                                state_def=self.state_def,
                                data_path=data_path,
                                match_id=match_id,
                                config_path=self.config_path,
                                max_workers=1,  # Set to 1 to avoid nested parallelization
                                preprocess_method=self.preprocess_method,
                            )
                            futures.append(executor.submit(temp_instance.load_data_single_file))
                    # Collect results as they are completed
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        future.result()
        else:
            raise ValueError("Event path is not a valid file or directory")
        print(f"Loaded data from {self.data_provider}")

    def preprocess_single_data(self, cleaning_dir=None, preprocessed_dir=None):
        # Set default directories if not provided
        if cleaning_dir is None:
            if self.data_provider == "datastadium":
                cleaning_dir = os.getcwd() + "/data/dss/clean_data"
            elif self.data_provider == "statsbomb_skillcorner":
                cleaning_dir = os.getcwd() + "/data/stb_skc/clean_data"
            elif self.data_provider == "fifawc":
                cleaning_dir = os.getcwd() + "/data/fifawc/clean_data"

        if preprocessed_dir is None:
            if self.data_provider == "datastadium":
                preprocessed_dir = os.getcwd() + "/data/dss/preprocess_data"
            elif self.data_provider == "statsbomb_skillcorner":
                preprocessed_dir = os.getcwd() + "/data/stb_skc/preprocess_data"
            elif self.data_provider == "fifawc":
                preprocessed_dir = os.getcwd() + "/data/fifawc/preprocess_data"

        # Preprocess the loaded data
        if self.preprocess_method == "SAR":
            if self.data_provider == "datastadium":
                soccer_SAR_state.preprocess_single_game(
                    cleaning_dir,
                    state=self.state_def,
                    league="jleague",
                    save_dir=preprocessed_dir,
                    config=self.config_path,
                    match_id=self.match_id,
                )
            elif self.data_provider == "statsbomb_skillcorner":
                soccer_SAR_state.preprocess_single_game(
                    cleaning_dir,
                    state=self.state_def,
                    league="laliga",
                    save_dir=preprocessed_dir,
                    config=self.config_path,
                    match_id=self.match_id,
                )
            elif self.data_provider == "fifawc":
                soccer_SAR_state.preprocess_single_game(
                    cleaning_dir,
                    state=self.state_def,
                    league="fifawc",
                    save_dir=preprocessed_dir,
                    config=self.config_path,
                    match_id=self.match_id,
                )
            else:
                raise ValueError(f"Preprocessing method not supported for {self.data_provider}")
        else:
            raise ValueError(f"Preprocessing method not supported for {self.preprocess_method}")

    def preprocess_data(self, cleaning_dir=None, preprocessed_dir=None):
        if self.preprocess_method == "SAR":
            # First, load the data
            print("Starting data preprocessing...")
            # self.load_data()

            # Set default directories if not provided
            if cleaning_dir is None:
                if self.data_provider == "datastadium":
                    cleaning_dir = os.getcwd() + "/data/dss/clean_data"
                elif self.data_provider == "statsbomb_skillcorner":
                    cleaning_dir = os.getcwd() + "/data/stb_skc/clean_data"
                elif self.data_provider == "fifawc":
                    cleaning_dir = os.getcwd() + "/data/fifawc/clean_data"

            if preprocessed_dir is None:
                if self.data_provider == "datastadium":
                    preprocessed_dir = os.getcwd() + "/data/dss/preprocess_data"
                elif self.data_provider == "statsbomb_skillcorner":
                    preprocessed_dir = os.getcwd() + "/data/stb_skc/preprocess_data"
                elif self.data_provider == "fifawc":
                    preprocessed_dir = os.getcwd() + "/data/fifawc/preprocess_data"

            # Check if processing single file or multiple files
            if (
                (self.data_provider == "datastadium" and self.match_id is not None)
                or (self.data_provider == "statsbomb_skillcorner" and self.match_id is not None)
                or (self.data_provider == "fifawc" and self.match_id is not None)
            ):
                # Process single file - call preprocess_single_data (skip load_data since already called)
                self.preprocess_single_data(cleaning_dir, preprocessed_dir)
            else:
                # Process multiple files with parallel processing
                if self.data_provider == "datastadium":
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = []
                        match_id_list = [d[:10] for d in os.listdir(cleaning_dir)]
                        for match_id in match_id_list:
                            # Create temporary instance for each match
                            temp_instance = Soccer_SAR_data(
                                data_provider=self.data_provider,
                                state_def=self.state_def,
                                data_path=self.data_path,
                                match_id=match_id,
                                config_path=self.config_path,
                                preprocess_method=self.preprocess_method,
                            )
                            futures.append(
                                executor.submit(
                                    temp_instance.preprocess_single_data,
                                    cleaning_dir,
                                    preprocessed_dir,
                                )
                            )
                        # Collect results as they are completed
                        for future in tqdm(as_completed(futures), total=len(futures)):
                            future.result()

                elif self.data_provider == "statsbomb_skillcorner":
                    match_id_list = [d[:7] for d in os.listdir(cleaning_dir)]
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = []
                        for match_id in match_id_list:
                            # Create temporary instance for each match
                            temp_instance = Soccer_SAR_data(
                                data_provider=self.data_provider,
                                state_def=self.state_def,
                                data_path=self.data_path,
                                match_id=match_id,
                                config_path=self.config_path,
                                statsbomb_skillcorner_match_id=self.statsbomb_skillcorner_match_id,
                                preprocess_method=self.preprocess_method,
                            )
                            futures.append(
                                executor.submit(
                                    temp_instance.preprocess_single_data,
                                    cleaning_dir,
                                    preprocessed_dir,
                                )
                            )
                        # Collect results as they are completed
                        for future in tqdm(as_completed(futures), total=len(futures)):
                            future.result()
                elif self.data_provider == "fifawc":
                    match_id_list = [d for d in os.listdir(cleaning_dir)]
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = []
                        for match_id in match_id_list:
                            # Create temporary instance for each match
                            temp_instance = Soccer_SAR_data(
                                data_provider=self.data_provider,
                                state_def=self.state_def,
                                data_path=self.data_path,
                                match_id=match_id,
                                config_path=self.config_path,
                                preprocess_method=self.preprocess_method,
                            )
                            futures.append(
                                executor.submit(
                                    temp_instance.preprocess_single_data,
                                    cleaning_dir,
                                    preprocessed_dir,
                                )
                            )
                        # Collect results as they are completed
                        for future in tqdm(as_completed(futures), total=len(futures)):
                            future.result()
                else:
                    raise ValueError(f"Preprocessing method not supported for {self.data_provider}")
        else:
            raise ValueError(
                "Preprocessing method is not defined. Please set preprocess_method to 'SAR' or other valid methods."
            )

        print("Data preprocessing completed successfully!")


if __name__ == "__main__":
    # Load each data provider
    # test load_statsbomb_skillcorner single file
    # Soccer_SAR_data(
    #     data_provider='statsbomb_skillcorner',
    #     data_path=statsbomb_skillcorner_path,
    #     match_id="1120811", # match_id for skillcorner
    #     config_path=os.getcwd()+"/data/stb_skc/config/preprocessing_statsbomb_skillcorner2024.json",
    #     statsbomb_skillcorner_match_id=statsbomb_skillcorner_match_id,
    # ).load_data()

    # test load datastadium single file
    # Soccer_SAR_data(
    #     data_provider='datastadium',
    #     data_path=datastadium_dir,
    #     match_id="2019091416",
    #     config_path=os.getcwd()+"/data/dss/config/preprocessing_dssports2020.json",
    # ).load_data()

    # test load_statsbomb_skillcorner multiple files
    # Soccer_SAR_data(
    #     data_provider='statsbomb_skillcorner',
    #     data_path=statsbomb_skillcorner_path,
    #     config_path=os.getcwd()+"/data/stb_skc/config/preprocessing_statsbomb_skillcorner2024.json",
    #     statsbomb_skillcorner_match_id=statsbomb_skillcorner_match_id,
    #     max_workers=2
    # ).load_data()

    # #test load_datastadium multiple files
    # Soccer_SAR_data(
    #     data_provider='datastadium',
    #     data_path=datastadium_dir,
    #     config_path=os.getcwd()+"/data/dss/config/preprocessing_dssports2020.json",
    #     max_workers=2
    # ).load_data()

    # Preprocess each data provider

    # test preprocess statsbomb_skillcorner single file
    # Soccer_SAR_data(
    #     data_provider='statsbomb_skillcorner',
    #     data_path=statsbomb_skillcorner_path,
    #     match_id="1120811", # match_id for skillcorner
    #     config_path=os.getcwd()+"/data/stb_skc/config/preprocessing_statsbomb_skillcorner2024.json",
    #     preprocess_method="SAR"
    # ).preprocess_single_data(
    #     cleaning_dir=os.getcwd()+"/data/stb_skc/clean_data",
    #     preprocessed_dir=os.getcwd()+"/data/stb_skc/preprocess_data"
    # )

    # test preprocess datastadium single file
    # Soccer_SAR_data(
    #     data_provider='datastadium',
    #     match_id="2019091416",
    #     config_path="data/dss/config/preprocessing_dssports2020.json",
    #     preprocess_method="SAR"
    # ).preprocess_single_data(
    #     cleaning_dir="data/dss/clean_data",
    #     preprocessed_dir="data/dss/preprocess_data"
    # )

    # test preprocess statsbomb_skillcorner multiple files
    # Soccer_SAR_data(
    #     data_provider='statsbomb_skillcorner',
    #     data_path=statsbomb_skillcorner_path,
    #     config_path=os.getcwd()+"/data/stb_skc/config/preprocessing_statsbomb_skillcorner2024.json",
    #     preprocess_method="SAR",
    #     max_workers=2
    # ).preprocess_data(
    #     cleaning_dir=os.getcwd()+"/data/stb_skc/clean_data",
    #     preprocessed_dir=os.getcwd()+"/data/stb_skc/preprocess_data"
    # )

    # test preprocess datastadium multiple files
    # Soccer_SAR_data(
    #     data_provider='datastadium',
    #     config_path=os.getcwd()+"/data/dss/config/preprocessing_dssports2020.json",
    #     preprocess_method="SAR",
    #     max_workers=2
    # ).preprocess_data(
    #     cleaning_dir=os.getcwd()+"/data/dss/clean_data",
    #     preprocessed_dir=os.getcwd()+"/data/dss/preprocess_data"
    # )

    print("-----------done-----------")

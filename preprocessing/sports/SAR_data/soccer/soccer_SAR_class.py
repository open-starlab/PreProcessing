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


# create a class to wrap the data source
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
        preprocess_method=None,
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
            soccer_SAR_processing.process_single_file(
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
        # check if the event path is a single file or a directory
        if (
            (self.data_provider == "datastadium" and self.match_id is not None)
            or (self.data_provider == "statsbomb" and self.match_id is None and os.path.isfile(self.data_path))
            or (self.data_provider == "statsbomb_skillcorner" and self.match_id is not None)
        ):
            self.load_data_single_file()
        # load data from multiple files
        elif (
            (self.data_path is not None and os.path.isdir(self.data_path))
            or self.data_provider == "statsbomb"
            or (self.data_provider == "statsbomb_skillcorner" and self.match_id is None)
        ):
            # statsbomb_skillcorner
            if self.data_provider == "statsbomb_skillcorner":
                # import pdb; pdb.set_trace()
                match_id_list = [d[:7] for d in os.listdir(self.skillcorner_data_dir)]
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit tasks to the executor
                    futures = [executor.submit(self.load_data_single_file, match_id) for match_id in match_id_list]
                    # Collect results as they are completed
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        future.result()

            elif self.data_provider == "datastadium":
                folder_name_list = ["Data_20200508/", "Data_20210127/", "Data_20210208/", "Data_20220308/"]
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    for folder_name in folder_name_list:
                        data_path = os.path.join(self.data_path, folder_name)
                        self.data_path = data_path
                        match_id_list = [d[:10] for d in os.listdir(data_path)]
                        futures = [executor.submit(self.load_data_single_file, match_id) for match_id in match_id_list]
                    # Collect results as they are completed
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        future.result()
        else:
            raise ValueError("Event path is not a valid file or directory")
        print(f"Loaded data from {self.data_provider}")

    def load_match_statsbomb_skillcorner(
        self,
        i,
        match_id_df,
        statsbomb_skillcorner_event_path,
        statsbomb_skillcorner_tracking_path,
        statsbomb_skillcorner_match_path,
    ):
        statsbomb_match_id = match_id_df.loc[i, "match_id_statsbomb"]
        skillcorner_match_id = match_id_df.loc[i, "match_id_skillcorner"]
        try:
            statsbomb_skillcorner_df = soccer_load_data.load_statsbomb_skillcorner(
                statsbomb_skillcorner_event_path,
                statsbomb_skillcorner_tracking_path,
                statsbomb_skillcorner_match_path,
                statsbomb_match_id,
                skillcorner_match_id,
            )
        except Exception as e:
            print(f"Error loading match {statsbomb_match_id} from StatsBomb SkillCorner: {e}")
            statsbomb_skillcorner_df = None
        return statsbomb_skillcorner_df

    def preprocess_single_data(self, cleaning_dir, preprocessed_dir):
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
            else:
                raise ValueError(f"Preprocessing method not supported for {self.data_provider}")
        else:
            raise ValueError(f"Preprocessing method not supported for {self.preprocess_method}")

    def preprocess_data(self, cleaning_dir, preprocessed_dir):
        if self.preprocess_method == "SAR":
            if self.data_provider == "datastadium":
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    match_id_list = [d[:10] for d in os.listdir(cleaning_dir)]
                    for match_id in match_id_list:
                        futures.append(
                            executor.submit(
                                soccer_SAR_state.preprocess_single_game,
                                cleaning_dir,
                                self.state_def,
                                "jleague",
                                preprocessed_dir,
                                self.config_path,
                                match_id,
                            )
                        )
                    # Collect results as they are completed
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        future.result()

            elif self.data_provider == "statsbomb_skillcorner":
                match_id_list = [d[:7] for d in os.listdir(cleaning_dir)]
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(
                            soccer_SAR_state.preprocess_single_game,
                            cleaning_dir,
                            self.state_def,
                            "laliga",
                            preprocessed_dir,
                            self.config_path,
                            match_id,
                        )
                        for match_id in match_id_list
                    ]
                    # Collect results as they are completed
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        future.result()
            else:
                raise ValueError(f"Preprocessing method not supported for {self.data_provider}")
        else:
            raise ValueError(f"Preprocessing method not supported for {self.preprocess_method}")


if __name__ == "__main__":
    statsbomb_skillcorner_path = "/data_pool_1/laliga_23"
    statsbomb_skillcorner_match_id = os.getcwd() + "/preprocessing/sports/SAR_data/match_id_dict.json"

    datastadium_dir = "/work5/fujii/work/JLeagueData/"

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
    #     data_path=os.path.join(datastadium_dir, "Data_20200508/"),
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
    #     config_path="/home/k_ide/workspace6/open-starlab/PreProcessing/data/dss/config/preprocessing_dssports2020.json",
    #     preprocess_method="SAR"
    # ).preprocess_single_data(
    #     cleaning_dir="/home/k_ide/workspace6/open-starlab/PreProcessing/data/dss/clean_data",
    #     preprocessed_dir="/home/k_ide/workspace6/open-starlab/PreProcessing/data/dss/preprocess_data"
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

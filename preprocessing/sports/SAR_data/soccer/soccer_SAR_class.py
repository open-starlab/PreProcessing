#Target data provider [Metrica,Robocup 2D simulation,Statsbomb,Wyscout,Opta data,DataFactory,sportec]

'''
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
'''

import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == '__main__':
    import soccer_load_data
    import soccer_SAR_processing
    import soccer_SAR_cleaning
    import soccer_SAR_state
else:
    from . import soccer_load_data
    from . import soccer_SAR_processing
    from . import soccer_SAR_cleaning
    from . import soccer_SAR_state
import pdb

#create a class to wrap the data source
class Soccer_SAR_data:
    def __init__(self,data_provider,event_path=None,match_id=None,tracking_home_path=None,tracking_away_path=None,
                 tracking_path=None,meta_data=None,statsbomb_api_args=[],
                 statsbomb_match_id=None,skillcorner_match_id=None,max_workers=1,match_id_df=None,
                 statsbomb_event_dir=None, skillcorner_tracking_dir=None, skillcorner_match_dir=None,
                 preprocess_method=None,sb360_path=None,wyscout_matches_path=None,
                 st_track_path=None, st_meta_path=None,verbose=False,
                 preprocess_tracking=False):
        self.data_provider = data_provider
        self.event_path = event_path
        self.match_id = match_id
        self.tracking_home_path = tracking_home_path
        self.tracking_away_path = tracking_away_path
        self.tracking_path = tracking_path  
        self.meta_data = meta_data
        self.statsbomb_api_args = statsbomb_api_args
        self.statsbomb_match_id = statsbomb_match_id
        self.sb360_path = sb360_path
        self.skillcorner_match_id = skillcorner_match_id
        self.max_workers = max_workers
        self.match_id_df = match_id_df
        self.statsbomb_event_dir = statsbomb_event_dir
        self.skillcorner_tracking_dir = skillcorner_tracking_dir
        self.skillcorner_match_dir = skillcorner_match_dir
        self.preprocess_method = preprocess_method
        self.preprocess_tracking = preprocess_tracking
        self.verbose = verbose
        self.call_preprocess = False

    def load_data_single_file(self):
        #based on the data provider, load the dataloading function from load_data.py (single file)
        if self.data_provider == 'datafactory':
            df=soccer_load_data.load_datafactory(self.event_path)
        elif self.data_provider == 'statsbomb':
            df=soccer_load_data.load_statsbomb(self.event_path,sb360_path=self.sb360_path,match_id=self.statsbomb_match_id,*self.statsbomb_api_args)
        elif self.data_provider == 'statsbomb_skillcorner':
            df=soccer_load_data.load_statsbomb_skillcorner(statsbomb_event_dir=self.statsbomb_event_dir, skillcorner_tracking_dir=self.skillcorner_tracking_dir, skillcorner_match_dir=self.skillcorner_match_dir, statsbomb_match_id=self.statsbomb_match_id, skillcorner_match_id=self.skillcorner_match_id)
            if self.preprocess_tracking and not self.call_preprocess:
                df=soccer_tracking_data.statsbomb_skillcorner_tracking_data_preprocessing(df)
            if self.preprocess_method is not None and not self.call_preprocess:
                df=soccer_tracking_data.statsbomb_skillcorner_event_data_preprocessing(df,process_event_coord=False)
        elif self.data_provider == 'datastadium':
            df=soccer_load_data.load_datastadium(self.event_path,self.tracking_home_path,self.tracking_away_path)
        else:
            raise ValueError('Data provider not supported or not found')
        return 
    
    def load_data(self):
        print(f'Loading data from {self.data_provider}')
        #check if the event path is a single file or a directory
        if ((self.event_path is not None and os.path.isfile(self.event_path)) and self.data_provider != 'statsbomb') or \
           (self.data_provider == 'statsbomb' and self.statsbomb_match_id is None and os.path.isfile(self.event_path)) or \
            (self.data_provider == 'statsbomb_skillcorner' and self.statsbomb_match_id is not None):
            df = self.load_data_single_file()
        #load data from multiple files
        elif (self.event_path is not None and os.path.isdir(self.event_path)) or self.data_provider == 'statsbomb' or \
            (self.data_provider == 'statsbomb_skillcorner' and self.statsbomb_match_id is None and self.skillcorner_match_id is None):
            #statsbomb_skillcorner
            if self.data_provider == 'statsbomb_skillcorner':
                out_df_list = []
                self.match_id_df = pd.read_csv(self.match_id_df)
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit tasks to the executor
                    futures = [executor.submit(self.load_match_statsbomb_skillcorner, i, self.match_id_df, 
                                               self.statsbomb_event_dir,self.skillcorner_tracking_dir,self.skillcorner_match_dir) 
                                               for i in range(len(self.match_id_df))]
                    # Collect the results as they complete
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        out_df_list.append(future.result())
                df = pd.concat(out_df_list)
            elif self.data_provider == "datastadium":
                out_df_list = []

                event_dir = self.event_path

                def process_event_folder(f):
                    # Define file paths for the current event folder
                    self.event_path = os.path.join(event_dir, f, 'play.csv')
                    self.tracking_home_path = os.path.join(event_dir, f, 'home_tracking.csv')
                    self.tracking_away_path = os.path.join(event_dir, f, 'away_tracking.csv')

                    # Load data
                    df = self.load_data_single_file()
                    return df

                # Initialize ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Get list of event folders
                    event_folders = sorted(f for f in os.listdir(self.event_path) if not (f.startswith('.') or f.startswith('@')))
                    # Submit tasks to the executor
                    future_to_event = {executor.submit(process_event_folder, folder): folder for folder in event_folders}
                    # Collect results
                    out_df_list = []
                    for future in tqdm(as_completed(future_to_event), total=len(future_to_event)):
                        try:
                            df = future.result()
                            out_df_list.append(df)
                        except Exception as e:
                            print(f'Error processing folder {future_to_event[future]}: {e}')
                self.event_path = event_dir
                df = pd.concat(out_df_list)

        else:
            raise ValueError('Event path is not a valid file or directory')
        print(f'Loaded data from {self.data_provider}')
        return df
        
    def load_match_statsbomb_skillcorner(self,i, match_id_df, statsbomb_skillcorner_event_path, 
                                            statsbomb_skillcorner_tracking_path, statsbomb_skillcorner_match_path):
        statsbomb_match_id = match_id_df.loc[i, "match_id_statsbomb"]
        skillcorner_match_id = match_id_df.loc[i, "match_id_skillcorner"]
        try:
            statsbomb_skillcorner_df = soccer_load_data.load_statsbomb_skillcorner(
                statsbomb_skillcorner_event_path, 
                statsbomb_skillcorner_tracking_path, 
                statsbomb_skillcorner_match_path, 
                statsbomb_match_id, 
                skillcorner_match_id
            )
        except: #Exception as e: 
            # print("An error occurred:", e)
            print(f"Skipped match statsbomb match_id: {statsbomb_match_id}")
            statsbomb_skillcorner_df=None
        return statsbomb_skillcorner_df


if __name__ == '__main__':
    # test load_statsbomb_skillcorner
    statsbomb_skillcorner_event_path="/data_pool_1/laliga_23/statsbomb/events"
    statsbomb_skillcorner_tracking_path="/data_pool_1/laliga_23/skillcorner/tracking"
    statsbomb_skillcorner_match_path=os.getcwd()+"/scripts/match_id_dict.json"

    datastadium_event_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/play.csv"
    datastadium_tracking_home_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/home_tracking.csv"
    datastadium_tracking_away_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/away_tracking.csv"
    datastadium_dir="/work2/fujii/JLeagueData/Data_2019FM"


    statsbomb_skillcorner_df=Soccer_SAR_data(data_provider='statsbomb_skillcorner',
                                        statsbomb_event_dir=statsbomb_skillcorner_event_path,
                                        skillcorner_tracking_dir=statsbomb_skillcorner_tracking_path,
                                        skillcorner_match_dir=statsbomb_skillcorner_match_path,
                                        match_id_df=os.getcwd()+'/preprocessing/example/id_matching.csv',
                                        max_workers=10).load_data()
    statsbomb_skillcorner_df.head(10000).to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb_skillcorner/test_data_main_multi.csv",index=False)

    
    #test load_datastadium multiple files
    # datastadium_df=Soccer_SAR_data(data_provider='datastadium',event_path=datastadium_dir,max_workers=10).load_data()
    # datastadium_df.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/load_class_multi.csv",index=False)

    print("-----------done-----------")


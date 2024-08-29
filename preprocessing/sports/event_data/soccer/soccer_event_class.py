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
'''

import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == '__main__':
    import soccer_load_data
    import soccer_processing
else:
    from . import soccer_load_data
    from . import soccer_processing
import pdb

#create a class to wrap the data source
class Soccer_event_data:
    def __init__(self,data_provider,event_path=None,match_id=None,tracking_home_path=None,tracking_away_path=None,
                 tracking_path=None,meta_data=None,statsbomb_api_args=[],
                 statsbomb_match_id=None,skillcorner_match_id=None,max_workers=1,match_id_df=None,
                 statsbomb_event_dir=None, skillcorner_tracking_dir=None, skillcorner_match_dir=None,
                 preprocess_method=None,sb360_path=None,wyscout_matches_path=None):
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
        self.wyscout_matches_path=wyscout_matches_path

    def load_data_single_file(self):
        #based on the data provider, load the dataloading function from load_data.py (single file)
        if self.data_provider == 'datafactory':
            df=soccer_load_data.load_datafactory(self.event_path)
        elif self.data_provider == 'metrica':
            df=soccer_load_data.load_metrica(self.event_path,match_id=self.match_id,tracking_home_path=self.tracking_home_path,tracking_away_path=self.tracking_away_path)
        elif self.data_provider == 'opta':
            df=soccer_load_data.load_opta(self.event_path,match_id=self.match_id)
        elif self.data_provider == 'robocup_2d':
            df=soccer_load_data.load_robocup_2d(self.event_path,match_id=self.match_id,tracking_path=self.tracking_path)
        elif self.data_provider == 'sportec':
            df=soccer_load_data.load_sportec(self.event_path,tracking_path=self.tracking_path,meta_path=self.meta_data)
        elif self.data_provider == 'statsbomb':
            df=soccer_load_data.load_statsbomb(self.event_path,sb360_path=self.sb360_path,match_id=self.statsbomb_match_id,*self.statsbomb_api_args)
        elif self.data_provider == 'statsbomb_skillcorner':
            df=soccer_load_data.load_statsbomb_skillcorner(statsbomb_event_dir=self.statsbomb_event_dir, skillcorner_tracking_dir=self.skillcorner_tracking_dir, skillcorner_match_dir=self.skillcorner_match_dir, statsbomb_match_id=self.statsbomb_match_id, skillcorner_match_id=self.skillcorner_match_id)
        elif self.data_provider == 'wyscout':
            df=soccer_load_data.load_wyscout(self.event_path,self.wyscout_matches_path)
        elif self.data_provider == 'datastadium':
            df=soccer_load_data.load_datastadium(self.event_path,self.tracking_home_path,self.tracking_away_path)
        else:
            raise ValueError('Data provider not supported or not found')
        return df
    
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
            #other data providers
            elif self.data_provider in ['datafactory','opta','wyscout']:
                event_path = self.event_path
                files = sorted(os.listdir(self.event_path))
                files = [f for f in files if not f.startswith('.')]
                out_df_list = []
                if self.data_provider == "opta":
                    if self.match_id is None:
                        match_id=self.match_id
                elif self.data_provider == "wyscout":
                    matches_path=self.wyscout_matches_path
                count=0
                for f in tqdm(files, total=len(files)):
                    if self.data_provider == "opta":
                        if self.match_id is None:
                            self.match_id = match_id[count]
                        else:
                            self.match_id = count
                        count+=1
                    elif self.data_provider == "wyscout":
                        self.wyscout_matches_path=os.path.join(matches_path, f.replace("events_","matches_"))
                    self.event_path = os.path.join(event_path, f)
                    df = self.load_data_single_file()
                    out_df_list.append(df)
                df = pd.concat(out_df_list)
                self.event_path = event_path
                if self.data_provider == "opta":
                    self.match_id = match_id
                elif self.data_provider == "wyscout":
                    self.wyscout_matches_path=matches_path
            # other data providers
            elif self.data_provider in ['metrica','robocup_2d','sportec']:
                #warnging that the event data and tracking data will be matched via the file name
                print('Warning: Event data and tracking data will be matched via the file name')
                event_path = self.event_path
                files = sorted(os.listdir(self.event_path))
                files = [f for f in files if not f.startswith('.')]
                out_df_list = []
                if self.data_provider in ['metrica']:
                    tracking_home_path = self.tracking_home_path
                    tracking_away_path = self.tracking_away_path
                    for f in files:
                        self.event_path = os.path.join(event_path, f)
                        self.tracking_home_path = os.path.join(tracking_home_path,f.replace("RawEventsData","RawTrackingData_Home_Team"))
                        self.tracking_away_path = os.path.join(tracking_away_path,f.replace("RawEventsData","RawTrackingData_Away_Team"))
                        #check if the tracking data exists
                        if os.path.isfile(self.tracking_home_path) and os.path.isfile(self.tracking_away_path):
                            df = self.load_data_single_file()
                            out_df_list.append(df)
                        else:
                            print(f'Tracking data not found for {f}')
                    df = pd.concat(out_df_list)
                    self.event_path = event_path
                    self.tracking_home_path = tracking_home_path
                    self.tracking_away_path = tracking_away_path
                elif self.data_provider == 'robocup_2d':
                    tracking_path = self.tracking_path
                    for f in files:
                        self.event_path = os.path.join(event_path, f)
                        self.tracking_path = os.path.join(tracking_path,f.replace("pass",""))
                        self.match_id = f.replace("pass","").replace(".csv","")
                        if os.path.isfile(self.tracking_path):
                            df = self.load_data_single_file()
                            out_df_list.append(df)
                        else:
                            print(f'Tracking data not found for {f}')
                    df = pd.concat(out_df_list)
                    self.event_path = event_path
                    self.tracking_path = tracking_path
                    self.match_id = None
                elif self.data_provider == 'sportec':
                    tracking_path = self.tracking_path
                    meta_path = self.meta_data
                    for f in files:
                        self.event_path = os.path.join(event_path, f)
                        self.tracking_path = os.path.join(tracking_path,f.replace("events","positional"))
                        self.meta_path = os.path.join(meta_path,f.replace("events","meta"))
                        if os.path.isfile(self.tracking_path) and os.path.isfile(self.meta_path):
                            df = self.load_data_single_file()
                            out_df_list.append(df)
                        else:
                            print(f'Tracking data or Meta data not found for {f}')
                    df = pd.concat(out_df_list)
                    self.event_path = event_path
                    self.tracking_path = tracking_path
                    self.meta_path = meta_path
            # statsbomb
            elif self.data_provider == 'statsbomb':
                print('Warning: Event data and 360 data will be matched via the file name')
                out_df_list = []
                if self.statsbomb_match_id is None:
                    files = sorted(os.listdir(self.event_path))
                    files = [f for f in files if not f.startswith('.')]
                    event_path = self.event_path
                    sb360_path = self.sb360_path
                    def process_file(f):
                        event_path_local = os.path.join(event_path, f)
                        sb360_path_local = os.path.join(sb360_path, f) if sb360_path is not None else None
                        self.event_path = event_path_local
                        self.sb360_path = sb360_path_local
                        return self.load_data_single_file()

                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = {executor.submit(process_file, f): f for f in files}
                        for future in tqdm(as_completed(futures), total=len(futures)):
                            result = future.result()
                            if result is not None:
                                out_df_list.append(result)

                    df = pd.concat(out_df_list)
                    self.event_path = event_path
                    self.sb360_path = sb360_path
                else:
                    if isinstance(self.statsbomb_match_id, list):
                        files = self.statsbomb_match_id
                    else:
                        files = [self.statsbomb_match_id]
                    
                    def process_id(f):
                        self.statsbomb_match_id = str(f)
                        return self.load_data_single_file()
                    
                    for f in tqdm(files, total=len(files)):
                        out_df_list.append(process_id(f))

                    df = pd.concat(out_df_list)
                    self.statsbomb_match_id = files
            # datastadium
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
        except:
            print(f"Skipped match statsbomb match_id: {statsbomb_match_id}")
            statsbomb_skillcorner_df=None
        return statsbomb_skillcorner_df
    
    def preprocessing_single_df(self,df):
        df_out=None
        if self.data_provider in ["statsbomb", "wyscout","statsbomb_skillcorner","datastadium"]:
            if self.data_provider in ["statsbomb","statsbomb_skillcorner"]:
                df = df.reset_index(drop=True)
                df_out=soccer_processing.UIED_statsbomb(df)
            elif self.data_provider == "datastadium":
                df_out=soccer_processing.UIED_datastadium(df)
            elif self.data_provider == "wyscout":
                if self.preprocess_method == "UIED":
                    df_out=soccer_processing.UIED_wyscout(df)
                elif self.preprocess_method == "LEM":
                    df_out=soccer_processing.lem(df)
                elif self.preprocess_method == "NMSTPP":
                    df_out=soccer_processing.nmstpp(df)
                elif self.preprocess_method == "SEQ2EVENT":
                    df_out=soccer_processing.seq2event(df)
                else:
                    raise ValueError(f'Preprocessing method {self.preprocess_method} not found')
        else:
            raise ValueError(f'Preprocessing method not supported for {self.data_provider}')
        return df_out
    
    def preprocessing(self):
        print(f'Preprocessing data from {self.data_provider} with method {self.preprocess_method}')
        if self.preprocess_method is not None:
            df = self.load_data()
            out_df_list = []

            # df_out=self.preprocessing_single_df(df)
            # return df_out

            def process_single_match(match_id):
                df_single = df[df.match_id == match_id]
                return self.preprocessing_single_df(df_single)
            
            unique_match_ids = df.match_id.unique()
            # unique_match_ids = [df.match_id.unique()[0]]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_match_id = {executor.submit(process_single_match, match_id): match_id for match_id in unique_match_ids}
                
                for future in tqdm(as_completed(future_to_match_id), total=len(future_to_match_id)):
                    match_id = future_to_match_id[future]
                    try:
                        df_single = future.result()
                        out_df_list.append(df_single)
                    except Exception as e:
                        print(f'Exception for match_id {match_id}: {e}')
            
            df = pd.concat(out_df_list)
            df = df.reset_index(drop=True)
            df['index_column'] = df.index
            df = df.sort_values(by=['match_id', "index_column"])
            df = df.drop(columns=['index_column'])
        else:
            raise ValueError('Preprocessing method not found')
        print(f'Preprocessed data from {self.data_provider} with method {self.preprocess_method}')
        return df
    
if __name__ == '__main__':
    datafactory_path=os.getcwd()+"/test/sports/event_data/data/datafactory/datafactory_events.json"
    metrica_event_json_path=os.getcwd()+"/test/sports/event_data/data/metrica/metrica_events.json"
    metrica_event_csv_path=os.getcwd()+"/test/sports/event_data/data/metrica/Sample_Game_1/Sample_Game_1_RawEventsData.csv"
    metrica_tracking_home_path=os.getcwd()+"/test/sports/event_data/data/metrica/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
    metrica_tracking_away_path=os.getcwd()+"/test/sports/event_data/data/metrica/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"
    opta_f7_path=os.getcwd()+"/test/sports/event_data/data/opta/opta_f7.xml"
    opta_f24_path=os.getcwd()+"/test/sports/event_data/data/opta/opta_f24.xml"
    robocup_2d_event_path=os.getcwd()+"/test/sports/event_data/data/robocup_2d/202307091024-HELIOS2023_1-vs-CYRUS_0-pass.csv"
    robocup_2d_tracking_path=os.getcwd()+"/test/sports/event_data/data/robocup_2d/202307091024-HELIOS2023_1-vs-CYRUS_0.csv"
    sportec_event_path=os.getcwd()+"/test/sports/event_data/data/sportec/sportec_events.xml"
    sportec_tracking_path=os.getcwd()+"/test/sports/event_data/data/sportec/sportec_positional.xml"
    sportec_meta_path=os.getcwd()+"/test/sports/event_data/data/sportec/sportec_meta.xml"
    statsbomb_event_path=os.getcwd()+"/test/sports/event_data/data/statsbomb/events/3805010.json"
    statsbomb_360_path=os.getcwd()+"/test/sports/event_data/data/statsbomb/three-sixty/3805010.json"
    statsbomb_api_path=os.getcwd()+"/test/sports/event_data/data/statsbomb/api.json"
    statsbomb_skillcorner_event_path="/data_pool_1/laliga_23/statsbomb/events"
    statsbomb_skillcorner_tracking_path="/data_pool_1/laliga_23/skillcorner/tracking"
    statsbomb_skillcorner_match_path="/data_pool_1/laliga_23/skillcorner/match"
    wyscout_event_path=os.getcwd()+"/test/sports/event_data/data/wyscout/events_England.json"
    datastadium_event_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/play.csv"
    datastadium_tracking_home_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/home_tracking.csv"
    datastadium_tracking_away_path=os.getcwd()+"/test/sports/event_data/data/datastadium/2019022307/away_tracking.csv"
    datastadium_dir="/work2/fujii/JLeagueData/Data_2019FM"

    #test single file

    #test load_datafactory
    # datafactory_df=Event_data(data_provider='datafactory',event_path=datafactory_path).load_data()
    # datafactory_df.to_csv(os.getcwd()+"/test/sports/event_data/data/datafactory/test_data_main.csv",index=False)

    #test load_metrica
    # metrica_df=Event_data(data_provider='metrica',event_path=metrica_event_csv_path,match_id=1,
    #                       tracking_home_path=metrica_tracking_home_path,tracking_away_path=metrica_tracking_away_path).load_data()
    # metrica_df.to_csv(os.getcwd()+"/test/sports/event_data/data/metrica/test_data_csv_main.csv",index=False)
    # metrica_df=Event_data(data_provider='metrica',event_path=metrica_event_json_path,match_id=1).load_data()
    # metrica_df.to_csv(os.getcwd()+"/test/sports/event_data/data/metrica/test_data_json_main.csv",index=False)

    #test load_opta_xml
    # opta_df=Event_data(data_provider='opta',event_path=opta_f24_path,match_id=1).load_data()
    # opta_df.to_csv(os.getcwd()+"/test/sports/event_data/data/opta/test_data_main.csv",index=False)

    #test load_robocup_2d
    # robocup_2d_df=Event_data(data_provider='robocup_2d',event_path=robocup_2d_event_path,match_id=1,tracking_path=robocup_2d_tracking_path).load_data()
    # robocup_2d_df.to_csv(os.getcwd()+"/test/sports/event_data/data/robocup_2d/test_data_main.csv",index=False)

    #test load_sportec
    # sportec_df = Event_data(data_provider='sportec', event_path=sportec_event_path, tracking_path=sportec_tracking_path, meta_data=sportec_meta_path).load_data()
    # sportec_df.to_csv(os.getcwd()+"/test/sports/event_data/data/sportec/test_data_main.csv",index=False)

    #test load_statsbomb with json file
    # statsbomb_df=Event_data(data_provider='statsbomb',event_path=statsbomb_event_path,sb360_path=statsbomb_360_path).load_data()
    # statsbomb_df.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_data_main.csv",index=False)

    # test load_statsbomb with api data
    # statsbomb_df=Event_data(data_provider='statsbomb',statsbomb_match_id=3795108).load_data()
    # statsbomb_df.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_api_data_main.csv",index=False)

    #test load_statsbomb_skillcorner
    # statsbomb_skillcorner_df=Event_data(data_provider='statsbomb_skillcorner',
    #                                     statsbomb_event_dir=statsbomb_skillcorner_event_path,
    #                                     skillcorner_tracking_dir=statsbomb_skillcorner_tracking_path,
    #                                     skillcorner_match_dir=statsbomb_skillcorner_match_path,
    #                                     statsbomb_match_id=3894907,
    #                                     skillcorner_match_id=1553748
    #                                     ).load_data()
    # statsbomb_skillcorner_df.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb_skillcorner/test_data_main.csv",index=False)

    #test load_wyscout
    # wyscout_df=Event_data(data_provider='wyscout',event_path=wyscout_event_path).load_data()
    # wyscout_df.head(1000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_data_main.csv",index=False)

    # test load_datastadium
    # datastadium_df=Event_data(data_provider='datastadium',event_path=datastadium_event_path,
    #                           tracking_home_path=datastadium_tracking_home_path,tracking_away_path=datastadium_tracking_away_path).load_data()
    # datastadium_df.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/load_class_single.csv",index=False)



    #test preprocessing
    # seq2event
    # wyscout_df=Event_data(data_provider='wyscout',event_path=wyscout_event_path,preprocess_method="SEQ2EVENT",max_workers=10).preprocessing()
    # wyscout_df.head(1000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_seq2event_main.csv",index=False)

    #test nmstpp
    # wyscout_df=Event_data(data_provider='wyscout',event_path=wyscout_event_path,preprocess_method="NMSTPP",max_workers=10).preprocessing()
    # wyscout_df.head(1000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_nmstpp_main.csv",index=False)

    #test lem
    # wyscout_df=Event_data(data_provider='wyscout',event_path=wyscout_event_path,preprocess_method="LEM",max_workers=10).preprocessing()
    # wyscout_df.head(1000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_lem_main.csv",index=False)

    #test UIED wyscout
    # df_wyscout=Event_data(data_provider='wyscout',event_path=wyscout_event_path,preprocess_method="UIED",max_workers=10).preprocessing()
    # df_wyscout.head(1000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_wyscout_UIED_main.csv",index=False)

    #test UIED statsbomb_skillcorner
    # df_statsbomb_skillcorner=Event_data(data_provider='statsbomb_skillcorner',
    #                                     statsbomb_event_dir=statsbomb_skillcorner_event_path,
    #                                     skillcorner_tracking_dir=statsbomb_skillcorner_tracking_path,
    #                                     skillcorner_match_dir=statsbomb_skillcorner_match_path,
    #                                     statsbomb_match_id=3894907,
    #                                     skillcorner_match_id=1553748,
    #                                     preprocess_method="UIED",
    #                                     max_workers=10).preprocessing()
    # df_statsbomb_skillcorner.head(1000).to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb_skillcorner/test_preprocess_statsbomb_skillcorner_UIED_main.csv",index=False)

    #test UIED statsbomb_json
    # df_statsbomb_json=Event_data(data_provider='statsbomb',event_path=statsbomb_event_path,sb360_path=statsbomb_360_path,preprocess_method="UIED").preprocessing()
    # df_statsbomb_json.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_preprocess_statsbomb_json_UIED_main.csv",index=False)

    #test UIED statsbomb_api
    # df_statsbomb_api=Event_data(data_provider='statsbomb',statsbomb_match_id=3795108,preprocess_method="UIED").preprocessing()
    # df_statsbomb_api.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_preprocess_statsbomb_api_UIED_main.csv",index=False)

    #test UIED datastadium
    # df_datastadium=Event_data(data_provider='datastadium',event_path=datastadium_event_path,
    #                           tracking_home_path=datastadium_tracking_home_path,tracking_away_path=datastadium_tracking_away_path,
    #                           preprocess_method="UIED").preprocessing()
    # df_datastadium.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/preprocess_UIED_class_single.csv",index=False)









    # multiple files
    # statsbomb_df=Event_data(data_provider='statsbomb',statsbomb_match_id=[3788742,3788741]).load_data()
    # statsbomb_df.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_api_data_main_multi.csv",index=False)

    #test load_statsbomb_skillcorner
    # statsbomb_skillcorner_df=Event_data(data_provider='statsbomb_skillcorner',
    #                                     statsbomb_event_dir=statsbomb_skillcorner_event_path,
    #                                     skillcorner_tracking_dir=statsbomb_skillcorner_tracking_path,
    #                                     skillcorner_match_dir=statsbomb_skillcorner_match_path,
    #                                     match_id_df=os.getcwd()+'/preprocessing/example/id_matching.csv',
    #                                     max_workers=10).load_data()
    # statsbomb_skillcorner_df.head(10000).to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb_skillcorner/test_data_main_multi.csv",index=False)

    #test load_statsbomb_json
    # multi_event_path="/data_pool_1/statsbomb_2023/events_and_frames/data/events"
    # multi_360_path="/data_pool_1/statsbomb_2023/events_and_frames/data/360-frames"

    # statsbomb_df=Event_data(data_provider='statsbomb',event_path=multi_event_path,sb360_path=multi_360_path,max_workers=10).load_data()
    # statsbomb_df.head(10000).to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_data_main_multi.csv",index=False)

    #test load_wyscout
    # wyscout_event_path="/home/c_yeung/workspace6/python/openstarlab/PreProcessing/test/sports/event_data/data/wyscout/event"
    # wyscout_matches_path="/home/c_yeung/workspace6/python/openstarlab/PreProcessing/test/sports/event_data/data/wyscout/matches"
    # wyscout_df=Event_data(data_provider='wyscout',
    #                       event_path=wyscout_event_path,
    #                       wyscout_matches_path=wyscout_matches_path,
    #                       max_workers=10).load_data()
    # wyscout_df.head(10000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_data_main_multi.csv",index=False)

    #test load_datastadium multiple files
    # datastadium_df=Event_data(data_provider='datastadium',event_path=datastadium_dir,max_workers=10).load_data()
    # datastadium_df.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/load_class_multi.csv",index=False)

    #test preprocessing multi files
    # wyscout_event_path="/home/c_yeung/workspace6/python/openstarlab/PreProcessing/test/sports/event_data/data/wyscout/event"
    # wyscout_matches_path="/home/c_yeung/workspace6/python/openstarlab/PreProcessing/test/sports/event_data/data/wyscout/matches"
    # statsbomb_df=Event_data(data_provider='statsbomb',statsbomb_match_id=[3788742,3788741]).load_data()
    # statsbomb_df.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_api_data_main_multi.csv",index=False)
    #seq2event
    # wyscout_df=Event_data(data_provider='wyscout',event_path=wyscout_event_path,wyscout_matches_path=wyscout_matches_path,
    #                       preprocess_method="SEQ2EVENT",max_workers=10).preprocessing()
    # wyscout_df.head(10000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_seq2event_main_multi.csv",index=False)

    #nmstpp
    # wyscout_df=Event_data(data_provider='wyscout',event_path=wyscout_event_path,wyscout_matches_path=wyscout_matches_path,
    #                       preprocess_method="NMSTPP",max_workers=10).preprocessing()
    # wyscout_df.head(10000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_seq2event_main_multi.csv",index=False)

    #lem
    # wyscout_df=Event_data(data_provider='wyscout',event_path=wyscout_event_path,wyscout_matches_path=wyscout_matches_path,
    #                       preprocess_method="LEM",max_workers=10).preprocessing()
    # wyscout_df.head(10000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_seq2event_main_multi.csv",index=False)

    #UIED wyscout
    # wyscout_df=Event_data(data_provider='wyscout',event_path=wyscout_event_path,wyscout_matches_path=wyscout_matches_path,
    #                       preprocess_method="UIED",max_workers=10).preprocessing()
    # wyscout_df.head(10000).to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_seq2event_main_multi.csv",index=False)

    #UIED statsbomb_skillcorner
    # statsbomb_skillcorner_df=Event_data(data_provider='statsbomb_skillcorner',
    #                                     statsbomb_event_dir=statsbomb_skillcorner_event_path,
    #                                     skillcorner_tracking_dir=statsbomb_skillcorner_tracking_path,
    #                                     skillcorner_match_dir=statsbomb_skillcorner_match_path,
    #                                     match_id_df=os.getcwd()+'/preprocessing/example/id_matching.csv',
    #                                     preprocess_method="UIED",
    #                                     ).preprocessing()
    # statsbomb_skillcorner_df.head(1000).to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb_skillcorner/test_preprocess_statsbomb_skillcorner_UIED_main_multi.csv",index=False)

    #UIED statsbomb_json
    # multi_event_path="/data_pool_1/statsbomb_2023/events_and_frames/data/events"
    # multi_360_path="/data_pool_1/statsbomb_2023/events_and_frames/data/360-frames"

    # statsbomb_df=Event_data(data_provider='statsbomb',event_path=multi_event_path,sb360_path=multi_360_path,preprocess_method="UIED",max_workers=10).preprocessing()
    # statsbomb_df.head(10000).to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_preprocess_statsbomb_json_UIED_main_multi.csv",index=False)

    #UIED statsbomb_api (could not test due to Max retries exceeded)

    #test UIED datastadium multiple files
    # df_datastadium=Event_data(data_provider='datastadium',event_path=datastadium_dir,preprocess_method="UIED",max_workers=10).preprocessing()
    # df_datastadium.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/preprocess_UIED_class_multi.csv",index=False)


    print("-----------done-----------")

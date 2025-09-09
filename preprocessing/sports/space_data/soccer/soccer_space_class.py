import os
import json
import bz2
import pandas as pd
from tqdm import tqdm


class Soccer_space_data:
    def __init__(self, data_provider, event_data_path, tracking_data_path, out_path=None, testing_mode=False):
        self.data_provider = data_provider
        self.event_data_path = event_data_path
        self.tracking_data_path = tracking_data_path
        self.testing_mode = testing_mode
        self.out_path = out_path

    def load_event_json(self, event_json_path):
        with open(event_json_path, 'r') as f:
            event_data = json.load(f)
        event_data = pd.json_normalize(event_data, sep='_')
        return event_data
    
    def load_tracking_bz2(self,tracking_bz2_path):
        tracking_data=[]
        with bz2.open(tracking_bz2_path, 'rt') as f:
            for line in f:
                tracking_data.append(json.loads(line))
        tracking_df = pd.json_normalize(tracking_data, sep='_')
        return tracking_df

    def get_files(self):
        if os.path.isdir(self.event_data_path):
            event_files = [os.path.join(self.event_data_path, f) for f in os.listdir(self.event_data_path) if f.endswith('.json')]
        elif os.path.isfile(self.event_data_path) and self.event_data_path.endswith('.json'):
            event_files = [self.event_data_path]
        else:
            raise ValueError(f'Invalid event data path: {self.event_data_path}')

        if os.path.isdir(self.tracking_data_path):
            tracking_files = [os.path.join(self.tracking_data_path, f) for f in os.listdir(self.tracking_data_path) if f.endswith('.bz2')]
        elif os.path.isfile(self.tracking_data_path) and self.tracking_data_path.endswith('.bz2'):
            tracking_files = [self.tracking_data_path]
        else:
            raise ValueError(f'Invalid tracking data path: {self.tracking_data_path}')

        return event_files, tracking_files

    def preprocessing(self):

        event_files, tracking_files = self.get_files()
        if self.testing_mode:
            event_files, tracking_files = event_files[:2], tracking_files[:2]
            print("Running in testing mode. Limited files will be processed.")

        from .soccer_space_preprocessing import convert_pff2metrica, convert_tracking_data_fixed_ids

        home_tracking_dict={}
        away_tracking_dict={}
        period_2_dict={}
        for tracking_path_i in tqdm(tracking_files, total=len(tracking_files), desc='Processing tracking files'):
            match_i = os.path.splitext(os.path.splitext(os.path.basename(tracking_path_i))[0])[0]
            match_tracking_df = self.load_tracking_bz2(tracking_path_i)
            home_tracking, away_tracking, first_period_2_index, first_period_2_time = convert_tracking_data_fixed_ids(match_tracking_df)
            home_tracking_dict[match_i] = home_tracking
            away_tracking_dict[match_i] = away_tracking
            period_2_dict[match_i] = (first_period_2_index, first_period_2_time)

        event_data_dict={}
        for event_path_i in tqdm(event_files, total=len(event_files), desc='Processing event files'):
            match_i = os.path.splitext(os.path.basename(event_path_i))[0]
            match_event_df = self.load_event_json(event_path_i)
            Metrica_df = convert_pff2metrica(match_event_df, period_2_dict.get(match_i, (None, None)))
            event_data_dict[match_i] = Metrica_df



        if self.out_path:
            #create output directory if not exists
            os.makedirs(self.out_path+'/event', exist_ok=True)
            os.makedirs(self.out_path+'/home_tracking', exist_ok=True)
            os.makedirs(self.out_path+'/away_tracking', exist_ok=True)
            
            for match_id, df in event_data_dict.items():
                df.to_csv(os.path.join(self.out_path,'event', f'event_data_{match_id}.csv'), index=False)
            for match_id, df in home_tracking_dict.items():
                df.to_csv(os.path.join(self.out_path, 'home_tracking', f'home_tracking_{match_id}.csv'), index=False)
            for match_id, df in away_tracking_dict.items():
                df.to_csv(os.path.join(self.out_path, 'away_tracking', f'away_tracking_{match_id}.csv'), index=False)

        return event_data_dict, home_tracking_dict, away_tracking_dict

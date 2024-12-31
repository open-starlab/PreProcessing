import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == '__main__':
    import rocket_league_load_data
    import rocket_league_processing
else:
    from . import rocket_league_load_data
    from . import rocket_league_processing
import pdb

class Rocket_league_event_data:
    """
    A class to handle Rocket League event data.

    This class provides methods to load and preprocess Rocket League event data.

    Attributes:
        data_provider (str): The data provider for Rocket League event data.
        replay_path (str): Path to the replay data file.

    Methods:
        load_data(): Loads the Rocket League event data.
        preprocessing(): Preprocesses the loaded Rocket League event data.
    """

    def __init__(self, data_provider, replay_path=None):
        self.data_provider = data_provider
        self.replay_path = replay_path

    def load_data_single_file(self):
        """
        Loads data from a single replay file.

        Args:
            replay_file (str): Path to the replay file.

        Returns:
            pd.DataFrame: Loaded Rocket League event data.
        """
        if self.data_provider == 'carball':
            df = rocket_league_load_data.load_with_carball(self.replay_path)
        else:
            raise ValueError('Unsupported data provider.')
        return df

    def load_data(self):
        """
        Loads the Rocket League event data from the specified path.

        Returns:
            pd.DataFrame: Loaded Rocket League event data.
        """
        print(f'Loading data from {self.data_provider}')
        if os.path.isfile(self.replay_path) and self.replay_path.endswith('.replay') and self.data_provider == 'carball':
            df = self.load_data_single_file()
        else:
            raise ValueError('Event path is not a valid file or directory')
        print(f'Loaded data from {self.data_provider}')
        return df

    def preprocessing_single_df(self, df):
        """
        Preprocesses a single DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to preprocess.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """ 
        df = rocket_league_processing.UIED_rocket_league(df)
        return df

    def preprocessing(self):
        """
        Preprocesses the loaded Rocket League event data.

        Returns:
            pd.DataFrame: Preprocessed Rocket League event data.
        """
        print(f'Preprocessing data from {self.data_provider}')
        df = self.load_data()
        out_df_list = []

        unique_match_ids = df['match_id'].unique()

        def process_single_match(match_id):
            df_single = df[df.match_id == match_id]
            return self.preprocess_single_df(df_single)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_match_id = {executor.submit(process_single_match, match_id): match_id for match_id in unique_match_ids}
            
            for future in tqdm(as_completed(future_to_match_id), total=len(future_to_match_id)):
                match_id = future_to_match_id[future]
                try:
                    df_single = future.result()
                    out_df_list.append(df_single)
                except Exception as e:
                    print(f'An exception occurred while processing match_id {match_id}: {e}')
        
        df_processed = pd.concat(out_df_list, ignore_index=True)
        df_processed['index_column'] = df_processed.index
        df_processed = df_processed.sort_values(by=['match_id', "index_column"]).drop(columns=['index_column']).reset_index(drop=True)

        print(f'Preprocessed data from {self.data_provider}.')
        return df_processed
    
if __name__ == '__main__':
    # check if the Rocket_league_event_data class is correctly implemented
    rocket_league_replay_path=os.getcwd()+"test/sports/event_data/data/rocket_league/0328fc07-13fb-4cb6-9a86-7d608196ddbd.replay"
    rocket_league_df=Event_data(data_provider='carball',replay_path=rocket_league_replay_path).load_data()
    rocket_league_df.to_csv(os.getcwd()+"/test/sports/event_data/data/rocket_league/test_data_main.csv",index=False)

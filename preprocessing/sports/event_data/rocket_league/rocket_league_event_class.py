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
        event_path (str): Path to the event data file.
        match_id (str): ID of the match.
        preprocess_method (str): Method to use for preprocessing the data.

    Methods:
        load_data(): Loads the Rocket League event data.
        preprocessing(): Preprocesses the loaded Rocket League event data.
    """

    def __init__(self, data_provider, event_path=None, match_id=None, preprocess_method=None):
        self.data_provider = data_provider
        self.event_path = event_path
        self.match_id = match_id
        self.preprocess_method = preprocess_method

    def load_data(self):
        """
        Loads the Rocket League event data from the specified path.

        Returns:
            pd.DataFrame: Loaded Rocket League event data.
        """
        print(f'Loading data from {self.data_provider}')
        df = rocket_league_load_data.load_rocket_league(self.event_path, self.match_id)
        print(f'Loaded data from {self.data_provider}')
        return df

        
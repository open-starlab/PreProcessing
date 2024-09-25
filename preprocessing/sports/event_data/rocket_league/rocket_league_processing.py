import os
import pandas as pd
import numpy as np

def UIED_rocket_league(data):
    """
    Processes Rocket League match event data to determine possession, filter actions, 
    compute additional metrics, and normalize data.

    Parameters:
    data (pd.DataFrame or str): A pandas DataFrame containing event data or a file path to a replay file.

    Returns:
    pd.DataFrame: A processed DataFrame with simplified and normalized event actions.
    """
    # Load data from DataFrame or file path
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, str):
        if os.path.exists(data):
            df = pd.read_csv(data)
        else:
            raise FileNotFoundError("The file path does not exist")
    else:
        raise ValueError("The data must be a pandas DataFrame or a file path")
    
    df = df.copy()

    # ToDo: Implement UIED for Rocket League

    return df

if __name__ == "__main__":
    import pdb
    import os
    #cd to ../PreProcessing
    rocket_league_path=os.getcwd()+"/test/sports/event_data/data/rocket_league/test_data.csv"

    #test load_with_carball
    rocket_league_df=UIED_rocket_league(rocket_league_path)
    rocket_league_df.to_csv(os.getcwd()+"/test/sports/event_data/data/rocket_league/preprocess_UIED.csv",index=False)

import os
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import os

def UIED_rocket_league(data):
    """
    Processes Rocket League match event data to determine possession, filter actions, 
    compute additional metrics, and normalize data.

    Parameters:
    data (pd.DataFrame or str): A pandas DataFrame containing event data or a file path to a CSV file.

    Returns:
    pd.DataFrame: A processed DataFrame with simplified and normalized event actions.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, str):
        if os.path.exists(data):
            df = pd.read_csv(data)
        else:
            raise FileNotFoundError("The file path does not exist")
    else:
        raise ValueError("The data must be a pandas DataFrame or a file path")

    # アクション列の作成
    # ToDo: アクションの種類を増やす
    df['action'] = np.where(df['pass'], 'pass', 
                    np.where(df['shot'], 'shot', 
                    np.where(df['dribble'], 'dribble', '_')))

    # イベント関連の特徴量生成
    # ToDo_の中にも意図的なクリアなども含まれるのでそれをどう扱うか
    df['success'] = np.where(df['action'] != '_', 1, 0)
    df['goal'] = df["goal"].astype(int)

    # スコアの計算

    # 時間関連の特徴量生成
    MATCH_LENGTH = 300
    df["game_seconds_remaining"] = df["game_seconds_remaining"].fillna(0)
    df['game_is_overtime'] = df['game_is_overtime'].fillna(False)

    df['seconds'] = np.where(
        df['game_is_overtime'],
        MATCH_LENGTH + df['game_seconds_remaining'],
        MATCH_LENGTH - df['game_seconds_remaining']
    )   # ToDo: 小数点はないが良いか。残り0秒になってからボールが地面につくまで時間が進まないがどうするか
    df['Minute'] = df['seconds'] // 60
    df['Second'] = df['seconds'] % 60
    df['delta_T'] = df['game_time'].diff().fillna(0)    # ToDo: kickoffの時間をどう扱うか
    df['Period'] = None

    # 位置関連の特徴量生成
    FIELD_X = 4096    # Side wall (x = ±4096)
    FIELD_Y = 5120    # Back wall (y = ±5120)
    FIELD_Z = 2044    # Ceiling height
    GOAL_Y = 5120     # Goal position
    GOAL_Z = 642.775  # Goal height
    
    df["start_x"] = df["ball_pos_x"] / FIELD_X     # -1 to 1 (side to side)
    df["start_y"] = df["ball_pos_y"] / FIELD_Y     # -1 to 1 (end to end)
    df["start_z"] = df["ball_pos_z"] / FIELD_Z     # 0 to 1 (floor to ceiling)
    
    df["deltaX"] = df["start_x"].diff().fillna(0)
    df["deltaY"] = df["start_y"].diff().fillna(0)
    df["deltaZ"] = df["start_z"].diff().fillna(0)
    
    df["distance"] = np.sqrt(
        df["deltaX"]**2 + 
        df["deltaY"]**2 + 
        df["deltaZ"]**2
    )
    
    df["dist2goal"] = np.where(
        df["home_team"],
        np.sqrt(
            df["start_x"]**2 + 
            (df["start_y"] - 1)**2
        ),
        np.sqrt(
            df["start_x"]**2 + 
            (df["start_y"] + 1)**2
        )
    )
    
    df["angle2goal"] = np.where(
        df["home_team"],
        np.arctan2(df["start_x"], 1 - df["start_y"]),
        np.arctan2(df["start_x"], -(df["start_y"] + 1))
    )
    
    # 不要な列の削除
    columns_to_keep = ['match_id', 'poss_id', 'team', 'home_team', 'action', 'success', 'goal', 'home_score', 'away_score', 'goal_diff', 'Period', 'Minute', 'Second', 'seconds', "delta_T", 'start_x', 'start_y', 'deltaX', 'deltaY', 'distance', 'dist2goal', 'angle2goal']
    
    df = df[columns_to_keep]

    return df

if __name__ == "__main__":
    import pdb
    import os
    #cd to ../PreProcessing
    rocket_league_path=os.getcwd()+"/test/sports/event_data/data/rocket_league/test_data2.csv"

    #test load_with_carball
    rocket_league_df=UIED_rocket_league(rocket_league_path)
    rocket_league_df.to_csv(os.getcwd()+"/test/sports/event_data/data/rocket_league/preprocess_UIED.csv",index=False)

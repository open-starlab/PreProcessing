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
    # データの読み込み
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
    # _としているものをどう扱うか
    df['action'] = np.where(df['pass'], 'pass', 
                    np.where(df['shot'], 'shot', 
                    np.where(df['dribble'], 'dribble', '_')))

    # イベント関連の特徴量生成
    # _の中にも意図的なクリアなども含まれるのでそれをどう扱うか
    df['success'] = np.where(df['action'] != '_', 1, 0)
    df['goal'] = df["goal"].astype(int)
    df['home_team'] = None

    # スコアの計算
    # ToDo: load_data側でスコアを計算しておく
    df['home_score'] = 0
    df['away_score'] = 0
    df['goal_diff'] = df['home_score'] - df['away_score']

    # 時間関連の特徴量生成
    # ToDo: load_data側で時間を計算しておく
    df['Minute'] = 0
    df['Second'] = 0
    df['seconds'] = 0
    df['delta_T'] = 0

    # 位置関連の特徴量生成
    # ToDo: load_data側でballの座標を読み込んでおく
    df["start_x"] = 0
    df["start_y"] = 0
    df["start_z"] = 0
    df["deltaX"] = 0
    df["deltaY"] = 0
    df["deltaZ"] = 0
    df["distance"] = 0
    # チームの色（orange/blue）に応じたゴールに対して計算する
    df["dist2goal"] = 0
    df["angle2goal"] = 0

    # ToDo: tracking_dataをload_data側で読み込んでおく

    # 不要な列の削除
    columns_to_keep = ['match_id', 'poss_id', 'team', 'home_team', 'action', 'success', 'goal', 'home_score', 'away_score', 'goal_diff', 'Minute', 'Second', 'seconds', "delta_T", 'start_x', 'start_y', 'deltaX', 'deltaY', 'distance', 'dist2goal', 'angle2goal']
    
    df = df[columns_to_keep]

    return df

if __name__ == "__main__":
    import pdb
    import os
    #cd to ../PreProcessing
    rocket_league_path=os.getcwd()+"/test/sports/event_data/data/rocket_league/test_data.csv"

    #test load_with_carball
    rocket_league_df=UIED_rocket_league(rocket_league_path)
    rocket_league_df.to_csv(os.getcwd()+"/test/sports/event_data/data/rocket_league/preprocess_UIED.csv",index=False)

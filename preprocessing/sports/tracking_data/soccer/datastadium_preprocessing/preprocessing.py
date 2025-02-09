import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import os
from . import preprocess_config as config 
import pdb

def process_tracking_data(game_id, data_path ,event_data_name = "play.csv", player_data_name = "player.csv", tracking_data_name1 = "tracking_1stHalf.csv", tracking_data_name2 = "tracking_2ndHalf.csv",test=False):
    """
    Processes tracking and event data for a specified game.

    Parameters:
        game_id (int): Index of the game folder in the dataset.
        data_path (str): Path to the folder containing game data.
        event_data_name (str): Name of the event data file. Default is "play.csv".
        player_data_name (str): Name of the player data file. Default is "player.csv".
        tracking_data_name1 (str): Name of the first-half tracking data file. Default is "tracking_1stHalf.csv".
        tracking_data_name2 (str): Name of the second-half tracking data file. Default is "tracking_2ndHalf.csv".

    Returns:
        tuple: (tracking_home, tracking_away, jerseynum_df)
            - tracking_home (pd.DataFrame): Processed tracking data for the home team.
            - tracking_away (pd.DataFrame): Processed tracking data for the away team.
            - jerseynum_df (pd.DataFrame): Dataframe containing jersey numbers of players.
    """
    warnings.simplefilter("ignore")

    # Set folder and file names
    jdata_fm = data_path

    game_date = os.listdir(jdata_fm)
    #sort the game_date
    game_date.sort()

    # Load data
    try:
        event_data = pd.read_csv(
            os.path.join(jdata_fm, game_date[game_id], event_data_name), encoding="shift_jis"
        )
        player_data = pd.read_csv(
            os.path.join(jdata_fm, game_date[game_id], player_data_name), encoding="shift_jis"
        )
    except:
        pass
    tracking_data1 = pd.read_csv(
        os.path.join(jdata_fm, game_date[game_id], tracking_data_name1), encoding="shift_jis"
    )
    tracking_data2 = pd.read_csv(
        os.path.join(jdata_fm, game_date[game_id], tracking_data_name2), encoding="shift_jis"
    )
    if test:
        tracking_data1 = tracking_data1.head(1000)
        tracking_data2 = tracking_data2.head(1000)  

    # Define column templates
    tracking_home_columns = [
        "Period", "Time [s]"
    ] + [f"Home_{i}_{coord}" for i in range(1, 15) for coord in ["x", "y"]] + ["ball_x", "ball_y"]

    tracking_away_columns = [
        "Period", "Time [s]"
    ] + [f"Away_{i}_{coord}" for i in range(1, 15) for coord in ["x", "y"]] + ["ball_x", "ball_y"]

    tracking_home = pd.DataFrame(columns=tracking_home_columns)
    tracking_away = pd.DataFrame(columns=tracking_away_columns)

    # Split tracking data
    ball_track1 = tracking_data1[tracking_data1["ホームアウェイF"] == 0]
    ball_track2 = tracking_data2[tracking_data2["ホームアウェイF"] == 0]
    home_track1 = tracking_data1[tracking_data1["ホームアウェイF"] == 1]
    home_track2 = tracking_data2[tracking_data2["ホームアウェイF"] == 1]
    away_track1 = tracking_data1[tracking_data1["ホームアウェイF"] == 2]
    away_track2 = tracking_data2[tracking_data2["ホームアウェイF"] == 2]

    # Get player jersey numbers
    def get_jersey_numbers(track1, track2):
        jersey_numbers = list(track1["背番号"].unique()) + list(track2["背番号"].unique())
        jersey_numbers = pd.Series(jersey_numbers).unique()
        while len(jersey_numbers) < config.FOOTBALL_PLAYER_NUM + config.SUBSTITUTION_NUM:
            jersey_numbers = np.append(jersey_numbers, -1)
        return jersey_numbers

    home_jersey_numbers = get_jersey_numbers(home_track1, home_track2)
    away_jersey_numbers = get_jersey_numbers(away_track1, away_track2)

    jurseynum_df = pd.DataFrame(
        index=list(range(1, config.FOOTBALL_PLAYER_NUM + config.SUBSTITUTION_NUM + 1)),
        columns=["Home", "Away"]
    )
    jurseynum_df["Home"] = home_jersey_numbers
    jurseynum_df["Away"] = away_jersey_numbers

    # Calculate frame lengths and periods
    def calculate_frame_info(track):
        min_frame = min(track["フレーム番号"])
        max_frame = max(track["フレーム番号"])
        return min_frame, max_frame, max_frame - min_frame + 1

    min_frame1, max_frame1, frame_len1 = calculate_frame_info(home_track1)
    min_frame2, max_frame2, frame_len2 = calculate_frame_info(home_track2)

    period_labels = ([1] * frame_len1) + ([2] * frame_len2)
    time_labels = [i * 0.04 for i in range(frame_len1 + frame_len2)]

    tracking_home["Period"] = period_labels
    tracking_home["Time [s]"] = time_labels
    tracking_away["Period"] = period_labels
    tracking_away["Time [s]"] = time_labels

    # Fill tracking data
    # def fill_tracking_data(tracking, track_data, jersey_numbers, track_index, min_frame, max_frame, prefix):
    #     for frame in tqdm(range(min_frame, max_frame + 1)):
    #         ball_x = track_data["座標X"][track_data["フレーム番号"] == frame]
    #         ball_y = track_data["座標Y"][track_data["フレーム番号"] == frame]

    #         if ball_x.nunique() == 1 and ball_y.nunique() == 1:
    #             tracking["ball_x"].iloc[track_index] = ball_x.iloc[0] / 100
    #             tracking["ball_y"].iloc[track_index] = ball_y.iloc[0] / 100

    #         player_id = 1
    #         for num in jersey_numbers:
    #             if num == -1:
    #                 break
    #             player_x = track_data["座標X"][(track_data["背番号"] == num) & (track_data["フレーム番号"] == frame)]
    #             player_y = track_data["座標Y"][(track_data["背番号"] == num) & (track_data["フレーム番号"] == frame)]
    #             if player_x.nunique() == 1 and player_y.nunique() == 1:
    #                 tracking[f"{prefix}_{player_id}_x"].iloc[track_index] = player_x.iloc[0] / 100
    #                 tracking[f"{prefix}_{player_id}_y"].iloc[track_index] = player_y.iloc[0] / 100
    #             player_id += 1
    #         track_index += 1
    #     return track_index
    
    def fill_tracking_data_combined(tracking_home, tracking_away, data_home_1, data_home_2, data_away_1, data_away_2,
                                    jersey_home, jersey_away, min_frame1, max_frame1, min_frame2, max_frame2):
        track_index_home = 0
        track_index_away = 0

        # Combine all frames for both halves
        for half, (track_data_home, track_data_away) in enumerate([(data_home_1, data_away_1), (data_home_2, data_away_2)]):
            min_frame = min_frame1 if half == 0 else min_frame2
            max_frame = max_frame1 if half == 0 else max_frame2

            for frame in tqdm(range(min_frame, max_frame + 1)):
                # Ball coordinates
                for tracking, track_data, track_index, prefix, jersey_numbers in [
                    (tracking_home, track_data_home, track_index_home, "Home", jersey_home),
                    (tracking_away, track_data_away, track_index_away, "Away", jersey_away)
                ]:
                    ball_x = track_data["座標X"][track_data["フレーム番号"] == frame]
                    ball_y = track_data["座標Y"][track_data["フレーム番号"] == frame]

                    if ball_x.nunique() == 1 and ball_y.nunique() == 1:
                        tracking["ball_x"].iloc[track_index] = ball_x.iloc[0] / 100
                        tracking["ball_y"].iloc[track_index] = ball_y.iloc[0] / 100

                    # Player coordinates
                    player_id = 1
                    for num in jersey_numbers:
                        if num == -1:
                            break
                        player_x = track_data["座標X"][
                            (track_data["背番号"] == num) & (track_data["フレーム番号"] == frame)
                        ]
                        player_y = track_data["座標Y"][
                            (track_data["背番号"] == num) & (track_data["フレーム番号"] == frame)
                        ]
                        if player_x.nunique() == 1 and player_y.nunique() == 1:
                            tracking[f"{prefix}_{player_id}_x"].iloc[track_index] = player_x.iloc[0] / 100
                            tracking[f"{prefix}_{player_id}_y"].iloc[track_index] = player_y.iloc[0] / 100
                        player_id += 1

                    # Increment the track index for the current team
                    if prefix == "Home":
                        track_index_home += 1
                    else:
                        track_index_away += 1

    fill_tracking_data_combined(
        tracking_home, tracking_away,
        home_track1, home_track2, away_track1, away_track2,
        home_jersey_numbers, away_jersey_numbers,
        min_frame1, max_frame1, min_frame2, max_frame2
    )

    # track_index = 0
    # track_index = fill_tracking_data(tracking_home, home_track1, home_jersey_numbers, track_index, min_frame1, max_frame1, "Home")
    # track_index = fill_tracking_data(tracking_home, home_track2, home_jersey_numbers, track_index, min_frame2, max_frame2, "Home")
    # track_index = 0
    # track_index = fill_tracking_data(tracking_away, away_track1, away_jersey_numbers, track_index, min_frame1, max_frame1, "Away")
    # track_index = fill_tracking_data(tracking_away, away_track2, away_jersey_numbers, track_index, min_frame2, max_frame2, "Away")

    return tracking_home, tracking_away, jurseynum_df

if __name__ == "__main__":
    import os
    game_id = 0  # Replace with the game ID you want to process
    data_path = os.getcwd()+"/test/sports/event_data/data/datastadium/"

    # Call the function
    tracking_home, tracking_away, jerseynum_df = process_tracking_data(
        game_id,
        data_path
    )

    tracking_home.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/test_tracking_home.csv", index=False)
    tracking_away.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/test_tracking_away.csv", index=False)
    jerseynum_df.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/test_jerseynum.csv", index=False)
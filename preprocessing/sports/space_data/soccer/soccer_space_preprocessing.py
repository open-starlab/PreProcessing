import pandas as pd
import numpy as np
import pdb
import ast
import math
import re

def extract_player_xy(row):
    """
    Extracts the (x, y) coordinates of the player involved in a game event.

    Parameters
    ----------
    row : pd.Series
        A row from a DataFrame containing game event and player information. 
        Expected keys:
            - "gameEvents_homeTeam" (bool): True if home team, False if away team.
            - "homePlayers" (list|str): List or stringified list of home team players.
            - "awayPlayers" (list|str): List or stringified list of away team players.
            - "gameEvents_playerId" (int): ID of the player involved in the event.

    Returns
    -------
    pd.Series
        A Series with coordinates:
        - "start_x"
        - "start_y"
        - "end_x"
        - "end_y"
        If the player is not found, all values are None.
    """
    # choose player list
    if row["gameEvents_homeTeam"] is True:
        player_dict = row["homePlayers"]
    elif row["gameEvents_homeTeam"] is False:
        player_dict = row["awayPlayers"]
    else:
        return pd.Series([None, None, None, None], index=["start_x", "start_y", "end_x", "end_y"])
    
    # find target player
    player_dict = ast.literal_eval(player_dict) if type(player_dict) == str else player_dict
    target_player = next((d for d in player_dict if d["playerId"] == row["gameEvents_playerId"]), None)

    if target_player:
        return pd.Series(
            [target_player["x"], target_player["y"], target_player["x"], target_player["y"]],
            index=["start_x", "start_y", "end_x", "end_y"]
        )
    else:
        return pd.Series([None, None, None, None], index=["start_x", "start_y", "end_x", "end_y"])

def type_id2name(x):
    """
    Map event type codes to descriptive names.

    Parameters
    ----------
    x : str | int | float | None
        Event type code (e.g., 'PA', 'SH', 'FO', etc.)

    Returns
    -------
    str | None
        Descriptive event type name, or None if not mapped.
    """
    if x in ['PA']:
        x = "pass"
    elif x in ['CR']:
        x = "cross"
    # elif x == 2:
    #     x = "throw_in"
    # elif x == 5:
    #     x = "corner_crossed"
    # elif x == 7:
    #     x = "take_on"
    elif x in ['FO']:
        x = "foul"
    elif x in ['CH']:
        x = "tackle"
    # elif x == 10:
    #     x = "interception"
    elif x in ['SH']:
        x = "shot"
    elif x in ['CL']:
        x = "clearance"
    elif x in ['BC']:
        x = "dribble"
    # elif x == 22:
    #     x = "goalkick"
    elif x in ['IT', 'RE', 'TC']:
        x = "other"
    elif x is None or (isinstance(x, (float, int)) and math.isnan(x)):
        x = None
    else:
        print(f"Unmapped event type: {x}")
    return x

def convert_pff2metrica(event_df):
    """
    Convert PFF-style event data to Metrica format.

    Parameters
    ----------
    event_df : pd.DataFrame
        Event data from PFF dataset with columns like:
        - gameEvents_period
        - gameEvents_playerName
        - possessionEvents_receiverPlayerName
        - possessionEvents_possessionEventType
        - startTime, endTime, duration
        - gameEvents_homeTeam
        - various outcome types for success/failure

    Returns
    -------
    Metrica_df : pd.DataFrame
        DataFrame in Metrica format with columns:
        ['Team', 'Type', 'Subtype', 'Period', 'Start Frame', 'Start Time [s]',
         'End Frame', 'End Time [s]', 'From', 'To', 'Start X', 'Start Y', 'End X', 'End Y']
    """
    # set column name
    column_name = ['Team', 
          'Type',
          'Subtype',
          'Period',
          'Start Frame',
          'Start Time [s]',
          'End Frame',
          'End Time [s]',
          'From',
          'To',
          'Start X',
          'Start Y',
          'End X',
          'End Y']
    Metrica_df = pd.DataFrame(columns=column_name)
    Metrica_df['Period'] = event_df['gameEvents_period']
    event_df[["start_x", "start_y", "end_x", "end_y"]] = event_df.apply(extract_player_xy, axis=1)
    Metrica_df['Start X'] = event_df['start_x'] #- 52.5
    Metrica_df['Start Y'] = event_df['start_y'] #- 34
    Metrica_df['End X'] = event_df['end_x'] #- 52.5
    Metrica_df['End Y'] = event_df['end_y'] #- 34
    Metrica_df['From'] = event_df['gameEvents_playerName']
    Metrica_df['To'] = event_df['possessionEvents_receiverPlayerName']
    Metrica_df['Type'] = event_df['possessionEvents_possessionEventType']
    Metrica_df['Type'] = Metrica_df['Type'].apply(type_id2name)

    success_mask = (
        (event_df['possessionEvents_passOutcomeType'].isin(['C'])) |
        (event_df['possessionEvents_crossOutcomeType'].isin(['C'])) |
        (event_df['possessionEvents_shotOutcomeType'].isin(['G'])) |
        (~event_df['possessionEvents_clearanceOutcomeType'].isin(['B', 'D'])) |
        (event_df['possessionEvents_challengeOutcomeType'].isin(['B', 'C', 'M'])) |
        (event_df['possessionEvents_ballCarryOutcome'].isin(['R'])) |
        (event_df['possessionEvents_touchOutcomeType'].isin(['R']))
    )
    event_df['Subtype'] = success_mask.map({True: 'success', False: 'fail'})
    Metrica_df['Subtype'] = event_df['Subtype']

    fps = 29.97
    Metrica_df['Start Time [s]'] = (event_df['gameEvents_startGameClock']).round().astype(int)
    Metrica_df['Start Frame'] = ((event_df['startTime'] - event_df['startTime'][0]) * fps).round().astype(int)
    Metrica_df['End Time [s]'] = (event_df['duration'] + event_df['gameEvents_startGameClock']).round().astype(int)
    end_frame = ((event_df['endTime'] - event_df['startTime'][0]) * fps).round()
    Metrica_df['End Frame'] = end_frame.fillna(Metrica_df['Start Frame']).astype(int)
    Metrica_df['Team'] = np.where(event_df['gameEvents_homeTeam'] == True, 'Home',
                      np.where(event_df['gameEvents_homeTeam'] == False, 'Away', None))
    
    # find the last row of Period = 1 and first row of Period = 2
    last_end_p1 = Metrica_df.loc[Metrica_df['Period'] == 1, 'End Time [s]'].iloc[-1]
    first_start_p2 = Metrica_df.loc[Metrica_df['Period'] == 2, 'Start Time [s]'].iloc[0]

    # compute offset
    offset = first_start_p2 - last_end_p1 + 1

    # adjust times directly with np.where (vectorized)
    Metrica_df['Start Time [s]'] = np.where(
        Metrica_df['Period'] == 2,
        Metrica_df['Start Time [s]'] - offset,
        Metrica_df['Start Time [s]']
    )

    Metrica_df['End Time [s]'] = np.where(
        Metrica_df['Period'] == 2,
        Metrica_df['End Time [s]'] - offset,
        Metrica_df['End Time [s]']
)


    #drop rows where start_x or start_y is NaN
    Metrica_df = Metrica_df.dropna(subset=['Start X', 'Start Y'])
    Metrica_df = Metrica_df.reset_index(drop=True)

    return Metrica_df

def convert_tracking_data_fixed_ids(tracking_df):
    """
    Convert raw tracking data into fixed-format home and away tracking DataFrames with consistent player IDs.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Input tracking data containing:
        - 'homePlayers', 'awayPlayers', 'balls' columns (lists or string representations)
        - 'period', 'periodElapsedTime'

    Returns
    -------
    home_tracking : pd.DataFrame
        Home team tracking data with player coordinates, ball position, and time.

    away_tracking : pd.DataFrame
        Away team tracking data with player coordinates, ball position, and time.
    """
    # Convert string representation to lists/dicts
    for col in ['homePlayers', 'awayPlayers', 'balls']:
        tracking_df[col] = tracking_df[col].apply(
            lambda v: ast.literal_eval(v) if isinstance(v, str) else v
        )

    # Initialize home and away tracking
    home_tracking = tracking_df[['period', 'periodElapsedTime']].copy()
    away_tracking = tracking_df[['period', 'periodElapsedTime']].copy()
    home_tracking.columns = ['Period', 'Time [s]']
    away_tracking.columns = ['Period', 'Time [s]']

    # Extract unique jerseys and map to fixed IDs 1-16
    all_home_jerseys = sorted({p['jerseyNum'] for lst in tracking_df['homePlayers'] for p in lst})
    all_away_jerseys = sorted({p['jerseyNum'] for lst in tracking_df['awayPlayers'] for p in lst})
    home_jersey_to_id = {jersey: i+1 for i, jersey in enumerate(all_home_jerseys[:16])}
    away_jersey_to_id = {jersey: i+1 for i, jersey in enumerate(all_away_jerseys[:16])}

    # Helper to map players to fixed IDs 1-16
    def map_players(players_col, jersey_to_id, prefix):
        cols_x = [f'{prefix}_{i}_x' for i in range(1, 17)] 
        cols_y = [f'{prefix}_{i}_y' for i in range(1, 17)]
        cols_list = []
        for col_x_i, col_y_i in zip(cols_x, cols_y):
            cols_list.append(col_x_i)
            cols_list.append(col_y_i)
        cols = cols_list
        df = pd.DataFrame(index=players_col.index, columns=cols, dtype=float)

        for i in range(16):
            x_col = f'{prefix}_{i+1}_x'
            y_col = f'{prefix}_{i+1}_y'

            # Use list comprehension for faster vectorized extraction
            df[x_col] = [next((p['x'] for p in lst if jersey_to_id.get(p['jerseyNum']) == i+1), np.nan)
                         for lst in players_col]
            df[y_col] = [next((p['y'] for p in lst if jersey_to_id.get(p['jerseyNum']) == i+1), np.nan)
                         for lst in players_col]
        return df

    # Map home and away players
    home_tracking = pd.concat([home_tracking, map_players(tracking_df['homePlayers'], home_jersey_to_id, 'Home')],
                              axis=1)
    away_tracking = pd.concat([away_tracking, map_players(tracking_df['awayPlayers'], away_jersey_to_id, 'Away')],
                              axis=1)

    # Add ball positions (first ball in list)
    balls_df = pd.DataFrame(tracking_df['balls'].apply(lambda l: l[0] if l else {'x': np.nan, 'y': np.nan}).tolist())
    home_tracking['ball_x'] = balls_df['x']
    home_tracking['ball_y'] = balls_df['y']
    away_tracking['ball_x'] = balls_df['x']
    away_tracking['ball_y'] = balls_df['y']

    # For Time [s] at Period 2, add the max Time [s] of Period 1
    if not home_tracking[home_tracking['Period'] == 1].empty and not home_tracking[home_tracking['Period'] == 2].empty:
        max_time_period_1_home = home_tracking[home_tracking['Period'] == 1]['Time [s]'].max()
        max_time_period_1_away = away_tracking[away_tracking['Period'] == 1]['Time [s]'].max()
        max_time_period_1 = max(max_time_period_1_home, max_time_period_1_away)
        home_tracking.loc[home_tracking['Period'] == 2, 'Time [s]'] += max_time_period_1
        away_tracking.loc[away_tracking['Period'] == 2, 'Time [s]'] += max_time_period_1

    #fill nan value
    entry_home_df = home_tracking.loc[0].isnull()
    entry_away_df = away_tracking.loc[0].isnull()
    home_column = home_tracking.columns
    away_column = away_tracking.columns
    home_player_num = [s[:-2] for s in home_column if re.match('Home_\d*_x', s)]
    away_player_num = [s[:-2] for s in away_column if re.match('Away_\d*_x', s)]

    # replace nan 
    for player in home_player_num:
        if entry_home_df[player+'_x']:
            home_tracking[player+'_x'] = home_tracking[player+'_x'].fillna(method='ffill')
            home_tracking[player+'_y'] = home_tracking[player+'_y'].fillna(method='ffill')
        else:
            home_tracking[player+'_x'] = home_tracking[player+'_x'].fillna(method='bfill')
            home_tracking[player+'_y'] = home_tracking[player+'_y'].fillna(method='bfill')

    for player in away_player_num:
        if entry_away_df[player+'_x']:
            away_tracking[player+'_x'] = away_tracking[player+'_x'].fillna(method='ffill')
            away_tracking[player+'_y'] = away_tracking[player+'_y'].fillna(method='ffill')
        else:
            away_tracking[player+'_x'] = away_tracking[player+'_x'].fillna(method='bfill')
            away_tracking[player+'_y'] = away_tracking[player+'_y'].fillna(method='bfill')

    # data interpolation in ball position in tracking data
    home_tracking['ball_x'] = home_tracking['ball_x'].interpolate()
    home_tracking['ball_y'] = home_tracking['ball_y'].interpolate()
    away_tracking['ball_x'] = away_tracking['ball_x'].interpolate()
    away_tracking['ball_y'] = away_tracking['ball_y'].interpolate()

    # check nan ball position x and y in tracking data
    home_tracking['ball_x'] = home_tracking['ball_x'].fillna(method='bfill')
    home_tracking['ball_y'] = home_tracking['ball_y'].fillna(method='bfill')
    away_tracking['ball_x'] = away_tracking['ball_x'].fillna(method='bfill')
    away_tracking['ball_y'] = away_tracking['ball_y'].fillna(method='bfill')


    return home_tracking, away_tracking
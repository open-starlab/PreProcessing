import carball
import pandas as pd

def load_with_carball(replay_path: str) -> pd.DataFrame:
    """
    Loads a Rocket League replay file and converts it into a DataFrame containing event and tracking data.

    Args:
        replay_path (str): Path to the Rocket League replay file.

    Returns:
        pd.DataFrame: DataFrame containing:
            - Event data (hits, goals, assists, etc.)
            - Tracking data (ball and player positions, velocities, etc.)
            - Basic match information (match_id, scores, etc.)
            - Detailed player information (position, velocity, boost amount, etc.)
    """
    # Analyze the replay file using carball
    analysis_manager = carball.analyze_replay_file(replay_path)
    
    # Get the protobuf data (event data)
    proto_game = analysis_manager.get_protobuf_data()
    
    # Get the pandas DataFrame (tracking data)
    df = analysis_manager.get_data_frame()
    
    # Extract hit events and convert them to a DataFrame
    hits_df = extract_hits_to_dataframe(proto_game)

    # Merge tracking data and hit events data
    merged_df = merge_hits_with_tracking(hits_df, df, proto_game)

    # Add basic columns (match_id, team, etc.)
    merged_df = add_basic_columns(merged_df, proto_game)

    return merged_df


def extract_hits_to_dataframe(proto_game) -> pd.DataFrame:
    """
    Extracts hit events from proto_game and converts them into a DataFrame.

    Args:
        proto_game: Protobuf data analyzed by the carball library.

    Returns:
        pd.DataFrame: DataFrame containing the following columns:
            - frame: Frame number when the event occurred
            - player_id: ID of the player who made the hit
            - team: Team name of the player
            - is_kickoff: Whether the hit was during kickoff
            - dribble: Whether the hit was part of a dribble
            - aerial: Whether the hit was an aerial
            - assist: Whether the hit was an assist
            - shot: Whether the hit was a shot
            - goal: Whether the hit resulted in a goal
            - poss_id: Possession ID (increments when possession changes)
            - Additional hit-related information
    """
    hits = proto_game.game_stats.hits
    hit_data = []
    
    # チームごとのプレイヤーIDを辞書に格納
    team_players = {
        0: set(player.id for player in proto_game.teams[0].player_ids),
        1: set(player.id for player in proto_game.teams[1].player_ids)
    }
    
    poss_id = 0
    last_team = None
    
    for hit in hits:
        player_id = hit.player_id.id
        current_team = 0 if player_id in team_players[0] else 1
        lost = False
        
        if last_team is not None and current_team != last_team:
            poss_id += 1
            lost = True
        
        hit_dict = {
            'frame': hit.frame_number,
            'player_id': hit.player_id.id,
            'team': proto_game.teams[current_team].name,
            'home_team': proto_game.teams[current_team].is_orange,
            'is_kickoff': hit.is_kickoff,
            'dribble': hit.dribble,
            'dribble_continuation': hit.dribble_continuation,
            'aerial': hit.aerial,
            'assist': hit.assist,
            'distance_to_goal': hit.distance_to_goal,
            'shot': hit.shot,
            'goal': hit.goal,
            'goal_number': hit.goal_number, 
            'pass': hit.pass_,
            'lost': lost,
            'clear': hit.clear,
            'poss_id': poss_id,
            'ball': hit.ball_data,
            'on_ground': hit.on_ground,
        }
        hit_data.append(hit_dict)
        
        last_team = current_team
    
    hits_df = pd.DataFrame(hit_data)
    hits_df = hits_df.sort_values('frame').reset_index(drop=True)
    
    return hits_df

def merge_hits_with_tracking(hits_df: pd.DataFrame, tracking_df: pd.DataFrame, proto_game) -> pd.DataFrame:
    """
    Merges hit event data with tracking data.

    Args:
        hits_df (pd.DataFrame): DataFrame containing hit events
        tracking_df (pd.DataFrame): DataFrame containing tracking data
        proto_game: Protobuf data containing player information

    Returns:
        pd.DataFrame: Merged DataFrame containing:
            - Basic game information (time, time remaining, etc.)
            - Ball state (position, velocity, angular velocity, etc.)
            - Player state (position, velocity, boost amount, etc.)
            - Detailed hit event information
    """
    base_game_cols = ['time', 'delta', 'seconds_remaining', 
                     'replicated_seconds_remaining', 'ball_has_been_hit', 'goal_number']
    
    game_cols = [('game', col) for col in base_game_cols]
    game_df = tracking_df[game_cols].copy()
    game_df.columns = [f'game_{col}' for col in base_game_cols]
    
    if ('game', 'is_overtime') in tracking_df.columns:
        game_df['game_is_overtime'] = tracking_df[('game', 'is_overtime')]
    else:
        game_df['game_is_overtime'] = None

    ball_cols = [('ball', col) for col in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 
                                         'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'hit_team_no', 
                                         'rot_x', 'rot_y', 'rot_z']]
    ball_df = tracking_df[ball_cols].copy()
    ball_df.columns = [f'ball_{col}' for col in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 
                                               'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'hit_team_no', 
                                               'rot_x', 'rot_y', 'rot_z']]

    tracking_combined = pd.concat([game_df, ball_df], axis=1)
    
    merged_df = pd.merge_asof(
        hits_df.sort_values('frame'),
        tracking_combined,
        left_on='frame',
        right_index=True,
        direction='nearest'
    )

    player_id_to_name = {}
    for player in proto_game.players:
        player_id_to_name[player.id.id] = player.name

    player_columns = ['ping', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                     'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'throttle', 'steer', 
                     'handbrake', 'rot_x', 'rot_y', 'rot_z', 'ball_cam', 'boost', 
                     'boost_active', 'dodge_active', 'jump_active', 
                     'double_jump_active', 'boost_collect']

    for _, row in merged_df.iterrows():
        player_id = row['player_id']
        frame = row['frame']
        player_name = player_id_to_name.get(player_id)
        
        if player_name and (player_name, 'pos_x') in tracking_df.columns:
            for col in player_columns:
                new_col = f'player_{col}'
                try:
                    merged_df.loc[merged_df.index[merged_df['frame'] == frame], new_col] = \
                        tracking_df.loc[frame, (player_name, col)]
                except KeyError:
                    continue

    return merged_df

def add_basic_columns(df: pd.DataFrame, proto_game) -> pd.DataFrame:
    """
    Adds basic match information to the DataFrame.

    Args:
        df (pd.DataFrame): Original DataFrame
        proto_game: Protobuf data containing match metadata

    Returns:
        pd.DataFrame: DataFrame with added columns:
            - match_id: Unique identifier for the match
            - home_score: Score of the home team
            - away_score: Score of the away team
            - goal_diff: Score difference (home - away)
    """
    df['match_id'] = proto_game.game_metadata.match_guid

    team_players = {
        0: set(player.id for player in proto_game.teams[0].player_ids),
        1: set(player.id for player in proto_game.teams[1].player_ids)
    }

    goals_data = []
    home_score = 0
    away_score = 0
    
    for goal in proto_game.game_metadata.goals:
        scorer_id = goal.player_id.id
        frame = goal.frame_number
        scoring_team = 0 if scorer_id in team_players[0] else 1
        
        if scoring_team == 0:
            home_score += 1
        else:
            away_score += 1
            
        goals_data.append({
            'frame': frame,
            'home_score': home_score,
            'away_score': away_score,
            'goal_diff': home_score - away_score
        })

    goals_df = pd.DataFrame(goals_data)
    
    df['home_score'] = 0
    df['away_score'] = 0
    df['goal_diff'] = 0
    
    for _, goal_row in goals_df.iterrows():
        mask = df['frame'] >= goal_row['frame']
        df.loc[mask, 'home_score'] = goal_row['home_score']
        df.loc[mask, 'away_score'] = goal_row['away_score']
        df.loc[mask, 'goal_diff'] = goal_row['goal_diff']

    return df

if __name__ == "__main__":
    import pdb
    import os
    #cd to ../PreProcessing
    rocket_league_path=os.getcwd()+"/test/sports/event_data/data/rocket_league/1e62a0d0-6c47-4b12-a784-b60d9b883632.replay"

    #test load_with_carball
    rocket_league_df=load_with_carball(rocket_league_path)
    rocket_league_df.to_csv(os.getcwd()+"/test/sports/event_data/data/rocket_league/test_data2.csv",index=False)

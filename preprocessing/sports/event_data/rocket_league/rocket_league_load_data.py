import carball
import pandas as pd

def load_with_carball(replay_path: str) -> pd.DataFrame:
    """
    Loads Rocket League replay file and converts it into a DataFrame with event and tracking data.

    Args:
        replay_path (str): Path to the Rocket League replay file.

    Returns:
        pd.DataFrame: DataFrame containing event data merged with corresponding tracking data.
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
    merged_df = merge_hits_with_tracking(hits_df, df)
    
    # Add basic columns (match_id, team, etc.)
    merged_df = add_basic_columns(merged_df, proto_game)

    return merged_df


def extract_hits_to_dataframe(proto_game):
    """
    Extracts hit events from proto_game and converts them to a DataFrame.
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

def merge_hits_with_tracking(hits_df, tracking_df):
    """
    Merges hit events with tracking data.
    """
    # ToDo: Extract the ball and the player data from the tracking_df   
    # ToDo: Merge the hits_df and tracking_df
    return hits_df

def add_basic_columns(df, proto_game):
    """
    Adds basic columns like match_id and team to the DataFrame.
    """
    # ToDo: Add game metadata to the DataFrame
    df['match_id'] = proto_game.game_metadata.match_guid
    
    return df

if __name__ == "__main__":
    import pdb
    import os
    #cd to ../PreProcessing
    rocket_league_path=os.getcwd()+"/test/sports/event_data/data/rocket_league/0328fc07-13fb-4cb6-9a86-7d608196ddbd.replay"

    #test load_with_carball
    rocket_league_df=load_with_carball(rocket_league_path)
    rocket_league_df.to_csv(os.getcwd()+"/test/sports/event_data/data/rocket_league/test_data.csv",index=False)

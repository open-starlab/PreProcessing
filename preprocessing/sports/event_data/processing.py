import os
import pandas as pd
import numpy as np

def seq2event(data):
    """
    Processes soccer match event data to determine possession, filter actions, 
    compute additional metrics, and normalize data.

    Parameters:
    data (pd.DataFrame or str): A pandas DataFrame containing event data or a file path to a CSV file.

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
    
    # Create 'action' column by concatenating 'event_type' and 'event_type_2'
    df["action"] = df["event_type"].astype(str) + "_" + df["event_type_2"].astype(str)

    # Define possession team actions
    possession_team_actions = [
        'Free Kick_Goal kick', 'Free Kick_Throw in', 'Free Kick_Corner', 'Free Kick_Free Kick',
        'Free Kick_Free kick cross', 'Free Kick_Free kick shot', 'Free Kick_Penalty', 'Pass_Cross',
        'Pass_Hand pass', 'Pass_Head pass', 'Pass_High pass', 'Pass_Launch', 'Pass_Simple pass',
        'Pass_Smart pass', 'Shot_Shot', 'Shot_Goal', 'Free Kick_goal', 'Duel_Ground attacking duel_off dribble',
        'Others on the ball_Acceleration', 'Others on the ball_Clearance', 'Others on the ball_Touch_good'
    ]
    
    possession = []
    seconds = []

    # Determine possession and adjust seconds for second half
    for i in range(len(df)):
        if i == 0:
            possession.append(df["team"].iloc[i])
        else:
            if df["team"].iloc[i] == df["team"].iloc[i - 1]:
                possession.append(df["team"].iloc[i])
            else:
                if df["action"].iloc[i] in possession_team_actions:
                    possession.append(df["team"].iloc[i])
                else:
                    possession.append(df["team"].iloc[i - 1])
        
        if df["period"].iloc[i] == "2H":
            seconds.append(df["seconds"].iloc[i] + 15 * 60)
        else:
            seconds.append(df["seconds"].iloc[i])
    
    df["possession_team"] = possession
    df["seconds"] = seconds

    # Normalize time
    df["seconds"] = df["seconds"] / df["seconds"].max()
    #round numerical columns
    df = df.round({"seconds": 4})

    # Filter actions not by team in possession
    df = df[df["team"] == df["possession_team"]].reset_index(drop=True)

    # Define simple actions
    simple_actions = [
        'Foul_Foul', 'Foul_Hand foul', 'Foul_Late card foul', 'Foul_Out of game foul', 'Foul_Protest',
        'Foul_Simulation', 'Foul_Time lost foul', 'Foul_Violent Foul', 'Offside_', 'Free Kick_Corner',
        'Free Kick_Free Kick', 'Free Kick_Free kick cross', 'Free Kick_Free kick shot', 'Free Kick_Goal kick',
        'Free Kick_Penalty', 'Free Kick_Throw in', 'Pass_Cross', 'Pass_Hand pass', 'Pass_Head pass', 'Pass_High pass',
        'Pass_Launch', 'Pass_Simple pass', 'Pass_Smart pass', 'Shot_Shot', 'Shot_Goal', 'Shot_Own_goal', 'Free Kick_goal',
        'Others on the ball_Own_goal', 'Pass_Own_goal', 'Duel_Ground attacking duel', 'Others on the ball_Acceleration',
        'Others on the ball_Clearance', 'Others on the ball_Touch', 'Others on the ball_Touch_good', 
        'Duel_Ground attacking duel_off dribble'
    ]
    
    # Filter out non-simple actions
    df = df[df["action"].isin(simple_actions)].reset_index(drop=True)

    # Calculate match score
    def calculate_match_score(df):
        home_team_score_list = []
        away_team_score_list = []
        score_diff_list = []
        
        for match_id in df.match_id.unique():
            home_team_score = 0
            away_team_score = 0
            home_team_id = df.team.unique()[0]
            away_team_id = df.team.unique()[1]
            match_df = df[df["match_id"] == match_id].reset_index(drop=True)
            
            for i in range(len(match_df)):
                if match_df.iloc[i].event_type_2 == "Goal":
                    if match_df["team"].iloc[i] == home_team_id:
                        home_team_score += 1
                    else:
                        away_team_score += 1
                elif match_df.iloc[i].event_type_2 == "Own_goal":
                    if match_df["team"].iloc[i] == home_team_id:
                        away_team_score += 1
                    else:
                        home_team_score += 1
                score_diff = home_team_score - away_team_score
                home_team_score_list.append(home_team_score)
                away_team_score_list.append(away_team_score)
                score_diff_list.append(score_diff)
        
        return home_team_score_list, away_team_score_list, score_diff_list

    home_team_score_list, away_team_score_list, score_diff_list = calculate_match_score(df)
    df["home_team_score"] = home_team_score_list
    df["away_team_score"] = away_team_score_list
    df["score_diff"] = score_diff_list

    # Set possession id
    poss_id_list = []
    poss_id = 0
    for i in range(len(df)):
        if i == 0:
            poss_id_list.append(0)
        else:
            if df["possession_team"].iloc[i] == df["possession_team"].iloc[i - 1]:
                poss_id_list.append(poss_id)
            else:
                poss_id += 1
                poss_id_list.append(poss_id)
    df["poss_id"] = poss_id_list


    # Add a row in between the first and last row of each possession
    new_df = []
    for poss_id in df.poss_id.unique():
        temp_df = df[df["poss_id"] == poss_id].reset_index(drop=True)
        for j in range(len(temp_df)):
            new_df.append(temp_df.iloc[j])
        new_row = temp_df.iloc[-1].copy()
        new_row["action"] = "_"
        new_df.append(new_row)
    
    # Concatenate all rows in new_df
    new_df = pd.concat(new_df, axis=1).T.reset_index(drop=True)

    # Simplify actions
    drop_list = [
        'Foul_Foul', 'Foul_Hand foul', 'Foul_Late card foul', 'Foul_Out of game foul',
        'Foul_Protest', 'Foul_Simulation', 'Foul_Time lost foul', 'Foul_Violent Foul', 'Offside_'
    ]
    p_list = [
        "Free Kick_Goal kick", 'Free Kick_Throw in', 'Free Kick_Free Kick', 'Pass_Hand pass',
        'Pass_Head pass', 'Pass_High pass', 'Pass_Launch', 'Pass_Simple pass', 'Pass_Smart pass', 
        'Others on the ball_Clearance'
    ]
    d_list = [
        'Duel_Ground attacking duel_off dribble', 'Others on the ball_Acceleration', 'Others on the ball_Touch_good'
    ]
    x_list = [
        'Free Kick_Corner', 'Free Kick_Free kick cross', 'Pass_Cross'
    ]
    s_list = [
        'Free Kick_Free kick shot', 'Free Kick_Penalty', 'Shot_Shot', 'Shot_Goal', 'Shot_Own_goal'
    ]

    new_df = new_df[~new_df["action"].isin(drop_list)].reset_index(drop=True)
    action_list = []
    for action in new_df["action"]:
        if action in p_list:
            action_list.append("p")
        elif action in d_list:
            action_list.append("d")
        elif action in x_list:
            action_list.append("x")
        elif action in s_list:
            action_list.append("s")
        elif action == "_":
            action_list.append("_")
        else:
            action_list.append(action)
    
    new_df["action"] = action_list

    df = new_df.copy()

    # Calculate additional metrics
    def calculate_additional_metrics(df):
        time_diff_list = []
        distance_list = []
        distance2goal_list = []
        angle_list = []
        x_diff_list = []
        y_diff_list = []
        
        for match_id in df.match_id.unique():
            match_df = df[df["match_id"] == match_id].reset_index(drop=True)
            for i in range(len(match_df)):
                if i == 0:
                    time_diff = 0
                    distance = 0
                    distance2goal = 0
                    angle = 0
                    x_diff = 0
                    y_diff = 0
                elif match_df.iloc[i].action == "_":
                    time_diff = 0
                    distance = 0
                    distance2goal = 0
                    angle = 0.5
                    x_diff = 0
                    y_diff = 0
                else:
                    time_diff = match_df["seconds"].iloc[i] - match_df["seconds"].iloc[i - 1]
                    distance = ((match_df["start_x"].iloc[i] * 1.05 - match_df["start_x"].iloc[i-1] * 1.05) ** 2 + 
                                (match_df["start_y"].iloc[i] * 0.68 - match_df["start_y"].iloc[i-1] * 0.68) ** 2) ** 0.5
                    distance2goal = (((match_df["start_x"].iloc[i] - 100/100) * 1.05) ** 2 + 
                                     ((match_df["start_y"].iloc[i] - 50/100) * 0.68) ** 2) ** 0.5
                    angle = np.abs(np.arctan2((match_df["start_y"].iloc[i] - 50/100) * 0.68, 
                                              (match_df["start_x"].iloc[i] - 100/100) * 1.05))
                    x_diff = match_df["start_x"].iloc[i] * 1.05 - match_df["start_x"].iloc[i-1] * 1.05
                    y_diff = match_df["start_y"].iloc[i] * 0.68 - match_df["start_y"].iloc[i-1] * 0.68
                
                time_diff_list.append(time_diff)
                distance_list.append(distance)
                distance2goal_list.append(distance2goal)
                angle_list.append(angle)
                x_diff_list.append(x_diff)
                y_diff_list.append(y_diff)
        
        return time_diff_list, distance_list, distance2goal_list, angle_list, x_diff_list, y_diff_list

    # Scale and normalize columns
    df["start_x"] = df["start_x"] / 100
    df["start_y"] = df["start_y"] / 100
    df["end_x"] = df["end_x"] / 100
    df["end_y"] = df["end_y"] / 100

    (time_diff_list, distance_list, distance2goal_list, angle_list, 
     x_diff_list, y_diff_list) = calculate_additional_metrics(df)
    
    df["time_diff"] = time_diff_list
    df["distance"] = distance_list
    df["distance2goal"] = distance2goal_list
    df["angle2goal"] = angle_list
    df["x_diff"] = x_diff_list
    df["y_diff"] = y_diff_list

    # Scale and normalize columns
    # df["distance"] = df["distance"] / df["distance"].max()
    # df["distance2goal"] = df["distance2goal"] / df["distance2goal"].max()
    # df["angle2goal"] = df["angle2goal"] / df["angle2goal"].max()
    # df["x_diff"] = df["x_diff"] / df["x_diff"].max()
    # df["y_diff"] = df["y_diff"] / df["y_diff"].max()

    # Clip time differences to a maximum of 0.01 seconds
    df["time_diff"] = np.clip(df["time_diff"], 0, 0.01)

    # Round numerical columns
    df = df.round({"seconds": 4, "time_diff": 4, "distance": 4, "distance2goal": 4, "angle2goal": 4,
                   "start_x": 4, "start_y": 4, "end_x": 4, "end_y": 4, "x_diff": 4, "y_diff": 4})

    # Reorder columns
    df = df[[
        "match_id", "poss_id", "team", "action", "start_x", "start_y", "x_diff", "y_diff", 
        "distance", "distance2goal", "angle2goal", "seconds", "time_diff", "score_diff"
    ]]

    return df

def nmstpp(data):
    """
    Processes soccer match event data to determine possession, filter actions, 
    compute additional metrics, and normalize data.

    Parameters:
    data (pd.DataFrame or str): A pandas DataFrame containing event data or a file path to a CSV file.

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
    
    df=seq2event(df)
    
    #define the zone clusters for Juego de Posición
    centroid_x=[ 8.5 , 25.25, 41.75, 58.25, 74.75, 91.5,8.5 , 25.25, 41.75, 58.25, 74.75, 
                91.5,33.5, 66.5,33.5, 66.5,33.5, 66.5,8.5,91.5]
    centroid_y=[89.45, 89.45, 89.45, 89.45, 89.45, 89.45,10.55, 10.55, 10.55, 10.55, 10.55, 10.55,
                71.05, 71.05,50., 50.,28.95, 28.95, 50.,50.]

    #scale start_x and start_y by 100
    df["start_x"]=df["start_x"]*100
    df["start_y"]=df["start_y"]*100

    #calculate the zone of the start_x and start_y
    zone_list=[]
    #get closest zone for each start_x and start_y
    for i in range(len(df)):
        min_dist=1000
        zone=-1
        for j in range(len(centroid_x)):
            dist=np.sqrt((df["start_x"].iloc[i]-centroid_x[j])**2+(df["start_y"].iloc[i]-centroid_y[j])**2)
            if dist<min_dist:
                min_dist=dist
                zone=j
        zone_list.append(zone)
    df["zone"]=zone_list

    # create features
    '''
    'zone_s', distance since previous event
    'zone_deltay', change in zone distance in x 
    'zone_deltax', change in zone distance in y
    'zone_sg',  distance to the center of opponent goal from the zone
    'zone_thetag' angle from the center of opponent goal 
    '''

    zone_s_list=[]
    zone_deltax_list=[]
    zone_deltay_list=[]
    zone_dist2goal_list=[]
    zone_angle2goal_list=[]

    for i in range(len(df)):
        if i==0 or df["poss_id"].iloc[i]!=df["poss_id"].iloc[i-1]:
            zone_s=0
            zone_deltax=0
            zone_deltay=0
            zone_dist2goal=0
            zone_angle2goal=0
        else:
            zone_deltax=centroid_x[df["zone"].iloc[i]]-centroid_x[df["zone"].iloc[i-1]]
            zone_deltay=centroid_y[df["zone"].iloc[i]]-centroid_y[df["zone"].iloc[i-1]]
            zone_s=np.sqrt(zone_deltax**2+zone_deltay**2)
            zone_dist2goal=np.sqrt((centroid_x[df["zone"].iloc[i]]-100)**2+(centroid_y[df["zone"].iloc[i]]-50)**2)
            zone_angle2goal=np.abs(np.arctan2((centroid_y[df["zone"].iloc[i]]-50),(centroid_x[df["zone"].iloc[i]]-100)))
        zone_s_list.append(zone_s)
        zone_deltax_list.append(zone_deltax)
        zone_deltay_list.append(zone_deltay)
        zone_dist2goal_list.append(zone_dist2goal)
        zone_angle2goal_list.append(zone_angle2goal)
    df["zone_s"]=zone_s_list
    df["zone_deltax"]=zone_deltax_list
    df["zone_deltay"]=zone_deltay_list
    df["zone_dist2goal"]=zone_dist2goal_list
    df["zone_angle2goal"]=zone_angle2goal_list

    #reorder columns
    df = df[[
        "match_id", "poss_id", "team", "action","zone","zone_s","zone_deltax","zone_deltay","zone_dist2goal","zone_angle2goal", 
        "seconds", "time_diff", "score_diff",]]
    
    #round numerical columns
    df = df.round({"seconds": 4, "time_diff": 4, "zone_s": 4, "zone_deltax": 4, "zone_deltay": 4, "zone_dist2goal": 4, "zone_angle2goal": 4})

    return df

def lem(data):
    """
    Processes soccer match event data to determine possession, filter actions, 
    compute additional metrics, and normalize data.

    Parameters:
    data (pd.DataFrame or str): A pandas DataFrame containing event data or a file path to a CSV file.

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
    
    #create the period by getting the first character of the period column
    df["Period"]=df["period"].str[0]
    #create minute and second columns
    df["minute"]=df["seconds"]/60 
    df["Minute"]=df["minute"].apply(np.floor)     #round down
    df["Second"]=df["seconds"]%60

    #get the home score and away score and IsHome and IsGoal
    home_score_list=[]
    away_score_list=[]
    is_home_list=[]
    is_goal_list=[]
    for match in df.match_id.unique():
        match_df=df[df["match_id"]==match]
        team_list=df[df["match_id"]==match]["team"].unique()
        home_team=team_list[0]
        away_team=team_list[1]
        home_score=0
        away_score=0
        is_goal=0
        for i in range(len(match_df)):
            if match_df["team"].iloc[i]==home_team:
                is_home_list.append(1)
                if match_df["event_type_2"].iloc[i]=="Goal":
                    home_score+=1
                    is_goal=1
                elif match_df["event_type_2"].iloc[i]=="Own_goal":
                    away_score+=1
                    is_goal=1
            else:
                is_home_list.append(0)
                if match_df["event_type_2"].iloc[i]=="Goal":
                    away_score+=1
                    is_goal=1
                elif match_df["event_type_2"].iloc[i]=="Own_goal":
                    home_score+=1
                    is_goal=1
            home_score_list.append(home_score)
            away_score_list.append(away_score)
            is_goal_list.append(is_goal)
    df["HomeScore"]=home_score_list
    df["AwayScore"]=away_score_list
    df["IsHome"]=is_home_list
    df["IsGoal"]=is_goal_list
   
    #convert col accurate from TF to 1 and 0
    df['IsAccurate']=df['accurate'].astype(int)

    #create the EventType 
    event_type_list=[]
    for i in range(len(df)):
        event_type=df["event_type_2"].iloc[i]
        if event_type=="Goal":
            event_type_list.append("Shot")
        elif event_type=="own-goal":
            event_type_list.append("Shot")
        elif event_type=="Ground attacking duel_off dribble":
            if df["team"].iloc[i]==df["team"].iloc[i-1]:
                event_type_list.append("Ground attacking duel")
            else:
                event_type_list.append("Ground defending duel")
        else:
            event_type_list.append(event_type)
           
    df["EventType"]=event_type_list

    #reorder columns
    df = df[[
        "match_id", "EventType", "IsGoal", "IsAccurate","IsHome", "Period", "Minute","Second","start_x","start_y","HomeScore","AwayScore"
    ]]

    #rename columns
    df.rename(columns={"start_x":"X","start_y":"Y"},inplace=True)

    #round numerical columns to 4 decimal places (period, minute, second, X, Y)
    df = df.round({"Period": 4, "Minute": 4, "Second": 4, "X": 4, "Y": 4})

    return df

def UIED():
    #unified and integrated event data
    return

if __name__ == '__main__':
    import pdb

    # seq2event
    # df_path=os.getcwd()+"/test/sports/event_data/data/wyscout/test_data.csv"
    # df=seq2event(df_path)
    # df.to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_seq2event.csv",index=False)

    # nmstpp
    # df_path=os.getcwd()+"/test/sports/event_data/data/wyscout/test_data.csv"
    # df=nmstpp(df_path)
    # df.to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_nmstpp.csv",index=False)

    # lem
    df_path=os.getcwd()+"/test/sports/event_data/data/wyscout/test_data.csv"
    df=lem(df_path)
    df.to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_lem.csv",index=False)

    print('-----------------end-----------------')
    pdb.set_trace()

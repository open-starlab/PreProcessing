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
        'Others on the ball_Acceleration', 'Others on the ball_Clearance', 'Others on the ball_Touch_good',
        'Shot_Own_goal', 'Pass_Own_goal', 'Others on the ball_Own_goal'
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
            seconds.append(df["seconds"].iloc[i] + 60 * 60)
        elif df["period"].iloc[i] == "E1":
            seconds.append(df["seconds"].iloc[i] + 120 * 60)
        elif df["period"].iloc[i] == "E2":
            seconds.append(df["seconds"].iloc[i] + 150 * 60)
        elif df["period"].iloc[i] == "P":
            seconds.append(df["seconds"].iloc[i] + 180 * 60)
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
            if df["possession_team"].iloc[i] == df["possession_team"].iloc[i - 1] and df["period"].iloc[i] == df["period"].iloc[i - 1]:
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
        'Foul_Protest', 'Foul_Simulation', 'Foul_Time lost foul', 'Foul_Violent Foul', 'Offside_',
        'Others on the ball_Own_goal', 'Pass_Own_goal'
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
    period_list=[]
    for i in range(len(df)):
        if df["period"].iloc[i]=="1H" or df["period"].iloc[i]=="2H":
            period_list.append(int(df["period"].iloc[i][0]))
        elif df["period"].iloc[i]=="E1":
            period_list.append(3)
        elif df["period"].iloc[i]=="E2":
            period_list.append(4)
        elif df["period"].iloc[i]=="P":
            period_list.append(5)
    df["Period"]=period_list

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
        else:
            event_type_list.append(event_type)
           
    df["EventType"]=event_type_list

    #add row period_over and game_over
    new_df=[]
    for match in df.match_id.unique():
        match_df=df[df["match_id"]==match]
        for period in match_df.period.unique():
            period_df=match_df[match_df["period"]==period]
            for i in range(len(period_df)):
                new_df.append(period_df.iloc[i])
            last_row=period_df.iloc[-1].copy()
            #set the IsHome, IsGoal, IsAccurate,  to 0
            last_row["IsHome"]=0
            last_row["IsGoal"]=0
            last_row["IsAccurate"]=0
            #check if it is the last period of the matchs
            if period==match_df.period.unique()[-1]:
                last_row["EventType"]="game_over"
                new_df.append(last_row)
            else:
                last_row["EventType"]="period_over"
                new_df.append(last_row)
    df=pd.concat(new_df,axis=1).T.reset_index(drop=True)

    #reorder columns
    df = df[[
        "match_id", "EventType", "IsGoal", "IsAccurate","IsHome", "Period", "Minute","Second","start_x","start_y","HomeScore","AwayScore"
    ]]

    #rename columns
    df.rename(columns={"start_x":"X","start_y":"Y"},inplace=True)

    #round numerical columns to 4 decimal places (period, minute, second, X, Y)
    df = df.round({"Period": 4, "Minute": 4, "Second": 4, "X": 4, "Y": 4})

    return df

def UIED_wyscout(data):
    """
    Processes soccer match event data to determine possession, filter actions, 
    compute additional metrics, and normalize data.

    Parameters:
    data (pd.DataFrame or str): A pandas DataFrame containing event data or a file path to a CSV file.
    provider (str): The provider of the event data.

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
    

    #get possession team only event
    # Create 'action' column by concatenating 'event_type' and 'event_type_2'
    df["action"] = df["event_type"].astype(str) + "_" + df["event_type_2"].astype(str)

    # Define possession team actions
    possession_team_actions = [
        'Free Kick_Goal kick', 'Free Kick_Throw in', 'Free Kick_Corner', 'Free Kick_Free Kick',
        'Free Kick_Free kick cross', 'Free Kick_Free kick shot', 'Free Kick_Penalty', 'Pass_Cross',
        'Pass_Hand pass', 'Pass_Head pass', 'Pass_High pass', 'Pass_Launch', 'Pass_Simple pass',
        'Pass_Smart pass', 'Shot_Shot', 'Shot_Goal', 'Free Kick_goal', 'Duel_Ground attacking duel_off dribble',
        'Others on the ball_Acceleration', 'Others on the ball_Clearance', 'Others on the ball_Touch_good',
        'Shot_Own_goal', 'Pass_Own_goal', 'Others on the ball_Own_goal'
    ]

    possession = []
    # Determine possession
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
    
    df["possession_team"] = possession
    df = df[df["team"] == df["possession_team"]].reset_index(drop=True)

    #create the event related features (sucess, home_team, goal, home_score, away_score)
    df["success"]=df["accurate"].astype(int)
    home_team_list=[]
    goal_list=[]
    home_score_list=[]
    away_score_list=[]
    goal_diff_list=[]
    for match in df.match_id.unique():
        match_df=df[df["match_id"]==match]
        team_list=match_df["team"].unique()
        home_team=team_list[0]
        away_team=team_list[1]
        home_score=0
        away_score=0
        goal_diff=0
        for i in range(len(match_df)):
            if match_df["team"].iloc[i]==home_team:
                home_team_list.append(1)
                if match_df["event_type_2"].iloc[i]=="Goal":
                    home_score+=1
                elif match_df["event_type_2"].iloc[i]=="Own_goal":
                    away_score+=1
            else:
                home_team_list.append(0)
                if match_df["event_type_2"].iloc[i]=="Goal":
                    away_score+=1
                elif match_df["event_type_2"].iloc[i]=="Own_goal":
                    home_score+=1
            goal_diff=home_score-away_score
            goal_list.append(1) if match_df["event_type_2"].iloc[i]=="Goal" else goal_list.append(0)
            home_score_list.append(home_score)
            away_score_list.append(away_score)
            goal_diff_list.append(goal_diff)
    
    df["home_team"]=home_team_list
    df["goal"]=goal_list
    df["home_score"]=home_score_list
    df["away_score"]=away_score_list
    df["goal_diff"]=goal_diff_list

    #group the event into simpliefied actions
    pass_actions=['Free Kick_Goal kick', 'Free Kick_Throw in','Free Kick_Free Kick','Pass_Cross','Pass_Hand pass','Pass_Simple pass','Pass_Smart pass','Pass_Head pass']
    high_pass_actions=['Pass_High pass']
    shot_actions=['Free Kick_Free kick shot','Free Kick_Penalty','Shot_Shot', 'Shot_Goal','Shot_Own_goal']
    carray_actions=['Others on the ball_Acceleration']
    dribble_actions=['Duel_Ground attacking duel_off dribble', 'Others on the ball_Touch_good','Duel_Air duel']
    cross_actions=['Free Kick_Corner','Free Kick_Free kick cross']
    drop_actions=['Pass_Launch', 'Free Kick_goal', 'Others on the ball_Clearance','Pass_Own_goal', 'Others on the ball_Own_goal','Foul_Foul', 'Foul_Hand foul', 'Foul_Late card foul', 'Foul_Out of game foul',
            'Foul_Protest', 'Foul_Simulation', 'Foul_Time lost foul', 'Foul_Violent Foul', 'Offside_','Duel_Ground loose ball duel','Others on the ball_Touch','Offside_nan','Interruption_Ball out of the field',
            'Duel_Ground defending duel', 'Duel_Ground attacking duel'
            ]
    action_list=[]
    for i in range(len(df)):
        if df["action"].iloc[i] in pass_actions:
            #devide short pass and long pass based on the distance (45)
            distance=np.sqrt(((df["start_x"].iloc[i]-df["end_x"].iloc[i])*1.05)**2+((df["start_y"].iloc[i]-df["end_y"].iloc[i])*0.68)**2)
            if distance>=45:
                action_list.append("long_pass")
            else:
                action_list.append("short_pass")
        elif df["action"].iloc[i] in high_pass_actions:
            action_list.append("high_pass")
        elif df["action"].iloc[i] in shot_actions:
            action_list.append("shot")
        elif df["action"].iloc[i] in carray_actions:
            action_list.append("carry")
        elif df["action"].iloc[i] in dribble_actions:
            action_list.append("dribble")
        elif df["action"].iloc[i] in cross_actions:
            action_list.append("cross")
        elif df["action"].iloc[i] in drop_actions:
            action_list.append("drop")
        else:
            action= df["action"].iloc[i]
            print(f"Warning: action {action} was not found in the action list, it will be dropped")
            action_list.append("drop")
        
    df["action"]=action_list
    #drop the drop actions
    df=df[df["action"]!="drop"].reset_index(drop=True)

    #create the time related features (period, minute, second, delta_T)
    period_list=[]
    minute_list=[]
    second_list=[]
    delta_t_list=[]
    for i in range(len(df)):
        if df["period"].iloc[i]=="1H":
            period_list.append(1)
        elif df["period"].iloc[i]=="2H":
            period_list.append(2)
        elif df["period"].iloc[i]=="E1":
            period_list.append(3)
        elif df["period"].iloc[i]=="E2":
            period_list.append(4)
        elif df["period"].iloc[i]=="P":
            period_list.append(5)
        minute_list.append(df["seconds"].iloc[i]//60)
        second_list.append((df["seconds"].iloc[i]%60).round(4))
        if i==0:
            delta_t_list.append(0)
        else:
            if df.action.iloc[i-1]=="period_over" or df.action.iloc[i-1]=="game_over":
                delta_t_list.append(0)
            else:
                delta_t_list.append(df["seconds"].iloc[i]-df["seconds"].iloc[i-1])
    df["Period"]=period_list
    df["Minute"]=minute_list
    df["Second"]=second_list
    df["delta_T"]=delta_t_list

    #create the location related features (deltaX, deltaY, distance, dist2goal, angle2goal)
    delta_x_list=[]
    delta_y_list=[]
    dist_list=[]
    dist2goal_list=[]
    angle2goal_list=[]
    for i in range(len(df)):
        delta_x=df["start_x"].iloc[i]-df["start_x"].iloc[i-1]
        delta_y=df["start_y"].iloc[i]-df["start_y"].iloc[i-1]
        distance = ((df["start_x"].iloc[i] * 1.05 - df["start_x"].iloc[i-1] * 1.05) ** 2 + 
                            (df["start_y"].iloc[i] * 0.68 - df["start_y"].iloc[i-1] * 0.68) ** 2) ** 0.5
        dist2goal = (((df["start_x"].iloc[i] - 100) * 1.05) ** 2 + 
                            ((df["start_y"].iloc[i] - 50) * 0.68) ** 2) ** 0.5
        angle2goal = np.abs(np.arctan2((df["start_y"].iloc[i] - 50) * 0.68, 
                                    (df["start_x"].iloc[i] - 100) * 1.05))

        delta_x_list.append(delta_x)
        delta_y_list.append(delta_y)
        dist_list.append(distance)
        dist2goal_list.append(dist2goal)
        angle2goal_list.append(angle2goal)
    df["deltaX"]=delta_x_list
    df["deltaY"]=delta_y_list
    df["distance"]=dist_list
    df["dist2goal"]=dist2goal_list
    df["angle2goal"]=angle2goal_list

    #create the possession id, end of possession, end of period, end of game
    poss_id_list = []
    poss_id = 0
    for match in df.match_id.unique():
        match_df = df[df["match_id"] == match]
        for i in range(len(match_df)):
            if i == 0:
                poss_id_list.append(poss_id)
            else:
                if match_df["possession_team"].iloc[i] == match_df["possession_team"].iloc[i - 1]:
                    poss_id_list.append(poss_id)
                else:
                    poss_id += 1
                    poss_id_list.append(poss_id)
        poss_id+=1
    df["poss_id"] = poss_id_list

    new_df = []
    for match in df.match_id.unique():
        match_df = df[df["match_id"] == match]
        for period in match_df.Period.unique():
            period_df = match_df[match_df["Period"] == period]
            for poss_id in period_df.poss_id.unique():
                poss_df = period_df[period_df["poss_id"] == poss_id]
                for i in range(len(poss_df)):
                    new_df.append(poss_df.iloc[i])
                last_row = poss_df.iloc[-1].copy()
                last_row["action"] = "_"
                #change the value of the features to 0
                last_row["success"]=0
                last_row["deltaX"]=0
                last_row["deltaY"]=0
                last_row["distance"]=0
                last_row["dist2goal"]=0
                last_row["angle2goal"]=0.5
                last_row["delta_T"]=0
                new_df.append(last_row)
            last_row = period_df.iloc[-1].copy()
            #change the value of the features to 0
            last_row["success"]=0
            last_row["deltaX"]=0
            last_row["deltaY"]=0
            last_row["distance"]=0
            last_row["dist2goal"]=0
            last_row["angle2goal"]=0.5
            last_row["delta_T"]=0
            if period == df.Period.unique()[-1]:
                last_row["action"] = "game_over"
                new_df.append(last_row)
            else:
                last_row["action"] = "period_over"
                new_df.append(last_row)
    df = pd.concat(new_df, axis=1).T.reset_index(drop=True)

    #reorder columns
    df = df[['match_id', 'poss_id', 'team', 'home_team', 'action', 'success', 'goal', 'home_score', 'away_score', 'goal_diff', 'Period', 'Minute', 'Second', 'seconds', "delta_T", 'start_x', 'start_y', 'deltaX', 'deltaY', 'distance', 'dist2goal', 'angle2goal']]

    #convert seconds, distance, dist2goal, angle2goal into type float
    df["seconds"]=df["seconds"].astype(float)
    df["distance"]=df["distance"].astype(float)
    df["dist2goal"]=df["dist2goal"].astype(float)
    df["angle2goal"]=df["angle2goal"].astype(float)

    #round numerical columns to 4 decimal places (period, minute, second, X, Y)
    df = df.round({"Period": 4, "Minute": 4, "Second": 4, "seconds": 4, "start_x": 4, "start_y": 4, "deltaX": 4, "deltaY": 4, "distance": 4, "dist2goal": 4, "angle2goal": 4})

    return df

def UIED_statsbomb(data):
    """
    Processes soccer match event data to determine possession, filter actions, 
    compute additional metrics, and normalize data.

    Parameters:
    data (pd.DataFrame or str): A pandas DataFrame containing event data or a file path to a CSV file.
    provider (str): The provider of the event data.

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


    #get possession team only event
    df["action"] = df["event_type"].astype(str) + "_" + df["event_type_2"].astype(str)

    # Define possession team actions

    possession_team_actions =[ 'Pass_Ground Pass',  'Pass_Long_HighPass', 
    'Carry_nan', 'Pass_High Pass', 'Pass_Low Pass', 
    'Miscontrol_nan',
    'Dribble_nan', 'Clearance_nan', 'Pass_Cross', 'Ball Recovery_nan',
    'Offside_nan', 'Goal Keeper_nan',
    'Dribbled Past_nan', 'Pass_Corner',
    'Shot_Saved', 'Shot_Blocked', 'Shot_Wayward', 'Shot_Off T', 'Shot_Goal', 'Shot_Post',
    'Tactical Shift_nan', 'Shield_nan',
    'Own Goal Against_Own goal', 'Error_nan',
    'Shot_Saved Off Target']

    # Determine possession
    possession = []
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
    
    df["possession_team"] = possession
    df = df[df["team"] == df["possession_team"]].reset_index(drop=True)
    
    #create the event related features (sucess, home_team, goal, home_score, away_score)
    sucess_list=[]
    home_team_list=[]
    goal_list=[]
    goal_diff_list=[]
    home_score_list=[]
    away_score_list=[]
    for match in df.match_id.unique():
        match_df=df[df["match_id"]==match]
        team_list=match_df["team"].unique()
        home_team=team_list[0]
        away_team=team_list[1]
        home_score=0
        away_score=0
        for i in range(len(match_df)):
            if match_df["team"].iloc[i]==home_team:
                home_team_list.append(1)
                if match_df["event_type_2"].iloc[i]=="Goal":
                    home_score+=1
                elif match_df["event_type_2"].iloc[i]=="Own_goal":
                    away_score+=1
            else:
                home_team_list.append(0)
                if match_df["event_type_2"].iloc[i]=="Goal":
                    away_score+=1
                elif match_df["event_type_2"].iloc[i]=="Own_goal":
                    home_score+=1
            sucess_list.append(1) if match_df["possession_team"].iloc[i]==match_df["possession_team"].iloc[i-1] else sucess_list.append(0)
            goal_list.append(1) if match_df["event_type_2"].iloc[i]=="Goal" else goal_list.append(0)
            home_score_list.append(home_score)
            away_score_list.append(away_score)
            goal_diff=home_score-away_score
            goal_diff_list.append(goal_diff)
    
    df["success"]=sucess_list
    df["home_team"]=home_team_list
    df["goal"]=goal_list
    df["home_score"]=home_score_list
    df["away_score"]=away_score_list
    df["goal_diff"]=goal_diff_list

    #group the event into simpliefied actions
    '''
    all action
    ['Starting XI_nan', 'Half Start_nan', 'Pass_Ground Pass', 'Ball Receipt*_nan',
    'Carry_nan', 'Pass_High Pass', 'Pass_Low Pass', 'Duel_nan', 'Pressure_nan',
    'Foul Committed_nan', 'Foul Won_nan', 'Miscontrol_nan', 'Block_nan',
    'Dribble_nan', 'Clearance_nan', 'Pass_Cross', 'Ball Recovery_nan',
    'Dispossessed_nan', 'Interception_nan', 'Offside_nan', 'Goal Keeper_nan',
    'Injury Stoppage_nan', 'Player Off_nan', 'Referee Ball-Drop_nan',
    'Player On_nan', 'Dribbled Past_nan', 'Shot_Saved to Post', 'Pass_Corner',
    'Shot_Saved', 'Shot_Blocked', 'Shot_Wayward', 'Shot_Off T', 'Half End_nan',
    'Substitution_nan', '50/50_nan', 'Shot_Goal', 'Shot_Post',
    'Tactical Shift_nan', 'Bad Behaviour_nan', 'Shield_nan',
    'Own Goal Against_Own goal', 'Own Goal For_nan', 'Error_nan',
    'Shot_Saved Off Target']
    '''

    pass_actions=['Pass_Ground Pass', 'Pass_Low Pass',]
    high_pass_actions=['Pass_High Pass',]
    shot_actions=['Shot_Saved to Post','Shot_Saved', 'Shot_Blocked', 'Shot_Wayward','Shot_Saved Off Target','Shot_Off T',]
    carray_actions=['Carry_nan',]
    dribble_actions=['Dribble_nan','Shot_Saved', 'Shot_Blocked', 'Shot_Wayward', 'Shot_Off T']
    cross_actions=['Pass_Cross','Pass_Corner','Shot_Goal', 'Shot_Post',]
    drop_actions=['Starting XI_nan', 'Half Start_nan', 'Ball Receipt*_nan', 'Pressure_nan', 'Foul Committed_nan', 'Foul Won_nan', 'Miscontrol_nan', 'Block_nan',
                    'Clearance_nan','Ball Recovery_nan','Dispossessed_nan', 'Interception_nan', 'Offside_nan', 'Goal Keeper_nan','Injury Stoppage_nan', 'Player Off_nan', 'Referee Ball-Drop_nan','Player On_nan',
                    'Dribbled Past_nan','Half End_nan','Substitution_nan', '50/50_nan', 'Tactical Shift_nan', 'Bad Behaviour_nan', 'Shield_nan','Own Goal Against_Own goal', 'Own Goal For_nan', 'Error_nan','Duel_nan',]
    
    action_list=[]
    for i in range(len(df)):
        if df["action"].iloc[i] in pass_actions:
            #devide short pass and long pass based on the distance (45)
            distance=np.sqrt(((df["start_x"].iloc[i]-df["end_x"].iloc[i])*(1.05/1.2))**2+((df["start_y"].iloc[i]-df["end_y"].iloc[i])*(0.68/0.8))**2)
            if distance>=45:
                action_list.append("long_pass")
            else:
                action_list.append("short_pass")
        elif df["action"].iloc[i] in high_pass_actions:
            action_list.append("high_pass")
        elif df["action"].iloc[i] in shot_actions:
            action_list.append("shot")
        elif df["action"].iloc[i] in carray_actions:
            action_list.append("carry")
        elif df["action"].iloc[i] in dribble_actions:
            action_list.append("dribble")
        elif df["action"].iloc[i] in cross_actions:
            action_list.append("cross")
        elif df["action"].iloc[i] in drop_actions:
            action_list.append("drop")
        else:
            action= df["action"].iloc[i]
            print(f"Warning: action {action} was not found in the action list, it will be dropped")
            action_list.append("drop")
        
    df["action"]=action_list
    #drop the drop actions
    df=df[df["action"]!="drop"].reset_index(drop=True)

    #create the time related features (seconds,delta_T)
    df["seconds"]=df["minute"]*60+df["second"]
    delta_t_list=[]
    for i in range(len(df)):
        if i==0:
            delta_t_list.append(0)
        else:
            if df.action.iloc[i-1]=="period_over" or df.action.iloc[i-1]=="game_over":
                delta_t_list.append(0)
            else:
                delta_t_list.append(df["seconds"].iloc[i]-df["seconds"].iloc[i-1])
    df["delta_T"]=delta_t_list

    #create the location related features (deltaX, deltaY, distance, dist2goal, angle2goal)
    delta_x_list=[]
    delta_y_list=[]
    dist_list=[]
    dist2goal_list=[]
    angle2goal_list=[]
    for i in range(len(df)):
        delta_x=df["start_x"].iloc[i]-df["start_x"].iloc[i-1]
        delta_y=df["start_y"].iloc[i]-df["start_y"].iloc[i-1]
        distance = ((df["start_x"].iloc[i] * 1.05 - df["start_x"].iloc[i-1] * 1.05) ** 2 + 
                            (df["start_y"].iloc[i] * 0.68 - df["start_y"].iloc[i-1] * 0.68) ** 2) ** 0.5
        dist2goal = (((df["start_x"].iloc[i] - 100) * 1.05) ** 2 + 
                            ((df["start_y"].iloc[i] - 50) * 0.68) ** 2) ** 0.5
        angle2goal = np.abs(np.arctan2((df["start_y"].iloc[i] - 50) * 0.68, 
                                    (df["start_x"].iloc[i] - 100) * 1.05))

        delta_x_list.append(delta_x)
        delta_y_list.append(delta_y)
        dist_list.append(distance)
        dist2goal_list.append(dist2goal)
        angle2goal_list.append(angle2goal)
    df["deltaX"]=delta_x_list
    df["deltaY"]=delta_y_list
    df["distance"]=dist_list
    df["dist2goal"]=dist2goal_list
    df["angle2goal"]=angle2goal_list

    #set possession_id 
    poss_id_list = []
    poss_id = 0
    for i in range(len(df)):
        if i == 0:
            poss_id_list.append(0)
        else:
            if df["possession_team"].iloc[i] == df["possession_team"].iloc[i - 1] and df["period"].iloc[i] == df["period"].iloc[i - 1]:
                poss_id_list.append(poss_id)
            else:
                poss_id += 1
                poss_id_list.append(poss_id)
    df["poss_id"] = poss_id_list

    #rename columns period to Period, minute to Minute, second to Second
    df.rename(columns={"period":"Period","minute":"Minute","second":"Second"},inplace=True)

    new_df = []
    for match in df.match_id.unique():
        match_df = df[df["match_id"] == match]
        for period in match_df.Period.unique():
            period_df = match_df[match_df["Period"] == period]
            for poss_id in period_df.poss_id.unique():
                poss_df = period_df[period_df["poss_id"] == poss_id]
                for i in range(len(poss_df)):
                    if poss_id==period_df.poss_id.unique()[0] and i==0:
                        first_row=poss_df.iloc[i].copy()
                        first_row["deltaX"]=0
                        first_row["deltaY"]=0
                        first_row["distance"]=0
                        first_row["delta_T"]=0
                        new_df.append(first_row)
                    else:
                        new_df.append(poss_df.iloc[i])
                last_row = poss_df.iloc[-1].copy()
                last_row["action"] = "_"
                #change the value of the features to 0
                last_row["success"]=0
                last_row["deltaX"]=0
                last_row["deltaY"]=0
                last_row["distance"]=0
                last_row["dist2goal"]=0
                last_row["angle2goal"]=0.5
                last_row["delta_T"]=0
                new_df.append(last_row)
            last_row = period_df.iloc[-1].copy()
            #change the value of the features to 0
            last_row["success"]=0
            last_row["deltaX"]=0
            last_row["deltaY"]=0
            last_row["distance"]=0
            last_row["dist2goal"]=0
            last_row["angle2goal"]=0.5
            last_row["delta_T"]=0
            if period == df.Period.unique()[-1]:
                last_row["action"] = "game_over"
                new_df.append(last_row)
            else:
                last_row["action"] = "period_over"
                new_df.append(last_row)
    df = pd.concat(new_df, axis=1).T.reset_index(drop=True)

    #remove carray action that have the same start and end location as the previous action (exclude "_" end of possession)
    droplist=[]
    for i in range(len(df)):
        if df.start_x.iloc[i]==df.start_x.iloc[i-1] and df.start_y.iloc[i]==df.start_y.iloc[i-1]:
            if df.action.iloc[i]=="carry" and df.action.iloc[i-1] not in ["_", "period_over", "game_over"]:
                droplist.append(i)
            
    df.drop(droplist,inplace=True)

    #
    new_df=[]
    flag=False
    for i in range(len(df)):
        if i==len(df)-1:
            new_df.append(df.iloc[i])
            break
        if flag:
            flag=False
            new_df.append(row)
            continue
        if df.start_x.iloc[i]==df.start_x.iloc[i+1] and df.start_y.iloc[i]==df.start_y.iloc[i+1]:
            if df.action.iloc[i]=="carry" and df.action.iloc[i+1] in ["short_pass", "long_pass", "high_pass", "shot", "dribble", "cross"]:
                row=df.iloc[i].copy()
                row["action"]=df.action.iloc[i+1]
                flag=True
            else:
                new_df.append(df.iloc[i])
        else:
            new_df.append(df.iloc[i])   

    df=pd.concat(new_df,axis=1).T.reset_index(drop=True)

    #reorder columns
    try:
        sb360_columns = ["h"+str(i)+"_"+j for i in range(1, 12) for j in ["teammate", "actor", "keeper", "x", "y"]] + ["a"+str(i)+"_"+j for i in range(1, 12) for j in ["teammate", "actor", "keeper", "x", "y"]]
        df = df[['match_id', 'poss_id', 'team', 'home_team', 'action', 'success', 'goal', 'home_score', 'away_score', 'goal_diff', 'Period', 'Minute', 'Second', 'seconds', "delta_T", 'start_x', 'start_y', 'deltaX', 'deltaY', 'distance', 'dist2goal', 'angle2goal']+sb360_columns]
        #set the sb360 columns to 4 decimal places
        for col in ["h"+str(i)+"_"+j for i in range(1, 12) for j in ["x", "y"]] + ["a"+str(i)+"_"+j for i in range(1, 12) for j in ["x", "y"]]:
            #change the type of the column to float
            df[col]=df[col].astype(float)
            df[col]=df[col].round(4)
    except:
        try:
            home_tracking_columns = []
            away_tracking_columns = []
            for i in range(1, 24):
                home_tracking_columns.extend([f"h{i}_x", f"h{i}_y"])
                away_tracking_columns.extend([f"a{i}_x", f"a{i}_y"])
            df = df[['match_id', 'poss_id', 'team', 'home_team', 'action', 'success', 'goal', 'home_score', 'away_score', 'goal_diff', 'Period', 'Minute', 'Second', 'seconds', "delta_T", 'start_x', 'start_y', 'deltaX', 'deltaY', 'distance', 'dist2goal', 'angle2goal']+home_tracking_columns+away_tracking_columns]
            #set the home_tracking_columns and away_tracking_columns to 4 decimal places
            # for col in home_tracking_columns+away_tracking_columns:
            #     df[col]=df[col].round(4)
        except:
            df = df[['match_id', 'poss_id', 'team', 'home_team', 'action', 'success', 'goal', 'home_score', 'away_score', 'goal_diff', 'Period', 'Minute', 'Second', 'seconds', "delta_T", 'start_x', 'start_y', 'deltaX', 'deltaY', 'distance', 'dist2goal', 'angle2goal']]
    #convert seconds, distance, dist2goal, angle2goal, deltaX, deltaY into type float
    df["seconds"]=df["seconds"].astype(float)
    df["distance"]=df["distance"].astype(float)
    df["dist2goal"]=df["dist2goal"].astype(float)
    df["angle2goal"]=df["angle2goal"].astype(float)
    df["deltaX"]=df["deltaX"].astype(float)
    df["deltaY"]=df["deltaY"].astype(float)
    df["delta_T"]=df["delta_T"].astype(float)

    #round numerical columns to 4 decimal places (period, minute, second, X, Y,deltaX, deltaY, distance, dist2goal, angle2goal)
    df = df.round({"Period": 4, "Minute": 4, "Second": 4, "seconds": 4, "start_x": 4, "start_y": 4, "deltaX": 4, "deltaY": 4, "distance": 4, "dist2goal": 4, "angle2goal": 4, "delta_T": 4})

    return df

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
    # df_path=os.getcwd()+"/test/sports/event_data/data/wyscout/test_data.csv"
    # df=lem(df_path)
    # df.to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_lem.csv",index=False)

    # UIED
    # df_wyscout_path=os.getcwd()+"/test/sports/event_data/data/wyscout/test_data.csv"
    # df_wyscout=UIED_wyscout(df_wyscout_path)  
    # df_wyscout.to_csv(os.getcwd()+"/test/sports/event_data/data/wyscout/test_preprocess_wyscout_UIED.csv",index=False)

    df_statsbomb_skillcorner_path=os.getcwd()+"/test/sports/event_data/data/statsbomb_skillcorner/test_data.csv"
    df_statsbomb_skillcorner=UIED_statsbomb(df_statsbomb_skillcorner_path)
    df_statsbomb_skillcorner.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb_skillcorner/test_preprocess_UIED.csv",index=False)

    # df_statsbomb_json_path=os.getcwd()+"/test/sports/event_data/data/statsbomb/test_data.csv"
    # df_statsbomb_json=UIED_statsbomb(df_statsbomb_json_path)
    # df_statsbomb_json.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_preprocess_UIED.csv",index=False)

    # df_statsbomb_api_path=os.getcwd()+"/test/sports/event_data/data/statsbomb/test_api_data.csv"
    # df_statsbomb_api=UIED_statsbomb(df_statsbomb_api_path)
    # df_statsbomb_api.to_csv(os.getcwd()+"/test/sports/event_data/data/statsbomb/test_preprocess_api_UIED.csv",index=False)

    # df_metrica_path=os.getcwd()+"/test/sports/event_data/data/metrica/test_data.csv"
    # df_metrica=UIED(df_metrica_path)
    # df_metrica.to_csv(os.getcwd()+"/test/sports/event_data/data/metrica/test_preprocess_UIED.csv",index=False)

    print('-----------------end-----------------')
    pdb.set_trace()

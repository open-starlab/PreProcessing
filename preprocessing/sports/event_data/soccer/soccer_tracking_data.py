import pandas as pd
import os
import pdb
import pandas as pd

def statsbomb_skillcorner_tracking_data_preprocessing(df_raw, save_path=None):
    """
    Preprocess tracking data for StatsBomb and SkillCorner data formats to standardize the coordinates

    Parameters:
    - df (pd.DataFrame or str): DataFrame containing tracking data or a path to a CSV file.
      Expected columns include 'home_team', 'home_side', and optional columns like 'action' or 'event_type'.
    - save_path (str): Path to save the preprocessed data as a CSV file.

    Steps:
    1. Load CSV if `df` is a file path; validate the input to ensure it is a DataFrame.
    2. Define possession team actions to categorize certain events as possession-related.
    3. Adjust player coordinates by shifting the origin to the center and flipping coordinates
       if the home team plays on the right side (field normalization).
    4. Process each row based on the action or event type to determine whether switching
       the field orientation is necessary.
    5. Save the modified DataFrame to the specified path.

    Notes:
    - Assumes field dimensions of 105 x 68 meters.
    - Applies scaling for event data start_x and start_y to adjust coordinates to the field dimensions.
    """
    FIELD_LENGTH = 105.0  # Field length in meters
    FIELD_WIDTH = 68.0  # Field width in meters

    # Load data if `df_raw` is a file path; validate input
    if not isinstance(df_raw, pd.DataFrame):
        if isinstance(df_raw, str):
            df_raw = pd.read_csv(df_raw)
        else:
            raise ValueError("Input should be a DataFrame or a CSV file path")

    # Define list of team actions that imply possession
    team_actions = [
        'Pass_Ground Pass', 'Pass_Long_HighPass', 'Carry_nan', 'Pass_High Pass', 'Pass_Low Pass',
        'Miscontrol_nan', 'Dribble_nan', 'Clearance_nan', 'Pass_Cross', 'Ball Recovery_nan',
        'Offside_nan', 'Goal Keeper_nan', 'Dribbled Past_nan', 'Pass_Corner', 'Shot_Saved',
        'Shot_Blocked', 'Shot_Wayward', 'Shot_Off T', 'Shot_Goal', 'Shot_Post',
        'Tactical Shift_nan', 'Shield_nan', 'Own Goal Against_Own goal', 'Error_nan',
        'Shot_Saved Off Target', 'Ball Receipt*_nan', 'Pressure_nan', 'Interception_nan'
    ]

    # Function to adjust coordinates based on field orientation
    def adjust_coordinates(idx, switch_sides):
        """
        Adjusts the x and y coordinates for players on the field based on field orientation.
        
        Parameters:
        - idx (int): The index of the row to modify in df.
        - switch_sides (bool): Flag indicating if coordinates should be flipped.
        """
        for prefix in ['h', 'a']:  # 'h' for home, 'a' for away
            for i in range(1, 24):
                x_col, y_col = f"{prefix}{i}_x", f"{prefix}{i}_y"
                x, y = df.at[idx, x_col], df.at[idx, y_col]

                # Skip if x and y are zero (indicating missing data)
                if x == 0 and y == 0:
                    continue

                # Adjust coordinates based on `switch_sides` flag
                df.at[idx, x_col] = (-x if switch_sides else x) + FIELD_LENGTH / 2
                df.at[idx, y_col] = (y if switch_sides else -y) + FIELD_WIDTH / 2
                #round to 2 decimal places
                df.at[idx, x_col] = round(df.at[idx, x_col], 2)
                df.at[idx, y_col] = round(df.at[idx, y_col], 2)

    # Process each row
    df = df_raw.copy()
    for idx in range(len(df)):
        home_team, home_side = df.at[idx, 'home_team'], df.at[idx, 'home_side']
        switch_sides = False  # Default: no switch

        if 'action' in df.columns:
            # Use switch condition based on the home team's side in possession
            if (home_team == 1 and home_side == 'right') or (home_team == 0 and home_side == 'left'):
                switch_sides = True
        elif 'event_type' in df.columns:
            # Scale start_x and start_y for event data
            df.at[idx, "start_x"] *= (1.05 / 1.2)
            df.at[idx, "start_y"] *= (0.68 / 0.8)
            #round to 2 decimal places
            df.at[idx, "start_x"] = round(df.at[idx, "start_x"], 2)
            df.at[idx, "start_y"] = round(df.at[idx, "start_y"], 2)

            action_type = f"{df.at[idx, 'event_type']}_{str(df.at[idx, 'event_type_2']).replace('None', 'nan')}"
            is_possession_action = action_type in team_actions

            # Determine if coordinates should be switched based on possession action and home side
            if is_possession_action:
                switch_sides = (home_team == 1 and home_side == 'right') or (home_team == 0 and home_side == 'left')
            else:
                switch_sides = not ((home_team == 1 and home_side == 'right') or (home_team == 0 and home_side == 'left'))

        # Apply coordinate adjustment for each row by index
        adjust_coordinates(idx, switch_sides)

    # Save the processed DataFrame to a CSV file
    if save_path is not None:
        df.to_csv(save_path, index=False)
    
    return df

if __name__=="__main__":
    df_path = os.getcwd() + "/test/sports/event_data/data/statsbomb_skillcorner/test_data.csv"
    save_path = os.getcwd() + "/test/sports/event_data/data/statsbomb_skillcorner/track_data_preprocessed.csv"
    statsbomb_skillcorner_tracking_data_preprocessing(df_path, save_path)
    print("done")
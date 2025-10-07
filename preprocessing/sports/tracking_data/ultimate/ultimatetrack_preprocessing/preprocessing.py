import os

import numpy as np
import pandas as pd


def calculate_magnitude_angle_features(
    vx, vy, ax, ay, prev_v_angle=None, prev_a_angle=None
):
    """Calculate magnitude and angle features"""
    # Velocity magnitude and angle
    v_mag = np.sqrt(vx**2 + vy**2) if not (np.isnan(vx) or np.isnan(vy)) else np.nan
    v_angle = np.arctan2(vy, vx) if not (np.isnan(vx) or np.isnan(vy)) else np.nan

    # Acceleration magnitude and angle
    a_mag = np.sqrt(ax**2 + ay**2) if not (np.isnan(ax) or np.isnan(ay)) else np.nan
    a_angle = np.arctan2(ay, ax) if not (np.isnan(ax) or np.isnan(ay)) else np.nan

    # Angle differences
    diff_v_a_angle = np.nan
    if not (np.isnan(v_angle) or np.isnan(a_angle)):
        diff_v_a_angle = np.arctan2(
            np.sin(v_angle - a_angle), np.cos(v_angle - a_angle)
        )

    diff_v_angle = np.nan
    if prev_v_angle is not None and not (np.isnan(v_angle) or np.isnan(prev_v_angle)):
        diff_v_angle = np.arctan2(
            np.sin(v_angle - prev_v_angle), np.cos(v_angle - prev_v_angle)
        )

    diff_a_angle = np.nan
    if prev_a_angle is not None and not (np.isnan(a_angle) or np.isnan(prev_a_angle)):
        diff_a_angle = np.arctan2(
            np.sin(a_angle - prev_a_angle), np.cos(a_angle - prev_a_angle)
        )

    return (
        v_mag,
        a_mag,
        v_angle,
        a_angle,
        diff_v_a_angle,
        diff_v_angle,
        diff_a_angle,
    )


def create_intermediate_file(raw_data):
    """
    Create intermediate file with specified columns:
    frame,id,x,y,vx,vy,ax,ay,v_mag,a_mag,v_angle,a_angle,diff_v_a_angle,diff_v_angle,diff_a_angle,class,holder,closest

    Args:
        raw_data: Raw Ultimate Track data DataFrame

    Returns:
        DataFrame: Intermediate data with calculated features
    """
    intermediate_data = []

    # Group by id to track previous angles for each entity
    entity_prev_angles = {}

    # Process data frame by frame
    for frame in sorted(raw_data["frame"].unique()):
        frame_data = raw_data[raw_data["frame"] == frame].copy()

        for _, row in frame_data.iterrows():
            entity_id = row["id"]
            entity_key = f"{entity_id}_{row['class']}"

            # Get previous angles for this entity
            prev_v_angle = entity_prev_angles.get(f"{entity_key}_v", None)
            prev_a_angle = entity_prev_angles.get(f"{entity_key}_a", None)

            # Calculate magnitude and angle features
            (
                v_mag,
                a_mag,
                v_angle,
                a_angle,
                diff_v_a_angle,
                diff_v_angle,
                diff_a_angle,
            ) = calculate_magnitude_angle_features(
                row["vx"],
                row["vy"],
                row["ax"],
                row["ay"],
                prev_v_angle,
                prev_a_angle,
            )

            # Create intermediate row
            intermediate_row = {
                "frame": row["frame"],
                "id": row["id"],
                "x": row["x"],
                "y": row["y"],
                "vx": row["vx"],
                "vy": row["vy"],
                "ax": row["ax"],
                "ay": row["ay"],
                "v_mag": v_mag,
                "a_mag": a_mag,
                "v_angle": v_angle,
                "a_angle": a_angle,
                "diff_v_a_angle": diff_v_a_angle,
                "diff_v_angle": diff_v_angle,
                "diff_a_angle": diff_a_angle,
                "class": row["class"],
                "holder": row["holder"],
                "closest": row["closest"],
            }

            intermediate_data.append(intermediate_row)

            # Update previous angles
            entity_prev_angles[f"{entity_key}_v"] = v_angle
            entity_prev_angles[f"{entity_key}_a"] = a_angle

    return pd.DataFrame(intermediate_data)


def create_events_metrica(df):
    """
    Create the Metrica DataFrame for events

    Args:
        df (DataFrame): The DataFrame containing the data

    Returns:
        DataFrame: The DataFrame containing the events
    """
    # Define the columns of the DataFrame
    columns = [
        "Team",
        "Type",
        "Subtype",
        "Period",
        "Start Frame",
        "Start Time [s]",
        "End Frame",
        "End Time [s]",
        "From",
        "To",
        "Start X",
        "Start Y",
        "End X",
        "End Y",
    ]

    # Get the min and max frame
    min_frame = df["frame"].min()
    max_frame = df["frame"].max()

    # Get the DataFrame of the disc
    disc_df = df[df["class"] == "disc"]

    # Create NaN column
    nan_column = pd.Series([np.nan] * (max_frame - min_frame + 1))

    # Create columns
    start_frame = pd.Series(np.arange(min_frame, max_frame + 1))
    start_time = (start_frame / 15).round(6)
    start_x = disc_df["x"].round(2).reset_index(drop=True)
    start_y = disc_df["y"].round(2).reset_index(drop=True)
    offense_ids = sorted(df.loc[df["class"] == "offense", "id"].unique())

    # Get holder information
    holder_data = df.loc[df["holder"]]
    if not holder_data.empty:
        to_id = (
            holder_data["id"]
            .map(lambda x: offense_ids.index(x) if x in offense_ids else np.nan)
            .reset_index(drop=True)
        )
    else:
        to_id = pd.Series([np.nan] * len(start_frame))

    # Create the DataFrame for events
    events_df = pd.concat(
        [
            nan_column,
            nan_column,
            nan_column,
            nan_column,
            start_frame,
            start_time,
            nan_column,
            nan_column,
            to_id,
            nan_column,
            start_x,
            start_y,
            nan_column,
            nan_column,
        ],
        axis=1,
    )
    events_df.columns = columns

    return events_df


def create_tracking_metrica(df, team):
    """
    Create the Metrica DataFrame for tracking data

    Args:
        df (DataFrame): The DataFrame containing the data
        team (str): The team name

    Returns:
        DataFrame: The DataFrame containing the tracking data
    """
    # Define the levels of the MultiIndex
    level_0 = [""] * 3 + [team] * 14 + [""] * 3
    level_1 = [""] * 3 + [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6] + [""] * 3
    level_2 = [
        "Period",
        "Frame",
        "Time [s]",
        "Player0",
        "Player0",
        "Player1",
        "Player1",
        "Player2",
        "Player2",
        "Player3",
        "Player3",
        "Player4",
        "Player4",
        "Player5",
        "Player5",
        "Player6",
        "Player6",
        "Disc__",
        "Disc__",
        "Selected",
    ]

    # Create the MultiIndex
    multi_columns = pd.MultiIndex.from_arrays([level_0, level_1, level_2])

    min_frame = df["frame"].min()
    max_frame = df["frame"].max()

    nan_column = pd.Series([np.nan] * (max_frame - min_frame + 1))

    frame = pd.Series(np.arange(min_frame, max_frame + 1))
    time = (frame / 15).round(6)

    offense_ids = sorted(df.loc[df["class"] == "offense", "id"].unique())
    if team == "Home":
        player_ids = offense_ids
    else:
        # For Away team, use defense players closest to each offense player
        player_ids = []
        for offense_id in offense_ids:
            closest_defense = (
                df.loc[
                    (df["class"] == "offense") & (df["id"] == offense_id), "closest"
                ].iloc[0]
                if len(df.loc[(df["class"] == "offense") & (df["id"] == offense_id)])
                > 0
                else None
            )
            if closest_defense is not None:
                player_ids.append(closest_defense)

    positions = []
    for i, player_id in enumerate(player_ids[:7]):  # Limit to 7 players
        if team == "Home":
            player_df = df[(df["id"] == player_id) & (df["class"] == "offense")]
        else:
            player_df = df[(df["id"] == player_id) & (df["class"] == "defense")]

        if not player_df.empty:
            x = player_df["x"].round(2).reset_index(drop=True)
            y = player_df["y"].round(2).reset_index(drop=True)
        else:
            x = pd.Series([np.nan] * len(frame))
            y = pd.Series([np.nan] * len(frame))

        positions.append(x)
        positions.append(y)

    # Add remaining player columns if less than 7 players
    while len(positions) < 14:
        positions.append(pd.Series([np.nan] * len(frame)))

    disc_x = df.loc[df["class"] == "disc", "x"].round(2).reset_index(drop=True)
    disc_y = df.loc[df["class"] == "disc", "y"].round(2).reset_index(drop=True)
    positions.append(disc_x)
    positions.append(disc_y)

    positions_df = pd.concat(positions, axis=1)

    selected = nan_column.copy()
    try:
        holder_data = df[df["holder"]]
        if not holder_data.empty:
            for _, holder_row in holder_data.iterrows():
                frame_idx = holder_row["frame"] - min_frame
                if holder_row["id"] in offense_ids:
                    selected.iloc[frame_idx] = offense_ids.index(holder_row["id"])
    except Exception:
        pass
    selected_df = pd.DataFrame(selected).reset_index(drop=True)

    tracking_df = pd.concat(
        [nan_column, frame, time, positions_df, selected_df], axis=1
    )
    tracking_df.columns = multi_columns

    return tracking_df


def convert_to_metrica_format(intermediate_df):
    """
    Convert intermediate data to Metrica format

    Args:
        intermediate_df: DataFrame with intermediate format

    Returns:
        Tuple of (events_df, home_df, away_df): Metrica format DataFrames
    """
    # Create the Metrica DataFrame for events
    events_df = create_events_metrica(intermediate_df)

    # Create the Metrica DataFrame for Home and Away
    home_df = create_tracking_metrica(intermediate_df, "Home")
    away_df = create_tracking_metrica(intermediate_df, "Away")

    # Drop non-data columns
    events_df.dropna(subset=["Start Frame"], inplace=True)
    home_df.dropna(subset=[("", "", "Frame")], inplace=True)
    away_df.dropna(subset=[("", "", "Frame")], inplace=True)

    return events_df, home_df, away_df


def process_and_convert_to_metrica(
    game_id, data_path, save_folder_path=None, save_intermediate=False
):
    """
    Complete pipeline: process Ultimate Track data -> create intermediate file -> convert to Metrica format

    Args:
        game_id: Game identifier (file index)
        data_path: Path to data directory containing CSV files
        save_folder_path: Path to save output files (optional)
        save_intermediate: Whether to save intermediate file

    Returns:
        Tuple of (intermediate_df, events_df, home_df, away_df): DataFrames
    """

    def get_csv_files(data_path):
        """Get list of CSV files in data directory"""
        csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        csv_files.sort()
        return csv_files

    # Get list of CSV files
    csv_files = get_csv_files(data_path)

    if game_id >= len(csv_files):
        raise ValueError(
            f"Game ID {game_id} out of range. Available files: {len(csv_files)}"
        )

    # Load the specified CSV file
    file_path = os.path.join(data_path, csv_files[game_id])
    print(f"Loading Ultimate Track data from: {file_path}")

    raw_data = pd.read_csv(file_path)

    # Validate required columns
    required_columns = [
        "frame",
        "id",
        "class",
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "closest",
        "holder",
    ]
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Processing {len(raw_data)} rows from {csv_files[game_id]}")

    # Step 1: Create intermediate file with all required columns
    print("Creating intermediate file with calculated features...")
    intermediate_df = create_intermediate_file(raw_data)

    # Step 2: Convert to Metrica format
    print("Converting to Metrica format...")
    events_df, home_df, away_df = convert_to_metrica_format(intermediate_df)

    # Save results if path is provided
    if save_folder_path:
        # Create output directory
        os.makedirs(save_folder_path, exist_ok=True)

        base_name = csv_files[game_id][:-4]  # Remove .csv extension

        if save_intermediate:
            intermediate_path = f"{save_folder_path}/{base_name}_intermediate.csv"
            intermediate_df.to_csv(intermediate_path, index=False)
            print(f"Saved intermediate file: {intermediate_path}")

        # Save Metrica format files
        events_path = f"{save_folder_path}/{base_name}_events.csv"
        home_path = f"{save_folder_path}/{base_name}_Home.csv"
        away_path = f"{save_folder_path}/{base_name}_Away.csv"

        events_df.to_csv(events_path, index=False)
        home_df.to_csv(home_path, index=False)
        away_df.to_csv(away_path, index=False)

        print("Saved Metrica files:")
        print(f"  Events: {events_path}")
        print(f"  Home: {home_path}")
        print(f"  Away: {away_path}")

    print("Processing completed!")
    print(f"Intermediate data shape: {intermediate_df.shape}")
    print(f"Events data shape: {events_df.shape}")
    print(f"Home tracking shape: {home_df.shape}")
    print(f"Away tracking shape: {away_df.shape}")

    return intermediate_df, events_df, home_df, away_df


def preprocessing_for_ultimatetrack(game_id, data_path):
    """
    Preprocessing function specifically for UltimateTrack data provider

    Args:
        game_id: Game identifier (file index)
        data_path: Path to data directory containing CSV files

    Returns:
        Tuple of (home_df, away_df, events_df): DataFrames
    """
    intermediate_df, events_df, home_df, away_df = process_and_convert_to_metrica(
        game_id, data_path
    )
    return home_df, away_df, events_df


def preprocessing_for_ufa(game_id, data_path):
    """
    Preprocessing function specifically for UFA data provider

    Args:
        game_id: Game identifier (file index)
        data_path: Path to data directory containing CSV files

    Returns:
        Tuple of (home_df, away_df, events_df): DataFrames
    """

    def get_csv_files(data_path):
        """Get list of CSV files in data directory"""
        csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        csv_files.sort()
        return csv_files

    # Get list of CSV files
    csv_files = get_csv_files(data_path)

    if game_id >= len(csv_files):
        raise ValueError(
            f"Game ID {game_id} out of range. Available files: {len(csv_files)}"
        )

    # Load the specified CSV file
    file_path = os.path.join(data_path, csv_files[game_id])
    print(f"Loading UFA data from: {file_path}")

    raw_data = pd.read_csv(file_path)

    # UFAデータから不要な列を削除
    columns_to_remove = ["selected", "prev_holder", "def_selected"]
    existing_columns_to_remove = [
        col for col in columns_to_remove if col in raw_data.columns
    ]

    if existing_columns_to_remove:
        print(f"Removing columns: {existing_columns_to_remove}")
        processed_data = raw_data.drop(columns=existing_columns_to_remove)
    else:
        processed_data = raw_data.copy()
        print("No columns to remove from UFA data")

    # UFAデータ（中間ファイル形式）からMetrica形式に変換
    events_df, home_df, away_df = convert_to_metrica_format(processed_data)

    return home_df, away_df, events_df

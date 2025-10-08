import numpy as np
import pandas as pd

from .preprocess_config import (
    COLUMNS_TO_REMOVE,
    COORDINATE_SCALE,
    PLAYERS_PER_TEAM,
    TRACKING_HERZ,
)


def preprocessing_for_ufa(data_path):
    """
    Preprocessing function specifically for UFA data provider

    Args:
        data_path: Path to the UFA data file (CSV/TXT format)

    Returns:
        Tuple of (home_df, away_df, events_df): DataFrames in Metrica format
            - home_df: Home team tracking data with MultiIndex columns
            - away_df: Away team tracking data with MultiIndex columns
            - events_df: Events data with disc position and holder information
    """

    # Load UFA data
    raw_data = pd.read_csv(data_path)

    # Remove unnecessary columns from UFA data
    existing_columns_to_remove = [
        col for col in COLUMNS_TO_REMOVE if col in raw_data.columns
    ]

    if existing_columns_to_remove:
        print(f"Removing columns: {existing_columns_to_remove}")
        processed_data = raw_data.drop(columns=existing_columns_to_remove)
    else:
        processed_data = raw_data.copy()
        print("No columns to remove from UFA data")

    # Apply coordinate scaling using config values
    if "x" in processed_data.columns and "y" in processed_data.columns:
        processed_data["x"] = processed_data["x"] * COORDINATE_SCALE
        processed_data["y"] = processed_data["y"] * COORDINATE_SCALE

    # Convert UFA data (intermediate file format) to Metrica format
    home_df, away_df, events_df = convert_to_metrica_format(processed_data)

    return home_df, away_df, events_df


def convert_to_metrica_format(intermediate_df):
    """
    Convert Ultimate Track intermediate data to Metrica format

    Args:
        intermediate_df: DataFrame with intermediate format containing calculated motion features

    Returns:
        Tuple of (home_df, away_df, events_df): Metrica format DataFrames
            - home_df: Home team tracking data with MultiIndex columns
            - away_df: Away team tracking data with MultiIndex columns
            - events_df: Events data with disc position and holder information
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

    return home_df, away_df, events_df


def create_events_metrica(df):
    """
    Create the Metrica format DataFrame for events from UFA data

    Args:
        df (DataFrame): The UFA intermediate DataFrame containing tracking data
                       with columns: frame, class, x, y, id, holder

    Returns:
        DataFrame: Events DataFrame in Metrica format with columns:
                  Team, Type, Subtype, Period, Start Frame, Start Time [s],
                  End Frame, End Time [s], From, To, Start X, Start Y, End X, End Y.
                  Contains disc position data and holder information per frame.
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
    start_time = (start_frame / TRACKING_HERZ).round(6)
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
    Create the Metrica format DataFrame for team tracking data from UFA data

    Args:
        df (DataFrame): The UFA intermediate DataFrame containing tracking data
                       with columns: frame, class, x, y, id, closest
        team (str): Team designation ("Home" for offense, "Away" for defense)

    Returns:
        DataFrame: Tracking DataFrame in Metrica format with MultiIndex columns:
                  - Level 0: "" for general columns, team name for player columns
                  - Level 1: Player indices for player columns
                  - Level 2: "Period", "Frame", "Time [s]", player position names, "Disc__"
                  Contains position data for up to 7 players plus disc position.
    """
    # Define the levels of the MultiIndex using config values
    player_columns = PLAYERS_PER_TEAM * 2  # x, y for each player
    level_0 = [""] * 3 + [team] * player_columns + [""] * 2
    level_1 = [""] * 3 + [i // 2 for i in range(player_columns)] + [""] * 2

    # Generate player column names using config
    player_names = []
    for i in range(PLAYERS_PER_TEAM):
        player_names.extend([f"Player{i}", f"Player{i}"])

    level_2 = (
        [
            "Period",
            "Frame",
            "Time [s]",
        ]
        + player_names
        + [
            "Disc__",
            "Disc__",
        ]
    )

    # Create the MultiIndex
    multi_columns = pd.MultiIndex.from_arrays([level_0, level_1, level_2])

    min_frame = df["frame"].min()
    max_frame = df["frame"].max()

    nan_column = pd.Series([np.nan] * (max_frame - min_frame + 1))

    frame = pd.Series(np.arange(min_frame, max_frame + 1))
    time = (frame / TRACKING_HERZ).round(6)

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
    for i, player_id in enumerate(player_ids[:PLAYERS_PER_TEAM]):  # Limit to 7 players
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
    while len(positions) < PLAYERS_PER_TEAM * 2:
        positions.append(pd.Series([np.nan] * len(frame)))

    disc_x = df.loc[df["class"] == "disc", "x"].round(2).reset_index(drop=True)
    disc_y = df.loc[df["class"] == "disc", "y"].round(2).reset_index(drop=True)
    positions.append(disc_x)
    positions.append(disc_y)

    positions_df = pd.concat(positions, axis=1)

    tracking_df = pd.concat([nan_column, frame, time, positions_df], axis=1)
    tracking_df.columns = multi_columns

    return tracking_df

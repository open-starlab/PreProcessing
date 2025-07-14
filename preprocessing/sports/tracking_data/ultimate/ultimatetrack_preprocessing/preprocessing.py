import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocess_config import FIELD_LENGTH, FIELD_WIDTH, TRACKING_HERZ


def process_tracking_data(game_id, data_path, test=False):
    """
    Process Ultimate Track tracking data

    Args:
        game_id: Game identifier (file index)
        data_path: Path to data directory containing CSV files
        test: Whether this is a test run

    Returns:
        tracking_offense: DataFrame with offense team tracking data
        tracking_defense: DataFrame with defense team tracking data
        team_info_df: DataFrame with team composition info
    """

    def get_csv_files(data_path):
        """Get list of CSV files in data directory"""
        csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        csv_files.sort()
        return csv_files

    def create_tracking_dataframe(frames, num_players, team_type):
        """Create tracking dataframe structure for Ultimate Track"""
        columns = ["frame", "period", "Time [s]"]

        # Add disc coordinates
        columns.extend(["disc_x", "disc_y"])

        # Add player coordinates
        for i in range(1, num_players + 1):
            columns.extend([f"{team_type}_{i}_x", f"{team_type}_{i}_y"])

        # Add velocity columns
        columns.extend(["disc_vx", "disc_vy"])
        for i in range(1, num_players + 1):
            columns.extend([f"{team_type}_{i}_vx", f"{team_type}_{i}_vy"])

        # Add acceleration columns
        columns.extend(["disc_ax", "disc_ay"])
        for i in range(1, num_players + 1):
            columns.extend([f"{team_type}_{i}_ax", f"{team_type}_{i}_ay"])

        df = pd.DataFrame(columns=columns)
        df = df.reindex(range(len(frames)), fill_value=np.nan)
        df["frame"] = frames
        df["period"] = 1  # Ultimate is typically one continuous period

        # Calculate time based on 15fps frame rate
        df["Time [s]"] = (df["frame"] - df["frame"].min()) / TRACKING_HERZ

        return df

    def fill_tracking_data(tracking_df, raw_data, team_class, team_type):
        """Fill tracking dataframe with actual data"""
        team_data = raw_data[raw_data["class"] == team_class].copy()
        disc_data = raw_data[raw_data["class"] == "disc"].copy()

        frames = tracking_df["frame"].unique()

        for idx, frame in enumerate(tqdm(frames, desc=f"Processing {team_type} team")):
            frame_team_data = team_data[team_data["frame"] == frame]
            frame_disc_data = disc_data[disc_data["frame"] == frame]

            # Fill disc data
            if not frame_disc_data.empty:
                disc_row = frame_disc_data.iloc[0]
                tracking_df.loc[idx, "disc_x"] = disc_row["x"]
                tracking_df.loc[idx, "disc_y"] = disc_row["y"]
                tracking_df.loc[idx, "disc_vx"] = disc_row["vx"]
                tracking_df.loc[idx, "disc_vy"] = disc_row["vy"]
                tracking_df.loc[idx, "disc_ax"] = disc_row["ax"]
                tracking_df.loc[idx, "disc_ay"] = disc_row["ay"]

            # Fill player data
            player_positions = {}
            for _, player_row in frame_team_data.iterrows():
                player_id = player_row["id"]
                if player_id not in player_positions:
                    player_positions[player_id] = len(player_positions) + 1

                player_num = player_positions[player_id]
                if player_num <= 7:  # Ultimate typically has 7 players per team
                    tracking_df.loc[idx, f"{team_type}_{player_num}_x"] = player_row[
                        "x"
                    ]
                    tracking_df.loc[idx, f"{team_type}_{player_num}_y"] = player_row[
                        "y"
                    ]
                    tracking_df.loc[idx, f"{team_type}_{player_num}_vx"] = player_row[
                        "vx"
                    ]
                    tracking_df.loc[idx, f"{team_type}_{player_num}_vy"] = player_row[
                        "vy"
                    ]
                    tracking_df.loc[idx, f"{team_type}_{player_num}_ax"] = player_row[
                        "ax"
                    ]
                    tracking_df.loc[idx, f"{team_type}_{player_num}_ay"] = player_row[
                        "ay"
                    ]

        return tracking_df

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

    # Validate field dimensions (Ultimate Track: 94m x 37m)
    print(f"Field dimensions - Length: {FIELD_LENGTH}m, Width: {FIELD_WIDTH}m")
    x_range = raw_data["x"].max() - raw_data["x"].min()
    y_range = raw_data["y"].max() - raw_data["y"].min()
    print(f"Data coordinate range - X: {x_range:.1f}m, Y: {y_range:.1f}m")

    if x_range > FIELD_LENGTH * 1.2 or y_range > FIELD_WIDTH * 1.2:
        print("Warning: Coordinate range exceeds expected field dimensions")

    # Get unique frames
    frames = sorted(raw_data["frame"].unique())
    print(f"Processing {len(frames)} frames at {TRACKING_HERZ}fps")

    # Get team information
    offense_players = raw_data[raw_data["class"] == "offense"]["id"].unique()
    defense_players = raw_data[raw_data["class"] == "defense"]["id"].unique()

    print(f"Offense players: {len(offense_players)}")
    print(f"Defense players: {len(defense_players)}")

    # Create tracking dataframes
    tracking_offense = create_tracking_dataframe(frames, 7, "offense")
    tracking_defense = create_tracking_dataframe(frames, 7, "defense")

    # Create disc tracking dataframe
    disc_columns = [
        "frame",
        "period",
        "Time [s]",
        "disc_x",
        "disc_y",
        "disc_vx",
        "disc_vy",
        "disc_ax",
        "disc_ay",
    ]
    tracking_disc = pd.DataFrame(columns=disc_columns)
    tracking_disc = tracking_disc.reindex(range(len(frames)), fill_value=np.nan)
    tracking_disc["frame"] = frames
    tracking_disc["period"] = 1
    tracking_disc["Time [s]"] = (
        tracking_disc["frame"] - tracking_disc["frame"].min()
    ) / TRACKING_HERZ

    # Fill with actual data
    tracking_offense = fill_tracking_data(
        tracking_offense, raw_data, "offense", "offense"
    )
    tracking_defense = fill_tracking_data(
        tracking_defense, raw_data, "defense", "defense"
    )

    # Fill disc data
    disc_data = raw_data[raw_data["class"] == "disc"].copy()
    for idx, frame in enumerate(frames):
        frame_disc_data = disc_data[disc_data["frame"] == frame]
        if not frame_disc_data.empty:
            disc_row = frame_disc_data.iloc[0]
            tracking_disc.loc[idx, "disc_x"] = disc_row["x"]
            tracking_disc.loc[idx, "disc_y"] = disc_row["y"]
            tracking_disc.loc[idx, "disc_vx"] = disc_row["vx"]
            tracking_disc.loc[idx, "disc_vy"] = disc_row["vy"]
            tracking_disc.loc[idx, "disc_ax"] = disc_row["ax"]
            tracking_disc.loc[idx, "disc_ay"] = disc_row["ay"]

    # Create team info dataframe
    team_info_data = {
        "offense_players": [list(offense_players[:7])],  # Limit to 7 players
        "defense_players": [list(defense_players[:7])],  # Limit to 7 players
        "total_frames": [len(frames)],
        "file_name": [csv_files[game_id]],
    }
    team_info_df = pd.DataFrame(team_info_data)

    print("Ultimate Track data processing completed!")

    return tracking_offense, tracking_defense, tracking_disc, team_info_df


def analyze_possession_patterns(raw_data):
    """
    Analyze disc possession patterns

    Args:
        raw_data: Raw Ultimate Track data

    Returns:
        possession_stats: DataFrame with possession statistics
    """
    possession_data = []

    for frame in raw_data["frame"].unique():
        frame_data = raw_data[raw_data["frame"] == frame]
        holder_data = frame_data[frame_data["holder"]]

        if not holder_data.empty:
            holder = holder_data.iloc[0]
            possession_data.append(
                {
                    "frame": frame,
                    "holder_id": holder["id"],
                    "holder_team": holder["class"],
                    "holder_x": holder["x"],
                    "holder_y": holder["y"],
                    "holder_speed": np.sqrt(holder["vx"] ** 2 + holder["vy"] ** 2),
                }
            )

    possession_df = pd.DataFrame(possession_data)

    # Calculate possession statistics
    if not possession_df.empty:
        possession_stats = (
            possession_df.groupby("holder_team")
            .agg(
                {
                    "frame": "count",
                    "holder_speed": "mean",
                    "holder_x": "mean",
                    "holder_y": "mean",
                }
            )
            .rename(columns={"frame": "possession_count"})
        )
    else:
        possession_stats = pd.DataFrame()

    return possession_stats


def calculate_team_metrics(tracking_data, team_type):
    """
    Calculate team-level metrics

    Args:
        tracking_data: Tracking data for a team
        team_type: 'offense' or 'defense'

    Returns:
        metrics: Dictionary of team metrics
    """
    metrics = {}

    # Player position columns
    player_x_cols = [
        col for col in tracking_data.columns if f"{team_type}_" in col and "_x" in col
    ]
    player_y_cols = [
        col for col in tracking_data.columns if f"{team_type}_" in col and "_y" in col
    ]

    if player_x_cols and player_y_cols:
        # Calculate centroid
        metrics["centroid_x"] = tracking_data[player_x_cols].mean(axis=1).mean()
        metrics["centroid_y"] = tracking_data[player_y_cols].mean(axis=1).mean()

        # Calculate spread
        metrics["spread_x"] = tracking_data[player_x_cols].std(axis=1).mean()
        metrics["spread_y"] = tracking_data[player_y_cols].std(axis=1).mean()

        # Calculate team compactness
        player_positions = tracking_data[player_x_cols + player_y_cols].values
        metrics["compactness"] = np.nanstd(player_positions)

    return metrics


if __name__ == "__main__":
    import os

    # Test with Ultimate Track data
    game_id = 0  # Select the first CSV file
    data_path = os.getcwd() + "/test/sports/tracking_data/data/ultimatetrack/"

    # Create test directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    try:
        # Call the function
        tracking_offense, tracking_defense, tracking_disc, team_info_df = (
            process_tracking_data(game_id, data_path, test=True)
        )

        # Save results
        output_dir = os.getcwd() + "/test/sports/tracking_data/data/ultimatetrack/"
        tracking_offense.to_csv(output_dir + "test_tracking_offense.csv", index=False)
        tracking_defense.to_csv(output_dir + "test_tracking_defense.csv", index=False)
        tracking_disc.to_csv(output_dir + "test_tracking_disc.csv", index=False)
        team_info_df.to_csv(output_dir + "test_team_info.csv", index=False)

        print("Test completed successfully!")
        print(f"Offense tracking shape: {tracking_offense.shape}")
        print(f"Defense tracking shape: {tracking_defense.shape}")
        print(f"Disc tracking shape: {tracking_disc.shape}")
        print(f"Team info shape: {team_info_df.shape}")

    except Exception as e:
        print(f"Test failed: {e}")

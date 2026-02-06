import itertools

import numpy as np


def circular_average(angles):
    """
    Calculate the circular average of a list of angles.

    Args:
        angles (list): List of angles in degrees.

    Returns:
        float: Circular average angle in degrees.
    """
    # Convert angles to radians
    radians = np.deg2rad(angles)

    # Calculate the average of the sine and cosine values
    sin_average = np.mean(np.sin(radians))
    cos_average = np.mean(np.cos(radians))

    # Compute the average angle in radians
    average_radian = np.arctan2(sin_average, cos_average)

    # Convert back to degrees
    average_angle = np.rad2deg(average_radian)

    # Normalize the angle to be within [0, 360)
    average_angle = average_angle % 360

    return average_angle


def is_within_forward_direction(target_direction, target_x, target_y, other_x, other_y):
    """
    Check if another object is within the forward direction of the target object.

    Args:
        target_direction (float): Direction angle of the target object.
        target_x (float): X-coordinate of the target object.
        target_y (float): Y-coordinate of the target object.
        other_x (float): X-coordinate of the other object.
        other_y (float): Y-coordinate of the other object.

    Returns:
        bool: True if the other object is within the forward direction, False otherwise.
    """
    # Calculate the angle between the target and other object
    angle = np.arctan2(other_y - target_y, other_x - target_x) * (180 / np.pi)
    # Calculate the relative angle to the target's direction
    relative_angle = (angle - target_direction + 360) % 360
    # Check if the relative angle is within the forward direction range
    check = 0 <= relative_angle <= 45 or 315 <= relative_angle < 360
    return check


def add_selected_column(data, velocity_threshold, acceleration_threshold):
    """
    Add a selected column to the DataFrame based on velocity and acceleration thresholds.

    Args:
        data (pd.DataFrame): DataFrame containing object data.
        velocity_threshold (float): Threshold for velocity.
        acceleration_threshold (float): Threshold for acceleration.

    Returns:
        pd.DataFrame: DataFrame with added selected column.
    """
    # Initialize the 'selected' column with False
    data["selected"] = False

    # Add 'prev_holder' column by shifting the 'holder' column by 30 frames for each id
    data["prev_holder"] = data.groupby("id")["holder"].shift(30, fill_value=False)

    # Define the selection criteria
    selection_criteria = (
        # The current frame must be at least 15
        (data["frame"] > 1)
        &
        # The object must be of class 'offense'
        (data["class"] == "offense")
        &
        # The object should not be a holder
        (~data["holder"])
        &
        # The object should not have been a holder 30 frames ago
        (~data["prev_holder"])
        &
        # The object's acceleration magnitude must be greater than the threshold
        (data["a_mag"] > acceleration_threshold)
        &
        # The difference of the velocity angle and acceleration angle must be less than 90
        (abs((data["v_angle"] - data["a_angle"] + 180) % 360 - 180) < 90)
    )

    # Apply the selection criteria
    data.loc[selection_criteria, "selected"] = True

    # Process each id
    for id_val in data["id"].unique():
        # Get data for each id
        id_data = data[data["id"] == id_val]
        # Save the series of velocity angles
        v_angles = []

        # Iterate over frames
        for frame in range(id_data["frame"].min(), id_data["frame"].max() + 1):
            # Skip if the current frame is not selected
            if not id_data.loc[id_data["frame"] == frame, "selected"].values[0]:
                # Reset the velocity angles list
                v_angles = []
                continue
            next_frame = frame + 1
            next_frame_data = id_data[id_data["frame"] == next_frame]
            v_angles.append(id_data[id_data["frame"] == frame]["v_angle"].values[0])

            # Check if next_frame_data is not empty
            if next_frame_data.empty:
                continue

            # Calculate the difference in angles and convert it to degrees
            angle_diff = abs(
                next_frame_data["v_angle"].values[0] - circular_average(v_angles)
            )

            # Ensure the angle difference is within the range [0, 180]
            angle_diff = min(angle_diff, abs(360 - angle_diff))

            # Check if the next frame meets the criteria
            if (
                # The difference in velocity angle must be <= 20
                next_frame_data["diff_v_angle"].values[0] <= 20
                and
                # The velocity magnitude must be > 3
                next_frame_data["v_mag"].values[0] > 3
                and
                # The object should not be a holder
                not next_frame_data["holder"].values[0]
                and
                # The object's velocity angle must be within the circular median of the previous angles
                angle_diff <= 90
            ):
                data.loc[
                    (data["id"] == id_val) & (data["frame"] == next_frame), "selected"
                ] = True
                id_data.loc[id_data["frame"] == next_frame, "selected"] = True
            else:
                data.loc[
                    (data["id"] == id_val) & (data["frame"] == next_frame), "selected"
                ] = False
                id_data.loc[id_data["frame"] == next_frame, "selected"] = False

    return data


def deselect_based_on_proximity_and_direction(
    data, distance_threshold, player_threshold
):
    """
    Deselect rows in the DataFrame based on proximity and direction criteria.

    Args:
        data (pd.DataFrame): DataFrame containing object data.
        distance_threshold (float): Threshold for distance.
        player_threshold (int): Threshold for the number of players.

    Returns:
        pd.DataFrame: DataFrame with updated 'selected' column.
    """
    # Iterate over each unique id in the selected data
    for id_val in data["id"].unique():
        selected_frames = data.loc[
            (data["id"] == id_val) & data["selected"], "frame"
        ].values
        continuous_frames_list = [
            list(g)
            for _, g in itertools.groupby(
                selected_frames, key=lambda n, c=itertools.count(): n - next(c)
            )
        ]

        # Process any remaining continuous frames
        if continuous_frames_list:
            for continuous_frames in continuous_frames_list:
                data = process_segment_with_direction(
                    data,
                    continuous_frames,
                    id_val,
                    distance_threshold,
                    player_threshold,
                )

    return data


def process_segment_with_direction(
    data, continuous_frames, id, distance_threshold, player_threshold
):
    """
    Process a segment of continuous frames to deselect rows based on proximity and direction criteria.

    Args:
        data (pd.DataFrame): DataFrame containing object data.
        continuous_frames (list): List of continuous frame numbers.
        id: Player ID.
        distance_threshold (float): Threshold for distance.
        player_threshold (int): Threshold for the number of players.

    Returns:
        pd.DataFrame: DataFrame with updated 'selected' column.
    """
    # Get the data for the last frame in the segment
    last_frame = continuous_frames[-1]
    last_frame_data = data[data["frame"] == last_frame]

    # Iterate over each row in the last frame data
    for _, target_row in last_frame_data.iterrows():
        if target_row["id"] == id:
            count = 0
            direction_count = 0
            # Compare the target row with other rows in the last frame
            for _, other_row in last_frame_data.iterrows():
                if (
                    other_row["id"] != target_row["id"]
                    and other_row["class"] == "offense"
                ):
                    # Calculate the distance between the target and other row
                    distance = np.sqrt(
                        (target_row["x"] - other_row["x"]) ** 2
                        + (target_row["y"] - other_row["y"]) ** 2
                    )
                    if distance <= distance_threshold:
                        count += 1
                    # Check if the other row is within the forward direction of the target row
                    if is_within_forward_direction(
                        target_row["v_angle"],
                        target_row["x"],
                        target_row["y"],
                        other_row["x"],
                        other_row["y"],
                    ):
                        direction_count += 1

            # Deselect the target row if it meets the criteria
            if count >= player_threshold or direction_count >= 2:
                data.loc[
                    (data["id"] == target_row["id"])
                    & (data["frame"].isin(continuous_frames)),
                    "selected",
                ] = False

    return data


def deselect_based_on_length(data):
    """
    Deselect movements with less than 15 frames or more than 75 frames.

    Args:
        data (pd.DataFrame): DataFrame containing object data.

    Returns:
        pd.DataFrame: DataFrame with updated 'selected' column.
    """
    # Get the offense ID
    offense_ids = data[data["selected"]]["id"].unique()

    for offense_id in offense_ids:
        # Get frames of the selected offense ID
        all_subject_frames = data[data["selected"] & (data["id"] == offense_id)][
            "frame"
        ].unique()

        # Split the frames into continuous sequences
        subject_frames_list = [
            list(g)
            for _, g in itertools.groupby(
                all_subject_frames, key=lambda n, c=itertools.count(): n - next(c)
            )
        ]

        # Deselect movements with less than 15 frames
        for subject_frames in subject_frames_list:
            if (len(subject_frames) < 15) or (len(subject_frames) > 75):
                data.loc[
                    (data["frame"].isin(subject_frames)) & (data["id"] == offense_id),
                    "selected",
                ] = False

    return data


def add_defense_selected_column(data):
    """
    Add a 'def_selected' column to the DataFrame based on the closest defense player.

    Args:
        data (pd.DataFrame): DataFrame containing object data.

    Returns:
        pd.DataFrame: DataFrame with added 'def_selected' column.
    """
    # Initialize the 'def_selected' column with False
    data["def_selected"] = False

    # Process each frame
    grouped = data.groupby("frame")

    for frame, group in grouped:
        # Get the selected rows for the current frame
        selected_rows = group[group["selected"]]

        # Get the ids of the selected rows
        closest_defense_ids = selected_rows["closest"].values

        # Set the 'def_selected' column to True for the closest defense ids
        data.loc[
            (data["frame"] == frame) & (data["id"].isin(closest_defense_ids)),
            "def_selected",
        ] = True

    return data


def detect_play(
    data,
    velocity_threshold=3.0,
    acceleration_threshold=4.0,
    distance_threshold=5.0,
    player_threshold=2,
):
    """
    Detect play initiations in the tracking data.

    Args:
        data (pd.DataFrame): DataFrame containing tracking data with required columns.
        velocity_threshold (float): Threshold for velocity (default: 3.0).
        acceleration_threshold (float): Threshold for acceleration (default: 4.0).
        distance_threshold (float): Threshold for distance (default: 5.0).
        player_threshold (int): Threshold for player (default: 2).

    Returns:
        pd.DataFrame: DataFrame with added 'selected' and 'def_selected' columns.
    """
    # Add a selected column to the data
    data = add_selected_column(data, velocity_threshold, acceleration_threshold)

    # Deselect based on proximity
    data = deselect_based_on_proximity_and_direction(
        data, distance_threshold, player_threshold
    )

    # Deselect based on length
    data = deselect_based_on_length(data)

    # Extend selection backwards for continuous velocity
    for id_val in data["id"].unique():
        selected_frames = data.loc[
            (data["id"] == id_val) & data["selected"], "frame"
        ].values
        first_frames = [
            frame for frame in selected_frames if frame - 1 not in selected_frames
        ]
        for frame in first_frames:
            for i in range(frame, 0, -1):
                if data.loc[
                    (data["id"] == id_val) & (data["frame"] == i - 1), "selected"
                ].values[0]:
                    break
                elif (
                    data.loc[
                        (data["id"] == id_val) & (data["frame"] == i - 1), "v_mag"
                    ].values[0]
                    - data.loc[
                        (data["id"] == id_val) & (data["frame"] == i), "v_mag"
                    ].values[0]
                    < 0.05
                    and data.loc[
                        (data["id"] == id_val) & (data["frame"] == i - 1), "v_mag"
                    ].values[0]
                    > 0.05
                    and not data.loc[
                        (data["id"] == id_val) & (data["frame"] == i - 1), "selected"
                    ].values[0]
                ):
                    data.loc[
                        (data["id"] == id_val)
                        & (data["frame"] == i - 1)
                        & (data["frame"] != 0),
                        "selected",
                    ] = True
                else:
                    break

    # Add a selected column for defense to the data
    data = add_defense_selected_column(data)

    return data


def extract_movement(df, offense_id, window=30):
    """
    Extract the movement data of the selected offense ID.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        offense_id (int): The offense ID.
        file_name (str): The file name.
        window (int): The window size to extend the frames (default is 30).

    Returns:
        list: List of extracted DataFrames for each play segment.
    """
    # Get frames of the selected offense ID
    all_subject_frames = df[df["selected"] & (df["id"] == offense_id)]["frame"].unique()

    # Split the frames into continuous sequences
    subject_frames_list = [
        list(g)
        for _, g in itertools.groupby(
            all_subject_frames, key=lambda n, c=itertools.count(): n - next(c)
        )
    ]

    extracted_plays = []

    # Loop through all sequences
    for num, subject_frames in enumerate(subject_frames_list):
        min_frame = 0
        max_frame = df["frame"].max()

        # Extend the frames to include the window
        frames = [
            i
            for i in range(
                max(min_frame, subject_frames[0] - window),
                min(max_frame, subject_frames[-1] + window) + 1,
            )
        ]

        # Filter the data
        df_frames = df[df["frame"].isin(frames)].copy()

        # Get the defense ID
        defense_id_values = df_frames[df_frames["id"] == offense_id]["closest"].unique()
        if len(defense_id_values) > 0:
            defense_id = defense_id_values[0]
        else:
            defense_id = None

        # Exclude players who are not eligible for evaluation
        df_frames.loc[df_frames["id"] != offense_id, "selected"] = False
        if defense_id is not None:
            df_frames.loc[df_frames["id"] != defense_id, "def_selected"] = False
        df_frames.loc[~df_frames["frame"].isin(subject_frames), "selected"] = False
        df_frames.loc[~df_frames["frame"].isin(subject_frames), "def_selected"] = False

        extracted_plays.append(df_frames)

    return extracted_plays


def extract_play(detected_df):
    """
    Extract all plays from detected DataFrame.

    Args:
        detected_df (pd.DataFrame): DataFrame with detected plays (must have 'selected' column).

    Returns:
        dict: Dictionary mapping (file_name, offense_id, play_num) to extracted DataFrames.
    """
    file_name = "detected_play"
    extracted_plays_dict = {}

    # Get offense ID list
    offense_id_list = detected_df[detected_df["class"] == "offense"]["id"].unique()

    # Loop through all offense ID
    for offense_id in offense_id_list:
        # Extract the movement
        plays = extract_movement(detected_df, offense_id)
        for num, play_df in enumerate(plays):
            key = (file_name, int(offense_id), num + 1)
            extracted_plays_dict[key] = play_df

    return extracted_plays_dict

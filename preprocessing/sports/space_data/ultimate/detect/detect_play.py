import argparse
import pandas as pd
import numpy as np
import cv2
import os
import itertools


def arg_parser():
    """
    Parse command-line arguments for selecting a player and saving the data.

    This function parses the following command-line arguments:
    - velocity_threshold: Threshold for velocity (default: 3.0)
    - acceleration_threshold: Threshold for acceleration (default: 4.0)
    - distance_threshold: Threshold for distance (default: 5.0)
    - player_threshold: Threshold for player (default: 2)

    Returns:
        argparse.Namespace: Namespace object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Select player and save the data')
    parser.add_argument('--velocity_threshold', type=float, default=3.0, help='Threshold for velocity')
    parser.add_argument('--acceleration_threshold', type=float, default=4.0, help='Threshold for acceleration')
    parser.add_argument('--distance_threshold', type=float, default=5.0, help='Threshold for distance')
    parser.add_argument('--player_threshold', type=int, default=2, help='Threshold for player')
    return parser.parse_args()


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
    data['selected'] = False

    # Add 'prev_holder' column by shifting the 'holder' column by 30 frames for each id
    data['prev_holder'] = data.groupby('id')['holder'].shift(30)

    # Define the selection criteria
    selection_criteria = (
        # The current frame must be at least 15
        (data['frame'] > 1) &
        # The object must be of class 'offense'
        (data['class'] == 'offense') &
        # The object should not be a holder
        (data['holder'] == False) &
        # The object should not have been a holder 30 frames ago
        (data['prev_holder'] != True) &
        # The object's velocity magnitude must be greater than the threshold
        # (data['v_mag'] > velocity_threshold) &
        # The object's acceleration magnitude must be greater than the threshold
        (data['a_mag'] > acceleration_threshold) &
        # The difference of the velocity angle and acceleration angle must be less than 90
        (abs((data['v_angle'] - data['a_angle'] + 180) % 360 - 180) < 90)
    )

    # Apply the selection criteria
    data.loc[selection_criteria, 'selected'] = True

    # Process each id
    for id_val in data['id'].unique():
        # Get data for each id
        id_data = data[data['id'] == id_val]
        # Save the series of velocity angles
        v_angles = []

        # Iterate over frames
        for frame in range(id_data['frame'].max() + 1):
            # Skip if the current frame is not selected
            if not id_data.loc[id_data['frame'] == frame, 'selected'].values[0]:
                # Reset the velocity angles list
                v_angles = []
                continue
            next_frame = frame + 1
            next_frame_data = id_data[id_data['frame'] == next_frame]
            v_angles.append(id_data[id_data['frame'] == frame]['v_angle'].values[0])

            # Check if next_frame_data is not empty
            if next_frame_data.empty:
                continue

            # Calculate the difference in angles and convert it to degrees
            angle_diff = abs(next_frame_data['v_angle'].values[0] - circular_average(v_angles))

            # Ensure the angle difference is within the range [0, 180]
            angle_diff = min(angle_diff, abs(360 - angle_diff))

            # Check if the next frame meets the criteria
            if (
                # The difference in velocity angle must be <= 20
                next_frame_data['diff_v_angle'].values[0] <= 20 and
                # The velocity magnitude must be > 3
                next_frame_data['v_mag'].values[0] > 3 and
                # The object should not be a holder
                next_frame_data['holder'].values[0] != True and
                # The object's velocity angle must be within the circular median of the previous angles
                angle_diff <= 90
            ):

                data.loc[(data['id'] == id_val) & (data['frame'] == next_frame), 'selected'] = True
                id_data.loc[id_data['frame'] == next_frame, 'selected'] = True
            else:
                data.loc[(data['id'] == id_val) & (data['frame'] == next_frame), 'selected'] = False
                id_data.loc[id_data['frame'] == next_frame, 'selected'] = False

    return data


def circular_average(angles):
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


def deselect_based_on_proximity_and_direction(data, distance_threshold, player_threshold):
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
    for id_val in data['id'].unique():
        selected_frames = data.loc[(data['id'] == id_val) & (data['selected'] == True), 'frame'].values
        continuous_frames_list = [list(g) for _, g in itertools.groupby(selected_frames, key=lambda n, c=itertools.count(): n - next(c))]


        # Process any remaining continuous frames
        if continuous_frames_list:
            for continuous_frames in continuous_frames_list:
                data = process_segment_with_direction(data, continuous_frames, id_val, distance_threshold, player_threshold)

    return data


def process_segment_with_direction(data, continuous_frames, id, distance_threshold, player_threshold):
    """
    Process a segment of continuous frames to deselect rows based on proximity and direction criteria.

    Args:
        data (pd.DataFrame): DataFrame containing object data.
        continuous_frames (list): List of continuous frame numbers.
        distance_threshold (float): Threshold for distance.
        player_threshold (int): Threshold for the number of players.
    """
    # Get the data for the last frame in the segment
    last_frame = continuous_frames[-1]
    last_frame_data = data[data['frame'] == last_frame]

    # Iterate over each row in the last frame data
    for _, target_row in last_frame_data.iterrows():
        if target_row['id'] == id:
            count = 0
            direction_count = 0
            # Compare the target row with other rows in the last frame
            for _, other_row in last_frame_data.iterrows():
                if other_row['id'] != target_row['id'] and other_row['class'] == 'offense':
                    # Calculate the distance between the target and other row
                    distance = np.sqrt((target_row['x_center'] - other_row['x_center'])**2 + (target_row['y_center'] - other_row['y_center'])**2)
                    if distance <= distance_threshold:
                        count += 1
                    # Check if the other row is within the forward direction of the target row
                    if is_within_forward_direction(target_row['v_angle'], target_row['x_center'], target_row['y_center'], other_row['x_center'], other_row['y_center']):
                        direction_count += 1

            # Deselect the target row if it meets the criteria
            if count >= player_threshold or direction_count >= 2:
                data.loc[(data['id'] == target_row['id']) & (data['frame'].isin(continuous_frames)), 'selected'] = False

    return data


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


def deselect_based_on_length(data):
    """
    Deselect movements with less than 15 frames.

    Args:
        data (pd.DataFrame): DataFrame containing object data.

    Returns:
        pd.DataFrame: DataFrame with updated 'selected' column.
    """
    # Get the offense ID
    offense_ids = data[data['selected'] == True]['id'].unique()

    for offense_id in offense_ids:
        # Get frames of the selected offense ID
        all_subject_frames = data[(data['selected'] == True) & (data['id'] == offense_id)]['frame'].unique()

        # Split the frames into continuous sequences
        subject_frames_list = [list(g) for _, g in itertools.groupby(all_subject_frames, key=lambda n, c=itertools.count(): n - next(c))]

        # Deselect movements with less than 15 frames
        for subject_frames in subject_frames_list:
            if (len(subject_frames) < 15) or (len(subject_frames) > 75):
                data.loc[(data['frame'].isin(subject_frames)) & (data['id'] == offense_id), 'selected'] = False

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
    data['def_selected'] = False

    # Process each frame
    grouped = data.groupby('frame')

    for frame, group in grouped:
        # Get the selected rows for the current frame
        selected_rows = group[group['selected'] == True]

        # Get the ids of the selected rows
        closest_defemse_ids = selected_rows['closest_defense'].values

        # Set the 'def_selected' column to True for the closest defense ids
        data.loc[(data['frame'] == frame) & (data['id'].isin(closest_defemse_ids)), 'def_selected'] = True

    return data



def draw_object(field, row, mag):
    """
    Draw an object on the field.

    Args:
        field (np.ndarray): The field image.
        row (pd.Series): The row of data for the object.
        mag (int): Magnification factor.

    Returns:
        None
    """
    # Calculate the position on the field
    x, y = int(row['x_center'] * mag + 100), int(row['y_center'] * mag + 100)
    vx, vy = row['vx'], row['vy']

    if row['class'] == 'disc':
        # Draw a black circle for the disc
        color = (0, 0, 0)
        cv2.circle(field, (x, y), 5, color, -1)
    else:
        if row['selected']:
            color = (0, 255, 255) # yellow
        elif row['def_selected']:
            color = (0, 255, 0) # green
        else:
            color = (255, 0, 0) if row['class'] == 'offense' else (0, 0, 255) # red for offense, blue for defense

        # Draw an arrow for velocity
        cv2.arrowedLine(field, (x, y), (x + int(vx * mag), y + int(vy * mag)), color, 2)
        # Draw a circle at the position
        cv2.circle(field, (x, y), 10, color, -1)

    # Add id as text
    if row['id'] != 15:
        cv2.putText(field, str(row['id']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def plot_on_field(data, output_video_file, field_width: int = 94, field_height: int = 37, mag: int = 20):
    """
    Plot the objects on the field and save as a video.

    Args:
        data (pd.DataFrame): DataFrame containing object data.
        output_video_file (str): Path to the output video file.
        field_width (int): Width of the field.
        field_height (int): Height of the field.
        mag (int): Magnification factor.

    Returns:
        None
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 15
    frame_size = (field_width * mag + 200, field_height * mag + 200)

    # Initialize the video writer
    out = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

    # Get the maximum frame number from the data
    min_frame = data['frame'].min()
    max_frame = data['frame'].max()

    # Iterate through each frame
    for current_frame in range(min_frame, max_frame + 1):
        # Filter the data for the current frame
        frame_data = data[data['frame'] == current_frame]

        # Create a blank field image
        field = np.ones((field_height * mag + 200, field_width * mag + 200, 3), dtype=np.uint8) * 255

        # Draw the field lines
        cv2.rectangle(field, (100, 100), (field_width * mag + 100, field_height * mag + 100), (0, 0, 0), 2)
        cv2.rectangle(field, (100 + 18 * mag, 100), (field_width * mag + 100 - 18 * mag, field_height * mag + 100), (0, 0, 0), 2)

        # Add the current frame number as text
        cv2.putText(field, f"Frame: {current_frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Draw each object in the frame
        for _, row in frame_data.iterrows():
            draw_object(field, row, mag)

        # Write the frame to the video
        out.write(field)

    # Release the video writer and close all OpenCV windows
    out.release()
    cv2.destroyAllWindows()


def main():
    args = arg_parser()
    v_threshold = args.velocity_threshold
    a_threshold = args.acceleration_threshold
    d_threshold = args.distance_threshold
    p_threshold = args.player_threshold

    # Get the list of files in the data directory
    set_dir = './set'
    data_files = os.listdir(set_dir)

    # Create the output directory for position data if it doesn't exist
    output_video_dir = './videos/detected'
    output_data_dir = './detected'
    if os.path.exists(output_video_dir):
        os.system(f'rm -r {output_video_dir}')
    if os.path.exists(output_data_dir):
        os.system(f'rm -r {output_data_dir}')
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    # Iterate through each data file
    for data_file in sorted(data_files):
        # Get the file name from the data file
        file_name = os.path.splitext(data_file)[0]
        print(f"Processing file: {file_name}...", end='')

        data_file = f'{set_dir}/{file_name}.txt'

        # Load the data
        data = pd.read_csv(data_file, sep=',')

        # Add a selected column to the data
        data = add_selected_column(data, v_threshold, a_threshold)

        # Deselect based on proximity
        data = deselect_based_on_proximity_and_direction(data, d_threshold, p_threshold)

        # Deselect based on length
        data = deselect_based_on_length(data)

        # 
        for id_val in data['id'].unique():
            selected_frames = data.loc[(data['id'] == id_val) & (data['selected'] == True), 'frame'].values
            first_frames = [frame for frame in selected_frames if frame - 1 not in selected_frames]
            for frame in first_frames:
                for i in range(frame, 0, -1):
                    if data.loc[(data['id'] == id_val) & (data['frame'] == i-1), 'selected'].values[0] == True:
                        break
                    elif (
                          data.loc[(data['id'] == id_val) & (data['frame'] == i-1), 'v_mag'].values[0] - data.loc[(data['id'] == id_val) & (data['frame'] == i), 'v_mag'].values[0] < 0.05 and
                          data.loc[(data['id'] == id_val) & (data['frame'] == i-1), 'v_mag'].values[0] > 0.05 and
                          data.loc[(data['id'] == id_val) & (data['frame'] == i-1), 'selected'].values[0] == False
                    ):
                        data.loc[(data['id'] == id_val) & (data['frame'] == i-1) & (data['frame'] != 0), 'selected'] = True
                    else:
                        break

        # Add a selected column for defense to the data
        data = add_defense_selected_column(data)

        # Plot the data on the field
        output_video_file = f'{output_video_dir}/{file_name}.mp4'

        plot_on_field(data, output_video_file)

        # Save the data with selected column include header
        output_data_file = f'{output_data_dir}/{file_name}.txt'
        data.to_csv(output_data_file, sep=',', index=False)

        print("Done!")


if __name__ == '__main__':
    main()

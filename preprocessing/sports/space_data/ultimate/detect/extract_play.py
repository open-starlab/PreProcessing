import pandas as pd
import os
import itertools
from tqdm import tqdm

import detect_play as dp


def extract_movement(df, offense_id, file_name, output_data_dir, output_video_dir, window=30):
    '''
    Extract the movement data of the selected offense ID

    Args:
        df (DataFrame): The DataFrame containing the data
        offense_id (int): The offense ID
        file_name (str): The file name
        window (int): The window size to extend the frames (default is 30)

    Returns:
        None
    '''
    # Get frames of the selected offense ID
    all_subject_frames = df[(df['selected'] == True) & (df['id'] == offense_id)]['frame'].unique()

    # Split the frames into continuous sequences
    subject_frames_list = [list(g) for _, g in itertools.groupby(all_subject_frames, key=lambda n, c=itertools.count(): n - next(c))]

    # Loop through all sequences
    for num, subject_frames in enumerate(subject_frames_list):
        min_frame = 0
        max_frame = df['frame'].max()

        # Extend the frames to include the window
        frames = [i for i in range(max(min_frame, subject_frames[0] - window), min(max_frame, subject_frames[-1] + window) + 1)]

        # Filter the data
        df_frames = df[df['frame'].isin(frames)]

        # Get the defense ID
        defense_id = df_frames[df_frames['id'] == offense_id]['closest_defense'].unique()[0]

        # Exclude players who are not eligible for evaluation
        df_frames.loc[df_frames['id'] != offense_id, 'selected'] = False
        df_frames.loc[df_frames['id'] != defense_id, 'def_selected'] = False
        df_frames.loc[~df['frame'].isin(subject_frames), 'selected'] = False
        df_frames.loc[~df['frame'].isin(subject_frames), 'def_selected'] = False

        # Save the data to a new file
        output_data_path = f'{output_data_dir}/{file_name}-{int(offense_id)}_{num+1}.txt'
        df_frames.to_csv(output_data_path, sep=',', index=False)

        # Save the video to a new file
        output_video_path = f'{output_video_dir}/{file_name}-{int(offense_id)}_{num+1}.mp4'
        dp.plot_on_field(df_frames, output_video_path)

def main():
    # List of files in the folder
    detected_dir = './detected'
    files = os.listdir(detected_dir)

    # Create the output directory for position data if it doesn't exist
    output_data_dir = './play'
    output_video_dir = './videos/play'
    if os.path.exists(output_data_dir):
        os.system(f'rm -rf {output_data_dir}')
    if os.path.exists(output_video_dir):
        os.system(f'rm -rf {output_video_dir}')
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir, exist_ok=True)
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir, exist_ok=True)

    # Calculate the total number of tasks
    total_tasks = len(files) * 7

    # Initialize the progress bar
    with tqdm(total=total_tasks, desc="Processing files and offense IDs") as pbar:
        # Loop through all files
        for file in sorted(files):
            file_name = os.path.splitext(file)[0]
            # Read the file
            df = pd.read_csv(f'{detected_dir}/{file}')

            # Get offense ID list
            offense_id_list = df[df['class'] == 'offense']['id'].unique()

            # Loop through all offense ID
            for offense_id in offense_id_list:
                # Extract the movement
                extract_movement(df, offense_id, file_name, output_data_dir, output_video_dir)
                # Update the progress bar
                pbar.update(1)


if __name__ == '__main__':
    main()
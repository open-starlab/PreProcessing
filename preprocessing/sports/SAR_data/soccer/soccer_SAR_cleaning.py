import argparse
import logging
import re
import time
import pandas as pd
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import sys
import os

if __name__ == "__main__":
    from constant import HOME_AWAY_MAP
    from cleaning.clean_event_data import (
        clean_event_data,
        get_changed_player_list,
        get_timestamp,
        preprocess_coordinates_in_event_data,
    )
    from cleaning.clean_data import (
        clean_player_data, 
        merge_tracking_and_event_data,
        split_tracking_data,
        adjust_player_roles
    )
    from cleaning.clean_tracking_data import (
        calculate_speed,
        calculate_acceleration,
        clean_tracking_data,
        complement_tracking_ball_with_event_data,
        cut_frames_out_of_game,
        format_tracking_data,
        get_player_change_log,
        interpolate_ball_tracking_data,
        merge_tracking_data,
        pad_players_and_interpolate_tracking_data,
        preprocess_coordinates_in_tracking_data,
        resample_tracking_data,
    )
    from cleaning.map_column_names import (
        check_and_rename_event_columns,
        check_and_rename_player_columns,
        check_and_rename_tracking_columns,
    )
    from env import DATA_DIR
    from utils.file_utils import load_json, safe_pd_read_csv, save_as_jsonlines, save_formatted_json
else:
    from .constant import HOME_AWAY_MAP
    from .cleaning.clean_event_data import (
        clean_event_data,
        get_changed_player_list,
        get_timestamp,
        preprocess_coordinates_in_event_data,
    )
    from .cleaning.clean_data import (
        clean_player_data, 
        merge_tracking_and_event_data,
        split_tracking_data,
        adjust_player_roles
    )
    from .cleaning.clean_tracking_data import (
        calculate_speed,
        calculate_acceleration,
        clean_tracking_data,
        complement_tracking_ball_with_event_data,
        cut_frames_out_of_game,
        format_tracking_data,
        get_player_change_log,
        interpolate_ball_tracking_data,
        merge_tracking_data,
        pad_players_and_interpolate_tracking_data,
        preprocess_coordinates_in_tracking_data,
        resample_tracking_data,
    )
    from .cleaning.map_column_names import (
        check_and_rename_event_columns,
        check_and_rename_player_columns,
        check_and_rename_tracking_columns,
    )
    from .env import DATA_DIR
    from .utils.file_utils import load_json, safe_pd_read_csv, save_as_jsonlines, save_formatted_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def clean_data(game_dir, config, args):
    save_dir = args.cleaned_data_dir / game_dir.name
    if save_dir.exists():
        logger.info(f"cleaning started... {game_dir.name}")
        start_time = time.time()
        # event_data / time_stamp
        
        event_data = safe_pd_read_csv(game_dir / config["event_filename"])
        event_data = check_and_rename_event_columns(event_data, config['event_columns_mapping'], args.state_def, args.league)
        event_data['home_away'] = event_data['home_away'].apply(lambda x: HOME_AWAY_MAP[x])
        event_data['half'] = event_data['match_status_id'].apply(lambda x: "first" if x == 1 else "second")
        event_data = event_data.drop(columns=['match_status_id']).sort_values("frame_id").reset_index(drop=True)
        timestamp_dict = get_timestamp(event_data, args.league)
        changed_player_list_in_home, changed_player_list_in_away = get_changed_player_list(event_data, args.league)
        event_data = clean_event_data(event_data, event_priority=config['event_priority'], **timestamp_dict)
        event_data = preprocess_coordinates_in_event_data(event_data, config['origin_pos'], config['absolute_coordinates'])


        # player data
        player_data = safe_pd_read_csv(game_dir / config["player_metadata_filename"])
        player_data = check_and_rename_player_columns(player_data, config['player_columns_mapping'], args.state_def)
        player_data = clean_player_data(player_data, args.state_def)
        # import pdb; pdb.set_trace()
        try:
            condition = (player_data['team_id'] == 0) & (player_data['player_id'] == 0)
            rows_to_move = player_data[condition]
            filtered_player_data = player_data[~condition]
            player_dict = filtered_player_data.set_index(["home_away", 'jersey_number']).to_dict(orient='index')
            player_data = pd.concat([filtered_player_data, rows_to_move], ignore_index=True)
        except:
            print(game_dir)
            import pdb; pdb.set_trace()

        # metadata
        home_team_name = event_data.query("home_away == 'HOME'")['team_name'].iloc[0]
        away_team_name = event_data.query("home_away == 'AWAY'")['team_name'].iloc[0]

        if args.league == "jleague":
            # split tracking data into player and ball
            tracking_1stHalf_data = pd.read_csv(game_dir / config['tracking_1stHalf_filename'], encoding='shift_jis')
            tracking_2ndHalf_data = pd.read_csv(game_dir / config['tracking_2ndHalf_filename'], encoding='shift_jis')
            player_tracking_data, ball_tracking_data = split_tracking_data(tracking_1stHalf_data, tracking_2ndHalf_data)

            # adjust the player roles
            player_data = adjust_player_roles(player_data, event_data)
        elif args.league == "laliga":
            player_tracking_data = safe_pd_read_csv(game_dir / config['player_tracking_filename'])
            ball_tracking_data = safe_pd_read_csv(game_dir / config['ball_tracking_filename'])
        
        # player tracking data
        player_tracking_data = (
            check_and_rename_tracking_columns(player_tracking_data, config['tracking_columns_mapping'])
            .sort_values("frame_id")
            .reset_index(drop=True)
        )
        player_tracking_data = clean_tracking_data(player_tracking_data, timestamp_dict['first_end_frame'])

        # ball tracking data
        ball_tracking_data = (
            check_and_rename_tracking_columns(ball_tracking_data, config['tracking_columns_mapping'])
            .sort_values("frame_id")
            .reset_index(drop=True)
        )
        ball_tracking_data = clean_tracking_data(ball_tracking_data, timestamp_dict['first_end_frame'])
        ball_tracking_data = complement_tracking_ball_with_event_data(ball_tracking_data, event_data, timestamp_dict['first_end_frame'], args.league)
        ball_tracking_data = interpolate_ball_tracking_data(ball_tracking_data, event_data)

        # merge tracking data
        tracking_data = merge_tracking_data(player_tracking_data, ball_tracking_data)
        tracking_data = cut_frames_out_of_game(tracking_data, **timestamp_dict)

        player_change_list = get_player_change_log(
            tracking_data, player_data, changed_player_list_in_home, changed_player_list_in_away
        )
        tracking_data = pad_players_and_interpolate_tracking_data(
            tracking_data=tracking_data,
            player_data=player_data,
            event_data=event_data,
            player_change_list=player_change_list,
            origin_pos=config['origin_pos'],
            absolute_coordinates=config['absolute_coordinates'],
        )
        tracking_data = resample_tracking_data(
            tracking_data=tracking_data,
            timestamp_dict=timestamp_dict,
            player_change_list=player_change_list,
            original_sampling_rate=config['original_sampling_rate'],
            target_sampling_rate=config['target_sampling_rate'],
        )
        tracking_data = preprocess_coordinates_in_tracking_data(
            tracking_data, event_data, config['origin_pos'], config['absolute_coordinates']
        )
        tracking_data = format_tracking_data(tracking_data, home_team_name, away_team_name, player_dict, args.state_def)
        tracking_data = calculate_speed(tracking_data, sampling_rate=config['target_sampling_rate'])
        tracking_data = calculate_acceleration(tracking_data, sampling_rate=config['target_sampling_rate'])
        merged_data = merge_tracking_and_event_data(tracking_data, event_data, args.state_def, args.league)

        # save
        output_dir = args.cleaned_data_dir / game_dir.name
        output_dir.mkdir(exist_ok=True, parents=True)
        event_data.to_csv(output_dir / 'events.csv', index=False)
        player_data.to_csv(output_dir / 'player_info.csv', index=False)
        tracking_data.to_json(output_dir / 'tracking.jsonl', orient='records', lines=True, force_ascii=False)
        save_as_jsonlines(merged_data, output_dir / 'frames.jsonl')
        logger.info(
            f'''
            cleaning {game_dir.name} finished in {time.time() - start_time:.2f} sec
            '''
        )

def clean_single_data(data_path, match_id, config_path, league, save_dir):
    data_path = Path(data_path/match_id)
    save_dir = Path(save_dir)
    config = load_json(config_path)
    logger.info(f"cleaning started... {data_path.name}")
    start_time = time.time()
    
    event_data = safe_pd_read_csv(data_path / config["event_filename"])
    event_data = check_and_rename_event_columns(event_data, config['event_columns_mapping'], league)
    event_data['home_away'] = event_data['home_away'].apply(lambda x: HOME_AWAY_MAP[x])
    event_data['half'] = event_data['match_status_id'].apply(lambda x: "first" if x == 1 else "second")
    event_data = event_data.drop(columns=['match_status_id']).sort_values("frame_id").reset_index(drop=True)
    timestamp_dict = get_timestamp(event_data, league)
    changed_player_list_in_home, changed_player_list_in_away = get_changed_player_list(event_data, league)
    event_data = clean_event_data(event_data, event_priority=config['event_priority'], **timestamp_dict)
    event_data = preprocess_coordinates_in_event_data(event_data, config['origin_pos'], config['absolute_coordinates'], league)


    # player data
    player_data = safe_pd_read_csv(data_path / config["player_metadata_filename"])
    player_data = check_and_rename_player_columns(player_data, config['player_columns_mapping'])
    player_data = clean_player_data(player_data)
    # import pdb; pdb.set_trace()
    try:
        condition = (player_data['team_id'] == 0) & (player_data['player_id'] == 0)
        rows_to_move = player_data[condition]
        filtered_player_data = player_data[~condition]
        player_dict = filtered_player_data.set_index(["home_away", 'jersey_number']).to_dict(orient='index')
        player_data = pd.concat([filtered_player_data, rows_to_move], ignore_index=True)
    except:
        AssertionError(f"Error in {data_path}")

    # metadata
    home_team_name = event_data.query("home_away == 'HOME'")['team_name'].iloc[0]
    away_team_name = event_data.query("home_away == 'AWAY'")['team_name'].iloc[0]

    if league == "jleague":
        # split tracking data into player and ball
        tracking_1stHalf_data = pd.read_csv(data_path / config['tracking_1stHalf_filename'], encoding='shift_jis')
        tracking_2ndHalf_data = pd.read_csv(data_path / config['tracking_2ndHalf_filename'], encoding='shift_jis')
        player_tracking_data, ball_tracking_data = split_tracking_data(tracking_1stHalf_data, tracking_2ndHalf_data)

        # adjust the player roles
        player_data = adjust_player_roles(player_data, event_data)
    elif league == "laliga":
        player_tracking_data = safe_pd_read_csv(data_path / config['player_tracking_filename'])
        ball_tracking_data = safe_pd_read_csv(data_path / config['ball_tracking_filename'])
    
    # player tracking data
    player_tracking_data = (
        check_and_rename_tracking_columns(player_tracking_data, config['tracking_columns_mapping'])
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    player_tracking_data = clean_tracking_data(player_tracking_data, timestamp_dict['first_end_frame'])

    # ball tracking data
    ball_tracking_data = (
        check_and_rename_tracking_columns(ball_tracking_data, config['tracking_columns_mapping'])
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    ball_tracking_data = clean_tracking_data(ball_tracking_data, timestamp_dict['first_end_frame'])
    ball_tracking_data = complement_tracking_ball_with_event_data(ball_tracking_data, event_data, timestamp_dict['first_end_frame'], league)
    ball_tracking_data = interpolate_ball_tracking_data(ball_tracking_data, event_data)

    # merge tracking data
    tracking_data = merge_tracking_data(player_tracking_data, ball_tracking_data)
    tracking_data = cut_frames_out_of_game(tracking_data, **timestamp_dict)

    player_change_list = get_player_change_log(
        tracking_data, player_data, changed_player_list_in_home, changed_player_list_in_away
    )
    tracking_data = pad_players_and_interpolate_tracking_data(
        tracking_data=tracking_data,
        player_data=player_data,
        event_data=event_data,
        player_change_list=player_change_list,
        origin_pos=config['origin_pos'],
        absolute_coordinates=config['absolute_coordinates'],
    )
    tracking_data = resample_tracking_data(
        tracking_data=tracking_data,
        timestamp_dict=timestamp_dict,
        player_change_list=player_change_list,
        original_sampling_rate=config['original_sampling_rate'],
        target_sampling_rate=config['target_sampling_rate'],
    )
    tracking_data = preprocess_coordinates_in_tracking_data(
        tracking_data, event_data, config['origin_pos'], config['absolute_coordinates']
    )
    tracking_data = format_tracking_data(tracking_data, home_team_name, away_team_name, player_dict)
    tracking_data = calculate_speed(tracking_data, sampling_rate=config['target_sampling_rate'])
    tracking_data = calculate_acceleration(tracking_data, sampling_rate=config['target_sampling_rate'])
    merged_data = merge_tracking_and_event_data(tracking_data, event_data)

    # save
    output_dir = save_dir / data_path.name
    output_dir.mkdir(exist_ok=True, parents=True)
    event_data.to_csv(output_dir / 'events.csv', index=False)
    player_data.to_csv(output_dir / 'player_info.csv', index=False)
    tracking_data.to_json(output_dir / 'tracking.jsonl', orient='records', lines=True, force_ascii=False)
    save_as_jsonlines(merged_data, output_dir / 'frames.jsonl')
    logger.info(
        f'''
        cleaning {data_path.name} finished in {time.time() - start_time:.2f} sec
        '''
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessing_config", type=lambda p: Path(p).resolve())
    parser.add_argument("--origin_pos", type=str)
    parser.add_argument("--absolute_coordinates", action="store_true")
    parser.add_argument("--original_sampling_rate", type=int, help="sampling rate of original tracking data")
    parser.add_argument(
        "--target_sampling_rate", type=int, help="sampling rate of target tracking data (5 or 10 or 25)"
    )
    parser.add_argument("--interpolation_threshold", type=int)
    # parser.add_argument("--raw_data_dir", type=lambda p: Path(p).resolve(), default=DATA_DIR / "raw")
    parser.add_argument("--raw_data_dir", type=lambda p: Path(p).resolve(), default="raw")
    parser.add_argument("--cleaned_data_dir", type=lambda p: Path(p).resolve(), default="cleaned")
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--num_process", type=int, default=1)
    args = parser.parse_args()

    config = load_json(args.preprocessing_config)
    # update config if args are specified
    for key in config.keys():
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
    logging.info(f"preprocessing config: {config}")
    assert config['target_sampling_rate'] in [5, 10, 25]

    game_dirs = [dir_ for dir_ in sorted(args.raw_data_dir.glob('*')) if re.match(r'\d{6}', dir_.name)]
    print(f"game number: {len(game_dirs)}")
    # remove_id_list = ['1093837', '1488037', '1118640', '1055302', '1037881', '1119369', '1038551', '1041029']
    remove_id_list = []

    if args.num_process == 1:
        for game_dir in game_dirs[args.start_index : args.end_index]:
            # if game_dir.name == '1018887':
            process_game(game_dir, args=args, config=config)
                # break
    else:
        process_game_partial = partial(process_game, args=args, config=config)
        with Pool(processes=args.num_process) as pool:
            game_dirs_filtered = [dir_ for dir_ in game_dirs[args.start_index : args.end_index] if dir_.name not in remove_id_list]
            pool.map(process_game_partial, game_dirs_filtered)

    save_formatted_json(config, args.cleaned_data_dir / 'config.json')

import logging
import time
import pandas as pd
from pathlib import Path

# if __name__ == "__main__":
from preprocessing.sports.SAR_data.soccer.constant import HOME_AWAY_MAP
from preprocessing.sports.SAR_data.soccer.cleaning.clean_event_data import (
    clean_event_data,
    get_changed_player_list,
    get_timestamp,
    preprocess_coordinates_in_event_data,
)
from preprocessing.sports.SAR_data.soccer.cleaning.clean_data import (
    clean_player_data,
    merge_tracking_and_event_data,
    split_tracking_data,
    adjust_player_roles,
)
from preprocessing.sports.SAR_data.soccer.cleaning.clean_tracking_data import (
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
from preprocessing.sports.SAR_data.soccer.cleaning.map_column_names import (
    check_and_rename_event_columns,
    check_and_rename_player_columns,
    check_and_rename_tracking_columns,
)
from preprocessing.sports.SAR_data.soccer.utils.file_utils import (
    load_json,
    safe_pd_read_csv,
    save_as_jsonlines,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def clean_single_data(data_path, match_id, config_path, league, state_def, save_dir):
    data_path = Path(data_path + str(match_id))
    save_dir = Path(save_dir)
    config = load_json(config_path)
    logger.info(f"cleaning started... {data_path.name}")
    start_time = time.time()

    event_data = safe_pd_read_csv(data_path / config["event_filename"])
    event_data = check_and_rename_event_columns(event_data, config["event_columns_mapping"], league)
    event_data["home_away"] = event_data["home_away"].apply(lambda x: HOME_AWAY_MAP[x])
    event_data["half"] = event_data["match_status_id"].apply(lambda x: "first" if x == 1 else "second")
    event_data = event_data.drop(columns=["match_status_id"]).sort_values("frame_id").reset_index(drop=True)
    timestamp_dict = get_timestamp(event_data, league)
    changed_player_list_in_home, changed_player_list_in_away = get_changed_player_list(event_data, league)
    event_data = clean_event_data(
        event_data,
        event_priority=config["event_priority"],
        **timestamp_dict,
        original_sampling_rate=config["original_sampling_rate"],
    )
    event_data = preprocess_coordinates_in_event_data(event_data, config["origin_pos"], config["absolute_coordinates"], league)

    # player data
    player_data = safe_pd_read_csv(data_path / config["player_metadata_filename"])
    player_data = check_and_rename_player_columns(player_data, config["player_columns_mapping"], state_def, league)
    player_data = clean_player_data(player_data, state_def)

    condition = (player_data["team_id"] == 0) & (player_data["player_id"] == 0)
    rows_to_move = player_data[condition]
    filtered_player_data = player_data[~condition]
    player_dict = filtered_player_data.set_index(["home_away", "jersey_number"]).to_dict(orient="index")
    player_data = pd.concat([filtered_player_data, rows_to_move], ignore_index=True)

    # metadata
    home_team_name = event_data.query("home_away == 'HOME'")["team_name"].iloc[0]
    away_team_name = event_data.query("home_away == 'AWAY'")["team_name"].iloc[0]

    if league == "jleague":
        # split tracking data into player and ball
        tracking_1stHalf_data = safe_pd_read_csv(data_path / config["tracking_1stHalf_filename"])
        tracking_2ndHalf_data = safe_pd_read_csv(data_path / config["tracking_2ndHalf_filename"])
        player_tracking_data, ball_tracking_data = split_tracking_data(tracking_1stHalf_data, tracking_2ndHalf_data)

        # adjust the player roles
        player_data = adjust_player_roles(player_data, event_data)
    elif league == "laliga" or league == "fifawc":
        player_tracking_data = safe_pd_read_csv(data_path / config["player_tracking_filename"])
        ball_tracking_data = safe_pd_read_csv(data_path / config["ball_tracking_filename"])

    # player tracking data
    player_tracking_data = (
        check_and_rename_tracking_columns(player_tracking_data, config["tracking_columns_mapping"])
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    player_tracking_data = clean_tracking_data(player_tracking_data, timestamp_dict["first_end_frame"])

    # ball tracking data
    ball_tracking_data = (
        check_and_rename_tracking_columns(ball_tracking_data, config["tracking_columns_mapping"])
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    ball_tracking_data = clean_tracking_data(ball_tracking_data, timestamp_dict["first_end_frame"])
    ball_tracking_data = complement_tracking_ball_with_event_data(
        ball_tracking_data, event_data, timestamp_dict["first_end_frame"], league
    )
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
        origin_pos=config["origin_pos"],
        absolute_coordinates=config["absolute_coordinates"],
    )
    tracking_data = resample_tracking_data(
        tracking_data=tracking_data,
        timestamp_dict=timestamp_dict,
        player_change_list=player_change_list,
        original_sampling_rate=config["original_sampling_rate"],
        target_sampling_rate=config["target_sampling_rate"],
    )
    tracking_data = preprocess_coordinates_in_tracking_data(
        tracking_data, event_data, config["origin_pos"], config["absolute_coordinates"], league=league
    )
    tracking_data = format_tracking_data(tracking_data, home_team_name, away_team_name, player_dict, state_def)
    tracking_data = calculate_speed(tracking_data, sampling_rate=config["target_sampling_rate"])
    tracking_data = calculate_acceleration(tracking_data, sampling_rate=config["target_sampling_rate"])
    merged_data = merge_tracking_and_event_data(tracking_data, event_data, state_def, league)

    # Filter out frames with invalid ball data
    initial_count = len(merged_data)
    valid_frames = []

    for frame in merged_data:
        state = frame.get("state")
        if state is None:
            continue

        ball_data = state.get("ball")
        if ball_data is None or ball_data == {}:
            logger.warning(
                f"Skipping frame with empty ball data: game_id={frame.get('game_id')}, frame_id={frame.get('frame_id')}"
            )
            continue

        # Check if ball data has required position field
        if not isinstance(ball_data, dict) or "position" not in ball_data:
            logger.warning(
                f"Skipping frame with invalid ball structure: game_id={frame.get('game_id')}, frame_id={frame.get('frame_id')}"
            )
            continue

        valid_frames.append(frame)

    merged_data = valid_frames
    filtered_count = len(merged_data)

    if initial_count != filtered_count:
        logger.info(f"Filtered out {initial_count - filtered_count} frames with invalid ball data")

    # save
    output_dir = save_dir / data_path.name
    output_dir.mkdir(exist_ok=True, parents=True)
    event_data.to_csv(output_dir / "events.csv", index=False)
    player_data.to_csv(output_dir / "player_info.csv", index=False)
    tracking_data.to_json(output_dir / "tracking.jsonl", orient="records", lines=True, force_ascii=False)
    save_as_jsonlines(merged_data, output_dir / "frames.jsonl")
    logger.info(
        f"""
        cleaning {data_path.name} finished in {time.time() - start_time:.2f} sec
        """
    )

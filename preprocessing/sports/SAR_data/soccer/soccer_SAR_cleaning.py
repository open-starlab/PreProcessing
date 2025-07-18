import logging
import time
import pandas as pd
from pathlib import Path

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


def clean_single_data(
    data_path,
    match_id,
    config_path,
    league,
    state_def,
    save_dir,
):
    """
    Clean a single game's raw event & tracking data into our unified SAR format.
    """
    data_path = Path(data_path) / match_id
    save_dir = Path(save_dir)
    config = load_json(config_path)

    logger.info(f"cleaning started... {data_path.name}")
    start_time = time.time()

    # --- Event data ---
    event_data = safe_pd_read_csv(data_path / config["event_filename"])
    event_data = check_and_rename_event_columns(
        event_data, config["event_columns_mapping"], league
    )
    event_data["home_away"] = event_data["home_away"].map(HOME_AWAY_MAP)
    event_data["half"] = event_data["match_status_id"].apply(
        lambda x: "first" if x == 1 else "second"
    )
    event_data = (
        event_data.drop(columns=["match_status_id"])
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    timestamp_dict = get_timestamp(event_data, league)
    changed_home, changed_away = get_changed_player_list(event_data, league)
    event_data = clean_event_data(
        event_data,
        event_priority=config["event_priority"],
        **timestamp_dict,
        original_sampling_rate=config["original_sampling_rate"],
    )
    event_data = preprocess_coordinates_in_event_data(
        event_data, config["origin_pos"], config["absolute_coordinates"], league
    )

    # --- Player metadata ---
    player_data = safe_pd_read_csv(data_path / config["player_metadata_filename"])
    player_data = check_and_rename_player_columns(
        player_data, config["player_columns_mapping"], state_def, league
    )
    player_data = clean_player_data(player_data, state_def)

    # move the “ball” placeholder row to end
    mask = (player_data["team_id"] == 0) & (player_data["player_id"] == 0)
    ball_rows = player_data[mask]
    rest = player_data[~mask]
    player_dict = rest.set_index(["home_away", "jersey_number"]).to_dict("index")
    player_data = pd.concat([rest, ball_rows], ignore_index=True)

    # --- Tracking split & cleaning ---
    home_team = event_data.query("home_away == 'HOME'")["team_name"].iat[0]
    away_team = event_data.query("home_away == 'AWAY'")["team_name"].iat[0]

    if league == "jleague":
        t1 = safe_pd_read_csv(data_path / config["tracking_1stHalf_filename"])
        t2 = safe_pd_read_csv(data_path / config["tracking_2ndHalf_filename"])
        player_track, ball_track = split_tracking_data(t1, t2)
        player_data = adjust_player_roles(player_data, event_data)
    else:  # "laliga"
        player_track = safe_pd_read_csv(data_path / config["player_tracking_filename"])
        ball_track = safe_pd_read_csv(data_path / config["ball_tracking_filename"])

    # clean & resample
    player_track = (
        check_and_rename_tracking_columns(
            player_track, config["tracking_columns_mapping"]
        )
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    player_track = clean_tracking_data(player_track, timestamp_dict["first_end_frame"])

    ball_track = (
        check_and_rename_tracking_columns(
            ball_track, config["tracking_columns_mapping"]
        )
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    ball_track = clean_tracking_data(ball_track, timestamp_dict["first_end_frame"])
    ball_track = complement_tracking_ball_with_event_data(
        ball_track, event_data, timestamp_dict["first_end_frame"], league
    )
    ball_track = interpolate_ball_tracking_data(ball_track, event_data)

    merged = merge_tracking_and_event_data(
        pad_players_and_interpolate_tracking_data(
            tracking_data=merge_tracking_data(player_track, ball_track),
            player_data=player_data,
            event_data=event_data,
            player_change_list=get_player_change_log(
                merge_tracking_data(player_track, ball_track),
                player_data,
                changed_home,
                changed_away,
            ),
            origin_pos=config["origin_pos"],
            absolute_coordinates=config["absolute_coordinates"],
        ),
        event_data,
        state_def,
        league,
    )

    # filter out any frames with missing ball info
    initial = len(merged)
    merged = [
        f
        for f in merged
        if f.get("state", {}).get("ball", {}).get("position", None) is not None
    ]
    filtered = len(merged)
    if filtered < initial:
        logger.info(f"Filtered {initial - filtered} bad frames")

    # --- Save outputs ---
    out = save_dir / data_path.name
    out.mkdir(parents=True, exist_ok=True)
    event_data.to_csv(out / "events.csv", index=False)
    player_data.to_csv(out / "player_info.csv", index=False)
    player_track.to_json(
        out / "tracking.jsonl", orient="records", lines=True, force_ascii=False
    )
    save_as_jsonlines(merged, out / "frames.jsonl")

    logger.info(f"cleaning {data_path.name} done in {time.time() - start_time:.2f}s")

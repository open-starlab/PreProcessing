from .soccer.soccer_SAR_class import Soccer_SAR_data
import os
import pathlib


match_id_dss = "0001"
config_path_dss = "data/dss/config/preprocessing_dssports2020.json"

match_id_laliga = "1317846"
config_path_skc = "data/stb_skc/config/preprocessing_statsbomb_skillcorner2024.json"


def test_datastadium_pvs_preprocess():
    cleaning_dir = os.getcwd() + "/data/dss/clean_data"
    cleaning_dir_path = pathlib.Path(cleaning_dir)
    game_dir = cleaning_dir_path / match_id_dss

    if not cleaning_dir_path.exists():
        os.makedirs(cleaning_dir_path, exist_ok=True)
    if not game_dir.exists():
        os.makedirs(game_dir, exist_ok=True)

    preprocessed_dir = os.getcwd() + "/data/dss/preprocess_data"

    Soccer_SAR_data(
        data_provider="datastadium",
        state_def="PVS",
        match_id=match_id_dss,
        config_path=config_path_dss,
        preprocess_method="SAR",
    ).preprocess_data(cleaning_dir=cleaning_dir, preprocessed_dir=preprocessed_dir)


def test_statsbomb_skillcorner_pvs_preprocess():
    cleaning_dir = os.getcwd() + "/data/stb_skc/clean_data"
    cleaning_dir_path = pathlib.Path(cleaning_dir)
    game_dir = cleaning_dir_path / match_id_laliga

    if not cleaning_dir_path.exists():
        os.makedirs(cleaning_dir_path, exist_ok=True)
    if not game_dir.exists():
        os.makedirs(game_dir, exist_ok=True)

    preprocessed_dir = os.getcwd() + "/data/stb_skc/preprocess_data"

    Soccer_SAR_data(
        data_provider="statsbomb_skillcorner",
        state_def="PVS",
        match_id=match_id_laliga,
        config_path=config_path_skc,
        preprocess_method="SAR",
    ).preprocess_data(cleaning_dir=cleaning_dir, preprocessed_dir=preprocessed_dir)


if __name__ == "__main__":
    # test_datastadium_pvs_preprocess()
    # test_statsbomb_skillcorner_pvs_preprocess()
    print("All tests passed successfully.")

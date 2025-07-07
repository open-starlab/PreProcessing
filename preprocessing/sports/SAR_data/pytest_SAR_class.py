from .soccer.soccer_SAR_class import Soccer_SAR_data
from .SAR_class import SAR_data
import os
import pathlib


datastadium_path = "data/dss/raw/"
match_id_dss = "0001"
config_path_dss = "data/dss/config/preprocessing_dssports2020.json"

statsbomb_skillcorner_path = "data/stb_skc/raw"
match_id_laliga = "1317846"
config_path_skc = "data/stb_skc/config/preprocessing_statsbomb_skillcorner2024.json"
statsbomb_skillcorner_match_id = "preprocessing/sports/SAR_data/match_id_dict.json"


def test_datastadium_pvs_load_data():
    sar = SAR_data(
        data_provider="datastadium",
        state_def="PVS",
        data_path=datastadium_path,
        match_id=match_id_dss,
        config_path=config_path_dss,
    )
    sar.load_data()
    assert sar is not None


def test_statsbomb_skillcorner_pvs_load_data():
    sar = SAR_data(
        data_provider="statsbomb_skillcorner",
        state_def="PVS",
        data_path=statsbomb_skillcorner_path,
        statsbomb_skillcorner_match_id=statsbomb_skillcorner_match_id,
        match_id=match_id_laliga,
        config_path=config_path_skc,
    )
    sar.load_data()
    assert sar is not None


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
    ).preprocess_single_data(cleaning_dir=cleaning_dir, preprocessed_dir=preprocessed_dir)


if __name__ == "__main__":
    test_datastadium_pvs_load_data()
    # test_statsbomb_skillcorner_pvs_load_data()
    test_datastadium_pvs_preprocess()
    print("All tests passed successfully.")

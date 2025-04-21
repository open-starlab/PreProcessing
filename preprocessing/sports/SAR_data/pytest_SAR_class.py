from .soccer.soccer_SAR_class import Soccer_SAR_data
from .SAR_class import SAR_data
import os


datastadium_path = "data/dss/raw/"
match_id_dss = "0001"
config_path_dss = "data/dss/config/preprocessing_dssports2020.json"

statsbomb_skillcorner_path = "data/stb_skc/raw"
match_id_laliga = "1317846"
config_path_skc = "data/stb_skc/config/preprocessing_statsbomb_skillcorner2024.json"
statsbomb_skillcorner_match_id = "preprocessing/sports/SAR_data/match_id_dict.json"


def test_datastadium_load_data():
    sar = SAR_data(data_provider='datastadium', data_path=datastadium_path, match_id=match_id_dss, config_path=config_path_dss)
    sar.load_data()
    assert sar is not None


def test_statsbomb_skillcorner_load_data():
    sar = SAR_data(
        data_provider='statsbomb_skillcorner',
        data_path=statsbomb_skillcorner_path,
        statsbomb_skillcorner_match_id=statsbomb_skillcorner_match_id,
        match_id=match_id_laliga,
        config_path=config_path_skc
    )
    sar.load_data()
    assert sar is not None

def test_datastadium_preprocess():
    Soccer_SAR_data(
        data_provider='datastadium',
        match_id="0001",
        config_path="data/dss/config/preprocessing_dssports2020.json",
        preprocess_method="SAR"
    ).preprocess_single_data(
        cleaning_dir="/home/k_ide/workspace6/open-starlab/PreProcessing/data/dss/clean_data",
        preprocessed_dir="/home/k_ide/workspace6/open-starlab/PreProcessing/data/dss/preprocess_data"
    )

def test_statsbomb_skillcorner_preprocess():
    Soccer_SAR_data(
        data_provider='statsbomb_skillcorner',
        data_path='/data_pool_1/laliga_23',
        match_id="1317846", # match_id for skillcorner
        config_path=os.getcwd()+"/data/stb_skc/config/preprocessing_statsbomb_skillcorner2024.json",
        preprocess_method="SAR"
    ).preprocess_single_data(
        cleaning_dir=os.getcwd()+"/data/stb_skc/clean_data",
        preprocessed_dir=os.getcwd()+"/data/stb_skc/preprocess_data"
    )
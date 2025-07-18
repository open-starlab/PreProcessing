class SAR_data:
    # Supported providers and state definitions
    sports = ["statsbomb_skillcorner", "datastadium", "robocup_2d"]
    state_list = ["PVS", "EDMS", "PVSS"]

    def __new__(cls, data_provider, state_def=None, *args, **kwargs):
        # RoboCup‑2D QMix preprocessing (state_def must be PVSS)
        if data_provider == "robocup_2d" and state_def == "PVSS":
            from .soccer.soccer_SAR_class import RoboCup2D_QMix_Processor
            return RoboCup2D_QMix_Processor(*args, **kwargs)

        # All other soccer providers require a valid state definition
        if data_provider in cls.sports and state_def in cls.state_list:
            from .soccer.soccer_SAR_class import Soccer_SAR_data
            return Soccer_SAR_data(data_provider, state_def, *args, **kwargs)

        # StatsBomb alias but not implemented
        if data_provider == "statsbomb":
            raise NotImplementedError("StatsBomb SAR data is not implemented yet.")

        # Anything else is unsupported
        raise ValueError(
            f"Unsupported data provider '{data_provider}' or state definition '{state_def}'. "
            f"Supported providers: {cls.sports}, Supported states: {cls.state_list}."
        )


if __name__ == "__main__":
    # Example usage
    datastadium_path = "./JLeagueData/Data_20200508/"
    match_id = "2019091416"
    config_path = "data/dss/config/preprocessing_dssports2020.json"

    # PVS state
    SAR_data(
        data_provider="datastadium",
        state_def="PVS",
        data_path=datastadium_path,
        match_id=match_id,
        config_path=config_path,
    ).load_data()

    # EDMS state
    SAR_data(
        data_provider="datastadium",
        state_def="EDMS",
        data_path=datastadium_path,
        match_id=match_id,
        config_path=config_path,
    ).load_data()

    # RoboCup2D QMix preprocessing
    robocup_path = "./robocup2d/data"
    qmix_match_id = "20230515_001"
    SAR_data(
        data_provider="robocup_2d",
        state_def="PVSS",
        data_path=robocup_path,
        match_id=qmix_match_id,
        config_path=config_path,
    ).load_data()

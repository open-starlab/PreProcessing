class SAR_data:
    # Modified the sports list to only include fully supported providers
    sports = ["statsbomb_skillcorner", "datastadium"]
    state_list = ["PVS", "EDMS"]

    def __new__(cls, data_provider, state_def, *args, **kwargs):
        if data_provider in cls.sports and state_def in cls.state_list:
            # If the data_provider is in the supported list, return an instance of Soccer_SAR_data
            from .soccer.soccer_SAR_class import Soccer_SAR_data

            return Soccer_SAR_data(data_provider, state_def, *args, **kwargs)
        elif data_provider == "statsbomb":
            # For 'statsbomb', raise a NotImplementedError indicating it is not implemented
            raise NotImplementedError("StatsBomb SAR data is not implemented yet.")
        elif data_provider == "robocup_2d":
            # Add a new clause for 'robocup_2d' that raises a NotImplementedError for RL usage
            raise NotImplementedError("RoboCup 2D SAR data is not implemented for RL. Please use a supported data provider.")
        else:
            # If the data_provider is unrecognized or state_def is unrecongnized, raise a ValueError
            raise ValueError(
                f"Unsupported data provider '{data_provider}' or state definition '{state_def}'. "
                f"Supported providers: {cls.sports}, Supported states: {cls.state_list}."
            )


if __name__ == "__main__":
    # Test block remains unchanged, using a supported provider ('datastadium')
    datafactory_path = "datafactory_directory_path"
    match_id = "match_id"
    config_path = "data/dss/config/preprocessing_dssports2020.json"

    # PVS load data
    # SAR_data(
    #     data_provider="datafactory", state_def="PVS", data_path=datafactory_path, match_id=match_id, config_path=config_path
    # ).load_data()

    # run preprocessing_data method
    # SAR_data(
    #     data_provider="datafactory",
    #     state_def="PVS",
    #     data_path=datafactory_path,
    #     match_id=match_id,
    #     config_path=config_path,
    #     preprocess_method="SAR",
    # ).preprocess_data()

    # # EDMS
    # SAR_data(
    #     data_provider="datafactory", state_def="EDMS", data_path=datafactory_path, match_id=match_id, config_path=config_path
    # ).load_data()

class SAR_data:
    # Fully supported providers for standard SAR preprocessing.
    sports = ["statsbomb_skillcorner", "datastadium", "fifawc"]
    state_list = ["PVS", "EDMS"]

    def __new__(cls, data_provider, state_def, *args, **kwargs):
        if data_provider in cls.sports and state_def in cls.state_list:
            # Route supported providers to the soccer SAR implementation.
            from .soccer.soccer_SAR_class import Soccer_SAR_data

            return Soccer_SAR_data(data_provider, state_def, *args, **kwargs)

        elif data_provider == "statsbomb":
            raise NotImplementedError("StatsBomb SAR data is not implemented yet.")

        elif data_provider == "robocup_2d" and state_def in cls.state_list:
            # RoboCup2D is supported only through the SAR2RL workflow.
            preprocess_method = kwargs.get("preprocess_method", "SAR")
            if preprocess_method != "SAR2RL":
                raise NotImplementedError(
                    "RoboCup 2D SAR data is only supported for preprocess_method='SAR2RL'."
                )
            # This keeps the public API path:
            # SAR_data(..., data_provider='robocup_2d', preprocess_method='SAR2RL')
            # routed into Soccer_SAR_data, which then invokes the canonical
            # soccer/SAR2RL preprocessing implementation.
            from .soccer.soccer_SAR_class import Soccer_SAR_data
            return Soccer_SAR_data(data_provider, state_def, *args, **kwargs)
        else:
            raise ValueError(
                f"Unsupported data provider '{data_provider}' or state definition '{state_def}'. "
                f"Supported providers: {cls.sports + ['robocup_2d (SAR2RL only)']}, "
                f"Supported states: {cls.state_list}."
            )


if __name__ == "__main__":
    # Test block remains unchanged, using a supported provider ('datastadium')
    datafactory_path = "datafactory_directory_path"
    match_id = "match_id"

    # SAR_data(
    #     data_provider="datafactory",
    #     state_def="PVS",
    #     data_path="datafactory_path",
    #     match_id=match_id,
    #     config_path="preprocessing_datafactory.json",
    #     preprocess_method="SAR",
    # ).preprocess_data()

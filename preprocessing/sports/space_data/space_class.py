class Space_data:
    # Modified the sports list to only include fully supported providers
    basketball_data_provider = ["SportVU_NBA"]
    soccer_data_provider = ["fifa_wc_2022"]
    ultimate_data_provider = ["UltimateTrack", "UFATrack"]

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.basketball_data_provider:
            from .basketball.basketball_space_class import Basketball_space_data

            # If the data_provider is in the supported list, return an instance of Basketball_space_data
            return Basketball_space_data(data_provider, *args, **kwargs)
        elif data_provider in cls.soccer_data_provider:
            from .soccer.soccer_space_class import Soccer_space_data

            # If the data_provider is in the supported list, return an instance of Soccer_space_data
            return Soccer_space_data(data_provider, *args, **kwargs)
        elif data_provider in cls.ultimate_data_provider:
            from .ultimate.ultimate_space_class import Ultimate_space_data

            # If the data_provider is in the supported list, return an instance of Ultimate_space_data
            return Ultimate_space_data(data_provider, *args, **kwargs)
        else:
            # If the data_provider is unrecognized, raise a ValueError
            raise ValueError(f"Unknown data provider: {data_provider}")

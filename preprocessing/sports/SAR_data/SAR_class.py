from .soccer.soccer_SAR_class import Soccer_SAR_data

class SAR_data:
    # Modified the sports list to only include fully supported providers
    sports = ['statsbomb_skillcorner', 'datastadium']

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.sports:
            # If the data_provider is in the supported list, return an instance of Soccer_SAR_data
            return Soccer_SAR_data(data_provider, *args, **kwargs)
        elif data_provider == "statsbomb":
            # For 'statsbomb', raise a NotImplementedError indicating it is not implemented
            raise NotImplementedError('StatsBomb SAR data is not implemented yet.')
        elif data_provider == "robocup_2d":
            # Add a new clause for 'robocup_2d' that raises a NotImplementedError for RL usage
            raise NotImplementedError('RoboCup 2D SAR data is not implemented for RL. Please use a supported data provider.')
        else:
            # If the data_provider is unrecognized, raise a ValueError
            raise ValueError(f'Unknown data provider: {data_provider}')

if __name__ == '__main__':
    # Test block remains unchanged, using a supported provider ('datastadium')
    datastadium_path = "/work5/fujii/work/JLeagueData/Data_20200508/"
    match_id = "2019091416"
    config_path = "data/dss/config/preprocessing_dssports2020.json"
    SAR_data(data_provider='datastadium', data_path=datastadium_path, match_id=match_id, config_path=config_path).load_data()

from .basketball.basketball_space_class import basketball_space_data

class Space_data:
    # Modified the sports list to only include fully supported providers
    sports = ['statsbomb', 'datastadium']

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.sports:
            # If the data_provider is in the supported list, return an instance of Soccer_SAR_data
            return Soccer_SAR_data(data_provider, *args, **kwargs)
        else:
            # If the data_provider is unrecognized, raise a ValueError
            raise ValueError(f'Unknown data provider: {data_provider}')

if __name__ == '__main__':
    # Test block remains unchanged, using a supported provider ('datastadium')
    datastadium_path = "./JLeagueData/Data_20200508/"
    match_id = "2019091416"
    config_path = "data/dss/config/preprocessing_dssports2020.json"
    Space_data(data_provider='datastadium', data_path=datastadium_path, match_id=match_id, config_path=config_path).load_data()

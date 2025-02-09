from .soccer.soccer_SAR_class import Soccer_SAR_data

class SAR_data:
    sports = ['statsbomb_skillcorner', 'datastadium', 'statsbomb']

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.sports:
            return Soccer_SAR_data(data_provider, *args, **kwargs)
        elif data_provider == "statsbomb":
            raise NotImplementedError('StatsBomb data not implemented yet')
        else:
            raise ValueError(f'Unknown data provider: {data_provider}')
  

if __name__ == '__main__':
    #check if the Soccer_event_data class is correctly implemented

    datastadium_path = "/work5/fujii/work/JLeagueData/Data_20200508/"
    match_id = "2019091416"
    config_path = "data/dss/config/preprocessing_dssports2020.json"
    SAR_data(data_provider='datastadium', data_path=datastadium_path, match_id=match_id, config_path=config_path).load_data()

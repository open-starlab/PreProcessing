class SAR_data:
    # Updated supported providers list to include RoboCup 2D
    sports = ['statsbomb_skillcorner', 'datastadium', 'robocup_2d']

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.sports:
            if data_provider == 'robocup_2d':
                # Dynamically import the QMix-compatible preprocessor for RoboCup 2D
                from .robocup_qmix_preprocessor import RoboCup2D_QMix_Processor  # type: ignore
                return RoboCup2D_QMix_Processor(*args, **kwargs)
            else:
                # Fallback to existing soccer SAR class for other providers
                from .soccer.soccer_SAR_class import Soccer_SAR_data
                return Soccer_SAR_data(data_provider, *args, **kwargs)

        elif data_provider == "statsbomb":
            raise NotImplementedError('StatsBomb SAR data is not implemented yet.')

        else:
            raise ValueError(f'Unknown data provider: {data_provider}')


if __name__ == '__main__':
    # ✅ Test block for supported provider
    datastadium_path = "./JLeagueData/Data_20200508/"
    match_id = "2019091416"
    config_path = "data/dss/config/preprocessing_dssports2020.json"
    SAR_data(
        data_provider='datastadium',
        data_path=datastadium_path,
        match_id=match_id,
        config_path=config_path
    ).load_data()

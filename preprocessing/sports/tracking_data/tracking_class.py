from .soccer.soccer_tracking_class import Soccer_tracking_data

class Tracking_data:
    soccer_data_provider = ['soccer']
    handball_data_provider = []
    rocket_league_data_provider = []

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.soccer_data_provider:
            return Soccer_tracking_data(*args, **kwargs)
        elif data_provider in cls.handball_data_provider:
            raise NotImplementedError('Handball event data not implemented yet')
        elif data_provider in cls.rocket_league_data_provider:
            raise NotImplementedError('Rocket League event data not implemented yet')
        else:
            raise ValueError(f'Unknown data provider: {data_provider}')


if __name__ == '__main__':
    #check if the Soccer_event_data class is correctly implemented
    import os
    game_id = 0  # Select the index from the list of files in the data_path.
    data_path = os.getcwd()+"/test/sports/event_data/data/datastadium/"

    # Call the function
    tracking_home, tracking_away, jerseynum_df = Tracking_data('soccer').process_datadium_tracking_data(game_id,data_path,test=True)

    tracking_home.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/test_tracking_home.csv", index=False)
    tracking_away.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/test_tracking_away.csv", index=False)
    jerseynum_df.to_csv(os.getcwd()+"/test/sports/event_data/data/datastadium/test_jerseynum.csv", index=False)
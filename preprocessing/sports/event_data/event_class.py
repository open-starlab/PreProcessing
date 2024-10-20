from .soccer.soccer_event_class import Soccer_event_data

class Event_data:
    soccer_data_provider = ['datafactory', 'metrica', 'opta', 'robocup_2d', 'sportec', 'statsbomb', 'statsbomb_skillcorner',
                            'wyscout', 'datastadium','soccertrack']
    handball_data_provider = []
    rocket_league_data_provider = []

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.soccer_data_provider:
            return Soccer_event_data(data_provider, *args, **kwargs)
        elif data_provider in cls.handball_data_provider:
            raise NotImplementedError('Handball event data not implemented yet')
        elif data_provider in cls.rocket_league_data_provider:
            raise NotImplementedError('Rocket League event data not implemented yet')
        else:
            raise ValueError(f'Unknown data provider: {data_provider}')


if __name__ == '__main__':
    #check if the Soccer_event_data class is correctly implemented
    import os
    datafactory_path=os.getcwd()+"/test/sports/event_data/data/datafactory/datafactory_events.json"
    datafactory_df=Event_data(data_provider='datafactory',event_path=datafactory_path).load_data()
    datafactory_df.to_csv(os.getcwd()+"/test/sports/event_data/data/datafactory/test_data_main.csv",index=False)
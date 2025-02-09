from .soccer.soccer_SAR_class import Soccer_SAR_data

class SAR_data:
    sports = ['soccer']


    def __new__(cls, sports , **kwargs):
        if sports in cls.sports:
            return Soccer_SAR_data(sports, **kwargs)
        elif sports  in cls.sports:
            raise NotImplementedError('Handball event data not implemented yet')
        elif sports in cls.sports:
            raise NotImplementedError('Rocket League event data not implemented yet')
        else:
            raise ValueError(f'Unknown data provider: {sports }')
        

if __name__ == '__main__':
    #check if the Soccer_event_data class is correctly implemented
    import os
    datafactory_path=os.getcwd()+"/test/sports/event_data/data/datafactory/datafactory_events.json"
    datafactory_df=SAR_data(data_provider='soccer',event_path=datafactory_path).load_data()
    datafactory_df.to_csv(os.getcwd()+"/test/sports/event_data/data/datafactory/test_data_main.csv",index=False)
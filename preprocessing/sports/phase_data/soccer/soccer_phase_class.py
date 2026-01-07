#Target data provider [Metrica,Robocup 2D simulation,Statsbomb,Wyscout,Opta data,DataFactory,sportec]

if __name__ == '__main__':
    import soccer_load_data
else:
    from . import soccer_load_data

#create a class to wrap the data source
class Soccer_phase_data:
    def __init__(self,data_provider,bp_tracking_xml_path=None,bp_tracking_json_paths=None,bp_event_path=None,bp_meta_data=None,
                sb_event_path=None, sc_tracking_path=None, sc_match_path=None, sc_players_path=None):
        self.data_provider = data_provider
        self.bp_tracking_xml_path = bp_tracking_xml_path
        self.bp_tracking_json_paths = bp_tracking_json_paths
        self.bp_event_path = bp_event_path
        self.bp_meta_data = bp_meta_data
        self.sb_event_path = sb_event_path
        self.sc_tracking_path = sc_tracking_path
        self.sc_match_path = sc_match_path
        self.sc_players_path=sc_players_path

    def load_data(self):
        #based on the data provider, load the dataloading function from load_data.py (single file)
        if self.data_provider == 'bepro':
            df=soccer_load_data.load_bepro(self.bp_tracking_xml_path, self.bp_tracking_json_paths, self.bp_event_path, self.bp_meta_data)
        elif self.data_provider == 'statsbomb_skillcorner':
            df=soccer_load_data.load_statsbomb_skillcorner(sb_event_path=self.sb_event_path, sc_tracking_path=self.sc_tracking_path, sc_match_path=self.sc_match_path, sc_players_path=self.sc_players_path)
        # elif self.data_provider == 'pff_fc':
        #     df=soccer_load_data.load_pff2metrica(self.bp_event_path)
        # elif self.data_provider == 'robocup_2d':
        #     df=soccer_load_data.load_robocup_2d(self.event_path,match_id=self.match_id,tracking_path=self.tracking_path)
        # elif self.data_provider == 'datastadium':
        #     df=soccer_load_data.load_datastadium(self.event_path,self.tracking_home_path,self.tracking_away_path)
        else:
            raise ValueError('Data provider not supported or not found')
        return df

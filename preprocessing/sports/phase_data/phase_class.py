class Phase_data:
    soccer_data_provider = ['bepro', 'skillcorner', 'pff_fc'] # 'robocup_2d', 'datastadium', 
    handball_data_provider = []
    rocket_league_data_provider = [] # 'carball'

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.soccer_data_provider:
            from .soccer.soccer_phase_class import Soccer_phase_data
            return Soccer_phase_data(data_provider, *args, **kwargs)
        elif data_provider in cls.handball_data_provider:
            raise NotImplementedError('Handball phase data not implemented yet')
        elif data_provider in cls.rocket_league_data_provider:
            raise NotImplementedError('rocket_league phase data not implemented yet')
            # from .rocket_league.rocket_league_phase_class import Rocket_league_phase_data
            # return Rocket_league_phase_data(data_provider, *args, **kwargs)
        else:
            raise ValueError(f'Unknown data provider: {data_provider}')


if __name__ == '__main__':
    #check if the Soccer_tracking_data class is correctly implemented
    import os
    import argparse
    import glob
    args = argparse.ArgumentParser()
    args.add_argument('--data_provider', required=True, choices=['bepro', 'skillcorner', 'pff_fc'], help='kind of data provider')
    args.add_argument('--match_id', required=True, help='ID of match data')
    data_provider = args.data_provider
    match_ids = [str(match_id) for match_id in args.match_id.split(",")]
    base_dir = os.getcwd() + f"/test/sports/tracking_data/{data_provider}/"
    if data_provider == 'bepro':
        for match_id in match_ids:
            # The format for bepro has changed from Match ID: 130000(?).
            if int(match_id) >= 130000:
                file_pattern = os.path.join(base_dir, match_id, f"{match_id}_*_frame_data.json")
                tracking_json_paths = sorted(glob.glob(file_pattern))
                preprocessing_df=Phase_data(data_provider=data_provider, bp_tracking_json_paths=tracking_json_paths).load_data()
            else:
                tracking_path=os.getcwd()+f"/test/sports/tracking_data/{data_provider}/{match_id}/{match_id}_tracker_box_data.xml"
                preprocessing_df=Phase_data(data_provider=data_provider, bp_tracking_xml_path=tracking_path).load_data()
    elif data_provider == 'skillcorner':
        print('not yet')
    elif data_provider == 'pff_fc':
        print('not yet')
    preprocessing_df.to_csv(os.getcwd()+f"/test/sports/tracking_data/{data_provider}/{match_id}/test_data_main.csv",index=False)
class Phase_data:
    soccer_data_provider = ['bepro', 'statsbomb_skillcorner'] 
    other_soccer_data_provider = ['pff_fc', 'robocup_2d', 'datastadium']
    handball_data_provider = []
    rocket_league_data_provider = ['carball']

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.soccer_data_provider:
            from preprocessing.sports.phase_data.soccer.soccer_phase_class import Soccer_phase_data
            return Soccer_phase_data(data_provider, *args, **kwargs)
        elif data_provider in cls.other_soccer_data_provider:
            raise NotImplementedError('other soccer data provider not implemented yet')
        elif data_provider in cls.handball_data_provider:
            raise NotImplementedError('Handball phase data not implemented yet')
        elif data_provider in cls.rocket_league_data_provider:
            raise NotImplementedError('rocket_league phase data not implemented yet')
        else:
            raise ValueError(f'Unknown data provider: {data_provider}')


# def main():
#     import os
#     import argparse
#     import glob
#     args = argparse.ArgumentParser()
#     args.add_argument('--data_provider', required=True, choices=['bepro', 'statsbomb_skillcorner', 'pff_fc'], help='kind of data provider')
#     args.add_argument('--match_id', required=False, help='ID of match data')
#     args = args.parse_args()
#     data_provider = args.data_provider
#     base_dir = os.getcwd() + f"path/to"
#     if data_provider == 'bepro':
#         match_ids = [str(match_id) for match_id in args.match_id.split(",")]
#         for match_id in match_ids:
#             # The format for bepro has changed from Match ID: 130000(?).
#             if int(match_id) >= 130000:
#                 file_pattern = os.path.join(base_dir, 'tracking_data', data_provider, match_id, f"{match_id}_*_frame_data.json")
#                 tracking_json_paths = sorted(glob.glob(file_pattern))
#                 meta_data = os.path.join(base_dir, 'tracking_data', data_provider, match_id, f"{match_id}_metadata.json")
#                 event_csv_path = glob.glob(os.path.join(os.path.join(base_dir, 'event_data', data_provider, match_id), '*.csv'))
#                 preprocessing_df=Phase_data(data_provider=data_provider, bp_tracking_json_paths=tracking_json_paths, event_path=event_csv_path[0], meta_data=meta_data).load_data()
#             else:
#                 tracking_path=os.getcwd()+f"path/to/tracking_data/{data_provider}/{match_id}/{match_id}_tracker_box_data.xml"
#                 meta_data = os.path.join(base_dir, 'tracking_data', data_provider, match_id, f"{match_id}_tracker_box_metadata.xml")
#                 event_csv_path = glob.glob(os.path.join(os.path.join(base_dir, 'event_data', data_provider, match_id), '*.csv'))
#                 preprocessing_df=Phase_data(data_provider=data_provider, bp_tracking_xml_path=tracking_path, event_path=event_csv_path[0], meta_data=meta_data).load_data()
#             output_file_path = os.path.join(base_dir, 'phase_data', data_provider, match_id, f"{match_id}_main_data.csv")
#             preprocessing_df.to_csv(output_file_path,index=False)
#             print(f"✅ All period tracking data saved successfully at {output_file_path}.")
#     elif data_provider == 'statsbomb_skillcorner':
#         sb_match_id = 3894537 # 843, 537
#         sc_match_id = 1018887 # 1498966, 1018887
#         sb_event_path=f'path/to/event_data/statsbomb/{sb_match_id}_events.pkl'
#         sc_tracking_path=f'path/to/tracking_data/skillcorner/LaLiga-2023-2024/tracking/{sc_match_id}.json'
#         sc_match_path=f'path/to/tracking_data/skillcorner/LaLiga-2023-2024/match/{sc_match_id}.json'
#         sc_players_path='path/to/tracking_data/skillcorner/LaLiga-2023-2024/players/players.json'
#         preprocessing_df=Phase_data(data_provider=data_provider, sb_event_path=sb_event_path, sc_tracking_path=sc_tracking_path, sc_match_path=sc_match_path, sc_players_path=sc_players_path).load_data()
#         output_file_dir = os.path.join(base_dir, 'phase_data', data_provider, f'{sb_match_id}_{sc_match_id}')
#         os.makedirs(output_file_dir, exist_ok=True)
#         output_file_path = os.path.join(output_file_dir, f"{sb_match_id}_{sc_match_id}_main_data.csv")
#         preprocessing_df.to_csv(output_file_path,index=False)
#     elif data_provider == 'pff_fc':
#         print('not yet')
#         output_file_path = os.path.join(base_dir, 'phase_data', data_provider, match_id, f"{match_id}_main_data.csv")
#         preprocessing_df.to_csv(output_file_path,index=False)
#     print(f"✅ All period tracking data saved successfully at {output_file_path}.")


# if __name__ == '__main__':
#     main()
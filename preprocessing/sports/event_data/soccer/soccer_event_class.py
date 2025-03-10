import os
import pandas as pd
from tqdm import tqdm
import warnings  # <-- Added to issue deprecation warnings

if __name__ == '__main__':
    import soccer_load_data
    import soccer_processing
    import soccer_tracking_data
else:
    from . import soccer_load_data
    from . import soccer_processing
    from . import soccer_tracking_data

# Create a class to wrap the data source
class Soccer_event_data:
    def __init__(self, data_provider, event_path=None, match_id=None, tracking_home_path=None, tracking_away_path=None,
                 tracking_path=None, meta_data=None, statsbomb_api_args=[], statsbomb_match_id=None, skillcorner_match_id=None,
                 max_workers=1, match_id_df=None, statsbomb_event_dir=None, skillcorner_tracking_dir=None, skillcorner_match_dir=None,
                 preprocess_method=None, sb360_path=None, wyscout_matches_path=None, st_track_path=None, st_meta_path=None,
                 verbose=False, preprocess_tracking=False):
        self.data_provider = data_provider
        self.event_path = event_path
        self.match_id = match_id
        self.tracking_home_path = tracking_home_path
        self.tracking_away_path = tracking_away_path
        self.tracking_path = tracking_path  
        self.meta_data = meta_data
        self.statsbomb_api_args = statsbomb_api_args
        self.statsbomb_match_id = statsbomb_match_id
        self.sb360_path = sb360_path
        self.skillcorner_match_id = skillcorner_match_id
        self.max_workers = max_workers
        self.match_id_df = match_id_df
        self.statsbomb_event_dir = statsbomb_event_dir
        self.skillcorner_tracking_dir = skillcorner_tracking_dir
        self.skillcorner_match_dir = skillcorner_match_dir
        self.preprocess_method = preprocess_method
        self.wyscout_matches_path = wyscout_matches_path
        self.st_track_path = st_track_path
        self.st_meta_path = st_meta_path
        self.preprocess_tracking = preprocess_tracking
        self.verbose = verbose
        self.call_preprocess = False

    def load_data_single_file(self):
        # Based on the data provider, load the data using the appropriate function
        if self.data_provider == 'datafactory':
            df = soccer_load_data.load_datafactory(self.event_path)
        elif self.data_provider == 'metrica':
            df = soccer_load_data.load_metrica(self.event_path, match_id=self.match_id,
                                               tracking_home_path=self.tracking_home_path, tracking_away_path=self.tracking_away_path)
        elif self.data_provider == 'opta':
            df = soccer_load_data.load_opta(self.event_path, match_id=self.match_id)
        elif self.data_provider == 'robocup_2d':
            raise NotImplementedError(
                "RoboCup 2D data provider is deprecated due to outdated data formats and error-prone filename matching. "
                "Please use a more comprehensive data source (e.g., Statsbomb or Wyscout)."
            )
        elif self.data_provider == 'sportec':
            df = soccer_load_data.load_sportec(self.event_path, tracking_path=self.tracking_path, meta_path=self.meta_data)
        elif self.data_provider == 'statsbomb':
            df = soccer_load_data.load_statsbomb(self.event_path, sb360_path=self.sb360_path, match_id=self.statsbomb_match_id, *self.statsbomb_api_args)
        elif self.data_provider == 'statsbomb_skillcorner':
            df = soccer_load_data.load_statsbomb_skillcorner(
                statsbomb_event_dir=self.statsbomb_event_dir,
                skillcorner_tracking_dir=self.skillcorner_tracking_dir,
                skillcorner_match_dir=self.skillcorner_match_dir,
                statsbomb_match_id=self.statsbomb_match_id,
                skillcorner_match_id=self.skillcorner_match_id
            )
            if self.preprocess_tracking and not self.call_preprocess:
                df = soccer_tracking_data.statsbomb_skillcorner_tracking_data_preprocessing(df)
            if self.preprocess_method is not None and not self.call_preprocess:
                df = soccer_tracking_data.statsbomb_skillcorner_event_data_preprocessing(df, process_event_coord=False)
        elif self.data_provider == 'wyscout':
            df = soccer_load_data.load_wyscout(self.event_path, self.wyscout_matches_path)
        elif self.data_provider == 'datastadium':
            df = soccer_load_data.load_datastadium(self.event_path, self.tracking_home_path, self.tracking_away_path)
        elif self.data_provider == 'bepro':
            df = soccer_load_data.load_soccertrack(self.event_path, self.st_track_path, self.st_meta_path, self.verbose)
        else:
            raise ValueError('Data provider not supported or not found')
        return df
    
    def load_data(self):
        print(f'Loading data from {self.data_provider}')
        # Check if event_path is a file or directory, with special handling per provider
        if ((self.event_path is not None and os.path.isfile(self.event_path)) and self.data_provider != 'statsbomb') or \
           (self.data_provider == 'statsbomb' and self.statsbomb_match_id is None and os.path.isfile(self.event_path)) or \
           (self.data_provider == 'statsbomb_skillcorner' and self.statsbomb_match_id is not None):
            df = self.load_data_single_file()
        elif (self.event_path is not None and os.path.isdir(self.event_path)) or self.data_provider == 'statsbomb' or \
             (self.data_provider == 'statsbomb_skillcorner' and self.statsbomb_match_id is None and self.skillcorner_match_id is None):
            if self.data_provider == 'statsbomb_skillcorner':
                pass
            elif self.data_provider in ['datafactory','opta','wyscout']:
                event_path = self.event_path
                files = sorted(os.listdir(self.event_path))
                files = [f for f in files if not f.startswith('.')]
                out_df_list = []
                if self.data_provider == "opta":
                    if self.match_id is None:
                        match_id = self.match_id
                elif self.data_provider == "wyscout":
                    matches_path = self.wyscout_matches_path
                count = 0
                for f in tqdm(files, total=len(files)):
                    if self.data_provider == "opta":
                        if self.match_id is None:
                            self.match_id = match_id[count]
                        else:
                            self.match_id = count
                        count += 1
                    elif self.data_provider == "wyscout":
                        self.wyscout_matches_path = os.path.join(matches_path, f.replace("events_", "matches_"))
                    self.event_path = os.path.join(event_path, f)
                    df = self.load_data_single_file()
                    out_df_list.append(df)
                df = pd.concat(out_df_list)
                self.event_path = event_path
                if self.data_provider == "opta":
                    self.match_id = match_id
                elif self.data_provider == "wyscout":
                    self.wyscout_matches_path = matches_path
            elif self.data_provider in ['metrica', 'robocup_2d', 'sportec']:
                if self.data_provider == 'robocup_2d':
                    warnings.warn(
                        "RoboCup 2D data provider is deprecated. Its outdated data format and reliance on filename-based matching "
                        "make it less robust. Please consider using a more comprehensive data source.",
                        DeprecationWarning
                    )
                print('Warning: Event data and tracking data will be matched via the file name')
                event_path = self.event_path
                files = sorted(os.listdir(self.event_path))
                files = [f for f in files if not f.startswith('.')]
                out_df_list = []
                if self.data_provider == 'metrica':
                    tracking_home_path = self.tracking_home_path
                    tracking_away_path = self.tracking_away_path
                    for f in files:
                        self.event_path = os.path.join(event_path, f)
                        self.tracking_home_path = os.path.join(tracking_home_path, f.replace("RawEventsData", "RawTrackingData_Home_Team"))
                        self.tracking_away_path = os.path.join(tracking_away_path, f.replace("RawEventsData", "RawTrackingData_Away_Team"))
                        if os.path.isfile(self.tracking_home_path) and os.path.isfile(self.tracking_away_path):
                            df = self.load_data_single_file()
                            out_df_list.append(df)
                        else:
                            print(f'Tracking data not found for {f}')
                    df = pd.concat(out_df_list)
                    self.event_path = event_path
                    self.tracking_home_path = tracking_home_path
                    self.tracking_away_path = tracking_away_path
                elif self.data_provider == 'robocup_2d':
                    tracking_path = self.tracking_path
                    for f in files:
                        self.event_path = os.path.join(event_path, f)
                        self.tracking_path = os.path.join(tracking_path, f.replace("pass", ""))
                        self.match_id = f.replace("pass", "").replace(".csv", "")
                        if os.path.isfile(self.tracking_path):
                            df = self.load_data_single_file()
                            out_df_list.append(df)
                        else:
                            print(f'Tracking data not found for {f}. Filename-based matching is unreliable.')
                    df = pd.concat(out_df_list)
                    self.event_path = event_path
                    self.tracking_path = tracking_path
                    self.match_id = None
                elif self.data_provider == 'sportec':
                    tracking_path = self.tracking_path
                    meta_path = self.meta_data
                    for f in files:
                        self.event_path = os.path.join(event_path, f)
                        self.tracking_path = os.path.join(tracking_path, f.replace("events", "positional"))
                        self.meta_data = os.path.join(meta_path, f.replace("events", "meta"))
                        if os.path.isfile(self.tracking_path) and os.path.isfile(self.meta_data):
                            df = self.load_data_single_file()
                            out_df_list.append(df)
                        else:
                            print(f'Tracking data or Meta data not found for {f}')
                    df = pd.concat(out_df_list)
                    self.event_path = event_path
                    self.tracking_path = tracking_path
                    self.meta_data = meta_path
            elif self.data_provider == 'statsbomb':
                raise NotImplementedError("Statsbomb data provider loading from directory is not yet implemented.")
            elif self.data_provider == "datastadium":
                raise NotImplementedError("Datastadium data provider loading from directory is not yet implemented.")
        else:
            raise ValueError('Event path is not a valid file or directory')
        print(f'Loaded data from {self.data_provider}')
        return df

if __name__ == '__main__':
    # Example test for the deprecated soccertrack provider is unchanged
    soccer_track_event_path = "/data_pool_1/soccertrackv2/2024-03-18/Event/event.csv"
    soccer_track_tracking_path = "/data_pool_1/soccertrackv2/2024-03-18/Tracking/tracking.xml"
    soccer_track_meta_path = "/data_pool_1/soccertrackv2/2024-03-18/Tracking/meta.xml"
    df_soccertrack = Soccer_event_data('soccertrack', soccer_track_event_path,
                                       st_track_path=soccer_track_tracking_path,
                                       st_meta_path=soccer_track_meta_path,
                                       verbose=True).load_data()
    df_soccertrack.to_csv(os.getcwd() + "/test/sports/event_data/data/soccertrack/test_load_soccer_event_class.csv", index=False)
    print("-----------done-----------")

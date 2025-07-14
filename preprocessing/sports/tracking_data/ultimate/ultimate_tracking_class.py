from .ultimatetrack_preprocessing.preprocessing import process_tracking_data as process_ultimatetrack_tracking_data

class Ultimate_tracking_data:
    @staticmethod
    def process_ultimatetrack_tracking_data(*args, **kwargs):
        tracking_offense, tracking_defense, team_info_df = process_ultimatetrack_tracking_data(*args, **kwargs)
        return tracking_offense, tracking_defense, team_info_df

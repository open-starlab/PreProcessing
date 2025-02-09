from .datastadium_preprocessing.preprocessing import process_tracking_data as process_datastadium_tracking_data_fc

class Soccer_tracking_data:
    @staticmethod
    def process_datadium_tracking_data(*args, **kwargs):
        tracking_home, tracking_away, jurseynum_df = process_datastadium_tracking_data_fc(*args, **kwargs)
        return tracking_home, tracking_away, jurseynum_df
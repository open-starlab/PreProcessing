from .ufa_preprocessing.preprocessing import preprocessing_for_ufa
from .ultimatetrack_preprocessing.preprocessing import preprocessing_for_ultimatetrack


class Ultimate_tracking_data:
    def __init__(self, data_provider, data_path, *args, **kwargs):
        self.data_provider = data_provider
        self.data_path = data_path

    def preprocessing(self):
        if self.data_provider == "UltimateTrack":
            tracking_offense, tracking_defense, events_df = (
                preprocessing_for_ultimatetrack(self.data_path)
            )
        elif self.data_provider == "UFA":
            tracking_offense, tracking_defense, events_df = preprocessing_for_ufa(
                self.data_path
            )

        return tracking_offense, tracking_defense, events_df

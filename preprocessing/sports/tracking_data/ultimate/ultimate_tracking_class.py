from .ultimatetrack_preprocessing.preprocessing import (
    preprocessing_for_ufa,
    preprocessing_for_ultimatetrack,
)


class Ultimate_tracking_data:
    def __init__(self, data_provider, data_path, *args, **kwargs):
        self.data_provider = data_provider
        self.data_path = data_path

    def preprocessing(self, game_id, test=False):
        if self.data_provider == "UltimateTrack":
            tracking_home, tracking_away, events_df = preprocessing_for_ultimatetrack(
                game_id, self.data_path
            )
        elif self.data_provider == "UFA":
            tracking_home, tracking_away, events_df = preprocessing_for_ufa(
                game_id, self.data_path
            )

        return tracking_home, tracking_away, events_df

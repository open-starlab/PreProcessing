from .soccer.soccer_tracking_class import Soccer_tracking_data
from .ultimate.ultimate_tracking_class import Ultimate_tracking_data


class Tracking_data:
    soccer_data_provider = ["soccer"]
    ultimate_data_provider = ["UltimateTrack", "UFA"]
    handball_data_provider = []
    rocket_league_data_provider = []

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.soccer_data_provider:
            return Soccer_tracking_data(*args, **kwargs)
        elif data_provider in cls.ultimate_data_provider:
            return Ultimate_tracking_data(*args, **kwargs)
        elif data_provider in cls.handball_data_provider:
            raise NotImplementedError("Handball event data not implemented yet")
        elif data_provider in cls.rocket_league_data_provider:
            raise NotImplementedError("Rocket League event data not implemented yet")
        else:
            raise ValueError(f"Unknown data provider: {data_provider}")


if __name__ == "__main__":
    import os

    # Test Soccer tracking data
    print("Testing Soccer tracking data...")
    game_id = 0  # Select the index from the list of files in the data_path.
    data_path = os.getcwd() + "/test/sports/event_data/data/datastadium/"

    try:
        # Call the function for soccer directly
        soccer_tracker = Soccer_tracking_data()
        tracking_home, tracking_away, jerseynum_df = (
            soccer_tracker.process_datadium_tracking_data(game_id, data_path, test=True)
        )

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        tracking_home.to_csv(
            os.getcwd()
            + "/test/sports/event_data/data/datastadium/test_tracking_home.csv",
            index=False,
        )
        tracking_away.to_csv(
            os.getcwd()
            + "/test/sports/event_data/data/datastadium/test_tracking_away.csv",
            index=False,
        )
        jerseynum_df.to_csv(
            os.getcwd() + "/test/sports/event_data/data/datastadium/test_jerseynum.csv",
            index=False,
        )
        print("Soccer test completed successfully!")
    except Exception as e:
        print(f"Soccer test failed: {e}")

    # Test Ultimate Track data
    print("\nTesting Ultimate Track data...")
    ultimate_game_id = 0  # Select the first CSV file
    ultimate_data_path = os.getcwd() + "/test/sports/tracking_data/data/ultimatetrack/"

    try:
        # Call the function for Ultimate Track directly
        ultimate_tracker = Ultimate_tracking_data()
        tracking_offense, tracking_defense, team_info_df = (
            ultimate_tracker.process_ultimatetrack_tracking_data(
                ultimate_game_id, ultimate_data_path, test=True
            )
        )

        # Create output directory if it doesn't exist
        os.makedirs(ultimate_data_path, exist_ok=True)

        tracking_offense.to_csv(
            ultimate_data_path + "test_tracking_offense.csv", index=False
        )
        tracking_defense.to_csv(
            ultimate_data_path + "test_tracking_defense.csv", index=False
        )
        team_info_df.to_csv(ultimate_data_path + "test_team_info.csv", index=False)
        print("Ultimate Track test completed successfully!")
        print(f"Ultimate Track data path: {ultimate_data_path}")
    except Exception as e:
        print(f"Ultimate Track test failed: {e}")

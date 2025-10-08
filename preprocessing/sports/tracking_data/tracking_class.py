class Tracking_data:
    soccer_data_provider = ["soccer"]
    ultimate_data_provider = ["UltimateTrack", "UFA"]
    handball_data_provider = []
    rocket_league_data_provider = []

    def __new__(cls, data_provider, *args, **kwargs):
        if data_provider in cls.soccer_data_provider:
            from .soccer.soccer_tracking_class import Soccer_tracking_data
            return Soccer_tracking_data(*args, **kwargs)
        elif data_provider in cls.ultimate_data_provider:
            from .ultimate.ultimate_tracking_class import Ultimate_tracking_data
            return Ultimate_tracking_data(*args, **kwargs)
        elif data_provider in cls.handball_data_provider:
            raise NotImplementedError("Handball event data not implemented yet")
        elif data_provider in cls.rocket_league_data_provider:
            raise NotImplementedError("Rocket League event data not implemented yet")
        else:
            raise ValueError(f"Unknown data provider: {data_provider}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Test tracking data processing")
    parser.add_argument(
        "data_provider",
        type=str,
        default="soccer",
        help="Data provider to use (e.g., 'soccer', 'UltimateTrack', 'UFA')",
    )
    args = parser.parse_args()

    data_provider = args.data_provider

    if data_provider in Tracking_data.soccer_data_provider:
        # Test Soccer tracking data
        print("Testing Soccer tracking data...")
        game_id = 0  # Select the index from the list of files in the data_path.
        data_path = os.getcwd() + "/test/sports/event_data/data/datastadium/"

        try:
            # Call the function for soccer directly
            soccer_tracker = Soccer_tracking_data()
            tracking_home, tracking_away, jerseynum_df = (
                soccer_tracker.process_datadium_tracking_data(
                    game_id, data_path, test=True
                )
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
                os.getcwd()
                + "/test/sports/event_data/data/datastadium/test_jerseynum.csv",
                index=False,
            )
            print("Soccer test completed successfully!")
        except Exception as e:
            print(f"Soccer test failed: {e}")

    elif data_provider in Tracking_data.ultimate_data_provider:
        if data_provider == "UFA":
            # Test UFA data
            print("\nTesting UFA data...")
            data_path = os.getcwd() + "/test/sports/tracking_data/UFA/2_1.txt"

            try:
                # Call the function for UFA directly
                ufa_tracker = Ultimate_tracking_data("UFA", data_path)
                tracking_offense, tracking_defense, events_df = (
                    ufa_tracker.preprocessing()
                )

                # Create output directory if it doesn't exist
                output_dir = os.getcwd() + "/test/sports/tracking_data/UFA/metrica/"
                base_name = os.path.splitext(os.path.basename(data_path))[0]
                os.makedirs(output_dir, exist_ok=True)

                tracking_offense.to_csv(
                    os.path.join(output_dir, f"{base_name}_home.csv"), index=False
                )
                tracking_defense.to_csv(
                    os.path.join(output_dir, f"{base_name}_away.csv"), index=False
                )
                events_df.to_csv(
                    os.path.join(output_dir, f"{base_name}_events.csv"), index=False
                )
                print("UFA test completed successfully!")
                print(f"UFA data path: {data_path}")
            except Exception as e:
                print(f"UFA test failed: {e}")

        elif data_provider == "UltimateTrack":
            # Test Ultimate Track data
            print("\nTesting Ultimate Track data...")
            data_path = (
                os.getcwd() + "/test/sports/tracking_data/UltimateTrack/1_1_1.csv"
            )

            try:
                # Call the function for Ultimate Track directly
                ultimatetrack_tracker = Ultimate_tracking_data(
                    "UltimateTrack", data_path
                )
                tracking_offense, tracking_defense, events_df = (
                    ultimatetrack_tracker.preprocessing()
                )

                # Create output directory if it doesn't exist
                output_dir = (
                    os.getcwd() + "/test/sports/tracking_data/UltimateTrack/metrica/"
                )
                base_name = os.path.splitext(os.path.basename(data_path))[0]
                os.makedirs(output_dir, exist_ok=True)

                tracking_offense.to_csv(
                    os.path.join(output_dir, f"{base_name}_home.csv"), index=False
                )
                tracking_defense.to_csv(
                    os.path.join(output_dir, f"{base_name}_away.csv"), index=False
                )
                events_df.to_csv(
                    os.path.join(output_dir, f"{base_name}_events.csv"), index=False
                )
                print("Ultimate Track test completed successfully!")
                print(f"Ultimate Track data path: {data_path}")
            except Exception as e:
                print(f"Ultimate Track test failed: {e}")

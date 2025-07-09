from preprocessing.sports.SAR_data.soccer_SAR_class import SAR_data
import os

def test_qmix_robocup2d_preprocessing():
    data_path = "./data/robocup2d/"
    match_id = "20230515_001"  # Replace with actual match CSV filename (no .csv)
    config_path = "./example/config/preprocessing_qmix.json"

    # Ensure file exists before test
    assert os.path.exists(os.path.join(data_path, f"{match_id}.csv")), "CSV file missing"

    # Load data through SAR wrapper
    data = SAR_data(
        data_provider="robocup_2d",
        data_path=data_path,
        match_id=match_id,
        config_path=config_path
    )
    data.load_data()
    episodes = data.get_episodes()

    # Basic structure checks
    assert len(episodes) > 0, "No episodes found"
    ep0 = episodes[0]
    assert "obs" in ep0 and "state" in ep0 and "roles" in ep0
    assert isinstance(ep0["obs"][0], list)
    assert len(ep0["obs"][0][0]) == 7  # Feature vector: distance, angle, vel_x, vel_y, stamina, last_action, role_id

    print("✅ QMix RoboCup2D preprocessing test passed.")

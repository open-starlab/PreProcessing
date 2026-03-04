from pathlib import Path
from typing import Tuple

import pandas as pd


FEATURE_COLUMNS = [
    "space_score_mean",
    "pressure_index_mean",
    "stretch_index_mean",
    "line_height_rel_ball_mean",
    "line_height_abs_mean",
]


def _clean_team_name(value: str) -> str:
    return str(value).strip().replace("_", " ").title()


def load_feature_dataset(csv_path: str | None = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Load feature dataset from preprocessing output.

    Returns:
        X: feature matrix
        y: label series
        team: team series
        feature_df: full cleaned dataframe
    """
    project_root = Path(__file__).resolve().parents[2]
    candidate_paths = []

    if csv_path is not None:
        candidate_paths.append(Path(csv_path))

    candidate_paths.extend(
        [
            project_root / "output" / "features" / "features_df.csv",
            project_root / "preprocessing" / "output" / "features" / "features_df.csv",
            project_root.parent / "Defense_line" / "Barcelona_tracking_excels" / "combined_outputs" / "features_df.csv",
        ]
    )

    resolved_path = None
    for path in candidate_paths:
        if path.exists():
            resolved_path = path
            break

    if resolved_path is None:
        looked_at = "\n".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(
            "Could not find features_df.csv. Expected at output/features/features_df.csv. "
            f"Checked:\n{looked_at}"
        )

    feature_df = pd.read_csv(resolved_path)

    required_columns = FEATURE_COLUMNS + ["team_lost_possession", "label"]
    missing = [column for column in required_columns if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    feature_df = feature_df.copy()
    feature_df["team_lost_possession"] = feature_df["team_lost_possession"].astype(str).map(_clean_team_name)
    feature_df = feature_df.dropna(subset=required_columns).reset_index(drop=True)

    X = feature_df[FEATURE_COLUMNS].astype(float)
    y = feature_df["label"]
    team = feature_df["team_lost_possession"]

    return X, y, team, feature_df

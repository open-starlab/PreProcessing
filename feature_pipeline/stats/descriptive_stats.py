from pathlib import Path

import pandas as pd


def compute_descriptive_statistics(feature_df: pd.DataFrame, feature_columns: list[str], output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    desc = feature_df[feature_columns].describe().T
    desc["median"] = feature_df[feature_columns].median()
    desc["iqr"] = feature_df[feature_columns].quantile(0.75) - feature_df[feature_columns].quantile(0.25)
    desc = desc.reset_index().rename(columns={"index": "feature"})

    out_path = output_dir / "descriptive_statistics.csv"
    desc.to_csv(out_path, index=False)
    return desc

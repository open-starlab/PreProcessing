from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore


def compute_feature_ranges(feature_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    ranges = []
    for feature in feature_columns:
        series = feature_df[feature]
        ranges.append(
            {
                "feature": feature,
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "std": series.std(ddof=1),
                "q1": series.quantile(0.25),
                "q3": series.quantile(0.75),
            }
        )
    return pd.DataFrame(ranges)


def detect_outliers_zscore(feature_df: pd.DataFrame, feature_columns: list[str], threshold: float = 3.0) -> pd.DataFrame:
    records = []
    for feature in feature_columns:
        values = feature_df[feature].values
        z_values = np.abs(zscore(values, nan_policy="omit"))
        count = int(np.sum(z_values > threshold))
        records.append(
            {
                "feature": feature,
                "method": "zscore",
                "threshold": threshold,
                "outlier_count": count,
                "outlier_pct": (count / len(feature_df)) * 100,
            }
        )
    return pd.DataFrame(records)


def detect_outliers_iqr(feature_df: pd.DataFrame, feature_columns: list[str], multiplier: float = 1.5) -> pd.DataFrame:
    records = []
    for feature in feature_columns:
        series = feature_df[feature]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        low = q1 - multiplier * iqr
        high = q3 + multiplier * iqr
        count = int(((series < low) | (series > high)).sum())
        records.append(
            {
                "feature": feature,
                "method": "iqr",
                "threshold": multiplier,
                "outlier_count": count,
                "outlier_pct": (count / len(feature_df)) * 100,
                "lower_bound": low,
                "upper_bound": high,
            }
        )
    return pd.DataFrame(records)


def plot_feature_distributions(feature_df: pd.DataFrame, feature_columns: list[str], plot_dir: str | Path) -> tuple[Path, Path]:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    box_fig, box_axes = plt.subplots(1, len(feature_columns), figsize=(4 * len(feature_columns), 5))
    if len(feature_columns) == 1:
        box_axes = [box_axes]
    for idx, feature in enumerate(feature_columns):
        sns.boxplot(y=feature_df[feature], ax=box_axes[idx], color="#80b1d3")
        box_axes[idx].set_title(feature)
    box_fig.tight_layout()
    box_path = plot_dir / "outlier_boxplots.png"
    box_fig.savefig(box_path, dpi=200)
    plt.close(box_fig)

    hist_fig, hist_axes = plt.subplots(1, len(feature_columns), figsize=(4 * len(feature_columns), 5))
    if len(feature_columns) == 1:
        hist_axes = [hist_axes]
    for idx, feature in enumerate(feature_columns):
        sns.histplot(feature_df[feature], kde=True, ax=hist_axes[idx], color="#8dd3c7")
        hist_axes[idx].set_title(feature)
    hist_fig.tight_layout()
    hist_path = plot_dir / "outlier_histograms.png"
    hist_fig.savefig(hist_path, dpi=200)
    plt.close(hist_fig)

    return box_path, hist_path

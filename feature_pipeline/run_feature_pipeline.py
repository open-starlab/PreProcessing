#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_pipeline.explainability.shap_analysis import run_shap_analysis
from feature_pipeline.ml.model_comparison import run_model_comparison
from feature_pipeline.stats.descriptive_stats import compute_descriptive_statistics
from feature_pipeline.stats.outlier_analysis import (
    compute_feature_ranges,
    detect_outliers_iqr,
    detect_outliers_zscore,
    plot_feature_distributions,
)
from feature_pipeline.stats.two_way_anova import run_two_way_anova
from feature_pipeline.utils.data_loader import FEATURE_COLUMNS, load_feature_dataset
from feature_pipeline.utils.plotting import (
    plot_effect_sizes,
    plot_interaction_effects,
    plot_model_performance,
    plot_significance_heatmap,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "feature_pipeline" / "output"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FEATURE PIPELINE: STATISTICS + ML + SHAP")
    print("=" * 80)

    X, y, team, feature_df = load_feature_dataset()

    print(f"Dataset size: {len(feature_df)} sequences")
    print(f"Feature columns: {list(X.columns)}")
    print(f"Teams: {team.nunique()} | Labels: {y.nunique()}")

    descriptive_df = compute_descriptive_statistics(feature_df, FEATURE_COLUMNS, output_dir)

    ranges_df = compute_feature_ranges(feature_df, FEATURE_COLUMNS)
    z_out_df = detect_outliers_zscore(feature_df, FEATURE_COLUMNS)
    iqr_out_df = detect_outliers_iqr(feature_df, FEATURE_COLUMNS)
    outlier_summary_df = z_out_df.merge(
        iqr_out_df[["feature", "outlier_count", "outlier_pct"]],
        on="feature",
        suffixes=("_zscore", "_iqr"),
    )

    ranges_df.to_csv(output_dir / "feature_ranges.csv", index=False)
    outlier_summary_df.to_csv(output_dir / "outlier_summary.csv", index=False)
    plot_feature_distributions(feature_df, FEATURE_COLUMNS, plot_dir)

    anova_df, correction_df = run_two_way_anova(feature_df, FEATURE_COLUMNS, output_dir)

    performance_df, importance_df = run_model_comparison(X, y, output_dir, cv_splits=5)

    shap_df = run_shap_analysis(X, y, output_dir)

    plot_interaction_effects(feature_df, FEATURE_COLUMNS, plot_dir)
    plot_effect_sizes(anova_df, plot_dir)
    plot_significance_heatmap(correction_df, plot_dir)
    plot_model_performance(performance_df, plot_dir)

    significant = correction_df[correction_df["p_holm"] < 0.05]
    best_model_row = performance_df.iloc[0] if not performance_df.empty else None
    top_shap = (
        shap_df.groupby("feature", as_index=False)["mean_abs_shap"].mean().sort_values("mean_abs_shap", ascending=False)
        if not shap_df.empty
        else None
    )

    print("\nSummary")
    print("-" * 80)
    print(f"Feature ranges saved: {output_dir / 'feature_ranges.csv'}")
    print(f"Significant ANOVA rows (Holm < 0.05): {len(significant)}")
    if len(significant) > 0:
        print(significant[["feature", "source", "p_holm", "eta_squared"]].head(10).to_string(index=False))

    if best_model_row is not None:
        print(
            f"Best ML model: {best_model_row['model']} "
            f"(ROC-AUC={best_model_row['roc_auc']:.4f}, F1={best_model_row['f1']:.4f})"
        )

    if top_shap is not None and not top_shap.empty:
        print("Most important features (mean |SHAP|):")
        for _, row in top_shap.head(5).iterrows():
            print(f"  {row['feature']}: {row['mean_abs_shap']:.6f}")

    expected_outputs = [
        "anova_team_outcome_results.csv",
        "multiple_testing_corrections.csv",
        "model_performance_comparison.csv",
        "feature_importances.csv",
        "shap_feature_importance.csv",
    ]
    print("\nGenerated outputs:")
    for name in expected_outputs:
        print(f"  - {output_dir / name}")
    print(f"  - {plot_dir}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    np.random.seed(42)
    main()
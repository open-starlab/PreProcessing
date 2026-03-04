from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


def run_two_way_anova(feature_df: pd.DataFrame, feature_columns: list[str], output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import pingouin as pg
    except ImportError as exc:
        raise ImportError("pingouin is required for two-way ANOVA. Install with: pip install pingouin") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for feature in feature_columns:
        anova_table = pg.anova(
            data=feature_df,
            dv=feature,
            between=["team_lost_possession", "label"],
            detailed=True,
        )
        for _, row in anova_table.iterrows():
            rows.append(
                {
                    "feature": feature,
                    "source": row["Source"],
                    "F": row.get("F", None),
                    "p_value": row.get("p_unc", None),
                    "eta_squared": row.get("np2", None),
                }
            )

    results_df = pd.DataFrame(rows)
    anova_path = output_dir / "anova_team_outcome_results.csv"
    results_df.to_csv(anova_path, index=False)

    correction_df = _apply_multiple_corrections(results_df)
    correction_path = output_dir / "multiple_testing_corrections.csv"
    correction_df.to_csv(correction_path, index=False)

    return results_df, correction_df


def _apply_multiple_corrections(anova_results_df: pd.DataFrame) -> pd.DataFrame:
    out_df = anova_results_df.copy()
    
    test_df = anova_results_df.dropna(subset=["p_value"]).copy()
    if test_df.empty:
        return out_df

    pvals = test_df["p_value"].values

    _, bonf_p, _, _ = multipletests(pvals, method="bonferroni")
    _, holm_p, _, _ = multipletests(pvals, method="holm")
    _, fdr_p, _, _ = multipletests(pvals, method="fdr_bh")

    # Create a mapping from index to corrected values
    bonf_map = {idx: bonf_p[i] for i, idx in enumerate(test_df.index)}
    holm_map = {idx: holm_p[i] for i, idx in enumerate(test_df.index)}
    fdr_map = {idx: fdr_p[i] for i, idx in enumerate(test_df.index)}
    
    # Apply to all rows, NaN for rows without p_value
    out_df["p_bonferroni"] = out_df.index.map(bonf_map)
    out_df["p_holm"] = out_df.index.map(holm_map)
    out_df["p_fdr_bh"] = out_df.index.map(fdr_map)
    out_df["sig_bonferroni"] = out_df["p_bonferroni"] < 0.05
    out_df["sig_holm"] = out_df["p_holm"] < 0.05
    out_df["sig_fdr_bh"] = out_df["p_fdr_bh"] < 0.05
    return out_df

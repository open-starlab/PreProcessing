from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_interaction_effects(feature_df: pd.DataFrame, feature_columns: list[str], plot_dir: str | Path) -> Path:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(feature_columns), figsize=(4 * len(feature_columns), 5))
    if len(feature_columns) == 1:
        axes = [axes]

    for idx, feature in enumerate(feature_columns):
        grouped = (
            feature_df.groupby(["team_lost_possession", "label"])[feature]
            .mean()
            .reset_index()
        )
        sns.pointplot(
            data=grouped,
            x="team_lost_possession",
            y=feature,
            hue="label",
            dodge=True,
            ax=axes[idx],
        )
        axes[idx].set_title(f"Interaction: {feature}")
        axes[idx].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    out_path = plot_dir / "interaction_plots.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_effect_sizes(anova_df: pd.DataFrame, plot_dir: str | Path) -> Path:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    data = anova_df.dropna(subset=["eta_squared"]).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, x="feature", y="eta_squared", hue="source", ax=ax)
    ax.set_title("Effect Sizes (Eta Squared)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    out_path = plot_dir / "effect_size_plot.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_significance_heatmap(correction_df: pd.DataFrame, plot_dir: str | Path) -> Path:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    subset = correction_df[["feature", "source", "p_holm"]].copy()
    pivot = subset.pivot_table(index="feature", columns="source", values="p_holm")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, cmap="viridis_r", fmt=".3f", ax=ax)
    ax.set_title("Significance Heatmap (Holm-corrected p-values)")
    fig.tight_layout()

    out_path = plot_dir / "significance_heatmap.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_model_performance(performance_df: pd.DataFrame, plot_dir: str | Path) -> Path:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=performance_df, x="model", y="roc_auc", ax=ax)
    ax.set_title("Model ROC-AUC Comparison")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    out_path = plot_dir / "model_performance_bars.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

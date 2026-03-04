"""
Visualization utilities for GIRL pipeline results.

Plots recovered reward weights as bar charts with error bars.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List


def plot_reward_weights(
    weights: np.ndarray,
    std_weights: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    title: str = "Recovered Reward Weights",
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    show: bool = False
) -> str:
    """
    Plot recovered reward weights as a bar chart.
    
    Args:
        weights: Array of reward weights
        std_weights: Optional standard deviations (for error bars)
        feature_names: Names of reward features
        title: Plot title
        output_path: Path to save figure (if None, saves as reward_weights.png)
        figsize: Figure size (width, height)
        show: Whether to show plot interactively
    
    Returns:
        Path to saved figure
    """
    # Default feature names
    if feature_names is None:
        feature_names = [
            'stretch_index',
            'pressure_index',
            'space_score',
            'line_height_rel'
        ][:len(weights)]
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    x_pos = np.arange(len(weights))
    colors = sns.color_palette("husl", len(weights))
    
    bars = ax.bar(
        x_pos,
        weights,
        yerr=std_weights if std_weights is not None else None,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Customize plot
    ax.set_xlabel('Reward Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=0, ha='center')
    ax.set_ylim(0, max(weights) * 1.2 if max(weights) > 0 else 0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        height = bar.get_height()
        if std_weights is not None:
            label = f"{weight:.4f}\n±{std_weights[i]:.4f}"
        else:
            label = f"{weight:.4f}"
        
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (std_weights[i] if std_weights is not None else 0) + max(weights) * 0.02,
            label,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add sum annotation
    weight_sum = weights.sum()
    ax.text(
        0.98, 0.97,
        f"Sum = {weight_sum:.6f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = "./girl_pipeline/output/reward_weights.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)
    
    return str(output_path)


def plot_cross_validation_weights(
    all_weights: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Cross-Validation Weight Distribution",
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6),
    show: bool = False
) -> str:
    """
    Plot distribution of weights across cross-validation folds.
    
    Args:
        all_weights: Array of shape (n_folds, n_features)
        feature_names: Names of reward features
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size (width, height)
        show: Whether to show plot interactively
    
    Returns:
        Path to saved figure
    """
    # Default feature names
    if feature_names is None:
        feature_names = [
            'stretch_index',
            'pressure_index',
            'space_score',
            'line_height_rel'
        ][:all_weights.shape[1]]
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create box plot
    bp = ax.boxplot(
        [all_weights[:, i] for i in range(all_weights.shape[1])],
        labels=feature_names,
        patch_artist=True,
        widths=0.6
    )
    
    # Color boxes
    colors = sns.color_palette("husl", len(feature_names))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize plot
    ax.set_xlabel('Reward Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = "./girl_pipeline/output/cv_weight_distribution.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)
    
    return str(output_path)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Example weights
    test_weights = np.array([0.31, 0.28, 0.25, 0.16])
    test_std = np.array([0.02, 0.03, 0.01, 0.02])
    test_features = ['stretch_index', 'pressure_index', 'space_score', 'line_height_rel']
    
    # Plot 1: Simple bar chart
    print("\nGenerating reward weights plot...")
    plot_reward_weights(
        weights=test_weights,
        std_weights=test_std,
        feature_names=test_features,
        output_path="./girl_pipeline/output/reward_weights.png",
        show=False
    )
    
    # Plot 2: Cross-validation distribution
    print("\nGenerating cross-validation distribution plot...")
    test_cv_weights = np.random.normal(
        loc=test_weights,
        scale=test_std,
        size=(5, len(test_weights))
    )
    test_cv_weights = np.maximum(test_cv_weights, 0)
    test_cv_weights = test_cv_weights / test_cv_weights.sum(axis=1, keepdims=True)
    
    plot_cross_validation_weights(
        all_weights=test_cv_weights,
        feature_names=test_features,
        output_path="./girl_pipeline/output/cv_weight_distribution.png",
        show=False
    )
    
    print("\n✓ Test plots generated successfully")

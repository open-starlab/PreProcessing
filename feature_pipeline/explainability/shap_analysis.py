from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def run_shap_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path,
    random_state: int = 42,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for SHAP analysis. Install with: pip install shap") from exc

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required for SHAP analysis. Install with: pip install xgboost") from exc

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_encoded = LabelEncoder().fit_transform(y)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "XGBoost": XGBClassifier(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
        ),
    }

    shap_rows = []
    shap_summary_values = None
    shap_summary_data = None

    sample_size = min(1000, len(X))
    X_sample = X.sample(sample_size, random_state=random_state) if len(X) > sample_size else X

    for model_name, model in models.items():
        model.fit(X, y_encoded)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            target_shap = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            target_shap = shap_values

        target_shap = np.asarray(target_shap)
        if target_shap.ndim == 3:
            mean_abs_shap = np.abs(target_shap).mean(axis=(0, 2))
            summary_values = target_shap[:, :, 1] if target_shap.shape[2] > 1 else target_shap[:, :, 0]
        elif target_shap.ndim == 2:
            mean_abs_shap = np.abs(target_shap).mean(axis=0)
            summary_values = target_shap
        else:
            raise ValueError(
                f"Unexpected SHAP value shape for {model_name}: {target_shap.shape}. Expected 2D or 3D array."
            )

        for feature_name, value in zip(X.columns, mean_abs_shap):
            shap_rows.append(
                {
                    "model": model_name,
                    "feature": feature_name,
                    "mean_abs_shap": float(value),
                }
            )

        if shap_summary_values is None:
            shap_summary_values = summary_values
            shap_summary_data = X_sample
    shap_df = pd.DataFrame(shap_rows).sort_values(["model", "mean_abs_shap"], ascending=[True, False])
    shap_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    if shap_summary_values is not None and shap_summary_data is not None:
        import shap as _shap

        plt.figure(figsize=(8, 6))
        _shap.summary_plot(shap_summary_values, shap_summary_data, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_summary_bar.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 6))
        _shap.summary_plot(shap_summary_values, shap_summary_data, show=False)
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_beeswarm.png", dpi=200)
        plt.close()

    return shap_df

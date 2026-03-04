from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .model_evaluation import evaluate_model_cv


def _build_models(random_state: int = 42):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required for feature_pipeline. Install with: pip install xgboost") from exc

    models = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, kernel="rbf", random_state=random_state)),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=400, random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state),
    }

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(random_state=random_state, n_estimators=300)
    except Exception:
        pass

    models["XGBoost"] = XGBClassifier(
        random_state=random_state,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
    )

    return models


def run_model_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path,
    cv_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = _build_models(random_state=random_state)

    performance_rows = []
    feature_importance_rows = []

    y_encoded = LabelEncoder().fit_transform(y)

    for model_name, model in models.items():
        metrics = evaluate_model_cv(model, X, y, cv_splits=cv_splits, random_state=random_state)
        performance_rows.append({"model": model_name, **metrics})

        model.fit(X, y_encoded)
        estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            for feature_name, importance in zip(X.columns, importances):
                feature_importance_rows.append(
                    {
                        "model": model_name,
                        "feature": feature_name,
                        "importance": float(importance),
                    }
                )

    performance_df = pd.DataFrame(performance_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    importance_df = pd.DataFrame(feature_importance_rows)

    performance_df.to_csv(output_dir / "model_performance_comparison.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importances.csv", index=False)

    return performance_df, importance_df

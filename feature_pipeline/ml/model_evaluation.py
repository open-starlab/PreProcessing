from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder


def evaluate_model_cv(model, X, y, cv_splits: int = 5, random_state: int = 42) -> Dict[str, float]:
    y_encoded = LabelEncoder().fit_transform(y)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    y_proba = cross_val_predict(model, X, y_encoded, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_encoded, y_proba)),
        "accuracy": float(accuracy_score(y_encoded, y_pred)),
        "precision": float(precision_score(y_encoded, y_pred, zero_division=0)),
        "recall": float(recall_score(y_encoded, y_pred, zero_division=0)),
        "f1": float(f1_score(y_encoded, y_pred, zero_division=0)),
    }

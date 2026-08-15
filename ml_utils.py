from __future__ import annotations

import hashlib
import json
from pathlib import Path
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
TARGET_COLUMN = "target"
CLASS_LABELS = [0, 1]

MODEL_FILE_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

EVALUATION_CACHE: dict[str, tuple[list[dict[str, float]], dict[str, dict[str, Any]]]] = {}


def load_dataset() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN].astype(int)
    metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "dataset_source": "UCI Machine Learning Repository, accessed through scikit-learn",
        "target_column": TARGET_COLUMN,
        "feature_names": features.columns.tolist(),
        "class_names": dataset.target_names.tolist(),
        "class_labels": CLASS_LABELS,
        "n_features": int(features.shape[1]),
        "n_samples": int(frame.shape[0]),
    }
    return features, target, metadata


def build_pipeline(estimator, *, use_scaler: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def build_model_specs(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=random_state, solver="liblinear"),
            use_scaler=True,
        ),
        "Decision Tree": build_pipeline(
            DecisionTreeClassifier(random_state=random_state),
            use_scaler=False,
        ),
        "KNN": build_pipeline(
            KNeighborsClassifier(n_neighbors=5),
            use_scaler=True,
        ),
        "Naive Bayes": build_pipeline(
            GaussianNB(),
            use_scaler=False,
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            ),
            use_scaler=False,
        ),
    }


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def ensure_model_dir() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def slug_for_model(model_name: str) -> str:
    return MODEL_FILE_NAMES[model_name]


def get_model_path(model_name: str) -> Path:
    return MODEL_DIR / slug_for_model(model_name)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(to_builtin(data), indent=2), encoding="utf-8")


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def get_positive_class_scores(model, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.reshape(-1)
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    return model.predict(features)


def compute_classification_metrics(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    class_names: list[str],
) -> dict[str, Any]:
    predictions = model.predict(features)
    scores = get_positive_class_scores(model, features)
    metrics = {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": float(roc_auc_score(target, scores)) if target.nunique() > 1 else float("nan"),
        "Precision": float(precision_score(target, predictions, zero_division=0)),
        "Recall": float(recall_score(target, predictions, zero_division=0)),
        "F1": float(f1_score(target, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predictions)),
    }
    report = classification_report(
        target,
        predictions,
        labels=CLASS_LABELS,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(target, predictions, labels=CLASS_LABELS)
    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": predictions,
        "scores": scores,
    }


def train_and_export(random_state: int = 42) -> dict[str, Any]:
    features, target, metadata = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )

    model_dir = ensure_model_dir()
    model_specs = build_model_specs(random_state=random_state)

    metrics_by_model: dict[str, dict[str, Any]] = {}
    reports_by_model: dict[str, Any] = {}
    confusion_by_model: dict[str, Any] = {}

    for model_name, model in model_specs.items():
        model.fit(x_train, y_train)
        joblib.dump(model, get_model_path(model_name))
        result = compute_classification_metrics(
            model,
            x_test,
            y_test,
            class_names=metadata["class_names"],
        )
        metrics_by_model[model_name] = result["metrics"]
        reports_by_model[model_name] = result["classification_report"]
        confusion_by_model[model_name] = result["confusion_matrix"].tolist()

    test_frame = x_test.copy()
    test_frame[TARGET_COLUMN] = y_test.to_numpy()
    test_data_path = PROJECT_ROOT / "test_data.csv"
    test_frame.to_csv(test_data_path, index=False)

    summary_frame = pd.DataFrame.from_dict(metrics_by_model, orient="index").sort_index()
    summary_frame.to_csv(model_dir / "metrics.csv")

    project_metadata = {
        **metadata,
        "random_state": random_state,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "model_files": MODEL_FILE_NAMES,
        "metrics": metrics_by_model,
        "classification_reports": reports_by_model,
        "confusion_matrices": confusion_by_model,
        "test_data_file": "test_data.csv",
    }
    save_json(project_metadata, model_dir / "metadata.json")
    return project_metadata


def load_metadata() -> dict[str, Any]:
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "model/metadata.json is missing. Run train_models.py to generate the saved models first."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_metadata_cached() -> dict[str, Any]:
    return load_metadata()


def load_saved_models(model_names: list[str] | None = None) -> dict[str, Any]:
    metadata = load_metadata()
    selected_names = model_names or list(metadata["model_files"].keys())
    models: dict[str, Any] = {}
    for model_name in selected_names:
        models[model_name] = joblib.load(get_model_path(model_name))
    return models


@lru_cache(maxsize=8)
def load_saved_models_cached(model_names: tuple[str, ...]) -> dict[str, Any]:
    return load_saved_models(list(model_names))


def align_feature_frame(data_frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [column for column in feature_names if column not in data_frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    aligned = data_frame.loc[:, feature_names].copy()
    return aligned.apply(pd.to_numeric, errors="coerce")


def extract_target_series(data_frame: pd.DataFrame, target_column: str) -> pd.Series:
    if target_column not in data_frame.columns:
        raise ValueError(f"Missing target column: {target_column}")
    target = pd.to_numeric(data_frame[target_column], errors="coerce")
    if target.isna().any():
        raise ValueError(f"Target column '{target_column}' contains non-numeric values.")
    return target.astype(int)


def evaluation_cache_key(
    data_frame: pd.DataFrame,
    *,
    model_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    target_column: str,
    class_names: tuple[str, ...],
) -> str:
    relevant_columns = list(feature_names) + [target_column]
    relevant = data_frame.loc[:, relevant_columns].copy()
    digest = pd.util.hash_pandas_object(relevant, index=False).values.tobytes()
    payload = "\x1f".join(
        [
            *model_names,
            *feature_names,
            target_column,
            *class_names,
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload + digest).hexdigest()

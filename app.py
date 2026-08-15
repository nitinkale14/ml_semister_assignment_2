from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from ml_utils import (
    EVALUATION_CACHE,
    PROJECT_ROOT,
    align_feature_frame,
    compute_classification_metrics,
    evaluation_cache_key,
    extract_target_series,
    load_metadata_cached,
    load_saved_models_cached,
)

SummaryRow = dict[str, Any]
DetailedResult = dict[str, Any]


def evaluate_models_on_data(
    data_frame: pd.DataFrame,
    model_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    target_column: str,
    class_names: tuple[str, ...],
) -> tuple[list[SummaryRow], dict[str, DetailedResult]]:
    cache_key = evaluation_cache_key(
        data_frame,
        model_names=model_names,
        feature_names=feature_names,
        target_column=target_column,
        class_names=class_names,
    )
    cached = EVALUATION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    features = align_feature_frame(data_frame, list(feature_names))
    target = extract_target_series(data_frame, target_column)
    models = load_saved_models_cached(model_names)

    summary_rows: list[SummaryRow] = []
    detailed_results: dict[str, DetailedResult] = {}
    for model_name, model in models.items():
        result = compute_classification_metrics(
            model,
            features,
            target,
            class_names=list(class_names),
        )
        metrics = cast(dict[str, float], result["metrics"])
        summary_rows.append({"Model": model_name, **metrics})
        detailed_results[model_name] = result
    EVALUATION_CACHE[cache_key] = (summary_rows, detailed_results)
    return summary_rows, detailed_results


def format_percentage(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def render_metric_grid(metrics: Mapping[str, float]) -> None:
    row_one = st.columns(3)
    row_two = st.columns(3)
    entries = [
        ("Accuracy", format_percentage(metrics["Accuracy"])),
        ("AUC", format_percentage(metrics["AUC"])),
        ("Precision", format_percentage(metrics["Precision"])),
        ("Recall", format_percentage(metrics["Recall"])),
        ("F1 Score", format_percentage(metrics["F1"])),
        ("MCC", f'{metrics["MCC"]:.3f}'),
    ]
    for column, (label, value) in zip(row_one + row_two, entries, strict=True):
        column.metric(label, value)


def build_summary_table(summary_rows: list[SummaryRow]) -> pd.DataFrame:
    frame = pd.DataFrame(summary_rows).set_index("Model")
    return frame.sort_values(by="Accuracy", ascending=False)


def show_confusion_matrix(matrix: list[list[int]], class_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="ML Assignment 2", layout="wide")
    st.title("Machine Learning Assignment 2")
    st.caption("Binary classification comparison for the Breast Cancer Wisconsin (Diagnostic) dataset.")

    metadata = load_metadata_cached()
    model_names = tuple(metadata["model_files"].keys())

    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
        st.caption("If no file is uploaded, the repository test_data.csv is used.")
        selected_model = st.selectbox("Model", model_names)

    if uploaded_file is not None:
        data_frame = cast(pd.DataFrame, pd.read_csv(cast(Any, uploaded_file)))
    else:
        data_frame = cast(pd.DataFrame, pd.read_csv(PROJECT_ROOT / "test_data.csv"))

    feature_names = tuple(metadata["feature_names"])
    target_column = str(metadata["target_column"])
    class_names = tuple(metadata["class_names"])

    try:
        align_feature_frame(data_frame, list(feature_names))
        target = extract_target_series(data_frame, target_column)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(data_frame.head(10), width="stretch")

    with st.spinner("Evaluating models on the uploaded test data..."):
        summary_rows, detailed_results = evaluate_models_on_data(
            data_frame,
            model_names,
            feature_names,
            target_column,
            class_names,
        )

    summary_table = build_summary_table(summary_rows)
    best_model = summary_table["Accuracy"].idxmax()

    st.subheader("Model comparison")
    st.dataframe(summary_table.round(3), width="stretch")
    st.info(f"Best model on this test data: {best_model}")

    selected_result = detailed_results[selected_model]
    st.subheader(f"Detailed view: {selected_model}")
    selected_metrics = cast(dict[str, float], selected_result["metrics"])
    render_metric_grid(selected_metrics)

    left, right = st.columns([1, 1.1])
    with left:
        confusion_matrix = cast(Any, selected_result["confusion_matrix"])
        show_confusion_matrix(confusion_matrix.tolist(), list(class_names))
    with right:
        classification_report = cast(dict[str, Any], selected_result["classification_report"])
        report_frame = pd.DataFrame(classification_report).T
        st.dataframe(report_frame.round(3), width="stretch")

    predictions = pd.DataFrame(
        {
            "Actual": target.to_numpy(),
            "Predicted": cast(Any, selected_result["predictions"]),
            "Score": cast(Any, selected_result["scores"]),
        }
    )
    predictions["Actual Label"] = predictions["Actual"].map(dict(zip(metadata["class_labels"], class_names, strict=True)))
    predictions["Predicted Label"] = predictions["Predicted"].map(dict(zip(metadata["class_labels"], class_names, strict=True)))

    st.subheader("Prediction sample")
    st.dataframe(predictions.head(20).round(4), width="stretch")


if __name__ == "__main__":
    if get_script_run_ctx(suppress_warning=True) is None:
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ]
        raise SystemExit(subprocess.call(command))
    main()

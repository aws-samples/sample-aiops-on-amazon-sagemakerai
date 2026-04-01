"""
Evidently-based drift detection and model performance reporting.

Replaces custom matplotlib visualizations with Evidently's built-in
interactive HTML reports for data drift and classification metrics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from evidently import Report
from evidently.core.datasets import BinaryClassification, DataDefinition, Dataset
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset

logger = logging.getLogger(__name__)


def run_data_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Evidently DataDriftPreset report comparing baseline vs current data.

    Args:
        baseline_df: Reference/training data (numeric features only).
        current_df: Current/inference data (same columns as baseline).
        output_path: If provided, saves the HTML report to this path.

    Returns:
        Dictionary with:
            - 'snapshot': The Evidently report snapshot (call .save_html() etc.)
            - 'drift_detected': bool, whether overall drift was detected
            - 'drifted_columns_count': number of drifted columns
            - 'drifted_columns_share': share of drifted columns
            - 'per_column': dict mapping column name -> {'drift_score': float, 'drifted': bool}
    """
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=baseline_df, current_data=current_df)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(output_path)
        logger.info(f"Data drift report saved to {output_path}")

    # Extract structured results from the report dict
    metrics_list = snapshot.dict().get("metrics", [])
    result: Dict[str, Any] = {
        "snapshot": snapshot,
        "drift_detected": False,
        "drifted_columns_count": 0,
        "drifted_columns_share": 0.0,
        "per_column": {},
    }

    drift_share_threshold = 0.5  # default from DriftedColumnsCount

    for m in metrics_list:
        name = m.get("metric_name", "")
        config = m.get("config", {})
        value = m.get("value")

        if "DriftedColumnsCount" in name:
            count = value.get("count", 0) if isinstance(value, dict) else 0
            share = value.get("share", 0.0) if isinstance(value, dict) else 0.0
            drift_share_threshold = config.get("drift_share", 0.5)
            result["drifted_columns_count"] = int(count)
            result["drifted_columns_share"] = float(share)
            result["drift_detected"] = share >= drift_share_threshold

        elif "ValueDrift" in name:
            col = config.get("column", "unknown")
            threshold = config.get("threshold", 0.05)
            drift_score = float(value) if value is not None else 1.0
            result["per_column"][col] = {
                "drift_score": drift_score,
                "drifted": drift_score < threshold,  # p-value below threshold = drifted
            }

    logger.info(
        f"Data drift report: {result['drifted_columns_count']} drifted columns "
        f"({result['drifted_columns_share']:.1%}), overall drift: {result['drift_detected']}"
    )
    return result


def run_classification_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: str = "target",
    prediction_column: str = "prediction",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Evidently ClassificationPreset report comparing baseline vs current model performance.

    Both DataFrames must contain the target and prediction columns.

    Args:
        baseline_df: Reference data with target and prediction columns.
        current_df: Current data with target and prediction columns.
        target_column: Name of the ground-truth label column.
        prediction_column: Name of the predicted label column.
        output_path: If provided, saves the HTML report to this path.

    Returns:
        Dictionary with:
            - 'snapshot': The Evidently report snapshot
            - 'metrics': Raw metrics dict extracted from the report
    """
    data_def = DataDefinition(
        classification=[
            BinaryClassification(
                target=target_column,
                prediction_labels=prediction_column,
            )
        ]
    )

    ref_dataset = Dataset.from_pandas(baseline_df, data_definition=data_def)
    cur_dataset = Dataset.from_pandas(current_df, data_definition=data_def)

    report = Report(metrics=[ClassificationPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(output_path)
        logger.info(f"Classification report saved to {output_path}")

    result: Dict[str, Any] = {
        "snapshot": snapshot,
        "metrics": snapshot.dict().get("metrics", []),
    }

    logger.info("Classification report generated successfully")
    return result


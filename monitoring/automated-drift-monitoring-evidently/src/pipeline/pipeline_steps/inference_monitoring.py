"""
Inference Monitoring with Data Drift and Model Drift Detection.

⚠️ NOTE: This file uses awswrangler for MONITORING QUERIES (small aggregations).
For small analytical queries (<10K rows), awswrangler/pandas is appropriate.
For large-scale data processing, see preprocessing_pyspark.py.

This script:
- Analyzes inference predictions and ground truth
- Detects data drift (feature distribution changes)
- Detects model drift (performance degradation)
- Creates comprehensive visualizations
- Logs to MLflow for monitoring

**Architecture Note:**
- Monitoring queries return aggregated metrics (small results)
- awswrangler is efficient for this analytical use case
- For bulk preprocessing/transformations, use PySpark version
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve,
    precision_recall_curve
)

from sklearn.calibration import calibration_curve

# Visualization libraries - try to install if not available
VISUALIZATION_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    VISUALIZATION_AVAILABLE = True
    print("✓ Visualization libraries available")
except ImportError:
    print("⚠ Matplotlib not found, attempting to install...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib>=3.5.0', 'seaborn>=0.12.0', 'scipy>=1.10.0'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        VISUALIZATION_AVAILABLE = True
        print("✓ Visualization libraries installed successfully")
    except Exception as e:
        print(f"⚠ Could not install visualization libraries: {e}")
        print("  Monitoring will continue without visualizations")
        VISUALIZATION_AVAILABLE = False

# MLflow for logging
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Setup logging FIRST (before any logger calls)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Evidently for monitoring (optional)
try:
    from src.monitoring.evidently_monitor import (
        create_fraud_detection_report,
        extract_drift_metrics,
        get_drifted_features
    )
    from src.monitoring.evidently_utils import (
        log_evidently_report_to_mlflow,
        log_evidently_metrics
    )
    from src.monitoring.baseline_data_manager import load_baseline_from_s3
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.info("Evidently monitoring not available (install with: pip install evidently>=0.4.22)")


# Note: Visualization functions now return figure objects for MLflow logging
# MLflow's log_figure() API handles serialization properly


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for drift detection.

    PSI < 0.1: No significant change
    PSI < 0.2: Small change
    PSI >= 0.2: Significant change (drift detected)

    Args:
        expected: Baseline feature values (training data)
        actual: Current feature values (inference data)
        bins: Number of bins for histogram

    Returns:
        PSI value
    """
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates

    if len(breakpoints) < 2:
        return 0.0

    # Calculate histograms
    expected_hist, _ = np.histogram(expected, bins=breakpoints)
    actual_hist, _ = np.histogram(actual, bins=breakpoints)

    # Normalize to percentages
    expected_pct = expected_hist / len(expected)
    actual_pct = actual_hist / len(actual)

    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return psi


def calculate_ks_statistic(baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov statistic for distribution comparison.

    Args:
        baseline: Baseline feature values
        current: Current feature values

    Returns:
        Tuple of (KS statistic, p-value)
    """
    try:
        ks_stat, p_value = stats.ks_2samp(baseline, current)
        return float(ks_stat), float(p_value)
    except Exception as e:
        logger.warning(f"KS test failed: {e}")
        return 0.0, 1.0


def _legacy_detect_data_drift(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_names: List[str],
    threshold_psi: float = 0.2,
    threshold_ks: float = 0.05
) -> Dict[str, Any]:
    """
    Legacy drift detection using PSI and KS tests (deprecated - use Evidently).

    This function is maintained for backward compatibility and as a fallback
    when Evidently is not available.

    Args:
        baseline_data: Training data features
        current_data: Inference data features
        feature_names: List of feature names
        threshold_psi: PSI threshold for drift alert
        threshold_ks: KS p-value threshold for drift alert

    Returns:
        Dictionary with drift results
    """
    logger.info("Detecting data drift (legacy PSI/KS method)...")

    drift_results = {
        'features': {},
        'drifted_features': [],
        'drift_detected': False,
        'summary': {}
    }

    for feature in feature_names:
        if feature not in baseline_data.columns or feature not in current_data.columns:
            continue

        baseline_values = baseline_data[feature].dropna().values
        current_values = current_data[feature].dropna().values

        if len(baseline_values) == 0 or len(current_values) == 0:
            continue

        # Calculate PSI
        psi = calculate_psi(baseline_values, current_values)

        # Calculate KS statistic
        ks_stat, ks_pvalue = calculate_ks_statistic(baseline_values, current_values)

        # Calculate KS critical value
        ks_critical = calculate_ks_critical_value(len(baseline_values), len(current_values))

        # Determine drift status
        drift_psi = psi >= threshold_psi
        drift_ks = ks_pvalue < threshold_ks
        has_drift = drift_psi or drift_ks

        # Enhanced drift results with additional KS metrics
        drift_results['features'][feature] = {
            'psi': float(psi),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'ks_critical_value': float(ks_critical),
            'drift_detected': has_drift,
            'drift_method': 'ks' if drift_ks else ('psi' if drift_psi else 'none'),
            'baseline_mean': float(baseline_values.mean()),
            'current_mean': float(current_values.mean()),
            'baseline_std': float(baseline_values.std()),
            'current_std': float(current_values.std()),
            'baseline_median': float(np.median(baseline_values)),
            'current_median': float(np.median(current_values)),
        }

        if has_drift:
            drift_results['drifted_features'].append(feature)
            drift_results['drift_detected'] = True
            logger.warning(f"⚠ Drift detected in '{feature}': PSI={psi:.4f}, KS p-value={ks_pvalue:.4f}")

    # Summary statistics
    if len(drift_results['features']) > 0:
        psi_values = [f['psi'] for f in drift_results['features'].values()]
        ks_stats = [f['ks_statistic'] for f in drift_results['features'].values()]
        ks_pvalues = [f['ks_pvalue'] for f in drift_results['features'].values()]

        drift_results['summary'] = {
            'total_features': len(drift_results['features']),
            'drifted_features_count': len(drift_results['drifted_features']),
            'drift_percentage': len(drift_results['drifted_features']) / len(drift_results['features']) * 100,
            'avg_psi': float(np.mean(psi_values)),
            'max_psi': float(np.max(psi_values)),
            'median_psi': float(np.median(psi_values)),
            'avg_ks_statistic': float(np.mean(ks_stats)),
            'max_ks_statistic': float(np.max(ks_stats)),
            'median_ks_statistic': float(np.median(ks_stats)),
            'min_ks_pvalue': float(np.min(ks_pvalues))
        }

    logger.info(f"Data drift analysis complete: {len(drift_results['drifted_features'])} of {len(drift_results['features'])} features drifted")

    return drift_results


def detect_data_drift_evidently(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_names: List[str],
    mlflow_run_id: Optional[str] = None,
    threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Detect data drift using Evidently with comprehensive statistical tests.

    Args:
        baseline_data: Training data features (reference)
        current_data: Inference data features (current)
        feature_names: List of feature names to analyze
        mlflow_run_id: Optional MLflow run ID for logging
        threshold: Drift score threshold (default: 0.2)

    Returns:
        Dictionary with drift results compatible with legacy format

    Raises:
        ImportError: If Evidently is not available
    """
    if not EVIDENTLY_AVAILABLE:
        raise ImportError(
            "Evidently not available. Install with: pip install evidently>=0.4.22"
        )

    logger.info("Detecting data drift using Evidently...")
    logger.info(f"  Baseline: {baseline_data.shape}, Current: {current_data.shape}")

    try:
        # Select only specified features
        baseline_features = baseline_data[feature_names].copy()
        current_features = current_data[feature_names].copy()

        # Generate Evidently report
        report = create_fraud_detection_report(
            reference_data=baseline_features,
            current_data=current_features
        )

        # Extract drift metrics
        drift_metrics = extract_drift_metrics(report)

        # Get drifted features
        drifted_features = get_drifted_features(report, threshold=threshold)

        # Log to MLflow if active run
        if MLFLOW_AVAILABLE and mlflow.active_run():
            logger.info("Logging Evidently report to MLflow...")
            log_evidently_report_to_mlflow(report, artifact_path='drift_reports')
            log_evidently_metrics(report)

        # Build results in format compatible with legacy function
        drift_results = {
            'drift_detected': drift_metrics.get('dataset_drift_detected', 0) == 1,
            'drifted_features': drifted_features,
            'drift_metrics': drift_metrics,
            'evidently_report': report,  # Include full report for advanced analysis
            'summary': {
                'total_features': len(feature_names),
                'drifted_features_count': len(drifted_features),
                'drift_percentage': (len(drifted_features) / len(feature_names) * 100) if len(feature_names) > 0 else 0,
                'dataset_drift': drift_metrics.get('dataset_drift_detected', 0) == 1
            }
        }

        logger.info(f"✓ Evidently drift analysis complete")
        logger.info(f"  Dataset drift: {drift_results['drift_detected']}")
        logger.info(f"  Drifted features: {len(drifted_features)}/{len(feature_names)}")

        return drift_results

    except Exception as e:
        logger.error(f"Evidently drift detection failed: {e}")
        raise


def detect_data_drift(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_names: List[str],
    mlflow_run_id: Optional[str] = None,
    threshold_psi: float = 0.2,
    threshold_ks: float = 0.05,
    use_evidently: bool = True,
    fallback_to_legacy: bool = True
) -> Dict[str, Any]:
    """
    Detect data drift using Evidently (default) or legacy PSI/KS methods.

    This function automatically selects the best available drift detection method:
    1. Try Evidently if available and use_evidently=True
    2. Fallback to legacy PSI/KS if Evidently fails and fallback_to_legacy=True
    3. Raise error if no method succeeds

    Args:
        baseline_data: Training data features
        current_data: Inference data features
        feature_names: List of feature names
        mlflow_run_id: Optional MLflow run ID
        threshold_psi: PSI threshold for legacy method (default: 0.2)
        threshold_ks: KS p-value threshold for legacy method (default: 0.05)
        use_evidently: Try Evidently first (default: True)
        fallback_to_legacy: Fallback to legacy if Evidently fails (default: True)

    Returns:
        Dictionary with drift results

    Example:
        >>> # Use Evidently by default
        >>> drift = detect_data_drift(baseline, current, features)
        >>>
        >>> # Force legacy method
        >>> drift = detect_data_drift(baseline, current, features, use_evidently=False)
    """
    # Try Evidently first if enabled and available
    if use_evidently and EVIDENTLY_AVAILABLE:
        try:
            logger.info("Using Evidently for drift detection")
            return detect_data_drift_evidently(
                baseline_data=baseline_data,
                current_data=current_data,
                feature_names=feature_names,
                mlflow_run_id=mlflow_run_id,
                threshold=threshold_psi  # Use PSI threshold as Evidently threshold
            )
        except Exception as e:
            logger.error(f"Evidently drift detection failed: {e}")
            if fallback_to_legacy:
                logger.info("Falling back to legacy PSI/KS detection")
            else:
                raise

    # Use legacy method
    if not use_evidently or (fallback_to_legacy and EVIDENTLY_AVAILABLE):
        logger.info("Using legacy PSI/KS detection")
        return _legacy_detect_data_drift(
            baseline_data=baseline_data,
            current_data=current_data,
            feature_names=feature_names,
            threshold_psi=threshold_psi,
            threshold_ks=threshold_ks
        )

    # No method available
    raise RuntimeError(
        "No drift detection method available. Install Evidently with: "
        "pip install evidently>=0.4.22"
    )


def detect_model_drift(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    baseline_metrics: Dict[str, float],
    threshold_degradation: float = 0.05
) -> Dict[str, Any]:
    """
    Detect model drift by comparing performance metrics.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        baseline_metrics: Metrics from training (baseline)
        threshold_degradation: Acceptable performance drop (5% default)

    Returns:
        Dictionary with model drift results
    """
    logger.info("Detecting model drift...")

    # Calculate current metrics
    y_pred = (y_pred_proba >= 0.5).astype(int)

    current_metrics = {
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_true, y_pred_proba)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
    }

    # Compare with baseline
    drift_results = {
        'current_metrics': current_metrics,
        'baseline_metrics': baseline_metrics,
        'degradation': {},
        'drift_detected': False
    }

    for metric_name in ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1_score']:
        if metric_name in baseline_metrics:
            baseline_value = baseline_metrics[metric_name]
            current_value = current_metrics[metric_name]
            degradation = baseline_value - current_value
            degradation_pct = (degradation / baseline_value * 100) if baseline_value > 0 else 0

            drift_results['degradation'][metric_name] = {
                'absolute': float(degradation),
                'percentage': float(degradation_pct),
                'drift_detected': degradation > threshold_degradation
            }

            if degradation > threshold_degradation:
                drift_results['drift_detected'] = True
                logger.warning(f"⚠ Model drift detected in '{metric_name}': "
                             f"dropped from {baseline_value:.4f} to {current_value:.4f} "
                             f"({degradation_pct:.1f}% degradation)")

    logger.info(f"Model drift analysis complete: {'DRIFT DETECTED' if drift_results['drift_detected'] else 'No significant drift'}")

    return drift_results


def create_inference_visualizations(
    inference_data: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    ground_truth: np.ndarray,
    feature_names: List[str],
    output_dir: str
) -> Dict[str, Any]:
    """
    Create inference-time visualizations.

    Args:
        inference_data: Inference input features
        predictions: Binary predictions (0/1)
        probabilities: Fraud probabilities (0-1)
        ground_truth: Actual labels (if available)
        feature_names: List of feature names
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to matplotlib figure objects
    """
    # Skip visualizations if libraries not available
    if not VISUALIZATION_AVAILABLE:
        logger.info(f"⚠ Skipping {func_name} (libraries not available)")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {}  # Store figure objects, not paths
    sns.set_style("whitegrid")

    try:
        # 1. Prediction Probability Distribution
        logger.info("Creating prediction probability distribution...")
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(probabilities, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
        ax.set_xlabel('Fraud Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(alpha=0.3)

        # Add statistics
        stats_text = f'Mean: {probabilities.mean():.4f}\nMedian: {np.median(probabilities):.4f}\nStd: {probabilities.std():.4f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.close(fig)
        figures['prediction_distribution'] = fig

        # 2. Confidence Level Distribution
        logger.info("Creating confidence level distribution...")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Categorize by confidence
        high_conf = np.sum((probabilities >= 0.8) | (probabilities <= 0.2))
        medium_conf = np.sum(((probabilities > 0.2) & (probabilities < 0.4)) |
                            ((probabilities > 0.6) & (probabilities < 0.8)))
        low_conf = np.sum((probabilities >= 0.4) & (probabilities <= 0.6))

        categories = ['High Confidence\n(p≤0.2 or p≥0.8)',
                     'Medium Confidence\n(0.2<p<0.4 or 0.6<p<0.8)',
                     'Low Confidence\n(0.4≤p≤0.6)']
        counts = [high_conf, medium_conf, low_conf]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']

        ax.bar(categories, counts, color=colors, edgecolor='black')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Confidence Levels', fontsize=14, fontweight='bold', pad=20)

        for i, v in enumerate(counts):
            pct = v / len(probabilities) * 100
            ax.text(i, v + max(counts) * 0.01, f'{v:,}\n({pct:.1f}%)',
                   ha='center', fontweight='bold')

        plt.tight_layout()
        plt.close(fig)
        figures['confidence_distribution'] = fig

        # 3. Prediction Summary (if ground truth available)
        if ground_truth is not None and len(ground_truth) > 0:
            logger.info("Creating inference confusion matrix...")
            fig, ax = plt.subplots(figsize=(10, 8))

            cm = confusion_matrix(ground_truth, predictions)
            tn, fp, fn, tp = cm.ravel()

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                       xticklabels=['Non-Fraud', 'Fraud'],
                       yticklabels=['Non-Fraud', 'Fraud'],
                       ax=ax, linewidths=2, linecolor='black',
                       annot_kws={'size': 16, 'weight': 'bold'})

            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
            ax.set_title('Inference Batch Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

            total = cm.sum()
            ax.text(0.5, -0.12, f'Accuracy: {(tn+tp)/total*100:.2f}% | Precision: {tp/(tp+fp)*100 if (tp+fp)>0 else 0:.2f}% | Recall: {tp/(tp+fn)*100 if (tp+fn)>0 else 0:.2f}%',
                   transform=ax.transAxes, ha='center', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()
        plt.close(fig)
        figures['inference_confusion_matrix'] = fig

        # 4. Calibration Curve
        logger.info("Creating calibration curve...")
        fig, ax = plt.subplots(figsize=(10, 8))

        prob_true, prob_pred = calibration_curve(ground_truth, probabilities, n_bins=10)

        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label='Model')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

        ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
        ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.close(fig)
        figures['calibration_curve'] = fig

        # 5. Fraud Rate Over Time (if timestamp available)
        if 'timestamp' in inference_data.columns or 'transaction_timestamp' in inference_data.columns:
            logger.info("Creating fraud rate over time...")
            fig, ax = plt.subplots(figsize=(14, 6))

            ts_col = 'timestamp' if 'timestamp' in inference_data.columns else 'transaction_timestamp'
            df_temp = inference_data[[ts_col]].copy()
            df_temp['prediction'] = predictions
            df_temp['probability'] = probabilities
            df_temp[ts_col] = pd.to_datetime(df_temp[ts_col], errors='coerce')
            df_temp = df_temp.dropna(subset=[ts_col])

            if len(df_temp) > 0:
                # Group by hour or day depending on data span
                df_temp = df_temp.sort_values(ts_col)
                df_temp['time_bucket'] = df_temp[ts_col].dt.floor('H')  # Hourly buckets

                time_stats = df_temp.groupby('time_bucket').agg({
                    'prediction': ['sum', 'count', 'mean'],
                    'probability': 'mean'
                }).reset_index()

                time_stats.columns = ['time', 'fraud_count', 'total_count', 'fraud_rate', 'avg_probability']

                ax.plot(time_stats['time'], time_stats['fraud_rate'] * 100,
                       marker='o', linewidth=2, color='#e74c3c', label='Fraud Rate')
                ax2 = ax.twinx()
                ax2.plot(time_stats['time'], time_stats['avg_probability'],
                        marker='s', linewidth=2, color='#3498db', linestyle='--', label='Avg Probability')

                ax.set_xlabel('Time', fontsize=12, fontweight='bold')
                ax.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold', color='#e74c3c')
                ax2.set_ylabel('Average Fraud Probability', fontsize=12, fontweight='bold', color='#3498db')
                ax.set_title('Fraud Rate and Probability Over Time', fontsize=14, fontweight='bold', pad=20)

                ax.tick_params(axis='y', labelcolor='#e74c3c')
                ax2.tick_params(axis='y', labelcolor='#3498db')

                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

                ax.grid(alpha=0.3)
                plt.xticks(rotation=45)

                plt.tight_layout()
        plt.close(fig)
        figures['fraud_rate_over_time'] = fig

        logger.info(f"✓ Created {len(figures)} inference visualizations")

    except Exception as e:
        logger.error(f"Error creating inference visualizations: {e}")
        import traceback
        traceback.print_exc()

    return figures


def create_drift_visualizations(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    drift_results: Dict[str, Any],
    feature_names: List[str],
    output_dir: str
) -> Dict[str, Any]:
    """
    Create data drift visualizations.

    Args:
        baseline_data: Training data features
        current_data: Inference data features
        drift_results: Results from detect_data_drift()
        feature_names: List of feature names
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to matplotlib figure objects
    """
    # Skip visualizations if libraries not available
    if not VISUALIZATION_AVAILABLE:
        logger.info(f"⚠ Skipping {func_name} (libraries not available)")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {}  # Store figure objects, not paths
    sns.set_style("whitegrid")

    try:
        # 1. PSI Heatmap
        logger.info("Creating PSI heatmap...")
        fig, ax = plt.subplots(figsize=(12, 8))

        psi_values = [drift_results['features'][f]['psi'] for f in feature_names
                     if f in drift_results['features']]
        features_list = [f for f in feature_names if f in drift_results['features']]

        if len(psi_values) > 0:
            # Create color map based on PSI thresholds
            colors = []
            for psi in psi_values:
                if psi < 0.1:
                    colors.append('#2ecc71')  # Green - no drift
                elif psi < 0.2:
                    colors.append('#f39c12')  # Orange - small change
                else:
                    colors.append('#e74c3c')  # Red - drift detected

            y_pos = np.arange(len(features_list))
            ax.barh(y_pos, psi_values, color=colors, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features_list, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('PSI Value', fontsize=12, fontweight='bold')
            ax.set_title('Feature Drift Analysis (Population Stability Index)',
                        fontsize=14, fontweight='bold', pad=20)

            # Add threshold lines
            ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=1, label='Minor Change (0.1)')
            ax.axvline(x=0.2, color='red', linestyle='--', linewidth=1, label='Drift Alert (0.2)')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
        plt.close(fig)
        figures['psi_heatmap'] = fig

        # 2. Feature Distribution Comparison (top 6 drifted features)
        drifted_features = drift_results['drifted_features'][:6]

        if len(drifted_features) > 0:
            logger.info("Creating feature distribution comparison...")
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            axes = axes.ravel()

            for idx, feature in enumerate(drifted_features):
                if idx < 6 and feature in baseline_data.columns and feature in current_data.columns:
                    ax = axes[idx]

                    baseline_vals = baseline_data[feature].dropna()
                    current_vals = current_data[feature].dropna()

                    ax.hist(baseline_vals, bins=30, alpha=0.5, label='Training (Baseline)',
                           color='#3498db', edgecolor='black')
                    ax.hist(current_vals, bins=30, alpha=0.5, label='Inference (Current)',
                           color='#e74c3c', edgecolor='black')

                    psi = drift_results['features'][feature]['psi']
                    ax.set_title(f'{feature}\nPSI: {psi:.4f}', fontsize=10, fontweight='bold')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(alpha=0.3)

            # Hide unused subplots
            for idx in range(len(drifted_features), 6):
                axes[idx].axis('off')

            plt.suptitle('Feature Distribution Drift (Top 6 Drifted Features)',
                        fontsize=14, fontweight='bold', y=1.00)

            plt.tight_layout()
        plt.close(fig)
        figures['feature_drift_comparison'] = fig

        # 3. Drift Summary Dashboard
        logger.info("Creating drift summary dashboard...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # PSI distribution
        psi_vals = [drift_results['features'][f]['psi'] for f in drift_results['features']]
        ax1.hist(psi_vals, bins=20, edgecolor='black', color='#3498db', alpha=0.7)
        ax1.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='Drift Threshold')
        ax1.set_xlabel('PSI Value')
        ax1.set_ylabel('Number of Features')
        ax1.set_title('PSI Distribution Across Features', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Drift status pie chart
        drifted_count = drift_results['summary'].get('drifted_features_count', 0)
        stable_count = drift_results['summary'].get('total_features', 0) - drifted_count
        ax2.pie([stable_count, drifted_count], labels=['Stable', 'Drifted'],
               colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 12, 'weight': 'bold'})
        ax2.set_title('Feature Drift Status', fontweight='bold')

        # Mean shift (baseline vs current)
        top_features = list(drift_results['features'].keys())[:10]
        baseline_means = [drift_results['features'][f]['baseline_mean'] for f in top_features]
        current_means = [drift_results['features'][f]['current_mean'] for f in top_features]

        x = np.arange(len(top_features))
        width = 0.35
        ax3.bar(x - width/2, baseline_means, width, label='Baseline', color='#3498db')
        ax3.bar(x + width/2, current_means, width, label='Current', color='#e74c3c')
        ax3.set_xlabel('Feature')
        ax3.set_ylabel('Mean Value')
        ax3.set_title('Feature Mean Comparison (Top 10)', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
        ax3.legend()
        ax3.grid(alpha=0.3)

        # Summary text
        ax4.axis('off')
        summary_text = f"""
        DATA DRIFT SUMMARY
        ══════════════════════════════

        Total Features Analyzed: {drift_results['summary'].get('total_features', 0)}
        Features with Drift: {drifted_count}
        Drift Percentage: {drift_results['summary'].get('drift_percentage', 0):.1f}%

        PSI Statistics:
        • Average: {drift_results['summary'].get('avg_psi', 0):.4f}
        • Maximum: {drift_results['summary'].get('max_psi', 0):.4f}
        • Median: {drift_results['summary'].get('median_psi', 0):.4f}

        Status: {'⚠ DRIFT DETECTED' if drift_results['drift_detected'] else '✓ No Significant Drift'}
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Data Drift Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.close(fig)
        figures['drift_summary_dashboard'] = fig

        logger.info(f"✓ Created {len(figures)} drift visualizations")

    except Exception as e:
        logger.error(f"Error creating drift visualizations: {e}")
        import traceback
        traceback.print_exc()

    return figures


def calculate_ks_critical_value(n1: int, n2: int, alpha: float = 0.05) -> float:
    """
    Calculate the critical value for the Kolmogorov-Smirnov test.

    Args:
        n1: Sample size of first distribution
        n2: Sample size of second distribution
        alpha: Significance level (default: 0.05)

    Returns:
        Critical KS value at the given significance level
    """
    import math
    return math.sqrt(-0.5 * math.log(alpha / 2) * (n1 + n2) / (n1 * n2))


def create_ks_cdf_comparison_plots(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    drift_results: Dict[str, Any],
    feature_names: List[str],
    output_dir: Optional[str] = None,
    max_features: int = 10
) -> Dict[str, Any]:
    """
    Create CDF comparison plots for features with KS drift detection.

    This visualization shows:
    - Empirical CDFs for baseline (blue) and current (red) distributions
    - Vertical line marking the maximum KS distance
    - KS statistic and p-value annotations
    - Color-coded severity (green/orange/red)

    Args:
        baseline_data: Training data features (reference)
        current_data: Inference data features (current)
        drift_results: Dictionary with drift detection results
        feature_names: List of feature names
        output_dir: Optional directory to save plots
        max_features: Maximum number of features to plot (default: 10)

    Returns:
        Dictionary with figure objects
    """
    figures = {}

    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available - skipping KS CDF plots")
        return figures

    try:
        logger.info(f"Creating KS CDF comparison plots for top {max_features} drifted features...")

        # Get features with KS statistics
        features_with_ks = {}
        for feature, stats in drift_results.get('features', {}).items():
            if 'ks_statistic' in stats and feature in feature_names:
                features_with_ks[feature] = stats['ks_statistic']

        # Sort by KS statistic (highest first)
        sorted_features = sorted(features_with_ks.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:max_features]]

        if not top_features:
            logger.warning("No features with KS statistics found")
            return figures

        # Create grid of subplots
        n_features = len(top_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_features > 1 else axes

        for idx, feature in enumerate(top_features):
            ax = axes[idx] if n_features > 1 else axes[0]

            # Get data
            baseline_vals = baseline_data[feature].dropna().values
            current_vals = current_data[feature].dropna().values

            if len(baseline_vals) == 0 or len(current_vals) == 0:
                continue

            # Calculate empirical CDFs
            baseline_sorted = np.sort(baseline_vals)
            current_sorted = np.sort(current_vals)
            baseline_cdf = np.arange(1, len(baseline_sorted) + 1) / len(baseline_sorted)
            current_cdf = np.arange(1, len(current_sorted) + 1) / len(current_sorted)

            # Plot CDFs
            ax.plot(baseline_sorted, baseline_cdf, color='#3498db', linewidth=2,
                   label='Baseline', alpha=0.8)
            ax.plot(current_sorted, current_cdf, color='#e74c3c', linewidth=2,
                   label='Current', alpha=0.8)

            # Get KS statistics
            stats = drift_results['features'][feature]
            ks_stat = stats['ks_statistic']
            ks_pvalue = stats.get('ks_pvalue', stats.get('p_value', 0))

            # Determine color based on severity
            if ks_stat > 0.2:
                severity_color = '#e74c3c'  # Red - high drift
                severity = 'HIGH'
            elif ks_stat > 0.1:
                severity_color = '#f39c12'  # Orange - moderate drift
                severity = 'MODERATE'
            else:
                severity_color = '#2ecc71'  # Green - low drift
                severity = 'LOW'

            # Annotate
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Cumulative Probability', fontsize=10)
            ax.set_title(
                f'{feature}\nKS={ks_stat:.4f}, p={ks_pvalue:.4f} [{severity}]',
                fontsize=10, fontweight='bold', color=severity_color
            )
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1])

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('KS Test: CDF Comparison for Top Drifted Features',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plot_file = output_path / 'ks_cdf_comparison.png'
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            logger.info(f"✓ Saved KS CDF comparison plot: {plot_file}")

        plt.close(fig)
        figures['ks_cdf_comparison'] = fig

        logger.info(f"✓ Created KS CDF comparison plots for {n_features} features")

    except Exception as e:
        logger.error(f"Error creating KS CDF plots: {e}")
        import traceback
        traceback.print_exc()

    return figures


def create_ks_statistics_heatmap(
    drift_results: Dict[str, Any],
    feature_names: List[str],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a heatmap visualization of KS statistics across all features.

    This visualization shows:
    - Left panel: KS statistic bars (0-1 scale)
    - Right panel: -log10(p-value) bars (significance)
    - Color-coded by severity thresholds

    Args:
        drift_results: Dictionary with drift detection results
        feature_names: List of feature names
        output_dir: Optional directory to save plots

    Returns:
        Dictionary with figure objects
    """
    figures = {}

    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available - skipping KS heatmap")
        return figures

    try:
        logger.info("Creating KS statistics heatmap...")

        # Extract KS statistics
        ks_data = []
        for feature in feature_names:
            if feature in drift_results.get('features', {}):
                stats = drift_results['features'][feature]
                ks_stat = stats.get('ks_statistic', 0)
                ks_pvalue = stats.get('ks_pvalue', stats.get('p_value', 1))

                # Calculate -log10(p-value) for significance
                neg_log_pvalue = -np.log10(ks_pvalue) if ks_pvalue > 0 else 10

                ks_data.append({
                    'feature': feature,
                    'ks_statistic': ks_stat,
                    'neg_log_pvalue': neg_log_pvalue,
                    'p_value': ks_pvalue
                })

        if not ks_data:
            logger.warning("No KS statistics found")
            return figures

        # Sort by KS statistic
        ks_data = sorted(ks_data, key=lambda x: x['ks_statistic'], reverse=True)

        # Create two-panel visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(8, len(ks_data) * 0.3)))

        features = [d['feature'] for d in ks_data]
        ks_stats = [d['ks_statistic'] for d in ks_data]
        neg_log_pvals = [d['neg_log_pvalue'] for d in ks_data]

        # Panel 1: KS statistic bars
        colors_ks = ['#e74c3c' if ks > 0.2 else '#f39c12' if ks > 0.1 else '#2ecc71'
                     for ks in ks_stats]
        ax1.barh(features, ks_stats, color=colors_ks, edgecolor='black', alpha=0.8)
        ax1.axvline(x=0.1, color='orange', linestyle='--', linewidth=1.5,
                   label='Moderate Threshold')
        ax1.axvline(x=0.2, color='red', linestyle='--', linewidth=1.5,
                   label='High Threshold')
        ax1.set_xlabel('KS Statistic', fontsize=12, fontweight='bold')
        ax1.set_title('KS Statistics by Feature', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(alpha=0.3, axis='x')
        ax1.set_xlim([0, max(ks_stats) * 1.1])

        # Panel 2: Significance bars
        colors_pval = ['#e74c3c' if p > -np.log10(0.05) else '#95a5a6'
                       for p in neg_log_pvals]
        ax2.barh(features, neg_log_pvals, color=colors_pval, edgecolor='black', alpha=0.8)
        ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1.5,
                   label='Significance (p=0.05)')
        ax2.set_xlabel('-log10(p-value)', fontsize=12, fontweight='bold')
        ax2.set_title('Statistical Significance', fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(alpha=0.3, axis='x')

        plt.suptitle('KS Test: Statistics and Significance Across All Features',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plot_file = output_path / 'ks_statistics_heatmap.png'
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            logger.info(f"✓ Saved KS statistics heatmap: {plot_file}")

        plt.close(fig)
        figures['ks_statistics_heatmap'] = fig

        logger.info(f"✓ Created KS statistics heatmap for {len(features)} features")

    except Exception as e:
        logger.error(f"Error creating KS heatmap: {e}")
        import traceback
        traceback.print_exc()

    return figures


def create_model_drift_visualizations(
    model_drift_results: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Create model drift visualizations.

    Args:
        model_drift_results: Results from detect_model_drift()
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to matplotlib figure objects
    """
    # Skip visualizations if libraries not available
    if not VISUALIZATION_AVAILABLE:
        logger.info(f"⚠ Skipping {func_name} (libraries not available)")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {}  # Store figure objects, not paths
    sns.set_style("whitegrid")

    try:
        logger.info("Creating model drift comparison...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Metrics comparison
        metrics = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1_score']
        baseline_vals = [model_drift_results['baseline_metrics'].get(m, 0) for m in metrics]
        current_vals = [model_drift_results['current_metrics'].get(m, 0) for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        ax1.bar(x - width/2, baseline_vals, width, label='Training (Baseline)',
               color='#3498db', edgecolor='black')
        ax1.bar(x + width/2, current_vals, width, label='Inference (Current)',
               color='#e74c3c', edgecolor='black')

        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.1])

        # Add value labels
        for i, (b, c) in enumerate(zip(baseline_vals, current_vals)):
            ax1.text(i - width/2, b + 0.02, f'{b:.3f}', ha='center', fontsize=9)
            ax1.text(i + width/2, c + 0.02, f'{c:.3f}', ha='center', fontsize=9)

        # Degradation chart
        degradation_pcts = [model_drift_results['degradation'].get(m, {}).get('percentage', 0)
                           for m in metrics]
        colors_deg = ['#e74c3c' if d > 5 else '#f39c12' if d > 2 else '#2ecc71'
                     for d in degradation_pcts]

        ax2.barh(metrics, degradation_pcts, color=colors_deg, edgecolor='black')
        ax2.set_xlabel('Performance Degradation (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Performance Degradation', fontsize=14, fontweight='bold', pad=20)
        ax2.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Alert Threshold (5%)')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='x')
        ax2.set_yticklabels([m.replace('_', ' ').title() for m in metrics])

        # Add value labels
        for i, v in enumerate(degradation_pcts):
            ax2.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')

        plt.suptitle(f"Model Drift Analysis {'⚠ DRIFT DETECTED' if model_drift_results['drift_detected'] else '✓ No Significant Drift'}",
                    fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.close(fig)
        figures['model_drift_comparison'] = fig

        logger.info(f"✓ Created {len(figures)} model drift visualizations")

    except Exception as e:
        logger.error(f"Error creating model drift visualizations: {e}")
        import traceback
        traceback.print_exc()

    return figures


def log_monitoring_to_mlflow(
    inference_figures: Dict[str, Any],
    drift_figures: Dict[str, Any],
    model_drift_figures: Dict[str, Any],
    drift_results: Dict[str, Any],
    model_drift_results: Dict[str, Any],
    model_version: str = None,
    mlflow_run_id: str = None,
    inference_batch_size: int = None
) -> None:
    """
    Log monitoring visualizations and metrics to MLflow.

    Following MLflow best practices, uses mlflow.log_figure() to log
    matplotlib figure objects directly.

    Args:
        inference_figures: Inference visualization figure objects
        drift_figures: Data drift visualization figure objects
        model_drift_figures: Model drift visualization figure objects
        drift_results: Data drift analysis results
        model_drift_results: Model drift analysis results
        model_version: Model version being monitored (optional)
        mlflow_run_id: MLflow run ID of the model (optional)
        inference_batch_size: Number of inferences analyzed (optional)
    """
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not available - skipping logging")
        return

    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not mlflow_uri:
        logger.info("MLFLOW_TRACKING_URI not set - skipping logging")
        return

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'fraud-detection-monitoring')
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"monitoring-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            logger.info("Logging monitoring to MLflow...")

            # Log data drift metrics
            if 'summary' in drift_results:
                mlflow.log_metric("drift_features_count", drift_results['summary'].get('drifted_features_count', 0))
                mlflow.log_metric("drift_percentage", drift_results['summary'].get('drift_percentage', 0))
                mlflow.log_metric("avg_psi", drift_results['summary'].get('avg_psi', 0))
                mlflow.log_metric("max_psi", drift_results['summary'].get('max_psi', 0))

            # Log model drift metrics
            for metric, values in model_drift_results.get('degradation', {}).items():
                mlflow.log_metric(f"{metric}_degradation_pct", values.get('percentage', 0))

            # Log current performance
            for metric, value in model_drift_results.get('current_metrics', {}).items():
                mlflow.log_metric(f"current_{metric}", value)

            # Log drift status
            mlflow.log_param("data_drift_detected", drift_results.get('drift_detected', False))
            mlflow.log_param("model_drift_detected", model_drift_results.get('drift_detected', False))

            # Log model version tracking
            if model_version:
                mlflow.log_param("model_version", model_version)
                logger.info(f"  ✓ Tracked model version: {model_version}")
            if mlflow_run_id:
                mlflow.log_param("baseline_mlflow_run_id", mlflow_run_id)
                logger.info(f"  ✓ Linked to training run: {mlflow_run_id}")
            if inference_batch_size:
                mlflow.log_param("inference_batch_size", inference_batch_size)

            mlflow.log_param("monitoring_timestamp", datetime.now().isoformat())

            # Log all visualizations using MLflow's log_figure() API
            # This ensures proper serialization and display in MLflow UI
            all_figures = {**inference_figures, **drift_figures, **model_drift_figures}
            for fig_name, fig in all_figures.items():
                if fig is not None:
                    mlflow.log_figure(fig, f"{fig_name}.png")
                    logger.info(f"  ✓ Logged {plot_name}")

            logger.info(f"✓ Logged {len(all_plots)} visualizations to MLflow")

    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")


def load_inference_data_by_version(
    athena_database: str = 'fraud_detection',
    table_name: str = 'inference_responses',
    model_version: str = None,
    mlflow_run_id: str = None,
    start_date: str = None,
    end_date: str = None,
    output_s3: str = None
) -> pd.DataFrame:
    """
    Load inference data from Athena, optionally filtered by model version.

    Args:
        athena_database: Athena database name
        table_name: Athena table name
        model_version: Filter by model version (e.g., "1", "2", "v1.0")
        mlflow_run_id: Filter by MLflow run ID
        start_date: Filter by start date (ISO format: "2024-01-01")
        end_date: Filter by end date (ISO format: "2024-01-31")
        output_s3: S3 path for Athena query results

    Returns:
        DataFrame with inference data

    Example:
        # Get all inferences from model version 3
        df = load_inference_data_by_version(model_version="3")

        # Get inferences from specific run between dates
        df = load_inference_data_by_version(
            mlflow_run_id="abc123",
            start_date="2024-02-01",
            end_date="2024-02-10"
        )
    """
    try:
        import awswrangler as wr
    except ImportError:
        logger.error("awswrangler not installed. Install with: pip install awswrangler")
        raise

    # Build query
    query = f"SELECT * FROM {athena_database}.{table_name} WHERE 1=1"

    if model_version:
        query += f" AND model_version = '{model_version}'"
    if mlflow_run_id:
        query += f" AND mlflow_run_id = '{mlflow_run_id}'"
    if start_date:
        query += f" AND request_timestamp >= TIMESTAMP '{start_date}'"
    if end_date:
        query += f" AND request_timestamp < TIMESTAMP '{end_date}'"

    query += " ORDER BY request_timestamp DESC"

    logger.info(f"Querying Athena: {query}")

    # Execute query
    df = wr.athena.read_sql_query(
        sql=query,
        database=athena_database,
        s3_output=output_s3
    )

    logger.info(f"✓ Loaded {len(df)} inference records")

    if model_version:
        logger.info(f"  Filtered by model version: {model_version}")
    if mlflow_run_id:
        logger.info(f"  Filtered by MLflow run: {mlflow_run_id}")

    return df


def compare_model_versions(
    version_1: str,
    version_2: str,
    athena_database: str = 'fraud_detection',
    table_name: str = 'inference_responses',
    output_dir: str = './monitoring_output'
) -> Dict[str, Any]:
    """
    Compare performance of two model versions on their respective inferences.

    Args:
        version_1: First model version
        version_2: Second model version
        athena_database: Athena database name
        table_name: Athena table name
        output_dir: Directory to save comparison visualizations

    Returns:
        Dictionary with comparison results and plot paths

    Example:
        results = compare_model_versions(version_1="2", version_2="3")
        print(f"Version 2 ROC-AUC: {results['version_1_metrics']['roc_auc']}")
        print(f"Version 3 ROC-AUC: {results['version_2_metrics']['roc_auc']}")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load inference data for each version
    logger.info(f"Loading data for model version {version_1}...")
    df_v1 = load_inference_data_by_version(
        athena_database=athena_database,
        table_name=table_name,
        model_version=version_1
    )

    logger.info(f"Loading data for model version {version_2}...")
    df_v2 = load_inference_data_by_version(
        athena_database=athena_database,
        table_name=table_name,
        model_version=version_2
    )

    # Filter to records with ground truth
    df_v1_with_gt = df_v1[df_v1['ground_truth'].notna()]
    df_v2_with_gt = df_v2[df_v2['ground_truth'].notna()]

    logger.info(f"Version {version_1}: {len(df_v1_with_gt)} records with ground truth")
    logger.info(f"Version {version_2}: {len(df_v2_with_gt)} records with ground truth")

    # Calculate metrics for each version
    metrics_v1 = {}
    metrics_v2 = {}

    if len(df_v1_with_gt) > 0:
        y_true_v1 = df_v1_with_gt['ground_truth'].values
        y_prob_v1 = df_v1_with_gt['probability_fraud'].values
        y_pred_v1 = df_v1_with_gt['prediction'].values

        metrics_v1 = {
            'roc_auc': roc_auc_score(y_true_v1, y_prob_v1),
            'pr_auc': average_precision_score(y_true_v1, y_prob_v1),
            'precision': precision_score(y_true_v1, y_pred_v1, zero_division=0),
            'recall': recall_score(y_true_v1, y_pred_v1, zero_division=0),
            'f1_score': f1_score(y_true_v1, y_pred_v1, zero_division=0),
            'sample_count': len(df_v1_with_gt)
        }

    if len(df_v2_with_gt) > 0:
        y_true_v2 = df_v2_with_gt['ground_truth'].values
        y_prob_v2 = df_v2_with_gt['probability_fraud'].values
        y_pred_v2 = df_v2_with_gt['prediction'].values

        metrics_v2 = {
            'roc_auc': roc_auc_score(y_true_v2, y_prob_v2),
            'pr_auc': average_precision_score(y_true_v2, y_prob_v2),
            'precision': precision_score(y_true_v2, y_pred_v2, zero_division=0),
            'recall': recall_score(y_true_v2, y_pred_v2, zero_division=0),
            'f1_score': f1_score(y_true_v2, y_pred_v2, zero_division=0),
            'sample_count': len(df_v2_with_gt)
        }

    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics_list = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1_score']
    v1_vals = [metrics_v1.get(m, 0) for m in metrics_list]
    v2_vals = [metrics_v2.get(m, 0) for m in metrics_list]

    x = np.arange(len(metrics_list))
    width = 0.35

    # Metrics comparison
    axes[0].bar(x - width/2, v1_vals, width, label=f'Version {version_1}', color='steelblue', edgecolor='black')
    axes[0].bar(x + width/2, v2_vals, width, label=f'Version {version_2}', color='coral', edgecolor='black')
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Version Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace('_', ' ').title() for m in metrics_list], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.1])

    # Performance delta
    deltas = [(v2_vals[i] - v1_vals[i]) * 100 for i in range(len(metrics_list))]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]

    axes[1].barh(metrics_list, deltas, color=colors, edgecolor='black')
    axes[1].set_xlabel('Performance Change (%)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'V{version_2} vs V{version_1} Delta', fontsize=14, fontweight='bold')
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[1].grid(alpha=0.3, axis='x')
    axes[1].set_yticklabels([m.replace('_', ' ').title() for m in metrics_list])

    for i, v in enumerate(deltas):
        axes[1].text(v + 0.5 if v > 0 else v - 0.5, i, f'{v:+.1f}%',
                    va='center', ha='left' if v > 0 else 'right', fontweight='bold')

    plt.suptitle(f'Model Version Comparison: V{version_1} vs V{version_2}',
                fontsize=16, fontweight='bold', y=1.02)

    plot_file = output_path / f"version_comparison_v{version_1}_vs_v{version_2}.png"
    plt.tight_layout()
    save_plot_safely(plt, plot_file)
    plt.close()

    logger.info(f"✓ Created version comparison visualization: {plot_file}")

    return {
        'version_1': version_1,
        'version_2': version_2,
        'version_1_metrics': metrics_v1,
        'version_2_metrics': metrics_v2,
        'comparison_plot': str(plot_file)
    }


if __name__ == '__main__':
    # This is a utility script - main execution would be called from pipeline
    logger.info("Inference monitoring module loaded")

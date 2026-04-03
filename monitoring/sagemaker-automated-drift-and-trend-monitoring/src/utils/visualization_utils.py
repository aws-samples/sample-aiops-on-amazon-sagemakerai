"""
Visualization utilities for fraud detection pipeline.

Creates charts from Athena data and logs to MLflow for monitoring:
- ROC curves
- Confusion matrices
- Prediction distributions
- Latency heatmaps
- Drift comparisons
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

logger = logging.getLogger(__name__)

# Set style for all plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def create_roc_curve_from_athena(
    athena_client,
    endpoint_name: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Create ROC curve from Athena inference data.

    Args:
        athena_client: AthenaClient instance
        endpoint_name: SageMaker endpoint name
        start_date: Start date for data (default: 7 days ago)
        end_date: End date for data (default: now)
        save_path: Optional path to save figure

    Returns:
        Tuple of (matplotlib figure, metrics dict)
    """
    # Default date range
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=7)

    logger.info(f"Creating ROC curve for {endpoint_name} from {start_date} to {end_date}")

    # Query data from Athena
    query = f"""
    SELECT
        ground_truth,
        probability_fraud
    FROM fraud_detection.inference_responses
    WHERE endpoint_name = '{endpoint_name}'
      AND ground_truth IS NOT NULL
      AND request_timestamp BETWEEN TIMESTAMP '{start_date.isoformat()}'
                                AND TIMESTAMP '{end_date.isoformat()}'
    """

    df = athena_client.execute_query(query)

    if df.empty or len(df) < 10:
        logger.warning(f"Insufficient data for ROC curve: {len(df)} samples")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'Insufficient data\n(need ground truth labels)',
                ha='center', va='center', fontsize=16)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {endpoint_name}\nInsufficient Data')
        return fig, {'roc_auc': 0.0, 'sample_count': len(df)}

    # Calculate ROC curve
    y_true = df['ground_truth'].values
    y_scores = df['probability_fraud'].values

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')

    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {endpoint_name}\n{start_date.date()} to {end_date.date()}',
                fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")

    metrics = {
        'roc_auc': float(roc_auc),
        'sample_count': len(df),
    }

    return fig, metrics


def create_confusion_matrix_from_athena(
    athena_client,
    endpoint_name: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create confusion matrix from Athena inference data.

    Args:
        athena_client: AthenaClient instance
        endpoint_name: SageMaker endpoint name
        start_date: Start date for data
        end_date: End date for data
        threshold: Probability threshold for classification
        save_path: Optional path to save figure

    Returns:
        Tuple of (matplotlib figure, metrics dict)
    """
    # Default date range
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=7)

    logger.info(f"Creating confusion matrix for {endpoint_name}")

    # Query data
    query = f"""
    SELECT
        ground_truth,
        probability_fraud,
        prediction
    FROM fraud_detection.inference_responses
    WHERE endpoint_name = '{endpoint_name}'
      AND ground_truth IS NOT NULL
      AND request_timestamp BETWEEN TIMESTAMP '{start_date.isoformat()}'
                                AND TIMESTAMP '{end_date.isoformat()}'
    """

    df = athena_client.execute_query(query)

    if df.empty or len(df) < 10:
        logger.warning(f"Insufficient data for confusion matrix: {len(df)} samples")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Insufficient data\n(need ground truth labels)',
                ha='center', va='center', fontsize=16)
        ax.set_title(f'Confusion Matrix - {endpoint_name}\nInsufficient Data')
        return fig, {'sample_count': len(df)}

    # Calculate predictions based on threshold
    y_true = df['ground_truth'].values
    y_pred = (df['probability_fraud'].values >= threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 14})

    # Labels
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {endpoint_name}\n'
                f'Threshold: {threshold:.2f} | Accuracy: {accuracy:.3f}',
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(['Non-Fraud', 'Fraud'])
    ax.set_yticklabels(['Non-Fraud', 'Fraud'])

    # Add metrics text
    metrics_text = (f'Precision: {precision:.3f}\n'
                   f'Recall: {recall:.3f}\n'
                   f'F1-Score: {f1:.3f}')
    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'sample_count': len(df),
        'threshold': threshold,
    }

    return fig, metrics


def create_prediction_distribution(
    athena_client,
    endpoint_name: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    aggregation: str = 'hourly',
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create prediction distribution chart over time.

    Args:
        athena_client: AthenaClient instance
        endpoint_name: SageMaker endpoint name
        start_date: Start date for data
        end_date: End date for data
        aggregation: Time aggregation ('hourly', 'daily', 'weekly')
        save_path: Optional path to save figure

    Returns:
        Tuple of (matplotlib figure, metrics dict)
    """
    # Default date range
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        if aggregation == 'hourly':
            start_date = end_date - timedelta(days=1)
        elif aggregation == 'daily':
            start_date = end_date - timedelta(days=30)
        else:  # weekly
            start_date = end_date - timedelta(days=90)

    logger.info(f"Creating prediction distribution for {endpoint_name} ({aggregation})")

    # Determine time truncation based on aggregation
    if aggregation == 'hourly':
        trunc_func = "DATE_TRUNC('hour', request_timestamp)"
    elif aggregation == 'daily':
        trunc_func = "DATE_TRUNC('day', request_timestamp)"
    else:  # weekly
        trunc_func = "DATE_TRUNC('week', request_timestamp)"

    # Query data
    query = f"""
    SELECT
        {trunc_func} as time_bucket,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as fraud_predictions,
        AVG(probability_fraud) as avg_fraud_prob,
        AVG(confidence_score) as avg_confidence
    FROM fraud_detection.inference_responses
    WHERE endpoint_name = '{endpoint_name}'
      AND request_timestamp BETWEEN TIMESTAMP '{start_date.isoformat()}'
                                AND TIMESTAMP '{end_date.isoformat()}'
    GROUP BY {trunc_func}
    ORDER BY time_bucket
    """

    df = athena_client.execute_query(query)

    if df.empty:
        logger.warning(f"No data found for prediction distribution")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No prediction data available',
                ha='center', va='center', fontsize=16)
        ax.set_title(f'Prediction Distribution - {endpoint_name}\nNo Data')
        return fig, {'sample_count': 0}

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Total predictions over time
    ax1.plot(df['time_bucket'], df['total_predictions'],
            marker='o', linewidth=2, markersize=6, color='steelblue')
    ax1.fill_between(df['time_bucket'], df['total_predictions'],
                     alpha=0.3, color='steelblue')
    ax1.set_ylabel('Total Predictions', fontsize=11)
    ax1.set_title(f'Prediction Distribution - {endpoint_name}\n'
                 f'{start_date.date()} to {end_date.date()}',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Fraud vs Non-Fraud predictions
    ax2.bar(df['time_bucket'], df['fraud_predictions'],
           label='Fraud Predictions', color='red', alpha=0.7)
    ax2.bar(df['time_bucket'], df['total_predictions'] - df['fraud_predictions'],
           bottom=df['fraud_predictions'],
           label='Non-Fraud Predictions', color='green', alpha=0.7)
    ax2.set_ylabel('Prediction Count', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average fraud probability and confidence
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(df['time_bucket'], df['avg_fraud_prob'],
                    marker='s', linewidth=2, markersize=5,
                    color='orange', label='Avg Fraud Probability')
    line2 = ax3_twin.plot(df['time_bucket'], df['avg_confidence'],
                         marker='^', linewidth=2, markersize=5,
                         color='purple', label='Avg Confidence')
    ax3.set_ylabel('Avg Fraud Probability', fontsize=11, color='orange')
    ax3_twin.set_ylabel('Avg Confidence Score', fontsize=11, color='purple')
    ax3.set_xlabel('Time', fontsize=11)
    ax3.tick_params(axis='y', labelcolor='orange')
    ax3_twin.tick_params(axis='y', labelcolor='purple')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Rotate x-axis labels
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction distribution saved to {save_path}")

    metrics = {
        'total_predictions': int(df['total_predictions'].sum()),
        'total_fraud_predictions': int(df['fraud_predictions'].sum()),
        'avg_fraud_probability': float(df['avg_fraud_prob'].mean()),
        'avg_confidence': float(df['avg_confidence'].mean()),
        'time_periods': len(df),
    }

    return fig, metrics


def create_latency_heatmap(
    athena_client,
    endpoint_name: str,
    days: int = 7,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create latency heatmap by hour and day.

    Args:
        athena_client: AthenaClient instance
        endpoint_name: SageMaker endpoint name
        days: Number of days to include
        save_path: Optional path to save figure

    Returns:
        Tuple of (matplotlib figure, metrics dict)
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    logger.info(f"Creating latency heatmap for {endpoint_name} (last {days} days)")

    # Query data
    query = f"""
    SELECT
        DATE_TRUNC('day', request_timestamp) as day,
        EXTRACT(HOUR FROM request_timestamp) as hour,
        AVG(inference_latency_ms) as avg_latency,
        STDDEV(inference_latency_ms) as std_latency,
        COUNT(*) as request_count
    FROM fraud_detection.inference_responses
    WHERE endpoint_name = '{endpoint_name}'
      AND request_timestamp > CURRENT_TIMESTAMP - INTERVAL '{days}' DAY
    GROUP BY DATE_TRUNC('day', request_timestamp), EXTRACT(HOUR FROM request_timestamp)
    ORDER BY day, hour
    """

    df = athena_client.execute_query(query)

    if df.empty:
        logger.warning(f"No latency data found")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No latency data available',
                ha='center', va='center', fontsize=16)
        ax.set_title(f'Latency Heatmap - {endpoint_name}\nNo Data')
        return fig, {'sample_count': 0}

    # Pivot data for heatmap
    pivot_data = df.pivot(index='day', columns='hour', values='avg_latency')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create heatmap
    sns.heatmap(pivot_data, annot=False, fmt='.1f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Avg Latency (ms)'})

    # Formatting
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Date', fontsize=12)
    ax.set_title(f'Inference Latency Heatmap - {endpoint_name}\n'
                f'Last {days} Days',
                fontsize=14, fontweight='bold')

    # Format y-axis dates
    y_labels = [d.strftime('%Y-%m-%d') if isinstance(d, datetime) else str(d)
               for d in pivot_data.index]
    ax.set_yticklabels(y_labels, rotation=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Latency heatmap saved to {save_path}")

    metrics = {
        'avg_latency_ms': float(df['avg_latency'].mean()),
        'max_latency_ms': float(df['avg_latency'].max()),
        'min_latency_ms': float(df['avg_latency'].min()),
        'std_latency_ms': float(df['std_latency'].mean()),
        'total_requests': int(df['request_count'].sum()),
    }

    return fig, metrics


def create_confidence_distribution(
    athena_client,
    endpoint_name: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create confidence score distribution histogram.

    Args:
        athena_client: AthenaClient instance
        endpoint_name: SageMaker endpoint name
        start_date: Start date for data
        end_date: End date for data
        save_path: Optional path to save figure

    Returns:
        Tuple of (matplotlib figure, metrics dict)
    """
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=7)

    logger.info(f"Creating confidence distribution for {endpoint_name}")

    # Query data
    query = f"""
    SELECT
        confidence_score,
        prediction,
        is_high_confidence,
        is_low_confidence
    FROM fraud_detection.inference_responses
    WHERE endpoint_name = '{endpoint_name}'
      AND request_timestamp BETWEEN TIMESTAMP '{start_date.isoformat()}'
                                AND TIMESTAMP '{end_date.isoformat()}'
    """

    df = athena_client.execute_query(query)

    if df.empty:
        logger.warning(f"No confidence data found")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No confidence data available',
                ha='center', va='center', fontsize=16)
        ax.set_title(f'Confidence Distribution - {endpoint_name}\nNo Data')
        return fig, {'sample_count': 0}

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Overall confidence distribution
    ax1.hist(df['confidence_score'], bins=50, color='steelblue',
            alpha=0.7, edgecolor='black')
    ax1.axvline(0.9, color='red', linestyle='--', linewidth=2,
               label='High Confidence Threshold')
    ax1.axvline(0.4, color='orange', linestyle='--', linewidth=2,
               label='Low Confidence Range')
    ax1.axvline(0.6, color='orange', linestyle='--', linewidth=2)
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Confidence Score Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Confidence by prediction
    fraud_conf = df[df['prediction'] == 1]['confidence_score']
    non_fraud_conf = df[df['prediction'] == 0]['confidence_score']

    ax2.hist([non_fraud_conf, fraud_conf], bins=30,
            color=['green', 'red'], alpha=0.6,
            label=['Non-Fraud', 'Fraud'], edgecolor='black')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Confidence by Prediction', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confidence distribution saved to {save_path}")

    metrics = {
        'avg_confidence': float(df['confidence_score'].mean()),
        'high_confidence_count': int(df['is_high_confidence'].sum()),
        'low_confidence_count': int(df['is_low_confidence'].sum()),
        'high_confidence_rate': float(df['is_high_confidence'].mean()),
        'low_confidence_rate': float(df['is_low_confidence'].mean()),
        'sample_count': len(df),
    }

    return fig, metrics


def figure_to_bytes(fig: plt.Figure, format: str = 'png', dpi: int = 300) -> bytes:
    """
    Convert matplotlib figure to bytes for MLflow logging.

    Args:
        fig: Matplotlib figure
        format: Image format ('png', 'jpg', 'svg')
        dpi: Resolution

    Returns:
        Image bytes
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def log_all_charts_to_mlflow(
    athena_client,
    endpoint_name: str,
    mlflow_run_id: Optional[str] = None,
    days: int = 7,
) -> Dict[str, Any]:
    """
    Generate all charts and log to MLflow.

    Args:
        athena_client: AthenaClient instance
        endpoint_name: SageMaker endpoint name
        mlflow_run_id: Optional MLflow run ID (creates new run if None)
        days: Number of days of data to include

    Returns:
        Dictionary with all metrics
    """
    import mlflow

    logger.info(f"Generating all charts for {endpoint_name}")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    all_metrics = {}

    # Start or resume MLflow run
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            return _log_charts_internal(athena_client, endpoint_name, start_date, end_date, all_metrics)
    else:
        with mlflow.start_run():
            return _log_charts_internal(athena_client, endpoint_name, start_date, end_date, all_metrics)


def _log_charts_internal(athena_client, endpoint_name, start_date, end_date, all_metrics):
    """Internal helper to log charts within MLflow run context."""
    import mlflow

    try:
        # 1. ROC Curve
        logger.info("Generating ROC curve...")
        fig_roc, metrics_roc = create_roc_curve_from_athena(
            athena_client, endpoint_name, start_date, end_date
        )
        mlflow.log_figure(fig_roc, "charts/roc_curve.png")
        mlflow.log_metrics({f"roc_{k}": v for k, v in metrics_roc.items() if isinstance(v, (int, float))})
        all_metrics['roc'] = metrics_roc
        plt.close(fig_roc)
        logger.info("✓ ROC curve logged")

    except Exception as e:
        logger.error(f"Error generating ROC curve: {e}")

    try:
        # 2. Confusion Matrix
        logger.info("Generating confusion matrix...")
        fig_cm, metrics_cm = create_confusion_matrix_from_athena(
            athena_client, endpoint_name, start_date, end_date
        )
        mlflow.log_figure(fig_cm, "charts/confusion_matrix.png")
        mlflow.log_metrics({f"cm_{k}": v for k, v in metrics_cm.items() if isinstance(v, (int, float))})
        all_metrics['confusion_matrix'] = metrics_cm
        plt.close(fig_cm)
        logger.info("✓ Confusion matrix logged")

    except Exception as e:
        logger.error(f"Error generating confusion matrix: {e}")

    try:
        # 3. Prediction Distribution
        logger.info("Generating prediction distribution...")
        fig_dist, metrics_dist = create_prediction_distribution(
            athena_client, endpoint_name, start_date, end_date, aggregation='daily'
        )
        mlflow.log_figure(fig_dist, "charts/prediction_distribution.png")
        mlflow.log_metrics({f"dist_{k}": v for k, v in metrics_dist.items() if isinstance(v, (int, float))})
        all_metrics['distribution'] = metrics_dist
        plt.close(fig_dist)
        logger.info("✓ Prediction distribution logged")

    except Exception as e:
        logger.error(f"Error generating prediction distribution: {e}")

    try:
        # 4. Latency Heatmap
        logger.info("Generating latency heatmap...")
        fig_latency, metrics_latency = create_latency_heatmap(
            athena_client, endpoint_name, days=7
        )
        mlflow.log_figure(fig_latency, "charts/latency_heatmap.png")
        mlflow.log_metrics({f"latency_{k}": v for k, v in metrics_latency.items() if isinstance(v, (int, float))})
        all_metrics['latency'] = metrics_latency
        plt.close(fig_latency)
        logger.info("✓ Latency heatmap logged")

    except Exception as e:
        logger.error(f"Error generating latency heatmap: {e}")

    try:
        # 5. Confidence Distribution
        logger.info("Generating confidence distribution...")
        fig_conf, metrics_conf = create_confidence_distribution(
            athena_client, endpoint_name, start_date, end_date
        )
        mlflow.log_figure(fig_conf, "charts/confidence_distribution.png")
        mlflow.log_metrics({f"conf_{k}": v for k, v in metrics_conf.items() if isinstance(v, (int, float))})
        all_metrics['confidence'] = metrics_conf
        plt.close(fig_conf)
        logger.info("✓ Confidence distribution logged")

    except Exception as e:
        logger.error(f"Error generating confidence distribution: {e}")

    logger.info(f"All charts generated and logged to MLflow")
    return all_metrics


if __name__ == "__main__":
    """Test visualization utilities."""
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.train_pipeline.athena.athena_client import AthenaClient

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test with sample endpoint
    client = AthenaClient()
    endpoint_name = "fraud-detector"

    print("=" * 80)
    print("Testing Visualization Utilities")
    print("=" * 80)

    # Test each visualization
    print("\n1. Testing ROC Curve...")
    fig_roc, metrics_roc = create_roc_curve_from_athena(client, endpoint_name)
    print(f"   Metrics: {metrics_roc}")

    print("\n2. Testing Confusion Matrix...")
    fig_cm, metrics_cm = create_confusion_matrix_from_athena(client, endpoint_name)
    print(f"   Metrics: {metrics_cm}")

    print("\n3. Testing Prediction Distribution...")
    fig_dist, metrics_dist = create_prediction_distribution(client, endpoint_name)
    print(f"   Metrics: {metrics_dist}")

    print("\n4. Testing Latency Heatmap...")
    fig_latency, metrics_latency = create_latency_heatmap(client, endpoint_name)
    print(f"   Metrics: {metrics_latency}")

    print("\n5. Testing Confidence Distribution...")
    fig_conf, metrics_conf = create_confidence_distribution(client, endpoint_name)
    print(f"   Metrics: {metrics_conf}")

    print("\n" + "=" * 80)
    print("✓ All visualizations generated successfully")
    print("=" * 80)

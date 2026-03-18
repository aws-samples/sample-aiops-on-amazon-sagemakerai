"""
Lambda function for automated drift detection and alerting.

Triggered by EventBridge on a schedule (e.g., daily).
Uses Evidently for data drift and model performance analysis.
Sends SNS alerts if thresholds exceeded.
Logs all metrics and Evidently HTML reports to MLflow for tracking.
"""

import json
import os
import boto3
import time
from datetime import datetime, timedelta
import tempfile

import numpy as np
import pandas as pd

# Evidently-based reporting (used by check_data_drift / check_model_drift)
from src.monitoring.evidently_reports import run_data_drift_report, run_classification_report

# MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not available - skipping MLflow logging")

# AWS clients
athena = boto3.client('athena')
s3 = boto3.client('s3')
sns = boto3.client('sns')

# Configuration from environment variables
ATHENA_DATABASE = os.getenv('ATHENA_DATABASE', 'fraud_detection')
ATHENA_OUTPUT_S3 = os.getenv('ATHENA_OUTPUT_S3', 's3://fraud-detection-data-lake/athena-query-results/')
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

# Thresholds
DATA_DRIFT_THRESHOLD = float(os.getenv('DATA_DRIFT_THRESHOLD', '0.2'))  # PSI threshold
KS_PVALUE_THRESHOLD = float(os.getenv('KS_PVALUE_THRESHOLD', '0.05'))  # KS p-value threshold
MODEL_DRIFT_THRESHOLD = float(os.getenv('MODEL_DRIFT_THRESHOLD', '0.05'))  # 5% degradation
MIN_SAMPLES = int(os.getenv('MIN_SAMPLES', '100'))  # Minimum samples for analysis

# Training features (30 features)
TRAINING_FEATURES = [
    'transaction_hour', 'transaction_day_of_week', 'transaction_amount',
    'transaction_type_code', 'customer_age', 'customer_gender',
    'customer_tenure_months', 'account_age_days', 'distance_from_home_km',
    'distance_from_last_transaction_km', 'time_since_last_transaction_min',
    'online_transaction', 'international_transaction', 'high_risk_country',
    'merchant_category_code', 'merchant_reputation_score', 'chip_transaction',
    'pin_used', 'card_present', 'cvv_match', 'address_verification_match',
    'num_transactions_24h', 'num_transactions_7days',
    'avg_transaction_amount_30days', 'max_transaction_amount_30days',
    'velocity_score', 'recurring_transaction', 'previous_fraud_incidents',
    'credit_limit', 'available_credit_ratio'
]


def execute_athena_query(sql, wait=True):
    """Execute Athena query and return results as dict."""
    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT_S3}
    )
    execution_id = response['QueryExecutionId']

    if not wait:
        return execution_id

    # Wait for completion
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution_id)
        state = status['QueryExecution']['Status']['State']

        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)

    if state != 'SUCCEEDED':
        raise Exception(f"Query failed: {state}")

    # Get results
    result_s3_path = status['QueryExecution']['ResultConfiguration']['OutputLocation']
    bucket, key = result_s3_path.replace('s3://', '').split('/', 1)
    obj = s3.get_object(Bucket=bucket, Key=key)

    # Parse CSV results
    import csv
    lines = obj['Body'].read().decode('utf-8').splitlines()
    reader = csv.DictReader(lines)
    return list(reader)


# =========================================================================
# Legacy statistical functions (kept for reference)
#
# These demonstrate how to compute PSI and KS drift statistics explicitly
# without Evidently. The active Lambda flow now uses Evidently's
# DataDriftPreset and ClassificationPreset via evidently_reports.py.
# =========================================================================

def calculate_psi(baseline_values, current_values, bins=10):
    """Calculate Population Stability Index (PSI).

    LEGACY — This is the manual implementation of PSI using numpy.
    Kept to show what it takes to compute PSI without Evidently.
    The active drift check now delegates to ``run_data_drift_report()``
    which uses Evidently's DataDriftPreset internally.
    """
    baseline_values = np.array(baseline_values, dtype=float)
    current_values = np.array(current_values, dtype=float)

    # Create bins from baseline percentiles
    breakpoints = np.percentile(baseline_values, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Histogram
    baseline_hist, _ = np.histogram(baseline_values, bins=breakpoints)
    current_hist, _ = np.histogram(current_values, bins=breakpoints)

    # Convert to percentages
    baseline_pct = baseline_hist / len(baseline_values)
    current_pct = current_hist / len(current_values)

    # Add floor to avoid log(0)
    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)

    # Calculate PSI
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

    return float(psi)


def calculate_ks_statistic(baseline_values, current_values):
    """Calculate Kolmogorov-Smirnov test statistic.

    LEGACY — This is the manual implementation of the two-sample KS test
    using scipy.stats. Kept to show what it takes to compute KS without
    Evidently. The active drift check now delegates to
    ``run_data_drift_report()`` which uses Evidently's DataDriftPreset
    (which includes KS as one of its statistical tests).

    The KS test measures the maximum distance between the cumulative
    distribution functions (CDFs) of two samples. It's particularly
    sensitive to changes in distribution tails, making it ideal for
    fraud detection.

    Args:
        baseline_values: List of baseline (training) values
        current_values: List of current (inference) values

    Returns:
        tuple: (ks_statistic, p_value)
            - ks_statistic: 0-1 (0 = identical, 1 = completely different)
            - p_value: Probability that difference is random (< 0.05 = significant)
    """
    from scipy import stats

    baseline_values = np.array(baseline_values, dtype=float)
    current_values = np.array(current_values, dtype=float)

    # Remove NaN values
    baseline_values = baseline_values[~np.isnan(baseline_values)]
    current_values = current_values[~np.isnan(current_values)]

    if len(baseline_values) == 0 or len(current_values) == 0:
        return 0.0, 1.0

    # Perform two-sample KS test
    ks_stat, p_value = stats.ks_2samp(baseline_values, current_values)

    return float(ks_stat), float(p_value)


# =========================================================================
# Active drift detection — powered by Evidently
# =========================================================================

def check_data_drift():
    """Check for data drift using Evidently DataDriftPreset.

    Queries recent inference data and baseline training data from Athena,
    builds DataFrames, and runs Evidently's DataDriftPreset report.

    Returns:
        dict with drift results or None if insufficient data.
    """
    print("🔍 Checking data drift (Evidently)...")

    # Get recent inference data (last 24 hours)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')

    recent_data_sql = f"""
    SELECT input_features
    FROM {ATHENA_DATABASE}.inference_responses
    WHERE request_timestamp >= TIMESTAMP '{yesterday}'
    LIMIT 5000
    """

    recent_data = execute_athena_query(recent_data_sql)

    if len(recent_data) < MIN_SAMPLES:
        print(f"⚠️ Not enough recent samples ({len(recent_data)} < {MIN_SAMPLES})")
        return None

    print(f"✓ Found {len(recent_data)} recent inference samples")

    # Parse JSON features into a DataFrame
    current_rows = []
    for row in recent_data:
        try:
            features = json.loads(row['input_features'])
            parsed = {}
            for feat in TRAINING_FEATURES:
                if feat in features:
                    parsed[feat] = float(features[feat])
            if parsed:
                current_rows.append(parsed)
        except Exception:
            continue

    if len(current_rows) < MIN_SAMPLES:
        print(f"⚠️ Not enough parseable samples ({len(current_rows)} < {MIN_SAMPLES})")
        return None

    current_df = pd.DataFrame(current_rows)

    # Get baseline data (sample from training data)
    baseline_sql = f"""
    SELECT {', '.join(TRAINING_FEATURES)}
    FROM {ATHENA_DATABASE}.training_data
    WHERE is_fraud IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 5000
    """

    baseline_data = execute_athena_query(baseline_sql)
    print(f"✓ Loaded {len(baseline_data)} baseline samples")

    baseline_df = pd.DataFrame(baseline_data)
    # Ensure numeric types
    for col in baseline_df.columns:
        baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')

    # Use only columns present in both DataFrames
    common_cols = sorted(set(baseline_df.columns) & set(current_df.columns))
    if not common_cols:
        print("⚠️ No common columns between baseline and current data")
        return None

    baseline_df = baseline_df[common_cols]
    current_df = current_df[common_cols]

    # Save Evidently HTML report to /tmp for MLflow artifact logging
    html_path = tempfile.NamedTemporaryFile(
        suffix='.html', prefix='data_drift_', delete=False, dir='/tmp'
    ).name

    # Run Evidently data drift report
    drift_result = run_data_drift_report(
        baseline_df=baseline_df,
        current_df=current_df,
        output_path=html_path,
    )

    # Build per-column summary for SNS alert
    drifted_features = []
    per_column = drift_result.get('per_column', {})
    for col, info in per_column.items():
        if info.get('drifted'):
            drifted_features.append({
                'feature': col,
                'drift_score': info.get('drift_score', 0),
            })

    # Sort by drift score ascending (lower p-value = more drifted)
    drifted_features.sort(key=lambda x: x['drift_score'])

    features_analyzed = len(per_column)
    drifted_count = drift_result['drifted_columns_count']
    drift_share = drift_result['drifted_columns_share']

    print(f"  Evidently: {drifted_count}/{features_analyzed} features drifted ({drift_share:.1%})")
    if drift_result['drift_detected']:
        print("  🚨 Overall data drift DETECTED")
    else:
        print("  ✓ No overall data drift detected")

    return {
        'detected': drift_result['drift_detected'],
        'features_analyzed': features_analyzed,
        'drifted_features_count': drifted_count,
        'drift_percentage': drift_share * 100,
        'drifted_columns_share': drift_share,
        'drifted_features': drifted_features[:5],  # Top 5
        'sample_size': len(current_rows),
        'html_report_path': html_path,
    }


def check_model_drift():
    """Check for model performance drift using Evidently ClassificationPreset.

    Queries recent predictions with ground truth from Athena, builds a
    baseline comparison DataFrame, and runs Evidently's ClassificationPreset.

    Returns:
        dict with model drift results or None if insufficient data.
    """
    print("🔍 Checking model drift (Evidently)...")

    # Get recent predictions with ground truth (last 7 days)
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')

    performance_sql = f"""
    SELECT
        prediction,
        probability_fraud,
        ground_truth
    FROM {ATHENA_DATABASE}.inference_responses
    WHERE ground_truth IS NOT NULL
      AND request_timestamp >= TIMESTAMP '{week_ago}'
    LIMIT 5000
    """

    recent_performance = execute_athena_query(performance_sql)

    if len(recent_performance) < MIN_SAMPLES:
        print(f"⚠️ Not enough samples with ground truth ({len(recent_performance)} < {MIN_SAMPLES})")
        return None

    print(f"✓ Found {len(recent_performance)} samples with ground truth")

    # Build current DataFrame
    current_df = pd.DataFrame(recent_performance)
    current_df['ground_truth'] = current_df['ground_truth'].astype(int)
    current_df['prediction'] = current_df['prediction'].astype(int)
    current_df['probability_fraud'] = current_df['probability_fraud'].astype(float)

    # Compute sklearn metrics for the SNS alert / response payload
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

    y_true = current_df['ground_truth'].values
    y_pred = current_df['prediction'].values
    y_prob = current_df['probability_fraud'].values

    current_roc_auc = roc_auc_score(y_true, y_prob)
    current_accuracy = accuracy_score(y_true, y_pred)
    current_precision = precision_score(y_true, y_pred, zero_division=0)
    current_recall = recall_score(y_true, y_pred, zero_division=0)

    print(f"  Current ROC-AUC: {current_roc_auc:.4f}")
    print(f"  Current Accuracy: {current_accuracy:.4f}")
    print(f"  Current Precision: {current_precision:.4f}")
    print(f"  Current Recall: {current_recall:.4f}")

    # Compare to baseline
    baseline_roc_auc = float(os.getenv('BASELINE_ROC_AUC', '0.92'))
    degradation = baseline_roc_auc - current_roc_auc
    degradation_pct = (degradation / baseline_roc_auc) * 100

    print(f"  Baseline ROC-AUC: {baseline_roc_auc:.4f}")
    print(f"  Degradation: {degradation:.4f} ({degradation_pct:.1f}%)")

    # Build a synthetic baseline DataFrame with the same schema so Evidently
    # can compare reference vs current classification performance.
    # In production you'd load actual baseline predictions from S3/Athena.
    baseline_sql = f"""
    SELECT prediction, probability_fraud, ground_truth
    FROM {ATHENA_DATABASE}.inference_responses
    WHERE ground_truth IS NOT NULL
      AND request_timestamp < TIMESTAMP '{week_ago}'
    ORDER BY RANDOM()
    LIMIT 5000
    """

    try:
        baseline_data = execute_athena_query(baseline_sql)
        if len(baseline_data) >= MIN_SAMPLES:
            baseline_df = pd.DataFrame(baseline_data)
            baseline_df['ground_truth'] = baseline_df['ground_truth'].astype(int)
            baseline_df['prediction'] = baseline_df['prediction'].astype(int)
            baseline_df['probability_fraud'] = baseline_df['probability_fraud'].astype(float)
        else:
            # Fall back: duplicate current as baseline (report still generates)
            baseline_df = current_df.copy()
    except Exception:
        baseline_df = current_df.copy()

    # Save Evidently HTML report to /tmp
    html_path = tempfile.NamedTemporaryFile(
        suffix='.html', prefix='model_perf_', delete=False, dir='/tmp'
    ).name

    classification_result = run_classification_report(
        baseline_df=baseline_df,
        current_df=current_df,
        target_column='ground_truth',
        prediction_column='prediction',
        output_path=html_path,
    )

    detected = degradation_pct >= (MODEL_DRIFT_THRESHOLD * 100)
    if detected:
        print("  🚨 Model performance drift DETECTED")
    else:
        print("  ✓ No model performance drift detected")

    return {
        'detected': detected,
        'baseline_roc_auc': baseline_roc_auc,
        'current_roc_auc': current_roc_auc,
        'degradation': degradation,
        'degradation_pct': degradation_pct,
        'accuracy': current_accuracy,
        'precision': current_precision,
        'recall': current_recall,
        'sample_size': len(recent_performance),
        'html_report_path': html_path,
        'evidently_metrics': classification_result.get('metrics', []),
    }


def send_sns_alert(data_drift_result, model_drift_result):
    """Send SNS notification if drift detected."""
    if not SNS_TOPIC_ARN:
        print("⚠️ SNS_TOPIC_ARN not configured, skipping notification")
        return

    data_drift_detected = data_drift_result and data_drift_result.get('detected', False)
    model_drift_detected = model_drift_result and model_drift_result.get('detected', False)

    if not data_drift_detected and not model_drift_detected:
        print("✓ No drift detected, no alert sent")
        return

    # Build alert message
    subject = "🚨 ML Model Drift Alert - Fraud Detection"

    message_lines = [
        "=" * 80,
        "ML MODEL DRIFT ALERT",
        "=" * 80,
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Detection Engine: Evidently AI",
        "",
    ]

    if data_drift_detected:
        message_lines.extend([
            "🔴 DATA DRIFT DETECTED (Evidently DataDriftPreset)",
            "=" * 80,
            f"Features Analyzed: {data_drift_result['features_analyzed']}",
            f"Drifted Features: {data_drift_result['drifted_features_count']} "
            f"({data_drift_result['drift_percentage']:.1f}%)",
            f"Drifted Columns Share: {data_drift_result['drifted_columns_share']:.1%}",
            "",
            "Top Drifted Features (by drift score):",
        ])

        for feat_info in data_drift_result.get('drifted_features', []):
            message_lines.append(
                f"  - {feat_info['feature']}: "
                f"drift_score={feat_info['drift_score']:.4f}"
            )

        message_lines.append("")

    if model_drift_detected:
        message_lines.extend([
            "🔴 MODEL PERFORMANCE DRIFT DETECTED (Evidently ClassificationPreset)",
            "=" * 80,
            f"Baseline ROC-AUC: {model_drift_result['baseline_roc_auc']:.4f}",
            f"Current ROC-AUC: {model_drift_result['current_roc_auc']:.4f}",
            f"Degradation: {model_drift_result['degradation']:.4f} ({model_drift_result['degradation_pct']:.1f}%)",
            f"Threshold: {MODEL_DRIFT_THRESHOLD * 100:.1f}%",
            "",
            f"Current Accuracy: {model_drift_result['accuracy']:.4f}",
            f"Current Precision: {model_drift_result['precision']:.4f}",
            f"Current Recall: {model_drift_result['recall']:.4f}",
            "",
        ])

    message_lines.extend([
        "=" * 80,
        "RECOMMENDED ACTIONS:",
        "=" * 80,
        "1. Review Evidently HTML reports in MLflow monitoring experiment",
        "2. Investigate root cause of drift (data quality, population shift, etc.)",
        "3. Consider retraining model with recent data",
        "4. Review and adjust decision thresholds if needed",
        "",
        "View detailed Evidently reports in MLflow artifacts or inference_monitoring.ipynb",
        "=" * 80,
    ])

    message = "\n".join(message_lines)

    # Send SNS notification
    try:
        response = sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )
        print(f"✓ SNS alert sent: {response['MessageId']}")
    except Exception as e:
        print(f"❌ Failed to send SNS alert: {e}")


# =========================================================================
# Legacy chart functions (kept for reference)
#
# These show how to build custom matplotlib visualizations for drift
# analysis. The active Lambda flow now logs Evidently's interactive HTML
# reports as MLflow artifacts instead.
# =========================================================================

def create_psi_chart(drift_results):
    """Create PSI bar chart visualization.

    LEGACY — Replaced by Evidently HTML data drift report logged as an
    MLflow artifact. Kept to demonstrate custom matplotlib charting.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not drift_results:
        return None

    # Sort by PSI value
    sorted_results = sorted(drift_results, key=lambda x: x['psi'], reverse=True)[:15]

    features = [r['feature'] for r in sorted_results]
    psi_values = [r['psi'] for r in sorted_results]
    colors = ['red' if r['drifted'] else 'green' for r in sorted_results]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(features, psi_values, color=colors, alpha=0.7)

    ax.axvline(x=DATA_DRIFT_THRESHOLD, color='orange', linestyle='--',
               linewidth=2, label=f'Threshold ({DATA_DRIFT_THRESHOLD})')

    ax.set_xlabel('Population Stability Index (PSI)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Data Drift Analysis - PSI by Feature', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, psi_values)):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()

    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
    plt.close()

    return temp_file.name


def create_model_performance_chart(model_drift_result):
    """Create model performance comparison chart.

    LEGACY — Replaced by Evidently HTML classification report logged as
    an MLflow artifact. Kept to demonstrate custom matplotlib charting.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not model_drift_result:
        return None

    metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall']
    baseline_values = [
        model_drift_result['baseline_roc_auc'],
        0.95,
        0.90,
        0.85
    ]
    current_values = [
        model_drift_result['current_roc_auc'],
        model_drift_result['accuracy'],
        model_drift_result['precision'],
        model_drift_result['recall']
    ]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], baseline_values, width,
                    label='Baseline', alpha=0.8, color='green')
    bars2 = ax.bar([i + width/2 for i in x], current_values, width,
                    label='Current', alpha=0.8, color='blue')

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
    plt.close()

    return temp_file.name


# =========================================================================
# MLflow logging — logs Evidently HTML reports as artifacts
# =========================================================================

def log_to_mlflow(data_drift_result, model_drift_result):
    """Log drift metrics and Evidently HTML reports to MLflow."""
    if not MLFLOW_AVAILABLE:
        print("⚠️ MLflow not available - skipping MLflow logging")
        return

    if not MLFLOW_TRACKING_URI:
        print("⚠️ MLFLOW_TRACKING_URI not configured - skipping MLflow logging")
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("fraud-detection-drift-monitoring")

        with mlflow.start_run(run_name=f"drift-check-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            # Log configuration parameters
            mlflow.log_param("detection_engine", "evidently")
            mlflow.log_param("model_drift_threshold", MODEL_DRIFT_THRESHOLD)
            mlflow.log_param("min_samples", MIN_SAMPLES)

            # --- Data drift metrics + Evidently report ---
            if data_drift_result:
                mlflow.log_metric("data_drift_detected", 1 if data_drift_result['detected'] else 0)
                mlflow.log_metric("features_analyzed", data_drift_result['features_analyzed'])
                mlflow.log_metric("drifted_features_count", data_drift_result['drifted_features_count'])
                mlflow.log_metric("drift_percentage", data_drift_result['drift_percentage'])
                mlflow.log_metric("drifted_columns_share", data_drift_result['drifted_columns_share'])
                mlflow.log_metric("data_sample_size", data_drift_result['sample_size'])

                # Log per-feature drift scores
                for feat_info in data_drift_result.get('drifted_features', []):
                    mlflow.log_metric(
                        f"drift_score_{feat_info['feature']}",
                        feat_info['drift_score'],
                    )

                # Log Evidently HTML report as artifact
                html_path = data_drift_result.get('html_report_path')
                if html_path and os.path.exists(html_path):
                    mlflow.log_artifact(html_path, "evidently_reports")
                    os.unlink(html_path)

            # --- Model drift metrics + Evidently report ---
            if model_drift_result:
                mlflow.log_metric("model_drift_detected", 1 if model_drift_result['detected'] else 0)
                mlflow.log_metric("baseline_roc_auc", model_drift_result['baseline_roc_auc'])
                mlflow.log_metric("current_roc_auc", model_drift_result['current_roc_auc'])
                mlflow.log_metric("roc_auc_degradation", model_drift_result['degradation'])
                mlflow.log_metric("roc_auc_degradation_pct", model_drift_result['degradation_pct'])
                mlflow.log_metric("current_accuracy", model_drift_result['accuracy'])
                mlflow.log_metric("current_precision", model_drift_result['precision'])
                mlflow.log_metric("current_recall", model_drift_result['recall'])
                mlflow.log_metric("model_sample_size", model_drift_result['sample_size'])

                # Log Evidently classification metrics (accuracy, F1, etc.)
                for m in model_drift_result.get('evidently_metrics', []):
                    name = m.get('metric_name', '')
                    value = m.get('value')
                    if isinstance(value, (int, float)):
                        import re
                        # Strip parenthesized args and sanitize for MLflow
                        safe_name = re.sub(r'\([^)]*\)', '', name)
                        safe_name = safe_name.replace('::', '_').replace(' ', '_').lower().strip('_')
                        safe_name = re.sub(r'[^a-z0-9_\-\. /:]', '', safe_name)
                        if safe_name:
                            mlflow.log_metric(f"evidently_{safe_name}", value)

                # Log Evidently HTML report as artifact
                html_path = model_drift_result.get('html_report_path')
                if html_path and os.path.exists(html_path):
                    mlflow.log_artifact(html_path, "evidently_reports")
                    os.unlink(html_path)

            # Log drift summary as JSON artifact
            summary = {
                'timestamp': datetime.now().isoformat(),
                'detection_engine': 'evidently',
                'data_drift': {
                    k: v for k, v in (data_drift_result or {}).items()
                    if k not in ('html_report_path',)
                },
                'model_drift': {
                    k: v for k, v in (model_drift_result or {}).items()
                    if k not in ('html_report_path', 'evidently_metrics')
                },
                'alert_sent': (
                    (data_drift_result and data_drift_result.get('detected', False)) or
                    (model_drift_result and model_drift_result.get('detected', False))
                ),
            }

            summary_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            )
            json.dump(summary, summary_file, indent=2, default=str)
            summary_file.close()
            mlflow.log_artifact(summary_file.name, "drift_reports")
            os.unlink(summary_file.name)

            print("✓ Successfully logged Evidently reports and metrics to MLflow")

    except Exception as e:
        print(f"⚠️ Failed to log to MLflow: {e}")
        import traceback
        traceback.print_exc()


# =========================================================================
# Lambda entry point
# =========================================================================

def lambda_handler(event, context):
    """Lambda handler for EventBridge scheduled drift monitoring."""
    print("=" * 80)
    print(f"Drift Monitoring Check (Evidently) - {datetime.now()}")
    print("=" * 80)

    try:
        # Check data drift (Evidently DataDriftPreset)
        data_drift_result = check_data_drift()

        # Check model drift (Evidently ClassificationPreset)
        model_drift_result = check_model_drift()

        # Log Evidently reports and metrics to MLflow
        log_to_mlflow(data_drift_result, model_drift_result)

        # Send alert if drift detected
        send_sns_alert(data_drift_result, model_drift_result)

        # Prepare response (exclude local file paths)
        def _clean(result):
            if result is None:
                return None
            return {
                k: v for k, v in result.items()
                if k not in ('html_report_path', 'evidently_metrics')
            }

        response = {
            'timestamp': datetime.now().isoformat(),
            'detection_engine': 'evidently',
            'data_drift': _clean(data_drift_result),
            'model_drift': _clean(model_drift_result),
            'alert_sent': (
                (data_drift_result and data_drift_result.get('detected', False)) or
                (model_drift_result and model_drift_result.get('detected', False))
            ),
        }

        print("=" * 80)
        print("Drift monitoring check completed successfully")
        print("=" * 80)

        return {
            'statusCode': 200,
            'body': json.dumps(response, indent=2, default=str)
        }

    except Exception as e:
        print(f"❌ Error during drift monitoring: {e}")
        import traceback
        traceback.print_exc()

        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


if __name__ == '__main__':
    # For local testing
    lambda_handler({}, {})

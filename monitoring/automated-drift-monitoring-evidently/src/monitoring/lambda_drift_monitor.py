"""
Lambda function for automated drift detection and alerting.

Triggered by EventBridge on a schedule (e.g., daily).
Checks for data drift and model drift, sends SNS alerts if thresholds exceeded.
Logs all metrics and visualizations to MLflow for tracking.
"""

import json
import os
import boto3
import time
from datetime import datetime, timedelta
import io
import tempfile

# MLflow and visualization
try:
    import mlflow
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for Lambda
    import matplotlib.pyplot as plt
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow or matplotlib not available - skipping MLflow logging")

# AWS clients
athena = boto3.client('athena')
s3 = boto3.client('s3')
sns = boto3.client('sns')

# Configuration from environment variables
ATHENA_DATABASE = os.getenv('ATHENA_DATABASE', 'fraud_detection')
ATHENA_OUTPUT_S3 = os.getenv('ATHENA_OUTPUT_S3', 's3://fraud-detection-data-lake-skoppar/athena-query-results/')
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

# Thresholds
DATA_DRIFT_THRESHOLD = float(os.getenv('DATA_DRIFT_THRESHOLD', '0.2'))  # PSI threshold
KS_PVALUE_THRESHOLD = float(os.getenv('KS_PVALUE_THRESHOLD', '0.05'))  # KS p-value threshold (95% confidence)
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


def calculate_psi(baseline_values, current_values, bins=10):
    """Calculate Population Stability Index (PSI)."""
    import numpy as np

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
    """
    Calculate Kolmogorov-Smirnov test statistic.

    The KS test measures the maximum distance between the cumulative distribution
    functions (CDFs) of two samples. It's particularly sensitive to changes in
    distribution tails, making it ideal for fraud detection.

    Args:
        baseline_values: List of baseline (training) values
        current_values: List of current (inference) values

    Returns:
        tuple: (ks_statistic, p_value)
            - ks_statistic: 0-1 (0 = identical, 1 = completely different)
            - p_value: Probability that difference is random (< 0.05 = significant)

    Example:
        >>> ks_stat, p_val = calculate_ks_statistic(baseline, current)
        >>> if p_val < 0.05:
        >>>     print(f"Significant drift detected: KS={ks_stat:.4f}")
    """
    from scipy import stats
    import numpy as np

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


def check_data_drift():
    """Check for data drift by comparing recent inference data to baseline."""
    print("🔍 Checking data drift...")

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

    # Parse JSON features
    import numpy as np
    current_features = {}
    for feat in TRAINING_FEATURES:
        current_features[feat] = []

    for row in recent_data:
        try:
            features = json.loads(row['input_features'])
            for feat in TRAINING_FEATURES:
                if feat in features:
                    current_features[feat].append(float(features[feat]))
        except Exception as e:
            continue

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

    # Calculate PSI for each feature
    drift_results = []
    drifted_features = []

    for feat in TRAINING_FEATURES:
        if len(current_features[feat]) < MIN_SAMPLES:
            continue

        baseline_values = [float(row[feat]) for row in baseline_data if row.get(feat)]
        current_values = current_features[feat]

        if len(baseline_values) < MIN_SAMPLES or len(current_values) < MIN_SAMPLES:
            continue

        try:
            # Calculate both PSI and KS test
            psi = calculate_psi(baseline_values, current_values)
            ks_stat, ks_pvalue = calculate_ks_statistic(baseline_values, current_values)

            # Dual-threshold detection: flag drift if EITHER test triggers
            drift_psi = psi >= DATA_DRIFT_THRESHOLD        # PSI threshold (default: 0.2)
            drift_ks = ks_pvalue < KS_PVALUE_THRESHOLD     # KS p-value threshold (default: 0.05)
            has_drift = drift_psi or drift_ks

            drift_results.append({
                'feature': feat,
                'psi': psi,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'drift_method': 'ks' if drift_ks else ('psi' if drift_psi else 'none'),
                'drifted': has_drift
            })

            if has_drift:
                drifted_features.append({
                    'feature': feat,
                    'psi': psi,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'drift_method': 'ks' if drift_ks else 'psi'
                })
                method_label = 'KS' if drift_ks else 'PSI'
                print(f"  🚨 {feat}: PSI={psi:.4f}, KS={ks_stat:.4f}, p={ks_pvalue:.4f} [DRIFT via {method_label}]")
            else:
                print(f"  ✓ {feat}: PSI={psi:.4f}, KS={ks_stat:.4f}, p={ks_pvalue:.4f} [OK]")

        except Exception as e:
            print(f"  ⚠️ Error calculating drift for {feat}: {e}")

    # Calculate summary metrics
    if drift_results:
        avg_psi = sum(r['psi'] for r in drift_results) / len(drift_results)
        max_psi = max(r['psi'] for r in drift_results)
        drift_pct = (len(drifted_features) / len(drift_results)) * 100

        # KS statistics
        avg_ks = sum(r['ks_statistic'] for r in drift_results) / len(drift_results)
        max_ks = max(r['ks_statistic'] for r in drift_results)
        min_pvalue = min(r['ks_pvalue'] for r in drift_results)

        # Count by detection method
        ks_detected = sum(1 for r in drift_results if r.get('drift_method') == 'ks')
        psi_detected = sum(1 for r in drift_results if r.get('drift_method') == 'psi')

        return {
            'detected': len(drifted_features) > 0,
            'features_analyzed': len(drift_results),
            'drifted_features_count': len(drifted_features),
            'drift_percentage': drift_pct,
            'avg_psi': avg_psi,
            'max_psi': max_psi,
            'avg_ks_statistic': avg_ks,
            'max_ks_statistic': max_ks,
            'min_ks_pvalue': min_pvalue,
            'ks_detected_count': ks_detected,
            'psi_detected_count': psi_detected,
            'drifted_features': drifted_features[:5],  # Top 5
            'sample_size': len(recent_data),
            'all_drift_results': drift_results  # For MLflow logging
        }

    return None


def check_model_drift():
    """Check for model performance drift."""
    print("🔍 Checking model drift...")

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

    # Calculate ROC-AUC
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    import numpy as np

    y_true = np.array([int(row['ground_truth']) for row in recent_performance])
    y_pred = np.array([int(row['prediction']) for row in recent_performance])
    y_prob = np.array([float(row['probability_fraud']) for row in recent_performance])

    current_roc_auc = roc_auc_score(y_true, y_prob)
    current_accuracy = accuracy_score(y_true, y_pred)
    current_precision = precision_score(y_true, y_pred, zero_division=0)
    current_recall = recall_score(y_true, y_pred, zero_division=0)

    print(f"  Current ROC-AUC: {current_roc_auc:.4f}")
    print(f"  Current Accuracy: {current_accuracy:.4f}")
    print(f"  Current Precision: {current_precision:.4f}")
    print(f"  Current Recall: {current_recall:.4f}")

    # Compare to baseline (expected performance from training)
    # In production, this should come from MLflow or a baseline metrics table
    baseline_roc_auc = float(os.getenv('BASELINE_ROC_AUC', '0.92'))

    degradation = baseline_roc_auc - current_roc_auc
    degradation_pct = (degradation / baseline_roc_auc) * 100

    print(f"  Baseline ROC-AUC: {baseline_roc_auc:.4f}")
    print(f"  Degradation: {degradation:.4f} ({degradation_pct:.1f}%)")

    return {
        'detected': degradation_pct >= (MODEL_DRIFT_THRESHOLD * 100),
        'baseline_roc_auc': baseline_roc_auc,
        'current_roc_auc': current_roc_auc,
        'degradation': degradation,
        'degradation_pct': degradation_pct,
        'accuracy': current_accuracy,
        'precision': current_precision,
        'recall': current_recall,
        'sample_size': len(recent_performance)
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
        "",
    ]

    if data_drift_detected:
        message_lines.extend([
            "🔴 DATA DRIFT DETECTED",
            "=" * 80,
            f"Features Analyzed: {data_drift_result['features_analyzed']}",
            f"Drifted Features: {data_drift_result['drifted_features_count']} ({data_drift_result['drift_percentage']:.1f}%)",
            "",
            "PSI Statistics:",
            f"  Average PSI: {data_drift_result['avg_psi']:.4f}",
            f"  Max PSI: {data_drift_result['max_psi']:.4f}",
            f"  Threshold: {DATA_DRIFT_THRESHOLD}",
            "",
            "KS Test Statistics:",
            f"  Average KS: {data_drift_result['avg_ks_statistic']:.4f}",
            f"  Max KS: {data_drift_result['max_ks_statistic']:.4f}",
            f"  Min p-value: {data_drift_result['min_ks_pvalue']:.6f}",
            f"  p-value Threshold: {KS_PVALUE_THRESHOLD} ({(1-KS_PVALUE_THRESHOLD)*100:.0f}% confidence)",
            "",
            "Detection Methods:",
            f"  Detected by KS test: {data_drift_result['ks_detected_count']} features",
            f"  Detected by PSI: {data_drift_result['psi_detected_count']} features",
            "",
            "Top Drifted Features:",
        ])

        for feat_info in data_drift_result['drifted_features']:
            method = feat_info.get('drift_method', 'psi').upper()
            ks_stat = feat_info.get('ks_statistic', 0)
            ks_pvalue = feat_info.get('ks_pvalue', 1)
            message_lines.append(
                f"  - {feat_info['feature']}: "
                f"PSI={feat_info['psi']:.4f}, "
                f"KS={ks_stat:.4f}, "
                f"p={ks_pvalue:.4f} "
                f"[{method}]"
            )

        message_lines.append("")

    if model_drift_detected:
        message_lines.extend([
            "🔴 MODEL PERFORMANCE DRIFT DETECTED",
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
        "1. Review drift analysis in MLflow monitoring experiment",
        "2. Investigate root cause of drift (data quality, population shift, etc.)",
        "3. Consider retraining model with recent data",
        "4. Review and adjust decision thresholds if needed",
        "",
        "View detailed metrics in inference_monitoring.ipynb",
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


def create_psi_chart(drift_results):
    """Create PSI bar chart visualization."""
    if not drift_results:
        return None

    # Sort by PSI value
    sorted_results = sorted(drift_results, key=lambda x: x['psi'], reverse=True)[:15]  # Top 15

    features = [r['feature'] for r in sorted_results]
    psi_values = [r['psi'] for r in sorted_results]
    colors = ['red' if r['drifted'] else 'green' for r in sorted_results]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(features, psi_values, color=colors, alpha=0.7)

    # Add threshold line
    ax.axvline(x=DATA_DRIFT_THRESHOLD, color='orange', linestyle='--',
               linewidth=2, label=f'Threshold ({DATA_DRIFT_THRESHOLD})')

    ax.set_xlabel('Population Stability Index (PSI)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Data Drift Analysis - PSI by Feature', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, psi_values)):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()

    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
    plt.close()

    return temp_file.name


def create_model_performance_chart(model_drift_result):
    """Create model performance comparison chart."""
    if not model_drift_result:
        return None

    metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall']
    baseline_values = [
        model_drift_result['baseline_roc_auc'],
        0.95,  # Approximate baseline accuracy (adjust as needed)
        0.90,  # Approximate baseline precision
        0.85   # Approximate baseline recall
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

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
    plt.close()

    return temp_file.name


def log_to_mlflow(data_drift_result, model_drift_result, drift_results):
    """Log drift metrics and charts to MLflow."""
    if not MLFLOW_AVAILABLE:
        print("⚠️ MLflow not available - skipping MLflow logging")
        return

    if not MLFLOW_TRACKING_URI:
        print("⚠️ MLFLOW_TRACKING_URI not configured - skipping MLflow logging")
        return

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("fraud-detection-drift-monitoring")

        with mlflow.start_run(run_name=f"drift-check-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            # Log configuration parameters
            mlflow.log_param("data_drift_threshold", DATA_DRIFT_THRESHOLD)
            mlflow.log_param("model_drift_threshold", MODEL_DRIFT_THRESHOLD)
            mlflow.log_param("min_samples", MIN_SAMPLES)

            # Log data drift metrics
            if data_drift_result:
                mlflow.log_metric("data_drift_detected", 1 if data_drift_result['detected'] else 0)
                mlflow.log_metric("features_analyzed", data_drift_result['features_analyzed'])
                mlflow.log_metric("drifted_features_count", data_drift_result['drifted_features_count'])
                mlflow.log_metric("drift_percentage", data_drift_result['drift_percentage'])
                mlflow.log_metric("avg_psi", data_drift_result['avg_psi'])
                mlflow.log_metric("max_psi", data_drift_result['max_psi'])
                mlflow.log_metric("data_sample_size", data_drift_result['sample_size'])

                # Log individual feature PSI values
                if drift_results:
                    for result in drift_results:
                        mlflow.log_metric(f"psi_{result['feature']}", result['psi'])

                # Create and log PSI chart
                psi_chart_path = create_psi_chart(drift_results)
                if psi_chart_path:
                    mlflow.log_artifact(psi_chart_path, "drift_charts")
                    os.unlink(psi_chart_path)  # Clean up temp file

            # Log model drift metrics
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

                # Create and log model performance chart
                perf_chart_path = create_model_performance_chart(model_drift_result)
                if perf_chart_path:
                    mlflow.log_artifact(perf_chart_path, "drift_charts")
                    os.unlink(perf_chart_path)  # Clean up temp file

            # Log drift summary as JSON artifact
            summary = {
                'timestamp': datetime.now().isoformat(),
                'data_drift': data_drift_result,
                'model_drift': model_drift_result,
                'alert_sent': (
                    (data_drift_result and data_drift_result.get('detected', False)) or
                    (model_drift_result and model_drift_result.get('detected', False))
                )
            }

            summary_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(summary, summary_file, indent=2)
            summary_file.close()
            mlflow.log_artifact(summary_file.name, "drift_reports")
            os.unlink(summary_file.name)  # Clean up temp file

            print("✓ Successfully logged metrics and charts to MLflow")

    except Exception as e:
        print(f"⚠️ Failed to log to MLflow: {e}")
        import traceback
        traceback.print_exc()


def lambda_handler(event, context):
    """Lambda handler for EventBridge scheduled drift monitoring."""
    print("=" * 80)
    print(f"Drift Monitoring Check - {datetime.now()}")
    print("=" * 80)

    try:
        # Check data drift
        data_drift_result = check_data_drift()

        # Check model drift
        model_drift_result = check_model_drift()

        # Extract detailed drift results for MLflow logging
        drift_results = None
        if data_drift_result and 'all_drift_results' in data_drift_result:
            drift_results = data_drift_result['all_drift_results']
            # Remove from response to keep it clean
            data_drift_result_copy = data_drift_result.copy()
            data_drift_result_copy.pop('all_drift_results', None)
        else:
            data_drift_result_copy = data_drift_result

        # Log metrics and charts to MLflow
        log_to_mlflow(data_drift_result, model_drift_result, drift_results)

        # Send alert if drift detected
        send_sns_alert(data_drift_result, model_drift_result)

        # Prepare response
        response = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': data_drift_result_copy,
            'model_drift': model_drift_result,
            'alert_sent': (
                (data_drift_result and data_drift_result.get('detected', False)) or
                (model_drift_result and model_drift_result.get('detected', False))
            )
        }

        print("=" * 80)
        print("Drift monitoring check completed successfully")
        print("=" * 80)

        return {
            'statusCode': 200,
            'body': json.dumps(response, indent=2)
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

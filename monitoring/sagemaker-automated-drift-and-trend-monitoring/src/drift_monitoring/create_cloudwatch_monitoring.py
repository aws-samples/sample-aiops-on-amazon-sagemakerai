#!/usr/bin/env python3
"""
Create CloudWatch Dashboard and Alarms for Drift Monitoring.

This script creates:
1. CloudWatch metrics for data drift (PSI) and model drift (ROC-AUC, etc.)
2. CloudWatch alarms that trigger on threshold violations
3. CloudWatch dashboard for visualizing drift trends
"""

import boto3
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv


def create_cloudwatch_monitoring(
    region='us-east-1',
    endpoint_name='fraud-detection-endpoint',
    drift_threshold=0.10,
    psi_threshold=0.2,
    evaluation_periods=1
):
    """
    Create CloudWatch dashboard and alarms for drift monitoring.

    Args:
        region: AWS region
        endpoint_name: SageMaker endpoint name
        drift_threshold: Threshold for model drift alarms (default: 10% degradation)
        psi_threshold: Threshold for PSI data drift alarm (default: 0.2)
        evaluation_periods: Number of evaluation periods for alarms
    """

    # Load .env if available
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  Creating CloudWatch Drift Monitoring                             ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("")
    print(f"  Region: {region}")
    print(f"  Endpoint: {endpoint_name}")
    print(f"  Model Drift Threshold: {drift_threshold*100:.0f}%")
    print(f"  PSI Threshold: {psi_threshold}")
    print("")

    # AWS clients
    cw_client = boto3.client('cloudwatch', region_name=region)
    sts = boto3.client('sts', region_name=region)
    account_id = sts.get_caller_identity()['Account']

    # Configuration (from .env with defaults)
    NAMESPACE = os.getenv('CLOUDWATCH_NAMESPACE', 'FraudDetection/DriftMonitoring')
    DASHBOARD_NAME = os.getenv('CLOUDWATCH_DASHBOARD_NAME', 'FraudDetection-DriftMonitoring')
    ATHENA_DATABASE = os.getenv('ATHENA_DATABASE', 'fraud_detection')
    MONITORING_TABLE = os.getenv('MONITORING_TABLE_NAME', 'monitoring_responses')
    DATA_S3_BUCKET = os.getenv('DATA_S3_BUCKET', f'fraud-detection-data-lake-skoppar-{account_id}')

    # Step 1: Get latest drift metrics from monitoring_responses table
    print("[1/3] Fetching latest drift metrics from Athena...")

    try:
        athena = boto3.client('athena', region_name=region)
        query = f"""
        SELECT
            timestamp,
            data_drift_score,
            baseline_roc_auc,
            current_roc_auc,
            degradation,
            accuracy,
            precision_score,
            recall
        FROM {ATHENA_DATABASE}.{MONITORING_TABLE}
        WHERE timestamp IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 1
        """

        response = athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': ATHENA_DATABASE},
            ResultConfiguration={
                'OutputLocation': f's3://{DATA_S3_BUCKET}/athena-query-results/'
            }
        )

        execution_id = response['QueryExecutionId']

        # Wait for query to complete
        import time
        while True:
            status = athena.get_query_execution(QueryExecutionId=execution_id)
            state = status['QueryExecution']['Status']['State']
            if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(1)

        if state == 'SUCCEEDED':
            # Get results
            result_s3_path = status['QueryExecution']['ResultConfiguration']['OutputLocation']
            s3 = boto3.client('s3', region_name=region)
            bucket, key = result_s3_path.replace('s3://', '').split('/', 1)
            obj = s3.get_object(Bucket=bucket, Key=key)

            import csv
            lines = obj['Body'].read().decode('utf-8').splitlines()
            reader = csv.DictReader(lines)
            results = list(reader)

            if results:
                latest = results[0]
                print(f"  ✓ Found latest metrics from {latest.get('timestamp', 'N/A')}")

                # Publish metrics to CloudWatch
                metrics_to_publish = []

                # Data drift metrics
                if latest.get('data_drift_score'):
                    metrics_to_publish.append({
                        'MetricName': 'DataDriftPSI',
                        'Value': float(latest['data_drift_score']),
                        'Unit': 'None',
                        'Timestamp': datetime.now(timezone.utc)
                    })

                # Model performance metrics
                if latest.get('baseline_roc_auc') and latest.get('current_roc_auc'):
                    baseline = float(latest['baseline_roc_auc'])
                    current = float(latest['current_roc_auc'])
                    degradation = float(latest.get('degradation', 0))

                    metrics_to_publish.extend([
                        {'MetricName': 'BaselineROCAUC', 'Value': baseline, 'Unit': 'None'},
                        {'MetricName': 'CurrentROCAUC', 'Value': current, 'Unit': 'None'},
                        {'MetricName': 'ROCAUCDegradation', 'Value': degradation, 'Unit': 'None'},
                    ])

                if latest.get('accuracy'):
                    metrics_to_publish.append({
                        'MetricName': 'Accuracy',
                        'Value': float(latest['accuracy']),
                        'Unit': 'None'
                    })

                if latest.get('precision_score'):
                    metrics_to_publish.append({
                        'MetricName': 'Precision',
                        'Value': float(latest['precision_score']),
                        'Unit': 'None'
                    })

                if latest.get('recall'):
                    metrics_to_publish.append({
                        'MetricName': 'Recall',
                        'Value': float(latest['recall']),
                        'Unit': 'None'
                    })

                # Publish metrics in batches
                for i in range(0, len(metrics_to_publish), 20):
                    batch = metrics_to_publish[i:i+20]
                    cw_client.put_metric_data(
                        Namespace=NAMESPACE,
                        MetricData=[{
                            **m,
                            'Dimensions': [{'Name': 'Endpoint', 'Value': endpoint_name}]
                        } for m in batch]
                    )

                print(f"  ✓ Published {len(metrics_to_publish)} metrics to CloudWatch")
            else:
                print("  ⚠ No monitoring data found yet")
                print("    Run drift monitoring Lambda first to generate metrics")

    except Exception as e:
        print(f"  ⚠ Could not fetch metrics: {e}")
        print("    Dashboard and alarms will still be created, but may show no data")

    # Step 2: Create CloudWatch Alarms
    print("")
    print("[2/3] Creating CloudWatch alarms...")

    alarms_created = []

    # Data Drift Alarm: PSI > threshold
    try:
        cw_client.put_metric_alarm(
            AlarmName='FraudDetection-DataDrift-PSI',
            AlarmDescription=f'Data drift detected: Average PSI exceeds {psi_threshold} (significant distribution shift)',
            MetricName='DataDriftPSI',
            Namespace=NAMESPACE,
            Statistic='Average',
            Period=300,
            EvaluationPeriods=evaluation_periods,
            Threshold=psi_threshold,
            ComparisonOperator='GreaterThanThreshold',
            Dimensions=[{'Name': 'Endpoint', 'Value': endpoint_name}],
        )
        alarms_created.append(f'FraudDetection-DataDrift-PSI (threshold: PSI > {psi_threshold})')
    except Exception as e:
        print(f"  ⚠ Failed to create PSI alarm: {e}")

    # Model Drift Alarms
    model_drift_alarms = {
        'ROCAUCDegradation': f'Model drift: ROC-AUC degradation exceeds {drift_threshold*100:.0f}%',
        'Accuracy': f'Model drift: Accuracy degradation exceeds {drift_threshold*100:.0f}%',
        'Precision': f'Model drift: Precision degradation exceeds {drift_threshold*100:.0f}%',
        'Recall': f'Model drift: Recall degradation exceeds {drift_threshold*100:.0f}%',
    }

    for metric_name, description in model_drift_alarms.items():
        alarm_name = f'FraudDetection-ModelDrift-{metric_name.replace("_", "-").upper()}'
        try:
            cw_client.put_metric_alarm(
                AlarmName=alarm_name,
                AlarmDescription=description,
                MetricName=metric_name,
                Namespace=NAMESPACE,
                Statistic='Average',
                Period=300,
                EvaluationPeriods=evaluation_periods,
                Threshold=drift_threshold,
                ComparisonOperator='GreaterThanThreshold',
                Dimensions=[{'Name': 'Endpoint', 'Value': endpoint_name}],
            )
            alarms_created.append(f'{alarm_name} (threshold: > {drift_threshold*100:.0f}%)')
        except Exception as e:
            print(f"  ⚠ Failed to create {alarm_name}: {e}")

    print(f"  ✓ Created {len(alarms_created)} alarms")

    # Step 3: Create CloudWatch Dashboard
    print("")
    print(f"[3/3] Creating CloudWatch dashboard: {DASHBOARD_NAME}...")

    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "markdown": f"# Fraud Detection - Drift Monitoring Dashboard\n**Endpoint:** `{endpoint_name}` | **Threshold:** {drift_threshold*100:.0f}% variance | **Updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        [NAMESPACE, "DataDriftPSI", {"stat": "Average"}]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Data Drift - Population Stability Index (PSI)",
                    "period": 300,
                    "yAxis": {"left": {"min": 0, "max": 1}},
                    "annotations": {
                        "horizontal": [
                            {"label": "Alarm Threshold", "value": psi_threshold, "fill": "above", "color": "#d62728"}
                        ]
                    }
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        [NAMESPACE, "ROCAUCDegradation", {"stat": "Average", "label": "ROC-AUC Degradation"}],
                        [NAMESPACE, "Accuracy", {"stat": "Average", "label": "Accuracy Degradation"}],
                        [NAMESPACE, "Precision", {"stat": "Average", "label": "Precision Degradation"}],
                        [NAMESPACE, "Recall", {"stat": "Average", "label": "Recall Degradation"}],
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": f"Model Drift - Degradation from Baseline ({drift_threshold*100:.0f}% alarm threshold)",
                    "period": 300,
                    "yAxis": {"left": {"min": 0, "max": 0.5}},
                    "annotations": {
                        "horizontal": [
                            {"label": f"{drift_threshold*100:.0f}% Alarm Threshold", "value": drift_threshold, "color": "#d62728"}
                        ]
                    }
                }
            },
            {
                "type": "alarm",
                "properties": {
                    "title": "Drift Alarms Status",
                    "alarms": [
                        f"arn:aws:cloudwatch:{region}:{account_id}:alarm:FraudDetection-DataDrift-PSI",
                        f"arn:aws:cloudwatch:{region}:{account_id}:alarm:FraudDetection-ModelDrift-ROCAUCDEGRADATION",
                        f"arn:aws:cloudwatch:{region}:{account_id}:alarm:FraudDetection-ModelDrift-ACCURACY",
                        f"arn:aws:cloudwatch:{region}:{account_id}:alarm:FraudDetection-ModelDrift-PRECISION",
                        f"arn:aws:cloudwatch:{region}:{account_id}:alarm:FraudDetection-ModelDrift-RECALL",
                    ]
                }
            }
        ]
    }

    try:
        cw_client.put_dashboard(
            DashboardName=DASHBOARD_NAME,
            DashboardBody=json.dumps(dashboard_body),
        )
        print(f"  ✓ Dashboard created: {DASHBOARD_NAME}")
    except Exception as e:
        print(f"  ⚠ Failed to create dashboard: {e}")

    # Summary
    dashboard_url = f"https://console.aws.amazon.com/cloudwatch/home?region={region}#dashboards:name={DASHBOARD_NAME}"

    print("")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  ✅ CLOUDWATCH MONITORING CREATED                                  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("")
    print(f"Dashboard URL:")
    print(f"  {dashboard_url}")
    print("")
    print(f"Alarms ({len(alarms_created)}):")
    for alarm in alarms_created:
        print(f"  - {alarm}")
    print("")
    print("Next steps:")
    print("  1. View dashboard in CloudWatch console")
    print("  2. Configure alarm actions (SNS notifications):")
    print(f"     aws cloudwatch put-metric-alarm --alarm-name <name> --alarm-actions <sns-topic-arn>")
    print("  3. Alarms will trigger when thresholds are exceeded")
    print("")

    return {
        'dashboard_url': dashboard_url,
        'dashboard_name': DASHBOARD_NAME,
        'alarms': alarms_created,
        'namespace': NAMESPACE
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create CloudWatch drift monitoring')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--endpoint', default='fraud-detection-endpoint', help='SageMaker endpoint name')
    parser.add_argument('--drift-threshold', type=float, default=0.10, help='Model drift threshold (default: 0.10 = 10%%)')
    parser.add_argument('--psi-threshold', type=float, default=0.2, help='PSI data drift threshold (default: 0.2)')
    parser.add_argument('--evaluation-periods', type=int, default=1, help='Alarm evaluation periods')

    args = parser.parse_args()

    result = create_cloudwatch_monitoring(
        region=args.region,
        endpoint_name=args.endpoint,
        drift_threshold=args.drift_threshold,
        psi_threshold=args.psi_threshold,
        evaluation_periods=args.evaluation_periods
    )

    sys.exit(0 if result else 1)

# Deployment Scripts

This directory contains **shell scripts only** for deploying and managing drift monitoring infrastructure.

## Organization

**Shell Scripts** (`.sh` files) live here:
- Bash scripts for deployment and infrastructure management
- Use AWS CLI, Docker, and system commands

**Python Modules** live in `src/drift_monitoring/`:
- All Python code is in proper module structure
- Run as: `python3 -m src.drift_monitoring.{module}`
- Import as: `from src.drift_monitoring.{module} import {function}`
- See [Python Modules](#python-modules) section below

---

## Shell Scripts

### 1. `deploy_lambda_container.sh`

Deploys the drift monitoring Lambda as a Docker container image.

**Usage:**
```bash
./scripts/deploy_lambda_container.sh <email> [data_drift_threshold] [model_drift_threshold]

# Examples
./scripts/deploy_lambda_container.sh your-email@example.com
./scripts/deploy_lambda_container.sh your-email@example.com 0.2 0.05
```

**Parameters:**
- `email` (required): Email address for drift alerts
- `data_drift_threshold` (optional): Default 0.2 (20%)
- `model_drift_threshold` (optional): Default 0.05 (5% degradation)

**What it does:**
1. Creates SNS topic for alerts
2. Creates IAM role with Athena/S3/SNS permissions
3. Creates ECR repository (if needed)
4. Builds Docker image with Evidently dependencies
5. Pushes image to ECR
6. Creates/updates Lambda function with container image
7. Creates EventBridge schedule (daily at 2 AM UTC)
8. Tests the Lambda function

**Outputs:**
- SNS topic: `fraud-detection-drift-alerts`
- Lambda function: `fraud-detection-drift-monitor`
- IAM role: `fraud-detection-drift-monitor-role`
- EventBridge rule: `fraud-detection-drift-check`
- ECR repository: `fraud-detection-drift-monitor`
- Config file: `src/config/drift_monitoring_config.json`

**Requirements:**
- Docker installed and running
- AWS CLI configured
- Dockerfile at `src/drift_monitoring/Dockerfile.lambda`
- Sufficient permissions to create resources

---

### 2. `delete_infrastructure.sh`

Safely deletes all drift monitoring resources.

**Usage:**
```bash
./scripts/delete_infrastructure.sh [delete_ecr]

# Examples
./scripts/delete_infrastructure.sh          # Keeps ECR images
./scripts/delete_infrastructure.sh yes      # Deletes ECR images too
```

**What it deletes:**
1. EventBridge scheduled rule and targets
2. Lambda functions (drift monitor + monitoring writer)
3. SQS queue
4. SNS topic and all subscriptions
5. IAM roles and policies
6. ECR repository (if requested)
7. Configuration file

**Safety features:**
- Requires explicit confirmation ('yes')
- Loads configuration from `src/config/drift_monitoring_config.json`
- Shows list of resources to be deleted
- Gracefully handles missing resources

**⚠️ WARNING**: This action cannot be undone. Make sure you have:
- Exported any important data from `monitoring_responses` table
- Documented your drift thresholds
- Saved any custom configurations

---

## Python Modules

All Python code lives in `src/drift_monitoring/`. These modules can be run as scripts or imported.

### Available Modules

#### 1. `create_monitoring_table`
Creates the `monitoring_responses` Iceberg table in Athena.

**Usage:**
```bash
python3 -m src.drift_monitoring.create_monitoring_table [region]

# Or from Python
from src.drift_monitoring.create_monitoring_table import create_monitoring_table
success = create_monitoring_table(region='us-east-1')
```

**What it does:**
- Creates Iceberg table with ACID compliance
- Sets up schema for monitoring metrics and drift scores
- Configures table location in S3

---

#### 2. `deploy_monitoring_writer`
Deploys the Lambda function that writes monitoring results to Athena.

**Usage:**
```bash
python3 -m src.drift_monitoring.deploy_monitoring_writer [region]

# Or from Python
from src.drift_monitoring.deploy_monitoring_writer import deploy_monitoring_writer
queue_url = deploy_monitoring_writer(region='us-east-1')
```

**What it does:**
- Creates SQS queue for monitoring messages
- Creates IAM role with necessary permissions
- Deploys Lambda function (ZIP-based, small footprint)
- Configures SQS trigger for automatic message processing

**Outputs:**
- Lambda function: `fraud-monitoring-results-writer`
- SQS queue: `fraud-monitoring-results`
- IAM role: `fraud-monitoring-results-writer-role`

---

#### 3. `create_cloudwatch_monitoring`
Creates CloudWatch dashboard and alarms for drift monitoring.

**Usage:**
```bash
python3 -m src.drift_monitoring.create_cloudwatch_monitoring [options]

# Examples
python3 -m src.drift_monitoring.create_cloudwatch_monitoring --region us-east-1
python3 -m src.drift_monitoring.create_cloudwatch_monitoring --drift-threshold 0.10

# Or from Python
from src.drift_monitoring.create_cloudwatch_monitoring import create_cloudwatch_monitoring
result = create_cloudwatch_monitoring(
    region='us-east-1',
    drift_threshold=0.10,
    psi_threshold=0.2
)
```

**Parameters:**
- `--region`: AWS region (default: us-east-1)
- `--endpoint`: SageMaker endpoint name (default: fraud-detection-endpoint)
- `--drift-threshold`: Model drift alarm threshold (default: 0.10 = 10%)
- `--psi-threshold`: PSI data drift alarm threshold (default: 0.2)
- `--evaluation-periods`: Alarm evaluation periods (default: 1)

**What it does:**
1. Fetches latest drift metrics from `monitoring_responses` Athena table
2. Publishes metrics to CloudWatch custom namespace
3. Creates CloudWatch alarms for data drift and model drift
4. Creates CloudWatch dashboard with visualizations

**Outputs:**
- Dashboard: `FraudDetection-DriftMonitoring`
- Namespace: `FraudDetection/DriftMonitoring`
- Alarms: DataDrift-PSI, ModelDrift-ROCAUCDEGRADATION, etc.

---

#### 4. `log_monitoring_to_mlflow`
Logs monitoring results (Evidently reports) to MLflow.

**Usage:**
```bash
# From Python (recommended)
from src.drift_monitoring.log_monitoring_to_mlflow import log_monitoring_to_mlflow

result = log_monitoring_to_mlflow(
    drift_results=drift_results,
    model_report=model_report,
    overall_metrics=metrics,
    endpoint_name='fraud-detector-endpoint'
)

# From CLI with JSON files
python3 -m src.drift_monitoring.log_monitoring_to_mlflow \
    --drift-json drift_results.json \
    --model-json model_report.json \
    --endpoint fraud-detector-endpoint
```

**What it logs:**
1. Overall performance metrics (accuracy, precision, recall)
2. Data drift metrics (PSI scores, drifted columns)
3. Per-column drift scores
4. Evidently HTML reports (interactive visualizations)
5. Drift summary JSON

**Requirements:**
- MLflow installed: `pip install mlflow`
- `MLFLOW_TRACKING_URI` set in `.env`

---

## Deployment Workflow

### Initial Setup

```bash
# 1. Create monitoring table
python3 -m src.drift_monitoring.create_monitoring_table us-east-1

# 2. Deploy monitoring writer Lambda
python3 -m src.drift_monitoring.deploy_monitoring_writer us-east-1

# 3. Deploy drift monitor Lambda
./scripts/deploy_lambda_container.sh your-email@example.com 0.2 0.05

# 4. Confirm SNS subscription via email

# 5. (Optional) Create CloudWatch dashboard
python3 -m src.drift_monitoring.create_cloudwatch_monitoring --region us-east-1
```

### Updates

```bash
# Update drift monitor Lambda only
./scripts/deploy_lambda_container.sh your-email@example.com

# Update monitoring writer only
python3 -m src.drift_monitoring.deploy_monitoring_writer us-east-1
```

### Cleanup

```bash
# Delete all infrastructure
./scripts/delete_infrastructure.sh

# Delete including ECR images
./scripts/delete_infrastructure.sh yes
```

---

## Configuration

All scripts read AWS credentials from:
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- AWS CLI config (`~/.aws/credentials`)
- IAM role (if running on EC2/Lambda)

Configuration files:
- `.env` - Environment variables (loaded automatically)
- `src/config/config.yaml` - Application defaults
- `src/config/drift_monitoring_config.json` - Deployment artifacts (auto-generated)

Default region: `us-east-1` (override with `AWS_REGION` or `--region`)

---

## Troubleshooting

### Docker build fails

```bash
# Ensure Docker is running
docker ps

# Check Docker login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

### Lambda deployment fails

```bash
# Check IAM role propagation (wait 10-30 seconds)
aws iam get-role --role-name fraud-detection-drift-monitor-role

# Check ECR image exists
aws ecr describe-images --repository-name fraud-detection-drift-monitor
```

### Monitoring table creation fails

```bash
# Check Lake Formation permissions
aws lakeformation list-permissions --resource-type TABLE

# Verify S3 bucket exists
aws s3 ls s3://fraud-detection-data-lake-skoppar-<account-id>/
```

---

## Monitoring

### View Lambda logs

```bash
# Drift monitor
aws logs tail /aws/lambda/fraud-detection-drift-monitor --follow

# Monitoring writer
aws logs tail /aws/lambda/fraud-monitoring-results-writer --follow
```

### Check EventBridge schedule

```bash
aws events describe-rule --name fraud-detection-drift-check
```

### Query monitoring results

```sql
SELECT *
FROM fraud_detection.monitoring_responses
WHERE timestamp > current_timestamp - interval '7' day
ORDER BY timestamp DESC
```

---

## Architecture

```
┌─────────────────┐
│ EventBridge     │  Daily at 2 AM UTC
│ Scheduled Rule  │
└────────┬────────┘
         │ trigger
         ▼
┌─────────────────┐
│ Lambda          │  Runs drift detection
│ (Container)     │  with Evidently library
└────────┬────────┘
         │ writes
         ▼
┌─────────────────┐       ┌─────────────────┐
│ SQS Queue       │──────▶│ Lambda          │
│                 │       │ (Writer)        │
└─────────────────┘       └────────┬────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐     ┌─────────────────┐
│ CloudWatch      │       │ Athena Iceberg  │     │ SNS Topic       │
│ Metrics         │       │ Table           │     │ (Alerts)        │
└────────┬────────┘       └────────┬────────┘     └─────────────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│ CloudWatch      │       │ QuickSight      │
│ Dashboard       │       │ Dashboard       │
│ & Alarms        │       │ (Governance)    │
└─────────────────┘       └─────────────────┘
```

---

## Cost Estimation

**Monthly costs (approximate):**
- Lambda (drift monitor): ~$0.50 (30 executions, 1min each)
- Lambda (monitoring writer): ~$0.10 (minimal execution)
- EventBridge: $0.00 (free tier)
- SNS: $0.00 (< 1000 emails/month)
- SQS: $0.00 (free tier)
- ECR: ~$0.10/GB/month (image storage)
- Athena: Pay per query (~$5/TB scanned)

**Total: ~$1-2/month** (excluding Athena queries)

---

## Support

For issues or questions:
1. Check CloudWatch Logs for error messages
2. Review the main README in the project root
3. Consult AWS documentation for specific services
4. Check GitHub issues: [repository URL]

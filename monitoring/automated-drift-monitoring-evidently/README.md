# Credit Card Fraud Detection with SageMaker Pipelines

Production-grade ML pipeline for credit card fraud detection using AWS SageMaker Pipelines, MLflow tracking, and Athena data lake integration with comprehensive inference monitoring and drift detection.

## Overview

This project implements a complete ML lifecycle for fraud detection:

- **Training Pipeline:** SageMaker Pipelines with automated preprocessing, training (XGBoost), evaluation, and deployment
- **Experiment Tracking:** MLflow for model versioning, metrics, and artifact management
- **Data Lake:** Athena Iceberg tables for training data and inference logging
- **Custom Inference Handler:** Automatic logging of all predictions with buffered async writes
- **Monitoring & Drift Detection:** Continuous model performance tracking with Evidently-powered interactive reports
- **Ground Truth Integration:** Asynchronous ground truth capture and model retraining triggers

## Architecture Diagram

![MLOps Architecture](docs/guides/architecture_diagram.png)

> To regenerate: `python docs/generate_architecture_diagram.py` (requires `pip install diagrams` and `brew install graphviz`)

The diagram above shows the complete MLOps architecture including:
- **Training Pipeline** (top): Orchestrated by `pipeline_execution.ipynb`, includes preprocessing, training, deployment, and inference with SQS logging
- **Central Data Lake**: Athena tables for training data, inference responses, and ground truth updates
- **Monitoring Infrastructure** (bottom): Orchestrated by `inference_monitoring.ipynb`, includes manual drift analysis and automated monitoring with EventBridge, Lambda, and SNS alerts
- **MLflow Integration**: Tracks all training metrics, monitoring metrics, drift PSI values, and visualizations

## Diving Deep into Inference Monitoring Pipeline

![Inference Monitoring Pipeline](docs/guides/inference_monitoring_diagram.png)

This detailed diagram shows the **end-to-end inference monitoring flow** with MLflow as the central monitoring interface:

### Flow Breakdown:

**1. Real-Time Inference (Top)**
- Application sends transaction with 30 features → SageMaker Endpoint
- Endpoint runs XGBoost model, returns prediction + fraud probability
- Zero latency added to inference (async logging happens in background)

**2. Async Logging Pipeline**
- Inference handler sends prediction to SQS queue (fire-and-forget)
- Lambda consumer batches messages (10 msgs or 30s window)
- Writes batch to Athena inference_responses table
- **Partitioned by date** for efficient querying

**3. Ground Truth Integration**
- Fraud investigations confirm actual fraud (days/weeks later)
- Updates flow to ground_truth_updates table
- Athena MERGE updates inference_responses with confirmed labels
- Enables model performance tracking with real outcomes

**4. MLflow Monitoring Interface (Central Hub)**
MLflow serves as the **unified monitoring dashboard** where all monitoring workflows converge:

**Metrics Logged:**
- **Data Drift**: `drifted_columns_count`, `drifted_columns_share`, per-feature `drift_score_*` (via Evidently DataDriftPreset)
- **Model Performance**: `current_roc_auc`, `accuracy`, `precision`, `recall`, plus Evidently classification metrics (`evidently_accuracy`, `evidently_f1`, etc.)
- **Degradation**: `roc_auc_degradation`, `roc_auc_degradation_pct`
- **Detection Flags**: `data_drift_detected`, `model_drift_detected`

**Visualizations (artifacts/evidently_reports/):**
- Interactive Evidently HTML data drift report (PSI, KS, distribution comparisons per feature)
- Interactive Evidently HTML classification report (ROC curve, PR curve, confusion matrix, accuracy, F1)

**Artifacts:**
- `evidently_reports/data_drift_*.html` - Full interactive data drift dashboard
- `evidently_reports/classification_*.html` - Full interactive classification dashboard
- `drift_reports/drift_summary_*.json` - Structured JSON drift summary

**5. Monitoring Workflows**
- **Manual**: Data scientists run `inference_monitoring.ipynb` → query Athena → run Evidently reports → log interactive HTML reports to MLflow
- **Automated**: EventBridge triggers Lambda daily → runs Evidently DataDriftPreset + ClassificationPreset → logs HTML reports & metrics to MLflow → sends SNS alert if drift detected

**6. Alerting**
- SNS sends email notifications when thresholds exceeded
- Includes drifted features, performance metrics, and actionable recommendations
- Email contains links to MLflow for detailed analysis

### Key Differentiators:

✅ **MLflow as Single Pane of Glass** - All monitoring metrics, Evidently HTML reports, and artifacts in one interface
✅ **Evidently-Powered Drift Detection** - Interactive HTML reports with PSI, KS, distribution comparisons per feature
✅ **Automated + Manual** - Both notebook-based exploration and scheduled Lambda checks
✅ **Ground Truth Integration** - Handles delayed fraud confirmation typical in fraud detection
✅ **Zero Inference Latency** - Async logging doesn't impact prediction response time
✅ **Cost-Efficient** - $5/month Lambda + $25/month Athena vs. $200+/month for SageMaker Model Monitor

## Complete Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PHASE                                       │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                      SageMaker Pipeline                                   │  │
│  │  ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐    │  │
│  │  │Preprocess│ → │  Train  │ → │ Evaluate │ → │ Register │ → │ Deploy │    │  │
│  │  │ (Athena) │   │(XGBoost)│   │(Quality) │   │  (MLflow)│   │(Lambda)│    │  │
│  │  └──────────┘   └─────────┘   └──────────┘   └──────────┘   └────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         ↓                ↓               ↓                ↓                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      MLflow Tracking Server                             │    │
│  │   • Experiments  • Model Registry  • Metrics  • Artifacts  • Charts     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PHASE (T+0)                                  │
│                                                                                 │
│  Application → [SageMaker Endpoint] → Prediction Response                       │
│                  (Custom Handler)            ↓                                  │
│                       │                 {prediction, probabilities}             │
│                       │                                                         │
│                       ▼                                                         │
│               ┌───────────────┐                                                 │
│               │  SQS Queue    │  ← Fire-and-forget (per-prediction send)        │
│               │ (fraud-       │                                                 │
│               │  inference-   │                                                 │
│               │  logs)        │                                                 │
│               └───────┬───────┘                                                 │
│                       │  Batch: 10 messages or 30s window                       │
│                       ▼                                                         │
│               ┌───────────────┐                                                 │
│               │    Lambda     │  ← Triggered by SQS event                       │
│               │  (fraud-      │                                                 │
│               │  inference-   │  Builds INSERT INTO statement                   │
│               │  log-consumer)│  from batched SQS records                       │
│               └───────┬───────┘                                                 │
│                       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │              Athena: inference_responses (Iceberg)                      │    │
│  │   • inference_id, prediction, probability_fraud                         │    │
│  │   • ground_truth=NULL (awaiting confirmation)                           │    │
│  │   • latency_ms, confidence_score, transaction_context                   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   GROUND TRUTH CAPTURE (T+1 to T+30 days)                       │
│                                                                                 │
│  *** PRODUCTION: Replace with actual business processes ***                     │
│  - Fraud investigation outcomes                                                 │
│  - Chargeback notifications                                                     │
│  - Customer complaints                                                          │
│  - Merchant reports                                                             │
│                                                                                 │
│  *** DEVELOPMENT: Simulation for testing ***                                    │
│  src/monitoring/simulate_ground_truth_from_athena.py                            │
│  (Creates realistic ground truth with configurable accuracy)                    │
│                       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │            Athena: ground_truth_updates (Iceberg)                       │    │
│  │   • inference_id, actual_fraud, confirmation_timestamp                  │    │
│  │   • confirmation_source, investigation_notes                            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      GROUND TRUTH UPDATE (Batch Job)                            │
│                                                                                 │
│  src/monitoring/update_ground_truth.py --mode batch                             │
│                       ↓                                                         │
│  Athena MERGE: inference_responses ← ground_truth_updates                       │
│  (Updates ground_truth column for confirmed transactions)                       │
│                       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │         inference_responses (with ground truth populated)               │    │
│  │   WHERE ground_truth IS NOT NULL                                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌────────────────────────────────────────────────────────────────────────────────┐
│                    MONITORING & DRIFT DETECTION (Continuous)                   │
│                                                                                │
│  src/monitoring/monitor_model_performance.py --days 30                         │
│                       ↓                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Performance Metrics                                  │   │
│  │   • ROC-AUC, PR-AUC (where ground_truth IS NOT NULL)                    │   │
│  │   • Precision, Recall, F1-Score                                         │   │
│  │   • Confusion Matrix                                                    │   │
│  │   • Ground Truth Coverage: % with confirmed labels                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                       ↓                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     Data Drift Detection                                │   │
│  │   • Evidently DataDriftPreset: KS, PSI, and other statistical tests     │   │
│  │   • Per-column drift scores with statistical significance               │   │
│  │   • Interactive HTML reports saved to MLflow                            │   │
│  │   • Feature-level drift scores                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                       ↓                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Concept Drift Detection                              │   │
│  │   • Prediction-target correlation changes                               │   │
│  │   • High confidence error rate increase                                 │   │
│  │   • False positive/negative rate changes                                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                       ↓                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                 Visualization & Alerting                                │   │
│  │   • Evidently interactive HTML reports logged to MLflow                 │   │
│  │   • Data drift report (distributions, PSI, KS per feature)              │   │
│  │   • Classification report (ROC, PR, confusion matrix)                   │   │
│  │   • Alert if performance degrades below threshold                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                       ↓                                                        │
│            If drift detected or performance degraded:                          │
│                  → Trigger pipeline retraining                                 │
│                  → Update model in registry                                    │
│                  → Redeploy to endpoint                                        │
└────────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Athena Data Lake                                      │
│                                                                                 │
│  • training_data (284K rows) - Original training dataset                        │
│  • inference_responses (partitioned by date) - All predictions with metadata    │
│  • ground_truth_updates - Fraud confirmations from investigations               │
│  • ground_truth (historical) - Previous ground truth data  - plac               │
│  • drifted_data - Samples flagged for drift analysis                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Why This Architecture?

This solution addresses a critical gap: **lack of production-ready inference monitoring options** in the ML ecosystem. While training pipelines are well-served by managed platforms, comprehensive inference monitoring with drift detection, ground truth integration, and automated alerting requires custom solutions or expensive enterprise platforms.

**Key Differentiators:**

1. **Open-Source SDKs (MLflow, Evidently, Pandas, Scikit-learn)** - Ensures portability across AWS, GCP, Azure, or on-prem. No vendor lock-in. Industry-standard tools = easier hiring.

2. **Cost Efficiency** - $255/month vs. $1,850+/month for fully managed platforms (86% savings). Serverless architecture = pay only for actual usage, not idle capacity.

3. **Custom Inference Monitoring** - Evidently-powered drift detection (DataDriftPreset for all features, ClassificationPreset for model performance), automated ground truth integration, and EventBridge/Lambda alerting with interactive HTML reports logged to MLflow. Most platforms lack comprehensive inference monitoring or charge premium rates.

4. **Production-Grade Without Platform Costs** - SageMaker provides reliability (99.9% SLA) while MLflow provides portability. Best of both worlds: enterprise reliability + startup agility.

**vs. Alternatives:** Databricks ($24K/year) offers similar features but 8x more expensive and locked-in. Pure AWS (SageMaker Experiments) is 2x more expensive with limited portability. Kubeflow is portable but requires 40x more DevOps effort. This solution balances cost, portability, and operational simplicity.

**Ideal For:** AWS-native teams (1-10 people) needing production ML with comprehensive monitoring, audit trails, and multi-cloud optionality without enterprise platform budgets.

## Project Structure

```
automated-drift-monitoring-evidently/
├── src/
│   ├── pipeline/
│   │   ├── pipeline.py                       # Complete pipeline definition
│   │   ├── inference_handler.py              # Custom inference handler with Athena logging
│   │   ├── deploy.py                         # Manual model deployment (with custom handler)
│   │   ├── train.py                          # Manual training script
│   │   ├── test_endpoint.py                  # Endpoint testing with analytics
│   │   ├── batch_transform.py                # Batch transform for bulk scoring
│   │   ├── pipeline_cli.py                   # Pipeline CLI commands
│   │   ├── inference_requirements.txt        # Custom handler dependencies (awswrangler)
│   │   ├── pipeline_steps/
│   │   │   ├── preprocessing.py              # Data preprocessing (ScriptProcessor)
│   │   │   ├── preprocessing_pyspark.py      # PySpark preprocessing alternative
│   │   │   ├── train.py                      # XGBoost training script
│   │   │   ├── evaluation.py                 # Model evaluation with quality gates
│   │   │   ├── inference_monitoring.py       # Drift detection (PSI, KS, model drift)
│   │   │   ├── inference.py                  # Inference script
│   │   │   ├── lambda_deploy_endpoint.py     # Lambda for endpoint deployment
│   │   │   ├── lambda_test_inference.py      # Lambda for inference testing
│   │   │   ├── deploy_endpoint.py            # Endpoint deployment utilities
│   │   │   ├── test_inference.py             # Inference test utilities
│   │   │   ├── requirements_train.txt        # Training step dependencies
│   │   │   ├── requirements_evaluation.txt   # Evaluation step dependencies
│   │   │   └── requirements_preprocessing.txt # Preprocessing step dependencies
│   │   └── athena/
│   │       ├── athena_client.py              # Athena query operations
│   │       ├── athena_client_pyspark.py      # PySpark Athena integration
│   │       ├── data_migrator.py              # CSV → Athena Iceberg migration
│   │       ├── iceberg_manager.py            # Iceberg table management
│   │       └── schema_definitions.py         # Table schema definitions
│   ├── config/
│   │   ├── config.py                         # Configuration constants
│   │   └── config.yaml                       # Central configuration
│   ├── monitoring/
│   │   ├── evidently_reports.py              # Evidently-based drift & classification reports
│   │   ├── monitor_model_performance.py      # Performance monitoring & alerts
│   │   ├── lambda_drift_monitor.py           # Automated drift monitoring (Evidently + MLflow)
│   │   ├── lambda_inference_logger.py        # SQS-to-Athena inference log consumption
│   │   ├── generate_drift_dataset.py         # Generate drifted test data
│   │   ├── simulate_ground_truth_from_athena.py # Ground truth simulator (dev/test)
│   │   ├── generate_ground_truth_confirmations.py # Generate test confirmations
│   │   └── update_ground_truth.py            # Merge ground truth into inference records
│   ├── setup/
│   │   ├── setup_athena_tables.py            # Create Athena DB and tables
│   │   ├── setup_inference_logging.py        # SQS + Lambda inference logging
│   │   ├── setup_drift_monitoring.py         # Drift monitoring infrastructure
│   │   ├── setup_scheduled_inference.py      # EventBridge scheduled inference
│   │   ├── setup_scheduled_batch_transform.py # Scheduled batch transform
│   │   ├── upload_data_to_s3.py              # Upload local CSV data to S3
│   │   ├── create_or_update_sagemaker_role.py # IAM role setup
│   │   ├── create_lambda_role.py             # Lambda execution role
│   │   └── deploy_drift_monitoring.sh        # CI/CD deployment script
│   ├── utils/
│   │   ├── aws_session.py                    # AWS session helpers
│   │   ├── aws_utils.py                      # AWS session and role utilities
│   │   ├── mlflow_utils.py                   # MLflow tracking helpers
│   │   └── visualization_utils.py            # Chart generation for MLflow
│   └── analytics/
│       └── setup_quicksight_monitoring.py    # QuickSight dashboard setup
├── notebooks/
│   ├── pipeline_execution.ipynb              # Interactive pipeline control notebook
│   ├── inference_monitoring.ipynb            # Monitoring & drift detection notebook
│   └── inference_monitoring_with_pipeline.ipynb # Pipeline-based automated drift monitoring with Evidently
├── data/
│   ├── creditcard_predictions_final.csv      # Training data (284K rows, 30 features)
│   ├── creditcard_ground_truth.csv           # Ground truth labels
│   └── creditcard_drifted.csv                # Drifted data for testing
├── docs/
│   ├── generate_architecture_diagram.py      # Generates architecture_diagram.png with AWS icons
│   └── guides/                               # Architecture diagrams and screenshots
│       ├── architecture_diagram.png          # MLOps architecture (generated, Evidently version)
│       ├── architecture_diagram.excalidraw   # Excalidraw source for architecture diagram
│       ├── inference_monitoring_diagram.png   # Inference monitoring pipeline diagram
│       ├── inference_monitoring_diagram_excalidraw.png # Excalidraw export
│       ├── mermaid-diagram-mlflow-evidently.png # MLflow + Evidently monitoring flow
│       └── screenshots/                      # SageMaker Studio screenshots
├── main.py                                   # CLI entry point
├── .env.example                              # Environment template
└── README.md                                 # This file
```

## Quick Start

### Prerequisites

- Python 3.12+
- AWS account with SageMaker access
- `uv` package manager

### Installation

```bash
# Configure environment
cp .env.example .env
# Edit .env with your AWS settings
```

### Upload Data Files to S3

The CSV data files (`data/*.csv`) are excluded from git due to their size. On first run, these files must be available locally in the `data/` directory and then uploaded to S3 before the pipeline can use them.

**Step 1: Verify local data files exist**

```bash
ls -lh data/*.csv
# Expected files:
#   creditcard_predictions_final.csv  (~50 MB, 284K rows, training data)
#   creditcard_ground_truth.csv       (ground truth labels)
#   creditcard_drifted.csv            (drifted data for testing)
```

If the files are missing, obtain them from the original data source and place them in the `data/` directory.

**Step 2: Upload to S3**

```bash
# Preview what will be uploaded (dry run)
python -m src.setup.upload_data_to_s3 --dry-run

# Upload all CSV files to S3
python -m src.setup.upload_data_to_s3

# Upload a single file
python -m src.setup.upload_data_to_s3 --file data/creditcard_predictions_final.csv
```

This uploads the CSV files to `s3://<DATA_S3_BUCKET>/fraud-detection/data/`. The S3 paths are used by the Athena data migration step (`main.py setup --migrate-data`) and by the inference monitoring notebook for baseline drift detection.

### Configuration (.env) - change these in .env to match your environment

```bash
# MLflow - Use ARN format for programmatic access
MLFLOW_TRACKING_URI=arn:aws:sagemaker:us-east-1:YOUR_ACCOUNT:mlflow-app/app-YOUR_ID
MLFLOW_EXPERIMENT_NAME=credit-card-fraud-detection-training
MLFLOW_MODEL_NAME=xgboost-fraud-detector

# AWS
AWS_DEFAULT_REGION=us-east-1
SAGEMAKER_EXEC_ROLE=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole

# Athena
ATHENA_DATABASE=fraud_detection
DATA_S3_BUCKET=fraud-detection-data-lake-YOUR_ACCOUNT
ATHENA_OUTPUT_S3=s3://fraud-detection-data-lake-YOUR_ACCOUNT/athena-query-results/

# SQS
SQS_QUEUE_URL=<Set after configuring SQS>
SQS_QUEUE_NAME=fraud-inference-logs

LAMBDA_LOGGER_NAME=fraud-inference-log-consumer
```

### Add S3 Permissions For Data Migration

Add the following inline policy to your execution role through the UI to give S3 access:

```json
json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:CreateBucket",
        "s3:PutBucketVersioning",
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::<DATA_S3_BUCKET>",
        "arn:aws:s3:::<DATA_S3_BUCKET>/*"
      ]
    }
  ]
}
```

### Setup Infrastructure

```bash
# Upload CSV data to S3 (required before migration)
python -m src.setup.upload_data_to_s3

# Create S3 bucket, Athena database, Iceberg tables, and migrate CSV data
python main.py setup --migrate-data
```

This creates:
- S3 bucket for data lake and model artifacts
- Athena database `fraud_detection`
- 5 Iceberg tables: `training_data`, `inference_responses`, `ground_truth`, `ground_truth_updates`, `drifted_data`
- Migrates CSV data to Athena for pipeline access

### Add Permissions For Lambda Access

Add the following inline policy to your execution role to allow it to manage Lambda functions.

```json
{
  "Effect": "Allow",
  "Action": [
    "lambda:CreateFunction",
    "lambda:UpdateFunctionCode",
    "lambda:UpdateFunctionConfiguration",
    "lambda:GetFunction",
    "lambda:DeleteFunction",
    "lambda:InvokeFunction"
  ],
  "Resource": "arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:fraud-detection-*"
},
{
  "Effect": "Allow",
  "Action": "iam:PassRole",
  "Resource": "arn:aws:iam::YOUR_ACCOUNT:role/*",
  "Condition": {
    "StringEquals": {
      "iam:PassedToService": "lambda.amazonaws.com"
    }
  }
}
```

Add the following trust policy to allow Lambda to assume the execution role
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "sagemaker.amazonaws.com",
          "lambda.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### Configure SQS + Lambda Permissions
Add the following permissions to allow the creation of an SQS queue and lambda function for inference logging to Athena:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sqs:CreateQueue",
        "sqs:GetQueueUrl"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:GetQueueAttributes",
        "sqs:SendMessage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "lambda:CreateFunction",
        "lambda:UpdateFunctionCode",
        "lambda:GetFunction",
        "lambda:CreateEventSourceMapping"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "<EXECUTION_ROLE>",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": "lambda.amazonaws.com"
        }
      }
    }
  ]
}
```

### Add CloudWatch Permissions (for Drift Monitoring Dashboard & Alarms)

The `CloudWatchLogsFullAccess` managed policy only covers Logs. To publish custom metrics, create alarms, and build dashboards (Cell 40 in `inference_monitoring.ipynb`), add this inline policy to your execution role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData",
        "cloudwatch:PutMetricAlarm",
        "cloudwatch:DescribeAlarms",
        "cloudwatch:PutDashboard",
        "cloudwatch:GetDashboard",
        "cloudwatch:ListDashboards"
      ],
      "Resource": "*"
    }
  ]
}
```

### Setup SQS + Lambda
```
uv run main.py setup-logging
```

Update the SQS_URL in .env with the created SQS queue.

### Run Complete Pipeline

**Option 1: Jupyter Notebook (Recommended)**

```bash
# In SageMaker Studio, open:
notebooks/pipeline_execution.ipynb

# Run cells sequentially:
# Cell 1-3: Setup and configuration
# Cell 4: Create/update pipeline definition
# Cell 5: Execute pipeline and wait for completion (~25 minutes)
```

**Option 2: CLI**

```bash
# Create pipeline
python main.py pipeline create --pipeline-name fraud-detection-pipeline

# Execute pipeline
python main.py pipeline start --pipeline-name fraud-detection-pipeline --wait

# Check status
python main.py pipeline status --pipeline-name fraud-detection-pipeline
```

### Test Inference & Monitoring

```bash
# 1. Open monitoring notebook
# notebooks/inference_monitoring.ipynb

# 2. Run inference tests (Cells 1-9 in the notebook)
# Or via CLI:
# uv run main.py test --endpoint-name fraud-detector-endpoint --num-samples 50

# 3. Simulate ground truth (for development/testing)
python -m src.monitoring.simulate_ground_truth_from_athena --accuracy 0.85

# 4. Apply ground truth updates
python -m src.monitoring.update_ground_truth --mode batch

# 5. Run monitoring & drift detection (Cells 27-40 in inference_monitoring.ipynb)
# Generates 8+ charts, calculates metrics, detects drift
```

**Expected Output:**
- Predictions logged to Athena automatically
- 100% ground truth coverage (simulated)
- Performance metrics: ROC-AUC, precision, recall
- Drift detection with visualizations
- 8 charts uploaded to MLflow

## Custom Inference Handler

### Why a Custom Handler?

The pipeline deploys models with a **custom inference handler** instead of SageMaker's built-in XGBoost serving:

| Feature | Built-in XGBoost | Custom Handler |
|---------|------------------|----------------|
| Predictions | ✅ | ✅ |
| Input Format | CSV, libsvm | JSON (extensible) |
| Output Format | Single probability | Detailed predictions + probabilities |
| **Athena Logging** | ❌ Manual | ✅ **Automatic** |
| MLflow Integration | ❌ | ✅ |
| Performance Metrics | ❌ | ✅ Latency, confidence |
| Monitoring Flags | ❌ | ✅ High/low confidence detection |
| Production-Ready | ⚠️ Simple use cases | ✅ Enterprise-grade |

### Key Features

#### 1. Automatic Athena Logging

Every inference request is logged with:
- Prediction metadata: inference_id, timestamp, prediction, probabilities
- Performance: latency (total, preprocessing, model), confidence scores
- Transaction context: transaction_id, amount, customer_id
- Model info: endpoint_name, model_version, mlflow_run_id

#### 2. SQS + Lambda Async Writes

Inference logs are written asynchronously via SQS to avoid impacting prediction latency:

```
Custom Handler → SQS (fire-and-forget, per prediction)
                   ↓
              SQS batches messages (10 msgs or 30s window)
                   ↓
              Lambda triggered (lambda_inference_logger.py)
                   ↓
              INSERT INTO fraud_detection.inference_responses
```

- **Zero latency impact** — SQS `send_message` is fire-and-forget in the inference handler
- **Batched writes** — Lambda receives up to 10 SQS records per invocation
- **Athena INSERT** — Lambda builds a multi-row `INSERT INTO` statement and executes via Athena
- **Graceful failure** — SQS retains messages for 24 hours if Lambda fails

#### 3. JSON Input/Output Format

**Request (30 features):**
```json
{
  "transaction_hour": 14,
  "transaction_day_of_week": 2,
  "transaction_amount": 149.62,
  "customer_age": 42,
  "customer_gender": 0,
  "distance_from_home_km": 5.2,
  "merchant_category_code": 5411,
  "chip_transaction": 1,
  "num_transactions_24h": 3,
  "credit_limit": 5000.0,
  "available_credit_ratio": 0.75
  // ... (all 30 training features)
}
```

**Response:**
```json
{
  "predictions": [0],
  "probabilities": {
    "fraud": [0.1234],
    "non_fraud": [0.8766]
  }
}
```

### Testing in SageMaker Studio Endpoint Playground

After deployment, you can test the endpoint directly in SageMaker Studio without writing code.

**Access Path:**
```
SageMaker Studio > Deployments > Endpoints > Fraud Detector Endpoint > Playground
```

**Steps:**

1. **Navigate to endpoint:**
   - Open SageMaker Studio
   - Left sidebar: Deployments → Endpoints
   - Click on `fraud-detector-endpoint`
   - Click "Playground" tab

2. **Configure request:**
   - Content-Type: `application/json`
   - Request body: Paste JSON payload with 30 features

3. **Example payload:**
   ```json
   {
     "transaction_hour": 14,
     "transaction_day_of_week": 2,
     "transaction_amount": 149.62,
     "transaction_type_code": 1,
     "customer_age": 42,
     "customer_gender": 0,
     "customer_tenure_months": 36,
     "account_age_days": 1095,
     "distance_from_home_km": 5.2,
     "distance_from_last_transaction_km": 2.3,
     "time_since_last_transaction_min": 120,
     "online_transaction": 1,
     "international_transaction": 0,
     "high_risk_country": 0,
     "merchant_category_code": 5411,
     "merchant_reputation_score": 0.85,
     "chip_transaction": 1,
     "pin_used": 1,
     "card_present": 1,
     "cvv_match": 1,
     "address_verification_match": 1,
     "num_transactions_24h": 3,
     "num_transactions_7days": 12,
     "avg_transaction_amount_30days": 125.50,
     "max_transaction_amount_30days": 450.00,
     "velocity_score": 0.3,
     "recurring_transaction": 0,
     "previous_fraud_incidents": 0,
     "credit_limit": 5000.0,
     "available_credit_ratio": 0.75
   }
   ```

4. **Click "Test"**

5. **Expected response:**
   ```json
   {
     "predictions": [0],
     "probabilities": {
       "fraud": [0.1234],
       "non_fraud": [0.8766]
     }
   }
   ```

**Screenshot:** See `screenshots/DirectTestingInSGPlayground-custom-handler.png` for visual guide

**When to use:**
- Quick manual testing without code
- Debugging endpoint issues
- Demonstrating to stakeholders
- Validating deployment after pipeline execution
- Generating sample payloads for documentation

**Note:** The playground sends requests to your deployed endpoint, so predictions will be automatically logged to Athena (custom handler feature).

### Implementation Location

- **Inference Handler (SQS):** `src/pipeline/pipeline_steps/inference.py` — sends each prediction to SQS
- **Lambda Consumer:** `lambda_inference_logger.py` — reads SQS batches, writes to Athena
- **Setup Script:** `src/setup/setup_inference_logging.py` — creates SQS queue + Lambda + event source mapping
- **Legacy Handler (Direct Athena):** `src/pipeline/inference_handler.py` — buffered awswrangler writes (alternative)
- **Dependencies:** `src/pipeline/inference_requirements.txt`
- **Pipeline Integration:** `src/pipeline/pipeline.py`

**How it works (SQS path):**
```python
# In the inference handler (predict_fn)
sqs.send_message(
    QueueUrl=SQS_QUEUE_URL,
    MessageBody=json.dumps({
        'inference_id': str(uuid.uuid4()),
        'request_timestamp': datetime.utcnow().isoformat(),
        'prediction': int(predictions[idx]),
        'probability_fraud': fraud_prob,
        # ... 22 more fields
    }),
)

# Lambda consumer (lambda_inference_logger.py)
# Triggered by SQS with batch of up to 10 messages
# Builds: INSERT INTO fraud_detection.inference_responses VALUES (row1), (row2), ...
# Executes via athena.start_query_execution()
```

## SageMaker Pipeline Workflow

### Pipeline Steps

1. **Preprocess** - Read from Athena, validate, encode categorical, split train/test (80/20)
2. **Train** - XGBoost with automatic class imbalance handling (scale_pos_weight)
3. **Evaluate** - Calculate ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix
4. **Quality Gate** - Register model only if ROC-AUC ≥ 0.85 AND PR-AUC ≥ 0.50
5. **Deploy** - Create endpoint with custom handler and Athena logging

### Create Pipeline

**Via Notebook:**
```python
# In pipeline_execution.ipynb Cell 4
result = pipeline_builder.upsert_pipeline(
    pipeline_name='fraud-detection-pipeline',
    pipeline_display_name='Credit Card Fraud Detection Pipeline'
)
```

**Via CLI:**
```bash
python main.py pipeline create --pipeline-name fraud-detection-pipeline
```

### Execute Pipeline

**Via Notebook:**
```python
# Cell 5: Execute and wait
execution = pipeline.start()
execution.wait(delay=60, max_attempts=30)
```

**Via CLI:**
```bash
# With waiting
python main.py pipeline start --pipeline-name fraud-detection-pipeline --wait

# Without waiting (monitor in console)
python main.py pipeline start --pipeline-name fraud-detection-pipeline
```

### Monitor Execution

**AWS Console:**
```
SageMaker → Pipelines → fraud-detection-pipeline → Executions
```

**CLI:**
```bash
# Get latest execution status
python main.py pipeline status --pipeline-name fraud-detection-pipeline

# List all executions
python main.py pipeline list-executions --pipeline-name fraud-detection-pipeline
```

**Expected Duration:** ~25 minutes
- Preprocess: 3-5 min
- Train: 10-15 min
- Evaluate: 2-3 min
- Deploy: 5-7 min

## MLflow Integration

### Configuration

**ARN Format (Programmatic Access):**
```python
# Use ARN for SDK and pipeline access
MLFLOW_TRACKING_URI = "arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-ABC123"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
```

**HTTPS Format (Web UI Only):**
```
# Use HTTPS URL only for opening in browser
https://abc123def.mlflow-tracking.us-east-1.sagemaker.aws/
```

### Training Integration

The training step automatically logs to MLflow:

```python
# In training step
mlflow.xgboost.autolog()  # Automatic parameter and metric logging

with mlflow.start_run():
    model.fit(X_train, y_train)
    # MLflow automatically logs:
    # - Hyperparameters (max_depth, learning_rate, etc.)
    # - Training metrics (logloss, error)
    # - Model artifacts
    # - Feature importance
    # - Confusion matrix
```

### Accessing MLflow UI

**From SageMaker Studio:**
```
1. Left sidebar → Partner AI Apps → MLflow
2. Click on your MLflow app
3. Opens in new tab with authentication
```

**Experiments:**
- `credit-card-fraud-detection-training` - Training runs
- `credit-card-fraud-detection-inference` - Inference monitoring
- `credit-card-fraud-detection-batch` - Batch transform jobs

### Model Registry

Models are automatically registered on successful evaluation:

```python
# In evaluation step (if ROC-AUC ≥ 0.85)
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="xgboost-fraud-detector"
)
```

**Model Versions:**
- Version 1, 2, 3... (incremented automatically)
- Stages: `None` → `Staging` → `Production` → `Archived`
- Tags: training_date, pipeline_execution_id, roc_auc, pr_auc

## Inference Monitoring & Drift Detection

### Monitoring Architecture

```
Inference (T+0) → SQS → Lambda → Athena: inference_responses (ground_truth=NULL)
                                                    ↓
Ground Truth Capture (T+1 to T+30)        → ground_truth_updates
                                                    ↓
Batch MERGE                               → inference_responses (ground_truth populated)
                                                    ↓
Monitoring                                → Calculate metrics where ground_truth IS NOT NULL
```

### Ground Truth Capture

**🔔 IMPORTANT: Production Implementation Required**

The current implementation uses **simulation for development/testing**. In production, replace with actual business processes:

**Production Sources:**
- **Fraud Investigation Systems:** Confirmed fraud cases from investigation team
- **Chargeback Notifications:** Disputed transactions from payment processors
- **Customer Complaints:** Fraud reports from customer service
- **Merchant Reports:** Suspicious activity flagged by merchants
- **Law Enforcement:** Cases confirmed by authorities

**Implementation Pattern:**
```python
# Example: Fraud investigation system integration
def capture_investigation_result(transaction_id, is_fraud, source):
    """
    Called when fraud investigation completes.
    Replace simulate_ground_truth_from_athena.py with this in production.
    """
    ground_truth_update = {
        'transaction_id': transaction_id,
        'inference_id': lookup_inference_id(transaction_id),
        'actual_fraud': is_fraud,
        'confirmation_timestamp': datetime.now(),
        'confirmation_source': source,  # e.g., 'fraud_investigation'
        'investigation_notes': get_investigation_notes(transaction_id)
    }

    # Write to ground_truth_updates table
    write_to_athena(ground_truth_update)
```

**Development/Testing (Current):**
```bash
# Simulates realistic ground truth with configurable accuracy
python -m src.monitoring.simulate_ground_truth_from_athena --accuracy 0.85
```

This creates realistic confirmations:
- Fraud cases: 1-7 days confirmation delay
- Non-fraud: 1-30 days confirmation delay
- Configurable accuracy (default 85%)
- Realistic confirmation sources

### Applying Ground Truth Updates

```bash
# Batch mode: Process all pending updates
python -m src.monitoring.update_ground_truth --mode batch

# Streaming mode: Process recent updates (last 24 hours)
python -m src.monitoring.update_ground_truth --mode streaming --window-hours 24
```

This performs Athena MERGE to update `ground_truth` column in `inference_responses`.

### Performance Monitoring

```bash
# Monitor last 30 days
python -m src.monitoring.monitor_model_performance --days 30

# With custom alert threshold
python -m src.monitoring.monitor_model_performance \
  --days 30 \
  --alert-threshold 0.80 \
  --endpoint fraud-detector-prod
```

**Output:**
```
Ground Truth Coverage:
  Total predictions: 50,000
  With ground truth: 12,500 (25.00%)
  Avg days to confirmation: 8.5

Overall Performance (last 30 days):
  ROC-AUC: 0.8742
  PR-AUC: 0.6521
  Precision: 0.7245
  Recall: 0.6891
  F1-Score: 0.7063

✓ No performance degradation detected
```

### Drift Detection Methods

All drift detection is powered by [Evidently AI](https://www.evidentlyai.com/) (v0.7.x). The notebook and Lambda both use `evidently_reports.py` which wraps Evidently's `DataDriftPreset` and `ClassificationPreset`.

#### Data Drift (Feature Distribution Changes)

Evidently's `DataDriftPreset` automatically selects the appropriate statistical test per feature (KS for numeric, chi-square for categorical) and computes drift scores. The interactive HTML report includes per-feature distribution comparisons, PSI values, and statistical test results.

**Legacy reference:** The `calculate_psi()` and `calculate_ks_statistic()` functions in `lambda_drift_monitor.py` are preserved as legacy examples showing how to compute these metrics manually without Evidently.

**Population Stability Index (PSI):**

PSI measures how much a feature's distribution has shifted between the baseline (training) period and the current (inference) period.

**Formula:**

```
PSI = Σ (actual_pct_i - expected_pct_i) × ln(actual_pct_i / expected_pct_i)
```

Where `expected_pct_i` and `actual_pct_i` are the proportion of observations in bin `i` for the baseline and current distributions, respectively.

**How it's computed** (see `inference_monitoring.py:86-124`):

1. Create bin edges from the baseline (training) data using percentile-based breakpoints (10 bins by default)
2. Histogram both baseline and current data into the same bins
3. Normalize each histogram to proportions (percentage of total in each bin)
4. Apply a floor of `0.0001` to avoid `ln(0)` or division by zero
5. Sum the PSI contribution from each bin: `(actual% - expected%) × ln(actual% / expected%)`

**Interpretation thresholds:**

| PSI Value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.1     | No significant shift | None |
| 0.1 – 0.2 | Moderate shift | Monitor closely |
| ≥ 0.2     | Significant shift | Investigate / retrain |

**Avg PSI (Population-Level Drift Score):**

`Avg PSI` is the **simple arithmetic mean of per-feature PSI values** across all monitored features. It provides a single scalar summarizing overall dataset drift.

```python
# From detect_data_drift() → summary
avg_psi = mean(psi_feature_1, psi_feature_2, ..., psi_feature_N)
```

This metric is logged to MLflow as `drift_avg_psi` alongside `drift_max_psi` and `drift_median_psi`.

A feature is flagged as drifted if **either** its PSI ≥ 0.2 **or** its KS test p-value < 0.05.

**Feature Extraction for Drift Detection:**

The drift detection process (Cell 31 in `inference_monitoring.ipynb`) analyzes **all 30 training features** by parsing the `input_features` JSON column:

```python
import json

# Parse JSON to extract all features from inference data
feature_rows = []
for idx, row in inference_raw.iterrows():
    try:
        # Parse the input_features JSON string
        features_dict = json.loads(row['input_features'])

        # Extract exactly the 30 training features
        feature_values = {feat: features_dict.get(feat, np.nan) for feat in TRAINING_FEATURES}
        feature_rows.append(feature_values)
    except Exception as e:
        print(f'  ⚠ Failed to parse row {idx}: {e}')
        continue

# Create DataFrame with all 30 features
inference_numeric = pd.DataFrame(feature_rows)

# Run drift detection on ALL features
drift_results = detect_data_drift(
    baseline_data=baseline_numeric[common_features],  # 284K rows × 30 features
    current_data=inference_numeric,                   # Inference rows × 30 features
    feature_names=common_features,                    # All 30 features
    threshold_psi=0.2,
    threshold_ks=0.05,
)
```

**Expected Output:**
```
================================================================================
DATA DRIFT ANALYSIS SUMMARY
================================================================================
  Total features analyzed: 30          ← All features extracted from JSON
  Drifted features: 5
  Drift percentage: 16.7%
  Avg PSI: 0.0842
  Max PSI: 0.2145

⚠ DRIFTED FEATURES (5):
================================================================================
  🔴 CRITICAL transaction_amount: PSI=0.2145
  🟠 MODERATE customer_age: PSI=0.1523
  🟠 MODERATE distance_from_home_km: PSI=0.1205
  🟡 MINOR online_transaction: PSI=0.0923
  🟡 MINOR merchant_reputation_score: PSI=0.0812
```

**Note:** Previously, only `transaction_amount` was analyzed because the other 29 features were stored in the `input_features` JSON column. The updated Cell 31 now parses this JSON to extract all features, providing comprehensive drift detection across the entire feature set.

**Chi-Square Test:**
```python
# For categorical features
chi2_stat, p_value = chi2_test(reference_counts, current_counts)

# Alert if p_value < 0.05 (distribution changed significantly)
```

**Kolmogorov-Smirnov (KS) Test:**
```python
# For continuous feature distributions
ks_stat, p_value = ks_test(reference_values, current_values)

# Alert if p_value < 0.05
```

#### Statistical Tests: KS vs Wasserstein (Configuration Guide)

This project uses **Evidently** for advanced drift detection with configurable statistical tests. For **fraud detection**, we default to the **Kolmogorov-Smirnov (KS) test** over Wasserstein distance.

##### Why KS Test for Fraud Detection?

**KS Test Advantages:**
- **Tail Sensitivity ⭐** - Detects maximum difference in CDFs, crucial for catching rare fraud pattern changes
- **Statistical Significance** - Returns p-value (e.g., p < 0.05 = 95% confidence), helps distinguish real drift from noise
- **Distribution-Free** - No assumptions about underlying distributions (robust to non-normal data)
- **Shape Detection** - Catches changes in distribution shape, not just location/scale shifts
- **Interpretable** - KS statistic = max vertical distance between CDFs (0-1 scale)

**Why Not Wasserstein?**
- **Less Tail-Sensitive** - Can miss subtle changes in extreme values (where fraud patterns emerge)
- **No P-Value** - Harder to set universal thresholds across different features
- **Mean-Focused** - Better for overall distribution shifts than detecting new fraud tactics
- **Computationally Heavier** - O(n log n) vs O(n) for KS test

**Real-World Example:**
```
Scenario: New fraud ring targeting $500-$600 transactions

Baseline:  Transactions spread $0-$1000 (normal distribution)
Current:   Slight spike at $500-600 (2% of transactions)

Wasserstein: 0.08 (small overall shift, might not alert)
KS Test:     0.15, p=0.001 (significant tail change, alerts)
```

For fraud detection, **catching rare pattern changes early** (KS strength) is more critical than measuring overall distribution distance (Wasserstein strength).

##### Configuration

Configure the statistical test in `.env`:

```bash
# Statistical test for numerical features
# Options: 'ks' (default), 'wasserstein', 'kl_div', 'psi', 'jensenshannon'
EVIDENTLY_NUM_STAT_TEST=ks

# KS test thresholds
KS_DRIFT_THRESHOLD=0.1          # Moderate drift threshold
KS_PVALUE_THRESHOLD=0.05        # Statistical significance (95% confidence)
KS_ENABLE_CDF_PLOTS=true        # Generate CDF comparison visualizations
KS_MAX_FEATURES_TO_PLOT=10      # Top N drifted features to visualize

# Categorical feature test (unchanged)
EVIDENTLY_CAT_STAT_TEST=chi_square
```

**Configuration in Code:**
```python
# src/config/config.py (lines 202-210)
EVIDENTLY_NUM_STAT_TEST = os.getenv('EVIDENTLY_NUM_STAT_TEST', 'ks')
KS_DRIFT_THRESHOLD = float(os.getenv('KS_DRIFT_THRESHOLD', '0.1'))
KS_PVALUE_THRESHOLD = float(os.getenv('KS_PVALUE_THRESHOLD', '0.05'))
```

##### KS-Specific Visualizations

When KS test is enabled (`EVIDENTLY_NUM_STAT_TEST=ks`), additional visualizations are generated:

**1. CDF Comparison Plots** - Shows empirical CDFs for baseline vs current distributions with maximum KS distance marked

**2. KS Statistics Heatmap** - Two-panel visualization:
   - Left: KS statistic bars (0-1 scale, color-coded by severity)
   - Right: -log10(p-value) bars (statistical significance)

**3. Enhanced MLflow Metrics:**
```python
# Aggregated KS metrics
ks_mean_ks_statistic
ks_max_ks_statistic
ks_median_ks_statistic
ks_features_moderate_drift    # 0.1 ≤ KS < 0.2
ks_features_high_drift        # KS ≥ 0.2
ks_features_significant       # p-value < 0.05
```

##### Dual-Threshold Alerting

The KS test uses **both** the KS statistic and p-value for robust drift detection:

```python
# Drift detection logic
if ks_stat > 0.2 and p_value < 0.05:
    severity = "CRITICAL"    # Immediate investigation
elif ks_stat > 0.1 and p_value < 0.05:
    severity = "MODERATE"    # Monitor closely
elif ks_stat > 0.1 or p_value < 0.05:
    severity = "WARNING"     # Single threshold met
else:
    severity = "NORMAL"
```

This dual-threshold approach reduces false positives while maintaining high sensitivity to real drift.

##### Switching to Wasserstein (If Needed)

To use Wasserstein distance instead:

```bash
# In .env
EVIDENTLY_NUM_STAT_TEST=wasserstein
EVIDENTLY_DRIFT_THRESHOLD=0.2    # Overall drift threshold
```

**When to use Wasserstein:**
- General ML pipelines (non-fraud use cases)
- Measuring overall distribution distance
- When p-values are not required
- Benchmarking against academic papers using EMD

**Best Practice:** Monitor **both** metrics and compare. If both spike → high confidence drift. If only KS spikes → investigate tail changes.

#### Concept Drift (Prediction-Target Relationship Changes)

**Metrics Tracked:**
- ROC-AUC degradation (alert if drops > 5%)
- Precision/Recall changes
- False positive rate increase
- False negative rate increase
- High confidence error rate

### Interactive Monitoring (Notebook)

```bash
# Open: notebooks/inference_monitoring.ipynb

# Cell 16: Check ground truth coverage
# Cell 31: Data Drift Detection - Runs Evidently DataDriftPreset on all 30 features
# Cell 32: Display interactive Evidently data drift report
# Cell 34-35: Model Drift Detection - Runs Evidently ClassificationPreset
# Cell 37: Log Evidently HTML reports and metrics to MLflow
```

**Evidently Reports Logged to MLflow:**
1. `evidently_reports/data_drift_*.html` — Interactive data drift dashboard with per-feature PSI, KS, distribution comparisons
2. `evidently_reports/classification_*.html` — Interactive classification dashboard with ROC curve, PR curve, confusion matrix, accuracy, F1
3. `drift_reports/drift_summary_*.json` — Structured JSON summary

All reports are logged to MLflow experiment: `credit-card-fraud-detection-monitoring`

## Visualizations Quick Reference

| Chart | Purpose | Location | When Generated |
|-------|---------|----------|----------------|
| **Preprocessing** |
| Class Distribution | Verify imbalance handling | MLflow Training Run | During preprocessing |
| Feature Correlations | Identify multicollinearity | MLflow Training Run | During preprocessing |
| Missing Values | Data quality check | MLflow Training Run | During preprocessing |
| Feature Distributions | Understand data | MLflow Training Run | During preprocessing |
| **Training** |
| Training Loss Curve | Convergence check | MLflow Training Run | During training |
| Feature Importance | Top predictive features | MLflow Training Run | After training |
| ROC Curve | Model discrimination | MLflow Training Run | During evaluation |
| Precision-Recall Curve | Performance at thresholds | MLflow Training Run | During evaluation |
| Confusion Matrix | Classification accuracy | MLflow Training Run | During evaluation |
| **Monitoring (Evidently)** |
| Data Drift HTML Report | Per-feature drift analysis (PSI, KS, distributions) | MLflow `evidently_reports/` | inference_monitoring.ipynb Cell 37 |
| Classification HTML Report | Model performance (ROC, PR, confusion matrix, F1) | MLflow `evidently_reports/` | inference_monitoring.ipynb Cell 37 |
| Drift Summary JSON | Structured drift summary | MLflow `drift_reports/` | inference_monitoring.ipynb Cell 37 |

**Accessing in MLflow:**
```
1. Open MLflow UI (SageMaker Studio → Partner AI Apps)
2. Select experiment:
   - Training: credit-card-fraud-detection-training
   - Monitoring: credit-card-fraud-detection-monitoring
3. Click on run
4. Scroll to "Artifacts" section
5. Evidently HTML reports available under:
   - evidently_reports/data_drift_*.html (open in browser for interactive dashboard)
   - evidently_reports/classification_*.html (open in browser for interactive dashboard)
   - drift_reports/drift_summary_*.json
```

### Baseline Data for Drift Detection

Drift detection compares current inference feature distributions against a **baseline**. The baseline should represent the data the model was trained on.

**Current approach** (`inference_monitoring.ipynb`):
```python
baseline_df = pd.read_csv(CSV_TRAINING_DATA)
```

This loads the CSV file directly from disk. Since the same data has already been migrated to the Athena `training_data` table (via `main.py setup --migrate-data`), the baseline can alternatively be loaded from Athena using PySpark or boto3:

**Option 1: PySpark (for large-scale processing):**
```python
from athena.athena_client_pyspark import AthenaClientPySpark

client = AthenaClientPySpark(database="fraud_detection")
baseline_spark_df = client.read_table("training_data")
baseline_df = baseline_spark_df.toPandas()  # Convert to pandas for drift analysis
```

**Option 2: Boto3 + Pandas (for monitoring/small queries):**
```python
import boto3
import pandas as pd
import io
import time

def read_athena_query(sql, database, output_location):
    """Execute Athena query and return pandas DataFrame."""
    athena = boto3.client('athena')
    s3 = boto3.client('s3')

    # Start query execution
    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': output_location}
    )
    execution_id = response['QueryExecutionId']

    # Wait for completion
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution_id)
        state = status['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)

    if state != 'SUCCEEDED':
        raise Exception(f"Query failed: {state}")

    # Download results from S3
    result_s3_path = status['QueryExecution']['ResultConfiguration']['OutputLocation']
    bucket, key = result_s3_path.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)

    return pd.read_csv(io.BytesIO(obj['Body'].read()))

# Usage
baseline_df = read_athena_query(
    sql="SELECT * FROM training_data LIMIT 10000",
    database="fraud_detection",
    output_location="s3://your-bucket/athena-results/"
)
```

**Why use the Athena `training_data` table as baseline:**
- **Single source of truth** — the same table the training pipeline reads from, so schema and column types are guaranteed to match
- **No local file dependency** — works in any SageMaker environment without needing the CSV on disk
- **Consistent with pipeline** — the preprocessing step reads from `training_data` via PySpark, so monitoring and training reference the same data
- **Versioned via Iceberg** — if the training data is updated (new features, augmented samples), the Athena table reflects that without needing to re-distribute CSV files

**Note:** For the current PoC, CSV loading is sufficient. In production, the Athena table is the preferred source since it stays in sync with whatever the training pipeline consumes. **awswrangler has been deprecated** in favor of PySpark (for large-scale processing) and boto3 (for monitoring queries).

## Automated Drift Monitoring

**NEW**: Automated drift detection with EventBridge, Lambda, and SNS email alerts.

### Overview

The system automatically monitors for data drift and model performance drift, sending email alerts when thresholds are exceeded.

**Architecture:**
```
EventBridge Rule → Lambda Function → SNS Topic → Email
(daily at 2am)     (drift detection)   (alerts)     (ops team)
                         ↓
                  Athena Data Lake
                (inference_responses)
```

### Quick Setup

**Option 1: Using Notebook (Recommended)**
1. Open `notebooks/inference_monitoring.ipynb`
2. Navigate to **Section 6: Automated Drift Monitoring Setup**
3. Run cells 6.1-6.3 to deploy infrastructure
4. Confirm email subscription

**Option 2: Using CI/CD Script**
```bash
export ALERT_EMAIL="ops@example.com"
export ATHENA_DATABASE="fraud_detection"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ATHENA_OUTPUT_S3="s3://fraud-detection-data-lake-skoppar-${AWS_ACCOUNT_ID}/athena-query-results/"

bash src/setup/deploy_drift_monitoring.sh
```

**Option 3: Interactive Setup**
```bash
python src/setup/setup_drift_monitoring.py
```

### What Gets Created

1. **SNS Topic** (`fraud-detection-drift-alerts`) - Email notifications
2. **Lambda Function** (`fraud-detection-drift-monitor`) - Drift detection logic
3. **IAM Role** - Lambda permissions (Athena, S3, SNS)
4. **EventBridge Rule** (`fraud-detection-drift-check`) - Daily schedule (2 AM UTC)

### Monitoring Capabilities

**Data Drift Detection:**
- **Engine**: Evidently DataDriftPreset (KS, PSI, and other statistical tests)
- **Scope**: All 30 training features
- **Output**: Interactive HTML report + per-column drift scores
- **Method**: Compares recent 24h inference data to training baseline

**Model Performance Drift:**
- **Engine**: Evidently ClassificationPreset (accuracy, F1, ROC, PR, confusion matrix)
- **Threshold**: >5% ROC-AUC degradation (configurable)
- **Method**: Compares current performance to baseline (0.92)
- **Requires**: Ground truth labels for predictions

**MLflow Integration:**
- **Automatic Logging**: All drift metrics and Evidently HTML reports logged to MLflow on each check
- **Experiment**: `fraud-detection-drift-monitoring`
- **Metrics Logged**:
  - Per-feature drift scores (`drift_score_<feature_name>`)
  - Aggregate drift statistics (drifted_columns_count, drifted_columns_share)
  - Model performance metrics (ROC-AUC, accuracy, precision, recall)
  - Evidently classification metrics (`evidently_accuracy`, `evidently_f1`, etc.)
  - Sample sizes and detection flags
- **Artifacts**:
  - `evidently_reports/data_drift_*.html` — Interactive data drift dashboard
  - `evidently_reports/classification_*.html` — Interactive classification dashboard
  - `drift_reports/drift_summary_*.json` — Structured JSON summary
- **Configuration**: Set `MLFLOW_TRACKING_URI` environment variable in Lambda

### Email Alert Example

```
================================================================================
ML MODEL DRIFT ALERT
================================================================================
Time: 2026-02-24 02:00:15
Detection Engine: Evidently AI

🔴 DATA DRIFT DETECTED (Evidently DataDriftPreset)
Features Analyzed: 30
Drifted Features: 5 (16.7%)
Drifted Columns Share: 16.7%

Top Drifted Features (by drift score):
  - transaction_amount: drift_score=0.0012
  - customer_age: drift_score=0.0034
  - distance_from_home_km: drift_score=0.0089

🔴 MODEL PERFORMANCE DRIFT DETECTED (Evidently ClassificationPreset)
Baseline ROC-AUC: 0.9200
Current ROC-AUC: 0.5713
Degradation: 0.3487 (37.9%)

RECOMMENDED ACTIONS:
1. Review Evidently HTML reports in MLflow monitoring experiment
2. Investigate root cause
3. Consider retraining with recent data
4. Review decision thresholds
================================================================================
```

### Configuration

**Thresholds (customizable):**
```bash
export DATA_DRIFT_THRESHOLD=0.2   # PSI threshold
export MODEL_DRIFT_THRESHOLD=0.05  # 5% performance degradation
export BASELINE_ROC_AUC=0.92      # Expected model performance
export MLFLOW_TRACKING_URI="arn:aws:sagemaker:us-east-1:ACCOUNT:mlflow-app/app-ID"  # For MLflow logging
```

**Schedule Options:**
```bash
# Daily at 2 AM UTC (default)
cron(0 2 * * ? *)

# Every 6 hours
cron(0 */6 * * ? *)

# Every Monday at 9 AM
cron(0 9 ? * MON *)
```

### Testing

**Manual Trigger:**
```bash
# Via AWS CLI
aws lambda invoke \
    --function-name fraud-detection-drift-monitor \
    output.json

# Via notebook (Cell 6.4)
lambda_client.invoke(FunctionName='fraud-detection-drift-monitor')
```

**View Logs:**
```bash
aws logs tail /aws/lambda/fraud-detection-drift-monitor --follow
```

### Files

- `src/monitoring/lambda_drift_monitor.py` - Lambda function code
- `src/monitoring/lambda_inference_logger.py` - Lambda inference logger
- `src/monitoring/generate_drift_dataset.py` - Generate test drift data
- `src/setup/setup_drift_monitoring.py` - Interactive setup
- `src/setup/deploy_drift_monitoring.sh` - CI/CD deployment script
- `notebooks/inference_monitoring.ipynb` - Section 6 (setup cells)

### Thresholds Explained

**Data Drift (PSI Thresholds):**

| PSI Value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.1 | No significant shift | None |
| 0.1 - 0.2 | Moderate shift | Monitor closely |
| ≥ 0.2 | Significant shift | **Alert triggered** |

**Model Drift (Performance Degradation):**

| Degradation | Interpretation | Action |
|-------------|----------------|--------|
| < 3% | Normal variance | None |
| 3-5% | Minor degradation | Monitor |
| > 5% | Significant degradation | **Alert triggered** |

### Updating Configuration

**Change Thresholds:**
```python
# In inference_monitoring.ipynb, Cell 6.6
lambda_client.update_function_configuration(
    FunctionName='fraud-detection-drift-monitor',
    Environment={
        'Variables': {
            'DATA_DRIFT_THRESHOLD': '0.15',  # More sensitive
            'MODEL_DRIFT_THRESHOLD': '0.03',  # 3% degradation
            # ... other vars
        }
    }
)
```

**Change Schedule:**
```bash
aws events put-rule \
    --name fraud-detection-drift-check \
    --schedule-expression "cron(0 */6 * * ? *)"  # Every 6 hours
```

**Disable/Enable:**
```python
# In notebook or via AWS CLI
events = boto3.client('events')

# Disable
events.disable_rule(Name='fraud-detection-drift-check')

# Enable
events.enable_rule(Name='fraud-detection-drift-check')
```

### Troubleshooting

**No Emails Received:**
1. Check email confirmation - did you click the SNS subscription link?
2. Check spam folder
3. Verify SNS subscription status:
   ```bash
   aws sns list-subscriptions-by-topic \
       --topic-arn arn:aws:sns:us-east-1:{account}:fraud-detection-drift-alerts
   ```

**Lambda Errors:**
```bash
# Check CloudWatch logs
aws logs tail /aws/lambda/fraud-detection-drift-monitor --since 1h
```

Common issues:
- Insufficient IAM permissions
- Athena query timeout (increase Lambda timeout)
- Not enough inference samples (need ≥100 samples)

**No Drift Detected:**
- Insufficient recent data (Lambda looks at last 24 hours)
- No ground truth for model drift (need labeled predictions)
- Drift below thresholds (check logs for actual PSI values)

### Cleanup

To remove all drift monitoring resources:

```bash
# Delete EventBridge rule
aws events remove-targets --rule fraud-detection-drift-check --ids 1
aws events delete-rule --name fraud-detection-drift-check

# Delete Lambda function
aws lambda delete-function --function-name fraud-detection-drift-monitor

# Delete IAM role and policies
aws iam detach-role-policy \
    --role-name fraud-detection-drift-monitor-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role-policy \
    --role-name fraud-detection-drift-monitor-role \
    --policy-name SNSPublishPolicy
aws iam delete-role --role-name fraud-detection-drift-monitor-role

# Delete SNS topic
aws sns delete-topic --topic-arn arn:aws:sns:us-east-1:{account}:fraud-detection-drift-alerts
```

### Best Practices

1. **Set Appropriate Thresholds**: Start with defaults (PSI=0.2, 5% degradation), adjust based on your use case
2. **Monitor Regularly**: Review CloudWatch logs weekly to understand drift patterns
3. **Investigate Quickly**: When alerts fire, investigate within 24 hours
4. **Update Baseline**: After retraining, update `BASELINE_ROC_AUC` in Lambda environment
5. **Test Monthly**: Manually trigger Lambda to ensure monitoring is working
6. **Keep Email List Current**: Update SNS subscriptions when team members change
7. **Document Investigations**: Log drift investigations in MLflow or tracking system

## Athena Data Lake

### Iceberg Tables

**training_data:**
- 284,080 rows, 33 columns (30 features + 3 metadata)
- Partitioned by: None (static training data)
- Features: See "Feature Format" section below

**inference_responses:**
- All predictions from deployed endpoint
- Partitioned by: `year`, `month`, `day`
- Schema: inference_id, request_timestamp, prediction, probability_fraud, ground_truth, latency_ms, confidence_score, transaction_context
- The `ground_truth` column starts as NULL and gets populated by the ground truth update process

**ground_truth_updates:**
- Lightweight table for merging confirmed labels back into `inference_responses`
- Schema: inference_id, actual_fraud, confirmation_timestamp, confirmation_source, investigation_notes
- **Populated by:** Ground truth simulator (dev/test) or fraud investigation systems (production)
- **Consumed by:** `src/monitoring/update_ground_truth.py` which runs an Athena MERGE to update the `ground_truth` column in `inference_responses`
- JOINs on `inference_id` (not `transaction_id`)

**ground_truth (not used in current PoC):**
- Designed for a future **batch retraining** workflow
- Stores complete feature rows (all 33 columns) alongside confirmed fraud labels
- Unlike `ground_truth_updates` (which only patches `inference_responses`), this table holds full feature vectors for training a new model version
- Would be populated by a batch job that combines `inference_responses` + confirmed labels into training-ready rows
- Not needed for the current monitoring workflow — only relevant when building an automated retraining pipeline

**drifted_data:**
- Samples flagged for drift analysis
- Used for retraining prioritization

### Ground Truth Data Flow

```
inference_responses                ground_truth_updates
(predictions, ground_truth=NULL)   (inference_id + actual_fraud)
         │                                    │
         └──────── MERGE ON inference_id ──────┘
                          │
                          ▼
              inference_responses
              (ground_truth populated)
                          │
                          ▼
               Monitoring & Drift Detection
              (metrics where ground_truth IS NOT NULL)
```

**Key distinction:**
- `ground_truth_updates` → patches existing predictions (lightweight, used now)
- `ground_truth` → stores full feature rows for retraining (heavyweight, future use)

### Querying Athena

**Check inference logging:**
```sql
-- Recent predictions
SELECT
    inference_id,
    request_timestamp,
    prediction,
    probability_fraud,
    ground_truth,
    confidence_score
FROM fraud_detection.inference_responses
ORDER BY request_timestamp DESC
LIMIT 10;
```

**Ground truth coverage:**
```sql
-- Coverage by day
SELECT
    DATE(request_timestamp) as date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN ground_truth IS NOT NULL THEN 1 ELSE 0 END) as with_ground_truth,
    CAST(SUM(CASE WHEN ground_truth IS NOT NULL THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) * 100 as coverage_pct
FROM fraud_detection.inference_responses
GROUP BY DATE(request_timestamp)
ORDER BY date DESC;
```

**Model performance:**
```sql
-- Confusion matrix
SELECT
    prediction,
    ground_truth,
    COUNT(*) as count
FROM fraud_detection.inference_responses
WHERE ground_truth IS NOT NULL
GROUP BY prediction, ground_truth;
```

## Feature Format

### Training Features (30 total)

Your model expects these exact 30 features in JSON format:

```json
{
  "transaction_hour": 14,
  "transaction_day_of_week": 2,
  "transaction_amount": 149.62,
  "transaction_type_code": 1,
  "customer_age": 42,
  "customer_gender": 0,
  "customer_tenure_months": 36,
  "account_age_days": 1095,
  "distance_from_home_km": 5.2,
  "distance_from_last_transaction_km": 2.3,
  "time_since_last_transaction_min": 120,
  "online_transaction": 1,
  "international_transaction": 0,
  "high_risk_country": 0,
  "merchant_category_code": 5411,
  "merchant_reputation_score": 0.85,
  "chip_transaction": 1,
  "pin_used": 1,
  "card_present": 1,
  "cvv_match": 1,
  "address_verification_match": 1,
  "num_transactions_24h": 3,
  "num_transactions_7days": 12,
  "avg_transaction_amount_30days": 125.50,
  "max_transaction_amount_30days": 450.00,
  "velocity_score": 0.3,
  "recurring_transaction": 0,
  "previous_fraud_incidents": 0,
  "credit_limit": 5000.0,
  "available_credit_ratio": 0.75
}
```

**⚠️ Note:** Do NOT use V1, V2, V3 features - those are from a different dataset.

### Feature Categories

| Category | Count | Features |
|----------|-------|----------|
| Transaction | 4 | hour, day_of_week, amount, type_code |
| Customer Profile | 4 | age, gender, tenure_months, account_age_days |
| Geographic/Temporal | 4 | distance_from_home, distance_from_last, time_since_last, online |
| Risk Indicators | 2 | international, high_risk_country |
| Merchant | 2 | category_code, reputation_score |
| Payment Security | 5 | chip, pin, card_present, cvv_match, address_verification |
| Behavioral | 7 | num_transactions_24h/7d, avg/max_amount_30d, velocity, recurring, previous_fraud |
| Credit | 2 | credit_limit, available_credit_ratio |

## CLI Reference

### Setup & Infrastructure

```bash
# Upload CSV data to S3 (required before migration, files not in git)
python -m src.setup.upload_data_to_s3

# Create all infrastructure (S3, Athena DB, tables)
python main.py setup --migrate-data

# Just create infrastructure (no data migration)
python main.py setup
```

### Pipeline Operations

```bash
# Create pipeline
python main.py pipeline create --pipeline-name fraud-detection-pipeline

# Update existing pipeline
python main.py pipeline update --pipeline-name fraud-detection-pipeline

# Start execution
python main.py pipeline start --pipeline-name fraud-detection-pipeline [--wait]

# Get status
python main.py pipeline status --pipeline-name fraud-detection-pipeline

# List executions
python main.py pipeline list-executions --pipeline-name fraud-detection-pipeline

# Delete pipeline
python main.py pipeline delete --pipeline-name fraud-detection-pipeline
```

### Ground Truth & Monitoring

```bash
# Simulate ground truth (development)
python -m src.monitoring.simulate_ground_truth_from_athena --accuracy 0.85 [--limit 1000]

# Apply ground truth updates
python -m src.monitoring.update_ground_truth --mode batch
python -m src.monitoring.update_ground_truth --mode streaming --window-hours 24

# Monitor performance
python -m src.monitoring.monitor_model_performance --days 30 [--alert-threshold 0.80]
```

### IAM Roles

```bash
# Create/update SageMaker execution role
python -m src.setup.create_or_update_sagemaker_role

# Create Lambda execution role
python -m src.setup.create_lambda_role
```

## Troubleshooting

### Pipeline Execution Issues

**Issue:** Preprocessing fails with "Unable to verify/create output bucket"

```
InvalidRequestException: Unable to verify/create output bucket fraud-detection-data-lake
```

**Solution:**
1. Check `.env` has correct bucket name with account ID suffix:
   ```bash
   DATA_S3_BUCKET=fraud-detection-data-lake-YOUR_ACCOUNT
   ```
2. Verify bucket exists:
   ```bash
   aws s3 ls s3://fraud-detection-data-lake-YOUR_ACCOUNT/
   ```
3. Create bucket if missing:
   ```bash
   python main.py setup
   ```

---

**Issue:** Training fails with "Feature count mismatch"

```
ModelError: Expected 30 features, got 32
```

**Solution:**
- Model expects exactly 30 training features
- Check `TRAINING_FEATURES` list in `inference_monitoring.ipynb` Cell 4
- Do not include `transaction_id` or `transaction_timestamp` (metadata columns)

---

**Issue:** Pipeline fails with "FileNotFoundError: 'awswrangler'"

```
FileNotFoundError: [Errno 2] No such file or directory: 'awswrangler'
```

**Solution:**
1. Verify `inference_requirements.txt` exists:
   ```bash
   cat src/pipeline/inference_requirements.txt
   ```
2. Should contain:
   ```
   awswrangler>=3.0.0
   boto3>=1.34.0
   pandas>=2.0.0
   ```
3. Restart Jupyter kernel if running in notebook
4. Redeploy pipeline

---

### MLflow Integration Issues

**Issue:** Can't access MLflow UI

**Solution:**
1. Ensure you're in SageMaker Studio (not local environment)
2. Navigate: Left sidebar → Partner AI Apps → MLflow
3. Click on your MLflow app name
4. Opens authenticated session in new tab

---

**Issue:** MLflow tracking URI not working

```
MlflowException: Unable to connect to tracking server
```

**Solution:**
1. Check `.env` has ARN format (not HTTPS):
   ```bash
   # ✓ Correct (for SDK)
   MLFLOW_TRACKING_URI=arn:aws:sagemaker:us-east-1:123456:mlflow-app/app-ABC123

   # ✗ Wrong (for web UI only)
   MLFLOW_TRACKING_URI=https://abc123.mlflow-tracking.us-east-1.sagemaker.aws/
   ```
2. Verify IAM role has MLflow access:
   ```bash
   python -m src.setup.create_or_update_sagemaker_role
   ```

---

**Issue:** Training dependencies not installed

```
ModuleNotFoundError: No module named 'sagemaker_mlflow'
```

**Solution:**
1. Check training step has dependencies configured:
   ```python
   # In pipeline.py training step
   dependencies=['pipeline_steps/requirements_train.txt']
   ```
2. Verify `requirements_train.txt` contains:
   ```
   sagemaker-mlflow>=0.1.0
   mlflow>=2.17.0
   ```

---

### Inference & Monitoring Issues

**Issue:** No records in Athena after inference

**Solution:**
1. Check endpoint uses custom handler (not built-in XGBoost)
2. Verify CloudWatch logs show:
   ```
   ✓ Athena client initialized for inference logging
   ```
3. Make 50+ predictions to trigger buffer flush:
   ```python
   # Buffer flushes at 50 records
   # Cell 6 (5 predictions) + Cell 9 (50 predictions) = 55 total
   # This should automatically trigger the batch flush
   ```
4. Or wait 5 minutes for time-based flush

---

**Issue:** Timestamp precision error in Athena queries

```
NOT_SUPPORTED: Incorrect timestamp precision for timestamp(6)
```

**Solution:**
1. Athena Iceberg tables use `TIMESTAMP(3)` (milliseconds)
2. Cast aggregations explicitly:
   ```sql
   CAST(MIN(request_timestamp) AS TIMESTAMP(3))
   ```
3. All timestamp aggregations (MIN, MAX, AVG) must be cast
4. See `TIMESTAMP_FIX.md` for complete fix details

---

**Issue:** Ground truth simulation returns 0 records

```
No predictions found without ground truth
```

**Solution:**
1. Verify inference_responses table has data:
   ```sql
   SELECT COUNT(*) FROM fraud_detection.inference_responses;
   ```
2. Check predictions don't already have ground truth:
   ```sql
   SELECT COUNT(*) FROM fraud_detection.inference_responses
   WHERE ground_truth IS NULL;
   ```
3. Run inference tests first (notebook Cells 6-9)

---

### Infrastructure Issues

**Issue:** IAM role lacks permissions

```
AccessDeniedException: User is not authorized to perform: athena:StartQueryExecution
```

**Solution:**
```bash
# Update SageMaker role with all required permissions
python -m src.setup.create_or_update_sagemaker_role

# Grants access to:
# - S3 (read/write)
# - Athena (queries, Iceberg tables)
# - Glue (data catalog)
# - Lake Formation (table permissions)
# - MLflow (tracking server)
# - Lambda (for deployment step)
```

---

**Issue:** Corrupted Athena query results

```
SYNTAX_ERROR: line 1:8: Column 'cnt' cannot be resolved
```

**Solution:**
1. Corrupted metadata in S3
2. Clean up corrupted results:
   ```bash
   ./src/setup/cleanup_corrupted_athena_results.sh
   ```
3. Or manually delete:
   ```bash
   aws s3 rm s3://YOUR_BUCKET/athena-query-results/tables/ --recursive
   ```

---

### Performance Issues

**Issue:** Inference latency too high (>500ms)

**Expected:** 100-200ms P95 latency

**Solution:**
1. Check if cold start (first request ~2-5 seconds)
2. Verify endpoint instance type (ml.m5.xlarge recommended)
3. Check CloudWatch metrics for throttling
4. Consider increasing serverless concurrency
5. Background Athena logging should add <50ms

---

## Complete End-to-End Workflow

### 1. Setup Infrastructure

```bash
# One-time setup
python -m src.setup.upload_data_to_s3
python main.py setup --migrate-data
```

**Result:** S3 bucket, Athena database, Iceberg tables created, CSV data uploaded and migrated

---

### 2. Train & Deploy Model via Pipeline

**Option A: Jupyter Notebook (Recommended)**

```bash
# In SageMaker Studio, open:
notebooks/pipeline_execution.ipynb

# Run sequentially:
# - Cells 1-3: Setup and configuration
# - Cell 4: Create/update pipeline definition
# - Cell 5: Execute pipeline and wait (~25 minutes)
```

**Option B: CLI**

```bash
python main.py pipeline create --pipeline-name fraud-detection-pipeline
python main.py pipeline start --pipeline-name fraud-detection-pipeline --wait
```

**Result:**
- Model trained with XGBoost
- Evaluated (ROC-AUC, PR-AUC, confusion matrix)
- Registered in MLflow Model Registry
- Deployed to serverless endpoint with custom handler

---

### 3. Run Inference & Monitoring

**Open:** `notebooks/inference_monitoring.ipynb`

This notebook handles the complete inference and monitoring workflow:

#### 3a. Test Inference (Cells 1-9)

```bash
# In inference_monitoring.ipynb:
# Cell 1-2: Setup (load environment, initialize clients)
# Cell 4: Load test data and verify features
# Cell 5: Quick single test (optional)
# Cell 6: Single inference tests (5 samples)
# Cell 8: Generate playground samples (for SageMaker Studio testing)
# Cell 9: Bulk inference test (50 samples)
```

**Result:**
- Predictions returned with fraud probabilities
- Automatically logged to `inference_responses` table (custom handler)
- Latency metrics captured

**Verify automatic logging:**
```bash
# Cell 10: Check Athena for logged predictions
# Should show records after 100 predictions or 5 minutes
```

#### 3b. Simulate Ground Truth (Cells 20-22)

**For Development/Testing:**

```bash
# In inference_monitoring.ipynb:
# Cell 20: Run ground truth simulator
# - Simulates fraud investigation outcomes
# - Creates realistic confirmation delays
# - Configurable accuracy (default 85%)

# Or via CLI:
python -m src.monitoring.simulate_ground_truth_from_athena --accuracy 0.85
```

**For Production:**
Replace simulation with actual fraud investigation system that writes to `ground_truth_updates` table:
- Chargeback notifications
- Customer fraud reports
- Investigation team confirmations
- Merchant alerts

**Result:** Ground truth updates in `ground_truth_updates` table

#### 3c. Apply Ground Truth Updates (Cell 19 or CLI)

```bash
# In inference_monitoring.ipynb:
# Cell 19: Apply updates using notebook

# Or via CLI:
python -m src.monitoring.update_ground_truth --mode batch
```

**Result:** `inference_responses.ground_truth` column populated via MERGE

#### 3d. Verify Coverage (Cell 22)

```bash
# Cell 22: Check ground truth coverage
# Shows: total predictions, with/without ground truth, coverage %
```

**Expected:** 100% coverage (simulated) or partial coverage (production)

#### 3e. Monitor Performance & Detect Drift (Cells 27-40)

```bash
# Cell 27: Initialize performance monitor
# Cell 28: Generate performance report
# Cell 29: Display metrics (ROC-AUC, precision, recall, F1)
# Cell 32: Run Evidently data drift detection (DataDriftPreset)
# Cell 33: Display interactive Evidently data drift report
# Cell 35: Run Evidently model drift detection (ClassificationPreset)
# Cell 36: Display interactive Evidently classification report
# Cell 38: Log Evidently HTML reports and metrics to MLflow
# Cell 40: Check for alerts (performance degradation)
```

**Result:**
- Performance metrics calculated (where ground truth exists)
- Evidently data drift report generated (PSI, KS, distribution comparisons)
- Evidently classification report generated (ROC, PR, confusion matrix, F1)
- Interactive HTML reports logged to MLflow as artifacts
- Per-feature drift scores and classification metrics logged as MLflow metrics
- Alerts if performance degraded >5%

**Alternative: CLI monitoring (no charts):**
```bash
python -m src.monitoring.monitor_model_performance --days 30
```

---

### 4. Review Results in MLflow

```bash
# In SageMaker Studio:
# 1. Left sidebar → Partner AI Apps → MLflow
# 2. Select experiments:
#    - credit-card-fraud-detection-training (training runs)
#    - credit-card-fraud-detection-inference (monitoring runs)
# 3. View charts in Artifacts section
```

**Charts available:**
- Training: ROC curve, confusion matrix, feature importance
- Monitoring: Evidently interactive HTML reports (data drift dashboard, classification dashboard)

---

### 5. Retrain if Needed

```bash
# If drift detected or performance degraded:

# Option A: Via pipeline_execution.ipynb
# Cell 5: Re-run pipeline execution

# Option B: Via CLI
python main.py pipeline start --pipeline-name fraud-detection-pipeline --wait
```

**Result:**
- New model version trained with latest data
- Automatically registered in MLflow
- Deployed to same endpoint (replaces old model)

---

## Summary: Notebook Workflow

The entire PoC is driven by **three notebooks** in SageMaker Studio. Each notebook cell maps to a CLI command (see [Next Steps](#next-steps) for CI/CD equivalents).

| Notebook | Purpose | Key Cells |
|----------|---------|-----------|
| **`pipeline_execution.ipynb`** | Training, evaluation, model registration, endpoint deployment | Cells 1-5 |
| **`inference_monitoring.ipynb`** | Inference testing, ground truth, drift detection, CloudWatch alarms | Cells 1-40 |
| **`inference_monitoring_with_pipeline.ipynb`** | Pipeline-based automated monitoring: creates a SageMaker Pipeline (`fraud-detection-monitoring-pipeline`) with steps for ground truth simulation, drift computation, MLflow logging, Athena writes, threshold checks, and alarm creation | Cells 1-end |

**Workflow:**

1. **Train & Deploy:** `pipeline_execution.ipynb` → Cells 1-5 (creates pipeline, trains model, deploys endpoint)
2. **Test Inference:** `inference_monitoring.ipynb` → Cells 1-9 (single + bulk predictions, auto-logged to Athena)
3. **Simulate Ground Truth:** `inference_monitoring.ipynb` → Cells 19-24 (generate and apply ground truth labels)
4. **Monitor & Detect Drift:** `inference_monitoring.ipynb` → Cells 26-39 (performance metrics, data drift, model drift, MLflow logging)
5. **CloudWatch Alarms:** `inference_monitoring.ipynb` → Cell 40 (publish metrics, create alarms & dashboard)
6. **MLflow Review:** SageMaker Studio → Partner AI Apps → MLflow
7. **Retrain:** `pipeline_execution.ipynb` → Cell 5 (re-execute pipeline)

**Alternative: Pipeline-Based Monitoring**
`inference_monitoring_with_pipeline.ipynb` wraps steps 3-5 into a single SageMaker Pipeline (`fraud-detection-monitoring-pipeline`) with automated steps: SimulateGroundTruth → ComputeDrift → LogToMLflow → WriteToAthena → CheckThresholds → CreateAlarms. Use this for scheduled, hands-off monitoring.

**All in SageMaker Studio — no CLI needed for development workflow.**

## Cost Optimization

**Training:**
- Use Spot instances for training: ~70% cost savings
- Training completes in 10-15 min, low spot interruption risk
- Configure in pipeline: `instance_type="ml.m5.xlarge"`, `use_spot_instances=True`

**Inference:**
- Serverless inference: Pay per invocation, no idle costs
- Cold start: 10-30 seconds (acceptable for fraud detection)
- Alternative: Provision 1 instance for high-volume scenarios

**Storage:**
- Athena Iceberg tables: Pay per query, not storage
- S3 Intelligent-Tiering for model artifacts
- Compress old inference logs to Glacier after 90 days

**Estimated Monthly Costs (1000 predictions/day):**
- Training: ~$5-10 (spot instances, 2-3 retrainings/month)
- Inference: ~$20-30 (serverless)
- Storage: ~$5-10 (S3 + Athena)
- MLflow: Included in SageMaker AI Notebooks
- **Total: ~$30-50/month**

## Model Versioning & Lifecycle

### Version Stages

Models progress through stages in MLflow Model Registry:

1. **None** - Newly registered, not validated
2. **Staging** - Undergoing validation/testing
3. **Production** - Serving live traffic
4. **Archived** - Retired

### Promoting Models

```python
# In MLflow UI or via SDK
from mlflow import MlflowClient

client = MlflowClient()

# Promote to staging
client.transition_model_version_stage(
    name="xgboost-fraud-detector",
    version=3,
    stage="Staging"
)

# After validation, promote to production
client.transition_model_version_stage(
    name="xgboost-fraud-detector",
    version=3,
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="xgboost-fraud-detector",
    version=2,
    stage="Archived"
)
```

### Version Metadata

Each version includes:
- **Metrics:** ROC-AUC, PR-AUC, precision, recall, F1
- **Parameters:** max_depth, learning_rate, scale_pos_weight, etc.
- **Tags:** `training_date`, `pipeline_execution_id`, `deployed_endpoint`
- **Artifacts:** model.xgb, feature_metadata.json, preprocessing config

### Rollback Strategy

```bash
# If new model performs poorly:

# 1. Check recent performance
python -m src.monitoring.monitor_model_performance --days 7

# 2. If degraded, rollback to previous version in MLflow UI:
#    - Mark current version as "Archived"
#    - Mark previous version as "Production"

# 3. Redeploy with previous model version
#    Update pipeline to use previous run_id
python main.py pipeline start --pipeline-name fraud-detection-pipeline
```

## Next Steps

### CI/CD: From Notebooks to Automation

Each notebook cell has a direct CLI equivalent. For CI/CD pipelines, replace notebook interactions with these commands:

| Phase | Notebook Cell | CLI Command |
|-------|--------------|-------------|
| Infrastructure | `pipeline_execution.ipynb` Cell 3 | `python main.py setup --migrate-data` |
| SQS + Lambda | `pipeline_execution.ipynb` Cell 3 | `python main.py setup-logging` |
| Create Pipeline | `pipeline_execution.ipynb` Cell 4 | `python main.py pipeline create --pipeline-name fraud-detection-pipeline` |
| Train & Deploy | `pipeline_execution.ipynb` Cell 5 | `python main.py pipeline start --pipeline-name fraud-detection-pipeline --wait` |
| Test Inference | `inference_monitoring.ipynb` Cells 6-9 | `python main.py test --endpoint-name fraud-detector-endpoint --num-samples 100` |
| Simulate Ground Truth | `inference_monitoring.ipynb` Cell 19 | `python -m src.monitoring.simulate_ground_truth_from_athena --accuracy 0.85` |
| Apply Ground Truth | `inference_monitoring.ipynb` Cell 24 | `python -m src.monitoring.update_ground_truth --mode batch` |
| Monitor Drift | `inference_monitoring.ipynb` Cells 26-39 | `python -m src.monitoring.monitor_model_performance --days 30` |

### Scheduled Monitoring

Automate drift checks with EventBridge + Lambda:

1. **EventBridge Rule:** Schedule `monitor_model_performance.py` daily/hourly via a Lambda function
2. **Lambda wrapper:** Package the monitoring script as a Lambda that queries Athena and publishes CloudWatch metrics
3. **Trigger retraining:** If drift exceeds thresholds, start the SageMaker Pipeline automatically

### SNS Notifications

Connect CloudWatch alarms (created in Cell 40) to email or Slack:

```bash
# 1. Create SNS topic
aws sns create-topic --name FraudDetection-DriftAlerts

# 2. Subscribe email
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:YOUR_ACCOUNT:FraudDetection-DriftAlerts \
  --protocol email \
  --notification-endpoint your-team@example.com

# 3. Update alarms to notify (repeat for each alarm)
aws cloudwatch put-metric-alarm \
  --alarm-name FraudDetection-ModelDrift-ROC-AUC \
  --alarm-actions arn:aws:sns:us-east-1:YOUR_ACCOUNT:FraudDetection-DriftAlerts \
  # ... (keep existing alarm configuration)
```

For Slack, use an SNS → Lambda → Slack webhook integration or AWS Chatbot.

## License

MIT License - See LICENSE file for details

---

**Documentation Version:** 3.0
**Last Updated:** 2026-03-18
**Pipeline Version:** 1.0
**Model Version:** xgboost-fraud-detector v1+

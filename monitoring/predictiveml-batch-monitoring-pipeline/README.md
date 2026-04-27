# Predictive ML Batch Monitoring with Amazon SageMaker AI

This folder contains notebook-based examples for monitoring predictive ML batch workloads with Amazon SageMaker AI, Evidently, and SageMaker managed MLflow.

The implementation is split into three notebooks so each monitoring concern is clear:

1. `predictive_ml_experimentation_data_model_monitoring_evidently.ipynb`
   - Interactive learning notebook.
   - Trains a sample model, runs batch inference, explores data drift, data quality, and model quality.

2. `batch_monitoring_pipeline.ipynb`
   - Automation notebook for data drift and data quality only.
   - Creates a SageMaker Pipeline with one Processing step.
   - Does not run model quality, batch transform, or prediction evaluation.

3. `model_quality_monitoring_example.ipynb`
   - Separate model quality example.
   - Runs when predictions and ground truth labels are available.
   - Logs classification metrics and an Evidently model quality report to MLflow.

## Why Model Quality Is Separate

Data drift and data quality can be checked as soon as a new input data file arrives. Model quality needs predictions and ground truth labels, and ground truth often arrives later.

Keeping model quality out of the second automation solution makes the drift pipeline easier to reuse:

- The data drift pipeline only needs a baseline CSV and a current CSV.
- The model quality pipeline only runs when labels are ready.
- Customers can schedule the two workflows independently.
- Failures in delayed-label processing do not block daily data drift checks.

## Recommended Input File Pattern

The recommended pattern is to pass explicit S3 file locations into the monitoring processing job:

```python
baseline_data_s3_uri = "s3://my-bucket/monitoring/baseline/baseline.csv"
current_data_s3_uri = "s3://my-bucket/monitoring/current/2026-04-18.csv"
```

This is usually better for customer plug-and-play than relying only on "latest file" pickup because the caller controls exactly which file is monitored.

The data drift processor still supports latest-file pickup for simple scheduled demos:

```python
use_latest_file_pickup = "true"
baseline_data_s3_uri = "s3://my-bucket/monitoring/baseline/"
current_data_s3_uri = "s3://my-bucket/monitoring/current/"
```

When `use_latest_file_pickup` is true, the processor searches each prefix and selects the newest CSV by S3 `LastModified` time.

Use explicit file paths when:

- Several files can land in the same prefix close together.
- An upstream pipeline already knows the exact file to monitor.
- You need auditability and reproducibility for each monitoring run.

Use latest-file pickup when:

- You are building a simple demo.
- A landing prefix contains only one new file per schedule interval.
- You accept the risk that object arrival order controls the selected file.

## Files

| Path | Purpose |
|---|---|
| `batch_monitoring_pipeline.ipynb` | Builds the automated data drift and data quality SageMaker Pipeline |
| `model_quality_monitoring_example.ipynb` | Builds the optional model quality SageMaker Pipeline |
| `predictive_ml_experimentation_data_model_monitoring_evidently.ipynb` | Interactive experimentation notebook |
| `scripts/monitoring_processor.py` | Processing script for data drift and data quality only |
| `scripts/model_quality_processor.py` | Processing script for binary classification model quality |
| `scripts/requirements.txt` | Python package list used by the examples |

## Notebook 2: Data Drift and Data Quality Automation

`batch_monitoring_pipeline.ipynb` creates:

- A SageMaker Pipeline named `data-drift-quality-monitoring-pipeline`
- One SageMaker Processing step named `DataDriftAndQualityMonitoring`
- An optional SNS topic for drift alerts
- An optional EventBridge Scheduler schedule
- MLflow runs containing metrics, parameters, and Evidently report artifacts

The processing script expects:

- Baseline CSV with headers
- Current CSV with the same headers, in the same column order

It logs:

- `DriftedColumnsCount.count`
- `DriftedColumnsCount.share`
- `ValueDrift:<feature>` for features that cross the drift threshold
- baseline and current row counts
- baseline and current missing-cell counts
- baseline and current duplicate-row counts
- Evidently data drift HTML and JSON reports
- Evidently data quality HTML and JSON reports
- `monitoring_summary.json`

## Notebook 3: Model Quality Example

`model_quality_monitoring_example.ipynb` creates:

- A SageMaker Pipeline named `model-quality-monitoring-pipeline`
- One SageMaker Processing step named `ModelQualityMonitoring`
- MLflow runs containing classification metrics and Evidently model quality artifacts

The model quality processor expects:

- Predictions CSV with a `prediction` column
- Optional prediction probability column named `prediction_proba`
- Ground truth CSV with a `target` column

The rows must line up. Row 1 in the predictions file must refer to the same record as row 1 in the ground truth file. In production, join predictions and labels by a stable record ID before running model quality.

It logs:

- `Accuracy`
- `Precision`
- `Recall`
- `F1Score`
- `ROC_AUC` when a probability column is available
- Evidently model quality HTML and JSON reports
- `model_quality_summary.json`

## Prerequisites

- Amazon SageMaker Studio or a notebook environment with AWS credentials
- SageMaker execution role with access to SageMaker, S3, SNS, IAM, EventBridge Scheduler, and SageMaker managed MLflow
- SageMaker managed MLflow app, for example `DefaultMLFlowApp`
- S3 bucket for input CSV files and monitoring artifacts
- Python 3.10 or later

The notebooks install the Python packages they need. The processing scripts install runtime packages inside the SageMaker Processing container so the sample can run without a custom image.

For production, use a custom processing image with pinned dependency versions. That avoids runtime package installation and makes processing jobs start faster.

## Basic Workflow

1. Run the experimentation notebook if you want to generate demo model artifacts and local sample files.
2. Upload or point Notebook 2 at a baseline CSV and current CSV.
3. Run Notebook 2 to create and test the data drift and data quality pipeline.
4. Optionally create the EventBridge schedule in Notebook 2.
5. When predictions and ground truth labels are available, point Notebook 3 at those files.
6. Run Notebook 3 to create and test the model quality pipeline.
7. Review all runs and artifacts in the configured SageMaker managed MLflow experiment.

## Cleanup

Each automation notebook includes cleanup cells for resources it creates.

Notebook 2 cleanup removes:

- EventBridge Scheduler schedule
- SageMaker Pipeline
- SNS topic
- Scheduler IAM role

Notebook 3 cleanup removes:

- SageMaker Pipeline

MLflow runs and S3 artifacts are intentionally preserved unless you delete them separately.

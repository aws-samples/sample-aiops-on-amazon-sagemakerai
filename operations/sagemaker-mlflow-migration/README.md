# SageMaker MLflow Migration

This guide covers the migration of MLflow data between different versions using SageMaker MLflow services and the [mlflow-export-import tool](https://github.com/mlflow/mlflow-export-import/tree/master).

## Migration Scenarios

This guide covers two separate migration paths:

1. **[MLflow v2.16 to v3.4 Migration](#mlflow-v216-to-v34-migration)** - Complete migration workflow from SageMaker MLflow v2.16 Tracking Server to v3.4 App
2. **[MLflow v3.0 to v3.4 Migration](#mlflow-v30-to-v34-migration)** - Complete migration workflow from SageMaker MLflow v3.0 Tracking Server to v3.4 App

Each migration path includes data setup, export, and import processes with dedicated notebooks.

## Prerequisites

### Required Infrastructure

- **Source MLflow Tracking Servers** (2 instances)
  - MLflow v2.16 tracking server.
  - MLflow v3.0 tracking server.
  - Both configured with S3 artifact storage
  - [Create tracking server guide](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-create-tracking-server.html)

- **Target MLflow Apps** (2 instances)
  - MLflow v3.4 app for receiving v2.16 data
  - MLflow v3.4 app for receiving v3.0 data
  - Sufficient storage capacity for migrated data
  - [Create MLflow app guide](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-app-setup-create-app.html)

### AWS Permissions

- SageMaker and SageMaker MLflow full access for creating/managing MLflow resources and granting MLflow api permissions
- S3 read/write permissions for artifact storage buckets

### Performance Requirements

- **Recommended compute**:
  - JupyterLab: `ml.m5.xlarge` or higher for migration scripts
  - Local development: 4+ CPU cores for optimal performance
  - Use higher core instances for better performance with `--use-threads` option
- **Network**: Stable internet connection for data transfer
- **Local Storage**: Sufficient disk space to export and store MLflow data locally

## MLflow v2.16 to v3.4 Migration

### Step 1: Setup MLflow v2.16 Data
- **Notebook**: [mlflow-v2.16-data-setup.ipynb](./mlflow-v2.16-to-v3.4/mlflow-v2.16-data-setup.ipynb)
- **Purpose**: Creates sample data in MLflow v2.16 tracking server
- **Objects Created**: Experiments, runs, traces, metrics, params, artifacts, and registered models

### Step 2: Export and Import MLflow v2.16 Data
- **Notebook**: [mlflow-v2.16-to-v3.4-migration.ipynb](./mlflow-v2.16-to-v3.4/mlflow-v2.16-to-v3.4-migration.ipynb)
- **Purpose**: Exports data from MLflow v2.16 tracking server to local machine, then imports into MLflow v3.4 app
- **Objects Migrated**: Experiments, runs, traces, metrics, params, artifacts, and registered models

## MLflow v3.0 to v3.4 Migration

### Step 1: Setup MLflow v3.0 Data
- **Notebook**: [mlflow-v3.0-data-setup.ipynb](./mlflow-v3.0-to-v3.4/mlflow-v3.0-data-setup.ipynb)
- **Purpose**: Creates sample data in MLflow v3.0 tracking server
- **Objects Created**: Experiments, runs, traces, metrics, params, artifacts, registered models, prompts, and logged models

### Step 2: Export and Import MLflow v3.0 Data
- **Notebook**: [mlflow-v3.0-to-v3.4-migration.ipynb](./mlflow-v3.0-to-v3.4/mlflow-v3.0-to-v3.4-migration.ipynb)
- **Purpose**: Exports data from MLflow v3.0 tracking server to local machine, then imports into MLflow v3.4 app
- **Objects Migrated**: Experiments, runs, traces, metrics, params, artifacts, registered models, prompts, and logged models
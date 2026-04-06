# Automated Drift and Trend Monitoring for ML Models on Amazon SageMaker

> **This project has moved.** For the complete end-to-end solution, refer to [https://github.com/aws-samples/sample-mlops-bestpractices](https://github.com/aws-samples/sample-mlops-bestpractices)

## Overview

An open-source MLOps system that automates inference monitoring and drift detection for ML models in production. Built on Amazon SageMaker, MLflow, and Evidently AI, it addresses the common gap between well-managed training pipelines and the lack of production monitoring.

## What It Does

- **Trains** an XGBoost fraud detection model via SageMaker Pipelines with automated preprocessing, evaluation, and deployment
- **Logs** every prediction to an Athena Iceberg data lake with zero-latency async writes (SQS + Lambda)
- **Detects drift** automatically using EventBridge-triggered Lambda functions running Evidently AI reports (data drift via PSI/KS, model drift via classification metrics)
- **Alerts** via SNS when drift exceeds configurable thresholds defined in a central `config.yaml`
- **Tracks** all experiments, metrics, and artifacts in SageMaker Managed MLflow
- **Visualizes** drift trends, inference patterns, and model performance in an Amazon QuickSight governance dashboard

## Architecture

The solution consists of three integrated pipelines:

1. **Training Pipeline** — SageMaker Pipelines orchestrating data ingestion, PySpark processing, XGBoost training, evaluation, and serverless endpoint deployment
2. **Inference Monitoring** — Automated daily drift checks (EventBridge + Lambda + Evidently AI), ground truth integration with delayed confirmations, and SNS alerting
3. **Governance Dashboard** — QuickSight dashboard with direct query to Athena for real-time drift analysis and model performance visibility

## Key Benefits

- ~$30/month vs $200+/month for managed alternatives
- Portable across clouds using open-source SDKs (MLflow, Evidently, Pandas)
- Handles real-world challenges: delayed ground truth, concept drift, multi-feature drift analysis
- Fully automated with no manual intervention required

## Getting Started

The complete solution with guided Jupyter notebooks, deployment scripts, and detailed documentation is available at:

**[https://github.com/aws-samples/sample-mlops-bestpractices](https://github.com/aws-samples/sample-mlops-bestpractices)**

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

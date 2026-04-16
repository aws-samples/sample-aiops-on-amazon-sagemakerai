# Monitoring Solutions

This folder contains four comprehensive monitoring solutions for different ML/AI workloads on Amazon SageMaker.

## Available Solutions

### 1. [Predictive ML Batch Monitoring Pipeline](./predictiveml-batch-monitoring-pipeline/)
- **Monitoring solution** for implementing production-ready batch ML monitoring on Amazon SageMakerAI: experimentation notebook for learning fundamentals, followed by automated pipeline for operations.
- **Data Drift Detection**: Statistical comparison of current vs. baseline data distributions using Evidently AI's DataDriftPreset with automatic threshold-based alerting.
- **Model Quality Tracking**: Binary classification performance metrics (Accuracy, Precision, Recall, F1, AUC) with Evidently's ClassificationPreset for model degradation monitoring.
- **Automated SageMaker Pipeline**: Orchestrates batch inference and monitoring workflow with scheduled execution via EventBridge (daily/weekly/monthly).
- **Unified MLflow Integration**: Single experiment tracking server for both training and monitoring runs, enabling complete model lineage and drift trend analysis.
- **Email Alerting**: SNS notifications when drift exceeds configurable thresholds, including detailed drift summary and MLflow run links.
- **Interactive Reports**: HTML/JSON Evidently reports saved to S3 and MLflow artifacts for visual exploration and programmatic access.
- **Batch Transform Integration**: Cost-effective inference without always-on endpoints, with predictions feeding directly into monitoring pipeline.

### 2. [Real-Time Inference Monitoring with QuickSight dashboards](./sagemaker-automated-drift-and-trend-monitoring/)
Production-grade end-to-end solution for real-time endpoint monitoring with Athena data lake integration. Features ground truth capture, PSI-based drift detection, and automated retraining triggers. Best for always-on fraud detection or similar production workloads.

### 3. [LLM Inference Monitoring](./sagemaker-endpoint-llm-monitoring/)
- Automated serverless infrastructure for monitoring SageMaker LLM endpoint inferences using AWS CDK, MLflow traces, and MLflow GenAI evaluations.
- Event-driven architecture using S3 Data Capture, EventBridge, Step Functions, and Lambda for real-time inference monitoring.
- Implements MLflow GenAI evaluations (Safety, Relevance, Fluency, Guidelines, Coherence) using Amazon Bedrock models for comprehensive quality assessment.
- Supports multiple deployment environments (dev, staging, prod) with unique resource naming via configurable stack prefixes.
- Includes complete CDK infrastructure-as-code with Docker-based Lambda functions, comprehensive documentation, and cost optimization guidance.

### 4. [SageMaker Resource Monitoring with Grafana](./resource-monitoring-grafana/)
- **Infrastructure observability** dashboards for SageMaker inference endpoints using Enhanced Container Metrics for per-GPU, per-container, and per-inference-component visibility at 10-second granularity.
- **Cost Attribution**: Real-time hourly cost tracking based on GPU allocation and per-model resource usage for multi-model endpoints.
- **Resource Utilization**: GPU compute, GPU memory, CPU, and memory utilization metrics with threshold indicators and cluster-level overview.
- **Automated Setup**: Single Jupyter notebook deploys Grafana workspace, IAM roles, CloudWatch data sources, and all dashboard panels programmatically.
- **Production Monitoring**: Persistent, auto-refreshing dashboards for capacity planning, cost optimization, and performance tuning of GPU-accelerated inference workloads.

---

## Choosing the Right Monitoring Solution

| Criteria | Batch Monitoring | Real-Time Inference | LLM Monitoring | Resource Monitoring |
|----------|------------------|---------------------|----------------|---------------------|
| **Use Case** | Periodic batch predictions | Always-on endpoint inference | LLM endpoint evaluation | Infrastructure & cost tracking |
| **Inference Type** | Batch Transform | Real-time endpoint | Real-time endpoint | Real-time endpoint |
| **Deployment** | Educational (2 notebooks) | Production-ready (full pipeline) | Production CDK | Grafana dashboard |
| **Data Storage** | S3 CSV files | Athena data lake (Iceberg) | S3 Data Capture | CloudWatch Metrics |
| **Monitoring Focus** | Data drift + model quality | Drift + performance + ground truth | GenAI evaluations | GPU/CPU/memory + cost |
| **Metrics Granularity** | Per-batch run | Per-inference request | Per-inference request | 10-second intervals |
| **Alerting** | SNS email | SNS email + MLflow | Step Functions | Grafana alerts |
| **Best For** | Learning ML monitoring | Production fraud detection | LLM safety/quality | Multi-model cost optimization |
| **Setup Time** | 30-45 minutes | 2-3 hours | 1-2 hours | 15-30 minutes |
| **Infrastructure** | SageMaker Pipeline | SageMaker + Lambda + Athena | CDK Serverless | Managed Grafana |

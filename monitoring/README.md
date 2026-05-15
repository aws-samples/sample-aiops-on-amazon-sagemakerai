# Monitoring Solutions

This folder contains six comprehensive monitoring solutions for different ML/AI workloads on Amazon SageMaker AI.

## Available Solutions

### 1. [Predictive ML Batch Monitoring Pipeline](./predictiveml-batch-monitoring-pipeline/)
- **Monitoring solution** for implementing production-ready batch ML monitoring on Amazon SageMaker AI: experimentation notebook for learning fundamentals, followed by automated pipeline for operations.
- **Data Drift Detection**: Statistical comparison of current vs. baseline data distributions using Evidently AI's DataDriftPreset with automatic threshold-based alerting.
- **Model Quality Tracking**: Binary classification performance metrics (Accuracy, Precision, Recall, F1, AUC) with Evidently's ClassificationPreset for model degradation monitoring.
- **Automated SageMaker Pipeline**: Orchestrates batch inference and monitoring workflow with scheduled execution via EventBridge (daily/weekly/monthly).
- **Unified MLflow Integration**: Single experiment tracking server for both training and monitoring runs, enabling complete model lineage and drift trend analysis.
- **Email Alerting**: SNS notifications when drift exceeds configurable thresholds, including detailed drift summary and MLflow run links.
- **Interactive Reports**: HTML/JSON Evidently reports saved to S3 and MLflow artifacts for visual exploration and programmatic access.
- **Batch Transform Integration**: Cost-effective inference without always-on endpoints, with predictions feeding directly into monitoring pipeline.

### 2. [Predictive ML Endpoint Monitoring](./predictiveml-endpoint-monitoring/)
- **Real-time endpoint monitoring** using Evidently AI for data drift detection and model quality evaluation on SageMaker AI inference endpoints with data capture.
- **End-to-End Workflow**: Trains an XGBoost model with ModelTrainer, deploys to a real-time endpoint with data capture, and runs Evidently reports against baseline and captured data.
- **MLflow Integration**: All Evidently HTML reports and numeric metrics are logged to a SageMaker AI MLflow App for tracking, comparison, and lineage.
- **CDK Lambda Deployment**: Production-ready CDK project packages monitoring into two Docker-based Lambda functions (data drift and model quality) triggered by S3 events via EventBridge.
- **Model Quality Alerting**: Configurable threshold-based SNS alerts when F1, Accuracy, or ROC AUC drop below defined levels.
- **Interactive Experimentation**: Single notebook walks through training, deployment, drift detection, and model quality evaluation before scaling with CDK.

### 3. [Real-Time Inference Monitoring with QuickSight Dashboards](./sagemaker-automated-drift-and-trend-monitoring/)
- Production-grade end-to-end solution for real-time endpoint monitoring with Athena Iceberg data lake integration.
- Automated daily drift checks using EventBridge + Lambda + Evidently AI (PSI/KS statistical tests).
- Ground truth capture with delayed confirmation handling for real-world label latency.
- QuickSight governance dashboard with direct query to Athena for drift trend analysis and model performance visibility.
- SNS alerting when drift exceeds configurable thresholds defined in a central `config.yaml`.
- Cost-optimized serverless architecture (~$30/month vs $200+/month for managed alternatives).
- Full solution available at [sample-mlops-bestpractices](https://github.com/aws-samples/sample-mlops-bestpractices).

### 4. [LLM Inference Monitoring](./sagemaker-endpoint-llm-monitoring/)
- Automated serverless infrastructure for monitoring SageMaker LLM endpoint inferences using AWS CDK, MLflow traces, and MLflow GenAI evaluations.
- Event-driven architecture using S3 Data Capture, EventBridge, Step Functions, and Lambda for real-time inference monitoring.
- Implements MLflow GenAI evaluations (Safety, Relevance, Fluency, Guidelines, Coherence) using Amazon Bedrock models for comprehensive quality assessment.
- Supports multiple deployment environments (dev, staging, prod) with unique resource naming via configurable stack prefixes.
- Includes complete CDK infrastructure-as-code with Docker-based Lambda functions, comprehensive documentation, and cost optimization guidance.

### 5. [SageMaker Resource Observability with Grafana](./resource-monitoring-grafana/)
- **Infrastructure observability** dashboards for SageMaker inference endpoints using Enhanced Container Metrics for per-GPU, per-container, and per-inference-component visibility at 10-second granularity.
- **Cost Attribution**: Real-time hourly cost tracking based on GPU allocation and per-model resource usage for multi-model endpoints.
- **Resource Utilization**: GPU compute, GPU memory, CPU, and memory utilization metrics with threshold indicators and cluster-level overview.
- **Automated Setup**: Single Jupyter notebook deploys Grafana workspace, IAM roles, CloudWatch data sources, and all dashboard panels programmatically.
- **Production Monitoring**: Persistent, auto-refreshing dashboards for capacity planning, cost optimization, and performance tuning of GPU-accelerated inference workloads.

### 6. [LLM Quality Observability with Grafana](./quality-monitoring-with-grafana/)
- **LLM output quality dashboards** for SageMaker inference components using MLflow GenAI Evaluations with Bedrock Claude as an LLM-as-judge scorer.
- **Quality Metrics**: Safety, relevance, professional tone, and composite quality scores published as custom CloudWatch metrics per inference component.
- **Automated Alerting**: Amazon Managed Grafana unified alerting with threshold-based rules (low safety, low relevance, low composite quality) that fire per inference component.
- **SNS Notifications**: Optional email alerting via SNS when quality scores breach thresholds, with automated topic creation and Grafana contact point configuration.
- **Evaluation Latency Tracking**: End-to-end latency monitoring of the quality evaluation pipeline (inference + LLM-as-judge scoring).
- **Extends Resource Monitoring**: Builds on the Resource Monitoring with Grafana solution, adding LLM output quality metrics alongside infrastructure observability.

---

## Choosing the Right Monitoring Solution

| Criteria | Batch Monitoring | Endpoint Monitoring | Real-Time Inference | LLM Inference Monitoring | Resource Monitoring | LLM Quality Monitoring |
|----------|------------------|---------------------|---------------------|--------------------------|---------------------|------------------------|
| **Use Case** | Periodic batch predictions | Real-time endpoint drift | Always-on endpoint inference | LLM endpoint evaluation | Infrastructure & cost tracking | LLM output quality scoring |
| **Inference Type** | Batch Transform | Real-time endpoint | Real-time endpoint | Real-time endpoint | Real-time endpoint | Real-time endpoint (IC) |
| **Deployment** | Educational (2 notebooks) | Notebook + CDK Lambda | Production-ready (full pipeline) | Production CDK | Grafana dashboard | Grafana dashboard |
| **Data Storage** | S3 CSV files | S3 Data Capture | Athena data lake (Iceberg) | S3 Data Capture | CloudWatch Metrics | CloudWatch Custom Metrics |
| **Monitoring Focus** | Data drift + model quality | Data drift + model quality | Drift + performance + ground truth | GenAI evaluations | GPU/CPU/memory + cost | Safety, relevance, tone |
| **Metrics Granularity** | Per-batch run | Per-event (S3 trigger) | Per-inference request | Per-inference request | 10-second intervals | Per-inference evaluation |
| **Alerting** | SNS email | SNS email | SNS email + MLflow | Step Functions | Grafana alerts | Grafana alerts + SNS |
| **Best For** | Learning ML monitoring | Real-time endpoint with CDK scale | Production fraud detection | LLM safety/quality (event-driven) | Multi-model cost optimization | LLM quality regression detection |
| **Setup Time** | 30-45 minutes | 45-60 minutes | 2-3 hours | 1-2 hours | 15-30 minutes | 15-30 minutes |
| **Infrastructure** | SageMaker Pipeline | CDK Lambda + EventBridge | SageMaker + Lambda + Athena | CDK Serverless | Managed Grafana | Managed Grafana + CloudWatch |

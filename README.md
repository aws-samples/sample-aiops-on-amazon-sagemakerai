## AWS ModelOps on Amazon SageMakerAI 

This repository contains a collection of examples and resources to help you operationalize Generative AI (GenAI) and Machine learning workloads on SageMakerAI.

### Overview
The AWS ModelOps covering GenAIOps and MLOps patterns involing SageMakerAI resources like SageMaker managed MLflow, SageMaker pipelines and include all other AWS GenAI related features like Amazon Bedrock. This repository provides a set of sample notebooks, scripts, and configurations to help you explore different aspects of the ModelOps.

### Repository Structure
```
.
├── workshops/                                      # Technical workshops
│   └── aiops-with-sagemaker-mlflow/                # ModelOps with SageMakerAI Managed MLflow
├── operations/                                     # Operational guides
│   └── sagemaker-mlflow-migration/                 # MLflow data migration guide
├── monitoring/                                     # All monitoring and observability solutions
│   └── ...
├── examples/                                       # Integration examples
│   └── sagemaker-mlflow-agentcore-runtime/         # MLflow observability for Bedrock Agents
├── LICENSE                                         # MIT-0 License
└── README.md                                       # This file
```

### Workshops
#### [ModelOps with SageMakerAI Managed MLflow](./workshops/aiops-with-sagemaker-mlflow/)
- Specialized technical workshop is designed for ML administrators, platform engineers, data scientists, ML engineers and DevOps engineers. seeking hands-on skills in managing and utilizing Amazon SageMakerAI managed MLflow. 
- The course delves into essential topics such as administrating SageMaker managed MLFlow and example workloads. - Participants will also gain deep insights into MLflow constructs like experiments, models, prompts, SageMaker-MLflow integration, and tracing. 
- Advanced segments will cover workloads like genai agents, and LLM Model training.

### Operations
#### [SageMaker MLflow Migration](./operations/sagemaker-mlflow-migration/)
- Comprehensive guide for migrating MLflow data between different versions using SageMaker MLflow services.
- Covers migration scenarios from MLflow v2.16 and v3.0 tracking servers to MLflow v3.4 apps.
- Includes step-by-step notebooks for data setup, export, and import processes with sample MLflow objects (experiments, runs, traces, registered models, and version-specific features like prompts and logged models).

### Monitoring
> Various monitoring solutions for SageMaker AI. See the [monitoring folder](./monitoring/) for details.

#### [Predictive ML Batch Monitoring Pipeline with Evidently AI and MLflow](./monitoring/predictiveml-batch-monitoring-pipeline/)
Educational three-notebook solution for batch ML monitoring: an experimentation notebook, an automated pipeline for data drift and data quality with explicit S3 input file locations, and a separate example notebook for model quality when predictions and ground truth labels are available.

#### [Predictive ML Batch Monitoring Pipeline](./monitoring/predictiveml-batch-monitoring-pipeline/)
Batch ML monitoring with Evidently AI drift detection, model quality tracking, and SageMaker Pipelines automation with MLflow integration.

#### [Predictive ML Endpoint Monitoring](./monitoring/predictiveml-endpoint-monitoring/)
Real-time endpoint monitoring with Evidently AI data drift and model quality evaluation, CDK Lambda deployment for production scale, and MLflow tracking.

#### [Automated Drift and Trend Monitoring](./monitoring/sagemaker-automated-drift-and-trend-monitoring/)
Production-grade drift detection with Athena Iceberg data lake, QuickSight dashboards, and SNS alerting. Full solution at [sample-mlops-bestpractices](https://github.com/aws-samples/sample-mlops-bestpractices).

#### [LLM Inference Monitoring](./monitoring/sagemaker-endpoint-llm-monitoring/)
Serverless CDK infrastructure for LLM endpoint monitoring using MLflow GenAI evaluations and Amazon Bedrock, with event-driven Step Functions architecture.

#### [SageMaker Resource Observability with Grafana](./monitoring/resource-monitoring-grafana/)
Amazon Managed Grafana dashboards for endpoint infrastructure monitoring — per-GPU, per-container metrics, cost attribution, and resource utilization at 10-second granularity.

#### [LLM Quality Observability with Grafana](./monitoring/quality-monitoring-with-grafana/)
Grafana dashboards for LLM output quality scoring using GenAI Evaluations with Bedrock Claude as LLM-as-judge, with automated alerting via unified alerting and SNS.


### Examples
#### [SageMaker Managed MLflow Observability for Strands Agents on Amazon Bedrock AgentCore](./examples/sagemaker-mlflow-agentcore-runtime/)
-  Example with step-by-step instructions and deployment jupyter notebook to integrate Strands Agents in Amazon Bedrock AgentCore Runtime with Amazon SageMaker managed MLflow for observability. 

### Getting Started
To get started, follow these steps:

Clone the repository to your local machine:

```
git clone https://github.com/aws-samples/sample-aiops-on-amazon-sagemakerai.git
```
    
Navigate to the repository directory:

```
cd sample-aiops-on-amazon-sagemakerai
```

    
Explore the contents of the repository and follow the instructions in the `README.md` files within each subdirectory.

### Contributing

We welcome contributions to this repository! If you have any examples, improvements, or bug fixes to share, please see [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

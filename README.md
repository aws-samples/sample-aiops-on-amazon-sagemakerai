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
├── monitoring/                                     # All monitoring solutions
│   ├── predictiveml-batch-monitoring-pipeline/     # Batch ML monitoring with EvidentlyAI
│   ├── sagemaker-automated-drift-and-trend-monitoring/ # Real-time inference monitoring (→ moved to sample-mlops-bestpractices)
│   ├── sagemaker-endpoint-llm-monitoring/          # LLM inference monitoring
│   └── resource-monitoring-grafana/                # Resource & cost monitoring with Grafana
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
> Various Monitoring solutions for SageMakerAI. See the [folder directory for information](./monitoring/). 
#### [Predictive ML Batch Monitoring Pipeline with Evidently AI and MLflow](./monitoring/predictiveml-batch-monitoring-pipeline/)
Educational two-phase solution for learning and implementing batch ML monitoring. Uses Evidently AI for data drift detection and model quality tracking with SageMaker Pipelines automation. Ideal for periodic batch predictions and learning monitoring fundamentals.

#### [Automated Drift and Trend Monitoring for ML Models](./monitoring/sagemaker-automated-drift-and-trend-monitoring/)

Production-grade real-time inference monitoring with automated drift detection using SageMaker Pipelines, MLflow, Evidently AI, and QuickSight governance dashboards. Includes training pipeline, async inference logging, EventBridge-scheduled drift checks, SNS alerting, and configurable thresholds. **Full solution has moved to [https://github.com/aws-samples/sample-mlops-bestpractices](https://github.com/aws-samples/sample-mlops-bestpractices).**


#### [SageMaker Endpoint LLM Inference Monitoring with MLflow and GenAI Evaluations](./monitoring/sagemaker-endpoint-llm-monitoring/)
Serverless CDK infrastructure for monitoring LLM endpoints using MLflow GenAI evaluations (Safety, Relevance, Fluency, Guidelines, Coherence). Event-driven architecture with Amazon Bedrock integration for comprehensive quality assessment of large language models.

#### [SageMaker Resource Monitoring with Amazon Managed Grafana](./monitoring/resource-monitoring-grafana/)
Amazon Managed Grafana dashboards for SageMaker endpoint resource monitoring using Enhanced Container Metrics. Provides per-GPU, per-container, and per-inference-component observability at 10-second granularity with cost attribution, GPU/CPU/memory utilization tracking. Best for infrastructure monitoring and cost optimization of multi-model endpoints.


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

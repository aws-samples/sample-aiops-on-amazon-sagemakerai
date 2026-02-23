## AWS ModelOps on Amazon SageMakerAI 

This repository contains a collection of examples and resources to help you operationalize Generative AI (GenAI) and Machine learning workloads on SageMakerAI.

### Overview
The AWS ModelOps covering GenAIOps and MLOps patterns involing SageMakerAI resources like SageMaker managed MLflow, SageMaker pipelines and include all other AWS GenAI related features like Amazon Bedrock. This repository provides a set of sample notebooks, scripts, and configurations to help you explore different aspects of the ModelOps.

### Repository Structure
```
.
├── workshops/                               # Root folder for all workshops
│   └── ...                                  # Specific workshop folders
├── operations/                              # Root folder for operational guides
│   └── ...                                  # Specific operation folders
├── monitoring/                              # Root folder for monitoring solutions
│   └── ...                                  # Specific monitoring solution folders
├── examples/                                # Root folder for all examples
│   └── ...                                  # Specific example folders
├── LICENSE                                  # The Repository MIT-0 License
└── README.md                                # Root folder Repository documentation
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
#### [SageMaker Endpoint LLM Inference Monitoring with MLflow and GenAI Evaluations](./monitoring/sagemaker-endpoint-llm-monitoring/)
- Automated serverless infrastructure for monitoring SageMaker LLM endpoint inferences using AWS CDK, MLflow traces, and MLflow GenAI evaluations.
- Event-driven architecture using S3 Data Capture, EventBridge, Step Functions, and Lambda for real-time inference monitoring.
- Implements MLflow GenAI evaluations (Safety, Relevance, Fluency, Guidelines, Coherence) using Amazon Bedrock models for comprehensive quality assessment.
- Supports multiple deployment environments (dev, staging, prod) with unique resource naming via configurable stack prefixes.
- Includes complete CDK infrastructure-as-code with Docker-based Lambda functions, comprehensive documentation, and cost optimization guidance.

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

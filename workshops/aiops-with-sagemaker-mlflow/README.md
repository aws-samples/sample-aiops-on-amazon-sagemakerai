# Workshop: AIOps with SageMakerAI Managed MLflow

Welcome to the "AIOps with SageMakerAI Managed MLflow" workshop! The specialized technical workshop is designed for ML administrators and engineers seeking hands-on skills in managing and utilizing Amazon SageMakerAI managed MLflow. The course delves into essential topics such as administrating SageMaker managed MLFlow and example workloads. Participants will also gain deep insights into MLflow constructs like experiments, models, prompts, SageMaker-MLflow integration, and tracing. Advanced segments will cover workloads like genai agents, and LLM Model training. 

### What You'll Learn

By the end of this workshop, you'll be able to:

- Understand and implement SageMakerAI managed MLFlow
- Create and managed SageMakerAI managed MLFlow
- SageMakerAI managed MLFlow configurations and considerations for Administrators
- Operationalizing GenAI Agents with SageMakerAI managed MLFlow

### Workshop contents
The workshop contains jupyter notebooks where each notebook covers important AIOps topic in depth.
|Step|What|Notebook|
|---|---|---|
|1. |OSS MLflow fundamentals |[01-mlflow-fundamentals](01-mlflow-fundamentals.ipynb)|
|2. |LLM fine-tuning with SageMaker managed MLflow |[03-sagemaker-fine-tuning](03-sagemaker-fine-tuning.ipynb)|
|3. |Introduction to agent development with MLflow |[04-sagemaker-mlflow-agents-introduction](04-sagemaker-mlflow-agents-introduction.ipynb)|
|4. |OSS MLflow Agent evaluation |[04-sagemaker-mlflow-agents-evaluation](04-sagemaker-mlflow-agents-evaluation.ipynb)|
|5. |Amazon Bedrock agentCore integration with SageMaker managed MLflow |[04-sagemaker-mlflow-agentcore](04-sagemaker-mlflow-agentcore.ipynb)|
|6. |Introduction to DSPy |[04-sagemaker-mlflow-dspy](04-sagemaker-mlflow-dspy.ipynb)|

### How to run the workshop

This workshop follows a hands-on, self-paced format. Each module contains Jupyter notebooks that you'll run in your own JupyterLab environment (set up instructions provided in Module 0). The notebooks include:

- Step-by-step instructions and explanations
- Code samples that you can run and modify
- Exercises to reinforce your learning
- Links to additional resources

### Disclaimers

> This workshop is designed to run in the **us-east-1** (N. Virginia) region. Please ensure you are using this region throughout the workshop unless explicitly instructed otherwise.

> All code is covered under the [MIT-0 license](https://github.com/aws/mit-0)

> Please remember to clean up all resources created during this workshop to avoid ongoing charges to your AWS account.

### Security Best Practices

Throughout this workshop, we adhere to AWS service security best practices. We encourage you to familiarize yourself with the [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/) and apply them in your own implementations. Key points include:

- Using IAM roles and policies with least privilege
- Encrypting data at rest and in transit
- Implementing network security controls
- Regularly monitoring and auditing your resources

Remember to always follow security best practices when working with AWS services and sensitive data.
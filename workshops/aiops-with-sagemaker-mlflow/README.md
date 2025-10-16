# Workshop: AIOps with SageMakerAI Managed MLflow

Welcome to the "AIOps with SageMakerAI Managed MLflow" workshop! The specialized technical workshop is designed for platform administrators and ML engineers seeking hands-on skills in managing and utilizing Amazon SageMakerAI managed MLflow. The course delves into essential topics for administrating SageMaker AI managed MLFlow and AI/ML workloads with MLflow. Participants will gain deep insights into MLflow constructs like experiments, models, prompts, SageMaker-MLflow integrations. Advanced segments will cover AIML workloads using GenerativeAI agents, and Large-Language-Model training. 

### What You'll Learn

By the end of this workshop, you'll be able to:

- Understand and implement SageMakerAI managed MLFlow
- Create and managed SageMakerAI managed MLFlow
- SageMakerAI managed MLFlow configurations and considerations for Administrators
- Operationalizing GenAI Agents with SageMakerAI managed MLFlow

### Workshop contents
The workshop contains jupyter notebooks where each notebook covers important AIOps topic in depth.
|Step|What|Notebooks|
|---|---|---|
|1. |OSS MLflow fundamentals |[01-mlflow-fundamentals](01-mlflow-fundamentals.ipynb)|
|2. |Introduction to agent development with MLflow |[02-1-sagemaker-mlflow-agents-introduction.ipynb](02-1-sagemaker-mlflow-agents-introduction.ipynb)|
|3. |OSS MLflow Agent evaluation |[02-2-sagemaker-mlflow-agents-evaluation.ipynb](02-2-sagemaker-mlflow-agents-evaluation.ipynb)|
|4. |Amazon Bedrock agentCore integration with SageMaker managed MLflow |[02-3-sagemaker-mlflow-agentcore.ipynb](02-3-sagemaker-mlflow-agentcore.ipynb)|
|5. |LLM fine-tuning with SageMaker managed MLflow |Coming soon...|
|6. |Introduction to DSPy |Coming soon...|

### 🎓 Workshop stages 

| Module | Duration | Level | Target Audience 👥 |
|--------|----------|-------|-----------------|
| 1. Setup | 10-15 mins | Basic | All participants preparing setup and environment access |
| 2. AIOps Fundamentals | 30-40 mins | Intermediate | Data scientists, MLOps engineers, and AI practitioners new to MLflow concepts and AIOps practices |
| 3. SageMakerAI managed MLflow for Administrators | 30-40 mins | Advanced | Platform administrators, engineering leads responsible for governance, cost optimization, and security |
| 4. Operationalizing GenAI Agents | 50-70 mins | Advanced | ML Engineers, Data Scientists, Application developers |


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
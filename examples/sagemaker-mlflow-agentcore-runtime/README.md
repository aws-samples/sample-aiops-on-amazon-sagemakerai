

# SageMaker Managed MLflow Observability for Strands Agents on Amazon Bedrock AgentCore
This example provides step-by-step instructions, sample code, and deployment jupyter notebook to operationalize Strands Agents in Amazon Bedrock's AgentCore Runtime with Amazon SageMaker managed MLflow for observability. With this you will be able to observe Real-time agent interactions and tool invocations are recorded in MLflow for auditing and analytics of your agentic applications deployed in Amazon Bedrock AgentCore Runtime.

![image](./images/sagemaker-mlflow-agentCore.png)

## Features
- Deployment of tool-augmented financial analysis agents via Amazon Bedrock AgentCore Runtime
- AgentCore Runtime integration with SageMaker managed MLflow
- Automated tracing and experiment tracking using SageMaker managed MLflow (MLflow 3.4.0+ required). Sample output shown below.
- Example: Streaming real-time financial agent responses for investment advice

![image](./images/sagemaker-mlflow-output.png)

## Pre-requisites

- AWS Account 
- SageMakerAI Studio - [AWS Setup instructions](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html)
- SageMaker MLFlow tracking server - [AWS Setup instructions](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-create-tracking-server.html)
- Access to Bedrock LLM model - In this solution we are using the bedrock model with model ID `us.anthropic.claude-3-5-haiku-20241022-v1:0`
- Add the following SageMaker-mlflow IAM permission policy to the IAM role used for the notebook execution. 

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": "sagemaker-mlflow:*",
            "Resource": "*"
        }
    ]
}
```
- Configure the AgentCore runtime IAM permission settings as shown in [AWS documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html)
- To execute the sample code follow the notebook provided in this repository: [deploy_strands_agents_on_agentcore_with_sagemaker_MLflow.ipynb](./deploy_strands_agents_on_agentcore_with_sagemaker_MLflow.ipynb)

## Cleanup
Example cleanup commands and procedures to delete deployed agent, Cognito user pools, ECR repo, and runtime resources are provided in the notebook (see commented cleanup cells).

# License
This library is licensed under the MIT-0 License. See the LICENSE file.
---

*This solution demonstrates modern automation architecture patterns for AI/ML workloads on AWS, showcasing how to build scalable Agentic workloads.*
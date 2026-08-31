# Evaluation-Driven Development with Amazon SageMaker AI MLflow Apps

Companion code for the AWS Machine Learning Blog post **Evaluation-Driven Development with Amazon SageMaker AI MLflow Apps**.

This notebook walks through Evaluation-Driven Development (EDD): treating evaluation as an operational part of building an LLM application rather than an after-the-fact report. It builds a conversational Retrieval Augmented Generation (RAG) chatbot that answers Amazon Bedrock questions — retrieving from an Amazon Bedrock Knowledge Base and calling a `get_pricing` tool — and evaluates it end to end across response quality, retrieval, tool use, and whole-session behavior. The same `mlflow.genai.evaluate()` pass runs offline to gate changes before deployment and online against production traces.

Everything runs on AWS: MLflow 3.10.1 on Amazon SageMaker AI MLflow Apps provides tracking, versioned datasets, scorers, and the trace UI; Amazon Bedrock serves the generator models and the LLM-as-judge; and an Amazon Bedrock Knowledge Base backed by Amazon S3 Vectors powers retrieval.

> **Note:** This is sample code provided for demonstration and learning. It is not production-ready and should not be deployed to a production environment without additional review and security testing, including hardening the IAM policies, Amazon S3 bucket configuration, and credential handling for your own environment.

## Contents

- `evaluation-driven-development-with-mlflow.ipynb` — the end-to-end companion notebook (self-contained; the demo corpus is defined inline).

## What the notebook covers

1. Connect to a SageMaker AI MLflow App and enable Amazon Bedrock auto-tracing.
2. Provision an Amazon Bedrock Knowledge Base backed by Amazon S3 Vectors (idempotent).
3. Build a conversational RAG agent with one tool (`get_pricing`), traced end to end.
4. Create a versioned MLflow evaluation dataset.
5. Iterate with scoring — built-in scorers, deterministic `@scorer` functions, and custom `make_judge()` LLM judges, with judge validation and triangulation.
6. Run a per-turn A/B comparison across two model sizes.
7. Score whole conversations with session-level scorers.
8. Monitor production traces and read cost alongside quality.
9. Close the loop — feed failing production traces back into the dataset.

## Prerequisites

- An AWS account with permissions for Amazon Bedrock, Amazon S3 (including Amazon S3 Vectors), AWS Identity and Access Management (IAM, to create a Knowledge Base execution role), and Amazon SageMaker AI.
- An Amazon SageMaker AI MLflow App. If you don't have one, follow the [SageMaker AI MLflow Apps setup guide](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html) and copy its ARN into the notebook's Configuration cell.
- Access in your Region to the Amazon Bedrock models the notebook uses: the OpenAI gpt-oss-120b and gpt-oss-20b generators, the Claude Sonnet 4.6 judge, and Amazon Titan Text Embeddings V2.
- Python 3.10 or later, with AWS credentials configured in the environment (an Amazon SageMaker Studio notebook, or local Jupyter with credentials set up).

## Getting started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open `evaluation-driven-development-with-mlflow.ipynb`.

3. In the **Configuration** cell, set `MLFLOW_TRACKING_URI` (or edit `TRACKING_URI`) to your SageMaker AI MLflow App ARN. The Region and account ID are derived from the ARN, so it is the only value you need to provide.

4. Run the cells top to bottom.

## Cost and clean up

This notebook creates billable resources — an Amazon S3 source bucket, Amazon S3 Vectors storage, an Amazon Bedrock Knowledge Base, and an IAM role — and makes Amazon Bedrock API and embedding calls. The notebook's final cell deletes these resources; run it when you are finished to stop ongoing storage charges. Deletion is permanent and cannot be undone, so back up anything you need to keep first.

## Security

See [CONTRIBUTING](../../CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../../LICENSE) file.

# SageMaker Endpoint LLM Evaluation with MLflow GenAI

Comprehensive LLM evaluation example using Amazon SageMaker MLflow App and MLflow GenAI evaluation features to assess Large Language Models deployed on SageMaker Inference Endpoints.

## Overview

This example demonstrates how to systematically evaluate a LLM deployed on a SageMaker Inference Endpoint using MLflow's GenAI evaluation framework. The evaluation includes multiple metrics across different dimensions: safety, relevance, correctness, domain-specific guidelines, and third-party evaluation frameworks.

**Illustrative Example:** This notebook uses a medical domain LLM (fine-tuned Qwen model) for illustration purposes, but the evaluation framework and patterns are applicable to any domain-specific LLM use case (legal, financial, customer service, code generation, etc.).

## What You'll Learn

This example provides a reusable framework for evaluating LLMs across any domain. You will learn how to:

- Connect to a SageMaker MLflow App (managed MLflow tracking server)
- Define prediction wrappers for SageMaker Inference Endpoints
- Use `mlflow.genai.evaluate()` with multiple evaluation frameworks
- Implement custom scorers for domain-specific requirements
- Create LLM-as-a-judge evaluators using Amazon Bedrock
- Integrate third-party evaluation libraries (DeepEval, RAGAS)
- Analyze evaluation results in the MLflow UI

> **Note:** The notebook uses a medical domain LLM as a concrete example, but all techniques and patterns apply to any domain (legal, financial, customer service, code generation, etc.).

## Prerequisites

### AWS Resources

1. **SageMaker Inference Endpoint**
   - Deployed LLM model (e.g., Llama, Mistral, Qwen, or custom fine-tuned model for your domain)
   - Real-time endpoint with JSON input/output
   - Note the endpoint name for configuration

2. **SageMaker MLflow App**
   - Running MLflow tracking server (version 3.4.0+)
   - Note the ARN: `arn:aws:sagemaker:region:account-id:mlflow-app/app-id`

3. **IAM Permissions**
   - SageMaker endpoint invocation (`sagemaker:InvokeEndpoint`)
   - SageMaker MLflow operations (`sagemaker:CallMlflowAppApi`, `sagemaker:DescribeMlflowApp`)
   - Amazon Bedrock model access (`bedrock:InvokeModel`)
   - S3 access for MLflow artifacts

### Development Environment

- **SageMaker Studio** or **SageMaker Notebook Instance** (recommended)
- **Python 3.9+**
- **Jupyter Notebook** environment

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aws-samples/sample-aiops-on-amazon-sagemakerai.git
cd sample-aiops-on-amazon-sagemakerai/examples/sagemaker-endpoint-llm-evaluation
```

### 2. Install Dependencies
The installation steps are already added in the notebook solution so you can skip this steps

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `mlflow==3.8.0` - GenAI evaluation framework
- `sagemaker==2.245.0` - SageMaker Python SDK
- `sagemaker-mlflow==0.2.0` - SageMaker MLflow integration
- `deepeval==3.8.1` - DeepEval metrics integration
- `ragas==0.4.3` - RAGAS metrics for RAG evaluation
- `rouge_score==0.1.2` - ROUGE metrics
- `datasets==4.5.0` - HuggingFace datasets for evaluation data

### 3. Configure the Notebook

Open [SageMaker-mlflow-llm-evaluation.ipynb](SageMaker-mlflow-llm-evaluation.ipynb) and update the following variables:

```python
# Cell: Configure SageMaker AI Inference endpoint
SAGEMAKER_ENDPOINT_NAME = "your-endpoint-name"

# Cell: Configure SageMaker AI MLFlow
TRACKING_SERVER_ARN = "arn:aws:sagemaker:region:account:mlflow-app/app-id"
experiment_name = "your-experiment-name"

# Cell: Configure evaluation model
MLFLOW_EVALUATION_MODEL_ID = "bedrock:/global.anthropic.claude-sonnet-4-20250514-v1:0"
```

## Usage

### Run the Notebook

Execute the notebook cells sequentially:

1. **Environment Setup** - Install dependencies and import libraries
2. **SageMaker Configuration** - Configure endpoint and test invocation
3. **Evaluation Dataset** - Load and prepare evaluation dataset (question-answer pairs for your domain)
4. **MLflow Configuration** - Connect to MLflow tracking server
5. **Define Prediction Function** - Wrap endpoint calls for evaluation
6. **Configure Scorers** - Set up evaluation metrics
7. **Run Evaluation** - Execute `mlflow.genai.evaluate()`
8. **Review Results** - Analyze metrics in MLflow UI

### Quick Test

To evaluate a smaller subset for quick testing:

```python
# Evaluate 10 samples instead of full dataset
results = mlflow.genai.evaluate(
    data=eval_dataset[0:10],
    predict_fn=qa_predict_fn,
    scorers=scorers,
)
```

## Evaluation Metrics

The notebook implements a comprehensive evaluation suite across multiple dimensions:

### Built-in MLflow Scorers

| Scorer | Description | Requires LLM Judge |
|--------|-------------|-------------------|
| **Safety** | Content safety and policy violation detection | ✅ Bedrock |
| **RelevanceToQuery** | How well response addresses the question | ✅ Bedrock |
| **Equivalence** | Semantic similarity to expected response | ✅ Bedrock |
| **Correctness** | Factual and logical accuracy | ✅ Bedrock |

### Guidelines-Based Evaluation

Domain-specific guidelines tailored to your use case. The notebook demonstrates medical domain guidelines as examples:

- **follows_objective** - Adherence to domain-specific objectives (e.g., clinical objectives for medical, compliance for legal)
- **concise_communication** - Efficient, context-preserving responses
- **professional_tone** - Appropriate communication style for your domain (e.g., professional medical tone, formal legal language)
- **no_harmful_advice** - Domain-specific safety constraints (e.g., no diagnoses/prescriptions for medical, no legal advice for legal domain)
- **empathy_and_clarity** - User-centric communication with clear next steps

### Template-Based Judge

- **coherence_judge** - Custom prompt-based coherence evaluation

### Third-Party Integrations

**DeepEval Metrics** (20+ available):
- **Bias** - Detects bias in responses
- **AnswerRelevancy** - Context-aware relevance scoring
- **Faithfulness** - Grounding and factual consistency

**RAGAS Metrics** (RAG-focused):
- **ChrfScore** - Character n-gram F-score
- **BleuScore** - Lexical overlap metric

### Custom Scorers

- **is_brief** - Heuristic conciseness check (<15 words)
- **rougeL_fmeasure** - ROUGE-L F-measure for lexical similarity

### Automatic Metrics

- **Latency** - Per-sample inference time (via MLflow tracing)
- **Token counts** - Input/output tokens (requires custom implementation for SageMaker endpoints)

## Evaluation Dataset

The evaluation framework expects a dataset with input questions and expected responses.

**Example Format:**

```python
{
  "inputs": {
    "question": "Your domain-specific question..."
  },
  "expectations": {
    "expected_response": "Expected or reference answer..."
  }
}
```

**Illustrative Example:** The notebook uses the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset (medical Q&A) to demonstrate the evaluation workflow.

**Customize for Your Domain:**
Replace with your own evaluation dataset in the same format:
- `inputs.question` - Query/prompt for the model (e.g., customer inquiry, legal question, code generation request)
- `expectations.expected_response` - Reference answer or ground truth for comparison
- Use domain-specific datasets from HuggingFace, internal annotations, or synthetic data

## Results and Interpretation

### View in MLflow UI

1. Open your SageMaker MLflow App from the SageMaker console
2. Navigate to your experiment (e.g., `sagemaker-fine-tune-llm-evaluation` or your custom experiment name)
3. Select the evaluation run

### Metrics Dashboard

Review aggregated metrics across all samples:
- Safety scores (average)
- Relevance scores (average)
- Correctness scores (average)
- Custom metric distributions
- Latency statistics (p50, p95, p99)

### Traces View

Drill into individual samples:
- Per-sample latency breakdown
- Judge model invocations (inputs/outputs)
- Scorer-level spans (DeepEval, RAGAS)
- Token usage per scorer

### Compare Runs

Compare multiple evaluation runs:
- Different LLM versions
- Different prompt templates
- Different hyperparameters (temperature, top_p)
- Different judge models

## Project Structure

```
sagemaker-endpoint-llm-evaluation/
├── README.md                                # This file
├── SageMaker-mlflow-llm-evaluation.ipynb   # Main evaluation notebook
└── requirements.txt                         # Python dependencies
```

## Customization

### Adapt to Your Domain

The notebook uses medical domain as an illustrative example. To adapt to your domain:

1. **Replace the evaluation dataset** with your domain-specific Q&A pairs
2. **Update guidelines** to reflect your domain requirements (legal compliance, financial accuracy, brand voice, etc.)
3. **Modify custom scorers** to check domain-specific criteria
4. **Adjust terminology checks** for your domain vocabulary

**Example Domain Adaptations:**

| Domain | Dataset | Guidelines | Custom Scorers |
|--------|---------|------------|----------------|
| **Medical** | Medical Q&A pairs | Clinical accuracy, safety, empathy | Medical terminology check, diagnosis avoidance |
| **Legal** | Legal queries | Compliance, citations, disclaimers | Legal term usage, statute references |
| **Financial** | Investment queries | Regulatory compliance, risk disclosure | Financial metrics accuracy, disclaimer presence |
| **Customer Service** | Support tickets | Brand voice, empathy, resolution | Response time, escalation criteria |
| **Code Generation** | Programming problems | Code quality, security, efficiency | Syntax correctness, vulnerability checks |

### Use Your Own Model

Replace the endpoint name and adjust the prediction function:

```python
SAGEMAKER_ENDPOINT_NAME = "your-model-endpoint"

def qa_predict_fn(question: str) -> str:
    # Customize payload format for your endpoint
    response = predictor.predict({
        "prompt": question,
        "parameters": {...}
    })
    return response["generated_text"]
```

### Add Custom Scorers

Define domain-specific metrics tailored to your use case:

```python
from mlflow.genai import scorer

@scorer
def domain_terminology_check(outputs: str) -> bool:
    """Check if response uses appropriate domain-specific terminology

    Example for medical domain:
    domain_terms = ["diagnosis", "treatment", "symptoms", "condition"]

    Example for legal domain:
    domain_terms = ["liability", "statute", "plaintiff", "jurisdiction"]

    Example for financial domain:
    domain_terms = ["portfolio", "diversification", "risk", "return"]
    """
    domain_terms = ["term1", "term2", "term3"]  # Replace with your terms
    return any(term in outputs.lower() for term in domain_terms)

# Add to scorers list
scorers.append(domain_terminology_check)
```

### Use Different Judge Models

Change the LLM-as-a-judge model:

```python
# Use Claude Haiku for cost optimization
MLFLOW_EVALUATION_MODEL_ID = "bedrock:/us.anthropic.claude-3-5-haiku-20241022-v1:0"

# Or use different providers (OpenAI, Anthropic API)
MLFLOW_EVALUATION_MODEL_ID = "gpt-4o"
```

### Adjust Number of Samples

```python
# Evaluate subset for quick testing
results = mlflow.genai.evaluate(
    data=eval_dataset[0:10],  # First 10 samples
    predict_fn=qa_predict_fn,
    scorers=scorers,
)

# Full dataset evaluation
results = mlflow.genai.evaluate(
    data=eval_dataset,  # All samples
    predict_fn=qa_predict_fn,
    scorers=scorers,
)
```

## Cost Considerations

### SageMaker Inference Endpoint

- **Cost:** Varies by instance type (e.g., ml.g5.xlarge). See SageMakerAI pricing page for information on costs.
- **Optimization:** Use on-demand endpoints for evaluation, delete when done

### Amazon Bedrock (LLM-as-a-Judge)
- Input tokens
- Output tokens

**Cost Optimization:**
- Use **Claude Haiku** for simple scorers (80% cost reduction)
- Implement **sampling** (evaluate 10% of production traffic)
- Disable expensive scorers for quick tests

### MLflow Storage (S3)

- Negligible. See AWS S3 product pricing page for information on costs

## Troubleshooting

### Dependency Conflicts

The notebook may show dependency warnings when running in SageMaker Studio. These are typically safe to ignore if core libraries (mlflow, sagemaker) install successfully.

```bash
# Force reinstall if needed
pip install --force-reinstall -U mlflow==3.8.0 sagemaker==2.245.0
```

### Bedrock Rate Limits

If you encounter throttling errors:

```python
# Reduce concurrent judge invocations
MLFLOW_EVALUATION_MODEL_PARAM = {
    "temperature": 0,
    "max_tokens": 256,  # Reduce output tokens
}

# Evaluate smaller batches
for i in range(0, len(eval_dataset), 10):
    batch = eval_dataset[i:i+10]
    results = mlflow.genai.evaluate(data=batch, ...)
```

### Endpoint Timeout

If SageMaker endpoint times out:

```python
# Increase timeout in predictor
predictor = sagemaker.Predictor(
    endpoint_name=SAGEMAKER_ENDPOINT_NAME,
    sagemaker_session=sagemaker_session,
    serializer=sagemaker.serializers.JSONSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer(),
)
predictor._model_config["timeout"] = 300  # 5 minutes
```

### MLflow Connection Issues

Verify IAM permissions and network connectivity:

```python
# Test MLflow connection
import mlflow
mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
print(mlflow.get_tracking_uri())
print(mlflow.list_experiments())
```

## Best Practices

1. **Start Small**: Evaluate 10 samples first to verify setup
2. **Deterministic Generation**: Use `temperature=0` for reproducible evaluation
3. **Version Control**: Track evaluation datasets alongside code
4. **Experiment Naming**: Use descriptive names (model-version, date, dataset)
5. **Cost Monitoring**: Track Bedrock usage with CloudWatch metrics
6. **Ground Truth Quality**: Invest in high-quality reference answers
7. **Judge Model Selection**: Match judge model capability to task complexity

## Related Resources

### Documentation

- [MLflow GenAI Evaluation Guide](https://mlflow.org/docs/latest/genai/index.html)
- [SageMaker MLflow Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html)
- [Amazon Bedrock Model IDs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html)

### Related Examples

- [SageMaker Endpoint LLM Monitoring](../../monitoring/sagemaker-endpoint-llm-monitoring/) - Automated inference monitoring with MLflow traces
- [SageMaker MLflow AgentCore Runtime](../sagemaker-mlflow-agentcore-runtime/) - Observability for Strands Agents

### Evaluation Frameworks support in MLflow OSS

- [MLflow OSS DeepEval Documentation](https://mlflow.org/docs/3.9.0/genai/eval-monitor/scorers/third-party/deepeval/)
- [MLflow OSS RAGAS Documentation](https://mlflow.org/docs/3.9.0/genai/eval-monitor/scorers/third-party/ragas/)
- [MLflow OSS Arize Phoenix](https://mlflow.org/docs/3.9.0/genai/eval-monitor/scorers/third-party/phoenix/)

## Contributing

We welcome contributions! See [CONTRIBUTING](../../CONTRIBUTING.md) for guidelines.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../../LICENSE) file.

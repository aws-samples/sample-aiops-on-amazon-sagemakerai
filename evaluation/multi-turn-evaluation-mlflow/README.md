# Multi-Turn Evaluation & Conversation Simulation with MLflow on Amazon SageMaker AI

This folder contains the notebook [`multi_turn_eval_simulation.ipynb`](./multi_turn_eval_simulation.ipynb), which demonstrates **MLflow 3.10+ multi-turn evaluation** for assessing conversational AI quality across entire sessions — not just individual turns.

The agent under evaluation is built with the **Strands Agents SDK** and runs against a **SageMaker AI real-time inference endpoint** serving Qwen3.5-0.8B, with automatic tracing via `mlflow.strands.autolog()`.

## What the notebook covers

1. A traced conversational Strands agent backed by a SageMaker AI endpoint (`SageMakerAIModel`)
2. Session-ID tagging so MLflow groups turns into conversations
3. **Version tracking** with `mlflow.set_active_model()` so every trace and evaluation result links to a specific agent version
4. Evaluation of pre-recorded conversations with built-in session-level scorers
5. A custom multi-turn judge with `make_judge` and the `{{ conversation }}` template variable
6. Conversation simulation with `ConversationSimulator` using goals and personas
7. A v1-vs-v2 prompt-engineering comparison using version tracking and the simulator together
8. Single-turn regression testing against ground-truth answers with the `Correctness` judge

### Built-in multi-turn judges used

| Judge | What it evaluates |
|---|---|
| `ConversationCompleteness` | Were all user questions addressed by the end of the session? |
| `UserFrustration` | Did the user become frustrated? Was it resolved? |
| `KnowledgeRetention` | Does the agent remember information from earlier turns? |

## Prerequisites

- Python 3.10+
- An [MLflow App](https://aws.amazon.com/blogs/aws/accelerate-ai-development-using-amazon-sagemaker-ai-with-serverless-mlflow/) on Amazon SageMaker AI (serverless MLflow)
- Access to Anthropic Claude Sonnet 4.6 on Amazon Bedrock (used as the judge and simulated-user model)
- AWS credentials with `sagemaker:InvokeEndpoint` and `bedrock:InvokeModel` permissions
- A SageMaker AI endpoint serving Qwen3.5-0.8B — created by the deployment script below

Install dependencies (also run by the notebook's first cell):

```bash
pip install "mlflow==3.11.1" sagemaker-mlflow "strands-agents[sagemaker]" strands-agents-tools boto3
```

> **Why MLflow is pinned**: newer MLflow releases (observed with 3.14.0) have a regression in the native Bedrock judge adapter (`NotImplementedError: AmazonBedrockProvider does not implement get_endpoint_url`), which breaks the built-in LLM scorers unless `litellm` is installed. MLflow 3.11.1 invokes Bedrock judges natively without extra dependencies.

## Step 1 — Deploy the model endpoint

[`deploy_qwen_sagemaker.py`](./deploy_qwen_sagemaker.py) deploys **Qwen3.5-0.8B** to a SageMaker AI real-time endpoint using the AWS Deep Learning Container for vLLM (OpenAI-compatible chat completions API, `ml.g6.xlarge`):

```bash
# Required outside SageMaker Studio; inside Studio the execution role is auto-detected
export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account-id>:role/YourSageMakerExecutionRole"
python deploy_qwen_sagemaker.py
```

The script creates an endpoint named **`qwen3-5-0-8b`** (the name the notebook expects), waits until it is in service, and runs a smoke-test invocation. The endpoint is launched with vLLM's `--reasoning-parser qwen3`, so Qwen `<think>...</think>` blocks are stripped server-side and the LLM judges always see clean assistant messages.

> **Cost note**: `ml.g6.xlarge` is billed while the endpoint is running. Delete the endpoint when you are done (see Cleanup in the notebook).

## Step 2 — Configure and run the notebook

Open `multi_turn_eval_simulation.ipynb` and set the values in Section 1:

| Setting | Notes |
|---|---|
| `REGION` | Default `us-east-1` |
| `ENDPOINT_NAME` | Default `qwen3-5-0-8b` — matches the deploy script |
| `TRACKING_ARN` | Your MLflow App ARN (`arn:aws:sagemaker:...:mlflow-app/...`) |
| `EXPERIMENT_NAME` | Default `multi-turn-eval-demo` |
| `JUDGE_MODEL` | Default Bedrock Claude Sonnet 4.6 (`bedrock:/global.anthropic.claude-sonnet-4-6`) |

Then run all cells top to bottom.

## Notebook structure

| Section | Description |
|---|---|
| 0. Install Dependencies | `pip install` MLflow, Strands, and supporting packages |
| 1. Configuration | Region, endpoint name, MLflow App ARN, judge model |
| 2. Connect to MLflow | Set tracking URI, experiment, and enable `mlflow.strands.autolog()` |
| 3. Version-track your agent | `mlflow.set_active_model()` links traces and results to an agent version |
| 4. Define the Conversational Agent | Strands `Agent` + `SageMakerAIModel` with Qwen sampling presets |
| 5. Generate Pre-Recorded Conversations | One smooth and one frustrated multi-turn session |
| 6. Evaluate with Session-Level Scorers | `mlflow.genai.evaluate()` with the three built-in judges |
| 7. Custom Multi-Turn Judge | `tone_consistency` judge via `make_judge` + `{{ conversation }}` |
| 8. Conversation Simulation | `ConversationSimulator` with goals/personas and a `predict_fn` |
| 9. Compare Agent Versions | v1 (base prompt) vs v2 (modified prompt) — regression testing with the same simulator and scorers |
| 10. Ground-Truth Comparison | Single-turn `Correctness` regression testing of both versions |
| 11. Best Practices | Evaluation strategy across development, pre-release, and production |

## Implementation notes

- **Automatic tracing**: `mlflow.strands.autolog()` captures every Strands agent invocation — model calls, latencies, token usage — with no manual instrumentation.
- **Session grouping**: each turn is additionally tagged with `mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})` so MLflow can group turns into a session and apply session-level judges.
- **Version tracking**: `mlflow.set_active_model(name=...)` re-tags all downstream traces and evaluation results, enabling the v1-vs-v2 comparison in the MLflow UI Models view.
- **Simulation**: `mlflow.genai.evaluate(data=simulator, predict_fn=predict_fn, scorers=[...])` runs the simulated conversations and scores them in one call, using the same judges applied to pre-recorded data.

## Viewing results

Open the MLflow UI → experiment **`multi-turn-eval-demo`** → **Sessions** tab to see each conversation session with its scorer assessments and judge rationales, and the **Models** tab to compare agent versions side by side.

## References

- [MLflow Multi-Turn Evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/multi-turn/)
- [MLflow Conversation Simulation](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/conversation-simulation/)
- [MLflow Version Tracking](https://mlflow.org/docs/latest/genai/version-tracking/)
- [Strands Agents SDK — Amazon SageMaker model provider](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/amazon-sagemaker/)

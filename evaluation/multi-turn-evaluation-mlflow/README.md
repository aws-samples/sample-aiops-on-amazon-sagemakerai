# Multi-Turn Evaluation & Conversation Simulation with MLflow

This folder contains the notebook [`multi_turn_eval_simulation.ipynb`](./multi_turn_eval_simulation.ipynb), which demonstrates **MLflow 3.10+ multi-turn evaluation** for assessing conversational AI quality across entire sessions — not just individual turns.

The notebook evaluates a conversational agent that you provide. You supply the ARN of your own agent deployed on Amazon Bedrock AgentCore, and every turn is traced to MLflow so that sessions can be scored with session-level judges.

## What the notebook covers

1. Invoke a conversational agent and trace each turn to MLflow
2. Generate and trace pre-recorded multi-turn conversations
3. Evaluate existing conversations with built-in session-level scorers
4. Create a custom multi-turn judge with the `{{ conversation }}` template variable
5. Simulate conversations with `ConversationSimulator` using goals and personas
6. Run simulation and evaluation in a single `mlflow.genai.evaluate()` call

### Built-in multi-turn judges used

| Judge | What it evaluates |
|---|---|
| `ConversationCompleteness` | Were all user questions addressed by the end of the session? |
| `UserFrustration` | Did the user become frustrated? Was it resolved? |
| `KnowledgeRetention` | Does the agent remember information from earlier turns? |

## Prerequisites

- Python 3.10+
- `mlflow >= 3.10`, `boto3`, `pandas`, `sagemaker-mlflow`
- A SageMaker MLflow Tracking Server (MLflow App ARN)
- The ARN of an agent you have deployed on Amazon Bedrock AgentCore
- AWS credentials with Bedrock and SageMaker access

Install dependencies (also run by the notebook's first cell):

```bash
pip install --quiet "mlflow>=3.10" boto3 pandas sagemaker-mlflow
```

## Configuration

Before running, set the following values in the notebook:

| Setting | Where | Notes |
|---|---|---|
| `REGION` | Section 1 | Default `us-west-2` |
| `BEDROCK_MODEL_ID` | Section 1 | Judge/simulator model, default `global.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `AGENT_RUNTIME_ARN` | Section 1 | The ARN of your deployed agent |
| `TRACKING_URI` | Section 2 | Your MLflow App ARN (`YOUR_MLFLOW_APP_ARN`) |
| `EXP_NAME` | Section 2 | Experiment name, default `multi-turn-eval-agentcore` |

## How to run

1. Open `multi_turn_eval_simulation.ipynb`.
2. Fill in `AGENT_RUNTIME_ARN` (your agent's ARN) and `TRACKING_URI` (your MLflow App ARN), and confirm `REGION` / `BEDROCK_MODEL_ID`.
3. Run all cells top to bottom.

## Notebook structure

| Section | Description |
|---|---|
| 0. Install Dependencies | `pip install` MLflow and supporting packages |
| 1. Configuration | Region, Bedrock model, agent ARN |
| 2. Connect to MLflow | Set tracking URI, experiment, and enable `mlflow.bedrock.autolog()` |
| 3. Define Conversational Agent | The traced `chat_agent()` wrapper that tags each trace with a session ID |
| 4. Generate Pre-Recorded Conversations | Runs one smooth and one frustrated multi-turn session |
| 5. Evaluate with Session-Level Scorers | `mlflow.genai.evaluate()` with the three built-in judges |
| 6. Custom Multi-Turn Judge | `tone_consistency` judge via `make_judge` + `{{ conversation }}` |
| 7. Conversation Simulation | `ConversationSimulator` with goals/personas and a `predict_fn` |
| 8. Best Practices | Evaluation strategy across development, pre-release, and production |

## Implementation notes

- **Session grouping**: every trace is tagged with `mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})` so MLflow can group turns into a session and apply session-level judges.
- **Multi-turn context**: the agent receives a single `prompt` string; prior turns are concatenated into the prompt to give the agent conversation context.
- **Simulation**: `mlflow.genai.evaluate(data=simulator, predict_fn=predict_fn, scorers=[...])` runs the simulated conversations and scores them in one call, using the same judges applied to pre-recorded data.

## Viewing results

Open the MLflow UI → experiment **`multi-turn-eval-agentcore`** → **Sessions** tab to see each conversation session with its scorer assessments and the judge rationales.

## References

- [MLflow Multi-Turn Evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/multi-turn/)
- [MLflow Conversation Simulation](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/conversation-simulation/)
- [Strands Agents SDK](https://strandsagents.com/docs/user-guide/quickstart/python/)

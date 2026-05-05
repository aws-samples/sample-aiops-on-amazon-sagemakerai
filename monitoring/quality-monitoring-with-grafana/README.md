# LLM Quality Monitoring with CloudWatch and Grafana

Grafana dashboards for LLM quality monitoring on SageMaker Inference Components, using [MLflow GenAI Evaluations](https://mlflow.org/docs/latest/genai/eval-monitor/) and Bedrock Claude as an LLM-as-judge scorer. Quality scores are published as custom CloudWatch metrics and visualized in auto-refreshing Grafana dashboards.

This builds on the [Resource Monitoring with Grafana](../resource-monitoring-grafana) example — extending it from infrastructure metrics to LLM output quality metrics.

## Dashboard Panels

### Composite Quality & Relevance
Overall composite quality score and per-model relevance scores over time, with configurable threshold indicators.

![Composite Quality and Relevance](images/panel_1.png)

### Safety & Professional Tone
Safety score tracking for content policy compliance, and professional tone score measuring communication quality across inference components.

![Safety and Professional Tone](images/panel_2.png)

### Quality Evaluation Latency
End-to-end latency of the quality evaluation pipeline (inference + MLflow LLM-as-judge scoring) per inference component.

![Quality Evaluation Latency](images/panel_3.png)

## Setup

Run [`llm_quality_monitoring_cloudwatch_grafana.ipynb`](llm_quality_monitoring_cloudwatch_grafana.ipynb) — it creates the Grafana workspace, IAM role, CloudWatch data source, and deploys all dashboard panels programmatically. Requires an active SageMaker endpoint with inference components, Bedrock access, and IAM Identity Center (SSO).

## Quality Metrics Tracked

| Metric | Namespace | What it shows |
|---|---|---|
| `SafetyScore` | `SageMaker/InferenceQuality` | Content safety and policy compliance (0–1) |
| `RelevanceScore` | `SageMaker/InferenceQuality` | How well the response addresses the query (0–1) |
| `CoherenceScore` | `SageMaker/InferenceQuality` | Logical flow and consistency (0–1) |
| `ProfessionalToneScore` | `SageMaker/InferenceQuality` | Appropriate communication style (0–1) |
| `CompositeScore` | `SageMaker/InferenceQuality` | Weighted aggregate of all quality dimensions (0–1) |
| `EvaluationLatency` | `SageMaker/InferenceQuality` | End-to-end evaluation pipeline latency (ms) |

## Architecture

```
SageMaker Inference Component
        ↓
  Invoke endpoint
        ↓
  Log to CloudWatch Logs  ──→  MLflow Tracking (traces + evals)
        ↓
  Bedrock Claude (LLM-as-judge)
        ↓
  Publish custom metrics to CloudWatch
        ↓
  Grafana dashboard (auto-refresh)
```

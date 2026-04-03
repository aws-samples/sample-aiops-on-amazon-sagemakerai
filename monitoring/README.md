# Monitoring

#### Choosing the Right Monitoring Solution

| Criteria | Batch Monitoring Pipeline | Real-Time Inference Monitoring | LLM Monitoring |
|----------|---------------------------|--------------------------------|----------------|
| **Use Case** | Periodic batch predictions | Always-on endpoint inference | LLM endpoint evaluation |
| **Inference Type** | Batch Transform | Real-time endpoint | Real-time endpoint |
| **Deployment** | Educational (2 notebooks) | Production-ready (full pipeline) | Production CDK |
| **Data Storage** | S3 CSV files | Athena data lake (Iceberg) | S3 Data Capture |
| **Monitoring** | Data drift + model quality | Drift + performance + ground truth | GenAI evaluations |
| **Alerting** | SNS email | SNS email + MLflow | Step Functions |
| **Best For** | Learning ML monitoring | Production fraud detection | LLM safety/quality |
| **Setup Time** | 30-45 minutes | 2-3 hours | 1-2 hours |
| **Infrastructure** | SageMaker Pipeline | SageMaker + Lambda + Athena | CDK Serverless |
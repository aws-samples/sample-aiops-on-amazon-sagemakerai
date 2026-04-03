"""
Generate the Inference Monitoring Pipeline diagram — professional AWS style.

Usage:
    python docs/generate_inference_monitoring_diagram.py

Output:
    docs/guides/inference_monitoring_diagram.png
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
import os

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDES_DIR = os.path.join(DOCS_DIR, "guides")
ICONS = os.path.join(DOCS_DIR, "icons")

SAGEMAKER = os.path.join(ICONS, "sagemaker.png")
LAMBDA = os.path.join(ICONS, "lambda.png")
SQS = os.path.join(ICONS, "sqs.png")
EVENTBRIDGE = os.path.join(ICONS, "eventbridge.png")
SNS = os.path.join(ICONS, "sns.png")
ATHENA = os.path.join(ICONS, "athena.png")
S3 = os.path.join(ICONS, "s3.png")
CLOUDWATCH = os.path.join(ICONS, "cloudwatch.png")
MLFLOW = os.path.join(ICONS, "mlflow.png")
EVIDENTLY = os.path.join(ICONS, "evidently.png")
SAGEMAKER_NB = os.path.join(ICONS, "sagemaker_notebook.png")
CONFIG_FILE = os.path.join(ICONS, "config_file.png")

OUTPUT = os.path.join(GUIDES_DIR, "inference_monitoring_diagram")

GRAPH = {"fontsize": "11", "fontname": "Arial", "bgcolor": "white",
         "pad": "0.4", "ranksep": "0.8", "nodesep": "0.5",
         "splines": "spline", "dpi": "150"}
NODE = {"fontsize": "9", "fontname": "Arial", "fontcolor": "#232F3E"}
EDGE_S = {"fontsize": "8", "fontname": "Arial", "color": "#545B64"}
CL = {"bgcolor": "#FFFFFF", "style": "rounded,dashed", "pencolor": "#AAB7B8",
      "penwidth": "0.8", "fontsize": "9", "fontname": "Arial",
      "fontcolor": "#545B64", "labeljust": "l"}


with Diagram("", filename=OUTPUT, outformat="png", show=False,
             direction="TB", graph_attr=GRAPH, node_attr=NODE, edge_attr=EDGE_S):

    with Cluster("Inference Monitoring Pipeline", graph_attr={
        "bgcolor": "#FFFFFF", "style": "rounded", "pencolor": "#232F3E",
        "penwidth": "1.5", "fontsize": "12", "fontname": "Arial",
        "fontcolor": "#232F3E", "labeljust": "l"}):

        # Real-Time Inference
        with Cluster("Real-Time Inference", graph_attr={
            **CL, "bgcolor": "#F2F8FD", "pencolor": "#0972D3", "style": "rounded"}):
            ep = Custom("SageMaker Endpoint", SAGEMAKER)
            sqs_i = Custom("SQS\ninference-logs", SQS)
            lam_log = Custom("Lambda Logger", LAMBDA)
            ep >> Edge(label="async", color="#0972D3") >> sqs_i
            sqs_i >> Edge(label="batch", color="#0972D3") >> lam_log

        # Data Lake
        with Cluster("Athena Data Lake (Iceberg)", graph_attr={
            **CL, "bgcolor": "#F0FFF4", "pencolor": "#1B7742", "style": "rounded"}):
            athena_i = Custom("inference_responses", ATHENA)
            athena_t = Custom("training_data\n(baseline)", ATHENA)
            athena_m = Custom("monitoring_responses", ATHENA)
            s3 = Custom("S3 Data Lake", S3)

        lam_log >> Edge(label="INSERT", color="#0972D3") >> athena_i

        # Ground Truth
        with Cluster("Ground Truth (T+1 … T+30 d)", graph_attr={
            **CL, "bgcolor": "#FFF8F0", "pencolor": "#D45B07", "style": "rounded"}):
            gt = Custom("Ground Truth\nCapture", LAMBDA)
            gt_tbl = Custom("ground_truth_updates", ATHENA)
            merge = Custom("Athena MERGE", ATHENA)
            gt >> Edge(color="#D45B07") >> gt_tbl >> Edge(color="#D45B07") >> merge

        merge >> Edge(label="update gt", color="#D45B07") >> athena_i

        # Manual Monitoring
        with Cluster("Manual Monitoring (Notebook)", graph_attr={
            **CL, "bgcolor": "#F4F2FF", "pencolor": "#5B48D0", "style": "rounded"}):
            nb = Custom("inference_monitoring\n.ipynb", SAGEMAKER_NB)
            ev_m = Custom("Evidently AI", EVIDENTLY)

        nb >> Edge(color="#5B48D0") >> athena_i
        nb >> Edge(color="#5B48D0") >> ev_m

        # Drift Metrics Configuration
        with Cluster("Drift Metrics Configuration", graph_attr={
            **CL, "bgcolor": "#FFF3E0", "pencolor": "#E65100", "style": "rounded"}):
            cfg = Custom("config.yaml\n(drift thresholds)", CONFIG_FILE)

        # Automated Monitoring
        with Cluster("Automated Monitoring (Lambda)", graph_attr={
            **CL, "bgcolor": "#FFF8F0", "pencolor": "#D45B07", "style": "rounded"}):
            eb = Custom("EventBridge\n(daily cron)", EVENTBRIDGE)
            lam_d = Custom("Drift Monitor\nLambda", LAMBDA)
            ev_a = Custom("Evidently AI", EVIDENTLY)
            eb >> Edge(label="trigger", color="#D45B07") >> lam_d
            lam_d >> Edge(color="#D45B07") >> ev_a

        cfg >> Edge(label="data drift\nthresholds", style="dashed", color="#D45B07") >> lam_d
        cfg >> Edge(label="model drift\nthresholds", style="dashed", color="#D45B07") >> ev_a

        lam_d >> Edge(label="query baseline", color="#D45B07") >> athena_t
        lam_d >> Edge(label="query inference", color="#D45B07") >> athena_i

        # MLflow hub
        mlflow = Custom("SageMaker MLflow App\n(reports · metrics · artifacts)", MLFLOW)
        ev_m >> Edge(label="HTML reports", color="#7B1FA2") >> mlflow
        ev_a >> Edge(label="HTML reports", color="#7B1FA2") >> mlflow

        # Writer
        with Cluster("Monitoring Writer", graph_attr=CL):
            sqs_m = Custom("SQS", SQS)
            lam_w = Custom("Lambda Writer", LAMBDA)
            sqs_m >> Edge(color="#D45B07") >> lam_w

        lam_d >> Edge(label="results", color="#D45B07") >> sqs_m
        lam_w >> Edge(label="INSERT", color="#D45B07") >> athena_m

        # Alerting
        sns_a = Custom("SNS Alerts", SNS)
        cw = Custom("CloudWatch", CLOUDWATCH)
        lam_d >> Edge(label="if drift", color="#D32F2F") >> sns_a
        cw >> Edge(color="#545B64") >> lam_d


print(f"\n✅  Inference monitoring diagram → {OUTPUT}.png")
print(f"    {os.path.getsize(OUTPUT + '.png') / 1024:.0f} KB")

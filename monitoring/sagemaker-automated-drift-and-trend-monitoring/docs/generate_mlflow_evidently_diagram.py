"""
Generate the MLflow + Evidently Monitoring Flow diagram — professional AWS style.

Usage:
    python docs/generate_mlflow_evidently_diagram.py

Output:
    docs/guides/mermaid-diagram-mlflow-evidently.png
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
import os

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDES_DIR = os.path.join(DOCS_DIR, "guides")
ICONS = os.path.join(DOCS_DIR, "icons")

SAGEMAKER_NB = os.path.join(ICONS, "sagemaker_notebook.png")
LAMBDA = os.path.join(ICONS, "lambda.png")
EVENTBRIDGE = os.path.join(ICONS, "eventbridge.png")
SNS = os.path.join(ICONS, "sns.png")
ATHENA = os.path.join(ICONS, "athena.png")
MLFLOW = os.path.join(ICONS, "mlflow.png")
EVIDENTLY = os.path.join(ICONS, "evidently.png")
CONFIG_FILE = os.path.join(ICONS, "config_file.png")

OUTPUT = os.path.join(GUIDES_DIR, "mermaid-diagram-mlflow-evidently")

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

    with Cluster("MLflow + Evidently Monitoring Flow", graph_attr={
        "bgcolor": "#FFFFFF", "style": "rounded", "pencolor": "#232F3E",
        "penwidth": "1.5", "fontsize": "12", "fontname": "Arial",
        "fontcolor": "#232F3E", "labeljust": "l"}):

        # Data Sources
        with Cluster("Athena Data Lake", graph_attr={
            **CL, "bgcolor": "#F0FFF4", "pencolor": "#1B7742", "style": "rounded"}):
            a_train = Custom("training_data\n(baseline)", ATHENA)
            a_infer = Custom("inference_responses\n(current)", ATHENA)

        # Evidently Reports
        with Cluster("Evidently AI Reports", graph_attr={
            **CL, "bgcolor": "#FFF0F0", "pencolor": "#C62828", "style": "rounded"}):
            ev_drift = Custom("DataDriftPreset\n(PSI · KS · distributions)", EVIDENTLY)
            ev_class = Custom("ClassificationPreset\n(ROC · PR · confusion matrix)", EVIDENTLY)

        # Manual workflow
        with Cluster("Manual Workflow (Data Scientist)", graph_attr={
            **CL, "bgcolor": "#F4F2FF", "pencolor": "#5B48D0", "style": "rounded"}):
            nb = Custom("inference_monitoring\n.ipynb", SAGEMAKER_NB)

        nb >> Edge(color="#5B48D0") >> a_train
        nb >> Edge(color="#5B48D0") >> a_infer
        nb >> Edge(color="#5B48D0") >> ev_drift
        nb >> Edge(color="#5B48D0") >> ev_class

        # Drift Metrics Configuration
        with Cluster("Drift Metrics Configuration", graph_attr={
            **CL, "bgcolor": "#FFF3E0", "pencolor": "#E65100", "style": "rounded"}):
            cfg = Custom("config.yaml\n(drift thresholds)", CONFIG_FILE)

        # Automated workflow
        with Cluster("Automated Workflow (Scheduled)", graph_attr={
            **CL, "bgcolor": "#FFF8F0", "pencolor": "#D45B07", "style": "rounded"}):
            eb = Custom("EventBridge\n(daily 2 AM)", EVENTBRIDGE)
            lam = Custom("Drift Monitor\nLambda", LAMBDA)

        eb >> Edge(label="trigger", color="#D45B07") >> lam
        cfg >> Edge(label="data drift\nthresholds", style="dashed", color="#D45B07") >> lam
        cfg >> Edge(label="model drift\nthresholds", style="dashed", color="#D45B07") >> ev_drift
        lam >> Edge(color="#D45B07") >> a_train
        lam >> Edge(color="#D45B07") >> a_infer
        lam >> Edge(color="#D45B07") >> ev_drift
        lam >> Edge(color="#D45B07") >> ev_class

        # MLflow hub
        with Cluster("SageMaker MLflow App", graph_attr={
            **CL, "bgcolor": "#F2F8FD", "pencolor": "#0972D3", "style": "rounded"}):
            ml_metrics = Custom("Metrics\ndrift scores · ROC-AUC\naccuracy · precision · recall", MLFLOW)
            ml_artifacts = Custom("Artifacts\ndata_drift_*.html\nclassification_*.html\ndrift_summary_*.json", MLFLOW)

        ev_drift >> Edge(label="drift scores", color="#7B1FA2") >> ml_metrics
        ev_drift >> Edge(label="HTML report", color="#7B1FA2") >> ml_artifacts
        ev_class >> Edge(label="perf metrics", color="#7B1FA2") >> ml_metrics
        ev_class >> Edge(label="HTML report", color="#7B1FA2") >> ml_artifacts

        # Alerting
        sns_a = Custom("SNS Alerts\n(email)", SNS)
        lam >> Edge(label="if drift", color="#D32F2F") >> sns_a


print(f"\n✅  MLflow+Evidently diagram → {OUTPUT}.png")
print(f"    {os.path.getsize(OUTPUT + '.png') / 1024:.0f} KB")

"""
Generate the MLOps architecture diagram with AWS icons.

Usage:
    python docs/generate_architecture_diagram.py

Output:
    docs/guides/architecture_diagram.png

Requires:
    pip install diagrams
    brew install graphviz  (macOS)
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.ml import Sagemaker, SagemakerNotebook
from diagrams.aws.compute import Lambda
from diagrams.aws.integration import SQS, Eventbridge, SNS
from diagrams.aws.analytics import Athena
from diagrams.aws.storage import S3
from diagrams.custom import Custom
import os
import urllib.request

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDES_DIR = os.path.join(DOCS_DIR, "guides")
ICONS_DIR = os.path.join(DOCS_DIR, "icons")
os.makedirs(ICONS_DIR, exist_ok=True)

MLFLOW_ICON = os.path.join(ICONS_DIR, "mlflow.png")
EVIDENTLY_ICON = os.path.join(ICONS_DIR, "evidently.png")
XGBOOST_ICON = os.path.join(ICONS_DIR, "xgboost.png")


def download_icon(url, path):
    """Download icon if not already cached."""
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  Downloaded: {path}")
        except Exception as e:
            print(f"  Could not download {url}: {e}")
            _create_placeholder(path)


def _create_placeholder(path):
    """Create a minimal 1x1 PNG placeholder so diagrams doesn't crash."""
    import struct, zlib
    def create_minimal_png():
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        raw = zlib.compress(b'\x00\xff\xff\xff')
        idat_crc = zlib.crc32(b'IDAT' + raw) & 0xffffffff
        idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + struct.pack('>I', idat_crc)
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        return sig + ihdr + idat + iend
    with open(path, 'wb') as f:
        f.write(create_minimal_png())


download_icon("https://avatars.githubusercontent.com/u/39938107?s=200&v=4", MLFLOW_ICON)
download_icon("https://avatars.githubusercontent.com/u/82784750?s=200&v=4", EVIDENTLY_ICON)
download_icon(
    "https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png",
    XGBOOST_ICON,
)

OUTPUT_PATH = os.path.join(GUIDES_DIR, "architecture_diagram")

graph_attr = {
    "fontsize": "14",
    "fontname": "Helvetica",
    "bgcolor": "white",
    "pad": "0.6",
    "ranksep": "1.2",
    "nodesep": "0.5",
    "splines": "curved",
}

node_attr = {
    "fontsize": "11",
    "fontname": "Helvetica",
}

edge_attr = {
    "fontsize": "9",
    "fontname": "Helvetica",
    "color": "#555555",
}


with Diagram(
    "Credit Card Fraud Detection \u2014 MLOps Architecture",
    filename=OUTPUT_PATH,
    outformat="png",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):
    # MLflow (central hub)
    mlflow = Custom("MLflow Tracking Server\n(Experiments \xb7 Registry\nMetrics \xb7 Artifacts)", MLFLOW_ICON)

    # Training Pipeline (left)
    with Cluster(
        "Training Pipeline\n(SageMaker Pipelines)",
        graph_attr={"bgcolor": "#e8f4fd", "style": "rounded", "labeljust": "l"},
    ):
        nb_train = SagemakerNotebook("pipeline_execution\n.ipynb")
        preprocess = Sagemaker("Preprocess\n(ScriptProcessor)")
        train = Sagemaker("Train\n(XGBoost)")
        evaluate = Sagemaker("Evaluate\n(Quality Gate)")
        deploy_step = Lambda("Deploy Endpoint\n(Lambda)")

        nb_train >> Edge(label="orchestrates") >> preprocess
        preprocess >> train >> evaluate >> deploy_step

    # Real-Time Inference (right of training)
    with Cluster(
        "Real-Time Inference",
        graph_attr={"bgcolor": "#fef9e7", "style": "rounded", "labeljust": "l"},
    ):
        endpoint = Sagemaker("SageMaker Endpoint\n(Custom Handler)")
        sqs = SQS("SQS Queue\n(fraud-inference-logs)")
        lambda_logger = Lambda("Inference Logger\n(Lambda)")

        endpoint >> Edge(label="async fire-and-forget") >> sqs
        sqs >> Edge(label="batch 10 msgs / 30s") >> lambda_logger

    # Athena Data Lake (center)
    with Cluster(
        "Athena Data Lake (Iceberg)",
        graph_attr={"bgcolor": "#eafaf1", "style": "rounded", "labeljust": "l"},
    ):
        s3_bucket = S3("S3 Data Lake Bucket")
        athena_training = Athena("training_data\n(284K rows)")
        athena_inference = Athena("inference_responses\n(partitioned by date)")
        athena_gt = Athena("ground_truth_updates")

    # Ground Truth
    with Cluster(
        "Ground Truth Capture\n(T+1 to T+30 days)",
        graph_attr={"bgcolor": "#fdf2e9", "style": "rounded", "labeljust": "l"},
    ):
        gt_sim = Lambda("Simulate / Capture\nGround Truth")
        gt_merge = Athena("MERGE\nground_truth \u2192 inference")

    # Monitoring & Drift Detection
    with Cluster(
        "Monitoring & Drift Detection",
        graph_attr={"bgcolor": "#f5eef8", "style": "rounded", "labeljust": "l"},
    ):
        nb_monitor = SagemakerNotebook("inference_monitoring\n.ipynb")
        evidently = Custom("Evidently AI\nDataDriftPreset\nClassificationPreset", EVIDENTLY_ICON)

        with Cluster(
            "Automated Monitoring",
            graph_attr={"bgcolor": "#fdebd0", "style": "rounded"},
        ):
            eventbridge = Eventbridge("EventBridge\n(daily 2 AM UTC)")
            lambda_drift = Lambda("Drift Monitor\n(Lambda)")
            sns = SNS("SNS Alerts\n(Email)")

        eventbridge >> Edge(label="triggers daily") >> lambda_drift
        lambda_drift >> Edge(label="if drift detected") >> sns

    # Connections
    deploy_step >> Edge(label="creates endpoint") >> endpoint
    preprocess >> Edge(label="reads training data", style="dashed") >> athena_training
    train >> Edge(label="logs metrics & model", style="dashed") >> mlflow
    lambda_logger >> Edge(label="INSERT INTO") >> athena_inference
    gt_sim >> athena_gt
    athena_gt >> gt_merge
    gt_merge >> Edge(label="updates ground_truth col") >> athena_inference
    nb_monitor >> Edge(label="queries Athena") >> athena_inference
    nb_monitor >> Edge(label="runs reports") >> evidently
    evidently >> Edge(label="HTML reports & metrics") >> mlflow
    lambda_drift >> Edge(label="queries inference", style="dashed") >> athena_inference
    lambda_drift >> Edge(label="queries baseline", style="dashed") >> athena_training
    lambda_drift >> Edge(label="logs Evidently reports", style="dashed") >> mlflow
    s3_bucket >> Edge(style="dotted", color="#aaaaaa") >> athena_training
    s3_bucket >> Edge(style="dotted", color="#aaaaaa") >> athena_inference

print(f"\n\u2705 Architecture diagram generated: {OUTPUT_PATH}.png")
print(f"   Size: {os.path.getsize(OUTPUT_PATH + '.png') / 1024:.0f} KB")

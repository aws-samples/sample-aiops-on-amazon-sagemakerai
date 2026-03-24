"""
Generate the MLOps architecture diagram with official AWS icons.

Professional AWS reference-architecture style with 3 swim lanes inside
an AWS Cloud boundary:
  1. Training Pipeline
  2. Inference Monitoring
  3. Governance Dashboard

Usage:
    python docs/generate_architecture_diagram.py

Output:
    docs/guides/architecture_diagram.png

Requires:
    pip install diagrams
    brew install graphviz  (macOS)
    AWS icons in docs/icons/ (see README § Diagram Generation)
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
import os
import urllib.request

# ── Paths ────────────────────────────────────────────────────────────────
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDES_DIR = os.path.join(DOCS_DIR, "guides")
ICONS_DIR = os.path.join(DOCS_DIR, "icons")
os.makedirs(ICONS_DIR, exist_ok=True)

# AWS icons (official Architecture Icon package → docs/icons/)
SAGEMAKER = os.path.join(ICONS_DIR, "sagemaker.png")
SAGEMAKER_NB = os.path.join(ICONS_DIR, "sagemaker_notebook.png")
LAMBDA = os.path.join(ICONS_DIR, "lambda.png")
SQS = os.path.join(ICONS_DIR, "sqs.png")
EVENTBRIDGE = os.path.join(ICONS_DIR, "eventbridge.png")
SNS = os.path.join(ICONS_DIR, "sns.png")
ATHENA = os.path.join(ICONS_DIR, "athena.png")
QUICKSIGHT = os.path.join(ICONS_DIR, "quicksight.png")
S3 = os.path.join(ICONS_DIR, "s3.png")
CLOUDWATCH = os.path.join(ICONS_DIR, "cloudwatch.png")
GLUE = os.path.join(ICONS_DIR, "glue.png")

# Third-party icons (auto-downloaded on first run)
MLFLOW = os.path.join(ICONS_DIR, "mlflow.png")
EVIDENTLY = os.path.join(ICONS_DIR, "evidently.png")

# Config file icon (for configurability)
CONFIG_FILE = os.path.join(ICONS_DIR, "config_file.png")


# ── Icon helpers ─────────────────────────────────────────────────────────
def _download(url, path):
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            _placeholder(path)


def _placeholder(path):
    import struct, zlib
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    c = zlib.crc32(b'IHDR' + ihdr) & 0xffffffff
    ihdr_chunk = struct.pack('>I', 13) + b'IHDR' + ihdr + struct.pack('>I', c)
    raw = zlib.compress(b'\x00\xff\xff\xff')
    c2 = zlib.crc32(b'IDAT' + raw) & 0xffffffff
    idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + struct.pack('>I', c2)
    c3 = zlib.crc32(b'IEND') & 0xffffffff
    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', c3)
    with open(path, 'wb') as f:
        f.write(sig + ihdr_chunk + idat + iend)


_download("https://avatars.githubusercontent.com/u/39938107?s=200&v=4", MLFLOW)
_download("https://avatars.githubusercontent.com/u/82784750?s=200&v=4", EVIDENTLY)

# ── Diagram styling ─────────────────────────────────────────────────────
OUTPUT = os.path.join(GUIDES_DIR, "architecture_diagram")

# Clean, professional AWS reference-architecture style
# Increased spacing to prevent overlaps
GRAPH = {
    "fontsize": "11",
    "fontname": "Arial",
    "bgcolor": "white",
    "pad": "0.6",          # Increased padding
    "ranksep": "1.2",      # Increased vertical spacing
    "nodesep": "0.8",      # Increased horizontal spacing
    "splines": "ortho",    # Orthogonal edges (cleaner, less overlap)
    "dpi": "150",
    "compound": "true",    # Allow edges between clusters
}
NODE = {"fontsize": "9", "fontname": "Arial", "fontcolor": "#232F3E"}
EDGE_STYLE = {"fontsize": "8", "fontname": "Arial", "color": "#545B64"}

# Cluster palette — light fills, thin borders, AWS-style
CL_AWS = {"bgcolor": "#FFFFFF", "style": "rounded", "pencolor": "#232F3E",
           "penwidth": "2", "fontsize": "13", "fontname": "Arial Bold",
           "fontcolor": "#232F3E", "labeljust": "l", "margin": "20"}
CL_LANE1 = {"bgcolor": "#F2F8FD", "style": "rounded", "pencolor": "#0972D3",
             "penwidth": "1.5", "fontsize": "11", "fontname": "Arial Bold",
             "fontcolor": "#0972D3", "labeljust": "l", "margin": "16"}
CL_LANE2 = {"bgcolor": "#FFF8F0", "style": "rounded", "pencolor": "#D45B07",
             "penwidth": "1.5", "fontsize": "11", "fontname": "Arial Bold",
             "fontcolor": "#D45B07", "labeljust": "l", "margin": "16"}
CL_LANE3 = {"bgcolor": "#F4F2FF", "style": "rounded", "pencolor": "#5B48D0",
             "penwidth": "1.5", "fontsize": "11", "fontname": "Arial Bold",
             "fontcolor": "#5B48D0", "labeljust": "l", "margin": "16"}
CL_SUB = {"bgcolor": "#FFFFFF", "style": "rounded,dashed", "pencolor": "#AAB7B8",
           "penwidth": "0.8", "fontsize": "9", "fontname": "Arial",
           "fontcolor": "#545B64", "labeljust": "l", "margin": "12"}


# ── Build diagram ────────────────────────────────────────────────────────
with Diagram(
    "",
    filename=OUTPUT,
    outformat="png",
    show=False,
    direction="LR",  # Left-to-right for better flow
    graph_attr=GRAPH,
    node_attr=NODE,
    edge_attr=EDGE_STYLE,
):

    with Cluster("AWS Cloud", graph_attr=CL_AWS):

        # ════════════════════════════════════════════════════════════
        # LANE 1 — Training Pipeline (Left side)
        # ════════════════════════════════════════════════════════════
        with Cluster("Training Pipeline\n(SageMaker Pipelines)", graph_attr=CL_LANE1):

            # Data source
            s3_data = Custom("S3\nData Lake", S3)

            # Processing pipeline
            athena_train = Custom("Athena\ntraining_data", ATHENA)
            pyspark = Custom("PySpark\nProcessing", GLUE)
            xgb = Custom("XGBoost\nTraining", SAGEMAKER)
            evaluate = Custom("Evaluate\n(Quality Gate)", SAGEMAKER)

            # MLflow tracking
            mlflow_t = Custom("MLflow\nTracking", MLFLOW)

            # Deployment
            ep = Custom("SageMaker\nEndpoint", SAGEMAKER)

            # Inference logging pipeline
            sqs_i = Custom("SQS\nLogs", SQS)
            lam_log = Custom("λ Logger", LAMBDA)
            athena_i = Custom("Athena\ninference_\nresponses", ATHENA)

            # Vertical flow for training
            s3_data >> Edge(label="migrate", color="#0972D3", minlen="1") >> athena_train
            athena_train >> Edge(color="#0972D3", minlen="1") >> pyspark
            pyspark >> Edge(color="#0972D3", minlen="1") >> xgb
            xgb >> Edge(color="#0972D3", minlen="1") >> evaluate

            # MLflow connections
            xgb >> Edge(label="metrics", color="#7B1FA2", style="dashed") >> mlflow_t
            evaluate >> Edge(label="register", color="#7B1FA2", style="dashed") >> mlflow_t

            # Deployment flow
            evaluate >> Edge(label="deploy", color="#0972D3", minlen="1") >> ep

            # Inference logging flow
            ep >> Edge(label="async", color="#0972D3") >> sqs_i
            sqs_i >> Edge(label="batch", color="#0972D3") >> lam_log
            lam_log >> Edge(label="INSERT", color="#0972D3") >> athena_i

        # ════════════════════════════════════════════════════════════
        # LANE 2 — Inference Monitoring (Middle)
        # ════════════════════════════════════════════════════════════
        with Cluster("Inference Monitoring\n(Drift Detection)", graph_attr=CL_LANE2):

            # Configuration
            cfg = Custom("config.yaml\n(thresholds)", CONFIG_FILE)

            # Ground truth
            gt_sim = Custom("Ground Truth\nSimulator", LAMBDA)
            athena_gt = Custom("Athena\nground_truth", ATHENA)

            # Drift detection
            eb_drift = Custom("EventBridge\n2 AM daily", EVENTBRIDGE)
            lam_drift = Custom("λ Drift\nMonitor", LAMBDA)
            ev = Custom("Evidently AI\nDrift Analysis", EVIDENTLY)
            mlflow_m = Custom("MLflow\nMonitoring", MLFLOW)

            # Monitoring storage
            sqs_m = Custom("SQS\nResults", SQS)
            lam_w = Custom("λ Writer", LAMBDA)
            athena_m = Custom("Athena\nmonitoring_\nresponses", ATHENA)

            # Alerting
            sns_a = Custom("SNS\nAlerts", SNS)
            cw = Custom("CloudWatch\nLogs", CLOUDWATCH)

            # Ground truth flow
            gt_sim >> Edge(color="#D45B07") >> athena_gt
            athena_gt >> Edge(label="MERGE", color="#D45B07") >> athena_i

            # Drift monitoring flow
            eb_drift >> Edge(label="trigger", color="#D45B07") >> lam_drift
            cfg >> Edge(style="dashed", color="#999") >> lam_drift

            lam_drift >> Edge(label="query", color="#D45B07", style="dashed") >> athena_train
            lam_drift >> Edge(label="query", color="#D45B07", style="dashed") >> athena_i

            lam_drift >> Edge(color="#D45B07") >> ev
            ev >> Edge(label="reports", color="#7B1FA2", style="dashed") >> mlflow_m

            # Results storage
            lam_drift >> Edge(label="results", color="#D45B07") >> sqs_m
            sqs_m >> Edge(color="#D45B07") >> lam_w
            lam_w >> Edge(label="INSERT", color="#D45B07") >> athena_m

            # Alerting
            lam_drift >> Edge(label="if drift", color="#D32F2F") >> sns_a
            cw >> Edge(color="#999", style="dashed") >> lam_drift

        # ════════════════════════════════════════════════════════════
        # LANE 3 — Governance Dashboard (Right side)
        # ════════════════════════════════════════════════════════════
        with Cluster("Governance Dashboard\n(QuickSight)", graph_attr=CL_LANE3):

            # Data refresh automation
            eb_refresh = Custom("EventBridge\n3 AM daily", EVENTBRIDGE)
            lam_refresh = Custom("λ Dataset\nRefresh", LAMBDA)

            # Dashboard
            qs = Custom("QuickSight\nDashboard", QUICKSIGHT)

            # Dataset refresh flow
            eb_refresh >> Edge(label="trigger", color="#5B48D0") >> lam_refresh
            lam_refresh >> Edge(label="refresh", color="#5B48D0") >> qs

            # Data connections
            athena_i >> Edge(label="inference", color="#5B48D0", style="dashed") >> qs
            athena_m >> Edge(label="drift", color="#5B48D0", style="dashed") >> qs


print(f"\n✅  Architecture diagram → {OUTPUT}.png")
print(f"    {os.path.getsize(OUTPUT + '.png') / 1024:.0f} KB")
print("\n📊  Diagram improvements:")
print("    - Left-to-right layout for better readability")
print("    - Orthogonal edges to minimize overlaps")
print("    - Increased spacing between nodes and clusters")
print("    - Cleaner labels with line breaks")
print("    - Dashed lines for cross-lane queries")
print("    - Color-coded by lane (Blue/Orange/Purple)")

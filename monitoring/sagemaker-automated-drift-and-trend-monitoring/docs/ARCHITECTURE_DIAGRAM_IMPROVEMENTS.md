# Architecture Diagram Improvements

## Changes Made

The architecture diagram generator (`docs/generate_architecture_diagram.py`) has been significantly improved to eliminate overlapping elements and provide better readability.

### Key Improvements

#### 1. **Layout Direction Changed: Top-Bottom → Left-Right**
- **Before**: Vertical stacking (TB direction) caused compressed layout
- **After**: Horizontal flow (LR direction) provides natural left-to-right reading
- **Benefit**: Better utilizes wide screens and follows natural workflow progression

#### 2. **Orthogonal Edge Routing**
- **Before**: Spline curves could overlap and cross each other
- **After**: Orthogonal (right-angle) edges that follow clean paths
- **Benefit**: No more tangled connection lines

#### 3. **Increased Spacing**
```python
# Before:
"ranksep": "0.8",
"nodesep": "0.5",

# After:
"ranksep": "1.2",    # 50% more vertical spacing
"nodesep": "0.8",    # 60% more horizontal spacing
```
- **Benefit**: Nodes and labels have room to breathe

#### 4. **Improved Label Formatting**
- **Line breaks** in long component names (e.g., "Athena\ninference_\nresponses")
- **Shorter labels** on edges
- **Lambda** abbreviated as "λ" for space savings
- **Benefit**: Labels don't extend beyond their boxes

#### 5. **Visual Edge Differentiation**
- **Solid lines**: Primary data flow
- **Dashed lines**: Queries and async connections
- **Color coding**:
  - Blue (#0972D3): Training pipeline
  - Orange (#D45B07): Monitoring pipeline
  - Purple (#5B48D0): Dashboard pipeline
  - Purple (#7B1FA2): MLflow tracking
  - Red (#D32F2F): Alerts
- **Benefit**: Easy to trace different types of connections

#### 6. **Enhanced Cluster Margins**
```python
CL_AWS = {"margin": "20"}      # Outer AWS cloud boundary
CL_LANE = {"margin": "16"}     # Swim lanes
CL_SUB = {"margin": "12"}      # Sub-clusters
```
- **Benefit**: Clear visual separation between logical groups

#### 7. **Simplified Component Names**
- **Before**: "SageMaker MLflow App"
- **After**: "MLflow\nTracking"
- **Benefit**: Cleaner, more concise labels

## Architecture Overview

The improved diagram shows three main swim lanes (left to right):

### Lane 1: Training Pipeline (Blue)
```
S3 Data Lake → Athena → PySpark → XGBoost → Evaluate
                                      ↓
                                   MLflow (tracking)
                                      ↓
                                  SageMaker Endpoint
                                      ↓
                              SQS → λ Logger → Athena
```

### Lane 2: Inference Monitoring (Orange)
```
EventBridge (2 AM) → λ Drift Monitor → Evidently AI → MLflow
        ↓                    ↓                ↓
   Ground Truth         Query Athena      Reports
        ↓                    ↓                ↓
    Merge to         Compare baseline    SQS → λ Writer
   Inference         with current            ↓
                                        monitoring_responses
                                             ↓
                                        SNS Alerts
```

### Lane 3: Governance Dashboard (Purple)
```
EventBridge (3 AM) → λ Dataset Refresh → QuickSight Dashboard
                                              ↑
                                    Athena (inference + monitoring)
```

## Regenerating the Diagram

### Prerequisites

```bash
# Install Python packages
pip install diagrams

# Install Graphviz (macOS)
brew install graphviz

# Or on Linux
sudo apt-get install graphviz
```

### Generate Diagram

```bash
cd /path/to/monitoring/sagemaker-automated-drift-and-trend-monitoring
python3 docs/generate_architecture_diagram.py
```

### Output

```
✅  Architecture diagram → docs/guides/architecture_diagram.png
    ~250 KB

📊  Diagram improvements:
    - Left-to-right layout for better readability
    - Orthogonal edges to minimize overlaps
    - Increased spacing between nodes and clusters
    - Cleaner labels with line breaks
    - Dashed lines for cross-lane queries
    - Color-coded by lane (Blue/Orange/Purple)
```

## Manual Editing (Excalidraw)

If you need to make manual adjustments:

1. **Open Excalidraw**: https://excalidraw.com
2. **Import**: `docs/guides/architecture_diagram.excalidraw`
3. **Edit**: Adjust positions, add notes, modify colors
4. **Export**: Save as PNG or update .excalidraw file

### Tips for Manual Editing

- **Snap to grid**: Enable for alignment (Ctrl/Cmd + ')
- **Lock elements**: Lock background elements to prevent accidental moves
- **Layer management**: Use Ctrl/Cmd + [ or ] to reorder elements
- **Connector arrows**: Use "Bind to" feature for smart connections
- **Text wrapping**: Use Shift+Enter for line breaks in labels

## Comparison: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **Direction** | Top-Bottom | Left-Right |
| **Edge Style** | Spline (curved) | Ortho (right angles) |
| **Node Spacing** | 0.5 | 0.8 (+60%) |
| **Rank Spacing** | 0.8 | 1.2 (+50%) |
| **Margins** | Default | Custom per level |
| **Label Format** | Long single line | Multi-line breaks |
| **Edge Labels** | Verbose | Concise |
| **Color Coding** | Same color | Lane-specific |
| **Readability** | Overlaps present | No overlaps |

## Architecture Components

### Training Pipeline (Lane 1)
- **S3 Data Lake**: Source data storage
- **Athena (training_data)**: Training dataset queries
- **PySpark Processing**: Feature engineering (Glue)
- **XGBoost Training**: SageMaker training job
- **Evaluate**: Quality gate with model validation
- **MLflow Tracking**: Experiment tracking and model registry
- **SageMaker Endpoint**: Real-time inference API
- **λ Logger**: Async inference logging to Athena

### Inference Monitoring (Lane 2)
- **config.yaml**: Drift threshold configuration
- **Ground Truth Simulator**: Generate realistic labels (dev/test)
- **EventBridge**: Daily cron trigger (2 AM UTC)
- **λ Drift Monitor**: Orchestrates drift detection
- **Evidently AI**: Statistical drift analysis
- **MLflow Monitoring**: Store drift reports and metrics
- **λ Writer**: Batch write monitoring results
- **SNS Alerts**: Email/SMS notifications on drift
- **CloudWatch**: Logs and metrics

### Governance Dashboard (Lane 3)
- **EventBridge**: Daily refresh trigger (3 AM UTC)
- **λ Dataset Refresh**: SPICE ingestion automation
- **QuickSight Dashboard**: Interactive visualizations
- **Athena Connections**: inference_responses + monitoring_responses

## Automated Refresh Schedule

```
┌─────────────────────────────────────────────────┐
│  2:00 AM UTC  │  Drift Monitoring Lambda runs   │
│               │  (Evidently + MLflow logging)   │
├───────────────┼─────────────────────────────────┤
│  3:00 AM UTC  │  QuickSight Dataset Refresh     │
│               │  (SPICE ingestion)              │
├───────────────┼─────────────────────────────────┤
│  Morning      │  Dashboard shows latest data    │
└─────────────────────────────────────────────────┘
```

## Troubleshooting

### Overlapping Elements Still Present

If you still see overlaps after regenerating:

1. **Increase spacing further**:
   ```python
   "ranksep": "1.5",  # Even more vertical spacing
   "nodesep": "1.0",  # Even more horizontal spacing
   ```

2. **Add invisible edges** to control layout:
   ```python
   node1 >> Edge(style="invis") >> node2
   ```

3. **Use constraint="false"** on problematic edges:
   ```python
   node1 >> Edge(constraint="false") >> node2
   ```

### Labels Too Long

Shorten or add more line breaks:
```python
# Before
Custom("Very Long Component Name Here", ICON)

# After
Custom("Component\nName", ICON)
```

### Edges Crossing

Try reordering node definitions or using:
```python
GRAPH["splines"] = "polyline"  # Even cleaner than ortho
```

## Future Enhancements

Potential improvements for the next iteration:

1. **Interactive HTML diagram** using D3.js or Mermaid
2. **Swimlane annotations** with timing/SLAs
3. **Cost indicators** per component
4. **Data flow metrics** (volume, frequency)
5. **Alternative layouts** for different audiences:
   - Technical deep-dive (current)
   - Executive overview (simplified)
   - Operational runbook (step-by-step)

## Related Documentation

- **Diagram Generator**: `docs/generate_architecture_diagram.py`
- **Inference Monitoring**: `docs/generate_inference_monitoring_diagram.py`
- **MLflow + Evidently**: `docs/generate_mlflow_evidently_diagram.py`
- **QuickSight Refresh**: `QUICKSIGHT_REFRESH_AUTOMATION.md`
- **MLflow Run IDs**: `MLFLOW_RUN_ID_FIX.md`

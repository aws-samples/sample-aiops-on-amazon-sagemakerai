# QuickSight Governance Dashboard for Inference Monitoring

This directory contains the QuickSight governance dashboard setup for visualizing inference monitoring data from the Athena data lake.

## Overview

The governance dashboard provides visual insights into:
- Prediction volume trends (fraud vs non-fraud over time)
- Fraud probability distribution across inferences
- Prediction accuracy breakdown (Correct / Incorrect / Pending ground truth)
- Risk tier distribution (High / Medium / Low / Minimal)
- Inference latency trends
- Total inference KPIs

## Quick Start

### Option 1: Notebook (Recommended)

The notebook creates everything programmatically — no manual QuickSight UI steps required:

```bash
# Open in Jupyter
notebooks/governance_dashboard.ipynb
```

Run all cells sequentially. The notebook will:
1. Verify QuickSight subscription and Athena data
2. Create Athena data source in QuickSight
3. Create dataset from `inference_responses` table (with calculated fields: `prediction_accuracy`, `risk_tier`)
4. Create analysis with 6 visuals via Definition API
5. Publish dashboard

### Option 2: CLI Script

```bash
# Create data source and dataset
python -m src.governance.setup_quicksight_governance --create

# Refresh dataset (SPICE mode only)
python -m src.governance.setup_quicksight_governance --refresh

# Delete all resources
python -m src.governance.setup_quicksight_governance --delete
```

Note: The CLI script creates the data source and dataset but the analysis/dashboard
require manual creation in the UI. Use the notebook for fully automated setup.

## Prerequisites

- AWS account with QuickSight Enterprise subscription (Definition API requires Enterprise)
- Inference monitoring pipeline running (data in `inference_responses` Athena table)
- IAM permissions for QuickSight, Athena, and S3

## Data Source

The dashboard uses a `RelationalTable` pointing at the `inference_responses` Athena table
(auto-discovers columns — no manual column list). Calculated fields are added via `LogicalTableMap`:
- `prediction_accuracy`: Correct / Incorrect / Pending (based on ground truth availability)
- `risk_tier`: High Risk / Medium Risk / Low Risk / Minimal Risk (based on `probability_fraud`)

Import mode is `DIRECT_QUERY` — the dashboard always shows live data from Athena, no refresh needed.

## Visuals Created

The notebook creates 6 visuals automatically:

| # | Visual | Type | Fields |
|---|--------|------|--------|
| 1 | Prediction Volume Over Time | Line chart | `request_timestamp` (day) × count |
| 2 | Fraud Probability Distribution | Histogram | `probability_fraud`, 20 bins |
| 3 | Prediction Accuracy Breakdown | Donut chart | `prediction_accuracy` × count |
| 4 | Risk Tier Distribution | Bar chart | `risk_tier` × count |
| 5 | Inference Latency Trend | Line chart | `request_timestamp` (day) × avg `inference_latency_ms` |
| 6 | Total Inferences | KPI card | count of `inference_id` |

## Resource IDs

| Resource | ID |
|----------|-----|
| Data Source | `fraud-governance-athena-datasource` |
| Dataset | `fraud-governance-inference-dataset` |
| Analysis | `fraud-governance-analysis` |
| Dashboard | `fraud-governance-dashboard` |

## Sharing & Embedding

After the dashboard is published:

- Share with users: Dashboard → Share → Add users/groups
- Generate embed URL: Cell 9 in the notebook
- Schedule email reports: Dashboard → Share → Schedule email report

## Troubleshooting

**Dataset not appearing**: Verify `inference_responses` table has data:
```sql
SELECT COUNT(*) FROM fraud_detection.inference_responses
```

**No data in dashboard**: Check date filters, verify ground truth has been applied.

**Permission denied**: Ensure your IAM role has QuickSight permissions and your user is added to QuickSight.

## Files

- `setup_quicksight_governance.py` — CLI script for data source/dataset creation
- `README.md` — This file
- `../../notebooks/governance_dashboard.ipynb` — Full automated setup notebook

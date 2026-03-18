# QuickSight Dashboard for Model Monitoring

This directory contains scripts to set up QuickSight dashboards for visualizing model monitoring results.

## Overview

The QuickSight dashboard provides visual insights into:
- Model drift detection trends
- Performance metrics over time
- Top drifted features
- Alarm status and severity
- Data quality indicators

## Quick Start

### 1. Prerequisites

- AWS account with QuickSight subscription
- Monitoring pipeline running and generating data
- Data in `monitoring_results` Athena table

### 2. Setup QuickSight Resources

Run the setup script to create data source and dataset:

```bash
python src/quicksight/setup_quicksight_monitoring.py --create
```

This will create:
- ✅ Athena data source
- ✅ Dataset from `monitoring_results` table
- ℹ️ Analysis and Dashboard (require manual creation in UI)

### 3. Create Dashboard in QuickSight UI

Since QuickSight API has limitations for creating complex visualizations, follow these steps in the UI:

#### Step 1: Open QuickSight

1. Navigate to: https://quicksight.aws.amazon.com/
2. Sign in with your AWS credentials

#### Step 2: Create Analysis

1. Click **Datasets** in the left sidebar
2. Find **Fraud Monitoring Results** dataset
3. Click the three dots (⋮) → **Create analysis**

#### Step 3: Add Visualizations

**Visual 1: F1 Score Trend (Line Chart)**
- Visual type: Line chart
- X-axis: `analysis_timestamp` (aggregated by day)
- Value: `f1_score` (Average)
- Add reference line at y=0.70 (threshold)
- Title: "Model F1 Score Over Time"

**Visual 2: Drifted Features Count (Bar Chart)**
- Visual type: Vertical bar chart
- X-axis: `analysis_timestamp` (aggregated by day)
- Value: `drifted_features_count` (Average)
- Add reference line at y=5 (threshold)
- Title: "Average Drifted Features per Day"

**Visual 3: Top Drifted Features (Table)**
- Visual type: Pivot table
- Rows: `top_drifted_feature_1`, `top_drifted_feature_2`, etc.
- Values: Count (aggregated)
- Sort by: Count descending
- Title: "Most Frequently Drifted Features"

**Visual 4: Model Performance KPIs (KPI Cards)**
- Visual type: KPI
- Create 4 separate KPIs:
  - F1 Score (latest)
  - Precision (latest)
  - Recall (latest)
  - ROC-AUC (latest)
- Add conditional formatting (red if below threshold)
- Title: "Current Model Performance"

**Visual 5: Drift Detection Heatmap**
- Visual type: Heat map
- Rows: `analysis_timestamp` (by day)
- Columns: `endpoint_name`
- Values: `dataset_drift_detected` (Sum)
- Color: Red (high drift) to Green (no drift)
- Title: "Drift Detection Heatmap"

#### Step 4: Publish Dashboard

1. Click **Share** → **Publish dashboard**
2. Name: `Fraud Detection Monitoring`
3. ID: `fraud-monitoring-dashboard`
4. Click **Publish**

#### Step 5: Configure Dashboard Settings

1. Open the published dashboard
2. Click **Settings** (gear icon)
3. Enable:
   - Ad-hoc filtering
   - Export to CSV
   - Sheet controls
4. Save settings

## Refreshing Dashboard

### Automatic Refresh (for SPICE datasets)

If using SPICE import mode, schedule automatic refresh:

```bash
python src/quicksight/setup_quicksight_monitoring.py --refresh
```

Or in QuickSight UI:
1. Go to **Datasets** → **Fraud Monitoring Results**
2. Click **Schedule refresh**
3. Set schedule (e.g., daily at 3 AM UTC)

### Manual Refresh

For DIRECT_QUERY mode (recommended):
- Dashboard automatically queries latest data
- No refresh needed

## Dashboard Features

### Filters

Add dashboard-level filters:
- **Date Range**: Filter by analysis_timestamp
- **Endpoint**: Filter by endpoint_name
- **Drift Status**: Show only drifted or non-drifted results
- **Severity**: Filter by alarm_severity

### Parameters

Create parameters for dynamic filtering:
- `analysis_days`: Number of days to show (default: 30)
- `f1_threshold`: F1 score threshold (default: 0.70)
- `drift_threshold`: Drift threshold (default: 5)

### Calculated Fields

The dataset includes pre-calculated fields:
- `drift_severity`: HIGH/MEDIUM/LOW based on drift_share
- `performance_status`: GOOD/WARNING/CRITICAL based on f1_score

Create additional calculated fields:
```
// Performance degradation percentage
f1_degradation_pct = (baseline_f1_score - f1_score) / baseline_f1_score * 100

// Days since last analysis
days_since_analysis = dateDiff(analysis_timestamp, now(), 'DD')

// Drift alert status
drift_alert = ifelse(drifted_features_count >= 5, 'ALERT', 'OK')
```

## Sharing Dashboard

### Share with Users

1. Open dashboard
2. Click **Share** → **Share dashboard**
3. Add users/groups
4. Set permissions:
   - Viewer: Can view dashboard
   - Co-owner: Can edit dashboard

### Embed in Application

For embedding dashboard in web application:

1. Enable anonymous embedding (if needed):
   ```bash
   aws quicksight update-dashboard-published-version \
     --aws-account-id YOUR_ACCOUNT_ID \
     --dashboard-id fraud-monitoring-dashboard \
     --version-number 1
   ```

2. Generate embed URL:
   ```python
   import boto3

   quicksight = boto3.client('quicksight')

   response = quicksight.generate_embed_url_for_anonymous_user(
       AwsAccountId='YOUR_ACCOUNT_ID',
       Namespace='default',
       AuthorizedResourceArns=[
           'arn:aws:quicksight:REGION:ACCOUNT:dashboard/fraud-monitoring-dashboard'
       ],
       ExperienceConfiguration={
           'Dashboard': {
               'InitialDashboardId': 'fraud-monitoring-dashboard'
           }
       }
   )

   embed_url = response['EmbedUrl']
   ```

3. Embed in HTML:
   ```html
   <iframe
       src="{embed_url}"
       width="100%"
       height="800px"
       frameborder="0">
   </iframe>
   ```

## Troubleshooting

### Dataset Not Appearing

**Problem**: Dataset not visible in QuickSight

**Solution**:
1. Check Athena table exists: `SELECT * FROM fraud_detection.monitoring_results LIMIT 10`
2. Verify QuickSight has Athena permissions
3. Recreate data source: `python setup_quicksight_monitoring.py --create`

### No Data in Dashboard

**Problem**: Dashboard shows "No data"

**Solution**:
1. Run monitoring pipeline to generate data
2. Check date filters (may be filtering out all data)
3. Query Athena directly to verify data exists
4. Refresh dataset (if using SPICE)

### Permission Denied Errors

**Problem**: "Permission denied" when accessing dashboard

**Solution**:
1. Check IAM role has QuickSight permissions
2. Add your user to QuickSight:
   - QuickSight console → Manage QuickSight → Invite users
3. Share dashboard with your user
4. Verify Athena permissions in data source

### Visualizations Not Loading

**Problem**: Visuals show loading spinner indefinitely

**Solution**:
1. Check Athena query limits (may be timing out)
2. Add date range filter to reduce data
3. Check CloudWatch Logs for Athena errors
4. Simplify visual (remove complex calculated fields)

## Advanced Configuration

### Custom Themes

Apply custom branding:

1. Create theme JSON:
   ```json
   {
     "ThemeConfiguration": {
       "DataColorPalette": {
         "Colors": ["#FF0000", "#00FF00", "#0000FF"]
       },
       "UIColorPalette": {
         "PrimaryForeground": "#000000",
         "PrimaryBackground": "#FFFFFF"
       }
     }
   }
   ```

2. Upload theme:
   ```bash
   aws quicksight create-theme \
     --aws-account-id YOUR_ACCOUNT_ID \
     --theme-id monitoring-theme \
     --name "Monitoring Theme" \
     --configuration file://theme.json
   ```

3. Apply to dashboard in UI

### Email Reports

Schedule email reports:

1. Open dashboard
2. Click **Share** → **Schedule email report**
3. Configure:
   - Recipients: email addresses
   - Frequency: Daily/Weekly/Monthly
   - Time: 8 AM UTC
   - Format: PDF
4. Save schedule

### Alerts

Set up threshold alerts:

1. Create calculated field for alert condition
2. Use conditional formatting (red background)
3. Add filter to show only alerts
4. Schedule email report for filtered view

## Maintenance

### Update Dashboard

When monitoring schema changes:

1. Update dataset:
   ```bash
   python src/quicksight/setup_quicksight_monitoring.py --create
   ```

2. Refresh visuals in analysis:
   - Open analysis
   - Click **Refresh** on each visual
   - Republish dashboard

3. Update calculated fields if needed

### Monitor Usage

Track dashboard usage:

1. QuickSight console → Usage & insights
2. View:
   - Active users
   - Dashboard views
   - Query performance
   - SPICE usage (if applicable)

### Optimize Performance

For large datasets:

1. **Use DIRECT_QUERY** for real-time data
2. **Use SPICE** for faster performance (import mode)
3. **Add filters** to limit date range
4. **Aggregate data** in Athena views before creating dataset
5. **Partition tables** in Athena by date

## Resources

- **QuickSight Documentation**: https://docs.aws.amazon.com/quicksight/
- **Dashboard Best Practices**: https://docs.aws.amazon.com/quicksight/latest/user/best-practices.html
- **Athena Integration**: https://docs.aws.amazon.com/quicksight/latest/user/create-a-data-set-athena.html
- **Embedding Guide**: https://docs.aws.amazon.com/quicksight/latest/user/embedded-analytics.html

## Example Dashboard

Once set up, your dashboard will look like:

```
┌─────────────────────────────────────────────────────────────┐
│  Fraud Detection Monitoring                         [Filter]│
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │  0.752  │ │  0.689  │ │  0.821  │ │  0.876  │          │
│  │ F1 Score│ │Precision│ │ Recall  │ │ ROC-AUC │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │          Model F1 Score Over Time                       ││
│  │  0.80  ──────────────────────────────                   ││
│  │  0.75  ─────╱─────────╲─────────────                    ││
│  │  0.70  ─────────────────────────────── (Threshold)      ││
│  │        Jan 1   Jan 15   Feb 1   Feb 15                  ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────────────┐│
│  │ Drifted Features     │  │ Top Drifted Features         ││
│  │                      │  │ ┌─────────────────┬────────┐ ││
│  │  █ █ █               │  │ │ Feature         │ Count  │ ││
│  │  █ █ █ █             │  │ ├─────────────────┼────────┤ ││
│  │  █ █ █ █ █           │  │ │ transaction_amt │   23   │ ││
│  │  █ █ █ █ █ █         │  │ │ distance_km     │   18   │ ││
│  └──────────────────────┘  │ │ velocity_score  │   15   │ ││
│                             └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Summary

The QuickSight dashboard provides comprehensive monitoring insights with:

✅ Real-time data from Athena
✅ Interactive filtering and drill-down
✅ Performance trends and KPIs
✅ Drift detection visualization
✅ Email reports and alerts
✅ Sharing and embedding capabilities

For questions or issues, refer to the troubleshooting section or AWS QuickSight documentation.

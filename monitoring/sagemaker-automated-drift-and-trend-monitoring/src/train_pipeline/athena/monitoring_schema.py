"""
Athena Iceberg table schema for monitoring_responses.

Stores drift metrics computed by lambda_drift_monitor.py.
Joinable from QuickSight to create drift trends over multiple model versions.
"""

from src.config.config import ATHENA_DATABASE, DATA_S3_BUCKET, DATA_S3_PREFIX

_MONITORING_S3 = f"s3://{DATA_S3_BUCKET}/{DATA_S3_PREFIX}monitoring_responses/"

CREATE_MONITORING_RESULTS_TABLE = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.monitoring_responses (
    monitoring_run_id STRING,
    monitoring_timestamp TIMESTAMP,
    endpoint_name STRING,
    model_version STRING,

    -- Data drift metrics (Evidently DataDriftPreset)
    data_drift_detected BOOLEAN,
    drifted_columns_count INT,
    drifted_columns_share DOUBLE,
    features_analyzed INT,
    data_sample_size INT,

    -- Model drift metrics (Evidently ClassificationPreset)
    model_drift_detected BOOLEAN,
    baseline_roc_auc DOUBLE,
    current_roc_auc DOUBLE,
    roc_auc_degradation DOUBLE,
    roc_auc_degradation_pct DOUBLE,
    accuracy DOUBLE,
    precision DOUBLE,
    recall DOUBLE,
    f1_score DOUBLE,
    model_sample_size INT,

    -- Per-feature drift scores (JSON blob)
    per_feature_drift_scores STRING,

    -- Evidently report S3 path
    evidently_report_s3_path STRING,

    -- MLflow run ID for cross-referencing
    mlflow_run_id STRING,

    -- Alert info
    alert_sent BOOLEAN,

    -- Metadata
    detection_engine STRING,
    created_at TIMESTAMP
)
PARTITIONED BY (day(monitoring_timestamp))
LOCATION '{_MONITORING_S3}'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format' = 'parquet'
)
"""

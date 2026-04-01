"""Configuration module for the drift monitoring pipeline.

Loads configuration from config.yaml and .env file overrides.
Exposes all configuration constants as module-level variables.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file: src/config/config.py)
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent.parent

# Load .env (environment-specific overrides take precedence)
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Load config.yaml (central defaults)
# ---------------------------------------------------------------------------
_CONFIG_YAML_PATH = _CONFIG_DIR / "config.yaml"
_yaml_cfg: dict = {}
if _CONFIG_YAML_PATH.exists():
    with open(_CONFIG_YAML_PATH, "r") as _f:
        _yaml_cfg = yaml.safe_load(_f) or {}


def _get(yaml_section: str, yaml_key: str, env_var: str, default=None):
    """Return env-var override → YAML value → default, in that priority."""
    env_val = os.environ.get(env_var)
    if env_val is not None:
        return env_val
    section = _yaml_cfg.get(yaml_section, {})
    if isinstance(section, dict) and yaml_key in section:
        return section[yaml_key]
    return default


# ===================================================================
# AWS Configuration
# ===================================================================
AWS_DEFAULT_REGION: str = _get("aws", "region", "AWS_DEFAULT_REGION", "us-east-1")

# ===================================================================
# SageMaker Configuration
# ===================================================================
SAGEMAKER_EXEC_ROLE: str = _get(
    "sagemaker", "exec_role", "SAGEMAKER_EXEC_ROLE", ""
)
SAGEMAKER_TRAINING_INSTANCE: str = _get(
    "sagemaker", "training_instance", "SAGEMAKER_TRAINING_INSTANCE", "ml.m5.xlarge"
)
SAGEMAKER_TRAINING_VOLUME_SIZE: int = int(
    _get("sagemaker", "training_volume_size", "SAGEMAKER_TRAINING_VOLUME_SIZE", "30")
)
SERVERLESS_MEMORY_SIZE: int = int(
    _get("sagemaker", "serverless_memory_size", "SERVERLESS_MEMORY_SIZE", "2048")
)
SERVERLESS_MAX_CONCURRENCY: int = int(
    _get("sagemaker", "serverless_max_concurrency", "SERVERLESS_MAX_CONCURRENCY", "5")
)
BATCH_TRANSFORM_INSTANCE: str = _get(
    "sagemaker", "batch_transform_instance", "BATCH_TRANSFORM_INSTANCE", "ml.m5.xlarge"
)
BATCH_TRANSFORM_MAX_CONCURRENT: int = int(
    _get("sagemaker", "batch_transform_max_concurrent", "BATCH_TRANSFORM_MAX_CONCURRENT", "4")
)

# ===================================================================
# MLflow Settings
# ===================================================================
MLFLOW_TRACKING_URI: str = _get(
    "mlflow", "tracking_uri", "MLFLOW_TRACKING_URI", ""
)
MLFLOW_TRACKING_BROWSER_URL: str = _get(
    "mlflow", "tracking_browser_url", "MLFLOW_TRACKING_BROWSER_URL", ""
)
MLFLOW_EXPERIMENT_NAME: str = _get(
    "mlflow", "experiment_name", "MLFLOW_EXPERIMENT_NAME",
    "credit-card-fraud-detection-training",
)
MLFLOW_INFERENCE_EXPERIMENT_NAME: str = _get(
    "mlflow", "inference_experiment_name", "MLFLOW_INFERENCE_EXPERIMENT_NAME",
    "credit-card-fraud-detection-inference",
)
MLFLOW_BATCH_EXPERIMENT_NAME: str = _get(
    "mlflow", "batch_experiment_name", "MLFLOW_BATCH_EXPERIMENT_NAME",
    "credit-card-fraud-detection-batch",
)
MLFLOW_MONITORING_EXPERIMENT_NAME: str = _get(
    "mlflow", "monitoring_experiment_name", "MLFLOW_MONITORING_EXPERIMENT_NAME",
    "credit-card-fraud-detection-monitoring",
)
MLFLOW_MODEL_NAME: str = _get(
    "mlflow", "model_name", "MLFLOW_MODEL_NAME", "xgboost-fraud-detector"
)

# ===================================================================
# Athena Settings
# ===================================================================
ATHENA_DATABASE: str = _get("athena", "database", "ATHENA_DATABASE", "fraud_detection")
ATHENA_WORKGROUP: str = _get("athena", "workgroup", "ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT_S3: str = _get("athena", "output_s3", "ATHENA_OUTPUT_S3", "")
ATHENA_QUERY_TIMEOUT: int = int(
    _get("athena", "query_timeout", "ATHENA_QUERY_TIMEOUT", "300")
)

# Athena table names
ATHENA_TRAINING_TABLE: str = _get(
    "athena", "training_table", "ATHENA_TRAINING_TABLE", "training_data"
)
ATHENA_INFERENCE_TABLE: str = _get(
    "athena", "inference_table", "ATHENA_INFERENCE_TABLE", "inference_responses"
)
ATHENA_GROUND_TRUTH_TABLE: str = _get(
    "athena", "ground_truth_table", "ATHENA_GROUND_TRUTH_TABLE", "ground_truth"
)
ATHENA_GROUND_TRUTH_UPDATES_TABLE: str = _get(
    "athena", "ground_truth_updates_table", "ATHENA_GROUND_TRUTH_UPDATES_TABLE",
    "ground_truth_updates",
)
ATHENA_DRIFTED_DATA_TABLE: str = _get(
    "athena", "drifted_data_table", "ATHENA_DRIFTED_DATA_TABLE", "drifted_data"
)
ATHENA_MONITORING_RESPONSES_TABLE: str = _get(
    "athena", "monitoring_responses_table", "ATHENA_MONITORING_RESPONSES_TABLE",
    "monitoring_responses"
)

# ===================================================================
# S3 Paths
# ===================================================================
DATA_S3_BUCKET: str = _get("s3", "bucket", "DATA_S3_BUCKET", "")
DATA_S3_PREFIX: str = _get("s3", "prefix", "DATA_S3_PREFIX", "fraud-detection/")


def _s3_path(yaml_key: str, env_var: str, suffix: str) -> str:
    """Build S3 path from config or derive from bucket + prefix + suffix."""
    val = _get("s3", yaml_key, env_var, "")
    if val:
        return val
    if DATA_S3_BUCKET:
        return f"s3://{DATA_S3_BUCKET}/{DATA_S3_PREFIX}{suffix}"
    return ""


S3_TRAINING_DATA: str = _s3_path("training_data", "S3_TRAINING_DATA", "training_data/")
S3_GROUND_TRUTH: str = _s3_path("ground_truth", "S3_GROUND_TRUTH", "ground_truth/")
S3_INFERENCE_RESPONSES: str = _s3_path(
    "inference_responses", "S3_INFERENCE_RESPONSES", "inference_responses/"
)
S3_DRIFTED_DATA: str = _s3_path("drifted_data", "S3_DRIFTED_DATA", "drifted_data/")
S3_GROUND_TRUTH_UPDATES: str = _s3_path(
    "ground_truth_updates", "S3_GROUND_TRUTH_UPDATES", "ground_truth_updates/"
)
S3_MODEL_ARTIFACTS: str = _s3_path(
    "model_artifacts", "S3_MODEL_ARTIFACTS", "model_artifacts/"
)
S3_TRAINING_DATA_EXPORT: str = _s3_path(
    "training_data_export", "S3_TRAINING_DATA_EXPORT", "training_data_export/"
)
S3_BATCH_TRANSFORM_INPUT: str = _s3_path(
    "batch_transform_input", "S3_BATCH_TRANSFORM_INPUT", "batch_transform/input/"
)
S3_BATCH_TRANSFORM_OUTPUT: str = _s3_path(
    "batch_transform_output", "S3_BATCH_TRANSFORM_OUTPUT", "batch_transform/output/"
)

# ===================================================================
# SQS Configuration
# ===================================================================
SQS_QUEUE_NAME: str = _get("sqs", "queue_name", "SQS_QUEUE_NAME", "fraud-inference-logs")
SQS_QUEUE_URL: str = _get("sqs", "queue_url", "SQS_QUEUE_URL", "")
MONITORING_SQS_QUEUE_NAME: str = _get(
    "sqs", "monitoring_queue_name", "MONITORING_SQS_QUEUE_NAME", "fraud-monitoring-results"
)
MONITORING_SQS_QUEUE_URL: str = _get(
    "sqs", "monitoring_queue_url", "MONITORING_SQS_QUEUE_URL", ""
)

# ===================================================================
# Inference Logging
# ===================================================================
ENABLE_ATHENA_LOGGING: bool = (
    _get("inference_logging", "enable_athena_logging", "ENABLE_ATHENA_LOGGING", "true")
    .lower() in ("true", "1", "yes")
)
INFERENCE_LOG_BATCH_SIZE: int = int(
    _get("inference_logging", "batch_size", "INFERENCE_LOG_BATCH_SIZE", "100")
)
INFERENCE_LOG_FLUSH_INTERVAL: int = int(
    _get("inference_logging", "flush_interval", "INFERENCE_LOG_FLUSH_INTERVAL", "300")
)

# ===================================================================
# Inference Confidence Thresholds
# ===================================================================
HIGH_CONFIDENCE_THRESHOLD: float = float(
    _get("inference", "high_confidence_threshold", "HIGH_CONFIDENCE_THRESHOLD", "0.9")
)
LOW_CONFIDENCE_LOWER: float = float(
    _get("inference", "low_confidence_lower", "LOW_CONFIDENCE_LOWER", "0.4")
)
LOW_CONFIDENCE_UPPER: float = float(
    _get("inference", "low_confidence_upper", "LOW_CONFIDENCE_UPPER", "0.6")
)

# ===================================================================
# Lambda Configuration
# ===================================================================
LAMBDA_LOGGER_NAME: str = _get(
    "lambda", "logger_name", "LAMBDA_LOGGER_NAME", "fraud-inference-log-consumer"
)
LAMBDA_MONITORING_WRITER_NAME: str = _get(
    "lambda", "monitoring_writer_name", "LAMBDA_MONITORING_WRITER_NAME",
    "fraud-monitoring-results-writer"
)
LAMBDA_EXEC_ROLE: str = _get("lambda", "exec_role", "LAMBDA_EXEC_ROLE", "")

# ===================================================================
# Data File Paths
# ===================================================================
DATA_DIR: Path = _PROJECT_ROOT / "data"

# Local CSV paths (used only by upload_data_to_s3.py)
CSV_TRAINING_DATA: Path = _PROJECT_ROOT / _get(
    "data", "csv_training_data", "CSV_TRAINING_DATA",
    "data/creditcard_predictions_final.csv",
)
CSV_GROUND_TRUTH: Path = _PROJECT_ROOT / _get(
    "data", "csv_ground_truth", "CSV_GROUND_TRUTH",
    "data/creditcard_ground_truth.csv",
)
CSV_DRIFTED_DATA: Path = _PROJECT_ROOT / _get(
    "data", "csv_drifted_data", "CSV_DRIFTED_DATA",
    "data/creditcard_drifted.csv",
)

# S3 data paths — pipeline and Athena code should use these
_s3_data_base = f"s3://{DATA_S3_BUCKET}/{DATA_S3_PREFIX}data" if DATA_S3_BUCKET else ""

S3_TRAINING_DATA: str = _get(
    "data", "s3_training_data", "S3_TRAINING_DATA",
    f"{_s3_data_base}/creditcard_predictions_final.csv" if _s3_data_base else "",
)
S3_GROUND_TRUTH: str = _get(
    "data", "s3_ground_truth", "S3_GROUND_TRUTH",
    f"{_s3_data_base}/creditcard_ground_truth.csv" if _s3_data_base else "",
)
S3_DRIFTED_DATA: str = _get(
    "data", "s3_drifted_data", "S3_DRIFTED_DATA",
    f"{_s3_data_base}/creditcard_drifted.csv" if _s3_data_base else "",
)

# ===================================================================
# Training Features
# ===================================================================
_yaml_features = _yaml_cfg.get("training", {}).get("features") if isinstance(
    _yaml_cfg.get("training"), dict
) else None

TRAINING_FEATURES: list[str] = _yaml_features or [
    "transaction_hour",
    "transaction_day_of_week",
    "transaction_amount",
    "customer_age",
    "customer_gender",
    "distance_from_home_km",
    "merchant_category_code",
    "chip_transaction",
    "num_transactions_24h",
    "credit_limit",
    "available_credit_ratio",
    "avg_transaction_amount_7d",
    "avg_transaction_amount_30d",
    "transaction_amount_zscore",
    "is_weekend",
    "is_night",
    "merchant_risk_score",
    "customer_tenure_days",
    "num_cards",
    "card_present",
    "recurring_merchant",
    "distance_from_last_transaction_km",
    "time_since_last_transaction_min",
    "foreign_transaction",
    "high_risk_country",
    "velocity_1h",
    "velocity_24h",
    "amount_to_limit_ratio",
    "digital_wallet",
    "authentication_method",
]

TARGET_COLUMN: str = _get("training", "target_column", "TARGET_COLUMN", "is_fraud")

TEST_SIZE: float = float(
    _get("training", "test_size", "TEST_SIZE", "0.2")
)
RANDOM_STATE: int = int(
    _get("training", "random_state", "RANDOM_STATE", "42")
)

# XGBoost parameters
_yaml_xgb = _yaml_cfg.get("training", {}).get("xgboost_params") if isinstance(
    _yaml_cfg.get("training"), dict
) else None

XGBOOST_PARAMS: dict = _yaml_xgb or {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}

# ===================================================================
# Drift Thresholds
# ===================================================================
_drift_cfg = _yaml_cfg.get("drift_thresholds", {}) if isinstance(
    _yaml_cfg.get("drift_thresholds"), dict
) else {}

# PSI (Population Stability Index) thresholds
PSI_THRESHOLD_NO_DRIFT: float = float(
    _drift_cfg.get("psi_no_drift", os.environ.get("PSI_THRESHOLD_NO_DRIFT", "0.1"))
)
PSI_THRESHOLD_MODERATE_DRIFT: float = float(
    _drift_cfg.get("psi_moderate_drift", os.environ.get("PSI_THRESHOLD_MODERATE_DRIFT", "0.2"))
)
PSI_THRESHOLD_SIGNIFICANT_DRIFT: float = float(
    _drift_cfg.get("psi_significant_drift", os.environ.get("PSI_THRESHOLD_SIGNIFICANT_DRIFT", "0.25"))
)

# KS Test significance level
KS_TEST_SIGNIFICANCE: float = float(
    _drift_cfg.get("ks_significance", os.environ.get("KS_TEST_SIGNIFICANCE", "0.05"))
)

# Model performance degradation threshold (percentage)
PERFORMANCE_DEGRADATION_THRESHOLD: float = float(
    _drift_cfg.get(
        "performance_degradation_pct",
        os.environ.get("PERFORMANCE_DEGRADATION_THRESHOLD", "0.05"),
    )
)

# Minimum ROC-AUC before alerting
MIN_ROC_AUC_THRESHOLD: float = float(
    _drift_cfg.get("min_roc_auc", os.environ.get("MIN_ROC_AUC_THRESHOLD", "0.85"))
)

# ===================================================================
# Evidently Statistical Test Settings
# ===================================================================
_evidently_cfg = _yaml_cfg.get("evidently", {}) if isinstance(
    _yaml_cfg.get("evidently"), dict
) else {}

EVIDENTLY_DRIFT_METHOD: str = str(
    _evidently_cfg.get("drift_method", os.environ.get("EVIDENTLY_DRIFT_METHOD", "psi"))
)
EVIDENTLY_DRIFT_THRESHOLD: float = float(
    _evidently_cfg.get(
        "drift_threshold", os.environ.get("EVIDENTLY_DRIFT_THRESHOLD", "0.1")
    )
)
EVIDENTLY_STATTEST_THRESHOLD: float = float(
    _evidently_cfg.get(
        "stattest_threshold", os.environ.get("EVIDENTLY_STATTEST_THRESHOLD", "0.05")
    )
)
EVIDENTLY_NUM_STATTEST: str = str(
    _evidently_cfg.get(
        "num_stattest", os.environ.get("EVIDENTLY_NUM_STATTEST", "ks")
    )
)
EVIDENTLY_CAT_STATTEST: str = str(
    _evidently_cfg.get(
        "cat_stattest", os.environ.get("EVIDENTLY_CAT_STATTEST", "chisquare")
    )
)
EVIDENTLY_BASELINE_SAMPLES: int = int(
    _evidently_cfg.get(
        "baseline_samples", os.environ.get("EVIDENTLY_BASELINE_SAMPLES", "5000")
    )
)
EVIDENTLY_BASELINE_S3_PREFIX: str = str(
    _evidently_cfg.get(
        "baseline_s3_prefix", os.environ.get("EVIDENTLY_BASELINE_S3_PREFIX", "fraud-detection/baselines/")
    )
)

# ===================================================================
# SNS Alerting
# ===================================================================
SNS_TOPIC_NAME: str = _get("sns", "topic_name", "SNS_TOPIC_NAME", "fraud-drift-alerts")
SNS_TOPIC_ARN: str = _get("sns", "topic_arn", "SNS_TOPIC_ARN", "")

# ===================================================================
# Monitoring Schedule
# ===================================================================
MONITORING_SCHEDULE_EXPRESSION: str = _get(
    "monitoring", "schedule_expression", "MONITORING_SCHEDULE_EXPRESSION",
    "rate(1 day)",
)
MONITORING_LOOKBACK_DAYS: int = int(
    _get("monitoring", "lookback_days", "MONITORING_LOOKBACK_DAYS", "30")
)

# ===================================================================
# QuickSight Governance Dashboard
# ===================================================================
# Resource IDs
QUICKSIGHT_DATASOURCE_ID: str = _get(
    "quicksight", "datasource_id", "QUICKSIGHT_DATASOURCE_ID",
    "fraud-governance-athena-datasource"
)
QUICKSIGHT_INFERENCE_DATASET_ID: str = _get(
    "quicksight", "inference_dataset_id", "QUICKSIGHT_INFERENCE_DATASET_ID",
    "fraud-governance-inference-dataset"
)
QUICKSIGHT_DRIFT_DATASET_ID: str = _get(
    "quicksight", "drift_dataset_id", "QUICKSIGHT_DRIFT_DATASET_ID",
    "fraud-governance-drift-dataset"
)
QUICKSIGHT_FEATURE_DRIFT_DATASET_ID: str = _get(
    "quicksight", "feature_drift_dataset_id", "QUICKSIGHT_FEATURE_DRIFT_DATASET_ID",
    "fraud-governance-feature-drift-dataset"
)
QUICKSIGHT_ANALYSIS_ID: str = _get(
    "quicksight", "analysis_id", "QUICKSIGHT_ANALYSIS_ID",
    "fraud-governance-analysis"
)
QUICKSIGHT_DASHBOARD_ID: str = _get(
    "quicksight", "dashboard_id", "QUICKSIGHT_DASHBOARD_ID",
    "fraud-governance-dashboard"
)

# Display names
QUICKSIGHT_DATASOURCE_NAME: str = _get(
    "quicksight", "datasource_name", "QUICKSIGHT_DATASOURCE_NAME",
    "Fraud Governance - Athena"
)
QUICKSIGHT_INFERENCE_DATASET_NAME: str = _get(
    "quicksight", "inference_dataset_name", "QUICKSIGHT_INFERENCE_DATASET_NAME",
    "Fraud Governance - Inference Monitoring"
)
QUICKSIGHT_DRIFT_DATASET_NAME: str = _get(
    "quicksight", "drift_dataset_name", "QUICKSIGHT_DRIFT_DATASET_NAME",
    "Fraud Governance - Drift Monitoring"
)
QUICKSIGHT_FEATURE_DRIFT_DATASET_NAME: str = _get(
    "quicksight", "feature_drift_dataset_name", "QUICKSIGHT_FEATURE_DRIFT_DATASET_NAME",
    "Fraud Governance - Feature Drift Analysis"
)
QUICKSIGHT_ANALYSIS_NAME: str = _get(
    "quicksight", "analysis_name", "QUICKSIGHT_ANALYSIS_NAME",
    "Fraud Detection Governance Analysis"
)
QUICKSIGHT_DASHBOARD_NAME: str = _get(
    "quicksight", "dashboard_name", "QUICKSIGHT_DASHBOARD_NAME",
    "Fraud Detection Governance"
)

# QuickSight service role (for Lake Formation permissions)
QUICKSIGHT_SERVICE_ROLE_NAME: str = _get(
    "quicksight", "service_role_name", "QUICKSIGHT_SERVICE_ROLE_NAME",
    "aws-quicksight-service-role-v0"
)

# ===================================================================
# Ground Truth Simulation (Development/Testing)
# ===================================================================
_gt_sim_cfg = _yaml_cfg.get("ground_truth_simulation", {}) if isinstance(
    _yaml_cfg.get("ground_truth_simulation"), dict
) else {}

GROUND_TRUTH_SIM_ACCURACY: float = float(
    _gt_sim_cfg.get("default_accuracy", os.environ.get("GROUND_TRUTH_SIM_ACCURACY", "0.85"))
)
GROUND_TRUTH_SIM_FRAUD_DAYS_MIN: int = int(
    _gt_sim_cfg.get("fraud_confirmation_days_min", os.environ.get("GROUND_TRUTH_SIM_FRAUD_DAYS_MIN", "1"))
)
GROUND_TRUTH_SIM_FRAUD_DAYS_MAX: int = int(
    _gt_sim_cfg.get("fraud_confirmation_days_max", os.environ.get("GROUND_TRUTH_SIM_FRAUD_DAYS_MAX", "7"))
)
GROUND_TRUTH_SIM_NON_FRAUD_DAYS_MIN: int = int(
    _gt_sim_cfg.get("non_fraud_confirmation_days_min", os.environ.get("GROUND_TRUTH_SIM_NON_FRAUD_DAYS_MIN", "1"))
)
GROUND_TRUTH_SIM_NON_FRAUD_DAYS_MAX: int = int(
    _gt_sim_cfg.get("non_fraud_confirmation_days_max", os.environ.get("GROUND_TRUTH_SIM_NON_FRAUD_DAYS_MAX", "30"))
)
GROUND_TRUTH_SIM_FEATURE_DRIFT_MAG: float = float(
    _gt_sim_cfg.get("feature_drift_magnitude", os.environ.get("GROUND_TRUTH_SIM_FEATURE_DRIFT_MAG", "0.0"))
)
GROUND_TRUTH_SIM_FEATURE_DRIFT_COUNT: int = int(
    _gt_sim_cfg.get("feature_drift_count", os.environ.get("GROUND_TRUTH_SIM_FEATURE_DRIFT_COUNT", "0"))
)
GROUND_TRUTH_SIM_FEATURE_DRIFT_IMPACT: float = float(
    _gt_sim_cfg.get("feature_drift_impact", os.environ.get("GROUND_TRUTH_SIM_FEATURE_DRIFT_IMPACT", "0.0"))
)
GROUND_TRUTH_SIM_MODEL_DRIFT_MAG: float = float(
    _gt_sim_cfg.get("model_drift_magnitude", os.environ.get("GROUND_TRUTH_SIM_MODEL_DRIFT_MAG", "0.0"))
)
GROUND_TRUTH_SIM_RANDOM_SEED: int = int(
    _gt_sim_cfg.get("random_seed", os.environ.get("GROUND_TRUTH_SIM_RANDOM_SEED", "42"))
)

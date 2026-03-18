"""
Athena table schema definitions for fraud detection pipeline.

This module contains DDL statements for all Iceberg and external tables:
- training_data: Training dataset with all features
- ground_truth: Labeled ground truth data (partitioned)
- inference_responses: Prediction logs (partitioned)
- drifted_data: Data drift samples (external table)
- monitoring_results: Model monitoring results with drift metrics (partitioned)
"""

from src.config.config import (
    ATHENA_DATABASE,
    S3_TRAINING_DATA,
    S3_GROUND_TRUTH,
    S3_INFERENCE_RESPONSES,
    S3_DRIFTED_DATA,
    S3_GROUND_TRUTH_UPDATES,
)

# Import monitoring schema
from .monitoring_schema import CREATE_MONITORING_RESULTS_TABLE

# =============================================================================
# Training Data Table (Iceberg)
# =============================================================================

CREATE_TRAINING_DATA_TABLE = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.training_data (
    -- Transaction identifiers
    transaction_id STRING,

    -- Transaction features (5 columns)
    transaction_timestamp DOUBLE,
    transaction_hour DOUBLE,
    transaction_day_of_week DOUBLE,
    transaction_amount DOUBLE,
    transaction_type_code DOUBLE,

    -- Customer profile (4 columns)
    customer_age DOUBLE,
    customer_gender STRING,
    customer_tenure_months DOUBLE,
    account_age_days DOUBLE,

    -- Geographic & temporal (6 columns)
    distance_from_home_km DOUBLE,
    distance_from_last_transaction_km DOUBLE,
    time_since_last_transaction_min DOUBLE,
    online_transaction DOUBLE,
    international_transaction DOUBLE,
    high_risk_country DOUBLE,

    -- Merchant info (2 columns)
    merchant_category_code DOUBLE,
    merchant_reputation_score DOUBLE,

    -- Payment security (5 columns)
    chip_transaction DOUBLE,
    pin_used DOUBLE,
    card_present DOUBLE,
    cvv_match DOUBLE,
    address_verification_match DOUBLE,

    -- Behavioral patterns (7 columns)
    num_transactions_24h DOUBLE,
    num_transactions_7days DOUBLE,
    avg_transaction_amount_30days DOUBLE,
    max_transaction_amount_30days DOUBLE,
    velocity_score DOUBLE,
    recurring_transaction DOUBLE,
    previous_fraud_incidents DOUBLE,

    -- Credit info (2 columns)
    credit_limit DOUBLE,
    available_credit_ratio DOUBLE,

    -- Target & predictions
    fraud_prediction BOOLEAN,
    fraud_probability DOUBLE,
    is_fraud BOOLEAN,

    -- Metadata
    data_version STRING,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
LOCATION '{S3_TRAINING_DATA}'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format' = 'parquet'
)
"""

# =============================================================================
# Ground Truth Table (Iceberg, Partitioned)
# =============================================================================

CREATE_GROUND_TRUTH_TABLE = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.ground_truth (
    transaction_id STRING,
    prediction_timestamp TIMESTAMP,
    window_id INT,

    -- All 33 feature columns
    transaction_timestamp DOUBLE,
    transaction_hour DOUBLE,
    transaction_day_of_week DOUBLE,
    transaction_amount DOUBLE,
    transaction_type_code DOUBLE,
    customer_age DOUBLE,
    customer_gender STRING,
    customer_tenure_months DOUBLE,
    account_age_days DOUBLE,
    distance_from_home_km DOUBLE,
    distance_from_last_transaction_km DOUBLE,
    time_since_last_transaction_min DOUBLE,
    online_transaction DOUBLE,
    international_transaction DOUBLE,
    high_risk_country DOUBLE,
    merchant_category_code DOUBLE,
    merchant_reputation_score DOUBLE,
    chip_transaction DOUBLE,
    pin_used DOUBLE,
    card_present DOUBLE,
    cvv_match DOUBLE,
    address_verification_match DOUBLE,
    num_transactions_24h DOUBLE,
    num_transactions_7days DOUBLE,
    avg_transaction_amount_30days DOUBLE,
    max_transaction_amount_30days DOUBLE,
    velocity_score DOUBLE,
    recurring_transaction DOUBLE,
    previous_fraud_incidents DOUBLE,
    credit_limit DOUBLE,
    available_credit_ratio DOUBLE,

    -- Ground truth labels
    ground_truth_fraud BOOLEAN,
    observed_fraud BOOLEAN,
    fraud_probability DOUBLE,

    -- Metadata
    data_source STRING,
    ingestion_timestamp TIMESTAMP,
    batch_id STRING
)
PARTITIONED BY (day(prediction_timestamp))
LOCATION '{S3_GROUND_TRUTH}'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format' = 'parquet'
)
"""

# =============================================================================
# Inference Responses Table (Iceberg, Partitioned)
# =============================================================================

CREATE_INFERENCE_RESPONSES_TABLE = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.inference_responses (
    -- Identifiers
    inference_id STRING,
    request_timestamp TIMESTAMP,
    endpoint_name STRING,
    model_version STRING,
    mlflow_run_id STRING,

    -- Input features (JSON for flexibility)
    input_features STRING,

    -- Prediction outputs
    prediction INT,
    probability_fraud DOUBLE,
    probability_non_fraud DOUBLE,
    confidence_score DOUBLE,

    -- Ground truth (populated asynchronously after fraud investigation)
    ground_truth INT,
    ground_truth_timestamp TIMESTAMP,
    ground_truth_source STRING,
    days_to_ground_truth DOUBLE,

    -- Performance metadata
    inference_latency_ms DOUBLE,
    model_load_time_ms DOUBLE,
    preprocessing_time_ms DOUBLE,

    -- Business context
    transaction_id STRING,
    transaction_amount DOUBLE,
    customer_id STRING,

    -- Derived fields for analytics
    is_high_confidence BOOLEAN,
    is_low_confidence BOOLEAN,
    prediction_bucket STRING,

    -- Error tracking
    request_id STRING,
    response_time TIMESTAMP,
    error_message STRING,
    inference_mode STRING
)
PARTITIONED BY (day(request_timestamp), endpoint_name)
LOCATION '{S3_INFERENCE_RESPONSES}'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format' = 'parquet'
)
"""

# =============================================================================
# Drifted Data Table (Regular External Table)
# =============================================================================

CREATE_DRIFTED_DATA_TABLE = f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_DATABASE}.drifted_data (
    transaction_id STRING,

    -- All 33 feature columns
    transaction_timestamp DOUBLE,
    transaction_hour DOUBLE,
    transaction_day_of_week DOUBLE,
    transaction_amount DOUBLE,
    transaction_type_code DOUBLE,
    customer_age DOUBLE,
    customer_gender STRING,
    customer_tenure_months DOUBLE,
    account_age_days DOUBLE,
    distance_from_home_km DOUBLE,
    distance_from_last_transaction_km DOUBLE,
    time_since_last_transaction_min DOUBLE,
    online_transaction DOUBLE,
    international_transaction DOUBLE,
    high_risk_country DOUBLE,
    merchant_category_code DOUBLE,
    merchant_reputation_score DOUBLE,
    chip_transaction DOUBLE,
    pin_used DOUBLE,
    card_present DOUBLE,
    cvv_match DOUBLE,
    address_verification_match DOUBLE,
    num_transactions_24h DOUBLE,
    num_transactions_7days DOUBLE,
    avg_transaction_amount_30days DOUBLE,
    max_transaction_amount_30days DOUBLE,
    velocity_score DOUBLE,
    recurring_transaction DOUBLE,
    previous_fraud_incidents DOUBLE,
    credit_limit DOUBLE,
    available_credit_ratio DOUBLE,

    -- Target
    is_fraud BOOLEAN
)
STORED AS PARQUET
LOCATION '{S3_DRIFTED_DATA}'
TBLPROPERTIES ('parquet.compression'='SNAPPY')
"""

# =============================================================================
# Ground Truth Updates Table (Iceberg, Partitioned)
# =============================================================================

CREATE_GROUND_TRUTH_UPDATES_TABLE = f"""
CREATE TABLE IF NOT EXISTS {ATHENA_DATABASE}.ground_truth_updates (
    -- Link to inference
    transaction_id STRING,
    inference_id STRING,

    -- Confirmed fraud status
    actual_fraud BOOLEAN,
    confirmation_timestamp TIMESTAMP,
    confirmation_source STRING,

    -- Timing information
    transaction_timestamp TIMESTAMP,
    prediction_timestamp TIMESTAMP,
    days_since_transaction DOUBLE,
    days_since_prediction DOUBLE,

    -- Investigation details
    investigation_notes STRING,
    investigation_priority STRING,
    false_positive BOOLEAN,
    false_negative BOOLEAN,

    -- Window info for drift testing
    window_id INT,

    -- Metadata
    batch_id STRING,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
PARTITIONED BY (day(confirmation_timestamp))
LOCATION '{S3_GROUND_TRUTH_UPDATES}'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format' = 'parquet'
)
"""

# =============================================================================
# All Tables Dictionary
# =============================================================================

ALL_TABLE_DEFINITIONS = {
    'training_data': CREATE_TRAINING_DATA_TABLE,
    'ground_truth': CREATE_GROUND_TRUTH_TABLE,
    'inference_responses': CREATE_INFERENCE_RESPONSES_TABLE,
    'drifted_data': CREATE_DRIFTED_DATA_TABLE,
    'ground_truth_updates': CREATE_GROUND_TRUTH_UPDATES_TABLE,
    'monitoring_results': CREATE_MONITORING_RESULTS_TABLE,
}

# =============================================================================
# Column Mappings
# =============================================================================

# Map CSV columns to Athena columns (if different)
CSV_TO_ATHENA_COLUMN_MAP = {
    'class': 'is_fraud',  # Rename target column
    # Add more mappings as needed
}

# Columns that need type conversion
TYPE_CONVERSIONS = {
    # Boolean columns
    'is_fraud': 'boolean',
    'fraud_prediction': 'boolean',
    'ground_truth_fraud': 'boolean',
    'observed_fraud': 'boolean',
    'false_positive': 'boolean',
    'false_negative': 'boolean',
    'is_high_confidence': 'boolean',
    'is_low_confidence': 'boolean',

    # String columns
    'transaction_id': 'string',
    'customer_id': 'string',
    'inference_id': 'string',
    'batch_id': 'string',
    'customer_gender': 'string',
    'endpoint_name': 'string',
    'model_version': 'string',
    'mlflow_run_id': 'string',
    'data_source': 'string',
    'ground_truth_source': 'string',
    'confirmation_source': 'string',
    'investigation_notes': 'string',
    'investigation_priority': 'string',
    'prediction_bucket': 'string',
    'error_message': 'string',
    'inference_mode': 'string',
    'request_id': 'string',
    'data_version': 'string',
    'input_features': 'string',

    # Timestamp columns
    'prediction_timestamp': 'timestamp',
    'request_timestamp': 'timestamp',
    'response_time': 'timestamp',
    'ground_truth_timestamp': 'timestamp',
    'confirmation_timestamp': 'timestamp',
    'ingestion_timestamp': 'timestamp',
    'created_at': 'timestamp',
    'updated_at': 'timestamp',

    # Integer columns (INT = int32)
    'window_id': 'int',
    'prediction': 'int',
    'ground_truth': 'int',
}

# =============================================================================
# Validation Queries
# =============================================================================

def get_row_count_query(table_name: str) -> str:
    """Get query to count rows in a table."""
    return f"SELECT COUNT(*) as row_count FROM {ATHENA_DATABASE}.{table_name}"


def get_table_info_query(table_name: str) -> str:
    """Get query to describe table structure."""
    return f"DESCRIBE {ATHENA_DATABASE}.{table_name}"


def get_sample_data_query(table_name: str, limit: int = 10) -> str:
    """Get query to sample data from table."""
    return f"SELECT * FROM {ATHENA_DATABASE}.{table_name} LIMIT {limit}"


def get_partition_info_query(table_name: str) -> str:
    """Get query to list table partitions."""
    return f"SHOW PARTITIONS {ATHENA_DATABASE}.{table_name}"


# =============================================================================
# Helper Functions
# =============================================================================

def get_create_statement(table_name: str) -> str:
    """Get CREATE TABLE statement for a given table name."""
    if table_name not in ALL_TABLE_DEFINITIONS:
        raise ValueError(f"Unknown table: {table_name}. Valid tables: {list(ALL_TABLE_DEFINITIONS.keys())}")
    return ALL_TABLE_DEFINITIONS[table_name]


def list_all_tables() -> list:
    """List all defined table names."""
    return list(ALL_TABLE_DEFINITIONS.keys())


def get_iceberg_tables() -> list:
    """Get list of Iceberg tables (excludes external tables)."""
    return ['training_data', 'ground_truth', 'inference_responses', 'ground_truth_updates', 'monitoring_results']


def get_partitioned_tables() -> list:
    """Get list of partitioned tables."""
    return ['ground_truth', 'inference_responses', 'ground_truth_updates', 'monitoring_results']


if __name__ == '__main__':
    """Print all table definitions."""
    print("=" * 80)
    print("Athena Table Schema Definitions")
    print("=" * 80)

    for table_name, ddl in ALL_TABLE_DEFINITIONS.items():
        print(f"\n{'=' * 80}")
        print(f"Table: {table_name}")
        print(f"{'=' * 80}")
        print(ddl)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total tables: {len(ALL_TABLE_DEFINITIONS)}")
    print(f"Iceberg tables: {', '.join(get_iceberg_tables())}")
    print(f"Partitioned tables: {', '.join(get_partitioned_tables())}")

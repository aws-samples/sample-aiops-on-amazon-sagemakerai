"""
Enhanced SageMaker inference handler with Athena logging.

⚠️ NOTE: This file intentionally uses awswrangler for BUFFERED WRITES.
This is a valid use case - awswrangler is efficient for small batch writes.
The main pipeline uses PySpark for large-scale data processing.

This handler extends the base inference.py with:
- Buffered async logging to Athena for inference responses
- Metadata tracking (latency, confidence scores)
- Business context enrichment
- Error tracking

**Architecture Note:**
- Uses awswrangler.athena.to_iceberg() for efficient buffered writes
- Batch size: 50 records or 5-minute interval
- NOT deprecated - this is the correct tool for this use case
- For bulk data processing, see preprocessing_pyspark.py
"""

import json
import os
import pickle
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List
import threading

import pandas as pd
import numpy as np
import xgboost as xgb

# Configure logging for CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Athena logging globals (initialized lazily)
inference_buffer: List[Dict[str, Any]] = []
last_flush_time = time.time()
athena_client = None
buffer_lock = threading.Lock()

# Configuration from environment variables
ENABLE_ATHENA_LOGGING = os.getenv('ENABLE_ATHENA_LOGGING', 'true').lower() == 'true'
BUFFER_SIZE = int(os.getenv('INFERENCE_LOG_BATCH_SIZE', '50'))  # Flush after 50 predictions
FLUSH_INTERVAL = int(os.getenv('INFERENCE_LOG_FLUSH_INTERVAL', '300'))  # 5 minutes
ENDPOINT_NAME = os.getenv('ENDPOINT_NAME', 'unknown')
MODEL_VERSION = os.getenv('MODEL_VERSION', 'unknown')
MLFLOW_RUN_ID = os.getenv('MLFLOW_RUN_ID', 'unknown')
ATHENA_DATABASE = os.getenv('ATHENA_DATABASE', 'fraud_detection')
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.9'))
LOW_CONFIDENCE_LOWER = float(os.getenv('LOW_CONFIDENCE_LOWER', '0.4'))
LOW_CONFIDENCE_UPPER = float(os.getenv('LOW_CONFIDENCE_UPPER', '0.6'))


def get_athena_client():
    """Lazy initialization of Athena client."""
    global athena_client
    if athena_client is None and ENABLE_ATHENA_LOGGING:
        try:
            import awswrangler as wr
            # Store the module for use in flush_to_athena
            athena_client = wr
            logger.info("✓ Athena client initialized for inference logging")
        except ImportError:
            logger.warning("⚠ awswrangler not available, Athena logging disabled")
            athena_client = False  # Mark as unavailable
    return athena_client


def get_prediction_bucket(fraud_probability: float) -> str:
    """Categorize prediction by probability."""
    if fraud_probability < 0.2:
        return "very_low"
    elif fraud_probability < 0.4:
        return "low"
    elif fraud_probability < 0.6:
        return "medium"
    elif fraud_probability < 0.8:
        return "high"
    else:
        return "very_high"


def flush_to_athena():
    """
    Flush buffered inference logs to Athena.

    This is called when buffer is full or flush interval is reached.
    Uses Iceberg table append for ACID compliance.
    """
    global inference_buffer, last_flush_time

    with buffer_lock:
        if not inference_buffer:
            return

        # Get copy of buffer and clear it
        records_to_flush = inference_buffer.copy()
        inference_buffer.clear()
        last_flush_time = time.time()

    # Flush outside the lock to avoid blocking predictions
    try:
        wr = get_athena_client()
        if wr and wr is not False:
            # Convert to DataFrame
            df = pd.DataFrame(records_to_flush)

            # Write to Athena Iceberg table
            table_name = 'inference_responses'
            s3_output = os.getenv('ATHENA_OUTPUT_S3', 's3://fraud-detection-data-lake/athena-query-results/')

            wr.athena.to_iceberg(
                df=df,
                database=ATHENA_DATABASE,
                table=table_name,
                temp_path=f"{s3_output}temp/",
                keep_files=False,
            )

            logger.info(f"✓ Flushed {len(records_to_flush)} inference records to Athena")

    except Exception as e:
        logger.error(f"⚠ Failed to flush to Athena: {e}", exc_info=True)
        # Re-add records to buffer for retry
        with buffer_lock:
            inference_buffer.extend(records_to_flush)


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load the model and feature metadata for inference.

    Args:
        model_dir: Directory where model artifacts are stored

    Returns:
        Dictionary containing model and feature names
    """
    start_time = time.time()

    # Try to find the model file (support multiple formats)
    model_path = None
    for filename in ["xgboost-model.json", "xgboost-model", "model.pkl", "model.xgb", "model.ubj"]:
        potential_path = os.path.join(model_dir, filename)
        if os.path.exists(potential_path):
            model_path = potential_path
            break

    if model_path is None:
        raise FileNotFoundError(f"Model file not found in {model_dir}")

    print(f"Loading model from: {model_path}")

    # Load the model based on file type
    if model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        # Load as XGBoost Booster (supports .json, .ubj, .xgb formats)
        model = xgb.Booster()
        model.load_model(model_path)

    feature_names_path = os.path.join(model_dir, "feature_names.json")

    # Load feature names
    with open(feature_names_path, "r") as f:
        feature_metadata = json.load(f)
        feature_names = feature_metadata["feature_names"]

    load_time_ms = (time.time() - start_time) * 1000
    print(f"✓ Model loaded in {load_time_ms:.2f}ms")

    return {
        "model": model,
        "feature_names": feature_names,
        "model_load_time_ms": load_time_ms,
    }


def input_fn(request_body: str, content_type: str = "application/json") -> pd.DataFrame:
    """
    Parse input data for prediction.

    Args:
        request_body: The request body (JSON string)
        content_type: Content type of the request

    Returns:
        DataFrame with input features
    """
    if content_type == "application/json":
        data = json.loads(request_body)

        # Support both single instance and batch predictions
        if isinstance(data, dict):
            # Single instance: {"feature1": value1, "feature2": value2, ...}
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Batch: [{"feature1": value1, ...}, {"feature1": value1, ...}]
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")

        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: pd.DataFrame, model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make predictions using the loaded model with enhanced logging.

    Args:
        input_data: Input features as DataFrame
        model_dict: Dictionary containing model and metadata

    Returns:
        Dictionary with predictions and probabilities
    """
    start_time = time.time()
    preprocessing_start = time.time()

    model = model_dict["model"]
    feature_names = model_dict["feature_names"]
    model_load_time_ms = model_dict.get("model_load_time_ms", 0)

    # Ensure all required features are present
    missing_features = set(feature_names) - set(input_data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Reorder columns to match training feature order
    input_data = input_data[feature_names]

    preprocessing_time_ms = (time.time() - preprocessing_start) * 1000

    # Make predictions
    if isinstance(model, xgb.Booster):
        # XGBoost Booster object (loaded from JSON/UBJ)
        dmatrix = xgb.DMatrix(input_data)
        probabilities = model.predict(dmatrix)

        # For binary classification, probabilities are fraud probabilities
        # Convert to class predictions using 0.5 threshold
        predictions = (probabilities > 0.5).astype(int)

        # Format results
        results = {
            "predictions": predictions.tolist(),
            "probabilities": {
                "non_fraud": (1 - probabilities).tolist(),
                "fraud": probabilities.tolist(),
            },
        }
    else:
        # sklearn wrapper (has predict and predict_proba methods)
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        # Format results
        results = {
            "predictions": predictions.tolist(),
            "probabilities": {
                "non_fraud": probabilities[:, 0].tolist(),
                "fraud": probabilities[:, 1].tolist(),
            },
        }

    # Calculate total inference latency
    inference_latency_ms = (time.time() - start_time) * 1000

    # Log to Athena (if enabled)
    if ENABLE_ATHENA_LOGGING:
        try:
            # Process each prediction in the batch
            for idx in range(len(predictions)):
                fraud_prob = results["probabilities"]["fraud"][idx]
                non_fraud_prob = results["probabilities"]["non_fraud"][idx]
                prediction = int(results["predictions"][idx])

                # Calculate confidence score
                confidence_score = max(fraud_prob, non_fraud_prob)

                # Extract transaction context if available
                transaction_id = input_data.iloc[idx].get('transaction_id', None)
                transaction_amount = input_data.iloc[idx].get('transaction_amount', None)
                customer_id = input_data.iloc[idx].get('customer_id', None)

                # Create log entry
                log_entry = {
                    'inference_id': str(uuid.uuid4()),
                    'request_timestamp': datetime.utcnow(),
                    'endpoint_name': ENDPOINT_NAME,
                    'model_version': MODEL_VERSION,
                    'mlflow_run_id': MLFLOW_RUN_ID,
                    'input_features': json.dumps(input_data.iloc[idx].to_dict()),
                    'prediction': prediction,
                    'probability_fraud': float(fraud_prob),
                    'probability_non_fraud': float(non_fraud_prob),
                    'confidence_score': float(confidence_score),
                    'ground_truth': None,  # To be populated later
                    'ground_truth_timestamp': None,
                    'inference_latency_ms': float(inference_latency_ms),
                    'model_load_time_ms': float(model_load_time_ms),
                    'preprocessing_time_ms': float(preprocessing_time_ms),
                    'transaction_id': str(transaction_id) if transaction_id else None,
                    'transaction_amount': float(transaction_amount) if transaction_amount else None,
                    'customer_id': str(customer_id) if customer_id else None,
                    'is_high_confidence': confidence_score > HIGH_CONFIDENCE_THRESHOLD,
                    'is_low_confidence': LOW_CONFIDENCE_LOWER <= confidence_score <= LOW_CONFIDENCE_UPPER,
                    'prediction_bucket': get_prediction_bucket(fraud_prob),
                    'request_id': str(uuid.uuid4()),
                    'response_time': datetime.utcnow(),
                    'error_message': None,
                    'inference_mode': 'realtime',
                }

                # Add to buffer
                with buffer_lock:
                    inference_buffer.append(log_entry)

                    # Check if we should flush
                    should_flush = (
                        len(inference_buffer) >= BUFFER_SIZE or
                        (time.time() - last_flush_time) >= FLUSH_INTERVAL
                    )

                if should_flush:
                    # Flush in background thread to avoid blocking
                    logger.info(f"Buffer full ({len(inference_buffer)} records), triggering flush")
                    flush_thread = threading.Thread(target=flush_to_athena, daemon=True)
                    flush_thread.start()

        except Exception as e:
            logger.error(f"⚠ Error logging to Athena: {e}", exc_info=True)
            # Don't fail the prediction due to logging errors

    return results


def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    """
    Format the prediction output.

    Args:
        prediction: Prediction results
        accept: Accept header content type

    Returns:
        Formatted prediction output as JSON string
    """
    if accept == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# Cleanup handler (called on container shutdown)
def shutdown_handler():
    """Flush any remaining logs on container shutdown."""
    print("Shutting down inference handler, flushing remaining logs...")
    flush_to_athena()


# Register shutdown handler
import atexit
atexit.register(shutdown_handler)

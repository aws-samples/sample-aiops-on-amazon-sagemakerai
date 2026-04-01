import json
import os
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np
import xgboost as xgb
import boto3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
ENABLE_ATHENA_LOGGING = os.getenv('ENABLE_ATHENA_LOGGING', 'true').lower() == 'true'
ENDPOINT_NAME = os.getenv('ENDPOINT_NAME', 'unknown')
MODEL_VERSION = os.getenv('MODEL_VERSION', 'unknown')
MLFLOW_RUN_ID = os.getenv('MLFLOW_RUN_ID', 'unknown')
SQS_QUEUE_URL = os.getenv('SQS_QUEUE_URL', 'unknown')
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.9'))
LOW_CONFIDENCE_LOWER = float(os.getenv('LOW_CONFIDENCE_LOWER', '0.4'))
LOW_CONFIDENCE_UPPER = float(os.getenv('LOW_CONFIDENCE_UPPER', '0.6'))

sqs_client = None


def get_sqs_client():
    global sqs_client
    if sqs_client is None:
        region = os.getenv('SAGEMAKER_REGION', 'us-east-1')
        sqs_client = boto3.client('sqs', region_name=region)
    return sqs_client


def get_prediction_bucket(fraud_probability: float) -> str:
    if fraud_probability < 0.2:
        return "very_low"
    elif fraud_probability < 0.4:
        return "low"
    elif fraud_probability < 0.6:
        return "medium"
    elif fraud_probability < 0.8:
        return "high"
    return "very_high"


def model_fn(model_dir: str) -> Dict[str, Any]:
    start_time = time.time()

    model_path = None
    for filename in ["xgboost-model.json", "xgboost-model", "model.pkl", "model.xgb", "model.ubj"]:
        potential_path = os.path.join(model_dir, filename)
        if os.path.exists(potential_path):
            model_path = potential_path
            break

    if model_path is None:
        raise FileNotFoundError(f"Model file not found in {model_dir}")

    logger.info(f"Loading model from: {model_path}")

    if model_path.endswith(".pkl"):
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = xgb.Booster()
        model.load_model(model_path)

    feature_names = None
    meta_path = os.path.join(model_dir, "feature_names.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            feature_names = json.load(f)["feature_names"]

    load_time_ms = (time.time() - start_time) * 1000
    logger.info(f"✓ Model loaded in {load_time_ms:.2f}ms with {len(feature_names or [])} features")

    return {"model": model, "feature_names": feature_names, "model_load_time_ms": load_time_ms}


def input_fn(request_body: str, content_type: str = "application/json") -> pd.DataFrame:
    if content_type == "application/json":
        data = json.loads(request_body)

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")

        # Convert object columns to numeric for XGBoost
        cols_to_drop = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains unhashable types (lists, dicts)
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample_val, (list, dict)):
                    cols_to_drop.append(col)
                    continue

                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # String column - label encode
                    try:
                        unique_vals = df[col].dropna().astype(str).unique()
                        mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                        df[col] = df[col].astype(str).map(mapping).fillna(0).astype(float)
                    except Exception:
                        cols_to_drop.append(col)

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: pd.DataFrame, model_dict: Dict[str, Any]) -> Dict[str, Any]:
    start_time = time.time()

    model = model_dict["model"]
    feature_names = model_dict["feature_names"]
    model_load_time_ms = model_dict.get("model_load_time_ms", 0)

    missing = set(feature_names) - set(input_data.columns)
    for f in missing:
        input_data[f] = 0.0

    original_data = input_data.copy()
    input_data = input_data[feature_names]

    preprocessing_time_ms = (time.time() - start_time) * 1000

    dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)
    probabilities = model.predict(dmatrix)
    predictions = (probabilities > 0.5).astype(int)

    results = {
        "predictions": predictions.tolist(),
        "probabilities": {
            "non_fraud": (1 - probabilities).tolist(),
            "fraud": probabilities.tolist(),
        },
    }

    inference_latency_ms = (time.time() - start_time) * 1000

    # Send to SQS (fire-and-forget)
    if ENABLE_ATHENA_LOGGING and SQS_QUEUE_URL:
        try:
            sqs = get_sqs_client()
            for idx in range(len(predictions)):
                fraud_prob = float(results["probabilities"]["fraud"][idx])
                confidence_score = max(fraud_prob, 1 - fraud_prob)

                log_entry = {
                    'inference_id': str(uuid.uuid4()),
                    'request_timestamp': datetime.utcnow().isoformat(),
                    'endpoint_name': ENDPOINT_NAME,
                    'model_version': MODEL_VERSION,
                    'mlflow_run_id': MLFLOW_RUN_ID,
                    'input_features': json.dumps(input_data.iloc[idx].to_dict()),
                    'prediction': int(predictions[idx]),
                    'probability_fraud': fraud_prob,
                    'probability_non_fraud': float(1 - fraud_prob),
                    'confidence_score': confidence_score,
                    'ground_truth': None,
                    'ground_truth_timestamp': None,
                    'inference_latency_ms': inference_latency_ms,
                    'model_load_time_ms': model_load_time_ms,
                    'preprocessing_time_ms': preprocessing_time_ms,
                    'transaction_id': str(original_data.iloc[idx].get('transaction_id', '')) or None,
                    'transaction_amount': float(original_data.iloc[idx].get('transaction_amount', 0)) if 'transaction_amount' in original_data.columns else None,
                    'customer_id': str(original_data.iloc[idx].get('customer_id', '')) or None,
                    'is_high_confidence': confidence_score > HIGH_CONFIDENCE_THRESHOLD,
                    'is_low_confidence': LOW_CONFIDENCE_LOWER <= confidence_score <= LOW_CONFIDENCE_UPPER,
                    'prediction_bucket': get_prediction_bucket(fraud_prob),
                    'request_id': str(uuid.uuid4()),
                    'response_time': datetime.utcnow().isoformat(),
                    'error_message': None,
                    'inference_mode': 'realtime',
                }

                sqs.send_message(
                    QueueUrl=SQS_QUEUE_URL,
                    MessageBody=json.dumps(log_entry),
                )
        except Exception as e:
            print(f"⚠ SQS send failed: {e}")

    return results


def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    if accept == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported accept type: {accept}")
"""
Lambda function to process SageMaker data capture files and log to MLflow with GenAI evaluations.

This function:
1. Reads JSONL data capture files from S3
2. Parses request/response data (handles base64 encoding for errors)
3. Logs traces to MLflow with spans
4. Runs GenAI evaluations on the traces
"""
import json
import logging
import os
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime

import boto3
import mlflow
from mlflow.genai import scorer
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Safety, RelevanceToQuery, Fluency, Guidelines
from typing import Literal

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')

# Environment variables
MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
MLFLOW_EXPERIMENT_NAME = os.environ['MLFLOW_EXPERIMENT_NAME']
SAGEMAKER_ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']
BEDROCK_MODEL_ID = os.environ['BEDROCK_MODEL_ID']
DATA_CAPTURE_BUCKET = os.environ['DATA_CAPTURE_BUCKET']

# MLflow evaluation model parameters
MLFLOW_EVALUATION_MODEL_PARAM = {
    "temperature": 0,
    "max_tokens": 512,
    "anthropic_version": "bedrock-2023-05-31",
    "top_p": 0.9,
    "stop_sequences": ["}"]
}


def decode_base64_data(encoded_data: str) -> Any:
    """Decode base64 encoded data and parse as JSON if possible."""
    try:
        decoded_bytes = base64.b64decode(encoded_data)
        decoded_str = decoded_bytes.decode('utf-8')

        # Try to parse as JSON
        try:
            return json.loads(decoded_str)
        except json.JSONDecodeError:
            return decoded_str
    except Exception as e:
        logger.error(f"Error decoding base64 data: {e}")
        return f"Error decoding: {str(e)}"


def parse_data_capture_record(record: Dict) -> Dict:
    """Parse a SageMaker data capture record."""
    parsed_record = {
        "timestamp": record.get("eventMetadata", {}).get("inferenceTime"),
        "event_id": record.get("eventMetadata", {}).get("eventId"),
    }

    # Parse request (input)
    capture_data = record.get("captureData", {})
    endpoint_input = capture_data.get("endpointInput", {})
    endpoint_output = capture_data.get("endpointOutput", {})

    # Decode request
    if endpoint_input:
        encoding = endpoint_input.get("encoding", "")
        data = endpoint_input.get("data", "")

        if encoding == "BASE64":
            parsed_record["request"] = decode_base64_data(data)
        else:
            try:
                parsed_record["request"] = json.loads(data) if data else {}
            except:
                parsed_record["request"] = data

    # Decode response
    if endpoint_output:
        encoding = endpoint_output.get("encoding", "")
        data = endpoint_output.get("data", "")
        observed_content_type = endpoint_output.get("observedContentType")

        if encoding == "BASE64":
            parsed_record["response"] = decode_base64_data(data)
        else:
            try:
                parsed_record["response"] = json.loads(data) if data else {}
            except:
                parsed_record["response"] = data

        # Determine if this is an error response
        response_data = parsed_record.get("response", {})
        if isinstance(response_data, dict):
            if "code" in response_data:
                parsed_record["status_code"] = response_data["code"]
                parsed_record["error_message"] = response_data.get("message", "")
                parsed_record["is_error"] = True
            elif observed_content_type is None:
                # If observedContentType is None, it's likely an error
                parsed_record["is_error"] = True
                parsed_record["status_code"] = response_data.get("code", 500)
            else:
                parsed_record["status_code"] = 200
                parsed_record["is_error"] = False

    return parsed_record


def log_trace_to_mlflow(event_data: Dict, s3_file_key: str) -> None:
    """
    Log a single inference event as an MLflow trace with spans.

    Args:
        event_data: Parsed data capture event
        s3_file_key: S3 key of the source file
    """
    logger.info(f'Processing event to log mlflow trace: {event_data.get("event_id")}')

    # Extract data
    event_id = event_data.get('event_id', 'unknown')
    inference_time = event_data.get('timestamp', datetime.now().isoformat())

    # Parse input/output
    input_data = event_data.get('request', {})
    output_data = event_data.get('response', {})
    is_error = event_data.get('is_error', False)
    status_code = event_data.get('status_code', 200)

    # Additional trace attributes
    additional_trace_attr = {
        "s3_bucket_name": DATA_CAPTURE_BUCKET,
        "s3_file_key": s3_file_key,
        "sagemaker_endpoint_name": SAGEMAKER_ENDPOINT_NAME,
        "event_id": event_id,
        "inference_time": inference_time,
        "status_code": str(status_code),
    }

    # Create trace with span
    # Use event_id as request_id for idempotency - prevents duplicate traces on retry
    with mlflow.start_span(name=s3_file_key) as span:
        # Set inputs
        span.set_inputs({
            "prompt": input_data.get('inputs', ''),
            "parameters": input_data.get('parameters', {})
        })

        # Set attributes
        span.set_attributes(additional_trace_attr)

        # Set status for errors
        if is_error:
            span.set_status("ERROR")

        # Set outputs
        span.set_outputs(output_data)

    logger.info(f'Logged trace for event {event_id}')


def lambda_handler(event: Dict, context: Any) -> Dict:
    """
    Lambda handler for processing SageMaker data capture files.

    Args:
        event: Lambda event containing S3 bucket and key information
        context: Lambda context

    Returns:
        Dict with processing results
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        # Set MLflow tracking URI and experiment
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Extract S3 information from event
        # Event can come from Step Functions or direct S3 event
        if 'records' in event:
            # Batch processing from Step Functions
            records = event['records']
            s3_bucket = event.get('s3_bucket', DATA_CAPTURE_BUCKET)
            s3_key = event.get('s3_key', '')
        elif 's3_key' in event:
            # Direct from EventBridge S3 event
            s3_bucket = event.get('s3_bucket', DATA_CAPTURE_BUCKET)
            s3_key = event.get('s3_key', '')
            records = None
        else:
            # Fallback: read from S3 directly
            s3_bucket = event.get('bucket', DATA_CAPTURE_BUCKET)
            s3_key = event.get('key', '')
            records = None
        
        # Filter: Only process .jsonl files
        if not s3_key.endswith('.jsonl'):
            logger.info(f"Skipping non-JSONL file: {s3_key}")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Skipped non-JSONL file',
                    's3_key': s3_key,
                })
            }

        logger.info(f"Processing file: s3://{s3_bucket}/{s3_key}")

        # Read JSONL file from S3 if records not provided
        if records is None:
            obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            file_content = obj['Body'].read().decode('utf-8')

            # Parse JSONL (each line is a JSON object)
            records = []
            for line in file_content.strip().split('\n'):
                if line.strip():
                    records.append(json.loads(line))

        logger.info(f"Found {len(records)} records to process")

        # Process each record and log to MLflow
        processed_count = 0
        error_count = 0

        #with mlflow.start_run(run_name=f"sagemaker_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        for record in records:
            try:
                # Parse the record
                if isinstance(record, str):
                    record = json.loads(record)

                parsed_record = parse_data_capture_record(record)

                # Log trace to MLflow
                log_trace_to_mlflow(parsed_record, s3_key)

                processed_count += 1

                if parsed_record.get('is_error'):
                    error_count += 1

            except Exception as e:
                logger.error(f"Error processing record: {e}", exc_info=True)
                continue

            # Log summary metrics
            # mlflow.log_param("total_records", len(records))
            # mlflow.log_param("processed_count", processed_count)
            # mlflow.log_param("error_count", error_count)
            # mlflow.log_param("s3_file_key", s3_key)

        logger.info(f"Successfully processed {processed_count}/{len(records)} records")

        # Run GenAI evaluations on the traces
        # Note: This is done separately to allow traces to be fully committed
        run_evaluations(s3_key)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully processed data capture file',
                'records_processed': processed_count,
                'errors_found': error_count,
                's3_key': s3_key,
            })
        }

    except Exception as e:
        logger.error(f"Error in lambda handler: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Error processing data capture file',
                'error': str(e)
            })
        }


def run_evaluations(s3_file_key: str) -> None:
    """
    Run GenAI evaluations on traces from the specified S3 file.

    Args:
        s3_file_key: S3 key to filter traces
    """
    try:
        logger.info(f"Running mlflow genai evaluations on traces from {s3_file_key}")

        # Define custom scorers
        @scorer
        def tokens_words(outputs) -> int:
            """Approximate words in the response"""
            try:
                if isinstance(outputs, dict):
                    words = len(outputs.get('generated_text', '').split())
                else:
                    words = len(str(outputs).split())
            except:
                return 0
            return words
        
        # Create a judge that evaluates coherence using MLflow template-based scorers
        coherence_judge = make_judge(
            name="coherence",
            instructions=(
                "Evaluate if the response is coherent, maintaining a constant tone "
                "and following a clear flow of thoughts/concepts"
                "Question: {{ inputs }}\n"
                "Response: {{ outputs }}\n"
            ),
            feedback_value_type=Literal["coherent", "somewhat coherent", "incoherent"],
            model= BEDROCK_MODEL_ID
        )
        mlflow.autolog(disable=True)
        # Search for traces from this file
        traces = mlflow.search_traces(
            filter_string=f"name = '{s3_file_key}'"
        )

        if len(traces) == 0:
            logger.warning(f"No traces found for {s3_file_key}")
            return

        logger.info(f"Found {len(traces)} traces to evaluate")

        # Define scorers for evaluation

        scorers = [
            Safety(
                model=BEDROCK_MODEL_ID,
                parameters=MLFLOW_EVALUATION_MODEL_PARAM,
            ),
            RelevanceToQuery(
                model=BEDROCK_MODEL_ID,
                parameters=MLFLOW_EVALUATION_MODEL_PARAM,
            ),
            Fluency(
                model=BEDROCK_MODEL_ID,
                parameters=MLFLOW_EVALUATION_MODEL_PARAM,
            ),
            Guidelines(
                name="follows_objective",
                guidelines="The generated response must follow the objective in the request.",
                model=BEDROCK_MODEL_ID,
                parameters=MLFLOW_EVALUATION_MODEL_PARAM,
            ),
            Guidelines(
                name="professional_tone",
                guidelines="The response must be in a professional tone.",
                model=BEDROCK_MODEL_ID,
                parameters=MLFLOW_EVALUATION_MODEL_PARAM,
            ),
            coherence_judge,
            tokens_words,
        ]

        logger.info(f"Start mlflow genai trace evaluate")
        # Run evaluations
        try:
            results = mlflow.genai.evaluate(
                data=traces,
                scorers=scorers,
            )
            return
        except Exception as e:
            logger.warning(f"First Traces evaluation pass, skipping with error: {e}")
            pass
        
        logger.info(f"Second pass at mlflow genai trace evaluate")
        # Second pass: run evaluation in batches
        # To handle the issue https://github.com/mlflow/mlflow/issues/21002
        results = mlflow.genai.evaluate(
                data=traces,
                scorers=scorers,
            )

        logger.info(f"Evaluations completed successfully")

    except Exception as e:
        logger.error(f"Error running evaluations: {e}", exc_info=True)
        # Don't raise exception - evaluations are optional

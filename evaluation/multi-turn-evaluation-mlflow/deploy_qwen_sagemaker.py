"""
Deploy Qwen3.5-0.8B to a SageMaker AI real-time inference endpoint.

Uses the AWS Deep Learning Container for vLLM with an OpenAI-compatible chat
completions API. The vLLM reasoning parser strips Qwen <think>...</think>
blocks server-side into a separate `reasoning` field, so assistant messages
are always clean text — which is what the multi-turn evaluation notebook
(and its LLM judges) expect.

Prerequisites:
    pip install boto3
    An IAM role SageMaker can assume (or run from SageMaker Studio, where the
    execution role is detected automatically).

Usage:
    # Optionally override the role (required outside SageMaker Studio):
    export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account-id>:role/YourSageMakerExecutionRole"
    python deploy_qwen_sagemaker.py

The endpoint is named `qwen3-5-0-8b`, matching the ENDPOINT_NAME used in
multi_turn_eval_simulation.ipynb.
"""

import json
import os
import re
import time

import boto3

# ── Configuration ────────────────────────────────────────────────────────────
REGION = "us-east-1"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
ENDPOINT_NAME = "qwen3-5-0-8b"  # must match the eval notebook's ENDPOINT_NAME
INSTANCE = {"type": "ml.g6.xlarge", "num_gpu": 1}
STARTUP_HEALTH_CHECK_TIMEOUT = 900
VARIANT_NAME = "v1"

boto_session = boto3.Session(region_name=REGION)
sm = boto_session.client("sagemaker")
sm_runtime = boto_session.client("sagemaker-runtime")


def get_sagemaker_role() -> str:
    """Return the IAM role ARN: env var override, or derive from the caller identity."""
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if role:
        return role
    sts = boto_session.client("sts")
    arn = sts.get_caller_identity()["Arn"]
    if "assumed-role" in arn:
        # e.g. running inside SageMaker Studio with an execution role.
        # Resolve via iam:GetRole — assumed-role ARNs do not include the IAM
        # path (e.g. "service-role/"), so reconstructing the ARN from the
        # STS ARN alone produces an invalid role ARN for roles with a path.
        role_name = arn.split("/")[-2]
        try:
            return boto_session.client("iam").get_role(RoleName=role_name)["Role"]["Arn"]
        except Exception:
            # Fall back to reconstruction (correct only for roles without a path)
            return re.sub(r"^(.+)sts::(\d+):assumed-role/(.+?)/.*$", r"\1iam::\2:role/\3", arn)
    raise SystemExit(
        "Could not determine a SageMaker execution role.\n"
        "Set the SAGEMAKER_ROLE_ARN environment variable, e.g.:\n"
        '  export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account-id>:role/YourSageMakerExecutionRole"'
    )


def delete_existing(endpoint_name: str) -> None:
    """Remove a previous (e.g. failed) deployment so the script can be re-run."""
    try:
        status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        print(f"Found existing endpoint '{endpoint_name}' (status: {status}) — deleting...")
        sm.delete_endpoint(EndpointName=endpoint_name)
        while True:
            try:
                sm.describe_endpoint(EndpointName=endpoint_name)
                time.sleep(10)
            except sm.exceptions.ClientError:
                break
    except sm.exceptions.ClientError:
        pass  # no endpoint
    for deleter, kwargs in [
        (sm.delete_endpoint_config, {"EndpointConfigName": endpoint_name}),
        (sm.delete_model, {"ModelName": endpoint_name}),
    ]:
        try:
            deleter(**kwargs)
        except sm.exceptions.ClientError:
            pass  # does not exist


def wait_for_endpoint(endpoint_name: str, sleep_time: int = 30) -> None:
    """Poll until the endpoint leaves 'Creating' status."""
    print(f"Waiting for endpoint '{endpoint_name}' ", end="", flush=True)
    while True:
        status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        if status != "Creating":
            break
        print(".", end="", flush=True)
        time.sleep(sleep_time)
    print(f"\nEndpoint: '{endpoint_name}', Status: '{status}'")
    if status != "InService":
        raise RuntimeError(f"Endpoint creation failed with status '{status}'")


def deploy() -> None:
    role = get_sagemaker_role()
    print(f"Using role: {role}")

    delete_existing(ENDPOINT_NAME)

    inference_image = (
        f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
        "vllm:0.20.0-gpu-py312-cu130-ubuntu22.04-sagemaker"
    )

    env = {
        "HF_MODEL_ID": MODEL_ID,
        "SM_VLLM_MODEL": MODEL_ID,
        "SM_VLLM_TENSOR_PARALLEL_SIZE": json.dumps(INSTANCE["num_gpu"]),
        "SM_VLLM_MAX_MODEL_LEN": "32768",
        "SM_VLLM_ENABLE_AUTO_TOOL_CHOICE": "true",
        "SM_VLLM_TOOL_CALL_PARSER": "qwen3_coder",
        "SM_VLLM_REASONING_PARSER": "qwen3",
        "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.90",
    }

    print(f"Creating model '{ENDPOINT_NAME}' ({MODEL_ID})...")
    sm.create_model(
        ModelName=ENDPOINT_NAME,
        ExecutionRoleArn=role,
        PrimaryContainer={"Image": inference_image, "Environment": env},
    )

    print(f"Creating endpoint config '{ENDPOINT_NAME}' ({INSTANCE['type']})...")
    sm.create_endpoint_config(
        EndpointConfigName=ENDPOINT_NAME,
        ProductionVariants=[
            {
                "VariantName": VARIANT_NAME,
                "ModelName": ENDPOINT_NAME,
                "InstanceType": INSTANCE["type"],
                "InitialInstanceCount": 1,
                "ContainerStartupHealthCheckTimeoutInSeconds": STARTUP_HEALTH_CHECK_TIMEOUT,
            },
        ],
    )

    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=ENDPOINT_NAME)
    wait_for_endpoint(ENDPOINT_NAME)


def smoke_test() -> None:
    """Send one chat message to verify the endpoint responds."""
    payload = {
        "messages": [{"role": "user", "content": "In one sentence, what is MLflow?"}],
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    print("\nRunning smoke test...")
    res = sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        Body=json.dumps(payload),
        ContentType="application/json",
    )
    response = json.loads(res["Body"].read().decode("utf8"))
    print(f"Response: {response['choices'][0]['message']['content']}")
    print("\n✅ Endpoint is ready. Use it in the notebook with:")
    print(f'   ENDPOINT_NAME = "{ENDPOINT_NAME}"')


if __name__ == "__main__":
    deploy()
    smoke_test()

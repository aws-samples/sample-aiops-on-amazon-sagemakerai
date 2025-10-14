import boto3

# Create a SageMaker client
sagemaker_client = boto3.client("sagemaker")


# tracking_server_name = "vlm-finetuning-server"
def get_mlflow_server_arn(tracking_server_name: str) -> str:
    # Specify the name of the MLflow Tracking Server you want to describe
    try:
        # Describe the MLflow Tracking Server
        response = sagemaker_client.describe_mlflow_tracking_server(
            TrackingServerName=tracking_server_name
        )

        # Print the retrieved information
        # print(response)

        # You can access specific details from the response, for example:
        tracking_server_arn = response.get("TrackingServerArn")
        tracking_server_status = response.get("TrackingServerStatus")
        # print(f"\nMLflow Tracking Server ARN: {tracking_server_arn}")
        # print(f"MLflow Tracking Server Status: {tracking_server_status}")

    except sagemaker_client.exceptions.ResourceNotFoundException:
        print(f"MLflow Tracking Server '{tracking_server_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return tracking_server_arn

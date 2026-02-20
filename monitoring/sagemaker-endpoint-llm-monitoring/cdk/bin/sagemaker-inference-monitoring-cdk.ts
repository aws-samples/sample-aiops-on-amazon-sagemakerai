#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { SageMakerInferenceMonitoringStack } from '../lib/sagemaker-inference-monitoring-stack';
import { config as loadDotenv } from 'dotenv';
import * as path from 'path';

// Load environment variables from .env file
loadDotenv({ path: path.join(__dirname, '../.env') });

const app = new cdk.App();

// REQUIRED: Stack prefix - must be provided for unique resource names
const stackPrefix = process.env.STACK_PREFIX;
if (!stackPrefix) {
  throw new Error('STACK_PREFIX is required. Please set it in .env file or as an environment variable.');
}

// REQUIRED: SageMaker endpoint name - must be provided
const sagemakerEndpointName = process.env.SAGEMAKER_ENDPOINT_NAME;
if (!sagemakerEndpointName) {
  throw new Error('SAGEMAKER_ENDPOINT_NAME is required. Please set it in .env file or as an environment variable.');
}

// REQUIRED: MLflow tracking server ARN - must be provided
const mlflowTrackingServerArn = process.env.MLFLOW_TRACKING_URI;
if (!mlflowTrackingServerArn) {
  throw new Error('MLFLOW_TRACKING_URI is required. Please set it in .env file or as an environment variable.');
}

// Configuration with smart defaults based on endpoint name
const config = {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'us-west-2',
  },

  // REQUIRED: MLflow app ARN
  mlflowTrackingServerArn,

  // REQUIRED: SageMaker endpoint name
  sagemakerEndpointName,

  // REQUIRED: CDK stack prefix for unique resource names
  stackPrefix,

  // OPTIONAL: S3 bucket where data capture is stored
  // Defaults to: sagemaker-{region}-{account}
  dataCaptureS3BucketName: process.env.DATA_CAPTURE_S3_BUCKET,

  // OPTIONAL: S3 prefix for data capture files
  // Defaults to: {endpoint-name}-data-capture/
  dataCaptureS3Prefix: process.env.DATA_CAPTURE_S3_PREFIX ||
    `${sagemakerEndpointName}`,

  // OPTIONAL: MLflow experiment name
  // Defaults to: sagemaker-endpoint-{endpoint-name}-monitoring
  mlflowExperimentName: process.env.MLFLOW_EXPERIMENT_NAME ||
    `${sagemakerEndpointName}`,

  // OPTIONAL: Bedrock model for evaluations
  bedrockModelId: process.env.BEDROCK_MODEL_ID ||
    'bedrock:/global.anthropic.claude-sonnet-4-20250514-v1:0',
};

// Create unique stack ID based on prefix to support multiple deployments
const stackId = config.stackPrefix
  ? `SageMakerInferenceMonitoringStack-${config.stackPrefix}`
  : 'SageMakerInferenceMonitoringStack';

new SageMakerInferenceMonitoringStack(app, stackId, {
  ...config,
  description: 'Infrastructure for automated SageMaker endpoint monitoring with MLflow and GenAI evaluations',

  tags: {
    Project: 'SageMaker-Inference-Monitoring',
    ManagedBy: 'CDK',
  },
});

app.synth();

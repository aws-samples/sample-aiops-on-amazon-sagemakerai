import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
export interface SageMakerInferenceMonitoringStackProps extends cdk.StackProps {
    /**
     * Existing S3 bucket name where SageMaker data capture is stored
     * If not provided, will look for default SageMaker bucket
     */
    dataCaptureS3BucketName?: string;
    /**
     * cdk stack prefix
     * @default ''
     */
    stackPrefix?: string;
    /**
     * S3 prefix where data capture files are stored
     */
    dataCaptureS3Prefix?: string;
    /**
     * MLflow tracking server ARN
     * Format: arn:aws:sagemaker:region:account-id:mlflow-app/app-id
     */
    mlflowTrackingServerArn: string;
    /**
     * MLflow experiment name
     * @default sagemakerEndpointName
     */
    mlflowExperimentName: string;
    /**
     * SageMaker endpoint name to monitor
     */
    sagemakerEndpointName: string;
    /**
     * Bedrock model ID for GenAI evaluations
     * @default 'bedrock:/global.anthropic.claude-sonnet-4-20250514-v1:0'
     */
    bedrockModelId: string;
}
export declare class SageMakerInferenceMonitoringStack extends cdk.Stack {
    readonly processorFunction: lambda.Function;
    readonly stateMachine: sfn.StateMachine;
    constructor(scope: Construct, id: string, props: SageMakerInferenceMonitoringStackProps);
}

import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3n from 'aws-cdk-lib/aws-s3-notifications';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as path from 'path';

export interface SageMakerInferenceMonitoringStackProps extends cdk.StackProps {
  /**
   * Existing S3 bucket name where SageMaker data capture is stored
   * If not provided, will look for default SageMaker bucket
   */
  dataCaptureS3BucketName: string;

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

export class SageMakerInferenceMonitoringStack extends cdk.Stack {
  public readonly processorFunction: lambda.Function;
  public readonly stateMachine: sfn.StateMachine;

  constructor(scope: Construct, id: string, props: SageMakerInferenceMonitoringStackProps) {
    super(scope, id, props);

    const dataCapturePrefix = props.dataCaptureS3Prefix;
    const mlflowExperimentName = props.mlflowExperimentName;
    const bedrockModelId = props.bedrockModelId;
    const stackPrefix = props.stackPrefix;

    // Reference existing S3 bucket where SageMaker data capture is stored
    const dataCaptureS3BucketName = props.dataCaptureS3BucketName;

    const datCaptureBucket = s3.Bucket.fromBucketName(
      this,
      'DataCaptureBucket',
      dataCaptureS3BucketName
    );

    // Lambda execution role
    const lambdaRole = new iam.Role(this, 'ProcessorLambdaRole', {
      roleName: `sagemaker-data-capture-processing-${stackPrefix}`,
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'Role for Lambda function that processes SageMaker data capture files',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      ],
    });

    // Grant Lambda access to S3 bucket for reading data capture files and writing MLflow trace artifacts
    datCaptureBucket.grantReadWrite(lambdaRole);
    
    // Grant Lambda access to default SageMaker bucket for MLflow trace artifacts
    const defaultSageMakerBucket = s3.Bucket.fromBucketName(
      this,
      'MLflowArtifactsBucket',
      `sagemaker-${this.region}-${this.account}`
    );
    defaultSageMakerBucket.grantReadWrite(lambdaRole);

    // Grant Lambda access to SageMaker MLflow App
    lambdaRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'sagemaker:DescribeMlflowApp',
        'sagemaker:CallMlflowAppApi',
        'sagemaker:ListMlflowApps',
        //'sagemaker:CreatePresignedMlflowAppUrl',
      ],
      resources: [props.mlflowTrackingServerArn],
    }));

    // Grant Lambda access to Bedrock for evaluations
    lambdaRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:InvokeModel',
        'bedrock:InvokeModelWithResponseStream',
        'bedrock:ListFoundationModels',
        'bedrock:GetFoundationModel',
      ],
      resources: ['*'], // Bedrock models don't have specific ARNs
    }));

    // Lambda function to process data capture files
    this.processorFunction = new lambda.DockerImageFunction(this, 'ProcessorLambda', {
      functionName: `sagemaker-data-capture-processor-${stackPrefix}`,
      code: lambda.DockerImageCode.fromImageAsset(
        path.join(__dirname, '../lambda')
      ),
      description: 'Processes SageMaker data capture files and logs to MLflow with GenAI evaluations',
      memorySize: 3008, // Increased for MLflow and evaluation libraries
      timeout: cdk.Duration.minutes(15), // Increased for GenAI evaluations
      architecture: lambda.Architecture.X86_64,
      role: lambdaRole,
      environment: {
        MLFLOW_TRACKING_URI: props.mlflowTrackingServerArn,
        MLFLOW_EXPERIMENT_NAME: props.mlflowExperimentName,
        SAGEMAKER_ENDPOINT_NAME: props.sagemakerEndpointName,
        BEDROCK_MODEL_ID: props.bedrockModelId,
        DATA_CAPTURE_BUCKET: dataCaptureS3BucketName,
        AWS_ROLE_ARN: lambdaRole.roleArn,
      },
    });

    // Step Functions State Machine Definition
    // Lambda reads entire JSONL file and processes all records
    // This is simpler than Distributed Map for JSONL files where each line is a record
    const processTask = new tasks.LambdaInvoke(this, 'ProcessDataCaptureFile', {
      lambdaFunction: this.processorFunction,
      payload: sfn.TaskInput.fromObject({
        's3_bucket': sfn.JsonPath.stringAt('$.detail.bucket.name'),
        's3_key': sfn.JsonPath.stringAt('$.detail.object.key'),
      }),
      retryOnServiceExceptions: true,
    });

    // Create the state machine
    const logGroup = new logs.LogGroup(this, 'StateMachineLogGroup', {
      logGroupName: `/solution/sagemaker-endpoint-llm-monitoring-${stackPrefix}`,
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    this.stateMachine = new sfn.StateMachine(this, 'MonitoringStateMachine', {
      stateMachineName: `sagemaker-data-capture-monitoring-${stackPrefix}`,
      definitionBody: sfn.DefinitionBody.fromChainable(processTask),
      logs: {
        destination: logGroup,
        level: sfn.LogLevel.ALL,
      },
      tracingEnabled: true,
      timeout: cdk.Duration.hours(1),
    });

    // Grant Step Functions permission to invoke Lambda
    this.processorFunction.grantInvoke(this.stateMachine);

    // S3 event notification to trigger Step Functions
    // This will be triggered when new .jsonl files are created in the data capture prefix
    const eventPattern = {
      detail: {
        bucket: {
          name: [dataCaptureS3BucketName],
        },
        object: {
          key: [{
            // prefix: dataCapturePrefix,
            suffix: '.jsonl',
          }],
        },
      },
    };

    // Create EventBridge rule to trigger Step Functions
    const rule = new cdk.aws_events.Rule(this, 'S3EventRule', {
      ruleName: `sagemaker-s3-event-rule-${stackPrefix}`,
      eventPattern: {
        source: ['aws.s3'],
        detailType: ['Object Created'],
        detail: eventPattern.detail,
      },
    });

    rule.addTarget(new cdk.aws_events_targets.SfnStateMachine(this.stateMachine));

    // Enable S3 event notifications to EventBridge
    // Note: This requires enabling EventBridge notifications on the S3 bucket manually
    // or using a custom resource to enable it
    new cdk.CfnOutput(this, 'S3BucketEventBridgeNote', {
      value: `Please enable EventBridge notifications on bucket ${dataCaptureS3BucketName} manually or via AWS CLI: aws s3api put-bucket-notification-configuration --bucket ${dataCaptureS3BucketName} --notification-configuration '{"EventBridgeConfiguration": {}}'`,
      description: 'Command to enable EventBridge notifications on S3 bucket',
    });

    // Outputs
    new cdk.CfnOutput(this, 'ProcessorLambdaArn', {
      value: this.processorFunction.functionArn,
      description: 'ARN of the Lambda function that processes data capture files',
      exportName: `ProcessorLambdaArn-${stackPrefix}`,
    });

    new cdk.CfnOutput(this, 'StateMachineArn', {
      value: this.stateMachine.stateMachineArn,
      description: 'ARN of the Step Functions state machine',
      exportName: `StateMachineArn-${stackPrefix}`,
    });

    new cdk.CfnOutput(this, 'DataCaptureBucketName', {
      value: dataCaptureS3BucketName,
      description: 'S3 bucket where data capture files are stored',
      exportName: `DataCaptureBucketName-${stackPrefix}`,
    });
  }
}

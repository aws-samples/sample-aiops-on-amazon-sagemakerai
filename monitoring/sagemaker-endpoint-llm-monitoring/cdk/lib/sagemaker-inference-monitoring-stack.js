"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.SageMakerInferenceMonitoringStack = void 0;
const cdk = __importStar(require("aws-cdk-lib"));
const lambda = __importStar(require("aws-cdk-lib/aws-lambda"));
const s3 = __importStar(require("aws-cdk-lib/aws-s3"));
const sfn = __importStar(require("aws-cdk-lib/aws-stepfunctions"));
const tasks = __importStar(require("aws-cdk-lib/aws-stepfunctions-tasks"));
const iam = __importStar(require("aws-cdk-lib/aws-iam"));
const logs = __importStar(require("aws-cdk-lib/aws-logs"));
const path = __importStar(require("path"));
class SageMakerInferenceMonitoringStack extends cdk.Stack {
    constructor(scope, id, props) {
        super(scope, id, props);
        const dataCapturePrefix = props.dataCaptureS3Prefix;
        const mlflowExperimentName = props.mlflowExperimentName;
        const bedrockModelId = props.bedrockModelId;
        const stackPrefix = props.stackPrefix;
        // Reference existing S3 bucket where SageMaker data capture is stored
        const dataCaptureS3BucketName = props.dataCaptureS3BucketName;
        const datCaptureBucket = s3.Bucket.fromBucketName(this, 'DataCaptureBucket', dataCaptureS3BucketName);
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
        const defaultSageMakerBucket = s3.Bucket.fromBucketName(this, 'MLflowArtifactsBucket', `sagemaker-${this.region}-${this.account}`);
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
            code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../lambda')),
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
exports.SageMakerInferenceMonitoringStack = SageMakerInferenceMonitoringStack;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2FnZW1ha2VyLWluZmVyZW5jZS1tb25pdG9yaW5nLXN0YWNrLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsic2FnZW1ha2VyLWluZmVyZW5jZS1tb25pdG9yaW5nLXN0YWNrLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsaURBQW1DO0FBRW5DLCtEQUFpRDtBQUNqRCx1REFBeUM7QUFFekMsbUVBQXFEO0FBQ3JELDJFQUE2RDtBQUM3RCx5REFBMkM7QUFDM0MsMkRBQTZDO0FBQzdDLDJDQUE2QjtBQTRDN0IsTUFBYSxpQ0FBa0MsU0FBUSxHQUFHLENBQUMsS0FBSztJQUk5RCxZQUFZLEtBQWdCLEVBQUUsRUFBVSxFQUFFLEtBQTZDO1FBQ3JGLEtBQUssQ0FBQyxLQUFLLEVBQUUsRUFBRSxFQUFFLEtBQUssQ0FBQyxDQUFDO1FBRXhCLE1BQU0saUJBQWlCLEdBQUcsS0FBSyxDQUFDLG1CQUFtQixDQUFDO1FBQ3BELE1BQU0sb0JBQW9CLEdBQUcsS0FBSyxDQUFDLG9CQUFvQixDQUFDO1FBQ3hELE1BQU0sY0FBYyxHQUFHLEtBQUssQ0FBQyxjQUFjLENBQUM7UUFDNUMsTUFBTSxXQUFXLEdBQUcsS0FBSyxDQUFDLFdBQVcsQ0FBQztRQUV0QyxzRUFBc0U7UUFDdEUsTUFBTSx1QkFBdUIsR0FBRyxLQUFLLENBQUMsdUJBQXVCLENBQUM7UUFFOUQsTUFBTSxnQkFBZ0IsR0FBRyxFQUFFLENBQUMsTUFBTSxDQUFDLGNBQWMsQ0FDL0MsSUFBSSxFQUNKLG1CQUFtQixFQUNuQix1QkFBdUIsQ0FDeEIsQ0FBQztRQUVGLHdCQUF3QjtRQUN4QixNQUFNLFVBQVUsR0FBRyxJQUFJLEdBQUcsQ0FBQyxJQUFJLENBQUMsSUFBSSxFQUFFLHFCQUFxQixFQUFFO1lBQzNELFFBQVEsRUFBRSxxQ0FBcUMsV0FBVyxFQUFFO1lBQzVELFNBQVMsRUFBRSxJQUFJLEdBQUcsQ0FBQyxnQkFBZ0IsQ0FBQyxzQkFBc0Is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dUIsRUFBRTtZQUMvQyxLQUFLLEVBQUUsdUJBQXVCO1lBQzlCLFdBQVcsRUFBRSwrQ0FBK0M7WUFDNUQsVUFBVSxFQUFFLHlCQUF5QixXQUFXLEVBQUU7U0FDbkQsQ0FBQyxDQUFDO0lBQ0wsQ0FBQztDQUNGO0FBL0tELDhFQStLQyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIGNkayBmcm9tICdhd3MtY2RrLWxpYic7XG5pbXBvcnQgeyBDb25zdHJ1Y3QgfSBmcm9tICdjb25zdHJ1Y3RzJztcbmltcG9ydCAqIGFzIGxhbWJkYSBmcm9tICdhd3MtY2RrLWxpYi9hd3MtbGFtYmRhJztcbmltcG9ydCAqIGFzIHMzIGZyb20gJ2F3cy1jZGstbGliL2F3cy1zMyc7XG5pbXBvcnQgKiBhcyBzM24gZnJvbSAnYXdzLWNkay1saWIvYXdzLXMzLW5vdGlmaWNhdGlvbnMnO1xuaW1wb3J0ICogYXMgc2ZuIGZyb20gJ2F3cy1jZGstbGliL2F3cy1zdGVwZnVuY3Rpb25zJztcbmltcG9ydCAqIGFzIHRhc2tzIGZyb20gJ2F3cy1jZGstbGliL2F3cy1zdGVwZnVuY3Rpb25zLXRhc2tzJztcbmltcG9ydCAqIGFzIGlhbSBmcm9tICdhd3MtY2RrLWxpYi9hd3MtaWFtJztcbmltcG9ydCAqIGFzIGxvZ3MgZnJvbSAnYXdzLWNkay1saWIvYXdzLWxvZ3MnO1xuaW1wb3J0ICogYXMgcGF0aCBmcm9tICdwYXRoJztcblxuZXhwb3J0IGludGVyZmFjZSBTYWdlTWFrZXJJbmZlcmVuY2VNb25pdG9yaW5nU3RhY2tQcm9wcyBleHRlbmRzIGNkay5TdGFja1Byb3BzIHtcbiAgLyoqXG4gICAqIEV4aXN0aW5nIFMzIGJ1Y2tldCBuYW1lIHdoZXJlIFNhZ2VNYWtlciBkYXRhIGNhcHR1cmUgaXMgc3RvcmVkXG4gICAqIElmIG5vdCBwcm92aWRlZCwgd2lsbCBsb29rIGZvciBkZWZhdWx0IFNhZ2VNYWtlciBidWNrZXRcbiAgICovXG4gIGRhdGFDYXB0dXJlUzNCdWNrZXROYW1lOiBzdHJpbmc7XG5cbiAgLyoqXG4gICAqIGNkayBzdGFjayBwcmVmaXggXG4gICAqIEBkZWZhdWx0ICcnXG4gICAqL1xuICBzdGFja1ByZWZpeD86IHN0cmluZztcbiAgXG4gIC8qKlxuICAgKiBTMyBwcmVmaXggd2hlcmUgZGF0YSBjYXB0dXJlIGZpbGVzIGFyZSBzdG9yZWRcbiAgICovXG4gIGRhdGFDYXB0dXJlUzNQcmVmaXg/OiBzdHJpbmc7XG5cbiAgLyoqXG4gICAqIE1MZmxvdyB0cmFja2luZyBzZXJ2ZXIgQVJOXG4gICAqIEZvcm1hdDogYXJuOmF3czpzYWdlbWFrZXI6cmVnaW9uOmFjY291bnQtaWQ6bWxmbG93LWFwcC9hcHAtaWRcbiAgICovXG4gIG1sZmxvd1RyYWNraW5nU2VydmVyQXJuOiBzdHJpbmc7XG5cbiAgLyoqXG4gICAqIE1MZmxvdyBleHBlcmltZW50IG5hbWVcbiAgICogQGRlZmF1bHQgc2FnZW1ha2VyRW5kcG9pbnROYW1lXG4gICAqL1xuICBtbGZsb3dFeHBlcmltZW50TmFtZTogc3RyaW5nO1xuXG4gIC8qKlxuICAgKiBTYWdlTWFrZXIgZW5kcG9pbnQgbmFtZSB0byBtb25pdG9yXG4gICAqL1xuICBzYWdlbWFrZXJFbmRwb2ludE5hbWU6IHN0cmluZztcblxuICAvKipcbiAgICogQmVkcm9jayBtb2RlbCBJRCBmb3IgR2VuQUkgZXZhbHVhdGlvbnNcbiAgICogQGRlZmF1bHQgJ2JlZHJvY2s6L2dsb2JhbC5hbnRocm9waWMuY2xhdWRlLXNvbm5ldC00LTIwMjUwNTE0LXYxOjAnXG4gICAqL1xuICBiZWRyb2NrTW9kZWxJZDogc3RyaW5nO1xufVxuXG5leHBvcnQgY2xhc3MgU2FnZU1ha2VySW5mZXJlbmNlTW9uaXRvcmluZ1N0YWNrIGV4dGVuZHMgY2RrLlN0YWNrIHtcbiAgcHVibGljIHJlYWRvbmx5IHByb2Nlc3NvckZ1bmN0aW9uOiBsYW1iZGEuRnVuY3Rpb247XG4gIHB1YmxpYyByZWFkb25seSBzdGF0ZU1hY2hpbmU6IHNmbi5TdGF0ZU1hY2hpbmU7XG5cbiAgY29uc3RydWN0b3Ioc2NvcGU6IENvbnN0cnVjdCwgaWQ6IHN0cmluZywgcHJvcHM6IFNhZ2VNYWtlckluZmVyZW5jZU1vbml0b3JpbmdTdGFja1Byb3BzKSB7XG4gICAgc3VwZXIoc2NvcGUsIGlkLCBwcm9wcyk7XG5cbiAgICBjb25zdCBkYXRhQ2FwdHVyZVByZWZpeCA9IHByb3BzLmRhdGFDYXB0dXJlUzNQcmVmaXg7XG4gICAgY29uc3QgbWxmbG93RXhwZXJpbWVudE5hbWUgPSBwcm9wcy5tbGZsb3dFeHBlcmltZW50TmFtZTtcbiAgICBjb25zdCBiZWRyb2NrTW9kZWxJZCA9IHByb3BzLmJlZHJvY2tNb2RlbElkO1xuICAgIGNvbnN0IHN0YWNrUHJlZml4ID0gcHJvcHMuc3RhY2tQcmVmaXg7XG5cbiAgICAvLyBSZWZlcmVuY2UgZXhpc3RpbmcgUzMgYnVja2V0IHdoZXJlIFNhZ2VNYWtlciBkYXRhIGNhcHR1cmUgaXMgc3RvcmVkXG4gICAgY29uc3QgZGF0YUNhcHR1cmVTM0J1Y2tldE5hbWUgPSBwcm9wcy5kYXRhQ2FwdHVyZVMzQnVja2V0TmFtZTtcblxuICAgIGNvbnN0IGRhdENhcHR1cmVCdWNrZXQgPSBzMy5CdWNrZXQuZnJvbUJ1Y2tldE5hbWUoXG4gICAgICB0aGlzLFxuICAgICAgJ0RhdGFDYXB0dXJlQnVja2V0JyxcbiAgICAgIGRhdGFDYXB0dXJlUzNCdWNrZXROYW1lXG4gICAgKTtcblxuICAgIC8vIExhbWJkYSBleGVjdXRpb24gcm9sZVxuICAgIGNvbnN0IGxhbWJkYVJvbGUgPSBuZXcgaWFtLlJvbGUodGhpcywgJ1Byb2Nlc3NvckxhbWJkYVJvbGUnLCB7XG4gICAgICByb2xlTmFtZTogYHNhZ2VtYWtlci1kYXRhLWNhcHR1cmUtcHJvY2Vzc2luZy0ke3N0YWNrUHJlZml4fWAsXG4gICAgICBhc3N1bWVkQnk6IG5ldyBpYW0uU2VydmljZVByaW5jaXBhbCgnbGFtYmRhLmFtYXpvbmF3cy5jb20nKSxcbiAgICAgIGRlc2NyaXB0aW9uOiAnUm9sZSBmb3IgTGFtYmRhIGZ1bmN0aW9uIHRoYXQgcHJvY2Vzc2VzIFNhZ2VNYWtlciBkYXRhIGNhcHR1cmUgZmlsZXMnLFxuICAgICAgbWFuYWdlZFBvbGljaWVzOiBbXG4gICAgICAgIGlhbS5NYW5hZ2VkUG9saWN5LmZyb21Bd3NNYW5hZ2VkUG9saWN5TmFtZSgnc2VydmljZS1yb2xlL0FXU0xhbWJkYUJhc2ljRXhlY3V0aW9uUm9sZScpLFxuICAgICAgXSxcbiAgICB9KTtcblxuICAgIC8vIEdyYW50IExhbWJkYSBhY2Nlc3MgdG8gUzMgYnVja2V0IGZvciByZWFkaW5nIGRhdGEgY2FwdHVyZSBmaWxlcyBhbmQgd3JpdGluZyBNTGZsb3cgdHJhY2UgYXJ0aWZhY3RzXG4gICAgZGF0Q2FwdHVyZUJ1Y2tldC5ncmFudFJlYWRXcml0ZShsYW1iZGFSb2xlKTtcbiAgICBcbiAgICAvLyBHcmFudCBMYW1iZGEgYWNjZXNzIHRvIGRlZmF1bHQgU2FnZU1ha2VyIGJ1Y2tldCBmb3IgTUxmbG93IHRyYWNlIGFydGlmYWN0c1xuICAgIGNvbnN0IGRlZmF1bHRTYWdlTWFrZXJCdWNrZXQgPSBzMy5CdWNrZXQuZnJvbUJ1Y2tldE5hbWUoXG4gICAgICB0aGlzLFxuICAgICAgJ01MZmxvd0FydGlmYWN0c0J1Y2tldCcsXG4gICAgICBgc2FnZW1ha2VyLSR7dGhpcy5yZWdpb259LSR7dGhpcy5hY2NvdW50fWBcbiAgICApO1xuICAgIGRlZmF1bHRTYWdlTWFrZXJCdWNrZXQuZ3JhbnRSZWFkV3JpdGUobGFtYmRhUm9sZSk7XG5cbiAgICAvLyBHcmFudCBMYW1iZGEgYWNjZXNzIHRvIFNhZ2VNYWtlciBNTGZsb3cgQXBwXG4gICAgbGFtYmRhUm9sZS5hZGRUb1BvbGljeShuZXcgaWFtLlBvbGljeVN0YXRlbWVudCh7XG4gICAgICBlZmZlY3Q6IGlhbS5FZmZlY3QuQUxMT1csXG4gICAgICBhY3Rpb25zOiBbXG4gICAgICAgICdzYWdlbWFrZXI6RGVzY3JpYmVNbGZsb3dBcHAnLFxuICAgICAgICAnc2FnZW1ha2VyOkNhbGxNbGZsb3dBcHBBcGknLFxuICAgICAgICAnc2FnZW1ha2VyOkxpc3RNbGZsb3dBcHBzJyxcbiAgICAgICAgLy8nc2FnZW1ha2VyOkNyZWF0ZVByZXNpZ25lZE1sZmxvd0FwcFVybCcsXG4gICAgICBdLFxuICAgICAgcmVzb3VyY2VzOiBbcHJvcHMubWxmbG93VHJhY2tpbmdTZXJ2ZXJBcm5dLFxuICAgIH0pKTtcblxuICAgIC8vIEdyYW50IExhbWJkYSBhY2Nlc3MgdG8gQmVkcm9jayBmb3IgZXZhbHVhdGlvbnNcbiAgICBsYW1iZGFSb2xlLmFkZFRvUG9saWN5KG5ldyBpYW0uUG9saWN5U3RhdGVtZW50KHtcbiAgICAgIGVmZmVjdDogaWFtLkVmZmVjdC5BTExPVyxcbiAgICAgIGFjdGlvbnM6IFtcbiAgICAgICAgJ2JlZHJvY2s6SW52b2tlTW9kZWwnLFxuICAgICAgICAnYmVkcm9jazpJbnZva2VNb2RlbFdpdGhSZXNwb25zZVN0cmVhbScsXG4gICAgICAgICdiZWRyb2NrOkxpc3RGb3VuZGF0aW9uTW9kZWxzJyxcbiAgICAgICAgJ2JlZHJvY2s6R2V0Rm91bmRhdGlvbk1vZGVsJyxcbiAgICAgIF0sXG4gICAgICByZXNvdXJjZXM6IFsnKiddLCAvLyBCZWRyb2NrIG1vZGVscyBkb24ndCBoYXZlIHNwZWNpZmljIEFSTnNcbiAgICB9KSk7XG5cbiAgICAvLyBMYW1iZGEgZnVuY3Rpb24gdG8gcHJvY2VzcyBkYXRhIGNhcHR1cmUgZmlsZXNcbiAgICB0aGlzLnByb2Nlc3NvckZ1bmN0aW9uID0gbmV3IGxhbWJkYS5Eb2NrZXJJbWFnZUZ1bmN0aW9uKHRoaXMsICdQcm9jZXNzb3JMYW1iZGEnLCB7XG4gICAgICBmdW5jdGlvbk5hbWU6IGBzYWdlbWFrZXItZGF0YS1jYXB0dXJlLXByb2Nlc3Nvci0ke3N0YWNrUHJlZml4fWAsXG4gICAgICBjb2RlOiBsYW1iZGEuRG9ja2VySW1hZ2VDb2RlLmZyb21JbWFnZUFzc2V0KFxuICAgICAgICBwYXRoLmpvaW4oX19kaXJuYW1lLCAnLi4vbGFtYmRhJylcbiAgICAgICksXG4gICAgICBkZXNjcmlwdGlvbjogJ1Byb2Nlc3NlcyBTYWdlTWFrZXIgZGF0YSBjYXB0dXJlIGZpbGVzIGFuZCBsb2dzIHRvIE1MZmxvdyB3aXRoIEdlbkFJIGV2YWx1YXRpb25zJyxcbiAgICAgIG1lbW9yeVNpemU6IDMwMDgsIC8vIEluY3JlYXNlZCBmb3IgTUxmbG93IGFuZCBldmFsdWF0aW9uIGxpYnJhcmllc1xuICAgICAgdGltZW91dDogY2RrLkR1cmF0aW9uLm1pbnV0ZXMoMTUpLCAvLyBJbmNyZWFzZWQgZm9yIEdlbkFJIGV2YWx1YXRpb25zXG4gICAgICBhcmNoaXRlY3R1cmU6IGxhbWJkYS5BcmNoaXRlY3R1cmUuWDg2XzY0LFxuICAgICAgcm9sZTogbGFtYmRhUm9sZSxcbiAgICAgIGVudmlyb25tZW50OiB7XG4gICAgICAgIE1MRkxPV19UUkFDS0lOR19VUkk6IHByb3BzLm1sZmxvd1RyYWNraW5nU2VydmVyQXJuLFxuICAgICAgICBNTEZMT1dfRVhQRVJJTUVOVF9OQU1FOiBwcm9wcy5tbGZsb3dFeHBlcmltZW50TmFtZSxcbiAgICAgICAgU0FHRU1BS0VSX0VORFBPSU5UX05BTUU6IHByb3BzLnNhZ2VtYWtlckVuZHBvaW50TmFtZSxcbiAgICAgICAgQkVEUk9DS19NT0RFTF9JRDogcHJvcHMuYmVkcm9ja01vZGVsSWQsXG4gICAgICAgIERBVEFfQ0FQVFVSRV9CVUNLRVQ6IGRhdGFDYXB0dXJlUzNCdWNrZXROYW1lLFxuICAgICAgICBBV1NfUk9MRV9BUk46IGxhbWJkYVJvbGUucm9sZUFybixcbiAgICAgIH0sXG4gICAgfSk7XG5cbiAgICAvLyBTdGVwIEZ1bmN0aW9ucyBTdGF0ZSBNYWNoaW5lIERlZmluaXRpb25cbiAgICAvLyBMYW1iZGEgcmVhZHMgZW50aXJlIEpTT05MIGZpbGUgYW5kIHByb2Nlc3NlcyBhbGwgcmVjb3Jkc1xuICAgIC8vIFRoaXMgaXMgc2ltcGxlciB0aGFuIERpc3RyaWJ1dGVkIE1hcCBmb3IgSlNPTkwgZmlsZXMgd2hlcmUgZWFjaCBsaW5lIGlzIGEgcmVjb3JkXG4gICAgY29uc3QgcHJvY2Vzc1Rhc2sgPSBuZXcgdGFza3MuTGFtYmRhSW52b2tlKHRoaXMsICdQcm9jZXNzRGF0YUNhcHR1cmVGaWxlJywge1xuICAgICAgbGFtYmRhRnVuY3Rpb246IHRoaXMucHJvY2Vzc29yRnVuY3Rpb24sXG4gICAgICBwYXlsb2FkOiBzZm4uVGFza0lucHV0LmZyb21PYmplY3Qoe1xuICAgICAgICAnczNfYnVja2V0Jzogc2ZuLkpzb25QYXRoLnN0cmluZ0F0KCckLmRldGFpbC5idWNrZXQubmFtZScpLFxuICAgICAgICAnczNfa2V5Jzogc2ZuLkpzb25QYXRoLnN0cmluZ0F0KCckLmRldGFpbC5vYmplY3Qua2V5JyksXG4gICAgICB9KSxcbiAgICAgIHJldHJ5T25TZXJ2aWNlRXhjZXB0aW9uczogdHJ1ZSxcbiAgICB9KTtcblxuICAgIC8vIENyZWF0ZSB0aGUgc3RhdGUgbWFjaGluZVxuICAgIGNvbnN0IGxvZ0dyb3VwID0gbmV3IGxvZ3MuTG9nR3JvdXAodGhpcywgJ1N0YXRlTWFjaGluZUxvZ0dyb3VwJywge1xuICAgICAgbG9nR3JvdXBOYW1lOiBgL3NvbHV0aW9uL3NhZ2VtYWtlci1lbmRwb2ludC1sbG0tbW9uaXRvcmluZy0ke3N0YWNrUHJlZml4fWAsXG4gICAgICByZXRlbnRpb246IGxvZ3MuUmV0ZW50aW9uRGF5cy5PTkVfV0VFSyxcbiAgICAgIHJlbW92YWxQb2xpY3k6IGNkay5SZW1vdmFsUG9saWN5LkRFU1RST1ksXG4gICAgfSk7XG5cbiAgICB0aGlzLnN0YXRlTWFjaGluZSA9IG5ldyBzZm4uU3RhdGVNYWNoaW5lKHRoaXMsICdNb25pdG9yaW5nU3RhdGVNYWNoaW5lJywge1xuICAgICAgc3RhdGVNYWNoaW5lTmFtZTogYHNhZ2VtYWtlci1kYXRhLWNhcHR1cmUtbW9uaXRvcmluZy0ke3N0YWNrUHJlZml4fWAsXG4gICAgICBkZWZpbml0aW9uQm9keTogc2ZuLkRlZmluaXRpb25Cb2R5LmZyb21DaGFpbmFibGUocHJvY2Vzc1Rhc2spLFxuICAgICAgbG9nczoge1xuICAgICAgICBkZXN0aW5hdGlvbjogbG9nR3JvdXAsXG4gICAgICAgIGxldmVsOiBzZm4uTG9nTGV2ZWwuQUxMLFxuICAgICAgfSxcbiAgICAgIHRyYWNpbmdFbmFibGVkOiB0cnVlLFxuICAgICAgdGltZW91dDogY2RrLkR1cmF0aW9uLmhvdXJzKDEpLFxuICAgIH0pO1xuXG4gICAgLy8gR3JhbnQgU3RlcCBGdW5jdGlvbnMgcGVybWlzc2lvbiB0byBpbnZva2UgTGFtYmRhXG4gICAgdGhpcy5wcm9jZXNzb3JGdW5jdGlvbi5ncmFudEludm9rZSh0aGlzLnN0YXRlTWFjaGluZSk7XG5cbiAgICAvLyBTMyBldmVudCBub3RpZmljYXRpb24gdG8gdHJpZ2dlciBTdGVwIEZ1bmN0aW9uc1xuICAgIC8vIFRoaXMgd2lsbCBiZSB0cmlnZ2VyZWQgd2hlbiBuZXcgLmpzb25sIGZpbGVzIGFyZSBjcmVhdGVkIGluIHRoZSBkYXRhIGNhcHR1cmUgcHJlZml4XG4gICAgY29uc3QgZXZlbnRQYXR0ZXJuID0ge1xuICAgICAgZGV0YWlsOiB7XG4gICAgICAgIGJ1Y2tldDoge1xuICAgICAgICAgIG5hbWU6IFtkYXRhQ2FwdHVyZVMzQnVja2V0TmFtZV0sXG4gICAgICAgIH0sXG4gICAgICAgIG9iamVjdDoge1xuICAgICAgICAgIGtleTogW3tcbiAgICAgICAgICAgIC8vIHByZWZpeDogZGF0YUNhcHR1cmVQcmVmaXgsXG4gICAgICAgICAgICBzdWZmaXg6ICcuanNvbmwnLFxuICAgICAgICAgIH1dLFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICB9O1xuXG4gICAgLy8gQ3JlYXRlIEV2ZW50QnJpZGdlIHJ1bGUgdG8gdHJpZ2dlciBTdGVwIEZ1bmN0aW9uc1xuICAgIGNvbnN0IHJ1bGUgPSBuZXcgY2RrLmF3c19ldmVudHMuUnVsZSh0aGlzLCAnUzNFdmVudFJ1bGUnLCB7XG4gICAgICBydWxlTmFtZTogYHNhZ2VtYWtlci1zMy1ldmVudC1ydWxlLSR7c3RhY2tQcmVmaXh9YCxcbiAgICAgIGV2ZW50UGF0dGVybjoge1xuICAgICAgICBzb3VyY2U6IFsnYXdzLnMzJ10sXG4gICAgICAgIGRldGFpbFR5cGU6IFsnT2JqZWN0IENyZWF0ZWQnXSxcbiAgICAgICAgZGV0YWlsOiBldmVudFBhdHRlcm4uZGV0YWlsLFxuICAgICAgfSxcbiAgICB9KTtcblxuICAgIHJ1bGUuYWRkVGFyZ2V0KG5ldyBjZGsuYXdzX2V2ZW50c190YXJnZXRzLlNmblN0YXRlTWFjaGluZSh0aGlzLnN0YXRlTWFjaGluZSkpO1xuXG4gICAgLy8gRW5hYmxlIFMzIGV2ZW50IG5vdGlmaWNhdGlvbnMgdG8gRXZlbnRCcmlkZ2VcbiAgICAvLyBOb3RlOiBUaGlzIHJlcXVpcmVzIGVuYWJsaW5nIEV2ZW50QnJpZGdlIG5vdGlmaWNhdGlvbnMgb24gdGhlIFMzIGJ1Y2tldCBtYW51YWxseVxuICAgIC8vIG9yIHVzaW5nIGEgY3VzdG9tIHJlc291cmNlIHRvIGVuYWJsZSBpdFxuICAgIG5ldyBjZGsuQ2ZuT3V0cHV0KHRoaXMsICdTM0J1Y2tldEV2ZW50QnJpZGdlTm90ZScsIHtcbiAgICAgIHZhbHVlOiBgUGxlYXNlIGVuYWJsZSBFdmVudEJyaWRnZSBub3RpZmljYXRpb25zIG9uIGJ1Y2tldCAke2RhdGFDYXB0dXJlUzNCdWNrZXROYW1lfSBtYW51YWxseSBvciB2aWEgQVdTIENMSTogYXdzIHMzYXBpIHB1dC1idWNrZXQtbm90aWZpY2F0aW9uLWNvbmZpZ3VyYXRpb24gLS1idWNrZXQgJHtkYXRhQ2FwdHVyZVMzQnVja2V0TmFtZX0gLS1ub3RpZmljYXRpb24tY29uZmlndXJhdGlvbiAne1wiRXZlbnRCcmlkZ2VDb25maWd1cmF0aW9uXCI6IHt9fSdgLFxuICAgICAgZGVzY3JpcHRpb246ICdDb21tYW5kIHRvIGVuYWJsZSBFdmVudEJyaWRnZSBub3RpZmljYXRpb25zIG9uIFMzIGJ1Y2tldCcsXG4gICAgfSk7XG5cbiAgICAvLyBPdXRwdXRzXG4gICAgbmV3IGNkay5DZm5PdXRwdXQodGhpcywgJ1Byb2Nlc3NvckxhbWJkYUFybicsIHtcbiAgICAgIHZhbHVlOiB0aGlzLnByb2Nlc3NvckZ1bmN0aW9uLmZ1bmN0aW9uQXJuLFxuICAgICAgZGVzY3JpcHRpb246ICdBUk4gb2YgdGhlIExhbWJkYSBmdW5jdGlvbiB0aGF0IHByb2Nlc3NlcyBkYXRhIGNhcHR1cmUgZmlsZXMnLFxuICAgICAgZXhwb3J0TmFtZTogYFByb2Nlc3NvckxhbWJkYUFybi0ke3N0YWNrUHJlZml4fWAsXG4gICAgfSk7XG5cbiAgICBuZXcgY2RrLkNmbk91dHB1dCh0aGlzLCAnU3RhdGVNYWNoaW5lQXJuJywge1xuICAgICAgdmFsdWU6IHRoaXMuc3RhdGVNYWNoaW5lLnN0YXRlTWFjaGluZUFybixcbiAgICAgIGRlc2NyaXB0aW9uOiAnQVJOIG9mIHRoZSBTdGVwIEZ1bmN0aW9ucyBzdGF0ZSBtYWNoaW5lJyxcbiAgICAgIGV4cG9ydE5hbWU6IGBTdGF0ZU1hY2hpbmVBcm4tJHtzdGFja1ByZWZpeH1gLFxuICAgIH0pO1xuXG4gICAgbmV3IGNkay5DZm5PdXRwdXQodGhpcywgJ0RhdGFDYXB0dXJlQnVja2V0TmFtZScsIHtcbiAgICAgIHZhbHVlOiBkYXRhQ2FwdHVyZVMzQnVja2V0TmFtZSxcbiAgICAgIGRlc2NyaXB0aW9uOiAnUzMgYnVja2V0IHdoZXJlIGRhdGEgY2FwdHVyZSBmaWxlcyBhcmUgc3RvcmVkJyxcbiAgICAgIGV4cG9ydE5hbWU6IGBEYXRhQ2FwdHVyZUJ1Y2tldE5hbWUtJHtzdGFja1ByZWZpeH1gLFxuICAgIH0pO1xuICB9XG59XG4iXX0=
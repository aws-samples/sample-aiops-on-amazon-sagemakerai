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
        const dataCaptureS3BucketName = props.dataCaptureS3BucketName ||
            `sagemaker-${this.region}-${this.account}`;
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2FnZW1ha2VyLWluZmVyZW5jZS1tb25pdG9yaW5nLXN0YWNrLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsic2FnZW1ha2VyLWluZmVyZW5jZS1tb25pdG9yaW5nLXN0YWNrLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsaURBQW1DO0FBRW5DLCtEQUFpRDtBQUNqRCx1REFBeUM7QUFFekMsbUVBQXFEO0FBQ3JELDJFQUE2RDtBQUM3RCx5REFBMkM7QUFDM0MsMkRBQTZDO0FBQzdDLDJDQUE2QjtBQTRDN0IsTUFBYSxpQ0FBa0MsU0FBUSxHQUFHLENBQUMsS0FBSztJQUk5RCxZQUFZLEtBQWdCLEVBQUUsRUFBVSxFQUFFLEtBQTZDO1FBQ3JGLEtBQUssQ0FBQyxLQUFLLEVBQUUsRUFBRSxFQUFFLEtBQUssQ0FBQyxDQUFDO1FBRXhCLE1BQU0saUJBQWlCLEdBQUcsS0FBSyxDQUFDLG1CQUFtQixDQUFDO1FBQ3BELE1BQU0sb0JBQW9CLEdBQUcsS0FBSyxDQUFDLG9CQUFvQixDQUFDO1FBQ3hELE1BQU0sY0FBYyxHQUFHLEtBQUssQ0FBQyxjQUFjLENBQUM7UUFDNUMsTUFBTSxXQUFXLEdBQUcsS0FBSyxDQUFDLFdBQVcsQ0FBQztRQUV0QyxzRUFBc0U7UUFDdEUsTUFBTSx1QkFBdUIsR0FBRyxLQUFLLENBQUMsdUJBQXVCO1lBQzNELGFBQWEsSUFBSSxDQUFDLE1BQU0sSUFBSSxJQUFJLENBQUMsT0FBTyxFQUFFLENBQUM7UUFFN0MsTUFBTSxnQkFBZ0IsR0FBRyxFQUFFLENBQUMsTUFBTSxDQUFDLGNBQWMsQ0FDL0MsSUFBSSxFQUNKLG1CQUFtQixFQUNuQix1QkFBdUIsQ0FDeEIsQ0FBQztRQUVGLHdCQUF3QjtRQUN4QixNQUFNLFVBQVUsR0FBRyxJQUFJLEdBQUcsQ0FBQyxJQUFJLENBQUMsSUFBSSxFQUFFLHFCQUFxQixFQUFFO1lBQzNELFFBQVEsRUFBRSxxQ0FBcUMsV0FBVyxFQUFF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Y2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBjZGsgZnJvbSAnYXdzLWNkay1saWInO1xuaW1wb3J0IHsgQ29uc3RydWN0IH0gZnJvbSAnY29uc3RydWN0cyc7XG5pbXBvcnQgKiBhcyBsYW1iZGEgZnJvbSAnYXdzLWNkay1saWIvYXdzLWxhbWJkYSc7XG5pbXBvcnQgKiBhcyBzMyBmcm9tICdhd3MtY2RrLWxpYi9hd3MtczMnO1xuaW1wb3J0ICogYXMgczNuIGZyb20gJ2F3cy1jZGstbGliL2F3cy1zMy1ub3RpZmljYXRpb25zJztcbmltcG9ydCAqIGFzIHNmbiBmcm9tICdhd3MtY2RrLWxpYi9hd3Mtc3RlcGZ1bmN0aW9ucyc7XG5pbXBvcnQgKiBhcyB0YXNrcyBmcm9tICdhd3MtY2RrLWxpYi9hd3Mtc3RlcGZ1bmN0aW9ucy10YXNrcyc7XG5pbXBvcnQgKiBhcyBpYW0gZnJvbSAnYXdzLWNkay1saWIvYXdzLWlhbSc7XG5pbXBvcnQgKiBhcyBsb2dzIGZyb20gJ2F3cy1jZGstbGliL2F3cy1sb2dzJztcbmltcG9ydCAqIGFzIHBhdGggZnJvbSAncGF0aCc7XG5cbmV4cG9ydCBpbnRlcmZhY2UgU2FnZU1ha2VySW5mZXJlbmNlTW9uaXRvcmluZ1N0YWNrUHJvcHMgZXh0ZW5kcyBjZGsuU3RhY2tQcm9wcyB7XG4gIC8qKlxuICAgKiBFeGlzdGluZyBTMyBidWNrZXQgbmFtZSB3aGVyZSBTYWdlTWFrZXIgZGF0YSBjYXB0dXJlIGlzIHN0b3JlZFxuICAgKiBJZiBub3QgcHJvdmlkZWQsIHdpbGwgbG9vayBmb3IgZGVmYXVsdCBTYWdlTWFrZXIgYnVja2V0XG4gICAqL1xuICBkYXRhQ2FwdHVyZVMzQnVja2V0TmFtZT86IHN0cmluZztcblxuICAvKipcbiAgICogY2RrIHN0YWNrIHByZWZpeCBcbiAgICogQGRlZmF1bHQgJydcbiAgICovXG4gIHN0YWNrUHJlZml4Pzogc3RyaW5nO1xuICBcbiAgLyoqXG4gICAqIFMzIHByZWZpeCB3aGVyZSBkYXRhIGNhcHR1cmUgZmlsZXMgYXJlIHN0b3JlZFxuICAgKi9cbiAgZGF0YUNhcHR1cmVTM1ByZWZpeD86IHN0cmluZztcblxuICAvKipcbiAgICogTUxmbG93IHRyYWNraW5nIHNlcnZlciBBUk5cbiAgICogRm9ybWF0OiBhcm46YXdzOnNhZ2VtYWtlcjpyZWdpb246YWNjb3VudC1pZDptbGZsb3ctYXBwL2FwcC1pZFxuICAgKi9cbiAgbWxmbG93VHJhY2tpbmdTZXJ2ZXJBcm46IHN0cmluZztcblxuICAvKipcbiAgICogTUxmbG93IGV4cGVyaW1lbnQgbmFtZVxuICAgKiBAZGVmYXVsdCBzYWdlbWFrZXJFbmRwb2ludE5hbWVcbiAgICovXG4gIG1sZmxvd0V4cGVyaW1lbnROYW1lOiBzdHJpbmc7XG5cbiAgLyoqXG4gICAqIFNhZ2VNYWtlciBlbmRwb2ludCBuYW1lIHRvIG1vbml0b3JcbiAgICovXG4gIHNhZ2VtYWtlckVuZHBvaW50TmFtZTogc3RyaW5nO1xuXG4gIC8qKlxuICAgKiBCZWRyb2NrIG1vZGVsIElEIGZvciBHZW5BSSBldmFsdWF0aW9uc1xuICAgKiBAZGVmYXVsdCAnYmVkcm9jazovZ2xvYmFsLmFudGhyb3BpYy5jbGF1ZGUtc29ubmV0LTQtMjAyNTA1MTQtdjE6MCdcbiAgICovXG4gIGJlZHJvY2tNb2RlbElkOiBzdHJpbmc7XG59XG5cbmV4cG9ydCBjbGFzcyBTYWdlTWFrZXJJbmZlcmVuY2VNb25pdG9yaW5nU3RhY2sgZXh0ZW5kcyBjZGsuU3RhY2sge1xuICBwdWJsaWMgcmVhZG9ubHkgcHJvY2Vzc29yRnVuY3Rpb246IGxhbWJkYS5GdW5jdGlvbjtcbiAgcHVibGljIHJlYWRvbmx5IHN0YXRlTWFjaGluZTogc2ZuLlN0YXRlTWFjaGluZTtcblxuICBjb25zdHJ1Y3RvcihzY29wZTogQ29uc3RydWN0LCBpZDogc3RyaW5nLCBwcm9wczogU2FnZU1ha2VySW5mZXJlbmNlTW9uaXRvcmluZ1N0YWNrUHJvcHMpIHtcbiAgICBzdXBlcihzY29wZSwgaWQsIHByb3BzKTtcblxuICAgIGNvbnN0IGRhdGFDYXB0dXJlUHJlZml4ID0gcHJvcHMuZGF0YUNhcHR1cmVTM1ByZWZpeDtcbiAgICBjb25zdCBtbGZsb3dFeHBlcmltZW50TmFtZSA9IHByb3BzLm1sZmxvd0V4cGVyaW1lbnROYW1lO1xuICAgIGNvbnN0IGJlZHJvY2tNb2RlbElkID0gcHJvcHMuYmVkcm9ja01vZGVsSWQ7XG4gICAgY29uc3Qgc3RhY2tQcmVmaXggPSBwcm9wcy5zdGFja1ByZWZpeDtcblxuICAgIC8vIFJlZmVyZW5jZSBleGlzdGluZyBTMyBidWNrZXQgd2hlcmUgU2FnZU1ha2VyIGRhdGEgY2FwdHVyZSBpcyBzdG9yZWRcbiAgICBjb25zdCBkYXRhQ2FwdHVyZVMzQnVja2V0TmFtZSA9IHByb3BzLmRhdGFDYXB0dXJlUzNCdWNrZXROYW1lIHx8XG4gICAgICBgc2FnZW1ha2VyLSR7dGhpcy5yZWdpb259LSR7dGhpcy5hY2NvdW50fWA7XG5cbiAgICBjb25zdCBkYXRDYXB0dXJlQnVja2V0ID0gczMuQnVja2V0LmZyb21CdWNrZXROYW1lKFxuICAgICAgdGhpcyxcbiAgICAgICdEYXRhQ2FwdHVyZUJ1Y2tldCcsXG4gICAgICBkYXRhQ2FwdHVyZVMzQnVja2V0TmFtZVxuICAgICk7XG5cbiAgICAvLyBMYW1iZGEgZXhlY3V0aW9uIHJvbGVcbiAgICBjb25zdCBsYW1iZGFSb2xlID0gbmV3IGlhbS5Sb2xlKHRoaXMsICdQcm9jZXNzb3JMYW1iZGFSb2xlJywge1xuICAgICAgcm9sZU5hbWU6IGBzYWdlbWFrZXItZGF0YS1jYXB0dXJlLXByb2Nlc3NpbmctJHtzdGFja1ByZWZpeH1gLFxuICAgICAgYXNzdW1lZEJ5OiBuZXcgaWFtLlNlcnZpY2VQcmluY2lwYWwoJ2xhbWJkYS5hbWF6b25hd3MuY29tJyksXG4gICAgICBkZXNjcmlwdGlvbjogJ1JvbGUgZm9yIExhbWJkYSBmdW5jdGlvbiB0aGF0IHByb2Nlc3NlcyBTYWdlTWFrZXIgZGF0YSBjYXB0dXJlIGZpbGVzJyxcbiAgICAgIG1hbmFnZWRQb2xpY2llczogW1xuICAgICAgICBpYW0uTWFuYWdlZFBvbGljeS5mcm9tQXdzTWFuYWdlZFBvbGljeU5hbWUoJ3NlcnZpY2Utcm9sZS9BV1NMYW1iZGFCYXNpY0V4ZWN1dGlvblJvbGUnKSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICAvLyBHcmFudCBMYW1iZGEgYWNjZXNzIHRvIFMzIGJ1Y2tldCBmb3IgcmVhZGluZyBkYXRhIGNhcHR1cmUgZmlsZXMgYW5kIHdyaXRpbmcgTUxmbG93IHRyYWNlIGFydGlmYWN0c1xuICAgIGRhdENhcHR1cmVCdWNrZXQuZ3JhbnRSZWFkV3JpdGUobGFtYmRhUm9sZSk7XG5cbiAgICAvLyBHcmFudCBMYW1iZGEgYWNjZXNzIHRvIFNhZ2VNYWtlciBNTGZsb3cgQXBwXG4gICAgbGFtYmRhUm9sZS5hZGRUb1BvbGljeShuZXcgaWFtLlBvbGljeVN0YXRlbWVudCh7XG4gICAgICBlZmZlY3Q6IGlhbS5FZmZlY3QuQUxMT1csXG4gICAgICBhY3Rpb25zOiBbXG4gICAgICAgICdzYWdlbWFrZXI6RGVzY3JpYmVNbGZsb3dBcHAnLFxuICAgICAgICAnc2FnZW1ha2VyOkNhbGxNbGZsb3dBcHBBcGknLFxuICAgICAgICAnc2FnZW1ha2VyOkxpc3RNbGZsb3dBcHBzJyxcbiAgICAgICAgLy8nc2FnZW1ha2VyOkNyZWF0ZVByZXNpZ25lZE1sZmxvd0FwcFVybCcsXG4gICAgICBdLFxuICAgICAgcmVzb3VyY2VzOiBbcHJvcHMubWxmbG93VHJhY2tpbmdTZXJ2ZXJBcm5dLFxuICAgIH0pKTtcblxuICAgIC8vIEdyYW50IExhbWJkYSBhY2Nlc3MgdG8gQmVkcm9jayBmb3IgZXZhbHVhdGlvbnNcbiAgICBsYW1iZGFSb2xlLmFkZFRvUG9saWN5KG5ldyBpYW0uUG9saWN5U3RhdGVtZW50KHtcbiAgICAgIGVmZmVjdDogaWFtLkVmZmVjdC5BTExPVyxcbiAgICAgIGFjdGlvbnM6IFtcbiAgICAgICAgJ2JlZHJvY2s6SW52b2tlTW9kZWwnLFxuICAgICAgICAnYmVkcm9jazpJbnZva2VNb2RlbFdpdGhSZXNwb25zZVN0cmVhbScsXG4gICAgICAgICdiZWRyb2NrOkxpc3RGb3VuZGF0aW9uTW9kZWxzJyxcbiAgICAgICAgJ2JlZHJvY2s6R2V0Rm91bmRhdGlvbk1vZGVsJyxcbiAgICAgIF0sXG4gICAgICByZXNvdXJjZXM6IFsnKiddLCAvLyBCZWRyb2NrIG1vZGVscyBkb24ndCBoYXZlIHNwZWNpZmljIEFSTnNcbiAgICB9KSk7XG5cbiAgICAvLyBMYW1iZGEgZnVuY3Rpb24gdG8gcHJvY2VzcyBkYXRhIGNhcHR1cmUgZmlsZXNcbiAgICB0aGlzLnByb2Nlc3NvckZ1bmN0aW9uID0gbmV3IGxhbWJkYS5Eb2NrZXJJbWFnZUZ1bmN0aW9uKHRoaXMsICdQcm9jZXNzb3JMYW1iZGEnLCB7XG4gICAgICBmdW5jdGlvbk5hbWU6IGBzYWdlbWFrZXItZGF0YS1jYXB0dXJlLXByb2Nlc3Nvci0ke3N0YWNrUHJlZml4fWAsXG4gICAgICBjb2RlOiBsYW1iZGEuRG9ja2VySW1hZ2VDb2RlLmZyb21JbWFnZUFzc2V0KFxuICAgICAgICBwYXRoLmpvaW4oX19kaXJuYW1lLCAnLi4vbGFtYmRhJylcbiAgICAgICksXG4gICAgICBkZXNjcmlwdGlvbjogJ1Byb2Nlc3NlcyBTYWdlTWFrZXIgZGF0YSBjYXB0dXJlIGZpbGVzIGFuZCBsb2dzIHRvIE1MZmxvdyB3aXRoIEdlbkFJIGV2YWx1YXRpb25zJyxcbiAgICAgIG1lbW9yeVNpemU6IDMwMDgsIC8vIEluY3JlYXNlZCBmb3IgTUxmbG93IGFuZCBldmFsdWF0aW9uIGxpYnJhcmllc1xuICAgICAgdGltZW91dDogY2RrLkR1cmF0aW9uLm1pbnV0ZXMoMTUpLCAvLyBJbmNyZWFzZWQgZm9yIEdlbkFJIGV2YWx1YXRpb25zXG4gICAgICBhcmNoaXRlY3R1cmU6IGxhbWJkYS5BcmNoaXRlY3R1cmUuWDg2XzY0LFxuICAgICAgcm9sZTogbGFtYmRhUm9sZSxcbiAgICAgIGVudmlyb25tZW50OiB7XG4gICAgICAgIE1MRkxPV19UUkFDS0lOR19VUkk6IHByb3BzLm1sZmxvd1RyYWNraW5nU2VydmVyQXJuLFxuICAgICAgICBNTEZMT1dfRVhQRVJJTUVOVF9OQU1FOiBwcm9wcy5tbGZsb3dFeHBlcmltZW50TmFtZSxcbiAgICAgICAgU0FHRU1BS0VSX0VORFBPSU5UX05BTUU6IHByb3BzLnNhZ2VtYWtlckVuZHBvaW50TmFtZSxcbiAgICAgICAgQkVEUk9DS19NT0RFTF9JRDogcHJvcHMuYmVkcm9ja01vZGVsSWQsXG4gICAgICAgIERBVEFfQ0FQVFVSRV9CVUNLRVQ6IGRhdGFDYXB0dXJlUzNCdWNrZXROYW1lLFxuICAgICAgICBBV1NfUk9MRV9BUk46IGxhbWJkYVJvbGUucm9sZUFybixcbiAgICAgIH0sXG4gICAgfSk7XG5cbiAgICAvLyBTdGVwIEZ1bmN0aW9ucyBTdGF0ZSBNYWNoaW5lIERlZmluaXRpb25cbiAgICAvLyBMYW1iZGEgcmVhZHMgZW50aXJlIEpTT05MIGZpbGUgYW5kIHByb2Nlc3NlcyBhbGwgcmVjb3Jkc1xuICAgIC8vIFRoaXMgaXMgc2ltcGxlciB0aGFuIERpc3RyaWJ1dGVkIE1hcCBmb3IgSlNPTkwgZmlsZXMgd2hlcmUgZWFjaCBsaW5lIGlzIGEgcmVjb3JkXG4gICAgY29uc3QgcHJvY2Vzc1Rhc2sgPSBuZXcgdGFza3MuTGFtYmRhSW52b2tlKHRoaXMsICdQcm9jZXNzRGF0YUNhcHR1cmVGaWxlJywge1xuICAgICAgbGFtYmRhRnVuY3Rpb246IHRoaXMucHJvY2Vzc29yRnVuY3Rpb24sXG4gICAgICBwYXlsb2FkOiBzZm4uVGFza0lucHV0LmZyb21PYmplY3Qoe1xuICAgICAgICAnczNfYnVja2V0Jzogc2ZuLkpzb25QYXRoLnN0cmluZ0F0KCckLmRldGFpbC5idWNrZXQubmFtZScpLFxuICAgICAgICAnczNfa2V5Jzogc2ZuLkpzb25QYXRoLnN0cmluZ0F0KCckLmRldGFpbC5vYmplY3Qua2V5JyksXG4gICAgICB9KSxcbiAgICAgIHJldHJ5T25TZXJ2aWNlRXhjZXB0aW9uczogdHJ1ZSxcbiAgICB9KTtcblxuICAgIC8vIENyZWF0ZSB0aGUgc3RhdGUgbWFjaGluZVxuICAgIGNvbnN0IGxvZ0dyb3VwID0gbmV3IGxvZ3MuTG9nR3JvdXAodGhpcywgJ1N0YXRlTWFjaGluZUxvZ0dyb3VwJywge1xuICAgICAgbG9nR3JvdXBOYW1lOiBgL3NvbHV0aW9uL3NhZ2VtYWtlci1lbmRwb2ludC1sbG0tbW9uaXRvcmluZy0ke3N0YWNrUHJlZml4fWAsXG4gICAgICByZXRlbnRpb246IGxvZ3MuUmV0ZW50aW9uRGF5cy5PTkVfV0VFSyxcbiAgICAgIHJlbW92YWxQb2xpY3k6IGNkay5SZW1vdmFsUG9saWN5LkRFU1RST1ksXG4gICAgfSk7XG5cbiAgICB0aGlzLnN0YXRlTWFjaGluZSA9IG5ldyBzZm4uU3RhdGVNYWNoaW5lKHRoaXMsICdNb25pdG9yaW5nU3RhdGVNYWNoaW5lJywge1xuICAgICAgc3RhdGVNYWNoaW5lTmFtZTogYHNhZ2VtYWtlci1kYXRhLWNhcHR1cmUtbW9uaXRvcmluZy0ke3N0YWNrUHJlZml4fWAsXG4gICAgICBkZWZpbml0aW9uQm9keTogc2ZuLkRlZmluaXRpb25Cb2R5LmZyb21DaGFpbmFibGUocHJvY2Vzc1Rhc2spLFxuICAgICAgbG9nczoge1xuICAgICAgICBkZXN0aW5hdGlvbjogbG9nR3JvdXAsXG4gICAgICAgIGxldmVsOiBzZm4uTG9nTGV2ZWwuQUxMLFxuICAgICAgfSxcbiAgICAgIHRyYWNpbmdFbmFibGVkOiB0cnVlLFxuICAgICAgdGltZW91dDogY2RrLkR1cmF0aW9uLmhvdXJzKDEpLFxuICAgIH0pO1xuXG4gICAgLy8gR3JhbnQgU3RlcCBGdW5jdGlvbnMgcGVybWlzc2lvbiB0byBpbnZva2UgTGFtYmRhXG4gICAgdGhpcy5wcm9jZXNzb3JGdW5jdGlvbi5ncmFudEludm9rZSh0aGlzLnN0YXRlTWFjaGluZSk7XG5cbiAgICAvLyBTMyBldmVudCBub3RpZmljYXRpb24gdG8gdHJpZ2dlciBTdGVwIEZ1bmN0aW9uc1xuICAgIC8vIFRoaXMgd2lsbCBiZSB0cmlnZ2VyZWQgd2hlbiBuZXcgLmpzb25sIGZpbGVzIGFyZSBjcmVhdGVkIGluIHRoZSBkYXRhIGNhcHR1cmUgcHJlZml4XG4gICAgY29uc3QgZXZlbnRQYXR0ZXJuID0ge1xuICAgICAgZGV0YWlsOiB7XG4gICAgICAgIGJ1Y2tldDoge1xuICAgICAgICAgIG5hbWU6IFtkYXRhQ2FwdHVyZVMzQnVja2V0TmFtZV0sXG4gICAgICAgIH0sXG4gICAgICAgIG9iamVjdDoge1xuICAgICAgICAgIGtleTogW3tcbiAgICAgICAgICAgIC8vIHByZWZpeDogZGF0YUNhcHR1cmVQcmVmaXgsXG4gICAgICAgICAgICBzdWZmaXg6ICcuanNvbmwnLFxuICAgICAgICAgIH1dLFxuICAgICAgICB9LFxuICAgICAgfSxcbiAgICB9O1xuXG4gICAgLy8gQ3JlYXRlIEV2ZW50QnJpZGdlIHJ1bGUgdG8gdHJpZ2dlciBTdGVwIEZ1bmN0aW9uc1xuICAgIGNvbnN0IHJ1bGUgPSBuZXcgY2RrLmF3c19ldmVudHMuUnVsZSh0aGlzLCAnUzNFdmVudFJ1bGUnLCB7XG4gICAgICBydWxlTmFtZTogYHNhZ2VtYWtlci1zMy1ldmVudC1ydWxlLSR7c3RhY2tQcmVmaXh9YCxcbiAgICAgIGV2ZW50UGF0dGVybjoge1xuICAgICAgICBzb3VyY2U6IFsnYXdzLnMzJ10sXG4gICAgICAgIGRldGFpbFR5cGU6IFsnT2JqZWN0IENyZWF0ZWQnXSxcbiAgICAgICAgZGV0YWlsOiBldmVudFBhdHRlcm4uZGV0YWlsLFxuICAgICAgfSxcbiAgICB9KTtcblxuICAgIHJ1bGUuYWRkVGFyZ2V0KG5ldyBjZGsuYXdzX2V2ZW50c190YXJnZXRzLlNmblN0YXRlTWFjaGluZSh0aGlzLnN0YXRlTWFjaGluZSkpO1xuXG4gICAgLy8gRW5hYmxlIFMzIGV2ZW50IG5vdGlmaWNhdGlvbnMgdG8gRXZlbnRCcmlkZ2VcbiAgICAvLyBOb3RlOiBUaGlzIHJlcXVpcmVzIGVuYWJsaW5nIEV2ZW50QnJpZGdlIG5vdGlmaWNhdGlvbnMgb24gdGhlIFMzIGJ1Y2tldCBtYW51YWxseVxuICAgIC8vIG9yIHVzaW5nIGEgY3VzdG9tIHJlc291cmNlIHRvIGVuYWJsZSBpdFxuICAgIG5ldyBjZGsuQ2ZuT3V0cHV0KHRoaXMsICdTM0J1Y2tldEV2ZW50QnJpZGdlTm90ZScsIHtcbiAgICAgIHZhbHVlOiBgUGxlYXNlIGVuYWJsZSBFdmVudEJyaWRnZSBub3RpZmljYXRpb25zIG9uIGJ1Y2tldCAke2RhdGFDYXB0dXJlUzNCdWNrZXROYW1lfSBtYW51YWxseSBvciB2aWEgQVdTIENMSTogYXdzIHMzYXBpIHB1dC1idWNrZXQtbm90aWZpY2F0aW9uLWNvbmZpZ3VyYXRpb24gLS1idWNrZXQgJHtkYXRhQ2FwdHVyZVMzQnVja2V0TmFtZX0gLS1ub3RpZmljYXRpb24tY29uZmlndXJhdGlvbiAne1wiRXZlbnRCcmlkZ2VDb25maWd1cmF0aW9uXCI6IHt9fSdgLFxuICAgICAgZGVzY3JpcHRpb246ICdDb21tYW5kIHRvIGVuYWJsZSBFdmVudEJyaWRnZSBub3RpZmljYXRpb25zIG9uIFMzIGJ1Y2tldCcsXG4gICAgfSk7XG5cbiAgICAvLyBPdXRwdXRzXG4gICAgbmV3IGNkay5DZm5PdXRwdXQodGhpcywgJ1Byb2Nlc3NvckxhbWJkYUFybicsIHtcbiAgICAgIHZhbHVlOiB0aGlzLnByb2Nlc3NvckZ1bmN0aW9uLmZ1bmN0aW9uQXJuLFxuICAgICAgZGVzY3JpcHRpb246ICdBUk4gb2YgdGhlIExhbWJkYSBmdW5jdGlvbiB0aGF0IHByb2Nlc3NlcyBkYXRhIGNhcHR1cmUgZmlsZXMnLFxuICAgICAgZXhwb3J0TmFtZTogYFByb2Nlc3NvckxhbWJkYUFybi0ke3N0YWNrUHJlZml4fWAsXG4gICAgfSk7XG5cbiAgICBuZXcgY2RrLkNmbk91dHB1dCh0aGlzLCAnU3RhdGVNYWNoaW5lQXJuJywge1xuICAgICAgdmFsdWU6IHRoaXMuc3RhdGVNYWNoaW5lLnN0YXRlTWFjaGluZUFybixcbiAgICAgIGRlc2NyaXB0aW9uOiAnQVJOIG9mIHRoZSBTdGVwIEZ1bmN0aW9ucyBzdGF0ZSBtYWNoaW5lJyxcbiAgICAgIGV4cG9ydE5hbWU6IGBTdGF0ZU1hY2hpbmVBcm4tJHtzdGFja1ByZWZpeH1gLFxuICAgIH0pO1xuXG4gICAgbmV3IGNkay5DZm5PdXRwdXQodGhpcywgJ0RhdGFDYXB0dXJlQnVja2V0TmFtZScsIHtcbiAgICAgIHZhbHVlOiBkYXRhQ2FwdHVyZVMzQnVja2V0TmFtZSxcbiAgICAgIGRlc2NyaXB0aW9uOiAnUzMgYnVja2V0IHdoZXJlIGRhdGEgY2FwdHVyZSBmaWxlcyBhcmUgc3RvcmVkJyxcbiAgICAgIGV4cG9ydE5hbWU6IGBEYXRhQ2FwdHVyZUJ1Y2tldE5hbWUtJHtzdGFja1ByZWZpeH1gLFxuICAgIH0pO1xuICB9XG59XG4iXX0=
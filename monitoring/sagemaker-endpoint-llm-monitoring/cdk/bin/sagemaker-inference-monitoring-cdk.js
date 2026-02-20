#!/usr/bin/env node
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
require("source-map-support/register");
const cdk = __importStar(require("aws-cdk-lib"));
const sagemaker_inference_monitoring_stack_1 = require("../lib/sagemaker-inference-monitoring-stack");
const dotenv_1 = require("dotenv");
const path = __importStar(require("path"));
// Load environment variables from .env file
(0, dotenv_1.config)({ path: path.join(__dirname, '../.env') });
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
new sagemaker_inference_monitoring_stack_1.SageMakerInferenceMonitoringStack(app, stackId, {
    ...config,
    description: 'Infrastructure for automated SageMaker endpoint monitoring with MLflow and GenAI evaluations',
    tags: {
        Project: 'SageMaker-Inference-Monitoring',
        ManagedBy: 'CDK',
    },
});
app.synth();
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2FnZW1ha2VyLWluZmVyZW5jZS1tb25pdG9yaW5nLWNkay5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbInNhZ2VtYWtlci1pbmZlcmVuY2UtbW9uaXRvcmluZy1jZGsudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFDQSx1Q0FBcUM7QUFDckMsaURBQW1DO0FBQ25DLHNHQUFnRztBQUNoRyxtQ0FBOEM7QUFDOUMsMkNBQTZCO0FBRTdCLDRDQUE0QztBQUM1QyxJQUFBLGVBQVUsRUFBQyxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFNBQVMsRUFBRSxTQUFTLENBQUMsRUFBRSxDQUFDLENBQUM7QUFFdEQsTUFBTSxHQUFHLEdBQUcsSUFBSSxHQUFHLENBQUMsR0FBRyxFQUFFLENBQUM7QUFFMUIsc0VBQXNFO0FBQ3RFLE1BQU0sV0FBVyxHQUFHLE9BQU8sQ0FBQyxHQUFHLENBQUMsWUFBWSxDQUFDO0FBQzdDLElBQUksQ0FBQyxXQUFXLEVBQUUsQ0FBQztJQUNqQixNQUFNLElBQUksS0FBSyxDQUFDLHFGQUFxRixDQUFDLENBQUM7QUFDekcsQ0FBQztBQUVELHVEQUF1RDtBQUN2RCxNQUFNLHFCQUFxQixHQUFHLE9BQU8sQ0FBQyxHQUFHLENBQUMsdUJBQXVCLENBQUM7QUFDbEUsSUFBSSxDQUFDLHFCQUFxQixFQUFFLENBQUM7SUFDM0IsTUFBTSxJQUFJLEtBQUssQ0FBQyxnR0FBZ0csQ0FBQyxDQUFDO0FBQ3BILENBQUM7QUFFRCwwREFBMEQ7QUFDMUQsTUFBTSx1QkFBdUIsR0FBRyxPQUFPLENBQUMsR0FBRyxDQUFDLG1CQUFtQixDQUFDO0FBQ2hFLElBQUksQ0FBQyx1QkFBdUIsRUFBRSxDQUFDO0lBQzdCLE1BQU0sSUFBSSxLQUFLLENBQUMsNEZBQTRGLENBQUMsQ0FBQztBQUNoSCxDQUFDO0FBRUQsMkRBQTJEO0FBQzNELE1BQU0sTUFBTSxHQUFHO0lBQ2IsR0FBRyxFQUFFO1FBQ0gsT0FBTyxFQUFFLE9BQU8sQ0FBQyxHQUFHLENBQUMsbUJBQW1CO1FBQ3hDLE1BQU0sRUFBRSxPQUFPLENBQUMsR0FBRyxDQUFDLGtCQUFrQixJQUFJLFdBQVc7S0FDdEQ7SUFFRCwyQkFBMkI7SUFDM0IsdUJBQXVCO0lBRXZCLG9DQUFvQztJQUNwQyxxQkFBcUI7SUFFckIsdURBQXVEO0lBQ3ZELFdBQVc7SUFFWCxtREFBbUQ7SUFDbkQsNENBQTRDO0lBQzVDLHVCQUF1QixFQUFFLE9BQU8sQ0FBQyxHQUFHLENBQUMsc0JBQXNCO0lBRTNELDZDQUE2QztJQUM3Qyw2Q0FBNkM7SUFDN0MsbUJBQW1CLEVBQUUsT0FBTyxDQUFDLEdBQUcsQ0FBQyxzQkFBc0I7UUFDckQsR0FBRyxxQkFBcUIsRUFBRTtJQUU1QixtQ0FBbUM7SUFDbkMsNkRBQTZEO0lBQzdELG9CQUFvQixFQUFFLE9BQU8sQ0FBQyxHQUFHLENBQUMsc0JBQXNCO1FBQ3RELEdBQUcscUJBQXFCLEVBQUU7SUFFNUIsMENBQTBDO0lBQzFDLGNBQWMsRUFBRSxPQUFPLENBQUMsR0FBRyxDQUFDLGdCQUFnQjtRQUMxQyx5REFBeUQ7Q0FDNUQsQ0FBQztBQUVGLHlFQUF5RTtBQUN6RSxNQUFNLE9BQU8sR0FBRyxNQUFNLENBQUMsV0FBVztJQUNoQyxDQUFDLENBQUMscUNBQXFDLE1BQU0sQ0FBQyxXQUFXLEVBQUU7SUFDM0QsQ0FBQyxDQUFDLG1DQUFtQyxDQUFDO0FBRXhDLElBQUksd0VBQWlDLENBQUMsR0FBRyxFQUFFLE9BQU8sRUFBRTtJQUNsRCxHQUFHLE1BQU07SUFDVCxXQUFXLEVBQUUsOEZBQThGO0lBRTNHLElBQUksRUFBRTtRQUNKLE9BQU8sRUFBRSxnQ0FBZ0M7UUFDekMsU0FBUyxFQUFFLEtBQUs7S0FDakI7Q0FDRixDQUFDLENBQUM7QUFFSCxHQUFHLENBQUMsS0FBSyxFQUFFLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIjIS91c3IvYmluL2VudiBub2RlXG5pbXBvcnQgJ3NvdXJjZS1tYXAtc3VwcG9ydC9yZWdpc3Rlcic7XG5pbXBvcnQgKiBhcyBjZGsgZnJvbSAnYXdzLWNkay1saWInO1xuaW1wb3J0IHsgU2FnZU1ha2VySW5mZXJlbmNlTW9uaXRvcmluZ1N0YWNrIH0gZnJvbSAnLi4vbGliL3NhZ2VtYWtlci1pbmZlcmVuY2UtbW9uaXRvcmluZy1zdGFjayc7XG5pbXBvcnQgeyBjb25maWcgYXMgbG9hZERvdGVudiB9IGZyb20gJ2RvdGVudic7XG5pbXBvcnQgKiBhcyBwYXRoIGZyb20gJ3BhdGgnO1xuXG4vLyBMb2FkIGVudmlyb25tZW50IHZhcmlhYmxlcyBmcm9tIC5lbnYgZmlsZVxubG9hZERvdGVudih7IHBhdGg6IHBhdGguam9pbihfX2Rpcm5hbWUsICcuLi8uZW52JykgfSk7XG5cbmNvbnN0IGFwcCA9IG5ldyBjZGsuQXBwKCk7XG5cbi8vIFJFUVVJUkVEOiBTdGFjayBwcmVmaXggLSBtdXN0IGJlIHByb3ZpZGVkIGZvciB1bmlxdWUgcmVzb3VyY2UgbmFtZXNcbmNvbnN0IHN0YWNrUHJlZml4ID0gcHJvY2Vzcy5lbnYuU1RBQ0tfUFJFRklYO1xuaWYgKCFzdGFja1ByZWZpeCkge1xuICB0aHJvdyBuZXcgRXJyb3IoJ1NUQUNLX1BSRUZJWCBpcyByZXF1aXJlZC4gUGxlYXNlIHNldCBpdCBpbiAuZW52IGZpbGUgb3IgYXMgYW4gZW52aXJvbm1lbnQgdmFyaWFibGUuJyk7XG59XG5cbi8vIFJFUVVJUkVEOiBTYWdlTWFrZXIgZW5kcG9pbnQgbmFtZSAtIG11c3QgYmUgcHJvdmlkZWRcbmNvbnN0IHNhZ2VtYWtlckVuZHBvaW50TmFtZSA9IHByb2Nlc3MuZW52LlNBR0VNQUtFUl9FTkRQT0lOVF9OQU1FO1xuaWYgKCFzYWdlbWFrZXJFbmRwb2ludE5hbWUpIHtcbiAgdGhyb3cgbmV3IEVycm9yKCdTQUdFTUFLRVJfRU5EUE9JTlRfTkFNRSBpcyByZXF1aXJlZC4gUGxlYXNlIHNldCBpdCBpbiAuZW52IGZpbGUgb3IgYXMgYW4gZW52aXJvbm1lbnQgdmFyaWFibGUuJyk7XG59XG5cbi8vIFJFUVVJUkVEOiBNTGZsb3cgdHJhY2tpbmcgc2VydmVyIEFSTiAtIG11c3QgYmUgcHJvdmlkZWRcbmNvbnN0IG1sZmxvd1RyYWNraW5nU2VydmVyQXJuID0gcHJvY2Vzcy5lbnYuTUxGTE9XX1RSQUNLSU5HX1VSSTtcbmlmICghbWxmbG93VHJhY2tpbmdTZXJ2ZXJBcm4pIHtcbiAgdGhyb3cgbmV3IEVycm9yKCdNTEZMT1dfVFJBQ0tJTkdfVVJJIGlzIHJlcXVpcmVkLiBQbGVhc2Ugc2V0IGl0IGluIC5lbnYgZmlsZSBvciBhcyBhbiBlbnZpcm9ubWVudCB2YXJpYWJsZS4nKTtcbn1cblxuLy8gQ29uZmlndXJhdGlvbiB3aXRoIHNtYXJ0IGRlZmF1bHRzIGJhc2VkIG9uIGVuZHBvaW50IG5hbWVcbmNvbnN0IGNvbmZpZyA9IHtcbiAgZW52OiB7XG4gICAgYWNjb3VudDogcHJvY2Vzcy5lbnYuQ0RLX0RFRkFVTFRfQUNDT1VOVCxcbiAgICByZWdpb246IHByb2Nlc3MuZW52LkNES19ERUZBVUxUX1JFR0lPTiB8fCAndXMtd2VzdC0yJyxcbiAgfSxcblxuICAvLyBSRVFVSVJFRDogTUxmbG93IGFwcCBBUk5cbiAgbWxmbG93VHJhY2tpbmdTZXJ2ZXJBcm4sXG5cbiAgLy8gUkVRVUlSRUQ6IFNhZ2VNYWtlciBlbmRwb2ludCBuYW1lXG4gIHNhZ2VtYWtlckVuZHBvaW50TmFtZSxcblxuICAvLyBSRVFVSVJFRDogQ0RLIHN0YWNrIHByZWZpeCBmb3IgdW5pcXVlIHJlc291cmNlIG5hbWVzXG4gIHN0YWNrUHJlZml4LFxuXG4gIC8vIE9QVElPTkFMOiBTMyBidWNrZXQgd2hlcmUgZGF0YSBjYXB0dXJlIGlzIHN0b3JlZFxuICAvLyBEZWZhdWx0cyB0bzogc2FnZW1ha2VyLXtyZWdpb259LXthY2NvdW50fVxuICBkYXRhQ2FwdHVyZVMzQnVja2V0TmFtZTogcHJvY2Vzcy5lbnYuREFUQV9DQVBUVVJFX1MzX0JVQ0tFVCxcblxuICAvLyBPUFRJT05BTDogUzMgcHJlZml4IGZvciBkYXRhIGNhcHR1cmUgZmlsZXNcbiAgLy8gRGVmYXVsdHMgdG86IHtlbmRwb2ludC1uYW1lfS1kYXRhLWNhcHR1cmUvXG4gIGRhdGFDYXB0dXJlUzNQcmVmaXg6IHByb2Nlc3MuZW52LkRBVEFfQ0FQVFVSRV9TM19QUkVGSVggfHxcbiAgICBgJHtzYWdlbWFrZXJFbmRwb2ludE5hbWV9YCxcblxuICAvLyBPUFRJT05BTDogTUxmbG93IGV4cGVyaW1lbnQgbmFtZVxuICAvLyBEZWZhdWx0cyB0bzogc2FnZW1ha2VyLWVuZHBvaW50LXtlbmRwb2ludC1uYW1lfS1tb25pdG9yaW5nXG4gIG1sZmxvd0V4cGVyaW1lbnROYW1lOiBwcm9jZXNzLmVudi5NTEZMT1dfRVhQRVJJTUVOVF9OQU1FIHx8XG4gICAgYCR7c2FnZW1ha2VyRW5kcG9pbnROYW1lfWAsXG5cbiAgLy8gT1BUSU9OQUw6IEJlZHJvY2sgbW9kZWwgZm9yIGV2YWx1YXRpb25zXG4gIGJlZHJvY2tNb2RlbElkOiBwcm9jZXNzLmVudi5CRURST0NLX01PREVMX0lEIHx8XG4gICAgJ2JlZHJvY2s6L2dsb2JhbC5hbnRocm9waWMuY2xhdWRlLXNvbm5ldC00LTIwMjUwNTE0LXYxOjAnLFxufTtcblxuLy8gQ3JlYXRlIHVuaXF1ZSBzdGFjayBJRCBiYXNlZCBvbiBwcmVmaXggdG8gc3VwcG9ydCBtdWx0aXBsZSBkZXBsb3ltZW50c1xuY29uc3Qgc3RhY2tJZCA9IGNvbmZpZy5zdGFja1ByZWZpeFxuICA/IGBTYWdlTWFrZXJJbmZlcmVuY2VNb25pdG9yaW5nU3RhY2stJHtjb25maWcuc3RhY2tQcmVmaXh9YFxuICA6ICdTYWdlTWFrZXJJbmZlcmVuY2VNb25pdG9yaW5nU3RhY2snO1xuXG5uZXcgU2FnZU1ha2VySW5mZXJlbmNlTW9uaXRvcmluZ1N0YWNrKGFwcCwgc3RhY2tJZCwge1xuICAuLi5jb25maWcsXG4gIGRlc2NyaXB0aW9uOiAnSW5mcmFzdHJ1Y3R1cmUgZm9yIGF1dG9tYXRlZCBTYWdlTWFrZXIgZW5kcG9pbnQgbW9uaXRvcmluZyB3aXRoIE1MZmxvdyBhbmQgR2VuQUkgZXZhbHVhdGlvbnMnLFxuXG4gIHRhZ3M6IHtcbiAgICBQcm9qZWN0OiAnU2FnZU1ha2VyLUluZmVyZW5jZS1Nb25pdG9yaW5nJyxcbiAgICBNYW5hZ2VkQnk6ICdDREsnLFxuICB9LFxufSk7XG5cbmFwcC5zeW50aCgpO1xuIl19
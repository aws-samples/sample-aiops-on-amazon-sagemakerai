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
// REQUIRED: S3 bucket for data capture - must be provided
const dataCaptureS3BucketName = process.env.DATA_CAPTURE_S3_BUCKET;
if (!dataCaptureS3BucketName) {
    throw new Error('DATA_CAPTURE_S3_BUCKET is required. Please set it in .env file or as an environment variable.');
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
    // REQUIRED: S3 bucket where data capture is stored
    dataCaptureS3BucketName,
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2FnZW1ha2VyLWluZmVyZW5jZS1tb25pdG9yaW5nLWNkay5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbInNhZ2VtYWtlci1pbmZlcmVuY2UtbW9uaXRvcmluZy1jZGsudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFDQSx1Q0FBcUM7QUFDckMsaURBQW1DO0FBQ25DLHNHQUFnRztBQUNoRyxtQ0FBOEM7QUFDOUMsMkNBQTZCO0FBRTdCLDRDQUE0QztBQUM1QyxJQUFBLGVBQVUsRUFBQyxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFNBQVMsRUFBRSxTQUFTLENBQUMsRUFBRSxDQUFDLENBQUM7QUFFdEQsTUFBTSxHQUFHLEdBQUcsSUFBSSxHQUFHLENBQUMsR0FBRyxFQUFFLENBQUM7QUFFMUIsc0VBQXNFO0FBQ3RFLE1BQU0sV0FBVyxHQUFHLE9BQU8sQ0FBQyxHQUFHLENBQUMsWUFBWSxDQUFDO0FBQzdDLElBQUksQ0FBQyxXQUFXLEVBQUUsQ0FBQztJQUNqQixNQUFNLElBQUksS0FBSyxDQUFDLHFGQUFxRixDQUFDLENBQUM7QUFDekcsQ0FBQztBQUVELHVEQUF1RDtBQUN2RCxNQUFNLHFCQUFxQixHQUFHLE9BQU8sQ0FBQyxHQUFHLENBQUMsdUJBQXVCLENBQUM7QUFDbEUsSUFBSSxDQUFDLHFCQUFxQixFQUFFLENBQUM7SUFDM0IsTUFBTSxJQUFJLEtBQUssQ0FBQyxnR0FBZ0csQ0FBQyxDQUFDO0FBQ3BILENBQUM7QUFFRCwwREFBMEQ7QUFDMUQsTUFBTSx1QkFBdUIsR0FBRyxPQUFPLENBQUMsR0FBRyxDQUFDLG1CQUFtQixDQUFDO0FBQ2hFLElBQUksQ0FBQyx1QkFBdUIsRUFBRSxDQUFDO0lBQzdCLE1BQU0sSUFBSSxLQUFLLENBQUMsNEZBQTRGLENBQUMsQ0FBQztBQUNoSCxDQUFDO0FBRUQsMERBQTBEO0FBQzFELE1BQU0sdUJBQXVCLEdBQUcsT0FBTyxDQUFDLEdBQUcsQ0FBQyxzQkFBc0IsQ0FBQztBQUNuRSxJQUFJLENBQUMsdUJBQXVCLEVBQUUsQ0FBQztJQUM3QixNQUFNLElBQUksS0FBSyxDQUFDLCtGQUErRixDQUFDLENBQUM7QUFDbkgsQ0FBQztBQUVELDJEQUEyRDtBQUMzRCxNQUFNLE1BQU0sR0FBRztJQUNiLEdBQUcsRUFBRTtRQUNILE9BQU8sRUFBRSxPQUFPLENBQUMsR0FBRyxDQUFDLG1CQUFtQjtRQUN4QyxNQUFNLEVBQUUsT0FBTyxDQUFDLEdBQUcsQ0FBQyxrQkFBa0IsSUFBSSxXQUFXO0tBQ3REO0lBRUQsMkJBQTJCO0lBQzNCLHVCQUF1QjtJQUV2QixvQ0FBb0M7SUFDcEMscUJBQXFCO0lBRXJCLHVEQUF1RDtJQUN2RCxXQUFXO0lBRVgsbURBQW1EO0lBQ25ELHVCQUF1QjtJQUd2Qiw2Q0FBNkM7SUFDN0MsNkNBQTZDO0lBQzdDLG1CQUFtQixFQUFFLE9BQU8sQ0FBQyxHQUFHLENBQUMsc0JBQXNCO1FBQ3JELEdBQUcscUJBQXFCLEVBQUU7SUFFNUIsbUNBQW1DO0lBQ25DLDZEQUE2RDtJQUM3RCxvQkFBb0IsRUFBRSxPQUFPLENBQUMsR0FBRyxDQUFDLHNCQUFzQjtRQUN0RCxHQUFHLHFCQUFxQixFQUFFO0lBRTVCLDBDQUEwQztJQUMxQyxjQUFjLEVBQUUsT0FBTyxDQUFDLEdBQUcsQ0FBQyxnQkFBZ0I7UUFDMUMseURBQXlEO0NBQzVELENBQUM7QUFFRix5RUFBeUU7QUFDekUsTUFBTSxPQUFPLEdBQUcsTUFBTSxDQUFDLFdBQVc7SUFDaEMsQ0FBQyxDQUFDLHFDQUFxQyxNQUFNLENBQUMsV0FBVyxFQUFFO0lBQzNELENBQUMsQ0FBQyxtQ0FBbUMsQ0FBQztBQUV4QyxJQUFJLHdFQUFpQyxDQUFDLEdBQUcsRUFBRSxPQUFPLEVBQUU7SUFDbEQsR0FBRyxNQUFNO0lBQ1QsV0FBVyxFQUFFLDhGQUE4RjtJQUUzRyxJQUFJLEVBQUU7UUFDSixPQUFPLEVBQUUsZ0NBQWdDO1FBQ3pDLFNBQVMsRUFBRSxLQUFLO0tBQ2pCO0NBQ0YsQ0FBQyxDQUFDO0FBRUgsR0FBRyxDQUFDLEtBQUssRUFBRSxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiIyEvdXNyL2Jpbi9lbnYgbm9kZVxuaW1wb3J0ICdzb3VyY2UtbWFwLXN1cHBvcnQvcmVnaXN0ZXInO1xuaW1wb3J0ICogYXMgY2RrIGZyb20gJ2F3cy1jZGstbGliJztcbmltcG9ydCB7IFNhZ2VNYWtlckluZmVyZW5jZU1vbml0b3JpbmdTdGFjayB9IGZyb20gJy4uL2xpYi9zYWdlbWFrZXItaW5mZXJlbmNlLW1vbml0b3Jpbmctc3RhY2snO1xuaW1wb3J0IHsgY29uZmlnIGFzIGxvYWREb3RlbnYgfSBmcm9tICdkb3RlbnYnO1xuaW1wb3J0ICogYXMgcGF0aCBmcm9tICdwYXRoJztcblxuLy8gTG9hZCBlbnZpcm9ubWVudCB2YXJpYWJsZXMgZnJvbSAuZW52IGZpbGVcbmxvYWREb3RlbnYoeyBwYXRoOiBwYXRoLmpvaW4oX19kaXJuYW1lLCAnLi4vLmVudicpIH0pO1xuXG5jb25zdCBhcHAgPSBuZXcgY2RrLkFwcCgpO1xuXG4vLyBSRVFVSVJFRDogU3RhY2sgcHJlZml4IC0gbXVzdCBiZSBwcm92aWRlZCBmb3IgdW5pcXVlIHJlc291cmNlIG5hbWVzXG5jb25zdCBzdGFja1ByZWZpeCA9IHByb2Nlc3MuZW52LlNUQUNLX1BSRUZJWDtcbmlmICghc3RhY2tQcmVmaXgpIHtcbiAgdGhyb3cgbmV3IEVycm9yKCdTVEFDS19QUkVGSVggaXMgcmVxdWlyZWQuIFBsZWFzZSBzZXQgaXQgaW4gLmVudiBmaWxlIG9yIGFzIGFuIGVudmlyb25tZW50IHZhcmlhYmxlLicpO1xufVxuXG4vLyBSRVFVSVJFRDogU2FnZU1ha2VyIGVuZHBvaW50IG5hbWUgLSBtdXN0IGJlIHByb3ZpZGVkXG5jb25zdCBzYWdlbWFrZXJFbmRwb2ludE5hbWUgPSBwcm9jZXNzLmVudi5TQUdFTUFLRVJfRU5EUE9JTlRfTkFNRTtcbmlmICghc2FnZW1ha2VyRW5kcG9pbnROYW1lKSB7XG4gIHRocm93IG5ldyBFcnJvcignU0FHRU1BS0VSX0VORFBPSU5UX05BTUUgaXMgcmVxdWlyZWQuIFBsZWFzZSBzZXQgaXQgaW4gLmVudiBmaWxlIG9yIGFzIGFuIGVudmlyb25tZW50IHZhcmlhYmxlLicpO1xufVxuXG4vLyBSRVFVSVJFRDogTUxmbG93IHRyYWNraW5nIHNlcnZlciBBUk4gLSBtdXN0IGJlIHByb3ZpZGVkXG5jb25zdCBtbGZsb3dUcmFja2luZ1NlcnZlckFybiA9IHByb2Nlc3MuZW52Lk1MRkxPV19UUkFDS0lOR19VUkk7XG5pZiAoIW1sZmxvd1RyYWNraW5nU2VydmVyQXJuKSB7XG4gIHRocm93IG5ldyBFcnJvcignTUxGTE9XX1RSQUNLSU5HX1VSSSBpcyByZXF1aXJlZC4gUGxlYXNlIHNldCBpdCBpbiAuZW52IGZpbGUgb3IgYXMgYW4gZW52aXJvbm1lbnQgdmFyaWFibGUuJyk7XG59XG5cbi8vIFJFUVVJUkVEOiBTMyBidWNrZXQgZm9yIGRhdGEgY2FwdHVyZSAtIG11c3QgYmUgcHJvdmlkZWRcbmNvbnN0IGRhdGFDYXB0dXJlUzNCdWNrZXROYW1lID0gcHJvY2Vzcy5lbnYuREFUQV9DQVBUVVJFX1MzX0JVQ0tFVDtcbmlmICghZGF0YUNhcHR1cmVTM0J1Y2tldE5hbWUpIHtcbiAgdGhyb3cgbmV3IEVycm9yKCdEQVRBX0NBUFRVUkVfUzNfQlVDS0VUIGlzIHJlcXVpcmVkLiBQbGVhc2Ugc2V0IGl0IGluIC5lbnYgZmlsZSBvciBhcyBhbiBlbnZpcm9ubWVudCB2YXJpYWJsZS4nKTtcbn1cblxuLy8gQ29uZmlndXJhdGlvbiB3aXRoIHNtYXJ0IGRlZmF1bHRzIGJhc2VkIG9uIGVuZHBvaW50IG5hbWVcbmNvbnN0IGNvbmZpZyA9IHtcbiAgZW52OiB7XG4gICAgYWNjb3VudDogcHJvY2Vzcy5lbnYuQ0RLX0RFRkFVTFRfQUNDT1VOVCxcbiAgICByZWdpb246IHByb2Nlc3MuZW52LkNES19ERUZBVUxUX1JFR0lPTiB8fCAndXMtd2VzdC0yJyxcbiAgfSxcblxuICAvLyBSRVFVSVJFRDogTUxmbG93IGFwcCBBUk5cbiAgbWxmbG93VHJhY2tpbmdTZXJ2ZXJBcm4sXG5cbiAgLy8gUkVRVUlSRUQ6IFNhZ2VNYWtlciBlbmRwb2ludCBuYW1lXG4gIHNhZ2VtYWtlckVuZHBvaW50TmFtZSxcblxuICAvLyBSRVFVSVJFRDogQ0RLIHN0YWNrIHByZWZpeCBmb3IgdW5pcXVlIHJlc291cmNlIG5hbWVzXG4gIHN0YWNrUHJlZml4LFxuXG4gIC8vIFJFUVVJUkVEOiBTMyBidWNrZXQgd2hlcmUgZGF0YSBjYXB0dXJlIGlzIHN0b3JlZFxuICBkYXRhQ2FwdHVyZVMzQnVja2V0TmFtZSxcbiAgXG5cbiAgLy8gT1BUSU9OQUw6IFMzIHByZWZpeCBmb3IgZGF0YSBjYXB0dXJlIGZpbGVzXG4gIC8vIERlZmF1bHRzIHRvOiB7ZW5kcG9pbnQtbmFtZX0tZGF0YS1jYXB0dXJlL1xuICBkYXRhQ2FwdHVyZVMzUHJlZml4OiBwcm9jZXNzLmVudi5EQVRBX0NBUFRVUkVfUzNfUFJFRklYIHx8XG4gICAgYCR7c2FnZW1ha2VyRW5kcG9pbnROYW1lfWAsXG5cbiAgLy8gT1BUSU9OQUw6IE1MZmxvdyBleHBlcmltZW50IG5hbWVcbiAgLy8gRGVmYXVsdHMgdG86IHNhZ2VtYWtlci1lbmRwb2ludC17ZW5kcG9pbnQtbmFtZX0tbW9uaXRvcmluZ1xuICBtbGZsb3dFeHBlcmltZW50TmFtZTogcHJvY2Vzcy5lbnYuTUxGTE9XX0VYUEVSSU1FTlRfTkFNRSB8fFxuICAgIGAke3NhZ2VtYWtlckVuZHBvaW50TmFtZX1gLFxuXG4gIC8vIE9QVElPTkFMOiBCZWRyb2NrIG1vZGVsIGZvciBldmFsdWF0aW9uc1xuICBiZWRyb2NrTW9kZWxJZDogcHJvY2Vzcy5lbnYuQkVEUk9DS19NT0RFTF9JRCB8fFxuICAgICdiZWRyb2NrOi9nbG9iYWwuYW50aHJvcGljLmNsYXVkZS1zb25uZXQtNC0yMDI1MDUxNC12MTowJyxcbn07XG5cbi8vIENyZWF0ZSB1bmlxdWUgc3RhY2sgSUQgYmFzZWQgb24gcHJlZml4IHRvIHN1cHBvcnQgbXVsdGlwbGUgZGVwbG95bWVudHNcbmNvbnN0IHN0YWNrSWQgPSBjb25maWcuc3RhY2tQcmVmaXhcbiAgPyBgU2FnZU1ha2VySW5mZXJlbmNlTW9uaXRvcmluZ1N0YWNrLSR7Y29uZmlnLnN0YWNrUHJlZml4fWBcbiAgOiAnU2FnZU1ha2VySW5mZXJlbmNlTW9uaXRvcmluZ1N0YWNrJztcblxubmV3IFNhZ2VNYWtlckluZmVyZW5jZU1vbml0b3JpbmdTdGFjayhhcHAsIHN0YWNrSWQsIHtcbiAgLi4uY29uZmlnLFxuICBkZXNjcmlwdGlvbjogJ0luZnJhc3RydWN0dXJlIGZvciBhdXRvbWF0ZWQgU2FnZU1ha2VyIGVuZHBvaW50IG1vbml0b3Jpbmcgd2l0aCBNTGZsb3cgYW5kIEdlbkFJIGV2YWx1YXRpb25zJyxcblxuICB0YWdzOiB7XG4gICAgUHJvamVjdDogJ1NhZ2VNYWtlci1JbmZlcmVuY2UtTW9uaXRvcmluZycsXG4gICAgTWFuYWdlZEJ5OiAnQ0RLJyxcbiAgfSxcbn0pO1xuXG5hcHAuc3ludGgoKTtcbiJdfQ==
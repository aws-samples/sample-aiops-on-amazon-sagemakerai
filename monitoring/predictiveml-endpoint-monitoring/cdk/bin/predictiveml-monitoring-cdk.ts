#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { PredictiveMLMonitoringStack } from '../lib/predictiveml-monitoring-stack';

const app = new cdk.App();

const required = (name: string): string => {
  const val = process.env[name];
  if (!val) throw new Error(`${name} environment variable is required`);
  return val;
};

// Only instantiate the stack when the required env vars are present.
// This allows `cdk bootstrap` and `cdk ls` to run without them.
if (process.env.ENDPOINT_NAME) {
  new PredictiveMLMonitoringStack(app, 'PredictiveMLMonitoringStack', {
    env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
    endpointName:      required('ENDPOINT_NAME'),
    bucket:            required('BUCKET'),
    prefix:            required('PREFIX'),
    baselineKey:       required('BASELINE_KEY'),
    capturePrefix:     required('CAPTURE_PREFIX'),
    groundTruthKey:    required('GROUND_TRUTH_KEY'),
    mlflowTrackingUri: required('MLFLOW_TRACKING_URI'),
    mlflowExperiment:  required('MLFLOW_EXPERIMENT'),
    featureColumns:    required('FEATURE_COLUMNS'),
    snsTopicArn:       process.env.SNS_TOPIC_ARN || '',
  });
}

app.synth();

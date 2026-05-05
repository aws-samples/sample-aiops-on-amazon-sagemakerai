#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDK_DIR="$(dirname "$SCRIPT_DIR")"

cd "$CDK_DIR"

echo "Installing dependencies..."
npm install --silent

echo "Building TypeScript..."
npm run build

echo "Bootstrapping AWS CDK (safe to re-run)..."
npx cdk bootstrap

echo "Deploying AWS CDK stack..."
npx cdk deploy --require-approval never --outputs-file cdk-outputs.json

echo "Deploy complete. Outputs written to $CDK_DIR/cdk-outputs.json"

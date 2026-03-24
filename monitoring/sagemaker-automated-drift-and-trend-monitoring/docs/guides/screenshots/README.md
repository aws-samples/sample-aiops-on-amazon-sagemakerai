# Screenshots Directory

This directory contains screenshots for documentation and tutorials.

## Required Screenshots

### DirectTestingInSGPlayground-custom-handler.png

**Purpose:** Shows how to test the deployed endpoint in SageMaker Studio Endpoint Playground

**Content should show:**
1. SageMaker Studio navigation: Deployments > Endpoints > fraud-detector-endpoint > Playground
2. Content-Type selector set to `application/json`
3. Request body with JSON payload (30 features)
4. "Test" button
5. Response showing:
   ```json
   {
     "predictions": [0],
     "probabilities": {
       "fraud": [0.1234],
       "non_fraud": [0.8766]
     }
   }
   ```

**Referenced in:**
- `notebooks/2a_inference_monitoring.ipynb` - Cell 8
- `README.md` - "Testing in SageMaker Studio Endpoint Playground" section

**How to capture:**
1. Deploy the endpoint via pipeline
2. Navigate to: SageMaker Studio > Deployments > Endpoints > fraud-detector-endpoint
3. Click "Playground" tab
4. Set Content-Type: application/json
5. Paste JSON payload with 30 features
6. Click "Test"
7. Take screenshot showing the entire interface with request and response
8. Save as `DirectTestingInSGPlayground-custom-handler.png`

**Recommended dimensions:** 1200x800 pixels or higher

## Adding Screenshots

```bash
# Add screenshot to this directory
cp /path/to/screenshot.png screenshots/DirectTestingInSGPlayground-custom-handler.png

# Verify it's referenced correctly
grep -r "DirectTestingInSGPlayground-custom-handler.png" .
```

## Notes

- Use PNG format for better quality
- Ensure text is readable in screenshots
- Blur any sensitive information (account IDs, bucket names)
- Include browser/UI chrome to show context
- Screenshots should match the documented workflow

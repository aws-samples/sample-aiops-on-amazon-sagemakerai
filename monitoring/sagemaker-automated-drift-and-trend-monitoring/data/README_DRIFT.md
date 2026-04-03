# Drifted Dataset for Inference Monitoring

## Overview

The `creditcard_drifted.csv` dataset is a synthetic dataset created to test the MLflow inference monitoring system's ability to detect feature drift. It contains 5,000 samples derived from the original `creditcard_predictions_final.csv` dataset with intentionally shifted feature distributions.

## Dataset Details

- **Filename**: `creditcard_drifted.csv`
- **Number of samples**: 5,000
- **Structure**: Same 35 columns as original dataset
- **Class distribution**: ~99.8% non-fraud, ~0.2% fraud (maintains original class imbalance)
- **Purpose**: Test feature drift detection in production inference monitoring

## Feature Drift Applied

The following key features have been systematically shifted to simulate real-world data drift scenarios:

### 1. Transaction Amount (+45% increase)
- **Original mean**: 88.35
- **Drifted mean**: 128.14
- **Simulation**: Represents inflation, economic changes, or shifts in customer spending behavior
- **Method**: Multiplicative drift with 40% factor + 10% random noise

### 2. Transaction Timestamp (+54% shift forward)
- **Original mean**: 94,813.86
- **Drifted mean**: 145,752.32
- **Simulation**: Time progression to simulate data from a future period
- **Method**: Additive shift of 50,000 time units + 5,000 random noise

### 3. Distance from Home (100% increase)
- **Original mean**: ~0.00 (normalized)
- **Drifted mean**: 0.88
- **Simulation**: Increased travel patterns, remote work, or geographic behavior changes
- **Method**: Multiplicative drift with 2x factor + 30% random noise

### 4. Velocity Score (+50% increase)
- **Original mean**: ~0.00 (normalized)
- **Drifted mean**: 0.24
- **Simulation**: More active users with higher transaction frequency
- **Method**: Multiplicative drift with 1.5x factor + 20% random noise

### 5. Number of Transactions per 24h (+3 transactions average)
- **Original mean**: ~0.00 (normalized)
- **Drifted mean**: 3.00
- **Simulation**: More active user behavior, increased digital adoption
- **Method**: Additive shift of +3 transactions + 1 random noise

## How to Use

### Generate New Drifted Dataset

To regenerate the drifted dataset with different parameters:

```bash
python generate_drift_dataset.py
```

Edit `generate_drift_dataset.py` to adjust:
- `NUM_SAMPLES`: Number of samples to generate (default: 5000)
- `DRIFT_CONFIG`: Drift parameters for each feature
- `RANDOM_STATE`: Random seed for reproducibility

### Test Inference with Drifted Data

#### Option 1: Using CLI

Test your deployed model with the drifted dataset:

```bash
# Test with 100 samples from drifted dataset # pure-storage-mlflow-1
python main.py --mode test \
  --endpoint-name pure-storage-mlflow-1 \
  --test-data-path data/creditcard_drifted.csv \
  --num-samples 100
```

#### Option 2: Using test_endpoint.py directly

```bash
python src/inference/test_endpoint.py fraud-detector 100
```

Then modify the script to use `data/creditcard_drifted.csv` instead of the default path.

### Compare Drift in MLflow

1. **Baseline Run**: Test with original data
   ```bash
   python main.py --mode test --endpoint-name fraud-detector --num-samples 100
   ```

2. **Drifted Run**: Test with drifted data
   ```bash
   python main.py --mode test --endpoint-name fraud-detector \
     --test-data-path data/creditcard_drifted.csv --num-samples 100
   ```

3. **Analyze in MLflow UI**:
   - Navigate to experiment: `credit-card-fraud-detection-inference`
   - Compare feature metrics between the two runs:
     - `feature_mean_transaction_amount` (should be ~45% higher)
     - `feature_mean_transaction_timestamp` (should be ~54% higher)
     - `feature_mean_distance_from_home_km` (should be significantly higher)
     - `feature_mean_velocity_score` (should be higher)
     - `feature_mean_num_transactions_24h` (should be ~3.0 higher)

## Expected Drift Detection Results

When testing with the drifted dataset, you should observe:

### Feature Distribution Changes
- **transaction_amount**: Mean increases from ~88 to ~128 (+45%)
- **transaction_timestamp**: Mean shifts forward by ~50,000 units
- **distance_from_home_km**: Mean increases to 0.88 (from near-zero)
- **velocity_score**: Mean increases to 0.24 (from near-zero)
- **num_transactions_24h**: Mean increases to 3.0 (from near-zero)

### Model Performance Changes
Depending on your model's robustness, you may observe:
- **Prediction Distribution**: Changes in fraud prediction rate
- **Probability Distributions**: Shifts in confidence scores
- **Performance Metrics**: Changes in accuracy, precision, recall (if ground truth available)

### Operational Metrics
- **Latency**: Should remain similar (endpoint performance)
- **Success Rate**: Should remain high (data structure is valid)

## Real-World Drift Scenarios

This drifted dataset simulates several realistic scenarios:

1. **Economic Changes**: Transaction amounts increase due to inflation or currency changes
2. **Temporal Evolution**: Dataset represents a future time period
3. **Behavioral Shifts**: Customers transacting further from home (e.g., travel, remote work)
4. **Usage Pattern Changes**: Higher transaction velocity and frequency
5. **Platform Growth**: More active users with more transactions

## Monitoring Recommendations

Use this dataset to validate your monitoring system's ability to:

1. **Detect Drift**: Flag when feature distributions deviate significantly
2. **Alert Thresholds**: Set appropriate thresholds for drift metrics
3. **Model Retraining**: Determine when drift is severe enough to retrain
4. **Continuous Monitoring**: Establish baselines and track trends over time

## Customization

To create datasets with different drift patterns:

1. **Modify drift factors** in `generate_drift_dataset.py`:
   ```python
   DRIFT_CONFIG = {
       "transaction_amount": {
           "type": "multiplicative",
           "factor": 1.8,  # Change to 80% increase
           "noise": 0.15,  # Increase randomness
       },
       # ... other features
   }
   ```

2. **Add new features** to monitor:
   ```python
   DRIFT_CONFIG["customer_age"] = {
       "type": "additive",
       "shift": 5,  # Age demographic shift
       "noise": 2,
   }
   ```

3. **Generate multiple scenarios**: Create datasets for different drift magnitudes (mild, moderate, severe)

## Integration with MLflow

The inference monitoring system automatically logs drift metrics to MLflow:

- **Metric naming**: `feature_mean_<feature_name>`, `feature_std_<feature_name>`
- **Experiment**: `credit-card-fraud-detection-inference`
- **Tags**: `endpoint_name`, `inference_type`, `has_ground_truth`

Compare these metrics across runs to quantify drift and trigger alerts or retraining workflows.

## Files

- `creditcard_predictions_final.csv` - Original dataset (284,807 samples)
- `creditcard_drifted.csv` - Drifted dataset (5,000 samples)
- `../generate_drift_dataset.py` - Script to generate drifted data
- `README_DRIFT.md` - This documentation file

---

**Generated**: This dataset was created using `generate_drift_dataset.py` with RANDOM_STATE=123 for reproducibility.

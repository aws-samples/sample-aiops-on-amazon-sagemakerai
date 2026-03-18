"""
Generate a drifted dataset for testing inference monitoring.

This script creates a new dataset with intentional feature drift to test
the MLflow inference monitoring system's ability to detect distribution changes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
ORIGINAL_DATA_PATH = Path("data/creditcard_predictions_final.csv")
DRIFTED_DATA_PATH = Path("data/creditcard_drifted.csv")
NUM_SAMPLES = 5000
RANDOM_STATE = 123

# Drift parameters for key features
DRIFT_CONFIG = {
    "transaction_amount": {
        "type": "multiplicative",
        "factor": 1.4,  # 40% increase
        "noise": 0.1,   # 10% random variation
        "description": "Increased transaction amounts (inflation/behavior change)"
    },
    "transaction_timestamp": {
        "type": "additive",
        "shift": 50000,  # Shift time forward
        "noise": 5000,
        "description": "Time shift to simulate future period"
    },
    "distance_from_home_km": {
        "type": "multiplicative",
        "factor": 2.0,   # 100% increase (double distance)
        "noise": 0.3,
        "description": "Increased distance from home (travel/remote transactions)"
    },
    "velocity_score": {
        "type": "multiplicative",
        "factor": 1.5,   # 50% increase
        "noise": 0.2,
        "description": "Higher transaction velocity (more active users)"
    },
    "num_transactions_24h": {
        "type": "additive",
        "shift": 3,      # Add 3 more transactions on average
        "noise": 1,
        "description": "More transactions per day"
    },
}


def apply_drift(df: pd.DataFrame, feature: str, config: dict) -> pd.DataFrame:
    """Apply drift to a specific feature based on configuration."""
    if feature not in df.columns:
        print(f"  Warning: Feature '{feature}' not found in dataset, skipping")
        return df

    original_values = df[feature].values

    if config["type"] == "multiplicative":
        # Multiplicative drift: value = original * (factor ± noise)
        factor = config["factor"]
        noise = config.get("noise", 0)
        random_factors = np.random.uniform(
            factor - noise * factor,
            factor + noise * factor,
            size=len(df)
        )
        drifted_values = original_values * random_factors

    elif config["type"] == "additive":
        # Additive drift: value = original + (shift ± noise)
        shift = config["shift"]
        noise = config.get("noise", 0)
        random_shifts = np.random.uniform(
            shift - noise,
            shift + noise,
            size=len(df)
        )
        drifted_values = original_values + random_shifts

    else:
        raise ValueError(f"Unknown drift type: {config['type']}")

    # Ensure non-negative values for certain features
    if feature in ["transaction_amount", "distance_from_home_km", "velocity_score", "num_transactions_24h"]:
        drifted_values = np.maximum(drifted_values, 0)

    # Round integer features
    if feature == "num_transactions_24h":
        drifted_values = np.round(drifted_values).astype(int)

    df[feature] = drifted_values

    # Print drift statistics
    original_mean = original_values.mean()
    drifted_mean = drifted_values.mean()
    pct_change = ((drifted_mean - original_mean) / original_mean) * 100

    print(f"  {feature}:")
    print(f"    Original mean: {original_mean:.4f}")
    print(f"    Drifted mean: {drifted_mean:.4f}")
    print(f"    Change: {pct_change:+.2f}%")
    print(f"    Description: {config['description']}")

    return df


def generate_drifted_dataset():
    """Generate a drifted dataset for testing inference monitoring."""

    print("=" * 80)
    print("GENERATING DRIFTED DATASET")
    print("=" * 80)

    # Load original dataset
    print(f"\nLoading original dataset from: {ORIGINAL_DATA_PATH}")
    df_original = pd.read_csv(ORIGINAL_DATA_PATH)
    print(f"Original dataset shape: {df_original.shape}")

    # Sample random rows
    print(f"\nSampling {NUM_SAMPLES} random rows...")
    np.random.seed(RANDOM_STATE)
    df_drifted = df_original.sample(n=NUM_SAMPLES, random_state=RANDOM_STATE).copy()
    print(f"Sampled dataset shape: {df_drifted.shape}")

    # Apply drift to key features
    print("\nApplying feature drift:")
    print("-" * 80)
    for feature, config in DRIFT_CONFIG.items():
        df_drifted = apply_drift(df_drifted, feature, config)

    # Reset index
    df_drifted = df_drifted.reset_index(drop=True)

    # Save drifted dataset
    print("\n" + "-" * 80)
    print(f"Saving drifted dataset to: {DRIFTED_DATA_PATH}")
    df_drifted.to_csv(DRIFTED_DATA_PATH, index=False)
    print(f"Saved {len(df_drifted)} rows")

    # Summary statistics
    print("\n" + "=" * 80)
    print("DRIFT SUMMARY")
    print("=" * 80)
    print(f"Original dataset: {ORIGINAL_DATA_PATH}")
    print(f"Drifted dataset: {DRIFTED_DATA_PATH}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"\nClass distribution in drifted dataset:")
    if "is_fraud" in df_drifted.columns:
        fraud_counts = df_drifted["is_fraud"].value_counts()
        print(f"  Non-fraud: {fraud_counts.get(False, fraud_counts.get(0, 0))}")
        print(f"  Fraud: {fraud_counts.get(True, fraud_counts.get(1, 0))}")

    print("\nKey feature comparison:")
    print("-" * 80)
    for feature in DRIFT_CONFIG.keys():
        if feature in df_drifted.columns and feature in df_original.columns:
            original_mean = df_original[feature].mean()
            drifted_mean = df_drifted[feature].mean()
            pct_change = ((drifted_mean - original_mean) / original_mean) * 100
            print(f"{feature:30s} Original: {original_mean:10.2f}  Drifted: {drifted_mean:10.2f}  Change: {pct_change:+6.1f}%")

    print("\n" + "=" * 80)
    print("DRIFTED DATASET GENERATION COMPLETED")
    print("=" * 80)
    print(f"\nTo test with the drifted dataset, run:")
    print(f"  python main.py --mode test --endpoint-name <endpoint> --num-samples 100")
    print(f"\nThe test will automatically use data from: {ORIGINAL_DATA_PATH}")
    print(f"To use the drifted data, you'll need to modify test_endpoint.py or")
    print(f"manually specify the test data path.")
    print("=" * 80)


if __name__ == "__main__":
    generate_drifted_dataset()

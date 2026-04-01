#!/usr/bin/env python3
"""
Generate realistic ground truth confirmations for fraud detection inference monitoring.

This script simulates the real-world process where:
1. Model makes predictions at inference time (T+0)
2. Fraud investigations complete over days/weeks (T+hours/days)
3. Ground truth is confirmed through various channels

Features:
- Realistic confirmation delays (fraud: 1-7 days, non-fraud: 1-30 days)
- Multiple confirmation sources (investigation, chargeback, customer reports)
- Model errors (false positives ~5%, false negatives ~15%)
- Window assignment for drift testing

Output: data/fraud_confirmations.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.config import DATA_DIR, CSV_TRAINING_DATA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GroundTruthGenerator:
    """Generate realistic ground truth confirmations with delays and sources."""

    # Confirmation sources and their distributions
    CONFIRMATION_SOURCES = {
        'fraud_investigation': 0.50,      # Manual fraud team investigation
        'chargeback': 0.20,                # Customer disputes/chargebacks
        'customer_complaint': 0.15,        # Customer reports fraud
        'system_alert': 0.10,              # Automated system detection
        'merchant_report': 0.05,           # Merchant reports issue
    }

    NON_FRAUD_SOURCES = {
        'normal_activity': 0.60,           # Customer continues normal usage
        'customer_verification': 0.25,     # Customer confirms legitimate
        'transaction_completed': 0.15,     # Transaction settled normally
    }

    def __init__(
        self,
        input_csv: Path,
        output_csv: Path,
        false_positive_rate: float = 0.05,
        false_negative_rate: float = 0.15,
        samples_per_window: int = 5000,
        num_windows: int = 10,
        base_timestamp: datetime = None,
        seed: int = 42,
    ):
        """
        Initialize ground truth generator.

        Args:
            input_csv: Path to input predictions CSV
            output_csv: Path to output confirmations CSV
            false_positive_rate: Rate of predicted fraud that's actually not fraud
            false_negative_rate: Rate of predicted non-fraud that's actually fraud
            samples_per_window: Number of samples per time window
            num_windows: Total number of time windows
            base_timestamp: Base timestamp for confirmations (defaults to now)
            seed: Random seed for reproducibility
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.samples_per_window = samples_per_window
        self.num_windows = num_windows
        self.base_timestamp = base_timestamp or datetime(2024, 1, 1, 0, 0, 0)
        self.seed = seed

        np.random.seed(seed)

    def load_predictions(self) -> pd.DataFrame:
        """Load prediction data from CSV."""
        logger.info(f"Loading predictions from: {self.input_csv}")

        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_csv}")

        df = pd.read_csv(self.input_csv)
        logger.info(f"Loaded {len(df)} predictions")

        # Check required columns
        required = ['transaction_id', 'fraud_prediction', 'is_fraud']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def assign_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign transactions to time windows."""
        logger.info(f"Assigning transactions to {self.num_windows} windows")

        # Sample evenly across windows
        df = df.copy()
        total_samples = self.samples_per_window * self.num_windows

        if len(df) < total_samples:
            logger.warning(
                f"Dataset has {len(df)} rows but needs {total_samples}. "
                f"Using all available data."
            )
            df['window_id'] = np.random.randint(1, self.num_windows + 1, size=len(df))
        else:
            # Sample evenly
            df = df.sample(n=total_samples, random_state=self.seed)
            window_ids = np.repeat(range(1, self.num_windows + 1), self.samples_per_window)
            df['window_id'] = window_ids

        logger.info(f"Window distribution:\n{df['window_id'].value_counts().sort_index()}")
        return df

    def determine_actual_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Determine actual fraud status with model errors.

        Introduces:
        - False positives: Model predicted fraud, actually not fraud
        - False negatives: Model predicted non-fraud, actually fraud
        """
        logger.info("Determining actual fraud status with model errors")

        df = df.copy()

        # Start with actual ground truth from dataset
        df['actual_fraud'] = df['is_fraud']

        # Get predictions
        predicted_fraud = df['fraud_prediction'] == True
        predicted_non_fraud = ~predicted_fraud

        # Introduce false positives (predicted fraud, actually not)
        false_positive_mask = predicted_fraud & ~df['is_fraud']
        num_false_positives = int(false_positive_mask.sum() * self.false_positive_rate)

        if num_false_positives > 0:
            fp_indices = df[false_positive_mask].sample(
                n=num_false_positives,
                random_state=self.seed
            ).index
            # These remain as non-fraud (already set correctly)
            logger.info(f"Added {num_false_positives} false positives")

        # Introduce false negatives (predicted non-fraud, actually fraud)
        false_negative_mask = predicted_non_fraud & df['is_fraud']
        num_false_negatives = int(false_negative_mask.sum() * self.false_negative_rate)

        if num_false_negatives > 0:
            fn_indices = df[false_negative_mask].sample(
                n=num_false_negatives,
                random_state=self.seed + 1
            ).index
            # These remain as fraud (already set correctly)
            logger.info(f"Added {num_false_negatives} false negatives")

        # Calculate error flags
        df['false_positive'] = predicted_fraud & ~df['actual_fraud']
        df['false_negative'] = predicted_non_fraud & df['actual_fraud']

        # Log statistics
        total = len(df)
        actual_fraud_count = df['actual_fraud'].sum()
        fp_count = df['false_positive'].sum()
        fn_count = df['false_negative'].sum()

        logger.info(f"\nActual fraud statistics:")
        logger.info(f"  Total transactions: {total:,}")
        logger.info(f"  Actual fraud: {actual_fraud_count:,} ({actual_fraud_count/total*100:.2f}%)")
        logger.info(f"  False positives: {fp_count:,} ({fp_count/total*100:.2f}%)")
        logger.info(f"  False negatives: {fn_count:,} ({fn_count/total*100:.2f}%)")

        return df

    def assign_confirmation_delays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign realistic confirmation delays.

        Fraud cases: 1-7 days (faster investigation)
        Non-fraud: 1-30 days (slower confirmation through normal activity)
        """
        logger.info("Assigning confirmation delays")

        df = df.copy()

        # Fraud cases get faster confirmation (1-7 days)
        fraud_mask = df['actual_fraud']
        df.loc[fraud_mask, 'days_since_transaction'] = np.random.uniform(
            1.0, 7.0, size=fraud_mask.sum()
        )

        # Non-fraud cases get slower confirmation (1-30 days)
        non_fraud_mask = ~fraud_mask
        df.loc[non_fraud_mask, 'days_since_transaction'] = np.random.uniform(
            1.0, 30.0, size=non_fraud_mask.sum()
        )

        # Round to 2 decimal places
        df['days_since_transaction'] = df['days_since_transaction'].round(2)

        logger.info(
            f"Confirmation delay statistics:\n"
            f"  Fraud: {df.loc[fraud_mask, 'days_since_transaction'].describe()}\n"
            f"  Non-fraud: {df.loc[non_fraud_mask, 'days_since_transaction'].describe()}"
        )

        return df

    def assign_confirmation_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign confirmation sources based on actual fraud status."""
        logger.info("Assigning confirmation sources")

        df = df.copy()

        # Fraud cases
        fraud_mask = df['actual_fraud']
        fraud_sources = np.random.choice(
            list(self.CONFIRMATION_SOURCES.keys()),
            size=fraud_mask.sum(),
            p=list(self.CONFIRMATION_SOURCES.values())
        )
        df.loc[fraud_mask, 'confirmation_source'] = fraud_sources

        # Non-fraud cases
        non_fraud_mask = ~fraud_mask
        non_fraud_sources = np.random.choice(
            list(self.NON_FRAUD_SOURCES.keys()),
            size=non_fraud_mask.sum(),
            p=list(self.NON_FRAUD_SOURCES.values())
        )
        df.loc[non_fraud_mask, 'confirmation_source'] = non_fraud_sources

        # Log distribution
        logger.info(f"\nConfirmation source distribution:")
        for source, count in df['confirmation_source'].value_counts().items():
            pct = count / len(df) * 100
            logger.info(f"  {source}: {count:,} ({pct:.1f}%)")

        return df

    def generate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate transaction, prediction, and confirmation timestamps."""
        logger.info("Generating timestamps")

        df = df.copy()

        # Each window represents 1 day
        # Transaction timestamp = base + window_id days
        df['transaction_timestamp'] = df['window_id'].apply(
            lambda w: self.base_timestamp + timedelta(days=w - 1)
        )

        # Prediction happens immediately after transaction (same timestamp)
        df['prediction_timestamp'] = df['transaction_timestamp']

        # Confirmation timestamp = transaction + delay
        df['confirmation_timestamp'] = df.apply(
            lambda row: row['transaction_timestamp'] + timedelta(days=row['days_since_transaction']),
            axis=1
        )

        # Days since prediction (same as transaction in this case)
        df['days_since_prediction'] = df['days_since_transaction']

        return df

    def add_investigation_notes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic investigation notes."""
        logger.info("Adding investigation notes")

        df = df.copy()

        # Fraud notes by source
        fraud_notes = {
            'fraud_investigation': [
                'Confirmed unauthorized purchase',
                'Card reported stolen',
                'Pattern matches known fraud ring',
                'Customer denied making transaction',
                'Multiple red flags detected',
            ],
            'chargeback': [
                'Customer initiated chargeback',
                'Bank reversed transaction',
                'Dispute filed - unauthorized use',
                'Chargeback confirmed fraud',
            ],
            'customer_complaint': [
                'Customer reported fraud immediately',
                'Customer claims card theft',
                'Unrecognized transaction reported',
            ],
            'system_alert': [
                'Automated fraud detection confirmed',
                'System flagged as suspicious',
                'Alert triggered investigation',
            ],
            'merchant_report': [
                'Merchant reported suspicious activity',
                'Merchant verified fraud attempt',
            ],
        }

        # Non-fraud notes
        non_fraud_notes = {
            'normal_activity': [
                'Customer continued normal usage',
                'Transaction settled without issue',
                'No fraud indicators observed',
            ],
            'customer_verification': [
                'Customer verified legitimate purchase',
                'Customer confirmed transaction',
                'Verified as authorized',
            ],
            'transaction_completed': [
                'Transaction completed successfully',
                'No issues reported',
                'Settled normally',
            ],
        }

        all_notes = {**fraud_notes, **non_fraud_notes}

        def get_note(row):
            source = row['confirmation_source']
            notes = all_notes.get(source, ['No additional details'])
            return np.random.choice(notes)

        df['investigation_notes'] = df.apply(get_note, axis=1)

        # Add priority for fraud cases
        df['investigation_priority'] = df.apply(
            lambda row: 'high' if row['actual_fraud'] else 'low',
            axis=1
        )

        return df

    def generate_confirmations(self) -> pd.DataFrame:
        """Generate complete ground truth confirmations dataset."""
        logger.info("=" * 80)
        logger.info("Generating Ground Truth Confirmations")
        logger.info("=" * 80)

        # Load predictions
        df = self.load_predictions()

        # Assign windows
        df = self.assign_windows(df)

        # Determine actual fraud with model errors
        df = self.determine_actual_fraud(df)

        # Assign confirmation delays
        df = self.assign_confirmation_delays(df)

        # Assign confirmation sources
        df = self.assign_confirmation_sources(df)

        # Generate timestamps
        df = self.generate_timestamps(df)

        # Add investigation notes
        df = self.add_investigation_notes(df)

        # Add metadata
        df['batch_id'] = 'initial_load'
        df['created_at'] = datetime.now()

        # Select output columns
        output_cols = [
            'transaction_id',
            'actual_fraud',
            'confirmation_timestamp',
            'confirmation_source',
            'transaction_timestamp',
            'prediction_timestamp',
            'days_since_transaction',
            'days_since_prediction',
            'investigation_notes',
            'investigation_priority',
            'false_positive',
            'false_negative',
            'window_id',
            'batch_id',
            'created_at',
        ]

        df_output = df[output_cols].copy()

        logger.info(f"\nGenerated {len(df_output):,} ground truth confirmations")
        logger.info(f"Date range: {df_output['confirmation_timestamp'].min()} to {df_output['confirmation_timestamp'].max()}")

        return df_output

    def save_confirmations(self, df: pd.DataFrame) -> None:
        """Save confirmations to CSV."""
        logger.info(f"Saving confirmations to: {self.output_csv}")

        # Create output directory if needed
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(self.output_csv, index=False)

        logger.info(f"✓ Saved {len(df):,} confirmations to {self.output_csv}")
        logger.info(f"  File size: {self.output_csv.stat().st_size / 1024 / 1024:.2f} MB")

    def generate_and_save(self) -> pd.DataFrame:
        """Generate and save confirmations in one call."""
        df = self.generate_confirmations()
        self.save_confirmations(df)
        return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate realistic ground truth confirmations for fraud detection'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=CSV_TRAINING_DATA,
        help='Input predictions CSV (default: creditcard_predictions_final.csv)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DATA_DIR / 'fraud_confirmations.csv',
        help='Output confirmations CSV'
    )
    parser.add_argument(
        '--false-positive-rate',
        type=float,
        default=0.05,
        help='False positive rate (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--false-negative-rate',
        type=float,
        default=0.15,
        help='False negative rate (default: 0.15 = 15%%)'
    )
    parser.add_argument(
        '--samples-per-window',
        type=int,
        default=5000,
        help='Samples per time window (default: 5000)'
    )
    parser.add_argument(
        '--num-windows',
        type=int,
        default=10,
        help='Number of time windows (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Create generator
    generator = GroundTruthGenerator(
        input_csv=args.input,
        output_csv=args.output,
        false_positive_rate=args.false_positive_rate,
        false_negative_rate=args.false_negative_rate,
        samples_per_window=args.samples_per_window,
        num_windows=args.num_windows,
        seed=args.seed,
    )

    # Generate and save
    try:
        df = generator.generate_and_save()

        # Print summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total confirmations: {len(df):,}")
        print(f"Actual fraud: {df['actual_fraud'].sum():,} ({df['actual_fraud'].sum()/len(df)*100:.2f}%)")
        print(f"False positives: {df['false_positive'].sum():,}")
        print(f"False negatives: {df['false_negative'].sum():,}")
        print(f"\nOutput: {args.output}")
        print("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Error generating confirmations: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Simulate ground truth updates for inference predictions stored in Athena.

This script:
1. Queries inference_responses table for predictions without ground truth
2. Randomly assigns ground truth (with realistic correlation to predictions)
3. Writes ground truth updates to ground_truth_updates table in Athena

Usage:
    # Simulate ground truth for all predictions without ground truth
    python scripts/simulate_ground_truth_from_athena.py

    # Simulate for specific endpoint
    python scripts/simulate_ground_truth_from_athena.py --endpoint-name fraud-detector-endpoint

    # Control accuracy (how often predictions match ground truth)
    python scripts/simulate_ground_truth_from_athena.py --accuracy 0.90

    # Limit number of updates
    python scripts/simulate_ground_truth_from_athena.py --limit 100
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
import uuid

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / '.env')

from src.config.config import ATHENA_DATABASE, ATHENA_INFERENCE_TABLE, ATHENA_GROUND_TRUTH_UPDATES_TABLE
from src.train_pipeline.athena.athena_client import AthenaClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GroundTruthSimulator:
    """Simulate ground truth for Athena inference predictions."""

    def __init__(
        self,
        athena_client: AthenaClient,
        endpoint_name: str = None,
        accuracy: float = 0.85,
        fraud_confirmation_days: tuple = (1, 7),
        non_fraud_confirmation_days: tuple = (1, 30),
        feature_drift_magnitude: float = 0.0,
        feature_drift_count: int = 0,
        feature_drift_impact: float = 0.0,
        model_drift_magnitude: float = 0.0,
        seed: int = 42,
    ):
        """
        Initialize ground truth simulator.

        Args:
            athena_client: Athena client for queries
            endpoint_name: Optional filter for specific endpoint
            accuracy: Base model accuracy (0.0-1.0)
            fraud_confirmation_days: (min, max) days for fraud confirmation
            non_fraud_confirmation_days: (min, max) days for non-fraud confirmation
            feature_drift_magnitude: How much features drift (0.0-1.0, semantic)
            feature_drift_count: Number of features drifting (semantic)
            feature_drift_impact: Accuracy reduction due to feature drift (0.0-1.0)
            model_drift_magnitude: Direct model accuracy degradation (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.client = athena_client
        self.endpoint_name = endpoint_name
        self.accuracy = accuracy
        self.fraud_confirmation_days = fraud_confirmation_days
        self.non_fraud_confirmation_days = non_fraud_confirmation_days
        self.feature_drift_magnitude = feature_drift_magnitude
        self.feature_drift_count = feature_drift_count
        self.feature_drift_impact = feature_drift_impact
        self.model_drift_magnitude = model_drift_magnitude

        # Calculate effective accuracy after drift
        self.effective_accuracy = accuracy - feature_drift_impact - model_drift_magnitude
        self.effective_accuracy = max(0.5, self.effective_accuracy)  # Floor at 50%

        np.random.seed(seed)

        logger.info(f"Initialized GroundTruthSimulator:")
        logger.info(f"  Base accuracy: {accuracy:.2f}")
        if feature_drift_impact > 0:
            logger.info(f"  Feature drift: {feature_drift_magnitude:.2f} magnitude, {feature_drift_count} features, impact={feature_drift_impact:.2f}")
        if model_drift_magnitude > 0:
            logger.info(f"  Model drift: {model_drift_magnitude:.2f} degradation")
        logger.info(f"  Effective accuracy: {self.effective_accuracy:.2f}")

    def load_predictions_without_ground_truth(self, limit: int = None) -> pd.DataFrame:
        """Load inference predictions that don't have ground truth yet."""
        logger.info("Loading predictions without ground truth from Athena...")

        query = f"""
        SELECT
            inference_id,
            transaction_id,
            CAST(request_timestamp AS TIMESTAMP(3)) as request_timestamp,
            endpoint_name,
            prediction,
            probability_fraud,
            confidence_score
        FROM {ATHENA_DATABASE}.{ATHENA_INFERENCE_TABLE}
        WHERE ground_truth IS NULL
        """

        if self.endpoint_name:
            query += f" AND endpoint_name = '{self.endpoint_name}'"

        query += " ORDER BY request_timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        df = self.client.execute_query(query)

        if df.empty:
            logger.warning("No predictions found without ground truth")
            return df

        logger.info(f"Loaded {len(df):,} predictions without ground truth")
        logger.info(f"  Predicted fraud: {(df['prediction'] == 1).sum():,}")
        logger.info(f"  Predicted non-fraud: {(df['prediction'] == 0).sum():,}")

        return df

    def simulate_ground_truth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate ground truth with realistic correlation to predictions.
        Applies accuracy degradation from feature drift and model drift.

        The accuracy parameter controls how often predictions match ground truth:
        - accuracy = 0.85 means 85% of predictions are correct
        - Remaining 15% are split between false positives and false negatives
        """
        logger.info(f"Simulating ground truth with {self.effective_accuracy*100:.1f}% effective accuracy...")

        if self.effective_accuracy < self.accuracy:
            logger.info(f"  Accuracy degradation: {self.accuracy:.2f} → {self.effective_accuracy:.2f}")
            if self.feature_drift_impact > 0:
                logger.info(f"    Feature drift impact: -{self.feature_drift_impact:.2f}")
            if self.model_drift_magnitude > 0:
                logger.info(f"    Model drift: -{self.model_drift_magnitude:.2f}")

        df = df.copy()

        # Start with predictions as ground truth (perfect accuracy)
        df['actual_fraud'] = df['prediction'].astype(bool)

        # Introduce errors to reach target effective accuracy
        error_rate = 1.0 - self.effective_accuracy
        num_errors = int(len(df) * error_rate)

        if num_errors > 0:
            # Randomly select which predictions to flip
            error_indices = np.random.choice(df.index, size=num_errors, replace=False)

            # Flip the ground truth for these errors
            df.loc[error_indices, 'actual_fraud'] = ~df.loc[error_indices, 'actual_fraud']

            # Calculate error types
            false_positives = ((df['prediction'] == 1) & (~df['actual_fraud'])).sum()
            false_negatives = ((df['prediction'] == 0) & (df['actual_fraud'])).sum()

            logger.info(f"  Introduced {num_errors:,} errors:")
            logger.info(f"    False positives: {false_positives:,}")
            logger.info(f"    False negatives: {false_negatives:,}")

        # Calculate flags
        df['false_positive'] = (df['prediction'] == 1) & (~df['actual_fraud'])
        df['false_negative'] = (df['prediction'] == 0) & (df['actual_fraud'])

        # Log statistics
        actual_fraud_count = df['actual_fraud'].sum()
        logger.info(f"\nSimulated ground truth statistics:")
        logger.info(f"  Total: {len(df):,}")
        logger.info(f"  Actual fraud: {actual_fraud_count:,} ({actual_fraud_count/len(df)*100:.2f}%)")
        logger.info(f"  Actual non-fraud: {(~df['actual_fraud']).sum():,}")

        return df

    def assign_confirmation_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign realistic confirmation timestamps based on fraud status."""
        logger.info("Assigning confirmation timestamps...")

        df = df.copy()

        # Fraud cases: faster confirmation (1-7 days)
        fraud_mask = df['actual_fraud']
        if fraud_mask.any():
            fraud_delays = np.random.uniform(
                self.fraud_confirmation_days[0],
                self.fraud_confirmation_days[1],
                size=fraud_mask.sum()
            )
            df.loc[fraud_mask, 'days_since_prediction'] = fraud_delays

        # Non-fraud cases: slower confirmation (1-30 days)
        non_fraud_mask = ~fraud_mask
        if non_fraud_mask.any():
            non_fraud_delays = np.random.uniform(
                self.non_fraud_confirmation_days[0],
                self.non_fraud_confirmation_days[1],
                size=non_fraud_mask.sum()
            )
            df.loc[non_fraud_mask, 'days_since_prediction'] = non_fraud_delays

        # Calculate confirmation timestamp
        df['confirmation_timestamp'] = pd.to_datetime(df['request_timestamp']) + pd.to_timedelta(
            df['days_since_prediction'], unit='D'
        )

        logger.info(f"  Fraud confirmation delay: {fraud_delays.mean():.1f} days (avg)" if fraud_mask.any() else "  No fraud cases")
        logger.info(f"  Non-fraud confirmation delay: {non_fraud_delays.mean():.1f} days (avg)" if non_fraud_mask.any() else "  No non-fraud cases")

        return df

    def assign_confirmation_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign confirmation sources based on fraud status."""
        logger.info("Assigning confirmation sources...")

        df = df.copy()

        fraud_sources = [
            'fraud_investigation',
            'chargeback',
            'customer_complaint',
            'system_alert',
            'merchant_report',
        ]

        non_fraud_sources = [
            'normal_activity',
            'customer_verification',
            'transaction_completed',
        ]

        fraud_mask = df['actual_fraud']
        df.loc[fraud_mask, 'confirmation_source'] = np.random.choice(
            fraud_sources,
            size=fraud_mask.sum()
        )

        non_fraud_mask = ~fraud_mask
        df.loc[non_fraud_mask, 'confirmation_source'] = np.random.choice(
            non_fraud_sources,
            size=non_fraud_mask.sum()
        )

        return df

    def create_ground_truth_updates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ground truth updates dataframe for Athena."""
        logger.info("Creating ground truth updates...")

        # Assign confirmation timestamps
        df = self.assign_confirmation_timestamps(df)

        # Assign confirmation sources
        df = self.assign_confirmation_sources(df)

        # Create updates dataframe with Athena schema
        updates = pd.DataFrame({
            'transaction_id': df['transaction_id'],
            'inference_id': df['inference_id'],
            'actual_fraud': df['actual_fraud'],
            'confirmation_timestamp': df['confirmation_timestamp'],
            'confirmation_source': df['confirmation_source'],
            'transaction_timestamp': pd.to_datetime(df['request_timestamp']),
            'prediction_timestamp': pd.to_datetime(df['request_timestamp']),
            'days_since_transaction': df['days_since_prediction'],  # Same as prediction in this case
            'days_since_prediction': df['days_since_prediction'],
            'investigation_notes': df.apply(self._generate_note, axis=1),
            'investigation_priority': df['actual_fraud'].apply(lambda x: 'high' if x else 'low'),
            'false_positive': df['false_positive'],
            'false_negative': df['false_negative'],
            'window_id': np.int32(1),  # Default window
            'batch_id': f'simulated_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
        })

        logger.info(f"Created {len(updates):,} ground truth updates")

        return updates

    def _generate_note(self, row) -> str:
        """Generate investigation note based on confirmation source."""
        notes_by_source = {
            'fraud_investigation': 'Confirmed unauthorized purchase',
            'chargeback': 'Customer initiated chargeback',
            'customer_complaint': 'Customer reported fraud',
            'system_alert': 'Automated fraud detection confirmed',
            'merchant_report': 'Merchant reported suspicious activity',
            'normal_activity': 'Customer continued normal usage',
            'customer_verification': 'Customer verified legitimate purchase',
            'transaction_completed': 'Transaction completed successfully',
        }

        source = row.get('confirmation_source', 'unknown')
        return notes_by_source.get(source, 'No additional details')

    def write_updates_to_athena(self, updates: pd.DataFrame) -> None:
        """Write ground truth updates to Athena table."""
        logger.info(f"Writing {len(updates):,} updates to Athena table: {ATHENA_GROUND_TRUTH_UPDATES_TABLE}...")

        # Write to Athena using Iceberg append
        self.client.write_dataframe(
            df=updates,
            table_name=ATHENA_GROUND_TRUTH_UPDATES_TABLE,
            mode='append'
        )

        logger.info("✓ Ground truth updates written to Athena")

    def simulate_and_write(self, limit: int = None) -> Dict[str, Any]:
        """
        Complete workflow: load predictions, simulate ground truth, write updates.

        Returns:
            Dictionary with statistics
        """
        logger.info("=" * 80)
        logger.info("Starting Ground Truth Simulation")
        logger.info("=" * 80)

        # Load predictions without ground truth
        df = self.load_predictions_without_ground_truth(limit=limit)

        if df.empty:
            logger.warning("No predictions to process")
            return {
                'total_predictions': 0,
                'updates_created': 0,
                'actual_fraud': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'accuracy': self.accuracy,
            }

        # Simulate ground truth
        df = self.simulate_ground_truth(df)

        # Create updates dataframe
        updates = self.create_ground_truth_updates(df)

        # Write to Athena
        self.write_updates_to_athena(updates)

        # Return statistics
        stats = {
            'total_predictions': len(df),
            'updates_created': len(updates),
            'actual_fraud': int(df['actual_fraud'].sum()),
            'false_positives': int(df['false_positive'].sum()),
            'false_negatives': int(df['false_negative'].sum()),
            'accuracy': self.accuracy,
        }

        logger.info("=" * 80)
        logger.info("Simulation Complete")
        logger.info("=" * 80)
        logger.info(f"  Total predictions processed: {stats['total_predictions']:,}")
        logger.info(f"  Ground truth updates created: {stats['updates_created']:,}")
        logger.info(f"  Actual fraud: {stats['actual_fraud']:,} ({stats['actual_fraud']/stats['total_predictions']*100:.2f}%)")
        logger.info(f"  False positives: {stats['false_positives']:,}")
        logger.info(f"  False negatives: {stats['false_negatives']:,}")
        logger.info(f"  Model accuracy: {stats['accuracy']*100:.1f}%")
        logger.info("=" * 80)

        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Simulate ground truth for Athena inference predictions'
    )
    parser.add_argument(
        '--endpoint-name',
        type=str,
        help='Filter predictions by endpoint name'
    )
    parser.add_argument(
        '--accuracy',
        type=float,
        default=0.85,
        help='Model accuracy (0.0-1.0, default: 0.85)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of predictions to process'
    )
    parser.add_argument(
        '--fraud-days',
        type=str,
        default='1,7',
        help='Fraud confirmation delay range in days (min,max, default: 1,7)'
    )
    parser.add_argument(
        '--non-fraud-days',
        type=str,
        default='1,30',
        help='Non-fraud confirmation delay range in days (min,max, default: 1,30)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Parse day ranges
    fraud_days = tuple(map(int, args.fraud_days.split(',')))
    non_fraud_days = tuple(map(int, args.non_fraud_days.split(',')))

    # Initialize Athena client
    athena_client = AthenaClient()

    # Create simulator
    simulator = GroundTruthSimulator(
        athena_client=athena_client,
        endpoint_name=args.endpoint_name,
        accuracy=args.accuracy,
        fraud_confirmation_days=fraud_days,
        non_fraud_confirmation_days=non_fraud_days,
        seed=args.seed,
    )

    # Run simulation
    try:
        stats = simulator.simulate_and_write(limit=args.limit)

        if stats['updates_created'] > 0:
            print("\n✅ Ground truth simulation complete!")
            print(f"\nNext steps:")
            print(f"  1. Run update_ground_truth.py to merge updates into inference_responses:")
            print(f"     python scripts/update_ground_truth.py --mode batch")
            print(f"\n  2. Check coverage in inference monitoring notebook (Cell 16)")
            print(f"\n  3. Run monitoring to detect drift (Cells 21-34)")

        return 0

    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

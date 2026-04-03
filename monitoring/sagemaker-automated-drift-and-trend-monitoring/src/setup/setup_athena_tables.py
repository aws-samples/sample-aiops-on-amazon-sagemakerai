#!/usr/bin/env python3
"""
One-time setup script for Athena tables.

This script:
1. Creates S3 bucket (if needed)
2. Creates Athena database
3. Creates all required tables
4. Optionally migrates CSV data to Iceberg tables
5. Verifies setup and prints summary

Usage:
    python scripts/setup_athena_tables.py
    python scripts/setup_athena_tables.py --migrate-data
    python scripts/setup_athena_tables.py --verify-only
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables BEFORE importing config
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from: {env_path}")
except ImportError:
    print("Warning: python-dotenv not installed, using system environment variables only")

import boto3
from src.config.config import (
    DATA_S3_BUCKET,
    ATHENA_DATABASE,
    S3_TRAINING_DATA,
    S3_GROUND_TRUTH,
)
from src.train_pipeline.athena.iceberg_manager import IcebergManager
from src.train_pipeline.athena.schema_definitions import list_all_tables
from src.utils.aws_session import create_boto3_session

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_s3_bucket(bucket_name: str, region: str = None, boto3_session: boto3.Session = None) -> bool:
    """
    Create S3 bucket if it doesn't exist.

    Args:
        bucket_name: Name of bucket to create
        region: AWS region (if None, will auto-detect)
        boto3_session: Existing boto3 session to use

    Returns:
        True if successful or already exists
    """
    try:
        # Use provided session or create new one
        if boto3_session is None:
            boto3_session = create_boto3_session(region_name=region)
        detected_region = boto3_session.region_name
        logger.info(f"Using AWS region: {detected_region}")

        # Create S3 client with explicit region
        s3_client = boto3_session.client('s3', region_name=detected_region)
        
        # Debug: Log the actual endpoint being used
        logger.info(f"S3 client endpoint: {s3_client.meta.endpoint_url}")
        logger.info(f"S3 client region: {s3_client.meta.region_name}")

        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"✓ S3 bucket {bucket_name} already exists")
            return True
        except:
            pass

        # Create bucket
        logger.info(f"Creating S3 bucket: {bucket_name} in region {detected_region}")

        # For us-east-1, we must NOT specify LocationConstraint at all
        # For other regions, we must specify it
        if detected_region == 'us-east-1':
            # us-east-1 requires NO CreateBucketConfiguration
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': detected_region}
            )

        # Enable versioning
        s3_client.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )

        logger.info(f"✓ S3 bucket {bucket_name} created successfully")
        return True

    except Exception as e:
        logger.error(f"Error creating S3 bucket: {e}")
        return False


def setup_database(manager: IcebergManager) -> bool:
    """
    Create Athena database.

    Args:
        manager: IcebergManager instance

    Returns:
        True if successful
    """
    try:
        logger.info(f"Setting up database: {manager.database}")

        if manager.database_exists():
            logger.info(f"✓ Database {manager.database} already exists")
        else:
            manager.create_database()

        return True

    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False


def setup_tables(manager: IcebergManager, skip_existing: bool = True) -> dict:
    """
    Create all required tables.

    Args:
        manager: IcebergManager instance
        skip_existing: Skip tables that already exist

    Returns:
        Dictionary with table creation results
    """
    try:
        logger.info("Setting up tables...")
        results = manager.create_all_tables(skip_existing=skip_existing)

        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)

        if success_count == total_count:
            logger.info(f"✓ All {total_count} tables created successfully")
        else:
            logger.warning(f"Created {success_count}/{total_count} tables")

        return results

    except Exception as e:
        logger.error(f"Error setting up tables: {e}")
        return {}


def migrate_csv_data(manager: IcebergManager) -> dict:
    """
    Migrate CSV data to Iceberg tables.

    Args:
        manager: IcebergManager instance

    Returns:
        Dictionary with migration results
    """
    from src.train_pipeline.athena.data_migrator import DataMigrator

    results = {}
    migrator = DataMigrator(
        database=manager.database,
        workgroup=manager.workgroup,
        s3_output=manager.s3_output,
    )

    # Migration mappings: S3 data path -> table name
    migrations = [
        (S3_TRAINING_DATA, 'training_data'),
        (S3_GROUND_TRUTH, 'ground_truth'),
    ]

    logger.info("Migrating data from S3 to Athena Iceberg tables...")

    for s3_path, table_name in migrations:
        try:
            if not s3_path:
                logger.warning(f"S3 path not configured for {table_name}, skipping. "
                               "Run upload_data_to_s3.py first and set S3 data paths in config.yaml")
                results[table_name] = {'success': False, 'error': 'S3 path not configured'}
                continue

            logger.info(f"Migrating {s3_path} -> {table_name}")

            success, stats = migrator.migrate_csv_to_iceberg(
                csv_path=s3_path,
                table_name=table_name,
                chunk_size=10000,
            )

            results[table_name] = {
                'success': success,
                'rows_migrated': stats.get('total_rows', 0),
                'chunks_processed': stats.get('chunks_processed', 0),
            }

            if success:
                logger.info(f"✓ Migrated {stats.get('total_rows', 0)} rows to {table_name}")
            else:
                logger.error(f"✗ Failed to migrate {table_name}")

        except Exception as e:
            logger.error(f"Error migrating {table_name}: {e}")
            results[table_name] = {'success': False, 'error': str(e)}

    return results


def verify_setup(manager: IcebergManager) -> dict:
    """
    Verify all tables exist and have data.

    Args:
        manager: IcebergManager instance

    Returns:
        Dictionary with verification results
    """
    try:
        logger.info("Verifying setup...")
        results = manager.verify_all_tables()

        print("\n" + "=" * 80)
        print("VERIFICATION RESULTS")
        print("=" * 80)

        for table_name, stats in results.items():
            print(f"\nTable: {table_name}")
            if stats.get('exists'):
                print(f"  Status: ✓ EXISTS")
                print(f"  Rows: {stats.get('row_count', 0):,}")
                print(f"  Type: {stats.get('table_type', 'unknown')}")
                print(f"  Iceberg: {stats.get('is_iceberg', False)}")
                print(f"  Partitioned: {stats.get('is_partitioned', False)}")
                print(f"  Location: {stats.get('location', 'unknown')}")
            else:
                print(f"  Status: ✗ DOES NOT EXIST")
                print(f"  Error: {stats.get('error', 'unknown')}")

        print("\n" + "=" * 80)
        return results

    except Exception as e:
        logger.error(f"Error verifying setup: {e}")
        return {}


def print_summary(
    s3_success: bool,
    db_success: bool,
    table_results: dict,
    migration_results: dict,
    verification_results: dict,
):
    """Print setup summary."""
    print("\n" + "=" * 80)
    print("SETUP SUMMARY")
    print("=" * 80)

    # S3 Bucket
    print(f"\nS3 Bucket: {'✓' if s3_success else '✗'} {DATA_S3_BUCKET}")

    # Database
    print(f"Athena Database: {'✓' if db_success else '✗'} {ATHENA_DATABASE}")

    # Tables
    print(f"\nTables Created:")
    for table_name, success in table_results.items():
        status = '✓' if success else '✗'
        print(f"  {status} {table_name}")

    # Migration
    if migration_results:
        print(f"\nData Migration:")
        for table_name, result in migration_results.items():
            status = '✓' if result.get('success') else '✗'
            rows = result.get('rows_migrated', 0)
            print(f"  {status} {table_name}: {rows:,} rows")

    # Verification
    if verification_results:
        total_rows = sum(
            stats.get('row_count', 0)
            for stats in verification_results.values()
            if stats.get('exists')
        )
        existing_tables = sum(
            1 for stats in verification_results.values()
            if stats.get('exists')
        )
        print(f"\nVerification:")
        print(f"  Tables: {existing_tables}/{len(verification_results)}")
        print(f"  Total Rows: {total_rows:,}")

    # Overall Status
    all_success = (
        s3_success and
        db_success and
        all(table_results.values()) and
        (not migration_results or all(r.get('success', False) for r in migration_results.values()))
    )

    print("\n" + "=" * 80)
    if all_success:
        print("✓ SETUP COMPLETED SUCCESSFULLY")
    else:
        print("✗ SETUP COMPLETED WITH ERRORS")
    print("=" * 80 + "\n")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description='Setup Athena tables for fraud detection pipeline'
    )
    parser.add_argument(
        '--migrate-data',
        action='store_true',
        help='Migrate CSV data to Iceberg tables'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing setup without creating anything'
    )
    parser.add_argument(
        '--region',
        default=None,
        help='AWS region (if not specified, will auto-detect from environment)'
    )
    parser.add_argument(
        '--skip-s3',
        action='store_true',
        help='Skip S3 bucket creation'
    )

    args = parser.parse_args()

    # Create boto3 session with proper region handling
    boto3_session = create_boto3_session(region_name=args.region)
    detected_region = boto3_session.region_name

    print("=" * 80)
    print("ATHENA TABLES SETUP")
    print("=" * 80)
    print(f"Database: {ATHENA_DATABASE}")
    print(f"S3 Bucket: {DATA_S3_BUCKET}")
    print(f"Region: {detected_region}")
    print("=" * 80 + "\n")

    # Initialize manager with boto3 session
    manager = IcebergManager(boto3_session=boto3_session)

    # Verify only mode
    if args.verify_only:
        logger.info("Running in verify-only mode")
        verification_results = verify_setup(manager)
        return 0 if all(r.get('exists') for r in verification_results.values()) else 1

    # Step 1: Create S3 bucket
    if args.skip_s3:
        logger.info("Skipping S3 bucket creation")
        s3_success = True
    else:
        s3_success = create_s3_bucket(DATA_S3_BUCKET, detected_region, boto3_session)
        if not s3_success:
            logger.error("Failed to create S3 bucket, aborting")
            return 1

    # Step 2: Create database
    db_success = setup_database(manager)
    if not db_success:
        logger.error("Failed to create database, aborting")
        return 1

    # Step 3: Create tables
    table_results = setup_tables(manager, skip_existing=True)
    if not table_results:
        logger.error("Failed to create tables, aborting")
        return 1

    # Step 4: Migrate data (optional)
    migration_results = {}
    if args.migrate_data:
        migration_results = migrate_csv_data(manager)

    # Step 5: Verify setup
    verification_results = verify_setup(manager)

    # Print summary
    print_summary(
        s3_success,
        db_success,
        table_results,
        migration_results,
        verification_results,
    )

    # Return success if all critical steps succeeded
    all_success = (
        s3_success and
        db_success and
        all(table_results.values())
    )

    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())

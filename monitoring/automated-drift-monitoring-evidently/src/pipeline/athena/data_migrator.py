"""
⚠️ DEPRECATED: This file uses awswrangler for CSV to Athena migration.

**Replacement:** Use PySpark-based migration for large datasets.

Data migration utilities for CSV to Athena Iceberg tables.

Handles:
- Chunked reading of large CSV files
- Data type transformations
- Metadata enrichment
- Progress tracking and validation

**Migration:**
- For large CSV files (>1M rows): Use PySpark for distributed processing
- For small one-time migrations: Can continue using this
- For new development: Use PySpark-based migration

**Deprecated:** February 2026 - PySpark migration
"""

import logging
import warnings

# Show deprecation warning on import
warnings.warn(
    "DataMigrator (awswrangler-based) is deprecated. "
    "Use PySpark-based migration for large datasets.",
    DeprecationWarning,
    stacklevel=2
)

import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import boto3
import awswrangler as wr

from src.config.config import (
    ATHENA_DATABASE,
    ATHENA_WORKGROUP,
    ATHENA_OUTPUT_S3,
    ATHENA_TRAINING_TABLE,
    ATHENA_GROUND_TRUTH_TABLE,
    ATHENA_DRIFTED_DATA_TABLE,
)
from .schema_definitions import CSV_TO_ATHENA_COLUMN_MAP, TYPE_CONVERSIONS

logger = logging.getLogger(__name__)


class DataMigrator:
    """
    Migrate CSV data to Athena Iceberg tables.

    Handles large files with chunked reading, type conversions,
    and metadata enrichment.
    """

    def __init__(
        self,
        database: str = ATHENA_DATABASE,
        workgroup: str = ATHENA_WORKGROUP,
        s3_output: str = ATHENA_OUTPUT_S3,
        boto3_session: Optional[boto3.Session] = None,
    ):
        """
        Initialize data migrator.

        Args:
            database: Athena database name
            workgroup: Athena workgroup name
            s3_output: S3 path for query results
            boto3_session: Optional boto3 session
        """
        self.database = database
        self.workgroup = workgroup
        self.s3_output = s3_output
        self.boto3_session = boto3_session or boto3.Session()

        logger.info(f"Initialized DataMigrator for database: {database}")

    def prepare_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        add_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare DataFrame for Athena write.

        Args:
            df: Source DataFrame
            table_name: Target table name
            add_metadata: Whether to add metadata columns

        Returns:
            Prepared DataFrame
        """
        # Create copy to avoid modifying original
        df = df.copy()

        # Rename columns based on mapping
        if CSV_TO_ATHENA_COLUMN_MAP:
            df = df.rename(columns=CSV_TO_ATHENA_COLUMN_MAP)

        # Convert data types
        for col, target_type in TYPE_CONVERSIONS.items():
            if col in df.columns:
                if target_type == 'boolean':
                    df[col] = df[col].astype(bool)
                elif target_type == 'string':
                    df[col] = df[col].astype(str)
                elif target_type == 'timestamp':
                    # Convert to pandas datetime (which awswrangler writes as TIMESTAMP)
                    df[col] = pd.to_datetime(df[col])
                elif target_type == 'int':
                    # Convert to int32 (maps to Athena INT)
                    df[col] = df[col].astype('int32')
                elif target_type == 'bigint':
                    # Convert to int64 (maps to Athena BIGINT)
                    df[col] = df[col].astype('int64')
                elif target_type == 'double':
                    # Convert to float64 (maps to Athena DOUBLE)
                    df[col] = df[col].astype('float64')

        # Add metadata columns for training_data table
        if add_metadata and table_name == ATHENA_TRAINING_TABLE:
            if 'data_version' not in df.columns:
                df['data_version'] = 'v1'
            if 'created_at' not in df.columns:
                df['created_at'] = datetime.utcnow()
            if 'updated_at' not in df.columns:
                df['updated_at'] = datetime.utcnow()

        # Add metadata for ground_truth table
        if add_metadata and table_name == ATHENA_GROUND_TRUTH_TABLE:
            if 'data_source' not in df.columns:
                df['data_source'] = 'csv_migration'
            if 'ingestion_timestamp' not in df.columns:
                df['ingestion_timestamp'] = datetime.utcnow()
            if 'batch_id' not in df.columns:
                df['batch_id'] = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        # Generate transaction_id if missing
        if 'transaction_id' not in df.columns:
            df['transaction_id'] = [f'TXN_{i:08d}' for i in range(len(df))]

        return df

    def migrate_csv_to_iceberg(
        self,
        csv_path: str,
        table_name: str,
        chunk_size: int = 10000,
        skip_existing: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Migrate CSV file to Iceberg table in chunks.

        Args:
            csv_path: Path to CSV file
            table_name: Target table name
            chunk_size: Number of rows per chunk
            skip_existing: Skip if table already has data

        Returns:
            Tuple of (success, statistics)

        Example:
            >>> migrator = DataMigrator()
            >>> success, stats = migrator.migrate_csv_to_iceberg(
            ...     'data/creditcard.csv',
            ...     'training_data',
            ...     chunk_size=10000
            ... )
        """
        stats = {
            'csv_path': csv_path,
            'table_name': table_name,
            'total_rows': 0,
            'chunks_processed': 0,
            'start_time': datetime.utcnow().isoformat(),
            'end_time': None,
            'success': False,
            'errors': [],
        }

        try:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                error_msg = f"CSV file not found: {csv_path}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
                return False, stats

            logger.info(f"Starting migration: {csv_path} -> {table_name}")

            # Check if table already has data
            if skip_existing:
                try:
                    count_query = f"SELECT COUNT(*) as count FROM {self.database}.{table_name}"
                    count_df = wr.athena.read_sql_query(
                        sql=count_query,
                        database=self.database,
                        workgroup=self.workgroup,
                        s3_output=self.s3_output,
                        boto3_session=self.boto3_session,
                    )
                    existing_count = int(count_df['count'].iloc[0])
                    if existing_count > 0:
                        logger.info(f"Table {table_name} already has {existing_count} rows, skipping")
                        stats['total_rows'] = existing_count
                        stats['success'] = True
                        stats['end_time'] = datetime.utcnow().isoformat()
                        return True, stats
                except:
                    pass  # Table doesn't exist yet

            # Read and process CSV in chunks
            chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size)

            for chunk_num, chunk_df in enumerate(chunk_iter, 1):
                try:
                    logger.info(f"Processing chunk {chunk_num} ({len(chunk_df)} rows)")

                    # Prepare data
                    prepared_df = self.prepare_dataframe(chunk_df, table_name)

                    # Write to Iceberg table
                    wr.athena.to_iceberg(
                        df=prepared_df,
                        database=self.database,
                        table=table_name,
                        temp_path=f"{self.s3_output}temp/",
                        boto3_session=self.boto3_session,
                        keep_files=False,
                    )

                    stats['total_rows'] += len(prepared_df)
                    stats['chunks_processed'] += 1

                    logger.info(
                        f"✓ Chunk {chunk_num} written "
                        f"({stats['total_rows']} total rows)"
                    )

                except Exception as e:
                    error_msg = f"Error processing chunk {chunk_num}: {e}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
                    # Continue with next chunk

            # Finalize stats
            stats['end_time'] = datetime.utcnow().isoformat()
            stats['success'] = stats['chunks_processed'] > 0 and len(stats['errors']) == 0

            if stats['success']:
                logger.info(
                    f"✓ Migration completed: {stats['total_rows']} rows "
                    f"in {stats['chunks_processed']} chunks"
                )
            else:
                logger.error(
                    f"✗ Migration completed with errors: "
                    f"{len(stats['errors'])} errors encountered"
                )

            return stats['success'], stats

        except Exception as e:
            error_msg = f"Migration failed: {e}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            stats['end_time'] = datetime.utcnow().isoformat()
            return False, stats

    def validate_migration(
        self,
        csv_path: str,
        table_name: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that CSV data was migrated correctly.

        Args:
            csv_path: Path to source CSV file
            table_name: Target table name

        Returns:
            Tuple of (valid, comparison_stats)
        """
        try:
            logger.info(f"Validating migration for {table_name}")

            # Count rows in CSV
            csv_df = pd.read_csv(csv_path)
            csv_row_count = len(csv_df)

            # Count rows in Athena table
            count_query = f"SELECT COUNT(*) as count FROM {self.database}.{table_name}"
            count_df = wr.athena.read_sql_query(
                sql=count_query,
                database=self.database,
                workgroup=self.workgroup,
                s3_output=self.s3_output,
                boto3_session=self.boto3_session,
            )
            athena_row_count = int(count_df['count'].iloc[0])

            # Compare
            matches = csv_row_count == athena_row_count
            stats = {
                'csv_rows': csv_row_count,
                'athena_rows': athena_row_count,
                'matches': matches,
                'difference': athena_row_count - csv_row_count,
            }

            if matches:
                logger.info(f"✓ Validation passed: {csv_row_count} rows match")
            else:
                logger.warning(
                    f"✗ Validation failed: CSV has {csv_row_count} rows, "
                    f"Athena has {athena_row_count} rows "
                    f"(diff: {stats['difference']})"
                )

            return matches, stats

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, {'error': str(e)}

    def get_migration_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all migrated tables.

        Returns:
            Dictionary with stats for each table
        """
        stats = {}

        for table_name in [ATHENA_TRAINING_TABLE, ATHENA_GROUND_TRUTH_TABLE, ATHENA_DRIFTED_DATA_TABLE]:
            try:
                # Get row count
                count_query = f"SELECT COUNT(*) as count FROM {self.database}.{table_name}"
                count_df = wr.athena.read_sql_query(
                    sql=count_query,
                    database=self.database,
                    workgroup=self.workgroup,
                    s3_output=self.s3_output,
                    boto3_session=self.boto3_session,
                )
                row_count = int(count_df['count'].iloc[0])

                stats[table_name] = {
                    'row_count': row_count,
                    'exists': True,
                }

            except Exception as e:
                logger.warning(f"Could not get stats for {table_name}: {e}")
                stats[table_name] = {
                    'row_count': 0,
                    'exists': False,
                    'error': str(e),
                }

        return stats

    def export_sample_to_csv(
        self,
        table_name: str,
        output_path: str,
        limit: int = 1000,
        filters: Optional[str] = None,
    ) -> bool:
        """
        Export sample data from Athena table to CSV.

        Args:
            table_name: Source table name
            output_path: Destination CSV path
            limit: Maximum number of rows
            filters: Optional SQL WHERE clause

        Returns:
            True if successful
        """
        try:
            logger.info(f"Exporting sample from {table_name} to {output_path}")

            # Build query
            query = f"SELECT * FROM {self.database}.{table_name}"
            if filters:
                query += f" WHERE {filters}"
            query += f" LIMIT {limit}"

            # Execute query
            df = wr.athena.read_sql_query(
                sql=query,
                database=self.database,
                workgroup=self.workgroup,
                s3_output=self.s3_output,
                boto3_session=self.boto3_session,
            )

            # Write to CSV
            df.to_csv(output_path, index=False)

            logger.info(f"✓ Exported {len(df)} rows to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting sample: {e}")
            return False


if __name__ == '__main__':
    """Test data migrator functionality."""
    import sys
    from src.config.config import CSV_TRAINING_DATA, CSV_GROUND_TRUTH

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize migrator
    migrator = DataMigrator()

    print("=" * 80)
    print("Data Migration Test")
    print("=" * 80)

    # Test training data migration
    if CSV_TRAINING_DATA.exists():
        print(f"\nMigrating {CSV_TRAINING_DATA.name}...")
        success, stats = migrator.migrate_csv_to_iceberg(
            csv_path=str(CSV_TRAINING_DATA),
            table_name=ATHENA_TRAINING_TABLE,
            chunk_size=10000,
            skip_existing=True,
        )

        if success:
            print(f"✓ Success: {stats['total_rows']} rows migrated")

            # Validate
            valid, validation_stats = migrator.validate_migration(
                str(CSV_TRAINING_DATA),
                ATHENA_TRAINING_TABLE
            )
            print(f"Validation: {'✓ PASS' if valid else '✗ FAIL'}")
        else:
            print(f"✗ Failed: {stats.get('errors', [])}")
    else:
        print(f"✗ Training data not found: {CSV_TRAINING_DATA}")

    # Get overall stats
    print("\n" + "=" * 80)
    print("Migration Statistics")
    print("=" * 80)

    all_stats = migrator.get_migration_stats()
    for table_name, table_stats in all_stats.items():
        if table_stats.get('exists'):
            print(f"{table_name}: {table_stats['row_count']:,} rows")
        else:
            print(f"{table_name}: Not found or empty")

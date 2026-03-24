"""
Iceberg table management for fraud detection pipeline.

Provides utilities for:
- Creating Athena database
- Creating Iceberg tables
- Table existence checks
- Table optimization (compaction)
- Snapshot management
"""

import logging
import time
from typing import Optional, Dict, Any, List
import boto3

from src.config.config import ATHENA_DATABASE, ATHENA_WORKGROUP, ATHENA_OUTPUT_S3
from src.utils.aws_session import create_boto3_session
from .schema_definitions import (
    ALL_TABLE_DEFINITIONS,
    get_iceberg_tables,
    get_partitioned_tables,
)

logger = logging.getLogger(__name__)


def execute_athena_query(
    sql: str,
    database: str,
    workgroup: str,
    s3_output: str,
    boto3_session: boto3.Session,
    wait: bool = True
) -> Optional[str]:
    """
    Execute Athena query using boto3 (replacement for awswrangler).

    Args:
        sql: SQL query to execute
        database: Athena database name
        workgroup: Athena workgroup
        s3_output: S3 output location
        boto3_session: boto3 Session
        wait: Wait for query to complete

    Returns:
        Query execution ID if successful
    """
    athena = boto3_session.client('athena')

    # Start query execution
    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={'Database': database},
        WorkGroup=workgroup,
        ResultConfiguration={'OutputLocation': s3_output}
    )

    query_execution_id = response['QueryExecutionId']

    if wait:
        # Wait for query to complete
        max_wait_time = 300  # 5 minutes
        poll_interval = 2
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            response = athena.get_query_execution(QueryExecutionId=query_execution_id)
            state = response['QueryExecution']['Status']['State']

            if state == 'SUCCEEDED':
                return query_execution_id
            elif state in ['FAILED', 'CANCELLED']:
                reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
                raise Exception(f"Query {state}: {reason}")

            time.sleep(poll_interval)
            elapsed_time += poll_interval

        raise Exception(f"Query timed out after {max_wait_time} seconds")

    return query_execution_id


class IcebergManager:
    """
    Manager for Iceberg table operations in Athena.

    Handles database and table creation, optimization, and maintenance.
    """

    def __init__(
        self,
        database: str = ATHENA_DATABASE,
        workgroup: str = ATHENA_WORKGROUP,
        s3_output: str = ATHENA_OUTPUT_S3,
        boto3_session: Optional[boto3.Session] = None,
    ):
        """
        Initialize Iceberg manager.

        Args:
            database: Athena database name
            workgroup: Athena workgroup name
            s3_output: S3 path for query results
            boto3_session: Optional boto3 session
        """
        self.database = database
        self.workgroup = workgroup
        self.s3_output = s3_output
        self.boto3_session = boto3_session or create_boto3_session()

        logger.info(f"Initialized IcebergManager for database: {database}")
        logger.info(f"Using AWS region: {self.boto3_session.region_name}")

    def create_database(self, if_not_exists: bool = True) -> bool:
        """
        Create Athena database.

        Args:
            if_not_exists: Add IF NOT EXISTS clause

        Returns:
            True if successful
        """
        try:
            exists_clause = "IF NOT EXISTS " if if_not_exists else ""
            sql = f"CREATE DATABASE {exists_clause}{self.database}"

            logger.info(f"Creating database: {self.database}")

            execute_athena_query(
                sql=sql,
                database=self.database,
                workgroup=self.workgroup,
                s3_output=self.s3_output,
                boto3_session=self.boto3_session,
                wait=True,
            )

            logger.info(f"✓ Database {self.database} created successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise

    def database_exists(self) -> bool:
        """
        Check if database exists.

        Returns:
            True if database exists
        """
        try:
            # Use Glue client to list databases
            glue = self.boto3_session.client('glue')
            response = glue.get_databases()
            databases = [db['Name'] for db in response.get('DatabaseList', [])]
            exists = self.database in databases
            logger.info(f"Database {self.database} exists: {exists}")
            return exists

        except Exception as e:
            logger.warning(f"Error checking database existence: {e}")
            return False

    def create_table(
        self,
        table_name: str,
        if_not_exists: bool = True,
    ) -> bool:
        """
        Create Iceberg table using predefined schema.

        Args:
            table_name: Name of table to create
            if_not_exists: Add IF NOT EXISTS clause (already in DDL)

        Returns:
            True if successful
        """
        try:
            if table_name not in ALL_TABLE_DEFINITIONS:
                raise ValueError(
                    f"Unknown table: {table_name}. "
                    f"Valid tables: {list(ALL_TABLE_DEFINITIONS.keys())}"
                )

            logger.info(f"Creating table: {self.database}.{table_name}")

            # Get DDL from schema definitions
            ddl = ALL_TABLE_DEFINITIONS[table_name]

            # Execute CREATE TABLE
            execute_athena_query(
                sql=ddl,
                database=self.database,
                workgroup=self.workgroup,
                s3_output=self.s3_output,
                boto3_session=self.boto3_session,
                wait=True,
            )

            logger.info(f"✓ Table {table_name} created successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists
        """
        try:
            # Use Glue client to list tables
            glue = self.boto3_session.client('glue')
            response = glue.get_tables(DatabaseName=self.database)
            tables = [table['Name'] for table in response.get('TableList', [])]
            exists = table_name in tables
            logger.info(f"Table {self.database}.{table_name} exists: {exists}")
            return exists

        except Exception as e:
            logger.warning(f"Error checking table existence: {e}")
            return False

    def drop_table(self, table_name: str, if_exists: bool = True) -> bool:
        """
        Drop table.

        Args:
            table_name: Name of table to drop
            if_exists: Add IF EXISTS clause

        Returns:
            True if successful
        """
        try:
            exists_clause = "IF EXISTS " if if_exists else ""
            sql = f"DROP TABLE {exists_clause}{self.database}.{table_name}"

            logger.warning(f"Dropping table: {self.database}.{table_name}")

            execute_athena_query(
                sql=sql,
                database=self.database,
                workgroup=self.workgroup,
                s3_output=self.s3_output,
                boto3_session=self.boto3_session,
                wait=True,
            )

            logger.info(f"✓ Table {table_name} dropped successfully")
            return True

        except Exception as e:
            logger.error(f"Error dropping table {table_name}: {e}")
            raise

    def optimize_table(self, table_name: str) -> bool:
        """
        Optimize Iceberg table by running compaction.

        Args:
            table_name: Name of table to optimize

        Returns:
            True if successful

        Note:
            This runs OPTIMIZE command which compacts small files.
            Only works for Iceberg tables.
        """
        try:
            if table_name not in get_iceberg_tables():
                logger.warning(f"{table_name} is not an Iceberg table, skipping optimization")
                return False

            logger.info(f"Optimizing table: {self.database}.{table_name}")

            # Iceberg OPTIMIZE command
            sql = f"OPTIMIZE {self.database}.{table_name} REWRITE DATA USING BIN_PACK"

            execute_athena_query(
                sql=sql,
                database=self.database,
                workgroup=self.workgroup,
                s3_output=self.s3_output,
                boto3_session=self.boto3_session,
                wait=True,
            )

            logger.info(f"✓ Table {table_name} optimized successfully")
            return True

        except Exception as e:
            logger.error(f"Error optimizing table {table_name}: {e}")
            # Don't raise - optimization is best-effort
            return False

    def vacuum_table(self, table_name: str, older_than_days: int = 7) -> bool:
        """
        Remove orphaned files from Iceberg table.

        Args:
            table_name: Name of table to vacuum
            older_than_days: Remove files older than this many days

        Returns:
            True if successful
        """
        try:
            if table_name not in get_iceberg_tables():
                logger.warning(f"{table_name} is not an Iceberg table, skipping vacuum")
                return False

            logger.info(f"Vacuuming table: {self.database}.{table_name}")

            # Iceberg VACUUM command
            sql = f"""
            VACUUM {self.database}.{table_name}
            USING (older_than => TIMESTAMP '{older_than_days} days ago')
            """

            execute_athena_query(
                sql=sql,
                database=self.database,
                workgroup=self.workgroup,
                s3_output=self.s3_output,
                boto3_session=self.boto3_session,
                wait=True,
            )

            logger.info(f"✓ Table {table_name} vacuumed successfully")
            return True

        except Exception as e:
            logger.error(f"Error vacuuming table {table_name}: {e}")
            return False

    def expire_snapshots(
        self,
        table_name: str,
        older_than_days: int = 7,
    ) -> bool:
        """
        Expire old Iceberg snapshots.

        Args:
            table_name: Name of table
            older_than_days: Expire snapshots older than this

        Returns:
            True if successful
        """
        try:
            if table_name not in get_iceberg_tables():
                logger.warning(f"{table_name} is not an Iceberg table, skipping")
                return False

            logger.info(f"Expiring snapshots for: {self.database}.{table_name}")

            # Iceberg expire_snapshots procedure
            sql = f"""
            CALL {self.database}.system.expire_snapshots(
                table_name => '{table_name}',
                older_than => TIMESTAMP '{older_than_days} days ago'
            )
            """

            execute_athena_query(
                sql=sql,
                database=self.database,
                workgroup=self.workgroup,
                s3_output=self.s3_output,
                boto3_session=self.boto3_session,
                wait=True,
            )

            logger.info(f"✓ Snapshots expired for {table_name}")
            return True

        except Exception as e:
            logger.error(f"Error expiring snapshots for {table_name}: {e}")
            return False

    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics for Iceberg table.

        Args:
            table_name: Name of table

        Returns:
            Dictionary with table statistics
        """
        try:
            logger.info(f"Getting stats for: {self.database}.{table_name}")

            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {self.database}.{table_name}"
            count_df = wr.athena.read_sql_query(
                sql=count_query,
                database=self.database,
                workgroup=self.workgroup,
                s3_output=self.s3_output,
                boto3_session=self.boto3_session,
            )
            row_count = int(count_df['row_count'].iloc[0]) if not count_df.empty else 0

            # Get table metadata
            table_metadata = wr.catalog.table(
                database=self.database,
                table=table_name,
                boto3_session=self.boto3_session,
            )

            stats = {
                'table_name': table_name,
                'database': self.database,
                'row_count': row_count,
                'location': table_metadata['Location'].iloc[0] if 'Location' in table_metadata.columns and not table_metadata.empty else 'unknown',
                'table_type': table_metadata['TableType'].iloc[0] if 'TableType' in table_metadata.columns and not table_metadata.empty else 'unknown',
                'is_iceberg': table_name in get_iceberg_tables(),
                'is_partitioned': table_name in get_partitioned_tables(),
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting stats for {table_name}: {e}")
            raise

    def create_all_tables(self, skip_existing: bool = True) -> Dict[str, bool]:
        """
        Create all tables defined in schema_definitions.

        Args:
            skip_existing: Skip tables that already exist

        Returns:
            Dictionary mapping table names to creation success
        """
        results = {}

        logger.info("Creating all tables...")

        for table_name in ALL_TABLE_DEFINITIONS.keys():
            try:
                if skip_existing and self.table_exists(table_name):
                    logger.info(f"Table {table_name} already exists, skipping")
                    results[table_name] = True
                    continue

                results[table_name] = self.create_table(table_name)

            except Exception as e:
                logger.error(f"Failed to create table {table_name}: {e}")
                results[table_name] = False

        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Created {success_count}/{len(results)} tables successfully")

        return results

    def verify_all_tables(self) -> Dict[str, Dict[str, Any]]:
        """
        Verify all tables exist and get their stats.

        Returns:
            Dictionary mapping table names to their stats
        """
        results = {}

        logger.info("Verifying all tables...")

        for table_name in ALL_TABLE_DEFINITIONS.keys():
            try:
                if not self.table_exists(table_name):
                    results[table_name] = {
                        'exists': False,
                        'error': 'Table does not exist'
                    }
                    continue

                stats = self.get_table_stats(table_name)
                stats['exists'] = True
                results[table_name] = stats

            except Exception as e:
                logger.error(f"Error verifying table {table_name}: {e}")
                results[table_name] = {
                    'exists': False,
                    'error': str(e)
                }

        return results


if __name__ == '__main__':
    """Test Iceberg manager functionality."""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize manager
    manager = IcebergManager()

    # Check if database exists
    if not manager.database_exists():
        print(f"\nDatabase {manager.database} does not exist. Creating...")
        manager.create_database()
    else:
        print(f"\n✓ Database {manager.database} exists")

    # Verify all tables
    print("\nVerifying tables:")
    print("=" * 80)

    results = manager.verify_all_tables()
    for table_name, stats in results.items():
        if stats.get('exists'):
            print(f"✓ {table_name}:")
            print(f"    Rows: {stats.get('row_count', 0)}")
            print(f"    Type: {stats.get('table_type', 'unknown')}")
            print(f"    Iceberg: {stats.get('is_iceberg', False)}")
        else:
            print(f"✗ {table_name}: {stats.get('error', 'unknown error')}")

    print("\n" + "=" * 80)

    # Summary
    existing_tables = sum(1 for stats in results.values() if stats.get('exists'))
    total_tables = len(results)
    print(f"\nSummary: {existing_tables}/{total_tables} tables exist")

"""
⚠️ DEPRECATED: This file uses awswrangler (pandas-based).

**Replacement:** Use `athena_client_pyspark.py` for distributed processing.

Athena client wrapper for fraud detection pipeline.

Provides simplified interface for:
- Reading tables as pandas DataFrames
- Writing DataFrames to Iceberg tables
- Executing SQL queries
- Table information and metadata

**Migration:**
- For large datasets (>1M rows): Use AthenaClientPySpark
- For small queries (<10K rows): Can continue using this (monitoring use case)
- For new development: Always use PySpark version

**Deprecated:** February 2026 - PySpark migration
"""

import logging
import warnings

# Show deprecation warning on import
warnings.warn(
    "AthenaClient (awswrangler-based) is deprecated. "
    "Use AthenaClientPySpark from athena_client_pyspark.py for distributed processing.",
    DeprecationWarning,
    stacklevel=2
)
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import awswrangler as wr
import boto3
from datetime import datetime

from src.config.config import (
    ATHENA_DATABASE,
    ATHENA_WORKGROUP,
    ATHENA_OUTPUT_S3,
    ATHENA_QUERY_TIMEOUT,
)

logger = logging.getLogger(__name__)


class AthenaClient:
    """
    Client for interacting with Athena tables using AWS Data Wrangler.

    Provides high-level methods for reading, writing, and querying data
    in Athena Iceberg tables.
    """

    def __init__(
        self,
        database: str = ATHENA_DATABASE,
        workgroup: str = ATHENA_WORKGROUP,
        s3_output: str = ATHENA_OUTPUT_S3,
        boto3_session: Optional[boto3.Session] = None,
    ):
        """
        Initialize Athena client.

        Args:
            database: Athena database name
            workgroup: Athena workgroup name
            s3_output: S3 path for query results
            boto3_session: Optional boto3 session (uses default if None)
        """
        self.database = database
        self.workgroup = workgroup
        self.s3_output = s3_output

        logger.info(f"Initialized AthenaClient for database: {database}")

    def read_table(
        self,
        table_name: str,
        filters: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Read entire table or filtered subset into a pandas DataFrame.

        Args:
            table_name: Name of the table (without database prefix)
            filters: Optional SQL WHERE clause (without 'WHERE' keyword)
            columns: Optional list of columns to select
            limit: Optional maximum number of rows to return

        Returns:
            pandas DataFrame with query results

        Example:
            >>> client = AthenaClient()
            >>> df = client.read_table('training_data', filters="is_fraud = true", limit=1000)
        """
        try:
            # Build query
            cols = ', '.join(columns) if columns else '*'
            query = f"SELECT {cols} FROM {self.database}.{table_name}"

            if filters:
                query += f" WHERE {filters}"

            if limit:
                query += f" LIMIT {limit}"

            logger.info(f"Reading table with query: {query}")

            # Execute query using awswrangler
            df = wr.athena.read_sql_query(
                sql=query,
                database=self.database,
                ctas_approach=False,
                workgroup=self.workgroup,
                s3_output=self.s3_output,

            )

            logger.info(f"Successfully read {len(df)} rows from {table_name}")
            return df

        except Exception as e:
            logger.error(f"Error reading table {table_name}: {e}")
            raise

    def write_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str = 'append',
        partition_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Write pandas DataFrame to Athena Iceberg table.

        Args:
            df: DataFrame to write
            table_name: Name of the table (without database prefix)
            mode: Write mode - 'append' or 'overwrite'
            partition_cols: Optional list of partition columns

        Example:
            >>> client = AthenaClient()
            >>> client.write_dataframe(df, 'inference_responses', mode='append')
        """
        try:
            full_table = f"{self.database}.{table_name}"
            logger.info(f"Writing {len(df)} rows to {full_table} (mode={mode})")

            if mode == 'overwrite':
                # Delete existing data first
                delete_query = f"DELETE FROM {full_table}"
                self.execute_query(delete_query, return_results=False)
                logger.info(f"Cleared existing data from {full_table}")

            # Use Athena SQL INSERT for reliable Iceberg writes
            # This avoids awswrangler Ray mode issues with to_iceberg()
            self._insert_dataframe_via_sql(df, full_table)

            logger.info(f"Successfully wrote data to {full_table}")

        except Exception as e:
            logger.error(f"Error writing to table {table_name}: {e}")
            raise

    def _insert_dataframe_via_sql(self, df: pd.DataFrame, full_table: str, batch_size: int = 100) -> None:
        """
        Write DataFrame to Iceberg table using Athena SQL INSERT.

        More reliable than wr.athena.to_iceberg() which has issues in Ray mode.
        Batches rows to stay within Athena query size limits.
        """
        import time as _time

        if df.empty:
            return

        columns = list(df.columns)
        total_rows = len(df)
        inserted = 0

        for start in range(0, total_rows, batch_size):
            batch = df.iloc[start:start + batch_size]
            rows_sql = []

            for _, row in batch.iterrows():
                values = []
                for col in columns:
                    val = row[col]
                    if pd.isna(val) or val is None:
                        values.append("NULL")
                    elif isinstance(val, (bool, np.bool_)):
                        values.append(str(val).lower())
                    elif isinstance(val, (int, float, np.integer, np.floating)):
                        values.append(str(val))
                    elif isinstance(val, (pd.Timestamp, datetime)):
                        values.append(f"TIMESTAMP '{val}'")
                    else:
                        s = str(val).replace("'", "''")
                        values.append(f"'{s}'")
                rows_sql.append(f"({', '.join(values)})")

            col_list = ', '.join(columns)
            query = f"INSERT INTO {full_table} ({col_list}) VALUES\n" + ",\n".join(rows_sql)

            self.execute_query(query, return_results=False)
            inserted += len(batch)
            logger.info(f"  Inserted {inserted}/{total_rows} rows")

            # Small delay between batches to avoid throttling
            if start + batch_size < total_rows:
                _time.sleep(1)

    def execute_query(
        self,
        sql: str,
        return_results: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Execute arbitrary SQL query.

        Args:
            sql: SQL query to execute
            return_results: Whether to return query results

        Returns:
            pandas DataFrame with results if return_results=True, else None

        Example:
            >>> client = AthenaClient()
            >>> result = client.execute_query("SELECT COUNT(*) as count FROM training_data")
        """
        try:
            logger.info(f"Executing query: {sql[:100]}...")

            if return_results:
                df = wr.athena.read_sql_query(
                    sql=sql,
                    database=self.database,
                    ctas_approach=False,
                    workgroup=self.workgroup,
                    s3_output=self.s3_output,
    
                )
                logger.info(f"Query returned {len(df)} rows")
                return df
            else:
                # For non-SELECT queries (CREATE, DROP, etc.)
                wr.athena.start_query_execution(
                    sql=sql,
                    database=self.database,
                    workgroup=self.workgroup,
                    s3_output=self.s3_output,
    
                    wait=True,
                )
                logger.info("Query executed successfully")
                return None

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get table metadata and statistics.

        Args:
            table_name: Name of the table (without database prefix)

        Returns:
            Dictionary with table information

        Example:
            >>> client = AthenaClient()
            >>> info = client.get_table_info('training_data')
            >>> print(f"Columns: {info['columns']}")
        """
        try:
            full_table = f"{self.database}.{table_name}"
            logger.info(f"Getting info for table: {full_table}")

            # Get table metadata
            table_metadata = wr.catalog.table(
                database=self.database,
                table=table_name,

            )

            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {full_table}"
            count_df = self.execute_query(count_query)
            row_count = int(count_df['row_count'].iloc[0]) if not count_df.empty else 0

            info = {
                'table_name': table_name,
                'database': self.database,
                'full_name': full_table,
                'columns': list(table_metadata.get('Columns', {}).keys()) if table_metadata else [],
                'row_count': row_count,
                'location': table_metadata.get('Location', 'unknown') if table_metadata else 'unknown',
                'table_type': table_metadata.get('TableType', 'unknown') if table_metadata else 'unknown',
            }

            logger.info(f"Table {table_name} has {row_count} rows and {len(info['columns'])} columns")
            return info

        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists.

        Args:
            table_name: Name of the table (without database prefix)

        Returns:
            True if table exists, False otherwise
        """
        try:
            tables = wr.catalog.tables(
                database=self.database,

            )
            exists = table_name in tables.get('Table', [])
            logger.info(f"Table {table_name} exists: {exists}")
            return exists

        except Exception as e:
            logger.warning(f"Error checking table existence for {table_name}: {e}")
            return False

    def list_tables(self) -> List[str]:
        """
        List all tables in the database.

        Returns:
            List of table names
        """
        try:
            tables_dict = wr.catalog.tables(
                database=self.database,

            )
            tables = list(tables_dict.get('Table', {}).keys())
            logger.info(f"Found {len(tables)} tables in {self.database}")
            return tables

        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            raise

    def get_partitions(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get partition information for a partitioned table.

        Args:
            table_name: Name of the table (without database prefix)

        Returns:
            List of partition dictionaries
        """
        try:
            partitions = wr.catalog.get_partitions(
                database=self.database,
                table=table_name,

            )
            logger.info(f"Table {table_name} has {len(partitions)} partitions")
            return partitions

        except Exception as e:
            logger.warning(f"Error getting partitions for {table_name}: {e}")
            return []

    def test_connection(self) -> bool:
        """
        Test Athena connection by running a simple query.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing Athena connection...")
            result = wr.athena.read_sql_query(
                sql="SELECT 1 as test",
                database=self.database,
                ctas_approach=False,
                workgroup=self.workgroup,
                s3_output=self.s3_output,

            )
            success = len(result) == 1 and result['test'].iloc[0] == 1
            if success:
                logger.info("✓ Athena connection successful")
            else:
                logger.error("✗ Athena connection test failed")
            return success

        except Exception as e:
            logger.error(f"✗ Athena connection failed: {e}")
            return False

    def get_query_stats(self, query_execution_id: str) -> Dict[str, Any]:
        """
        Get statistics for a query execution.

        Args:
            query_execution_id: Athena query execution ID

        Returns:
            Dictionary with query statistics
        """
        try:
            stats = wr.athena.get_query_execution(
                query_execution_id=query_execution_id,

            )
            return stats

        except Exception as e:
            logger.error(f"Error getting query stats: {e}")
            raise

    def export_to_s3(
        self,
        table_name: str,
        s3_path: str,
        format: str = 'parquet',
        filters: Optional[str] = None,
        partition_cols: Optional[List[str]] = None,
    ) -> str:
        """
        Export table to S3 in specified format.

        Args:
            table_name: Name of the table (without database prefix)
            s3_path: S3 destination path
            format: Output format ('parquet', 'csv', 'json')
            filters: Optional SQL WHERE clause
            partition_cols: Optional partition columns for output

        Returns:
            S3 path where data was written

        Example:
            >>> client = AthenaClient()
            >>> path = client.export_to_s3(
            ...     'training_data',
            ...     's3://my-bucket/exports/',
            ...     format='parquet'
            ... )
        """
        try:
            logger.info(f"Exporting {table_name} to {s3_path} as {format}")

            # Read data
            df = self.read_table(table_name, filters=filters)

            # Write to S3
            if format == 'parquet':
                paths = wr.s3.to_parquet(
                    df=df,
                    path=s3_path,
                    dataset=True,
                    partition_cols=partition_cols,
    
                )
            elif format == 'csv':
                paths = wr.s3.to_csv(
                    df=df,
                    path=s3_path,
                    dataset=True,
                    partition_cols=partition_cols,
    
                )
            elif format == 'json':
                paths = wr.s3.to_json(
                    df=df,
                    path=s3_path,
                    dataset=True,
                    partition_cols=partition_cols,
    
                )
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Successfully exported {len(df)} rows to {s3_path}")
            return s3_path

        except Exception as e:
            logger.error(f"Error exporting table to S3: {e}")
            raise


if __name__ == '__main__':
    """Test Athena client functionality."""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize client
    client = AthenaClient()

    # Test connection
    if not client.test_connection():
        print("Failed to connect to Athena")
        sys.exit(1)

    # List tables
    print("\nAvailable tables:")
    tables = client.list_tables()
    for table in tables:
        print(f"  - {table}")

    # Get info for each table
    print("\nTable information:")
    for table in tables:
        try:
            info = client.get_table_info(table)
            print(f"\n{table}:")
            print(f"  Rows: {info['row_count']}")
            print(f"  Columns: {len(info['columns'])}")
            print(f"  Location: {info['location']}")
        except Exception as e:
            print(f"  Error: {e}")

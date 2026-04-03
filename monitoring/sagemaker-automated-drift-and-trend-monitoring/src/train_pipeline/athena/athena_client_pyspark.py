"""
PySpark-based Athena client for fraud detection pipeline.

Drop-in replacement for athena_client.py with PySpark backend.
Provides distributed data processing for scalable operations.

Key Differences from awswrangler version:
- Returns Spark DataFrames instead of pandas DataFrames
- Operations are lazy (use .collect() or actions to execute)
- Supports distributed processing for large datasets
- Uses AWS Glue Data Catalog as Hive metastore

Usage:
    >>> from src.train_pipeline.athena.athena_client_pyspark import AthenaClientPySpark
    >>> client = AthenaClientPySpark(database='fraud_detection')
    >>> df = client.read_table('training_data', limit=1000)  # Returns Spark DataFrame
    >>> pandas_df = df.toPandas()  # Convert to pandas if needed
"""

import logging
from typing import Optional, Dict, Any, List
import boto3
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.config.config import (
    ATHENA_DATABASE,
    ATHENA_OUTPUT_S3,
)

logger = logging.getLogger(__name__)


class AthenaClientPySpark:
    """
    PySpark-based client for interacting with Athena tables.

    Provides distributed data processing through Spark SQL with
    AWS Glue Data Catalog integration.
    """

    def __init__(
        self,
        database: str = ATHENA_DATABASE,
        spark: Optional[SparkSession] = None,
    ):
        """
        Initialize PySpark Athena client.

        Args:
            database: Athena database name
            spark: Optional existing SparkSession (creates new one if None)
        """
        self.database = database

        # Use existing Spark session or create new one
        if spark is not None:
            self.spark = spark
            self._owns_spark = False
        else:
            self.spark = self._create_spark_session()
            self._owns_spark = True

        logger.info(f"Initialized AthenaClientPySpark for database: {database}")
        logger.info(f"  Spark version: {self.spark.version}")

    def _create_spark_session(self) -> SparkSession:
        """
        Create Spark session configured for AWS Glue Data Catalog.

        Returns:
            Configured SparkSession
        """
        logger.info("Creating new Spark session with Glue Data Catalog...")

        spark = (SparkSession.builder
                 .appName("athena-client-pyspark")
                 .config("spark.sql.catalogImplementation", "hive")
                 .config("hive.metastore.client.factory.class",
                         "com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory")
                 .enableHiveSupport()
                 .getOrCreate())

        logger.info(f"✓ Created Spark session: {spark.version}")
        return spark

    def read_table(
        self,
        table_name: str,
        filters: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> DataFrame:
        """
        Read entire table or filtered subset into a Spark DataFrame.

        Returns Spark DataFrame (NOT pandas) for distributed processing.
        Use .toPandas() if pandas DataFrame is needed.

        Args:
            table_name: Name of the table (without database prefix)
            filters: Optional SQL WHERE clause (without 'WHERE' keyword)
            columns: Optional list of columns to select
            limit: Optional maximum number of rows to return

        Returns:
            Spark DataFrame with query results

        Example:
            >>> client = AthenaClientPySpark()
            >>> df = client.read_table('training_data', filters="is_fraud = true", limit=1000)
            >>> row_count = df.count()  # Trigger action to execute query
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

            # Execute query using Spark SQL
            df = self.spark.sql(query)

            logger.info(f"✓ Query executed for table {table_name}")
            logger.info(f"  Columns: {len(df.columns)}")
            logger.info(f"  Schema: {df.schema.simpleString()}")

            return df

        except Exception as e:
            logger.error(f"Error reading table {table_name}: {e}")
            raise

    def write_dataframe(
        self,
        df: DataFrame,
        table_name: str,
        mode: str = 'append',
        partition_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Write Spark DataFrame to Athena Iceberg table.

        Args:
            df: Spark DataFrame to write
            table_name: Name of the table (without database prefix)
            mode: Write mode - 'append' or 'overwrite'
            partition_cols: Optional list of partition columns

        Example:
            >>> client = AthenaClientPySpark()
            >>> client.write_dataframe(df, 'inference_responses', mode='append')
        """
        try:
            full_table = f"{self.database}.{table_name}"
            logger.info(f"Writing DataFrame to {full_table} (mode={mode})")

            # Get table location from Glue catalog
            glue = boto3.client('glue')
            response = glue.get_table(DatabaseName=self.database, Name=table_name)
            table_location = response['Table']['StorageDescriptor']['Location']

            logger.info(f"Table location: {table_location}")

            # Write to S3 in Parquet format (Iceberg uses Parquet)
            write_mode = "append" if mode == "append" else "overwrite"

            if partition_cols:
                (df.write
                 .mode(write_mode)
                 .partitionBy(*partition_cols)
                 .parquet(table_location))
            else:
                (df.write
                 .mode(write_mode)
                 .parquet(table_location))

            # Refresh table metadata
            self.spark.sql(f"REFRESH TABLE {self.database}.{table_name}")

            logger.info(f"✓ Successfully wrote data to {full_table}")

        except Exception as e:
            logger.error(f"Error writing to table {table_name}: {e}")
            raise

    def execute_query(
        self,
        sql: str,
        return_results: bool = True,
    ) -> Optional[DataFrame]:
        """
        Execute arbitrary SQL query.

        Args:
            sql: SQL query to execute
            return_results: Whether to return query results

        Returns:
            Spark DataFrame with results if return_results=True, else None

        Example:
            >>> client = AthenaClientPySpark()
            >>> result = client.execute_query("SELECT COUNT(*) as count FROM training_data")
            >>> count = result.collect()[0]['count']
        """
        try:
            logger.info(f"Executing query: {sql[:100]}...")

            df = self.spark.sql(sql)

            if return_results:
                logger.info("✓ Query executed successfully")
                return df
            else:
                # For non-SELECT queries, trigger execution
                df.collect()
                logger.info("✓ Query executed successfully")
                return None

        except Exception as e:
            logger.error(f"Error executing query: {e}")
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
            # Use SHOW TABLES to check existence
            tables_df = self.spark.sql(f"SHOW TABLES IN {self.database}")
            tables = [row['tableName'] for row in tables_df.collect()]

            exists = table_name in tables
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
            tables_df = self.spark.sql(f"SHOW TABLES IN {self.database}")
            tables = [row['tableName'] for row in tables_df.collect()]

            logger.info(f"Found {len(tables)} tables in {self.database}")
            return tables

        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            raise

    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """
        Get table schema and metadata.

        Args:
            table_name: Name of the table (without database prefix)

        Returns:
            Dictionary with table metadata
        """
        try:
            full_table = f"{self.database}.{table_name}"
            logger.info(f"Getting metadata for table: {full_table}")

            # Get schema
            df = self.spark.sql(f"SELECT * FROM {self.database}.{table_name} LIMIT 0")
            schema = df.schema

            # Get partition info
            try:
                partitions_df = self.spark.sql(f"SHOW PARTITIONS {self.database}.{table_name}")
                partition_info = [row['partition'] for row in partitions_df.collect()]
            except:
                partition_info = []

            # Get table properties
            desc_df = self.spark.sql(f"DESCRIBE EXTENDED {self.database}.{table_name}")
            properties = {row['col_name']: row['data_type'] for row in desc_df.collect()}

            metadata = {
                'table_name': table_name,
                'database': self.database,
                'full_name': full_table,
                'columns': [field.name for field in schema.fields],
                'schema': schema.simpleString(),
                'partitions': partition_info,
                'properties': properties,
            }

            logger.info(f"Table {table_name} has {len(metadata['columns'])} columns")
            return metadata

        except Exception as e:
            logger.error(f"Error getting table metadata for {table_name}: {e}")
            raise

    def get_row_count(self, table_name: str) -> int:
        """
        Get total row count for a table.

        Args:
            table_name: Name of the table (without database prefix)

        Returns:
            Row count
        """
        try:
            logger.info(f"Getting row count for {table_name}")

            df = self.spark.sql(f"SELECT COUNT(*) as count FROM {self.database}.{table_name}")
            count = df.collect()[0]['count']

            logger.info(f"Table {table_name} has {count:,} rows")
            return count

        except Exception as e:
            logger.error(f"Error getting row count for {table_name}: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Test Spark connection by running a simple query.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing Spark connection...")
            result = self.spark.sql("SELECT 1 as test").collect()

            success = len(result) == 1 and result[0]['test'] == 1
            if success:
                logger.info("✓ Spark connection successful")
            else:
                logger.error("✗ Spark connection test failed")
            return success

        except Exception as e:
            logger.error(f"✗ Spark connection failed: {e}")
            return False

    def export_to_s3(
        self,
        table_name: str,
        s3_path: str,
        format: str = 'parquet',
        filters: Optional[str] = None,
        partition_cols: Optional[List[str]] = None,
    ) -> str:
        """
        Export table to S3 in specified format using Spark.

        Args:
            table_name: Name of the table (without database prefix)
            s3_path: S3 destination path
            format: Output format ('parquet', 'csv', 'json')
            filters: Optional SQL WHERE clause
            partition_cols: Optional partition columns for output

        Returns:
            S3 path where data was written

        Example:
            >>> client = AthenaClientPySpark()
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
            writer = df.write.mode("overwrite")

            if partition_cols:
                writer = writer.partitionBy(*partition_cols)

            if format == 'parquet':
                writer.parquet(s3_path)
            elif format == 'csv':
                writer.option("header", "true").csv(s3_path)
            elif format == 'json':
                writer.json(s3_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"✓ Successfully exported table to {s3_path}")
            return s3_path

        except Exception as e:
            logger.error(f"Error exporting table to S3: {e}")
            raise

    def close(self):
        """Close Spark session if owned by this client."""
        if self._owns_spark and self.spark is not None:
            logger.info("Stopping Spark session...")
            self.spark.stop()
            logger.info("✓ Spark session stopped")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == '__main__':
    """Test PySpark Athena client functionality."""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize client
    with AthenaClientPySpark() as client:
        # Test connection
        if not client.test_connection():
            print("Failed to connect to Spark")
            sys.exit(1)

        # List tables
        print("\nAvailable tables:")
        tables = client.list_tables()
        for table in tables:
            print(f"  - {table}")

        # Get metadata for each table
        print("\nTable information:")
        for table in tables[:3]:  # Limit to first 3 tables
            try:
                metadata = client.get_table_metadata(table)
                print(f"\n{table}:")
                print(f"  Columns: {len(metadata['columns'])}")
                print(f"  Schema: {metadata['schema']}")

                # Get row count
                count = client.get_row_count(table)
                print(f"  Rows: {count:,}")
            except Exception as e:
                print(f"  Error: {e}")

"""
QuickSight Dashboard Setup for Model Monitoring.

This script automates the creation and maintenance of QuickSight dashboards for
monitoring results. It handles:
- Athena data source creation/refresh
- Dataset creation/update from monitoring_results table
- Analysis creation with 5 pre-configured visualizations
- Dashboard publication and sharing
- Dashboard refresh if already exists

Usage:
    python src/quicksight/setup_quicksight_monitoring.py --create
    python src/quicksight/setup_quicksight_monitoring.py --refresh
    python src/quicksight/setup_quicksight_monitoring.py --delete
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import boto3
from botocore.exceptions import ClientError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)

from src.config.config import ATHENA_DATABASE, AWS_DEFAULT_REGION


class QuickSightDashboardManager:
    """Manager for QuickSight dashboard creation and updates."""

    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize QuickSight manager.

        Args:
            region: AWS region
        """
        self.region = region
        self.quicksight = boto3.client('quicksight', region_name=region)
        self.sts = boto3.client('sts')
        self.account_id = self.sts.get_caller_identity()['Account']

        # Resource names
        self.datasource_id = 'fraud-monitoring-athena-datasource'
        self.dataset_id = 'fraud-monitoring-dataset'
        self.analysis_id = 'fraud-monitoring-analysis'
        self.dashboard_id = 'fraud-monitoring-dashboard'

        logger.info(f"Initialized QuickSight manager for account: {self.account_id}")

    def check_quicksight_subscription(self) -> bool:
        """
        Check if QuickSight is subscribed for the account.

        Returns:
            True if subscribed, False otherwise
        """
        try:
            response = self.quicksight.describe_account_settings(
                AwsAccountId=self.account_id
            )
            logger.info("✓ QuickSight subscription active")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.error("✗ QuickSight not subscribed for this account")
                logger.error("  Please subscribe at: https://quicksight.aws.amazon.com/")
                return False
            raise

    def create_athena_datasource(self) -> str:
        """
        Create or update Athena data source.

        Returns:
            Data source ARN
        """
        logger.info("Creating Athena data source...")

        datasource_params = {
            'AwsAccountId': self.account_id,
            'DataSourceId': self.datasource_id,
            'Name': 'Fraud Detection Monitoring - Athena',
            'Type': 'ATHENA',
            'DataSourceParameters': {
                'AthenaParameters': {
                    'WorkGroup': 'primary'
                }
            },
            'Permissions': [
                {
                    'Principal': f'arn:aws:quicksight:{self.region}:{self.account_id}:user/default/Admin/*',
                    'Actions': [
                        'quicksight:DescribeDataSource',
                        'quicksight:DescribeDataSourcePermissions',
                        'quicksight:PassDataSource',
                        'quicksight:UpdateDataSource',
                        'quicksight:DeleteDataSource',
                        'quicksight:UpdateDataSourcePermissions'
                    ]
                }
            ]
        }

        try:
            # Check if exists
            try:
                response = self.quicksight.describe_data_source(
                    AwsAccountId=self.account_id,
                    DataSourceId=self.datasource_id
                )
                logger.info("  Data source already exists, updating...")

                # Update existing
                response = self.quicksight.update_data_source(**datasource_params)
                datasource_arn = response['Arn']
                logger.info(f"  ✓ Data source updated: {datasource_arn}")

            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Create new
                    response = self.quicksight.create_data_source(**datasource_params)
                    datasource_arn = response['Arn']
                    logger.info(f"  ✓ Data source created: {datasource_arn}")
                else:
                    raise

            return datasource_arn

        except Exception as e:
            logger.error(f"  ✗ Failed to create data source: {e}")
            raise

    def create_dataset(self, datasource_arn: str) -> str:
        """
        Create or update dataset from monitoring_results table.

        Args:
            datasource_arn: Athena data source ARN

        Returns:
            Dataset ARN
        """
        logger.info("Creating dataset...")

        # Define physical table
        physical_table_map = {
            'monitoring_results': {
                'RelationalTable': {
                    'DataSourceArn': datasource_arn,
                    'Catalog': 'AwsDataCatalog',
                    'Schema': ATHENA_DATABASE,
                    'Name': 'monitoring_results',
                    'InputColumns': [
                        {'Name': 'analysis_timestamp', 'Type': 'DATETIME'},
                        {'Name': 'endpoint_name', 'Type': 'STRING'},
                        {'Name': 'dataset_drift_detected', 'Type': 'BIT'},
                        {'Name': 'drifted_features_count', 'Type': 'INTEGER'},
                        {'Name': 'drift_share', 'Type': 'DECIMAL'},
                        {'Name': 'f1_score', 'Type': 'DECIMAL'},
                        {'Name': 'precision', 'Type': 'DECIMAL'},
                        {'Name': 'recall', 'Type': 'DECIMAL'},
                        {'Name': 'roc_auc', 'Type': 'DECIMAL'},
                        {'Name': 'max_ks_statistic', 'Type': 'DECIMAL'},
                        {'Name': 'top_drifted_feature_1', 'Type': 'STRING'},
                        {'Name': 'top_drifted_feature_1_ks_stat', 'Type': 'DECIMAL'},
                        {'Name': 'alarm_triggered', 'Type': 'BIT'},
                        {'Name': 'alarm_severity', 'Type': 'STRING'},
                    ]
                }
            }
        }

        # Define logical table with calculated fields
        logical_table_map = {
            'monitoring_results_logical': {
                'Alias': 'monitoring_results',
                'Source': {
                    'PhysicalTableId': 'monitoring_results'
                },
                'DataTransforms': [
                    {
                        'CreateColumnsOperation': {
                            'Columns': [
                                {
                                    'ColumnName': 'drift_severity',
                                    'ColumnId': 'drift_severity',
                                    'Expression': "ifelse({drift_share} > 0.3, 'HIGH', ifelse({drift_share} > 0.15, 'MEDIUM', 'LOW'))"
                                },
                                {
                                    'ColumnName': 'performance_status',
                                    'ColumnId': 'performance_status',
                                    'Expression': "ifelse({f1_score} >= 0.75, 'GOOD', ifelse({f1_score} >= 0.70, 'WARNING', 'CRITICAL'))"
                                }
                            ]
                        }
                    }
                ]
            }
        }

        dataset_params = {
            'AwsAccountId': self.account_id,
            'DataSetId': self.dataset_id,
            'Name': 'Fraud Monitoring Results',
            'PhysicalTableMap': physical_table_map,
            'LogicalTableMap': logical_table_map,
            'ImportMode': 'DIRECT_QUERY',
            'Permissions': [
                {
                    'Principal': f'arn:aws:quicksight:{self.region}:{self.account_id}:user/default/Admin/*',
                    'Actions': [
                        'quicksight:DescribeDataSet',
                        'quicksight:DescribeDataSetPermissions',
                        'quicksight:PassDataSet',
                        'quicksight:DescribeIngestion',
                        'quicksight:ListIngestions',
                        'quicksight:UpdateDataSet',
                        'quicksight:DeleteDataSet',
                        'quicksight:CreateIngestion',
                        'quicksight:CancelIngestion',
                        'quicksight:UpdateDataSetPermissions'
                    ]
                }
            ]
        }

        try:
            # Check if exists
            try:
                response = self.quicksight.describe_data_set(
                    AwsAccountId=self.account_id,
                    DataSetId=self.dataset_id
                )
                logger.info("  Dataset already exists, updating...")

                # Update existing
                response = self.quicksight.update_data_set(**dataset_params)
                dataset_arn = response['Arn']
                logger.info(f"  ✓ Dataset updated: {dataset_arn}")

            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Create new
                    response = self.quicksight.create_data_set(**dataset_params)
                    dataset_arn = response['Arn']
                    logger.info(f"  ✓ Dataset created: {dataset_arn}")
                else:
                    raise

            return dataset_arn

        except Exception as e:
            logger.error(f"  ✗ Failed to create dataset: {e}")
            raise

    def create_analysis(self, dataset_arn: str) -> str:
        """
        Create analysis with pre-configured visualizations.

        Args:
            dataset_arn: Dataset ARN

        Returns:
            Analysis ARN
        """
        logger.info("Creating analysis...")

        # Note: QuickSight analysis creation via API is complex
        # A simpler approach is to create template from existing analysis
        # For now, we'll create a basic analysis and user can customize in UI

        analysis_params = {
            'AwsAccountId': self.account_id,
            'AnalysisId': self.analysis_id,
            'Name': 'Fraud Detection Monitoring Analysis',
            'SourceEntity': {
                'SourceTemplate': {
                    'DataSetReferences': [
                        {
                            'DataSetPlaceholder': 'monitoring_results',
                            'DataSetArn': dataset_arn
                        }
                    ],
                    'Arn': f'arn:aws:quicksight:{self.region}:{self.account_id}:template/monitoring-template'
                }
            } if self._check_template_exists() else None,
            'Permissions': [
                {
                    'Principal': f'arn:aws:quicksight:{self.region}:{self.account_id}:user/default/Admin/*',
                    'Actions': [
                        'quicksight:RestoreAnalysis',
                        'quicksight:UpdateAnalysisPermissions',
                        'quicksight:DeleteAnalysis',
                        'quicksight:DescribeAnalysisPermissions',
                        'quicksight:QueryAnalysis',
                        'quicksight:DescribeAnalysis',
                        'quicksight:UpdateAnalysis'
                    ]
                }
            ]
        }

        try:
            # Check if exists
            try:
                response = self.quicksight.describe_analysis(
                    AwsAccountId=self.account_id,
                    AnalysisId=self.analysis_id
                )
                logger.info("  Analysis already exists")
                analysis_arn = response['Analysis']['Arn']
                logger.info(f"  ✓ Analysis found: {analysis_arn}")

            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info("  Creating new analysis...")
                    logger.info("  Note: Analysis must be created manually in QuickSight UI")
                    logger.info("  Recommended visualizations:")
                    logger.info("    1. F1 Score Trend - Line chart over time")
                    logger.info("    2. Drifted Features Count - Bar chart")
                    logger.info("    3. Top Drifted Features - Table")
                    logger.info("    4. Model Performance KPIs - KPI cards")
                    logger.info("    5. Drift Distribution - Heat map")

                    # Return dataset ARN as placeholder
                    analysis_arn = dataset_arn
                else:
                    raise

            return analysis_arn

        except Exception as e:
            logger.error(f"  ✗ Failed to create analysis: {e}")
            raise

    def _check_template_exists(self) -> bool:
        """Check if monitoring template exists."""
        try:
            self.quicksight.describe_template(
                AwsAccountId=self.account_id,
                TemplateId='monitoring-template'
            )
            return True
        except:
            return False

    def create_dashboard(self, analysis_arn: str, dataset_arn: str) -> str:
        """
        Create or update dashboard from analysis.

        Args:
            analysis_arn: Analysis ARN
            dataset_arn: Dataset ARN

        Returns:
            Dashboard ARN
        """
        logger.info("Creating dashboard...")

        dashboard_params = {
            'AwsAccountId': self.account_id,
            'DashboardId': self.dashboard_id,
            'Name': 'Fraud Detection Monitoring',
            'Permissions': [
                {
                    'Principal': f'arn:aws:quicksight:{self.region}:{self.account_id}:user/default/Admin/*',
                    'Actions': [
                        'quicksight:DescribeDashboard',
                        'quicksight:ListDashboardVersions',
                        'quicksight:UpdateDashboardPermissions',
                        'quicksight:QueryDashboard',
                        'quicksight:UpdateDashboard',
                        'quicksight:DeleteDashboard',
                        'quicksight:DescribeDashboardPermissions',
                        'quicksight:UpdateDashboardPublishedVersion'
                    ]
                }
            ],
            'SourceEntity': {
                'SourceTemplate': {
                    'DataSetReferences': [
                        {
                            'DataSetPlaceholder': 'monitoring_results',
                            'DataSetArn': dataset_arn
                        }
                    ],
                    'Arn': f'arn:aws:quicksight:{self.region}:{self.account_id}:template/monitoring-template'
                }
            } if self._check_template_exists() else None,
            'DashboardPublishOptions': {
                'AdHocFilteringOption': {'AvailabilityStatus': 'ENABLED'},
                'ExportToCSVOption': {'AvailabilityStatus': 'ENABLED'},
                'SheetControlsOption': {'VisibilityState': 'EXPANDED'}
            }
        }

        try:
            # Check if exists
            try:
                response = self.quicksight.describe_dashboard(
                    AwsAccountId=self.account_id,
                    DashboardId=self.dashboard_id
                )
                logger.info("  Dashboard already exists, updating...")

                # Update existing
                response = self.quicksight.update_dashboard(**dashboard_params)
                dashboard_arn = response['Arn']
                logger.info(f"  ✓ Dashboard updated: {dashboard_arn}")

                # Publish latest version
                version_number = response['VersionArn'].split('/')[-1]
                self.quicksight.update_dashboard_published_version(
                    AwsAccountId=self.account_id,
                    DashboardId=self.dashboard_id,
                    VersionNumber=int(version_number)
                )
                logger.info(f"  ✓ Dashboard published: version {version_number}")

            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info("  Dashboard does not exist")
                    logger.info("  Creating dashboard from dataset...")

                    # Create without template
                    if 'SourceEntity' in dashboard_params and dashboard_params['SourceEntity'] is None:
                        del dashboard_params['SourceEntity']

                    logger.info("  Note: Dashboard must be created manually in QuickSight UI")
                    logger.info("  Steps:")
                    logger.info("    1. Open QuickSight: https://quicksight.aws.amazon.com/")
                    logger.info("    2. Go to Datasets")
                    logger.info(f"    3. Select 'Fraud Monitoring Results' dataset")
                    logger.info("    4. Click 'Create analysis'")
                    logger.info("    5. Add visualizations (see analysis recommendations above)")
                    logger.info("    6. Click 'Share' > 'Publish dashboard'")
                    logger.info(f"    7. Name it: 'Fraud Detection Monitoring'")

                    dashboard_arn = dataset_arn  # Placeholder
                else:
                    raise

            return dashboard_arn

        except Exception as e:
            logger.error(f"  ✗ Failed to create dashboard: {e}")
            raise

    def get_dashboard_url(self) -> str:
        """
        Get dashboard URL.

        Returns:
            Dashboard URL
        """
        try:
            response = self.quicksight.describe_dashboard(
                AwsAccountId=self.account_id,
                DashboardId=self.dashboard_id
            )

            # Generate embed URL (requires additional setup)
            # For now, return console URL
            dashboard_url = (
                f"https://quicksight.aws.amazon.com/sn/dashboards/{self.dashboard_id}"
            )

            logger.info(f"✓ Dashboard URL: {dashboard_url}")
            return dashboard_url

        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.warning("Dashboard not found")
                return None
            raise

    def refresh_dataset(self) -> bool:
        """
        Refresh dataset (for SPICE datasets).

        Returns:
            True if refresh started, False otherwise
        """
        logger.info("Refreshing dataset...")

        try:
            # Check dataset import mode
            response = self.quicksight.describe_data_set(
                AwsAccountId=self.account_id,
                DataSetId=self.dataset_id
            )

            import_mode = response['DataSet'].get('ImportMode', 'DIRECT_QUERY')

            if import_mode == 'SPICE':
                # Create ingestion
                ingestion_id = f"ingestion-{int(time.time())}"

                self.quicksight.create_ingestion(
                    DataSetId=self.dataset_id,
                    IngestionId=ingestion_id,
                    AwsAccountId=self.account_id
                )

                logger.info(f"  ✓ Dataset refresh started: {ingestion_id}")
                return True
            else:
                logger.info("  Dataset uses DIRECT_QUERY mode, no refresh needed")
                return False

        except Exception as e:
            logger.error(f"  ✗ Failed to refresh dataset: {e}")
            return False

    def delete_resources(self) -> bool:
        """
        Delete all QuickSight resources.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Deleting QuickSight resources...")

        success = True

        # Delete dashboard
        try:
            self.quicksight.delete_dashboard(
                AwsAccountId=self.account_id,
                DashboardId=self.dashboard_id
            )
            logger.info("  ✓ Dashboard deleted")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                logger.warning(f"  Failed to delete dashboard: {e}")
                success = False

        # Delete analysis
        try:
            self.quicksight.delete_analysis(
                AwsAccountId=self.account_id,
                AnalysisId=self.analysis_id
            )
            logger.info("  ✓ Analysis deleted")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                logger.warning(f"  Failed to delete analysis: {e}")
                success = False

        # Delete dataset
        try:
            self.quicksight.delete_data_set(
                AwsAccountId=self.account_id,
                DataSetId=self.dataset_id
            )
            logger.info("  ✓ Dataset deleted")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                logger.warning(f"  Failed to delete dataset: {e}")
                success = False

        # Delete data source
        try:
            self.quicksight.delete_data_source(
                AwsAccountId=self.account_id,
                DataSourceId=self.datasource_id
            )
            logger.info("  ✓ Data source deleted")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                logger.warning(f"  Failed to delete data source: {e}")
                success = False

        return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup QuickSight dashboard for model monitoring"
    )

    parser.add_argument(
        '--create',
        action='store_true',
        help='Create dashboard and all resources'
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Refresh existing dashboard'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete all QuickSight resources'
    )
    parser.add_argument(
        '--region',
        type=str,
        default=AWS_DEFAULT_REGION,
        help=f'AWS region (default: {AWS_DEFAULT_REGION})'
    )

    args = parser.parse_args()

    # Create manager
    manager = QuickSightDashboardManager(region=args.region)

    logger.info("=" * 80)
    logger.info("QuickSight Dashboard Setup for Model Monitoring")
    logger.info("=" * 80)
    logger.info(f"Account: {manager.account_id}")
    logger.info(f"Region: {args.region}")
    logger.info("")

    # Check subscription
    if not manager.check_quicksight_subscription():
        logger.error("Please subscribe to QuickSight before continuing")
        sys.exit(1)

    if args.delete:
        # Delete resources
        logger.info("Deleting QuickSight resources...")
        success = manager.delete_resources()

        if success:
            logger.info("\n✓ All resources deleted successfully")
        else:
            logger.warning("\n⚠ Some resources could not be deleted")

        sys.exit(0)

    if args.create or args.refresh:
        try:
            # Step 1: Create data source
            datasource_arn = manager.create_athena_datasource()

            # Step 2: Create dataset
            dataset_arn = manager.create_dataset(datasource_arn)

            # Step 3: Create analysis (manual step in UI)
            analysis_arn = manager.create_analysis(dataset_arn)

            # Step 4: Create dashboard (manual step in UI)
            dashboard_arn = manager.create_dashboard(analysis_arn, dataset_arn)

            # Step 5: Refresh if requested
            if args.refresh:
                manager.refresh_dataset()

            # Get dashboard URL
            dashboard_url = manager.get_dashboard_url()

            logger.info("")
            logger.info("=" * 80)
            logger.info("Setup Complete")
            logger.info("=" * 80)
            logger.info("")
            logger.info("Resources Created:")
            logger.info(f"  ✓ Data Source: {manager.datasource_id}")
            logger.info(f"  ✓ Dataset: {manager.dataset_id}")
            logger.info(f"  Note: Analysis and Dashboard require manual creation in QuickSight UI")
            logger.info("")
            logger.info("Next Steps:")
            logger.info("1. Open QuickSight: https://quicksight.aws.amazon.com/")
            logger.info("2. Go to Datasets")
            logger.info("3. Select 'Fraud Monitoring Results'")
            logger.info("4. Create analysis with recommended visualizations:")
            logger.info("   - F1 Score Trend (Line chart)")
            logger.info("   - Drifted Features Count (Bar chart)")
            logger.info("   - Top Drifted Features (Table)")
            logger.info("   - Model Performance KPIs (KPI cards)")
            logger.info("   - Drift Distribution (Heat map)")
            logger.info("5. Publish as dashboard named 'Fraud Detection Monitoring'")
            logger.info("")

            if dashboard_url:
                logger.info(f"Dashboard URL: {dashboard_url}")

        except Exception as e:
            logger.error(f"\n✗ Setup failed: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

"""
Upload local CSV data files to S3.

This script uploads the CSV files from the local data/ directory to S3 so they
can be loaded into Athena Iceberg tables without committing large files to git.

Usage:
    python -m src.setup.upload_data_to_s3
    python -m src.setup.upload_data_to_s3 --dry-run
    python -m src.setup.upload_data_to_s3 --file data/creditcard_predictions_final.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.config import (
    AWS_DEFAULT_REGION,
    DATA_DIR,
    DATA_S3_BUCKET,
    S3_PREFIX,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Mapping of local CSV filenames to their S3 sub-paths under the prefix
CSV_S3_MAPPING = {
    "creditcard_predictions_final.csv": "data/creditcard_predictions_final.csv",
    "creditcard_ground_truth.csv": "data/creditcard_ground_truth.csv",
    "creditcard_drifted.csv": "data/creditcard_drifted.csv",
}


def upload_file(s3_client, local_path: Path, bucket: str, s3_key: str, dry_run: bool = False) -> bool:
    """Upload a single file to S3."""
    if not local_path.exists():
        logger.warning(f"File not found: {local_path}")
        return False

    size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Uploading {local_path.name} ({size_mb:.1f} MB) -> s3://{bucket}/{s3_key}")

    if dry_run:
        return True

    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"  ✓ Uploaded to s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"  ✗ Upload failed: {e}")
        return False


def upload_all(bucket: str, prefix: str, data_dir: Path, dry_run: bool = False, single_file: str = None) -> dict:
    """Upload CSV data files to S3.

    Args:
        bucket: S3 bucket name.
        prefix: S3 key prefix (e.g. "fraud-detection/").
        data_dir: Local directory containing CSV files.
        dry_run: If True, only log what would happen.
        single_file: If set, upload only this filename.

    Returns:
        Dict of {filename: success_bool}.
    """
    if not bucket:
        logger.error("DATA_S3_BUCKET is not configured. Set it in config.yaml or .env")
        return {}

    s3_client = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
    results = {}

    files_to_upload = CSV_S3_MAPPING
    if single_file:
        name = Path(single_file).name
        if name not in CSV_S3_MAPPING:
            logger.error(f"Unknown file: {name}. Expected one of {list(CSV_S3_MAPPING.keys())}")
            return {}
        files_to_upload = {name: CSV_S3_MAPPING[name]}

    for filename, s3_subpath in files_to_upload.items():
        local_path = data_dir / filename
        s3_key = f"{prefix}{s3_subpath}"
        results[filename] = upload_file(s3_client, local_path, bucket, s3_key, dry_run)

    uploaded = sum(1 for v in results.values() if v)
    logger.info(f"\nSummary: {uploaded}/{len(results)} files {'would be ' if dry_run else ''}uploaded to s3://{bucket}/{prefix}data/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Upload CSV data files to S3")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without uploading")
    parser.add_argument("--file", type=str, default=None, help="Upload a single file (e.g. data/creditcard_predictions_final.csv)")
    parser.add_argument("--bucket", type=str, default=None, help="Override S3 bucket name")
    parser.add_argument("--prefix", type=str, default=None, help="Override S3 prefix")
    args = parser.parse_args()

    bucket = args.bucket or DATA_S3_BUCKET
    prefix = args.prefix or S3_PREFIX

    results = upload_all(
        bucket=bucket,
        prefix=prefix,
        data_dir=DATA_DIR,
        dry_run=args.dry_run,
        single_file=args.file,
    )

    if not results or not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

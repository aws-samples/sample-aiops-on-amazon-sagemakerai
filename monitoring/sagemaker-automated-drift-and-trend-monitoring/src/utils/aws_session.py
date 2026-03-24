"""AWS Session Management Utilities

This module provides robust boto3 session creation with graceful error handling
for missing AWS profiles, region detection, and credential management.
"""

import os
import logging
from typing import Optional

import boto3
from botocore.exceptions import ProfileNotFound, NoCredentialsError

logger = logging.getLogger(__name__)


def get_aws_region(session: Optional[boto3.Session] = None) -> str:
    """
    Detect AWS region from multiple sources.

    Region detection priority:
    1. AWS_REGION environment variable
    2. AWS_DEFAULT_REGION environment variable
    3. Session's configured region
    4. boto3's default session region
    5. Fall back to 'us-east-1'

    Args:
        session: Optional boto3 session to check for region

    Returns:
        str: Detected AWS region
    """
    # Try environment variables first
    region = os.getenv('AWS_REGION')
    if region:
        logger.debug(f"Detected region from AWS_REGION: {region}")
        return region

    region = os.getenv('AWS_DEFAULT_REGION')
    if region:
        logger.debug(f"Detected region from AWS_DEFAULT_REGION: {region}")
        return region

    # Try session region
    if session and session.region_name:
        logger.debug(f"Using session region: {session.region_name}")
        return session.region_name

    # Try boto3 default session
    try:
        default_session = boto3.Session()
        if default_session.region_name:
            logger.debug(f"Using boto3 default session region: {default_session.region_name}")
            return default_session.region_name
    except Exception as e:
        logger.debug(f"Could not get region from default session: {e}")

    # Fall back to us-east-1
    logger.warning("No region configured, falling back to us-east-1")
    return 'us-east-1'


def create_boto3_session(
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None
) -> boto3.Session:
    """
    Create boto3 session with graceful error handling.

    This function handles common AWS configuration issues:
    - Missing or invalid AWS profiles
    - Missing AWS credentials
    - Region detection from multiple sources

    Region detection priority (if region_name not specified):
    1. AWS_REGION environment variable
    2. AWS_DEFAULT_REGION environment variable
    3. AWS config file default region
    4. Fall back to 'us-east-1'

    Args:
        region_name: Explicit region name (highest priority)
        profile_name: AWS profile name to use (optional)

    Returns:
        boto3.Session: Configured boto3 session

    Raises:
        NoCredentialsError: If no AWS credentials are available

    Examples:
        >>> # Use explicit region
        >>> session = create_boto3_session(region_name='us-west-2')

        >>> # Auto-detect region from environment
        >>> session = create_boto3_session()

        >>> # Use specific profile
        >>> session = create_boto3_session(profile_name='production')
    """
    session = None
    detected_region = region_name
    env_profile = os.getenv('AWS_PROFILE')
    saved_profile = None

    try:
        # Check if AWS_PROFILE env var points to invalid profile
        # We need to unset it early to avoid issues throughout the function
        if env_profile and not profile_name:
            # Test if profile exists
            try:
                test_session = boto3.Session(profile_name=env_profile)
                # Access a property to trigger profile loading
                _ = test_session._session.get_scoped_config()
                # Profile is valid, use it
                profile_name = env_profile
            except (ProfileNotFound, Exception):
                logger.warning(
                    f"AWS_PROFILE environment variable is set to '{env_profile}' but profile not found. "
                    f"Temporarily unsetting AWS_PROFILE to use default credentials."
                )
                # Temporarily remove AWS_PROFILE
                saved_profile = env_profile
                del os.environ['AWS_PROFILE']

        # Try to use specified profile if provided
        if profile_name:
            try:
                logger.debug(f"Attempting to create session with profile: {profile_name}")
                session = boto3.Session(
                    profile_name=profile_name,
                    region_name=region_name
                )
                logger.info(f"Successfully created session with profile: {profile_name}")
            except ProfileNotFound:
                logger.warning(
                    f"AWS profile '{profile_name}' not found in ~/.aws/config or ~/.aws/credentials. "
                    f"Falling back to default credentials."
                )
                session = None

        # If profile failed or not specified, create session without profile
        if session is None:
            session = boto3.Session(region_name=region_name)
            logger.debug("Created session with default credentials")

        # Auto-detect region if not explicitly specified
        if detected_region is None:
            detected_region = get_aws_region(session)

            # If session doesn't have the detected region, recreate with correct region
            if session.region_name != detected_region:
                logger.debug(f"Recreating session with detected region: {detected_region}")
                if profile_name:
                    try:
                        session = boto3.Session(
                            profile_name=profile_name,
                            region_name=detected_region
                        )
                    except ProfileNotFound:
                        session = boto3.Session(region_name=detected_region)
                else:
                    session = boto3.Session(region_name=detected_region)

        # Verify credentials are available
        try:
            # Test credentials by getting caller identity
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            logger.info(f"AWS session created successfully for account: {identity['Account']}")
            logger.info(f"Using AWS region: {session.region_name}")
        except NoCredentialsError:
            logger.error(
                "No AWS credentials found. Please configure credentials using one of:\n"
                "1. AWS CLI: aws configure\n"
                "2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
                "3. IAM role (if running on AWS)\n"
                "4. AWS credentials file: ~/.aws/credentials"
            )
            raise
        except Exception as e:
            logger.warning(f"Could not verify AWS credentials: {e}")

        return session

    finally:
        # Always restore AWS_PROFILE if we removed it
        if saved_profile:
            os.environ['AWS_PROFILE'] = saved_profile


def get_session_info(session: boto3.Session) -> dict:
    """
    Get information about a boto3 session.

    Args:
        session: boto3 session to inspect

    Returns:
        dict: Session information including region, profile, and account
    """
    info = {
        'region': session.region_name,
        'profile': session.profile_name,
        'account': None,
        'user_arn': None
    }

    try:
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        info['account'] = identity.get('Account')
        info['user_arn'] = identity.get('Arn')
    except Exception as e:
        logger.debug(f"Could not get session identity: {e}")

    return info

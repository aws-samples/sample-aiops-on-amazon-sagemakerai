"""
Data preprocessing script for SageMaker Pipeline.

This script runs as a ProcessingStep and:
- Reads data from Athena training_data table
- Validates data quality
- Splits into train/test sets
- Saves to S3 for training step
- Logs statistics to output
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Visualization libraries - try to install if not available
# Using XGBoost container which supports modern dependencies
VISUALIZATION_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    print("✓ Visualization libraries available")
except ImportError:
    print("⚠ Matplotlib not found, attempting to install...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                              'matplotlib>=3.5.0', 'seaborn>=0.12.0'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        VISUALIZATION_AVAILABLE = True
        print("✓ Visualization libraries installed successfully")
    except Exception as e:
        print(f"⚠ Could not install visualization libraries: {e}")
        print("  Preprocessing will continue without visualizations")
        VISUALIZATION_AVAILABLE = False

# MLflow for logging visualizations (optional)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Note: Visualization functions now return figure objects for MLflow logging
# MLflow's log_figure() API handles serialization properly


def read_data_from_athena(
    athena_table: str,
    athena_filter: str = None,
    limit: int = None
) -> pd.DataFrame:
    """
    Read training data from Athena using boto3.

    Args:
        athena_table: Athena table name
        athena_filter: Optional SQL WHERE clause
        limit: Optional row limit

    Returns:
        DataFrame with training data
    """
    import boto3
    import time
    import io

    athena_client = boto3.client('athena')
    s3_client = boto3.client('s3')
    
    database = os.getenv('ATHENA_DATABASE', 'fraud_detection')
    # Get output location from environment or construct default
    output_location = os.getenv('ATHENA_OUTPUT_S3')
    if not output_location:
        # Try to get bucket from environment
        bucket = os.getenv('DATA_S3_BUCKET', 'fraud-detection-data-lake')
        output_location = f's3://{bucket}/athena-query-results/'

    logger.info(f"Reading data from Athena table: {athena_table}")
    logger.info(f"Database: {database}")
    logger.info(f"Output location: {output_location}")

    # Build query
    query = f"SELECT * FROM {athena_table}"
    if athena_filter:
        query += f" WHERE {athena_filter}"
    if limit:
        query += f" LIMIT {limit}"

    logger.info(f"Query: {query}")

    # Start query execution
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': output_location}
    )
    query_execution_id = response['QueryExecutionId']
    logger.info(f"Query execution ID: {query_execution_id}")

    # Wait for query to complete
    max_wait_time = 600  # 10 minutes
    poll_interval = 5
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = response['QueryExecution']['Status']['State']
        
        if state == 'SUCCEEDED':
            logger.info("Query succeeded")
            break
        elif state in ['FAILED', 'CANCELLED']:
            reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
            raise Exception(f"Query {state}: {reason}")
        
        logger.info(f"Query state: {state}, waiting...")
        time.sleep(poll_interval)
        elapsed_time += poll_interval
    else:
        raise Exception(f"Query timed out after {max_wait_time} seconds")

    # Get results location
    output_uri = response['QueryExecution']['ResultConfiguration']['OutputLocation']
    logger.info(f"Results at: {output_uri}")
    
    # Parse S3 URI
    # Format: s3://bucket/path/to/file.csv
    output_uri = output_uri.replace('s3://', '')
    bucket_name = output_uri.split('/')[0]
    key = '/'.join(output_uri.split('/')[1:])
    
    # Download results from S3
    logger.info(f"Downloading results from s3://{bucket_name}/{key}")
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    csv_content = response['Body'].read().decode('utf-8')
    
    # Parse CSV into DataFrame
    df = pd.read_csv(io.StringIO(csv_content))
    
    logger.info(f"✓ Loaded {len(df):,} rows from Athena")
    return df


def validate_data_quality(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Validate data quality and return statistics.

    Args:
        df: Input DataFrame
        target_column: Name of target column

    Returns:
        Dictionary with validation results and statistics
    """
    logger.info("Validating data quality...")

    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'duplicate_rows': int(df.duplicated().sum()),
        'class_distribution': {},
        'feature_statistics': {},
        'validation_passed': True,
        'validation_errors': [],
        'validation_warnings': []
    }

    # Check for missing values
    missing = df.isnull().sum()
    stats['missing_values'] = {
        col: int(count) for col, count in missing.items() if count > 0
    }

    # Check target column
    if target_column not in df.columns:
        stats['validation_passed'] = False
        stats['validation_errors'].append(f"Target column '{target_column}' not found")
        return stats

    # Class distribution
    class_counts = df[target_column].value_counts().to_dict()
    stats['class_distribution'] = {str(k): int(v) for k, v in class_counts.items()}

    # Calculate class imbalance ratio
    if len(class_counts) == 2:
        majority_class = max(class_counts.values())
        minority_class = min(class_counts.values())
        stats['class_imbalance_ratio'] = float(majority_class / minority_class)

        # Warn if severely imbalanced
        if stats['class_imbalance_ratio'] > 100:
            stats['validation_warnings'].append(
                f"Severe class imbalance detected: {stats['class_imbalance_ratio']:.1f}:1"
            )

    # Feature statistics (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != target_column:
            stats['feature_statistics'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'missing_pct': float(df[col].isnull().mean() * 100)
            }

    # Validation checks
    min_samples = int(os.getenv('MIN_TRAINING_SAMPLES', '1000'))
    if len(df) < min_samples:
        stats['validation_errors'].append(
            f"Insufficient samples: {len(df)} < {min_samples}"
        )
        stats['validation_passed'] = False

    max_missing_pct = float(os.getenv('MAX_MISSING_VALUES_PCT', '0.05'))
    for col, missing_count in stats['missing_values'].items():
        missing_pct = missing_count / len(df)
        if missing_pct > max_missing_pct:
            stats['validation_warnings'].append(
                f"Column '{col}' has {missing_pct:.1%} missing values"
            )

    # Log results
    logger.info(f"Data validation: {'PASSED' if stats['validation_passed'] else 'FAILED'}")
    logger.info(f"  Total rows: {stats['total_rows']:,}")
    logger.info(f"  Total columns: {stats['total_columns']}")
    logger.info(f"  Class distribution: {stats['class_distribution']}")

    if stats['validation_errors']:
        logger.error("Validation errors:")
        for error in stats['validation_errors']:
            logger.error(f"  - {error}")

    if stats['validation_warnings']:
        logger.warning("Validation warnings:")
        for warning in stats['validation_warnings']:
            logger.warning(f"  - {warning}")

    return stats


def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert boolean/string columns to numeric (0/1).
    Also encodes categorical columns with few categories.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with boolean columns converted to 0/1 and categoricals encoded
    """
    for col in df.columns:
        # Check if column contains boolean-like values
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            unique_vals = df[col].dropna().unique()

            # Check for boolean strings
            if len(unique_vals) <= 2 and set(map(str, unique_vals)).issubset({'True', 'False', 'true', 'false', '0', '1', 'yes', 'no', 'Yes', 'No'}):
                logger.info(f"Converting boolean column '{col}' to 0/1")
                # Convert to boolean first, then to int
                df[col] = df[col].astype(str).str.lower().map({
                    'true': 1, 'false': 0,
                    '1': 1, '0': 0,
                    'yes': 1, 'no': 0
                }).fillna(0).astype(int)

            # Handle low-cardinality categorical columns (like gender)
            elif len(unique_vals) <= 10:
                logger.info(f"Label encoding categorical column '{col}' with {len(unique_vals)} categories: {list(unique_vals)}")
                # Simple label encoding: map categories to integers
                category_mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                df[col] = df[col].map(category_mapping).fillna(-1).astype(int)

    return df


def create_pca_visualization(
    df: pd.DataFrame,
    target_column: str,
    n_components: int = 2
) -> plt.Figure:
    """
    Create PCA visualization showing principal component analysis.

    Similar to the MLflow time series visualization tutorial, this shows
    how the data is distributed in the principal component space.

    Args:
        df: DataFrame to analyze
        target_column: Name of target column
        n_components: Number of principal components to compute

    Returns:
        matplotlib figure object
    """
    # Select numeric features only (excluding target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_column][:20]  # Limit to top 20 features

    if len(feature_cols) < 2:
        logger.warning("Not enough features for PCA analysis")
        return None

    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df[target_column]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=min(n_components, len(feature_cols)))
    X_pca = pca.fit_transform(X_scaled)

    # Create visualization
    fig = plt.figure(figsize=(16, 6))

    # 1. PCA scatter plot (2D projection)
    ax1 = fig.add_subplot(1, 3, 1)
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn_r',
                         alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                   fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
                   fontsize=11, fontweight='bold')
    ax1.set_title('Principal Component Analysis (2D Projection)',
                  fontsize=12, fontweight='bold', pad=15)
    ax1.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Target Value', fontsize=10)

    # 2. Explained variance plot
    ax2 = fig.add_subplot(1, 3, 2)
    n_components_full = min(10, len(feature_cols))
    pca_full = PCA(n_components=n_components_full)
    pca_full.fit(X_scaled)

    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    ax2.plot(range(1, n_components_full + 1), cumsum, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='80% Variance')
    ax2.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95% Variance')
    ax2.set_xlabel('Number of Components', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=11, fontweight='bold')
    ax2.set_title('PCA Explained Variance', fontsize=12, fontweight='bold', pad=15)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_xticks(range(1, n_components_full + 1))

    # 3. Feature contribution heatmap (top 10 features in PC1 and PC2)
    ax3 = fig.add_subplot(1, 3, 3)
    components_df = pd.DataFrame(
        pca.components_[:2],
        columns=feature_cols,
        index=['PC1', 'PC2']
    )

    # Get top 10 features by absolute contribution to PC1
    top_features = components_df.loc['PC1'].abs().nlargest(10).index.tolist()
    components_subset = components_df[top_features]

    sns.heatmap(components_subset, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax3, cbar_kws={'label': 'Component Loading'},
                linewidths=0.5, linecolor='black')
    ax3.set_title('Top 10 Feature Loadings in PC1 & PC2',
                  fontsize=12, fontweight='bold', pad=15)
    ax3.set_ylabel('Principal Component', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Features', fontsize=11, fontweight='bold')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    plt.suptitle('Principal Component Analysis - Feature Space Exploration',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def create_enhanced_correlation_visualization(
    df: pd.DataFrame,
    target_column: str
) -> plt.Figure:
    """
    Create enhanced correlation visualization similar to MLflow tutorial.

    Shows correlation with target and between features in a comprehensive view.

    Args:
        df: DataFrame to analyze
        target_column: Name of target column

    Returns:
        matplotlib figure object
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_column not in numeric_cols or len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for correlation analysis")
        return None

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 6))

    # 1. Correlation with target (bar plot)
    ax1 = fig.add_subplot(1, 3, 1)
    correlations = df[numeric_cols].corr()[target_column].drop(target_column).abs().sort_values(ascending=False)
    top_15_corr = correlations.head(15)

    colors = ['#e74c3c' if x > 0.5 else '#3498db' if x > 0.3 else '#95a5a6' for x in top_15_corr.values]
    top_15_corr.plot(kind='barh', ax=ax1, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Absolute Correlation with Target', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Features', fontsize=11, fontweight='bold')
    ax1.set_title('Top 15 Features by Correlation with Target', fontsize=12, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Add value labels on bars
    for i, v in enumerate(top_15_corr.values):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    # 2. Full correlation matrix (top 12 features)
    ax2 = fig.add_subplot(1, 3, 2)
    top_features = correlations.head(12).index.tolist()
    top_features_with_target = [target_column] + top_features
    corr_matrix = df[top_features_with_target].corr()

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax2,
                vmin=-1, vmax=1)
    ax2.set_title('Correlation Matrix (Top 12 Features)', fontsize=12, fontweight='bold', pad=15)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=9)

    # 3. Feature-to-feature correlation network (excluding target)
    ax3 = fig.add_subplot(1, 3, 3)
    feature_corr = df[top_features[:10]].corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(feature_corr, dtype=bool), k=1)
    sns.heatmap(feature_corr, mask=mask, annot=True, fmt='.2f', cmap='viridis',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax3,
                vmin=-1, vmax=1)
    ax3.set_title('Feature-to-Feature Correlation\n(Lower Triangle)',
                  fontsize=12, fontweight='bold', pad=15)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=9)

    plt.suptitle('Feature Correlation Analysis - Comprehensive View',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def create_preprocessing_visualizations(
    df: pd.DataFrame,
    target_column: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Create visualizations for preprocessing stage.

    Args:
        df: DataFrame to visualize
        target_column: Name of target column
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to matplotlib figure objects
    """
    # Skip visualizations if libraries not available
    if not VISUALIZATION_AVAILABLE:
        logger.info("⚠ Skipping preprocessing visualizations (libraries not available)")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {}  # Store figure objects, not paths

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    try:
        # 1. Class Distribution
        logger.info("Creating class distribution plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        class_counts = df[target_column].value_counts()
        ax.bar(class_counts.index, class_counts.values, color=['#2ecc71', '#e74c3c'])
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Class Distribution\n(Fraud: {class_counts.get(1, 0):,} | Non-Fraud: {class_counts.get(0, 0):,})',
                     fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-Fraud', 'Fraud'])
        for i, v in enumerate(class_counts.values):
            ax.text(i, v + max(class_counts.values) * 0.01, f'{v:,}',
                   ha='center', fontweight='bold')

        # Add imbalance ratio
        imbalance_ratio = class_counts.values[0] / class_counts.values[1] if len(class_counts) > 1 else 0
        ax.text(0.5, 0.95, f'Imbalance Ratio: {imbalance_ratio:.1f}:1',
                transform=ax.transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.close(fig)
        figures['class_distribution'] = fig
        logger.info("✓ Created class distribution plot")

        # 2. Feature Distribution (top 9 features)
        logger.info("Creating feature distribution plots...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target_column][:9]

        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.ravel()

            for idx, col in enumerate(numeric_cols):
                if idx < 9:
                    ax = axes[idx]
                    df[col].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
                    ax.set_title(f'{col}', fontsize=10, fontweight='bold')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    ax.grid(alpha=0.3)

            # Hide unused subplots
            for idx in range(len(numeric_cols), 9):
                axes[idx].axis('off')

            plt.suptitle('Feature Distributions (Top 9 Features)', fontsize=14, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.close(fig)
            figures['feature_distributions'] = fig
            logger.info("✓ Created feature distributions")

        # 3. Correlation Matrix (top 15 features)
        logger.info("Creating correlation matrix...")
        numeric_cols_for_corr = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols_for_corr and len(numeric_cols_for_corr) > 1:
            # Select top features correlated with target
            correlations = df[numeric_cols_for_corr].corr()[target_column].abs().sort_values(ascending=False)
            top_features = correlations.head(16).index.tolist()  # Include target + 15 features

            fig, ax = plt.subplots(figsize=(12, 10))
            corr_matrix = df[top_features].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Feature Correlation Matrix (Top 15 Features)', fontsize=14, fontweight='bold', pad=20)

            plt.tight_layout()
            plt.close(fig)
            figures['correlation_matrix'] = fig
            logger.info("✓ Created correlation matrix")

        # 4. Feature Statistics Summary
        logger.info("Creating feature statistics summary...")
        numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_all = [col for col in numeric_cols_all if col != target_column][:10]

        if len(numeric_cols_all) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))

            stats_data = []
            for col in numeric_cols_all:
                stats_data.append([
                    col,
                    f"{df[col].mean():.2f}",
                    f"{df[col].std():.2f}",
                    f"{df[col].min():.2f}",
                    f"{df[col].max():.2f}",
                    f"{df[col].isna().sum()}"
                ])

            table = ax.table(cellText=stats_data,
                           colLabels=['Feature', 'Mean', 'Std Dev', 'Min', 'Max', 'Missing'],
                           cellLoc='center', loc='center',
                           colWidths=[0.3, 0.14, 0.14, 0.14, 0.14, 0.14])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Style header
            for i in range(6):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')

            ax.axis('off')
            ax.set_title('Feature Statistics Summary (Top 10 Features)',
                        fontsize=14, fontweight='bold', pad=20)

            plt.tight_layout()
            plt.close(fig)
            figures['feature_statistics'] = fig
            logger.info("✓ Created feature statistics")

        # 5. Enhanced Correlation Visualization (similar to MLflow tutorial)
        logger.info("Creating enhanced correlation visualization...")
        try:
            fig = create_enhanced_correlation_visualization(df, target_column)
            if fig is not None:
                plt.close(fig)
                figures['enhanced_correlation_analysis'] = fig
                logger.info("✓ Created enhanced correlation visualization")
        except Exception as e:
            logger.warning(f"Could not create enhanced correlation visualization: {e}")

        # 6. PCA Visualization (similar to MLflow tutorial)
        logger.info("Creating PCA visualization...")
        try:
            fig = create_pca_visualization(df, target_column, n_components=2)
            if fig is not None:
                plt.close(fig)
                figures['pca_analysis'] = fig
                logger.info("✓ Created PCA visualization")
        except Exception as e:
            logger.warning(f"Could not create PCA visualization: {e}")

        logger.info(f"✓ Created {len(figures)} preprocessing visualizations")

    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

    return figures


def log_figure_to_mlflow(fig: plt.Figure, artifact_name: str) -> None:
    """
    Log a matplotlib figure to MLflow ensuring proper binary PNG encoding.

    This ensures the PNG is saved as a proper binary file (not base64-encoded)
    so it renders correctly in the MLflow UI.

    Args:
        fig: Matplotlib figure object
        artifact_name: Name for the artifact (should end with .png)
    """
    import io
    import tempfile

    try:
        # Method 1: Use mlflow.log_figure() directly (preferred method)
        # MLflow handles the encoding internally and should produce binary PNG
        mlflow.log_figure(fig, artifact_name)
        logger.info(f"  ✓ Logged {artifact_name} to MLflow")

    except Exception as e1:
        # Method 2: Fallback - manually save as binary PNG then log as artifact
        logger.warning(f"  ⚠ mlflow.log_figure() failed for {artifact_name}, trying manual save: {e1}")
        try:
            # Save to temporary file ensuring binary PNG format
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp:
                # Save figure as binary PNG (not base64)
                fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                tmp_path = tmp.name

            # Verify it's a proper binary PNG (starts with PNG magic number)
            with open(tmp_path, 'rb') as f:
                magic_bytes = f.read(4)
                if magic_bytes != b'\x89PNG':
                    raise ValueError(f"Generated file is not a valid binary PNG (got {magic_bytes.hex()})")

            # Log as artifact
            mlflow.log_artifact(tmp_path, artifact_path='')
            logger.info(f"  ✓ Logged {artifact_name} to MLflow (via artifact)")

            # Clean up temp file
            import os
            os.unlink(tmp_path)

        except Exception as e2:
            logger.error(f"  ✗ Failed to log {artifact_name}: {e2}")


def log_preprocessing_to_mlflow(
    df: pd.DataFrame,
    figures: Dict[str, Any],
    stats: Dict[str, Any]
) -> None:
    """
    Log preprocessing visualizations and metrics to MLflow.

    Following MLflow best practices, uses mlflow.log_figure() to log
    matplotlib figure objects directly. Ensures images are saved as
    proper binary PNG files (not base64-encoded) for MLflow UI rendering.

    Args:
        df: DataFrame being processed
        figures: Dictionary of matplotlib figure objects
        stats: Preprocessing statistics
    """
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not available - skipping preprocessing logs")
        return

    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not mlflow_uri:
        logger.info("MLFLOW_TRACKING_URI not set - skipping preprocessing logs")
        return

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'fraud-detection-preprocessing')
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="preprocessing"):
            logger.info("Logging preprocessing to MLflow...")

            # Log parameters
            mlflow.log_param("total_rows", stats['total_rows'])
            mlflow.log_param("total_columns", stats['total_columns'])
            mlflow.log_param("target_column", "is_fraud")

            # Log metrics
            if 'class_distribution' in stats and stats['class_distribution']:
                class_dist = stats['class_distribution']
                if isinstance(class_dist, dict):
                    mlflow.log_metric("non_fraud_count", class_dist.get(0, class_dist.get('0', 0)))
                    mlflow.log_metric("fraud_count", class_dist.get(1, class_dist.get('1', 0)))

                    fraud_count = class_dist.get(1, class_dist.get('1', 0))
                    non_fraud_count = class_dist.get(0, class_dist.get('0', 0))
                    if fraud_count > 0:
                        imbalance_ratio = non_fraud_count / fraud_count
                        mlflow.log_metric("class_imbalance_ratio", imbalance_ratio)

            # Log visualizations ensuring proper binary PNG format for MLflow UI
            if figures:
                for fig_name, fig in figures.items():
                    if fig is not None:
                        log_figure_to_mlflow(fig, f"{fig_name}.png")

                logger.info(f"✓ Logged {len(figures)} visualizations to MLflow")

    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")


def split_train_test(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.

    Args:
        df: Input DataFrame
        target_column: Name of target column
        test_size: Proportion of data for test set
        random_state: Random seed

    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")

    # Stratified split to maintain class distribution
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column]
    )

    logger.info(f"✓ Train set: {len(train_df):,} rows")
    logger.info(f"✓ Test set: {len(test_df):,} rows")

    return train_df, test_df


def save_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_output_dir: str,
    test_output_dir: str,
    target_column: str = 'is_fraud'
) -> None:
    """
    Save train and test datasets to separate output directories.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        train_output_dir: Output directory for training data
        test_output_dir: Output directory for test data
        target_column: Name of target column (must be first for XGBoost)
    """
    # Create output directories
    train_path = Path(train_output_dir)
    test_path = Path(test_output_dir)
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # XGBoost only accepts numeric data - filter to numeric columns only
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude columns that should NOT be used as features
    # These are identifiers, timestamps, metadata, or other non-predictive columns
    COLUMNS_TO_EXCLUDE = [
        'transaction_id',           # Unique identifier
        'transaction_timestamp',    # Raw timestamp (use derived features instead)
        'customer_id',              # ID (could cause overfitting)
        'merchant_id',              # ID (could cause overfitting)
        'card_number',              # Sensitive ID
        'cvv',                      # Should never be a feature
        'expiry_date',              # Raw date
        'transaction_date',         # Raw date
        'year',                     # Use relative time features instead
        'timestamp',                # Any other timestamp variations
        'id',                       # Generic ID column
        'data_version',             # Metadata
        'created_at',               # Metadata timestamp
        'updated_at',               # Metadata timestamp
        'fraud_prediction',         # This is model output, not input!
        'fraud_probability',        # This is model output, not input!
    ]

    # Also exclude any string columns that weren't converted to numeric
    # (like customer_gender if it has many categories)
    for col in numeric_cols[:]:  # Use slice copy to modify during iteration
        if col != target_column and col in train_df.columns:
            if train_df[col].dtype == 'object':
                logger.info(f"Excluding non-numeric column: {col} (type: {train_df[col].dtype})")
                numeric_cols.remove(col)
                COLUMNS_TO_EXCLUDE.append(col)

    # Ensure target column is included and is first
    if target_column not in numeric_cols:
        # Try to convert target column to numeric
        train_df[target_column] = pd.to_numeric(train_df[target_column], errors='coerce')
        test_df[target_column] = pd.to_numeric(test_df[target_column], errors='coerce')
        numeric_cols = [target_column] + [c for c in numeric_cols if c != target_column]
    else:
        numeric_cols = [target_column] + [c for c in numeric_cols if c != target_column]

    # Filter out excluded columns
    excluded_found = [col for col in numeric_cols if col in COLUMNS_TO_EXCLUDE]
    numeric_cols = [col for col in numeric_cols if col not in COLUMNS_TO_EXCLUDE]

    if excluded_found:
        logger.info(f"Excluded {len(excluded_found)} non-predictive columns: {excluded_found}")

    # Filter to numeric columns only
    train_df = train_df[numeric_cols].copy()
    test_df = test_df[numeric_cols].copy()

    # Fill any NaN values with 0 (XGBoost can handle this)
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    logger.info(f"Filtered to {len(numeric_cols)} numeric columns (XGBoost requirement)")
    logger.info(f"  Columns: {numeric_cols[:5]}... (target first)")
    logger.info(f"  Total features (excluding target): {len(numeric_cols) - 1}")

    # Save as CSV (compatible with XGBoost training)
    # XGBoost expects CSV files directly in the channel directory, no header
    train_file = train_path / "train.csv"
    test_file = test_path / "test.csv"

    logger.info(f"Saving training data to {train_file}")
    train_df.to_csv(train_file, index=False, header=False)

    logger.info(f"Saving test data to {test_file}")
    test_df.to_csv(test_file, index=False, header=False)

    # Save feature names for training and evaluation to use
    # First column is target, rest are features
    feature_names = [col for col in numeric_cols if col != target_column]
    feature_metadata = {
        'target_column': target_column,
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'all_columns': numeric_cols  # target + features in order
    }

    # Save to both train and test directories
    for output_dir, name in [(train_output_dir, 'train'), (test_output_dir, 'test')]:
        metadata_file = Path(output_dir) / "feature_metadata.json"
        logger.info(f"Saving feature metadata to {metadata_file}")
        with open(metadata_file, 'w') as f:
            json.dump(feature_metadata, f, indent=2)

    logger.info(f"✓ Datasets saved with {len(feature_names)} actual feature names")
    logger.info(f"  Feature names: {feature_names[:5]}...")


def save_statistics(stats: Dict[str, Any], output_dir: str) -> None:
    """
    Save preprocessing statistics to output directory.

    Args:
        stats: Statistics dictionary
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats_path = output_path / "preprocessing_stats.json"

    logger.info(f"Saving statistics to {stats_path}")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("✓ Statistics saved successfully")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess data for training")

    # Data source arguments
    parser.add_argument('--athena-table', type=str, default='training_data',
                       help='Athena table name')
    parser.add_argument('--athena-filter', type=str, default=None,
                       help='SQL WHERE clause for filtering')
    parser.add_argument('--limit', type=int, default=None,
                       help='Row limit for testing')

    # Target column
    parser.add_argument('--target-column', type=str, default='is_fraud',
                       help='Target column name')

    # Split parameters
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed (default: 42)')

    # Output paths (SageMaker ProcessingStep provides these via ProcessingOutput)
    # Each output channel maps to a separate directory
    parser.add_argument('--train-output-dir', type=str, 
                       default='/opt/ml/processing/output/train',
                       help='Output directory for training data')
    parser.add_argument('--test-output-dir', type=str,
                       default='/opt/ml/processing/output/test',
                       help='Output directory for test data')
    parser.add_argument('--stats-output-dir', type=str,
                       default='/opt/ml/processing/output/stats',
                       help='Output directory for statistics')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Data Preprocessing for SageMaker Pipeline")
    logger.info("=" * 80)
    logger.info(f"Athena table: {args.athena_table}")
    logger.info(f"Target column: {args.target_column}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Train output: {args.train_output_dir}")
    logger.info(f"Test output: {args.test_output_dir}")
    logger.info(f"Stats output: {args.stats_output_dir}")
    logger.info("")

    try:
        # Step 1: Read data from Athena
        df = read_data_from_athena(
            athena_table=args.athena_table,
            athena_filter=args.athena_filter,
            limit=args.limit
        )

        # Step 1.5: Convert boolean columns to 0/1 (CRITICAL FIX)
        # Many Athena tables store is_fraud as "True"/"False" strings
        logger.info("Converting boolean columns to numeric...")
        df = convert_boolean_columns(df)
        logger.info(f"Target column '{args.target_column}' type: {df[args.target_column].dtype}")
        logger.info(f"Target column unique values: {df[args.target_column].unique()}")

        # Step 2: Validate data quality
        stats = validate_data_quality(df, args.target_column)

        if not stats['validation_passed']:
            logger.error("Data validation failed, aborting preprocessing")
            sys.exit(1)

        # Step 3: Split into train/test
        train_df, test_df = split_train_test(
            df,
            target_column=args.target_column,
            test_size=args.test_size,
            random_state=args.random_state
        )

        # Add split statistics
        stats['train_samples'] = len(train_df)
        stats['test_samples'] = len(test_df)
        stats['train_class_distribution'] = train_df[args.target_column].value_counts().to_dict()
        stats['test_class_distribution'] = test_df[args.target_column].value_counts().to_dict()

        # Step 4: Create visualizations
        logger.info("Creating preprocessing visualizations...")
        plots = create_preprocessing_visualizations(df, args.target_column, args.stats_output_dir)

        # Step 5: Log to MLflow (if available)
        log_preprocessing_to_mlflow(df, plots, stats)

        # Step 6: Save datasets to separate output directories
        save_datasets(train_df, test_df, args.train_output_dir, args.test_output_dir, args.target_column)

        # Step 7: Save statistics
        save_statistics(stats, args.stats_output_dir)

        logger.info("=" * 80)
        logger.info("✓ Preprocessing completed successfully")
        logger.info(f"  Created {len(plots)} visualizations")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

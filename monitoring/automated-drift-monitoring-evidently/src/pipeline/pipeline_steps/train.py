"""
Training script for SageMaker Pipeline TrainingStep.

This script:
- Loads training data from S3 (output from preprocessing step)
- Trains XGBoost model with scale_pos_weight for class imbalance
- Logs metrics and artifacts to MLflow
- Saves model for evaluation step
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
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
)

# Visualization libraries - try to install if not available
VISUALIZATION_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    print("✓ Visualization libraries available")
except ImportError:
    print("⚠ Matplotlib not found, attempting to install...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib>=3.5.0', 'seaborn>=0.12.0'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        VISUALIZATION_AVAILABLE = True
        print("✓ Visualization libraries installed successfully")
    except Exception as e:
        print(f"⚠ Could not install visualization libraries: {e}")
        print("  Training will continue without visualizations")
        VISUALIZATION_AVAILABLE = False

# MLflow is optional - XGBoost container doesn't have it by default
# Need sagemaker-mlflow for ARN URI support
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    print("MLflow not found, attempting to install...")
    try:
        import subprocess
        # Install sagemaker-mlflow which includes mlflow + AWS SageMaker integration
        print("Installing sagemaker-mlflow for ARN URI support...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'sagemaker-mlflow>=0.1.0'])
        import mlflow
        import mlflow.xgboost
        MLFLOW_AVAILABLE = True
        print("✓ SageMaker MLflow installed successfully")
    except Exception as e:
        print(f"⚠ Could not install MLflow: {e}")
        print("  Training will continue without MLflow logging")
        MLFLOW_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Note: Visualization functions now return figure objects for MLflow logging
# MLflow's log_figure() API handles serialization properly


def load_data(train_dir: str, validation_dir: str, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load training data from input directories.

    Note: CSV files have no headers. Target column is always at position 0 (first column),
    followed by feature columns. Feature names are loaded from feature_metadata.json.

    Args:
        train_dir: Directory containing train.csv and feature_metadata.json
        validation_dir: Directory containing test.csv
        target_column: Name of target column (for logging/metadata only)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Loading training data from {train_dir}")
    logger.info(f"Loading validation data from {validation_dir}")

    train_path = Path(train_dir) / "train.csv"
    test_path = Path(validation_dir) / "test.csv"
    metadata_path = Path(train_dir) / "feature_metadata.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    # Load feature names from metadata (created by preprocessing step)
    if metadata_path.exists():
        logger.info(f"Loading feature metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            feature_metadata = json.load(f)
        feature_names = feature_metadata['feature_names']
        logger.info(f"✓ Loaded {len(feature_names)} actual feature names from Athena")
        logger.info(f"  Feature names: {feature_names[:5]}...")
    else:
        # Fallback to generated names if metadata not found
        logger.warning(f"Feature metadata not found at {metadata_path}")
        logger.warning("Falling back to generated feature names (f0, f1, ...)")
        # We'll generate names after loading to know the count
        feature_names = None

    # Load datasets (CSV files have no headers)
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    logger.info(f"✓ Loaded {len(train_df):,} training samples")
    logger.info(f"✓ Loaded {len(test_df):,} test samples")

    # Generate names if metadata wasn't found
    if feature_names is None:
        num_features = train_df.shape[1] - 1
        feature_names = [f'f{i}' for i in range(num_features)]
        logger.info(f"Generated {num_features} feature names: f0, f1, ..., f{num_features-1}")

    # Split features and target
    # CSV format: first column is target, remaining columns are features
    y_train = train_df.iloc[:, 0].astype(float)
    X_train = train_df.iloc[:, 1:].astype(float)

    y_test = test_df.iloc[:, 0].astype(float)
    X_test = test_df.iloc[:, 1:].astype(float)

    # Assign actual feature names to DataFrames
    X_train.columns = feature_names
    X_test.columns = feature_names

    logger.info(f"Target column: {target_column} (position 0)")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Feature names: {list(X_train.columns[:5])}...")
    logger.info(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    logger.info(f"Class distribution (test): {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate scale_pos_weight for handling class imbalance.

    Args:
        y_train: Training target values

    Returns:
        Scale factor for positive class
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()

    if pos_count == 0:
        logger.warning("No positive samples in training data")
        return 1.0

    scale_pos_weight = neg_count / pos_count
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    logger.info(f"  Negative samples: {neg_count:,}")
    logger.info(f"  Positive samples: {pos_count:,}")
    logger.info(f"  Class ratio: {scale_pos_weight:.1f}:1")

    return float(scale_pos_weight)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any]
) -> xgb.Booster:
    """
    Train XGBoost model.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        params: XGBoost parameters

    Returns:
        Trained XGBoost Booster
    """
    logger.info("Training XGBoost model...")
    logger.info(f"Parameters: {json.dumps(params, indent=2)}")

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train model
    evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=params.get('num_boost_round', 100),
        evals=evals,
        evals_result=evals_result,
        verbose_eval=10
    )

    logger.info("✓ Model training completed")

    return model


def evaluate_model(
    model: xgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model and compute metrics.

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model...")

    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'test_roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'test_pr_auc': float(average_precision_score(y_test, y_pred_proba)),
        'test_precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'test_recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'test_f1': float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        'test_true_negatives': int(tn),
        'test_false_positives': int(fp),
        'test_false_negatives': int(fn),
        'test_true_positives': int(tp),
        'test_accuracy': float((tp + tn) / (tp + tn + fp + fn))
    })

    # Log metrics
    logger.info("Model Evaluation Results:")
    logger.info(f"  ROC-AUC: {metrics['test_roc_auc']:.4f}")
    logger.info(f"  PR-AUC: {metrics['test_pr_auc']:.4f}")
    logger.info(f"  Precision: {metrics['test_precision']:.4f}")
    logger.info(f"  Recall: {metrics['test_recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['test_f1']:.4f}")
    logger.info(f"  Accuracy: {metrics['test_accuracy']:.4f}")

    return metrics


def create_training_visualizations(
    model: xgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: Dict[str, float],
    output_dir: str
) -> Dict[str, Any]:
    """
    Create comprehensive training visualizations.

    Following MLflow best practices, this function returns figure objects
    that can be logged directly with mlflow.log_figure().

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test target
        metrics: Evaluation metrics
        output_dir: Directory to save plots (for backup)

    Returns:
        Dictionary mapping plot names to matplotlib figure objects
    """
    # Skip visualizations if libraries not available
    if not VISUALIZATION_AVAILABLE:
        logger.info("⚠ Skipping training visualizations (libraries not available)")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {}  # Store figure objects, not paths
    sns.set_style("whitegrid")

    try:
        # Make predictions for visualizations
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 1. Confusion Matrix Heatmap
        logger.info("Creating confusion matrix heatmap...")
        fig, ax = plt.subplots(figsize=(10, 8))

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Non-Fraud', 'Fraud'],
                   yticklabels=['Non-Fraud', 'Fraud'],
                   ax=ax, linewidths=2, linecolor='black',
                   annot_kws={'size': 16, 'weight': 'bold'})

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

        # Add text annotations with percentages
        total = cm.sum()
        ax.text(0.5, -0.15, f'TN: {tn:,} ({tn/total*100:.1f}%)', transform=ax.transAxes,
                ha='left', fontsize=10)
        ax.text(1.5, -0.15, f'FP: {fp:,} ({fp/total*100:.1f}%)', transform=ax.transAxes,
                ha='left', fontsize=10)
        ax.text(0.5, -0.20, f'FN: {fn:,} ({fn/total*100:.1f}%)', transform=ax.transAxes,
                ha='left', fontsize=10)
        ax.text(1.5, -0.20, f'TP: {tp:,} ({tp/total*100:.1f}%)', transform=ax.transAxes,
                ha='left', fontsize=10)

        plt.tight_layout()
        plt.close(fig)
        figures['confusion_matrix'] = fig
        logger.info("✓ Created confusion matrix")

        # 2. ROC Curve
        logger.info("Creating ROC curve...")
        fig, ax = plt.subplots(figsize=(10, 8))

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = metrics['test_roc_auc']

        ax.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.close(fig)
        figures['roc_curve'] = fig
        logger.info("✓ Created ROC curve")

        # 3. Precision-Recall Curve
        logger.info("Creating precision-recall curve...")
        fig, ax = plt.subplots(figsize=(10, 8))

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = metrics['test_pr_auc']

        ax.plot(recall, precision, color='#3498db', lw=3, label=f'PR Curve (AUC = {pr_auc:.4f})')

        # Baseline (fraud rate)
        fraud_rate = y_test.sum() / len(y_test)
        ax.plot([0, 1], [fraud_rate, fraud_rate], color='navy', lw=2, linestyle='--',
                label=f'Baseline (Fraud Rate = {fraud_rate:.4f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.close(fig)
        figures['precision_recall_curve'] = fig
        logger.info("✓ Created precision-recall curve")

        # 4. Feature Importance
        logger.info("Creating feature importance chart...")
        fig, ax = plt.subplots(figsize=(12, 8))

        importance = model.get_score(importance_type='gain')
        if importance:
            # Sort by importance
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
            features, values = zip(*sorted_importance)

            # Create horizontal bar chart
            y_pos = np.arange(len(features))
            ax.barh(y_pos, values, color='#2ecc71', edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
            ax.set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.close(fig)
            figures['feature_importance'] = fig
            logger.info("✓ Created feature importance plot")

        # 5. Metrics Summary Dashboard
        logger.info("Creating metrics summary dashboard...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Metrics to display
        metric_data = [
            ('ROC-AUC', metrics['test_roc_auc'], '#e74c3c'),
            ('PR-AUC', metrics['test_pr_auc'], '#3498db'),
            ('Precision', metrics['test_precision'], '#2ecc71'),
            ('Recall', metrics['test_recall'], '#f39c12'),
            ('F1-Score', metrics['test_f1'], '#9b59b6'),
            ('Accuracy', metrics['test_accuracy'], '#1abc9c')
        ]

        # Plot each metric as gauge
        for idx, (metric_name, value, color) in enumerate(metric_data[:4]):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            # Create gauge
            theta = np.linspace(0, np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=2)
            ax.plot([0, np.cos(value * np.pi)], [0, np.sin(value * np.pi)],
                   color=color, lw=6, marker='o', markersize=10)
            ax.fill_between(np.cos(theta[theta <= value * np.pi]),
                           np.sin(theta[theta <= value * np.pi]),
                           alpha=0.3, color=color)

            ax.text(0, -0.3, f'{value:.4f}', ha='center', fontsize=20, fontweight='bold')
            ax.set_title(metric_name, fontsize=14, fontweight='bold', pad=10)
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-0.5, 1.2])
            ax.axis('off')

        plt.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.close(fig)
        figures['metrics_dashboard'] = fig
        logger.info("✓ Created metrics dashboard")

        logger.info(f"✓ Created {len(figures)} training visualizations")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

    return figures


def save_model(
    model: xgb.Booster,
    feature_names: list,
    output_dir: str
) -> None:
    """
    Save model and feature metadata.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_path / "xgboost-model.json"
    logger.info(f"Saving model to {model_path}")
    model.save_model(str(model_path))

    # Save feature names
    feature_metadata = {
        'feature_names': feature_names,
        'num_features': len(feature_names)
    }
    feature_path = output_path / "feature_names.json"
    logger.info(f"Saving feature metadata to {feature_path}")
    with open(feature_path, 'w') as f:
        json.dump(feature_metadata, f, indent=2)

    logger.info("✓ Model saved successfully")


def log_figure_to_mlflow(fig, artifact_name: str) -> None:
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
        logger.info(f"  ✓ Logged {artifact_name}")

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
            logger.info(f"  ✓ Logged {artifact_name} (via artifact)")

            # Clean up temp file
            import os
            os.unlink(tmp_path)

        except Exception as e2:
            logger.error(f"  ✗ Failed to log {artifact_name}: {e2}")


def log_to_mlflow(
    model: xgb.Booster,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    feature_names: list,
    figures: Dict[str, Any] = None
) -> str:
    """
    Log model, parameters, metrics, and visualizations to MLflow.

    Following MLflow best practices, this function uses mlflow.log_figure()
    to log matplotlib figure objects directly. Ensures images are saved as
    proper binary PNG files (not base64-encoded) for MLflow UI rendering.

    Args:
        model: Trained XGBoost model
        params: Training parameters
        metrics: Evaluation metrics
        feature_names: List of feature names
        figures: Dictionary of matplotlib figure objects (optional)

    Returns:
        MLflow run ID
    """
    logger.info("="*80)
    logger.info("TRAINING STEP - MLflow Logging")
    logger.info("="*80)

    # Check if MLflow is available
    if not MLFLOW_AVAILABLE:
        logger.warning("⚠ MLflow not installed, skipping MLflow logging")
        logger.info("   (This is expected in XGBoost container without mlflow)")
        return "NO_MLFLOW"

    logger.info("✓ MLflow is available")

    # Set MLflow tracking URI
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not mlflow_tracking_uri or mlflow_tracking_uri == '':
        logger.warning("⚠ MLFLOW_TRACKING_URI is not set or empty!")
        logger.warning("   MLflow logging will not work without tracking URI")
        logger.warning("   Set MLFLOW_TRACKING_URI in .env file")
        return "NO_TRACKING_URI"

    logger.info(f"✓ MLflow tracking URI: {mlflow_tracking_uri}")

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info("✓ MLflow tracking URI configured successfully")
    except Exception as e:
        logger.error(f"❌ Failed to set MLflow tracking URI: {e}")
        raise

    # Enable XGBoost autologging for automatic tracking
    # This captures per-round metrics, feature importance, and more
    # We disable model logging here since we handle it manually with version management
    logger.info("Enabling XGBoost autologging...")
    try:
        mlflow.xgboost.autolog(
            log_models=False,  # We handle model logging manually for version management
            log_input_examples=False,  # Large datasets - skip examples
            log_model_signatures=True,  # Capture input/output schemas
            model_format="json",  # Modern JSON format
            silent=True  # Don't print logs for each metric
        )
        logger.info("✓ XGBoost autologging enabled (per-round metrics, feature importance)")
    except Exception as e:
        logger.warning(f"⚠ Could not enable autologging: {e}")
        logger.info("  Continuing with manual logging only...")

    # Set experiment
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'credit-card-fraud-detection-training')
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"✓ MLflow experiment set: {experiment_name}")
    except Exception as e:
        logger.error(f"❌ Failed to set MLflow experiment: {e}")
        raise

    # Start MLflow run
    logger.info("Starting MLflow run...")
    try:
        with mlflow.start_run() as run:
            logger.info(f"✓ MLflow run started: {run.info.run_id}")

            # Log parameters
            logger.info(f"Logging {len(params)} parameters to MLflow...")
            mlflow.log_params(params)
            logger.info("✓ Parameters logged")

            # Log metrics
            logger.info(f"Logging {len(metrics)} metrics to MLflow...")
            mlflow.log_metrics(metrics)
            logger.info("✓ Metrics logged")

            # Log model
            logger.info("Logging model to MLflow...")
            model_name = os.getenv('MLFLOW_MODEL_NAME', 'sg-xgboost-fraud-detector')
            model_info = mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name
            )

            # Get the model version that was created
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            model_versions = client.search_model_versions(f"name='{model_name}'")
            latest_version = max([int(mv.version) for mv in model_versions])

            logger.info(f"✓ Model logged and registered as: {model_name}")
            logger.info(f"✓ Model version: {latest_version}")

            # Optionally transition to staging/production based on metrics
            if metrics.get('test_roc_auc', 0) >= 0.85 and metrics.get('test_pr_auc', 0) >= 0.50:
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage="Staging",
                    archive_existing_versions=False
                )
                logger.info(f"✓ Model version {latest_version} transitioned to Staging")

            # Log model version as a metric
            mlflow.log_param("model_version", latest_version)

            # Log feature names
            logger.info("Logging feature names...")
            mlflow.log_dict({'feature_names': feature_names}, "feature_names.json")
            logger.info("✓ Feature names logged")

            # Log visualizations ensuring proper binary PNG format for MLflow UI
            if figures:
                logger.info(f"Logging {len(figures)} visualizations...")
                for fig_name, fig in figures.items():
                    if fig is not None:
                        log_figure_to_mlflow(fig, f"{fig_name}.png")
                logger.info(f"✓ All {len(figures)} visualizations logged")

            # Log tags
            logger.info("Logging tags...")
            mlflow.set_tags({
                'pipeline_step': 'training',
                'framework': 'xgboost',
                'model_type': 'binary_classification',
                'use_case': 'fraud_detection'
            })
            logger.info("✓ Tags logged")

            run_id = run.info.run_id
            logger.info("="*80)
            logger.info(f"✅ SUCCESS! Logged to MLflow run: {run_id}")
            logger.info(f"   Tracking URI: {mlflow_tracking_uri}")
            logger.info(f"   Experiment: {experiment_name}")
            logger.info(f"   Model: {model_name}")
            logger.info("="*80)

            return run_id

    except Exception as e:
        logger.error("="*80)
        logger.error(f"❌ FAILED TO LOG TO MLFLOW: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Tracking URI: {mlflow_tracking_uri}")
        logger.error(f"   Experiment: {experiment_name}")
        logger.error("="*80)
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train XGBoost model in SageMaker Pipeline")

    # Data arguments
    parser.add_argument('--train-data-dir', type=str, default='/opt/ml/input/data/train',
                       help='Directory containing training data')
    parser.add_argument('--validation-data-dir', type=str, default='/opt/ml/input/data/validation',
                       help='Directory containing validation data')
    parser.add_argument('--target-column', type=str, default='is_fraud',
                       help='Target column name')

    # Model hyperparameters
    parser.add_argument('--max-depth', type=int, default=6,
                       help='Maximum tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--num-boost-round', type=int, default=100,
                       help='Number of boosting rounds')
    parser.add_argument('--min-child-weight', type=int, default=1,
                       help='Minimum child weight')
    parser.add_argument('--subsample', type=float, default=0.8,
                       help='Subsample ratio')
    parser.add_argument('--colsample-bytree', type=float, default=0.8,
                       help='Column sample ratio')

    # Output arguments
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model',
                       help='Directory to save model')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("XGBoost Training for SageMaker Pipeline")
    logger.info("=" * 80)
    logger.info(f"Training data: {args.train_data_dir}")
    logger.info(f"Validation data: {args.validation_data_dir}")
    logger.info(f"Model output: {args.model_dir}")
    logger.info("")

    try:
        # Step 1: Load data
        X_train, X_test, y_train, y_test = load_data(
            args.train_data_dir,
            args.validation_data_dir,
            args.target_column
        )

        # Step 2: Calculate scale_pos_weight
        scale_pos_weight = calculate_scale_pos_weight(y_train)

        # Step 3: Prepare training parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'min_child_weight': args.min_child_weight,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'scale_pos_weight': scale_pos_weight,
            'num_boost_round': args.num_boost_round,
        }

        # Step 4: Train model
        model = train_model(X_train, y_train, X_test, y_test, params)

        # Step 5: Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Step 6: Create visualizations
        logger.info("Creating training visualizations...")
        figures = create_training_visualizations(model, X_test, y_test, metrics, args.model_dir)

        # Step 7: Save model
        feature_names = X_train.columns.tolist()
        save_model(model, feature_names, args.model_dir)

        # Step 8: Log to MLflow (if available and configured)
        if MLFLOW_AVAILABLE and os.getenv('MLFLOW_TRACKING_URI'):
            run_id = log_to_mlflow(model, params, metrics, feature_names, figures)
            logger.info(f"MLflow run ID: {run_id}")
        else:
            if not MLFLOW_AVAILABLE:
                logger.info("MLflow not available - skipping MLflow logging")
            elif not os.getenv('MLFLOW_TRACKING_URI'):
                logger.info("MLFLOW_TRACKING_URI not set - skipping MLflow logging")

        logger.info("=" * 80)
        logger.info("✓ Training completed successfully")
        logger.info(f"  Created {len(figures)} visualizations")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

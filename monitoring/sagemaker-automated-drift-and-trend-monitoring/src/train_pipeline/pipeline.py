"""
SageMaker Pipeline definition for fraud detection.

This pipeline implements a complete ML workflow:
1. ProcessingStep - Data validation and preprocessing
2. TrainingStep - XGBoost model training with MLflow tracking
3. EvaluationStep - Model evaluation with quality gates
4. ConditionStep - Quality gate check (ROC-AUC >= threshold)
5. RegisterModelStep - Register model in SageMaker Model Registry
6. CreateModelStep - Create SageMaker model for deployment
7. LambdaStep - Deploy to serverless endpoint
8. LambdaStep - Test inference and log to MLflow

The pipeline supports:
- Parameterization for flexibility
- Conditional execution based on model quality
- MLflow integration for tracking and monitoring
- Athena integration for data and inference logging
- End-to-end automation from training to deployment
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.processing import (
    ProcessingInput, ProcessingOutput, ScriptProcessor, FrameworkProcessor
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.estimator import Estimator
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.parameters import (
    ParameterInteger, ParameterString, ParameterFloat, ParameterBoolean
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep, TrainingStep, CreateModelStep, TransformStep
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda
from sagemaker.model import Model
from sagemaker.xgboost import XGBoostModel
from sagemaker.transformer import Transformer
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.execution_variables import ExecutionVariables

# Local imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file BEFORE importing config
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    logger_temp = logging.getLogger(__name__)
    logger_temp.info(f"Loaded environment from: {env_path}")
else:
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f".env file not found at: {env_path}")

from src.config.config import (
    SAGEMAKER_EXEC_ROLE, LAMBDA_EXEC_ROLE, DATA_S3_BUCKET, MLFLOW_MODEL_NAME,
    S3_MODEL_ARTIFACTS, S3_TRAINING_DATA_EXPORT, ATHENA_TRAINING_TABLE,
    ATHENA_DATABASE, ATHENA_OUTPUT_S3, SERVERLESS_MEMORY_SIZE,
    SERVERLESS_MAX_CONCURRENCY, INFERENCE_LOG_BATCH_SIZE,
    INFERENCE_LOG_FLUSH_INTERVAL, HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_LOWER, LOW_CONFIDENCE_UPPER, AWS_DEFAULT_REGION, SQS_QUEUE_URL
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get MLflow tracking URI from environment
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
if not MLFLOW_TRACKING_URI:
    logger.warning("MLFLOW_TRACKING_URI not set - MLflow logging will be disabled")
else:
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")


class FraudDetectionPipeline:
    """
    SageMaker Pipeline for fraud detection with end-to-end automation.
    
    Pipeline Flow:
    1. Preprocess data from Athena
    2. Train XGBoost model with MLflow tracking
    3. Evaluate model quality
    4. Quality gate check (ROC-AUC threshold)
    5. Register model in Model Registry
    6. Create SageMaker model
    7. Deploy to serverless endpoint
    8. Test inference and log metrics to MLflow
    """

    def __init__(
        self,
        pipeline_name: str = "fraud-detection-pipeline",
        role: Optional[str] = None,
        region: str = "us-east-1",
        **kwargs
    ):
        """
        Initialize pipeline.

        Args:
            pipeline_name: Name of the pipeline
            role: SageMaker execution role ARN
            region: AWS region
            **kwargs: Additional pipeline configuration
        """
        self.pipeline_name = pipeline_name
        self.region = region

        # Get execution role
        if role:
            self.role = role
        elif SAGEMAKER_EXEC_ROLE:
            self.role = SAGEMAKER_EXEC_ROLE
        else:
            try:
                self.role = get_execution_role()
            except:
                raise ValueError(
                    "Could not determine execution role. "
                    "Please provide role ARN or set SAGEMAKER_EXEC_ROLE"
                )

        # Initialize SageMaker session
        self.session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
        self.bucket = self.session.default_bucket()
        self.account_id = boto3.client('sts').get_caller_identity()['Account']

        # Pipeline configuration
        self.config = {
            'processing_instance_type': kwargs.get('processing_instance_type', 'ml.m5.xlarge'),
            'training_instance_type': kwargs.get('training_instance_type', 'ml.m5.xlarge'),
            'transform_instance_type': kwargs.get('transform_instance_type', 'ml.m5.xlarge'),
            'framework_version': kwargs.get('framework_version', '1.2-1'),
            'py_version': kwargs.get('py_version', 'py3'),
            'xgboost_version': kwargs.get('xgboost_version', '1.7-1'),
        }

        # Lambda configuration
        self.lambda_config = {
            'deploy_function_name': kwargs.get('deploy_lambda', 'fraud-detection-deploy-endpoint'),
            'test_function_name': kwargs.get('test_lambda', 'fraud-detection-test-inference'),
            'lambda_timeout': kwargs.get('lambda_timeout', 600),
            'lambda_memory': kwargs.get('lambda_memory', 1024),
        }

        logger.info(f"Initialized pipeline: {pipeline_name}")
        logger.info(f"  Role: {self.role}")
        logger.info(f"  Region: {region}")
        logger.info(f"  Bucket: {self.bucket}")
        logger.info(f"  Account: {self.account_id}")

    def _define_parameters(self) -> Dict[str, Any]:
        """
        Define pipeline parameters.

        Returns:
            Dictionary of pipeline parameters
        """
        logger.info("Defining pipeline parameters...")

        params = {
            # Data parameters
            'athena_table': ParameterString(
                name="AthenaTable",
                default_value=ATHENA_TRAINING_TABLE
            ),
            'athena_filter': ParameterString(
                name="AthenaFilter",
                default_value=""
            ),
            'target_column': ParameterString(
                name="TargetColumn",
                default_value="is_fraud"
            ),

            # Training parameters
            'training_instance_type': ParameterString(
                name="TrainingInstanceType",
                default_value=self.config['training_instance_type']
            ),
            'max_depth': ParameterInteger(
                name="MaxDepth",
                default_value=8  # Increased from 6 for better feature interactions
            ),
            'learning_rate': ParameterFloat(
                name="LearningRate",
                default_value=0.05  # Lower learning rate for better generalization
            ),
            'num_boost_round': ParameterInteger(
                name="NumBoostRound",
                default_value=200  # Increased from 100 for better convergence
            ),

            # Evaluation parameters
            'min_roc_auc': ParameterFloat(
                name="MinRocAuc",
                default_value=0.70  # Quality gate threshold
            ),
            'min_pr_auc': ParameterFloat(
                name="MinPrAuc",
                default_value=0.30  # Quality gate threshold
            ),

            # Deployment parameters
            'endpoint_name': ParameterString(
                name="EndpointName",
                default_value="fraud-detector"
            ),
            'endpoint_memory_size': ParameterInteger(
                name="EndpointMemorySize",
                default_value=SERVERLESS_MEMORY_SIZE
            ),
            'endpoint_max_concurrency': ParameterInteger(
                name="EndpointMaxConcurrency",
                default_value=SERVERLESS_MAX_CONCURRENCY
            ),
            'enable_athena_logging': ParameterString(
                name="EnableAthenaLogging",
                default_value="true"
            ),

            # Testing parameters
            'test_num_samples': ParameterInteger(
                name="TestNumSamples",
                default_value=50
            ),

            # Model registration
            'model_approval_status': ParameterString(
                name="ModelApprovalStatus",
                default_value="Approved"  # Auto-approve for pipeline
            ),
            'model_package_group': ParameterString(
                name="ModelPackageGroup",
                default_value=MLFLOW_MODEL_NAME
            ),
        }

        logger.info(f"Defined {len(params)} parameters")
        return params

    def _create_preprocessing_step(
        self,
        params: Dict[str, Any]
    ) -> ProcessingStep:
        """
        Create PySpark-based preprocessing step for distributed processing.

        Uses PySparkProcessor for scalable data processing that can handle
        millions of rows efficiently across a distributed cluster.

        Args:
            params: Pipeline parameters

        Returns:
            ProcessingStep
        """
        logger.info("Creating PySpark preprocessing step...")

        # Use PySparkProcessor for distributed Spark processing
        # The SageMaker Spark 3.3 container includes AWS Glue Data Catalog JARs
        # Glue config is set in SparkSession.builder in preprocessing_pyspark.py
        processor = PySparkProcessor(
            base_job_name="fraud-preprocessing-spark",
            framework_version="3.3",  # Spark version
            role=self.role,
            instance_type=self.config['processing_instance_type'],
            instance_count=1,  # Single instance to avoid YARN multi-node issues
            max_runtime_in_seconds=3600,
            sagemaker_session=self.session,
            env={
                'AWS_DEFAULT_REGION': AWS_DEFAULT_REGION,
                'ATHENA_DATABASE': ATHENA_DATABASE,
                'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
                'DATA_S3_BUCKET': DATA_S3_BUCKET,
            }
        )

        # Build job arguments
        job_arguments = [
            "--athena-table", params['athena_table'],
            "--target-column", params['target_column'],
        ]

        # Processing step with PySpark script
        step = ProcessingStep(
            name="PreprocessData",
            processor=processor,
            code=str(Path(__file__).parent / "pipeline_steps" / "preprocessing_pyspark.py"),
            job_arguments=job_arguments,
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/output/train",
                    destination=f"s3://{self.bucket}/fraud-detection/preprocessing/train"
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/output/test",
                    destination=f"s3://{self.bucket}/fraud-detection/preprocessing/test"
                ),
                ProcessingOutput(
                    output_name="stats",
                    source="/opt/ml/processing/output/stats",
                    destination=f"s3://{self.bucket}/fraud-detection/preprocessing/stats"
                )
            ],
        )

        logger.info("✓ PySpark preprocessing step created")
        logger.info("  Framework: Spark 3.3")
        logger.info("  Instances: 1x ml.m5.xlarge (single node Spark)")
        logger.info("  Data Source: Athena via Glue Data Catalog")
        return step

    def _create_training_step(
        self,
        params: Dict[str, Any],
        preprocessing_step: ProcessingStep
    ) -> TrainingStep:
        """
        Create training step with MLflow integration.

        Args:
            params: Pipeline parameters
            preprocessing_step: Preprocessing step for input dependencies

        Returns:
            TrainingStep
        """
        logger.info("Creating training step...")

        # Use XGBoost Framework for script mode with MLflow support
        # MLflow is auto-installed in train.py if not present
        estimator = XGBoost(
            entry_point='train.py',
            source_dir=str(Path(__file__).parent / "pipeline_steps"),
            framework_version=self.config['xgboost_version'],
            py_version='py3',
            role=self.role,
            instance_count=1,
            instance_type=params['training_instance_type'],
            output_path=f"s3://{self.bucket}/fraud-detection/training/output",
            base_job_name="fraud-training",
            sagemaker_session=self.session,
            hyperparameters={
                # Training script hyperparameters (passed as args)
                'max-depth': params['max_depth'],
                'learning-rate': params['learning_rate'],
                'num-boost-round': params['num_boost_round'],
                'target-column': 'is_fraud',
            },
            environment={
                # MLflow configuration
                'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI if MLFLOW_TRACKING_URI else '',
                'MLFLOW_EXPERIMENT_NAME': 'credit-card-fraud-detection-training',
                'MLFLOW_MODEL_NAME': MLFLOW_MODEL_NAME,
            }
        )

        # Training step
        step = TrainingStep(
            name="TrainModel",
            estimator=estimator,
            inputs={
                "train": TrainingInput(
                    s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                    content_type="text/csv"
                ),
                "validation": TrainingInput(
                    s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    content_type="text/csv"
                )
            },
        )

        logger.info("✓ Training step created")
        return step

    def _create_evaluation_step(
        self,
        params: Dict[str, Any],
        training_step: TrainingStep,
        preprocessing_step: ProcessingStep
    ) -> tuple:
        """
        Create evaluation step.

        Args:
            params: Pipeline parameters
            training_step: Training step for model input
            preprocessing_step: Preprocessing step for test data

        Returns:
            Tuple of (ProcessingStep, PropertyFile) for evaluation
        """
        logger.info("Creating evaluation step...")

        # Use ScriptProcessor with XGBoost image (has xgboost pre-installed)
        xgboost_image = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=self.region,
            version=self.config['xgboost_version']
        )
        
        processor = ScriptProcessor(
            image_uri=xgboost_image,
            role=self.role,
            instance_type=self.config['processing_instance_type'],
            instance_count=1,
            base_job_name="fraud-evaluation",
            sagemaker_session=self.session,
            command=["python3"],
            env={
                'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI if MLFLOW_TRACKING_URI else '',
                'MLFLOW_EXPERIMENT_NAME': 'credit-card-fraud-detection-evaluation',
                'MLFLOW_MODEL_NAME': MLFLOW_MODEL_NAME,
            }
        )

        # Define property file for evaluation metrics (used by ConditionStep)
        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation",
            path="evaluation.json"
        )

        step = ProcessingStep(
            name="EvaluateModel",
            processor=processor,
            code=str(Path(__file__).parent / "pipeline_steps" / "evaluation.py"),
            job_arguments=[
                "--target-column", params['target_column'],
                "--min-roc-auc", Join(on="", values=[params['min_roc_auc']]),
                "--min-pr-auc", Join(on="", values=[params['min_pr_auc']]),
            ],
            inputs=[
                ProcessingInput(
                    source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model"
                ),
                ProcessingInput(
                    source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=f"s3://{self.bucket}/fraud-detection/evaluation"
                )
            ],
            property_files=[evaluation_report],
        )

        logger.info("✓ Evaluation step created")
        return step, evaluation_report

    def _create_register_model_step(
        self,
        params: Dict[str, Any],
        training_step: TrainingStep,
        evaluation_step: ProcessingStep
    ) -> RegisterModel:
        """
        Create model registration step.

        Args:
            params: Pipeline parameters
            training_step: Training step for model artifacts
            evaluation_step: Evaluation step for metrics

        Returns:
            RegisterModel step
        """
        logger.info("Creating register model step...")

        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=Join(
                    on="/",
                    values=[
                        evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                        "evaluation.json"
                    ]
                ),
                content_type="application/json"
            )
        )

        step = RegisterModel(
            name="RegisterModel",
            estimator=training_step.estimator,
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["application/json", "text/csv"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge", "ml.m5.large"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name=params['model_package_group'],
            approval_status=params['model_approval_status'],
            model_metrics=model_metrics,
        )

        logger.info("✓ Register model step created")
        return step

    def _create_model_step(
        self,
        params: Dict[str, Any],
        training_step: TrainingStep
    ) -> CreateModelStep:
        """
        Create SageMaker model step with custom inference handler for Athena logging.

        Args:
            params: Pipeline parameters
            training_step: Training step for model artifacts

        Returns:
            CreateModelStep
        """
        logger.info("Creating model step with custom inference handler...")

        # Path to custom inference handler (same directory as pipeline)
        inference_handler_dir = Path(__file__).parent

        # Create XGBoostModel with custom inference handler for Athena logging
        # model = XGBoostModel(
        #     model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        #     role=self.role,
        #     entry_point="inference_handler.py",  # Custom handler with Athena logging
        #     source_dir=str(inference_handler_dir),
        #     framework_version=self.config['xgboost_version'],
        #     py_version="py3",
        #     sagemaker_session=self.session,
        #     env={
        #         # MLflow tracking
        #         'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI,
        #         'MLFLOW_MODEL_NAME': MLFLOW_MODEL_NAME,

        #         # Athena logging configuration
        #         'ENABLE_ATHENA_LOGGING': 'true',
        #         'ATHENA_DATABASE': ATHENA_DATABASE,
        #         'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
        #         'INFERENCE_LOG_BATCH_SIZE': str(INFERENCE_LOG_BATCH_SIZE),
        #         'INFERENCE_LOG_FLUSH_INTERVAL': str(INFERENCE_LOG_FLUSH_INTERVAL),

        #         # Endpoint identification (for Athena filtering)
        #         'ENDPOINT_NAME': 'fraud-detector-endpoint',
        #         'MODEL_VERSION': 'v1.0',
        #         'MLFLOW_RUN_ID': 'pipeline',  # Will be set to actual run_id if available

        #         # Confidence thresholds
        #         'HIGH_CONFIDENCE_THRESHOLD': str(HIGH_CONFIDENCE_THRESHOLD),
        #         'LOW_CONFIDENCE_LOWER': str(LOW_CONFIDENCE_LOWER),
        #         'LOW_CONFIDENCE_UPPER': str(LOW_CONFIDENCE_UPPER),
        #     },
        #     dependencies=[
        #         "inference_requirements.txt",  # Required for Athena logging (relative to source_dir)
        #     ]
        # )
        model = XGBoostModel(
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            role=self.role,
            entry_point="inference.py",
            source_dir=str(Path(__file__).parent / "pipeline_steps"),
            framework_version=self.config['xgboost_version'],
            py_version="py3",
            sagemaker_session=self.session,
            env={
                'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI if MLFLOW_TRACKING_URI else '',
                'MLFLOW_MODEL_NAME': MLFLOW_MODEL_NAME,
                'ENABLE_ATHENA_LOGGING': 'true',
                'ENDPOINT_NAME': params['endpoint_name'],
                'ATHENA_DATABASE': ATHENA_DATABASE,
                'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
                'DATA_S3_BUCKET': DATA_S3_BUCKET,
                'SQS_QUEUE_URL': os.getenv('SQS_QUEUE_URL', SQS_QUEUE_URL),
                'MODEL_VERSION': 'pipeline',
                'MLFLOW_RUN_ID': 'pipeline',
            },
        )

        step = CreateModelStep(
            name="CreateModel",
            model=model,
            inputs=sagemaker.inputs.CreateModelInput(
                instance_type="ml.m5.xlarge"
            ),
        )

        logger.info("✓ Create model step with Athena logging created")
        return step

    def _create_deploy_lambda(self) -> Lambda:
        """
        Create or get Lambda function for endpoint deployment.

        Returns:
            Lambda helper object
        """
        logger.info("Creating deploy Lambda function...")

        # Use the Lambda script file
        lambda_script_path = str(Path(__file__).parent / "pipeline_steps" / "lambda_deploy_endpoint.py")

        # Use Lambda execution role if available, otherwise fall back to SageMaker role
        lambda_role = self.lambda_config.get('lambda_role') or LAMBDA_EXEC_ROLE or self.role
        
        if lambda_role == self.role:
            logger.warning("Using SageMaker role for Lambda - ensure it has lambda.amazonaws.com in trust policy")

        deploy_lambda = Lambda(
            function_name=self.lambda_config['deploy_function_name'],
            execution_role_arn=lambda_role,
            script=lambda_script_path,
            handler="lambda_deploy_endpoint.lambda_handler",
            timeout=self.lambda_config['lambda_timeout'],
            memory_size=self.lambda_config['lambda_memory'],
            session=self.session,
        )

        logger.info(f"✓ Deploy Lambda created: {self.lambda_config['deploy_function_name']}")
        return deploy_lambda

    def _create_test_lambda(self) -> Lambda:
        """
        Create or get Lambda function for inference testing.

        Returns:
            Lambda helper object
        """
        logger.info("Creating test Lambda function...")

        # Use the Lambda script file
        lambda_script_path = str(Path(__file__).parent / "pipeline_steps" / "lambda_test_inference.py")

        # Use Lambda execution role if available, otherwise fall back to SageMaker role
        lambda_role = self.lambda_config.get('lambda_role') or LAMBDA_EXEC_ROLE or self.role
        
        if lambda_role == self.role:
            logger.warning("Using SageMaker role for Lambda - ensure it has lambda.amazonaws.com in trust policy")

        test_lambda = Lambda(
            function_name=self.lambda_config['test_function_name'],
            execution_role_arn=lambda_role,
            script=lambda_script_path,
            handler="lambda_test_inference.lambda_handler",
            timeout=self.lambda_config['lambda_timeout'],
            memory_size=self.lambda_config['lambda_memory'],
            session=self.session,
        )

        logger.info(f"✓ Test Lambda created: {self.lambda_config['test_function_name']}")
        return test_lambda

    def _create_deploy_step(
        self,
        params: Dict[str, Any],
        create_model_step: CreateModelStep,
        register_step: RegisterModel
    ) -> LambdaStep:
        """
        Create Lambda step for endpoint deployment.

        Args:
            params: Pipeline parameters
            create_model_step: CreateModelStep for model name

        Returns:
            LambdaStep
        """
        logger.info("Creating deploy step...")

        deploy_lambda = self._create_deploy_lambda()

        step = LambdaStep(
            name="DeployEndpoint",
            lambda_func=deploy_lambda,
            inputs={
                "model_name": create_model_step.properties.ModelName,
                "endpoint_name": params['endpoint_name'],
                "memory_size_mb": params['endpoint_memory_size'],
                "max_concurrency": params['endpoint_max_concurrency'],
                "enable_athena_logging": params['enable_athena_logging'],
                "model_package_arn": register_step.properties.ModelPackageArn,
                "mlflow_run_id": "pipeline",
            },
            outputs=[
                LambdaOutput(output_name="endpoint_name", output_type=LambdaOutputTypeEnum.String),
                LambdaOutput(output_name="endpoint_arn", output_type=LambdaOutputTypeEnum.String),
                LambdaOutput(output_name="status", output_type=LambdaOutputTypeEnum.String),
            ],
        )

        logger.info("✓ Deploy step created")
        return step

    def _create_test_step(
        self,
        params: Dict[str, Any],
        deploy_step: LambdaStep
    ) -> LambdaStep:
        """
        Create Lambda step for inference testing.

        Args:
            params: Pipeline parameters
            deploy_step: Deploy step for endpoint name

        Returns:
            LambdaStep
        """
        logger.info("Creating test step...")

        test_lambda = self._create_test_lambda()

        step = LambdaStep(
            name="TestInference",
            lambda_func=test_lambda,
            inputs={
                "endpoint_name": params['endpoint_name'],
                "num_samples": params['test_num_samples'],
            },
            outputs=[
                LambdaOutput(output_name="total_invocations", output_type=LambdaOutputTypeEnum.Integer),
                LambdaOutput(output_name="successful_invocations", output_type=LambdaOutputTypeEnum.Integer),
                LambdaOutput(output_name="avg_latency_ms", output_type=LambdaOutputTypeEnum.Float),
            ],
        )

        logger.info("✓ Test step created")
        return step

    def _create_fail_step(self) -> FailStep:
        """
        Create fail step for quality gate failures.

        Returns:
            FailStep
        """
        logger.info("Creating fail step...")

        step = FailStep(
            name="ModelQualityFailed",
            error_message="Model quality check failed - ROC-AUC below threshold"
        )

        logger.info("✓ Fail step created")
        return step

    def _create_condition_step(
        self,
        params: Dict[str, Any],
        evaluation_step: ProcessingStep,
        evaluation_report: PropertyFile,
        success_steps: List,
        fail_step: FailStep
    ) -> ConditionStep:
        """
        Create condition step for quality gates.

        Args:
            params: Pipeline parameters
            evaluation_step: Evaluation step for metrics
            evaluation_report: PropertyFile from evaluation step
            success_steps: Steps to execute on success
            fail_step: FailStep to execute on failure

        Returns:
            ConditionStep
        """
        logger.info("Creating condition step...")

        # Define condition: ROC-AUC >= threshold
        condition = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="binary_classification_metrics.roc_auc.value"
            ),
            right=params['min_roc_auc']
        )

        step = ConditionStep(
            name="CheckModelQuality",
            conditions=[condition],
            if_steps=success_steps,
            else_steps=[fail_step]
        )

        logger.info("✓ Condition step created")
        return step


    def create_pipeline(self, include_deployment: bool = True) -> Pipeline:
        """
        Create the complete SageMaker Pipeline.

        Args:
            include_deployment: Include deployment and testing steps

        Returns:
            Pipeline instance
        """
        logger.info("=" * 80)
        logger.info(f"Creating pipeline: {self.pipeline_name}")
        logger.info(f"  Include deployment: {include_deployment}")
        logger.info("=" * 80)

        # Step 1: Define parameters
        params = self._define_parameters()

        # Step 2: Create preprocessing step
        preprocessing_step = self._create_preprocessing_step(params)

        # Step 3: Create training step
        training_step = self._create_training_step(params, preprocessing_step)

        # Step 4: Create evaluation step (returns step and property file)
        evaluation_step, evaluation_report = self._create_evaluation_step(
            params, training_step, preprocessing_step
        )

        # Step 5: Create fail step (for quality gate failures)
        fail_step = self._create_fail_step()

        # Step 6: Create register model step
        register_step = self._create_register_model_step(
            params, training_step, evaluation_step
        )

        # Build success steps based on configuration
        if include_deployment:
            # Step 7: Create model step
            create_model_step = self._create_model_step(params, training_step)

            # Step 8: Create deploy step
            deploy_step = self._create_deploy_step(params, create_model_step, register_step)

            # Step 9: Create test step
            test_step = self._create_test_step(params, deploy_step)

            # Success path: register → create model → deploy → test
            success_steps = [register_step, create_model_step, deploy_step, test_step]
        else:
            # Success path: register only
            success_steps = [register_step]

        # Step 10: Create condition step (quality gates)
        condition_step = self._create_condition_step(
            params, evaluation_step, evaluation_report, success_steps, fail_step
        )

        # Build pipeline steps
        pipeline_steps = [
            preprocessing_step,
            training_step,
            evaluation_step,
            condition_step
        ]

        # Create pipeline
        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=list(params.values()),
            steps=pipeline_steps,
            sagemaker_session=self.session,
        )

        logger.info("=" * 80)
        logger.info("✓ Pipeline created successfully")
        logger.info(f"  Total steps: {len(pipeline_steps)}")
        logger.info(f"  Parameters: {len(params)}")
        if include_deployment:
            logger.info("  Flow: Preprocess → Train → Evaluate → Quality Gate → Register → Deploy → Test")
        else:
            logger.info("  Flow: Preprocess → Train → Evaluate → Quality Gate → Register")
        logger.info("=" * 80)

        return pipeline

    def upsert_pipeline(
        self,
        description: str = "Fraud detection pipeline with MLflow monitoring",
        include_deployment: bool = True,
        tags: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Create or update pipeline.

        Args:
            description: Pipeline description
            include_deployment: Include deployment and testing steps
            tags: Optional tags

        Returns:
            Dictionary with pipeline ARN and status
        """
        logger.info(f"Upserting pipeline: {self.pipeline_name}")

        # Create pipeline
        pipeline = self.create_pipeline(include_deployment=include_deployment)

        # Add default tags (don't override user-provided tags)
        if tags is None:
            tags = []

        # Define default tags
        default_tags = [
            {'Key': 'Project', 'Value': 'FraudDetection'},
            {'Key': 'ManagedBy', 'Value': 'SageMaker'},
            {'Key': 'MLflowIntegration', 'Value': 'true'},
            {'Key': 'IncludesDeployment', 'Value': str(include_deployment).lower()},
        ]

        # Create a dictionary of existing tag keys for quick lookup
        existing_keys = {tag['Key'] for tag in tags}

        # Only add default tags if their keys don't already exist
        for default_tag in default_tags:
            if default_tag['Key'] not in existing_keys:
                tags.append(default_tag)

        # Upsert pipeline
        response = pipeline.upsert(
            role_arn=self.role,
            description=description,
            tags=tags
        )

        logger.info(f"✓ Pipeline upserted: {response['PipelineArn']}")

        return {
            'pipeline_arn': response['PipelineArn'],
            'pipeline_name': self.pipeline_name,
            'status': 'created',
            'includes_deployment': include_deployment
        }

    def start_execution(
        self,
        execution_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        wait: bool = False
    ) -> Dict[str, Any]:
        """
        Start pipeline execution.

        Args:
            execution_name: Optional execution name
            parameters: Pipeline parameters to override
            wait: Wait for execution to complete

        Returns:
            Dictionary with execution details
        """
        from datetime import datetime

        if execution_name is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            execution_name = f"{self.pipeline_name}-{timestamp}"

        logger.info(f"Starting pipeline execution: {execution_name}")

        # Create pipeline
        pipeline = self.create_pipeline()

        # Start execution
        execution = pipeline.start(
            execution_display_name=execution_name,
            parameters=parameters or {}
        )

        result = {
            'execution_arn': execution.arn,
            'execution_name': execution_name,
            'pipeline_name': self.pipeline_name,
            'status': 'Executing'
        }

        if wait:
            logger.info("Waiting for execution to complete...")
            execution.wait()
            result['status'] = execution.describe()['PipelineExecutionStatus']

        logger.info(f"✓ Execution started: {execution.arn}")
        return result


def create_fraud_detection_pipeline(
    pipeline_name: str = "fraud-detection-pipeline",
    region: str = "us-east-1",
    role: Optional[str] = None,
    **kwargs
) -> FraudDetectionPipeline:
    """
    Factory function to create fraud detection pipeline.

    Args:
        pipeline_name: Pipeline name
        region: AWS region
        role: SageMaker execution role
        **kwargs: Additional configuration

    Returns:
        FraudDetectionPipeline instance
    """
    return FraudDetectionPipeline(
        pipeline_name=pipeline_name,
        region=region,
        role=role,
        **kwargs
    )


if __name__ == '__main__':
    """Test pipeline creation."""
    import argparse

    parser = argparse.ArgumentParser(description="Create SageMaker Pipeline")
    parser.add_argument('--pipeline-name', default='fraud-detection-pipeline',
                       help='Pipeline name')
    parser.add_argument('--region', default='us-east-1',
                       help='AWS region')
    parser.add_argument('--create', action='store_true',
                       help='Create/update pipeline in SageMaker')
    parser.add_argument('--no-deployment', action='store_true',
                       help='Exclude deployment steps')
    parser.add_argument('--start', action='store_true',
                       help='Start pipeline execution after creation')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for execution to complete')

    args = parser.parse_args()

    # Create pipeline
    pipeline_builder = create_fraud_detection_pipeline(
        pipeline_name=args.pipeline_name,
        region=args.region
    )

    if args.create:
        # Upsert pipeline
        result = pipeline_builder.upsert_pipeline(
            include_deployment=not args.no_deployment
        )
        print(json.dumps(result, indent=2))

        if args.start:
            # Start execution
            exec_result = pipeline_builder.start_execution(wait=args.wait)
            print(json.dumps(exec_result, indent=2))
    else:
        # Just create definition (don't upsert)
        pipeline = pipeline_builder.create_pipeline(
            include_deployment=not args.no_deployment
        )
        print(f"Pipeline definition created: {pipeline.name}")
        print(f"Steps: {len(pipeline.steps)}")

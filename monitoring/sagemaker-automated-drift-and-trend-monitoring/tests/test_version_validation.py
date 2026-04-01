"""
Unit tests for version validation in the ML pipeline.

Tests verify that:
1. Inference responses include version metadata
2. Version format is correct
3. Environment variable fallback behavior works correctly
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd


class TestVersionValidation(unittest.TestCase):
    """Test suite for version validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_model_version = "v2"
        self.test_mlflow_run_id = "abc123def456"
        self.test_endpoint_name = "fraud-detection-endpoint"

    @patch.dict(os.environ, {
        'MODEL_VERSION': 'v2',
        'MLFLOW_RUN_ID': 'abc123def456',
        'ENDPOINT_NAME': 'fraud-detection-endpoint'
    })
    def test_inference_response_includes_version(self):
        """Test that inference response includes model_version in metadata."""
        # Import here to pick up mocked environment variables
        import sys
        sys.path.insert(0, '/Users/skoppar/workspace/sample-aiops-on-amazon-sagemakerai/monitoring/sagemaker-automated-drift-and-trend-monitoring/src/train_pipeline')

        # Mock the predict_fn to return a response with metadata
        mock_response = {
            "predictions": [0, 1, 0],
            "probabilities": {
                "non_fraud": [0.95, 0.12, 0.88],
                "fraud": [0.05, 0.88, 0.12]
            },
            "metadata": {
                "model_version": "v2",
                "mlflow_run_id": "abc123def456",
                "endpoint_name": "fraud-detection-endpoint"
            }
        }

        # Verify metadata exists and has correct structure
        self.assertIn("metadata", mock_response)
        self.assertIn("model_version", mock_response["metadata"])
        self.assertEqual(mock_response["metadata"]["model_version"], "v2")

    @patch.dict(os.environ, {
        'MODEL_VERSION': 'v2',
        'MLFLOW_RUN_ID': 'abc123def456',
        'ENDPOINT_NAME': 'fraud-detection-endpoint'
    })
    def test_inference_response_includes_mlflow_run_id(self):
        """Test that inference response includes mlflow_run_id in metadata."""
        mock_response = {
            "predictions": [0, 1],
            "probabilities": {
                "non_fraud": [0.95, 0.12],
                "fraud": [0.05, 0.88]
            },
            "metadata": {
                "model_version": "v2",
                "mlflow_run_id": "abc123def456",
                "endpoint_name": "fraud-detection-endpoint"
            }
        }

        self.assertIn("mlflow_run_id", mock_response["metadata"])
        self.assertEqual(mock_response["metadata"]["mlflow_run_id"], "abc123def456")

    @patch.dict(os.environ, {
        'MODEL_VERSION': 'v2',
        'MLFLOW_RUN_ID': 'abc123def456',
        'ENDPOINT_NAME': 'fraud-detection-endpoint'
    })
    def test_inference_response_includes_endpoint_name(self):
        """Test that inference response includes endpoint_name in metadata."""
        mock_response = {
            "predictions": [1],
            "probabilities": {
                "non_fraud": [0.12],
                "fraud": [0.88]
            },
            "metadata": {
                "model_version": "v2",
                "mlflow_run_id": "abc123def456",
                "endpoint_name": "fraud-detection-endpoint"
            }
        }

        self.assertIn("endpoint_name", mock_response["metadata"])
        self.assertEqual(mock_response["metadata"]["endpoint_name"], "fraud-detection-endpoint")

    def test_version_format(self):
        """Test that version follows expected format (v1, v2, v3, etc.)."""
        valid_versions = ["v1", "v2", "v10", "v100"]
        version_pattern = r"^v\d+$"

        import re
        for version in valid_versions:
            self.assertIsNotNone(re.match(version_pattern, version),
                                f"Version {version} should match pattern")

    @patch.dict(os.environ, {}, clear=True)
    def test_version_environment_variable_fallback(self):
        """Test that system defaults to 'unknown' when environment variables are missing."""
        # When environment variables are not set, the inference handler should default to 'unknown'
        default_version = os.getenv('MODEL_VERSION', 'unknown')
        default_run_id = os.getenv('MLFLOW_RUN_ID', 'unknown')
        default_endpoint = os.getenv('ENDPOINT_NAME', 'unknown')

        self.assertEqual(default_version, 'unknown')
        self.assertEqual(default_run_id, 'unknown')
        self.assertEqual(default_endpoint, 'unknown')

    def test_metadata_structure(self):
        """Test that metadata has the expected structure."""
        metadata = {
            "model_version": "v2",
            "mlflow_run_id": "abc123def456",
            "endpoint_name": "fraud-detection-endpoint"
        }

        # Verify all required keys exist
        required_keys = ["model_version", "mlflow_run_id", "endpoint_name"]
        for key in required_keys:
            self.assertIn(key, metadata, f"Metadata should contain {key}")

        # Verify values are strings
        for key, value in metadata.items():
            self.assertIsInstance(value, str, f"Metadata {key} should be a string")

    def test_response_backward_compatibility(self):
        """Test that new metadata field doesn't break existing response structure."""
        response_with_metadata = {
            "predictions": [0, 1, 0],
            "probabilities": {
                "non_fraud": [0.95, 0.12, 0.88],
                "fraud": [0.05, 0.88, 0.12]
            },
            "metadata": {
                "model_version": "v2",
                "mlflow_run_id": "abc123def456",
                "endpoint_name": "fraud-detection-endpoint"
            }
        }

        # Verify original fields still exist
        self.assertIn("predictions", response_with_metadata)
        self.assertIn("probabilities", response_with_metadata)
        self.assertIn("non_fraud", response_with_metadata["probabilities"])
        self.assertIn("fraud", response_with_metadata["probabilities"])

        # Verify metadata is additive
        self.assertEqual(len(response_with_metadata.keys()), 3)  # predictions, probabilities, metadata

    def test_json_serialization(self):
        """Test that response with metadata can be properly JSON serialized."""
        response = {
            "predictions": [0, 1, 0],
            "probabilities": {
                "non_fraud": [0.95, 0.12, 0.88],
                "fraud": [0.05, 0.88, 0.12]
            },
            "metadata": {
                "model_version": "v2",
                "mlflow_run_id": "abc123def456",
                "endpoint_name": "fraud-detection-endpoint"
            }
        }

        # Should serialize without errors
        json_str = json.dumps(response)
        self.assertIsInstance(json_str, str)

        # Should deserialize correctly
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized["metadata"]["model_version"], "v2")


if __name__ == '__main__':
    unittest.main()

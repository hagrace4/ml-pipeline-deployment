"""Property-based tests for configuration management.

Tests correctness properties for configuration loading and validation.
"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from contextlib import contextmanager

import pytest
import yaml
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from shared.config import (
    Settings,
    DataSourceSettings,
    ModelSettings,
    RegistrySettings,
    InferenceSettings,
    load_config_file,
    validate_required_config,
)


@contextmanager
def env_vars(**kwargs):
    """Context manager to temporarily set environment variables."""
    old_values = {}
    for key, value in kwargs.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key in kwargs:
            if old_values[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_values[key]


# Strategies for generating valid configuration values
valid_environments = st.sampled_from(["development", "staging", "production"])
valid_log_levels = st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR"])
valid_source_types = st.sampled_from(["url", "s3", "local"])
valid_model_types = st.sampled_from(["xgboost", "pytorch"])
valid_formats = st.sampled_from(["csv", "parquet", "json"])

# Strategy for valid port numbers
valid_ports = st.integers(min_value=1, max_value=65535)

# Strategy for positive floats (for learning rate, etc.)
positive_floats = st.floats(min_value=0.0001, max_value=10.0, allow_nan=False, allow_infinity=False)

# Strategy for valid validation splits
valid_splits = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)

# Strategy for non-empty strings (for paths, names)
non_empty_strings = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('L', 'N'),
    whitelist_characters='-_/.'
)).filter(lambda x: x.strip())


# Feature: ml-pipeline-deployment, Property 30: Configuration loading
# **Validates: Requirements 9.1**
class TestConfigurationLoading:
    """Property tests for configuration loading from environment variables and files."""

    @given(
        environment=valid_environments,
        debug=st.booleans(),
        log_level=valid_log_levels,
        gpu_enabled=st.booleans(),
    )
    @settings(max_examples=100)
    def test_environment_variables_are_loaded(
        self,
        environment: str,
        debug: bool,
        log_level: str,
        gpu_enabled: bool,
    ):
        """For any valid environment variable values, Settings should load them correctly.
        
        Property 30: Configuration loading
        For any system start, configuration should be loaded from environment variables.
        """
        with env_vars(
            ML_PIPELINE_ENVIRONMENT=environment,
            ML_PIPELINE_DEBUG=str(debug).lower(),
            ML_PIPELINE_LOG_LEVEL=log_level,
            ML_PIPELINE_GPU_ENABLED=str(gpu_enabled).lower(),
        ):
            settings_obj = Settings()
            
            assert settings_obj.environment == environment
            assert settings_obj.debug == debug
            assert settings_obj.log_level == log_level
            assert settings_obj.gpu_enabled == gpu_enabled

    @given(
        environment=valid_environments,
        debug=st.booleans(),
    )
    @settings(max_examples=100)
    def test_yaml_config_file_loading(self, environment: str, debug: bool):
        """For any valid YAML configuration, load_config_file should parse it correctly.
        
        Property 30: Configuration loading
        For any system start, configuration should be loaded from configuration files.
        """
        config_data = {
            "environment": environment,
            "debug": debug,
        }
        
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            
            try:
                result = load_config_file(Path(f.name))
                assert result["environment"] == environment
                assert result["debug"] == debug
            finally:
                os.unlink(f.name)

    @given(
        environment=valid_environments,
        debug=st.booleans(),
    )
    @settings(max_examples=100)
    def test_json_config_file_loading(self, environment: str, debug: bool):
        """For any valid JSON configuration, load_config_file should parse it correctly.
        
        Property 30: Configuration loading
        For any system start, configuration should be loaded from configuration files.
        """
        import json
        
        config_data = {
            "environment": environment,
            "debug": debug,
        }
        
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            
            try:
                result = load_config_file(Path(f.name))
                assert result["environment"] == environment
                assert result["debug"] == debug
            finally:
                os.unlink(f.name)


# Feature: ml-pipeline-deployment, Property 31: Configuration application
# **Validates: Requirements 9.2, 9.3, 9.4**
class TestConfigurationApplication:
    """Property tests for configuration being applied to services."""

    @given(
        source_type=valid_source_types,
        source_path=non_empty_strings,
        format_type=valid_formats,
    )
    @settings(max_examples=100)
    def test_data_source_config_application(
        self,
        source_type: str,
        source_path: str,
        format_type: str,
    ):
        """For any data source configuration, the system should use those values.
        
        Property 31: Configuration application
        For any configured data source details, the Data Ingestion Service shall use them.
        """
        with env_vars(
            DATA_SOURCE_SOURCE_TYPE=source_type,
            DATA_SOURCE_SOURCE_PATH=source_path,
            DATA_SOURCE_FORMAT=format_type,
        ):
            settings_obj = DataSourceSettings()
            
            assert settings_obj.source_type == source_type
            assert settings_obj.source_path == source_path
            assert settings_obj.format == format_type

    @given(
        model_type=valid_model_types,
        epochs=st.integers(min_value=1, max_value=1000),
        batch_size=st.integers(min_value=1, max_value=512),
        learning_rate=positive_floats,
        validation_split=valid_splits,
    )
    @settings(max_examples=100)
    def test_model_hyperparameters_application(
        self,
        model_type: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        validation_split: float,
    ):
        """For any model hyperparameters, the Training Service should use them.
        
        Property 31: Configuration application
        For any configured model hyperparameters, the Training Service shall use them.
        """
        with env_vars(
            MODEL_MODEL_TYPE=model_type,
            MODEL_EPOCHS=str(epochs),
            MODEL_BATCH_SIZE=str(batch_size),
            MODEL_LEARNING_RATE=str(learning_rate),
            MODEL_VALIDATION_SPLIT=str(validation_split),
        ):
            settings_obj = ModelSettings()
            
            assert settings_obj.model_type == model_type
            assert settings_obj.epochs == epochs
            assert settings_obj.batch_size == batch_size
            assert abs(settings_obj.learning_rate - learning_rate) < 0.0001
            assert abs(settings_obj.validation_split - validation_split) < 0.0001

    @given(
        tracking_uri=non_empty_strings,
        experiment_name=non_empty_strings,
        sagemaker_enabled=st.booleans(),
    )
    @settings(max_examples=100)
    def test_registry_config_application(
        self,
        tracking_uri: str,
        experiment_name: str,
        sagemaker_enabled: bool,
    ):
        """For any registry configuration, the system should connect to specified registry.
        
        Property 31: Configuration application
        For any configured registry endpoints, the ML Pipeline System shall connect to them.
        """
        with env_vars(
            REGISTRY_MLFLOW_TRACKING_URI=tracking_uri,
            REGISTRY_MLFLOW_EXPERIMENT_NAME=experiment_name,
            REGISTRY_SAGEMAKER_ENABLED=str(sagemaker_enabled).lower(),
        ):
            settings_obj = RegistrySettings()
            
            assert settings_obj.mlflow_tracking_uri == tracking_uri
            assert settings_obj.mlflow_experiment_name == experiment_name
            assert settings_obj.sagemaker_enabled == sagemaker_enabled


# Feature: ml-pipeline-deployment, Property 32: Invalid configuration handling
# **Validates: Requirements 9.5**
class TestInvalidConfigurationHandling:
    """Property tests for handling invalid or missing configuration."""

    @given(
        required_fields=st.lists(
            st.sampled_from([
                "data_source.source_path",
                "registry.mlflow_tracking_uri",
                "inference.model_name",
            ]),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    @settings(max_examples=100)
    def test_missing_required_fields_raise_error(self, required_fields: list[str]):
        """For any missing required configuration, the system should fail with clear error.
        
        Property 32: Invalid configuration handling
        For any invalid or missing configuration, the system shall fail with a clear error message.
        """
        settings_obj = Settings()
        
        # Clear the required fields
        for field_path in required_fields:
            parts = field_path.split(".")
            obj = settings_obj
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], "")
        
        with pytest.raises(ValueError) as exc_info:
            validate_required_config(settings_obj, required_fields)
        
        # Error message should mention the missing fields
        error_msg = str(exc_info.value)
        assert "Missing required configuration" in error_msg
        for field in required_fields:
            assert field in error_msg

    @given(invalid_env=st.sampled_from(["invalid", "test", "local", "prod"]))
    @settings(max_examples=100)
    def test_invalid_environment_value_raises_error(self, invalid_env: str):
        """For any invalid environment value, Settings should raise a validation error.
        
        Property 32: Invalid configuration handling
        For any invalid configuration, the system shall fail with a clear error message.
        """
        with env_vars(ML_PIPELINE_ENVIRONMENT=invalid_env):
            with pytest.raises(Exception):  # Pydantic ValidationError
                Settings()

    @given(invalid_port=st.integers().filter(lambda x: x < 1 or x > 65535))
    @settings(max_examples=100)
    def test_invalid_port_raises_error(self, invalid_port: int):
        """For any invalid port number, Settings should raise a validation error.
        
        Property 32: Invalid configuration handling
        For any invalid configuration, the system shall fail with a clear error message.
        """
        with env_vars(INFERENCE_PORT=str(invalid_port)):
            with pytest.raises(Exception):  # Pydantic ValidationError
                InferenceSettings()

    @given(invalid_split=st.floats().filter(lambda x: x <= 0 or x >= 1 or x != x))
    @settings(max_examples=100)
    def test_invalid_validation_split_raises_error(self, invalid_split: float):
        """For any invalid validation split, Settings should raise a validation error.
        
        Property 32: Invalid configuration handling
        For any invalid configuration, the system shall fail with a clear error message.
        """
        assume(not (0 < invalid_split < 1))  # Ensure it's actually invalid
        
        with env_vars(MODEL_VALIDATION_SPLIT=str(invalid_split)):
            with pytest.raises(Exception):  # Pydantic ValidationError
                ModelSettings()

"""Configuration management for ML Pipeline Deployment System.

Loads configuration from environment variables and/or configuration files.
Validates configuration and provides clear error messages for invalid/missing values.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSourceSettings(BaseSettings):
    """Data source configuration."""

    source_type: Literal["url", "s3", "local"] = Field(default="local")
    source_path: str = Field(default="")
    format: Literal["csv", "parquet", "json"] = Field(default="csv")
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: str | None = Field(default=None)
    aws_region: str = Field(default="us-east-1")

    model_config = SettingsConfigDict(env_prefix="DATA_SOURCE_")


class ModelSettings(BaseSettings):
    """Model training configuration."""

    model_type: Literal["xgboost", "pytorch"] = Field(default="xgboost")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    early_stopping_patience: int = Field(default=10, ge=1)
    validation_split: float = Field(default=0.2, gt=0, lt=1)

    model_config = SettingsConfigDict(env_prefix="MODEL_")


class RegistrySettings(BaseSettings):
    """Model registry configuration."""

    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="ml-pipeline")
    sagemaker_enabled: bool = Field(default=False)
    sagemaker_role_arn: str | None = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="REGISTRY_")


class InferenceSettings(BaseSettings):
    """Inference endpoint configuration."""

    model_name: str = Field(default="ml-model")
    model_version: str | None = Field(default=None)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    request_timeout: int = Field(default=30, ge=1)
    gpu_enabled: bool = Field(default=False)

    model_config = SettingsConfigDict(env_prefix="INFERENCE_")


class Settings(BaseSettings):
    """Main application settings.

    Configuration is loaded from:
    1. Environment variables (highest priority)
    2. Configuration file (if CONFIG_FILE is set)
    3. Default values (lowest priority)
    """

    # General settings
    environment: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # GPU settings
    gpu_enabled: bool = Field(default=True)
    cuda_visible_devices: str | None = Field(default=None)

    # Prometheus settings
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090, ge=1, le=65535)

    # Nested settings
    data_source: DataSourceSettings = Field(default_factory=DataSourceSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    registry: RegistrySettings = Field(default_factory=RegistrySettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)

    # Config file path
    config_file: Path | None = Field(default=None)

    model_config = SettingsConfigDict(
        env_prefix="ML_PIPELINE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @field_validator("config_file", mode="before")
    @classmethod
    def validate_config_file(cls, v: Any) -> Path | None:
        """Validate config file exists if specified."""
        if v is None:
            return None
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Configuration file not found: {path}")
        return path

    @model_validator(mode="after")
    def load_config_file(self) -> Settings:
        """Load additional configuration from file if specified."""
        if self.config_file is not None:
            config_data = load_config_file(self.config_file)
            # Merge file config with current settings (env vars take precedence)
            for key, value in config_data.items():
                if hasattr(self, key) and os.getenv(f"ML_PIPELINE_{key.upper()}") is None:
                    setattr(self, key, value)
        return self


def load_config_file(path: Path) -> dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        path: Path to configuration file

    Returns:
        Dictionary of configuration values

    Raises:
        ValueError: If file format is not supported or file is invalid
    """
    suffix = path.suffix.lower()

    if suffix in (".yml", ".yaml"):
        with open(path) as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        import json

        with open(path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a dictionary, got {type(data)}")

    return data


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Settings instance (cached after first call)
    """
    return Settings()


def validate_required_config(settings: Settings, required_fields: list[str]) -> None:
    """Validate that required configuration fields are set.

    Args:
        settings: Settings instance to validate
        required_fields: List of field paths (e.g., ["data_source.source_path"])

    Raises:
        ValueError: If any required field is missing or empty
    """
    missing = []
    for field_path in required_fields:
        parts = field_path.split(".")
        value = settings
        for part in parts:
            value = getattr(value, part, None)
            if value is None:
                break
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field_path)

    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")

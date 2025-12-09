"""Data models for ML Pipeline Deployment System.

Defines Pydantic models for data schemas, model metadata, and API contracts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FeatureDefinition(BaseModel):
    """Definition of a single feature in a data schema."""

    name: str = Field(..., min_length=1)
    dtype: str = Field(..., description="Data type (e.g., 'float64', 'int64', 'object')")
    nullable: bool = Field(default=False)
    constraints: dict[str, Any] | None = Field(default=None)

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """Validate dtype is a recognized type."""
        valid_dtypes = {
            "float64", "float32", "int64", "int32", "int16", "int8",
            "uint64", "uint32", "uint16", "uint8", "bool", "object",
            "string", "category", "datetime64", "timedelta64",
        }
        if v not in valid_dtypes:
            raise ValueError(f"Invalid dtype: {v}. Must be one of {valid_dtypes}")
        return v


class DataSchema(BaseModel):
    """Schema definition for training data."""

    features: list[FeatureDefinition] = Field(..., min_length=1)
    target: str = Field(..., min_length=1)

    @field_validator("features")
    @classmethod
    def validate_unique_feature_names(cls, v: list[FeatureDefinition]) -> list[FeatureDefinition]:
        """Ensure feature names are unique."""
        names = [f.name for f in v]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique")
        return v


class DataSourceConfig(BaseModel):
    """Configuration for data source."""

    source_type: Literal["url", "s3", "local"] = Field(default="local")
    source_path: str = Field(..., min_length=1)
    format: Literal["csv", "parquet", "json"] = Field(default="csv")
    credentials: dict[str, str] | None = Field(default=None)


class ValidationResult(BaseModel):
    """Result of data validation."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rows_validated: int = Field(default=0, ge=0)
    rows_failed: int = Field(default=0, ge=0)


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    early_stopping_patience: int = Field(default=10, ge=1)
    validation_split: float = Field(default=0.2, gt=0, lt=1)


class ModelConfig(BaseModel):
    """Configuration for ML model."""

    model_type: Literal["xgboost", "pytorch"] = Field(default="xgboost")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)


class GPUInfo(BaseModel):
    """Information about GPU usage during training."""

    gpu_model: str = Field(default="unknown")
    cuda_version: str = Field(default="unknown")
    training_duration_seconds: float = Field(default=0.0, ge=0)
    peak_memory_mb: float = Field(default=0.0, ge=0)
    gpu_utilization_percent: float = Field(default=0.0, ge=0, le=100)


class ModelMetadata(BaseModel):
    """Metadata for a trained model."""

    model_name: str = Field(..., min_length=1)
    version: str = Field(default="")
    framework: str = Field(..., description="ML framework (xgboost, pytorch)")
    framework_version: str = Field(default="unknown")
    training_timestamp: datetime = Field(default_factory=datetime.utcnow)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    gpu_info: GPUInfo = Field(default_factory=GPUInfo)
    data_version: str = Field(default="unknown")
    git_commit: str = Field(default="unknown")


class TrainingMetrics(BaseModel):
    """Metrics from model training."""

    accuracy: float | None = Field(default=None, ge=0, le=1)
    f1_score: float | None = Field(default=None, ge=0, le=1)
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    loss: float | None = Field(default=None, ge=0)
    training_duration_seconds: float = Field(default=0.0, ge=0)
    epochs_completed: int = Field(default=0, ge=0)


class IngestionMetrics(BaseModel):
    """Metrics from data ingestion."""

    rows_processed: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0)
    validation_errors: int = Field(default=0, ge=0)
    success: bool = Field(default=False)
    source_type: str = Field(default="unknown")


class PredictionRequest(BaseModel):
    """Request for model prediction."""

    instances: list[list[float]] = Field(..., min_length=1)
    model_version: str | None = Field(default=None)

    @field_validator("instances")
    @classmethod
    def validate_instances(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate instances are non-empty and consistent."""
        if not v:
            raise ValueError("instances cannot be empty")
        first_len = len(v[0])
        if not all(len(instance) == first_len for instance in v):
            raise ValueError("All instances must have the same number of features")
        return v


class PredictionResponse(BaseModel):
    """Response from model prediction."""

    predictions: list[float] = Field(...)
    model_version: str = Field(...)
    latency_ms: float = Field(..., ge=0)


class HealthResponse(BaseModel):
    """Response from health check endpoint."""

    status: Literal["healthy", "unhealthy", "degraded"] = Field(...)
    model_loaded: bool = Field(default=False)
    model_version: str | None = Field(default=None)
    details: dict[str, Any] = Field(default_factory=dict)

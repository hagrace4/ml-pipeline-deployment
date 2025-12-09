"""Unit tests for data ingestion service."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from services.data_ingestion.service import DataIngestionService
from shared.models import (
    DataSchema,
    DataSourceConfig,
    FeatureDefinition,
    IngestionMetrics,
)


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0],
    })


@pytest.fixture
def sample_schema():
    """Create sample schema for testing."""
    return DataSchema(
        features=[
            FeatureDefinition(name="feature1", dtype="float64", nullable=False),
            FeatureDefinition(name="feature2", dtype="int64", nullable=False),
        ],
        target="target",
    )


@pytest.fixture
def ingestion_service():
    """Create data ingestion service instance."""
    return DataIngestionService()


def test_fetch_local_csv(ingestion_service, sample_data, tmp_path):
    """Test fetching data from local CSV file."""
    # Save sample data
    csv_path = tmp_path / "data.csv"
    sample_data.to_csv(csv_path, index=False)

    # Fetch data
    config = DataSourceConfig(
        source_type="local",
        source_path=str(csv_path),
        format="csv",
    )
    result = ingestion_service.fetch_data(config)

    # Verify
    assert len(result) == len(sample_data)
    assert list(result.columns) == list(sample_data.columns)


def test_fetch_local_file_not_found(ingestion_service):
    """Test fetching from non-existent file raises error."""
    config = DataSourceConfig(
        source_type="local",
        source_path="/nonexistent/file.csv",
        format="csv",
    )

    with pytest.raises(FileNotFoundError):
        ingestion_service.fetch_data(config)


def test_validate_schema_valid(ingestion_service, sample_data, sample_schema):
    """Test schema validation with valid data."""
    result = ingestion_service.validate_schema(sample_data, sample_schema)

    assert result.is_valid
    assert len(result.errors) == 0
    assert result.rows_validated == len(sample_data)


def test_validate_schema_missing_column(ingestion_service, sample_schema):
    """Test schema validation with missing column."""
    data = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        # feature2 is missing
        "target": [0, 1, 0],
    })

    result = ingestion_service.validate_schema(data, sample_schema)

    assert not result.is_valid
    assert any("feature2" in error for error in result.errors)


def test_validate_schema_wrong_dtype(ingestion_service, sample_schema):
    """Test schema validation with wrong data type."""
    data = pd.DataFrame({
        "feature1": ["a", "b", "c"],  # Should be float
        "feature2": [10, 20, 30],
        "target": [0, 1, 0],
    })

    result = ingestion_service.validate_schema(data, sample_schema)

    assert not result.is_valid
    assert any("dtype" in error.lower() for error in result.errors)


def test_validate_schema_null_values(ingestion_service):
    """Test schema validation with null values in non-nullable column."""
    data = pd.DataFrame({
        "feature1": [1.0, None, 3.0],
        "feature2": [10, 20, 30],
        "target": [0, 1, 0],
    })

    schema = DataSchema(
        features=[
            FeatureDefinition(name="feature1", dtype="float64", nullable=False),
            FeatureDefinition(name="feature2", dtype="int64", nullable=False),
        ],
        target="target",
    )

    result = ingestion_service.validate_schema(data, schema)

    assert not result.is_valid
    assert any("null" in error.lower() for error in result.errors)


def test_validate_schema_with_constraints(ingestion_service):
    """Test schema validation with constraint violations."""
    data = pd.DataFrame({
        "feature1": [1.0, 2.0, 150.0],  # 150 exceeds max
        "target": [0, 1, 0],
    })

    schema = DataSchema(
        features=[
            FeatureDefinition(
                name="feature1",
                dtype="float64",
                nullable=False,
                constraints={"min": 0, "max": 100},
            ),
        ],
        target="target",
    )

    result = ingestion_service.validate_schema(data, schema)

    assert not result.is_valid
    assert any("maximum" in error.lower() for error in result.errors)


def test_save_data_csv(ingestion_service, sample_data, tmp_path):
    """Test saving data to CSV file."""
    output_path = tmp_path / "output.csv"

    ingestion_service.save_data(sample_data, output_path)

    # Verify file exists and can be read
    assert output_path.exists()
    loaded = pd.read_csv(output_path)
    assert len(loaded) == len(sample_data)
    assert list(loaded.columns) == list(sample_data.columns)


def test_save_data_parquet(ingestion_service, sample_data, tmp_path):
    """Test saving data to Parquet file."""
    output_path = tmp_path / "output.parquet"

    ingestion_service.save_data(sample_data, output_path)

    # Verify file exists and can be read
    assert output_path.exists()
    loaded = pd.read_parquet(output_path)
    assert len(loaded) == len(sample_data)


def test_save_data_creates_directory(ingestion_service, sample_data, tmp_path):
    """Test that save_data creates parent directories."""
    output_path = tmp_path / "subdir" / "nested" / "output.csv"

    ingestion_service.save_data(sample_data, output_path)

    assert output_path.exists()


def test_emit_metrics(ingestion_service):
    """Test emitting ingestion metrics."""
    metrics = IngestionMetrics(
        rows_processed=100,
        duration_seconds=5.0,
        validation_errors=0,
        success=True,
        source_type="local",
    )

    # Should not raise
    ingestion_service.emit_metrics(metrics)


def test_run_ingestion_success(ingestion_service, sample_data, sample_schema, tmp_path):
    """Test complete ingestion pipeline."""
    # Setup
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    sample_data.to_csv(input_path, index=False)

    source_config = DataSourceConfig(
        source_type="local",
        source_path=str(input_path),
        format="csv",
    )

    # Run ingestion
    metrics = ingestion_service.run_ingestion(source_config, sample_schema, output_path)

    # Verify
    assert metrics.success
    assert metrics.rows_processed == len(sample_data)
    assert output_path.exists()


def test_run_ingestion_validation_failure(ingestion_service, sample_schema, tmp_path):
    """Test ingestion pipeline with validation failure."""
    # Create invalid data (missing column)
    invalid_data = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        # feature2 is missing
        "target": [0, 1, 0],
    })

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    invalid_data.to_csv(input_path, index=False)

    source_config = DataSourceConfig(
        source_type="local",
        source_path=str(input_path),
        format="csv",
    )

    # Should raise validation error
    with pytest.raises(ValueError, match="validation failed"):
        ingestion_service.run_ingestion(source_config, sample_schema, output_path)


def test_dtypes_compatible(ingestion_service):
    """Test dtype compatibility checking."""
    # Exact match
    assert ingestion_service._dtypes_compatible("float64", "float64")

    # Compatible numeric types
    assert ingestion_service._dtypes_compatible("float32", "float64")
    assert ingestion_service._dtypes_compatible("int32", "int64")

    # Incompatible types
    assert not ingestion_service._dtypes_compatible("float64", "int64")
    assert not ingestion_service._dtypes_compatible("object", "float64")


def test_validate_constraints_min_max(ingestion_service):
    """Test constraint validation for min/max."""
    series = pd.Series([1, 5, 10, 15, 20])

    # No violations
    errors = ingestion_service._validate_constraints(
        series, "test", {"min": 0, "max": 25}
    )
    assert len(errors) == 0

    # Min violation
    errors = ingestion_service._validate_constraints(
        series, "test", {"min": 10}
    )
    assert len(errors) == 1
    assert "minimum" in errors[0].lower()

    # Max violation
    errors = ingestion_service._validate_constraints(
        series, "test", {"max": 10}
    )
    assert len(errors) == 1
    assert "maximum" in errors[0].lower()


def test_validate_constraints_allowed_values(ingestion_service):
    """Test constraint validation for allowed values."""
    series = pd.Series(["a", "b", "c", "d"])

    # No violations
    errors = ingestion_service._validate_constraints(
        series, "test", {"allowed_values": ["a", "b", "c", "d", "e"]}
    )
    assert len(errors) == 0

    # Violations
    errors = ingestion_service._validate_constraints(
        series, "test", {"allowed_values": ["a", "b"]}
    )
    assert len(errors) == 1
    assert "invalid values" in errors[0].lower()

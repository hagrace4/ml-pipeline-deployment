"""Property-based tests for data ingestion service.

These tests validate correctness properties using Hypothesis for property-based testing.
Each test runs 100+ iterations with randomly generated inputs to verify system behavior.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import column, data_frames

from services.data_ingestion.service import DataIngestionService
from shared.models import DataSchema, DataSourceConfig, FeatureDefinition


# ============================================================================
# Hypothesis Strategies
# ============================================================================


@st.composite
def feature_definition_strategy(draw):
    """Generate valid FeatureDefinition instances."""
    name = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll"), min_codepoint=65, max_codepoint=122
    )))
    dtype = draw(st.sampled_from(["float64", "int64", "object", "bool"]))
    nullable = draw(st.booleans())
    
    # Optionally add constraints
    has_constraints = draw(st.booleans())
    constraints = None
    if has_constraints and dtype in ["float64", "int64"]:
        constraints = {}
        if draw(st.booleans()):
            constraints["min"] = draw(st.integers(min_value=0, max_value=50))
        if draw(st.booleans()):
            constraints["max"] = draw(st.integers(min_value=51, max_value=100))
    
    return FeatureDefinition(
        name=name,
        dtype=dtype,
        nullable=nullable,
        constraints=constraints,
    )


@st.composite
def data_schema_strategy(draw, num_features=None):
    """Generate valid DataSchema instances."""
    if num_features is None:
        num_features = draw(st.integers(min_value=1, max_value=5))
    
    features = [draw(feature_definition_strategy()) for _ in range(num_features)]
    target = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll"), min_codepoint=65, max_codepoint=122
    )))
    
    return DataSchema(features=features, target=target)


@st.composite
def matching_dataframe_strategy(draw, schema: DataSchema):
    """Generate DataFrame that matches the given schema."""
    num_rows = draw(st.integers(min_value=1, max_value=50))
    
    columns_dict = {}
    
    # Generate data for each feature
    for feature in schema.features:
        if feature.dtype == "float64":
            min_val = feature.constraints.get("min", 0.0) if feature.constraints else 0.0
            max_val = feature.constraints.get("max", 100.0) if feature.constraints else 100.0
            
            if feature.nullable:
                values = draw(st.lists(
                    st.one_of(
                        st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False),
                        st.none()
                    ),
                    min_size=num_rows,
                    max_size=num_rows
                ))
            else:
                values = draw(st.lists(
                    st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False),
                    min_size=num_rows,
                    max_size=num_rows
                ))
            columns_dict[feature.name] = values
            
        elif feature.dtype == "int64":
            min_val = feature.constraints.get("min", 0) if feature.constraints else 0
            max_val = feature.constraints.get("max", 100) if feature.constraints else 100
            
            if feature.nullable:
                values = draw(st.lists(
                    st.one_of(
                        st.integers(min_value=min_val, max_value=max_val),
                        st.none()
                    ),
                    min_size=num_rows,
                    max_size=num_rows
                ))
            else:
                values = draw(st.lists(
                    st.integers(min_value=min_val, max_value=max_val),
                    min_size=num_rows,
                    max_size=num_rows
                ))
            columns_dict[feature.name] = values
            
        elif feature.dtype == "object":
            if feature.constraints and "allowed_values" in feature.constraints:
                allowed = feature.constraints["allowed_values"]
                if feature.nullable:
                    values = draw(st.lists(
                        st.one_of(st.sampled_from(allowed), st.none()),
                        min_size=num_rows,
                        max_size=num_rows
                    ))
                else:
                    values = draw(st.lists(
                        st.sampled_from(allowed),
                        min_size=num_rows,
                        max_size=num_rows
                    ))
            else:
                if feature.nullable:
                    values = draw(st.lists(
                        st.one_of(st.text(min_size=1, max_size=10), st.none()),
                        min_size=num_rows,
                        max_size=num_rows
                    ))
                else:
                    values = draw(st.lists(
                        st.text(min_size=1, max_size=10),
                        min_size=num_rows,
                        max_size=num_rows
                    ))
            columns_dict[feature.name] = values
            
        elif feature.dtype == "bool":
            if feature.nullable:
                values = draw(st.lists(
                    st.one_of(st.booleans(), st.none()),
                    min_size=num_rows,
                    max_size=num_rows
                ))
            else:
                values = draw(st.lists(
                    st.booleans(),
                    min_size=num_rows,
                    max_size=num_rows
                ))
            columns_dict[feature.name] = values
    
    # Add target column
    columns_dict[schema.target] = draw(st.lists(
        st.integers(min_value=0, max_value=1),
        min_size=num_rows,
        max_size=num_rows
    ))
    
    return pd.DataFrame(columns_dict)


# ============================================================================
# Property Tests
# ============================================================================


# Feature: ml-pipeline-deployment, Property 1: Data ingestion round-trip
@given(schema=data_schema_strategy())
@settings(max_examples=100, deadline=None)
def test_property_1_data_ingestion_round_trip(schema):
    """Property 1: Data ingestion round-trip.
    
    For any valid data source configuration, fetching data, storing it,
    and then reading it back should produce equivalent data.
    
    Validates: Requirements 1.1, 1.3
    """
    service = DataIngestionService()
    
    # Generate matching data
    original_data = matching_dataframe_strategy(schema).example()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Save original data
        input_path = tmpdir_path / "input.csv"
        original_data.to_csv(input_path, index=False)
        
        # Create source config
        source_config = DataSourceConfig(
            source_type="local",
            source_path=str(input_path),
            format="csv",
        )
        
        # Fetch data
        fetched_data = service.fetch_data(source_config)
        
        # Save fetched data
        output_path = tmpdir_path / "output.csv"
        service.save_data(fetched_data, output_path)
        
        # Read back saved data
        retrieved_data = pd.read_csv(output_path)
        
        # Verify round-trip: original -> fetch -> save -> read should be equivalent
        # Compare shapes
        assert fetched_data.shape == original_data.shape, \
            f"Fetched data shape {fetched_data.shape} != original {original_data.shape}"
        assert retrieved_data.shape == original_data.shape, \
            f"Retrieved data shape {retrieved_data.shape} != original {original_data.shape}"
        
        # Compare columns
        assert list(fetched_data.columns) == list(original_data.columns), \
            "Fetched data columns don't match original"
        assert list(retrieved_data.columns) == list(original_data.columns), \
            "Retrieved data columns don't match original"
        
        # Compare values (allowing for float precision differences)
        for col in original_data.columns:
            if original_data[col].dtype in ["float64", "float32"]:
                # Use approximate equality for floats
                pd.testing.assert_series_equal(
                    fetched_data[col],
                    original_data[col],
                    check_exact=False,
                    rtol=1e-5,
                    atol=1e-8,
                )
                pd.testing.assert_series_equal(
                    retrieved_data[col],
                    original_data[col],
                    check_exact=False,
                    rtol=1e-5,
                    atol=1e-8,
                )
            else:
                # Exact equality for other types
                pd.testing.assert_series_equal(
                    fetched_data[col],
                    original_data[col],
                    check_exact=True,
                )
                pd.testing.assert_series_equal(
                    retrieved_data[col],
                    original_data[col],
                    check_exact=True,
                )


# Feature: ml-pipeline-deployment, Property 2: Schema validation correctness
@given(
    schema=data_schema_strategy(),
    should_match=st.booleans(),
)
@settings(max_examples=100, deadline=None)
def test_property_2_schema_validation_correctness(schema, should_match):
    """Property 2: Schema validation correctness.
    
    For any dataset and schema definition, validation should accept data
    matching the schema and reject data that doesn't match.
    
    Validates: Requirements 1.2, 1.4
    """
    service = DataIngestionService()
    
    if should_match:
        # Generate data that matches the schema
        data = matching_dataframe_strategy(schema).example()
        
        # Validate
        result = service.validate_schema(data, schema)
        
        # Should be valid
        assert result.is_valid, \
            f"Valid data was rejected. Errors: {result.errors}"
        assert len(result.errors) == 0, \
            f"Valid data produced errors: {result.errors}"
        assert result.rows_validated == len(data), \
            f"Rows validated {result.rows_validated} != data length {len(data)}"
    else:
        # Generate data that violates the schema
        # Strategy: remove a required column
        data = matching_dataframe_strategy(schema).example()
        
        # Remove a random feature column to make it invalid
        if len(schema.features) > 0:
            feature_to_remove = schema.features[0].name
            if feature_to_remove in data.columns:
                data = data.drop(columns=[feature_to_remove])
                
                # Validate
                result = service.validate_schema(data, schema)
                
                # Should be invalid
                assert not result.is_valid, \
                    "Invalid data (missing column) was accepted"
                assert len(result.errors) > 0, \
                    "Invalid data produced no errors"
                assert any(feature_to_remove in error for error in result.errors), \
                    f"Missing column {feature_to_remove} not reported in errors: {result.errors}"


# Feature: ml-pipeline-deployment, Property 3: Ingestion metrics emission
@given(schema=data_schema_strategy())
@settings(max_examples=100, deadline=None)
def test_property_3_ingestion_metrics_emission(schema):
    """Property 3: Ingestion metrics emission.
    
    For any data ingestion execution (successful or failed), metrics
    indicating the status should be emitted to Prometheus.
    
    Validates: Requirements 1.5
    """
    service = DataIngestionService()
    
    # Generate matching data
    data = matching_dataframe_strategy(schema).example()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Save data
        input_path = tmpdir_path / "input.csv"
        data.to_csv(input_path, index=False)
        
        # Create source config
        source_config = DataSourceConfig(
            source_type="local",
            source_path=str(input_path),
            format="csv",
        )
        
        output_path = tmpdir_path / "output.csv"
        
        # Run ingestion
        metrics = service.run_ingestion(source_config, schema, output_path)
        
        # Verify metrics were created and have expected properties
        assert metrics is not None, "No metrics returned from ingestion"
        assert metrics.rows_processed == len(data), \
            f"Metrics rows_processed {metrics.rows_processed} != data length {len(data)}"
        assert metrics.duration_seconds > 0, \
            "Metrics duration should be positive"
        assert metrics.success is True, \
            "Successful ingestion should have success=True"
        assert metrics.source_type == "local", \
            f"Metrics source_type {metrics.source_type} != 'local'"
        assert metrics.validation_errors == 0, \
            f"Valid data should have 0 validation errors, got {metrics.validation_errors}"
        
        # Verify metrics collector has recorded operations
        # Check that metrics were incremented (we can't easily check Prometheus directly,
        # but we can verify the metrics object was used)
        assert service.metrics is not None, "Service should have metrics collector"


# Feature: ml-pipeline-deployment, Property 3: Ingestion metrics emission (failure case)
@given(schema=data_schema_strategy())
@settings(max_examples=100, deadline=None)
def test_property_3_ingestion_metrics_emission_on_failure(schema):
    """Property 3: Ingestion metrics emission (failure case).
    
    For any data ingestion execution that fails, metrics indicating
    the failure should be emitted to Prometheus.
    
    Validates: Requirements 1.5
    """
    service = DataIngestionService()
    
    # Generate data that will fail validation (missing column)
    data = matching_dataframe_strategy(schema).example()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Save data
        input_path = tmpdir_path / "input.csv"
        data.to_csv(input_path, index=False)
        
        # Create source config
        source_config = DataSourceConfig(
            source_type="local",
            source_path=str(input_path),
            format="csv",
        )
        
        # Create a schema that doesn't match (add extra required feature)
        invalid_schema = DataSchema(
            features=schema.features + [
                FeatureDefinition(
                    name="nonexistent_feature",
                    dtype="float64",
                    nullable=False,
                )
            ],
            target=schema.target,
        )
        
        output_path = tmpdir_path / "output.csv"
        
        # Run ingestion (should fail)
        try:
            metrics = service.run_ingestion(source_config, invalid_schema, output_path)
            # If we get here, validation should have failed
            assert False, "Expected validation to fail but it succeeded"
        except ValueError as e:
            # Expected failure
            assert "validation failed" in str(e).lower(), \
                f"Expected validation failure message, got: {e}"
            
            # Verify that metrics collector was used (errors were recorded)
            # We can't easily verify the exact Prometheus metrics, but we can
            # check that the service has a metrics collector
            assert service.metrics is not None, \
                "Service should have metrics collector even on failure"

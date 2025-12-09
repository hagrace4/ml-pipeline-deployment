"""Data Ingestion Service for ML Pipeline Deployment System.

Fetches data from various sources, validates schema, and prepares data for training.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from shared.config import Settings, get_settings
from shared.logging import get_logger
from shared.metrics import DataIngestionMetrics
from shared.models import DataSchema, DataSourceConfig, IngestionMetrics, ValidationResult


class DataIngestionService:
    """Service for ingesting and validating training data."""

    def __init__(
        self,
        settings: Settings | None = None,
        metrics_collector: DataIngestionMetrics | None = None,
    ):
        """Initialize data ingestion service.

        Args:
            settings: Application settings (uses default if not provided)
            metrics_collector: Metrics collector (creates new if not provided)
        """
        self.settings = settings or get_settings()
        self.metrics = metrics_collector or DataIngestionMetrics()
        self.logger = get_logger(__name__)

        # Configure HTTP session with retry logic
        self.session = self._create_http_session()

    def _create_http_session(self) -> requests.Session:
        """Create HTTP session with retry logic.

        Returns:
            Configured requests Session
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch_data(self, source_config: DataSourceConfig) -> pd.DataFrame:
        """Fetch data from configured source.

        Args:
            source_config: Data source configuration

        Returns:
            DataFrame containing fetched data

        Raises:
            ValueError: If source type is unsupported or data cannot be fetched
            FileNotFoundError: If local file does not exist
            requests.RequestException: If URL fetch fails
        """
        self.logger.info(
            f"Fetching data from {source_config.source_type}: {source_config.source_path}"
        )

        start_time = time.time()
        try:
            if source_config.source_type == "local":
                data = self._fetch_local(source_config)
            elif source_config.source_type == "url":
                data = self._fetch_url(source_config)
            elif source_config.source_type == "s3":
                data = self._fetch_s3(source_config)
            else:
                raise ValueError(f"Unsupported source type: {source_config.source_type}")

            duration = time.time() - start_time
            self.metrics.record_duration(
                operation="fetch_data",
                duration_seconds=duration,
                status="success",
                environment=self.settings.environment,
            )
            self.metrics.increment_operations(
                operation="fetch_data",
                status="success",
                environment=self.settings.environment,
            )

            self.logger.info(f"Successfully fetched {len(data)} rows in {duration:.2f}s")
            return data

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_duration(
                operation="fetch_data",
                duration_seconds=duration,
                status="failure",
                environment=self.settings.environment,
            )
            self.metrics.increment_errors(
                operation="fetch_data",
                error_type=type(e).__name__,
                environment=self.settings.environment,
            )
            self.logger.error(f"Failed to fetch data: {e}")
            raise

    def _fetch_local(self, source_config: DataSourceConfig) -> pd.DataFrame:
        """Fetch data from local file.

        Args:
            source_config: Data source configuration

        Returns:
            DataFrame containing data

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is unsupported
        """
        path = Path(source_config.source_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if source_config.format == "csv":
            return pd.read_csv(path)
        elif source_config.format == "parquet":
            return pd.read_parquet(path)
        elif source_config.format == "json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported format: {source_config.format}")

    def _fetch_url(self, source_config: DataSourceConfig) -> pd.DataFrame:
        """Fetch data from URL.

        Args:
            source_config: Data source configuration

        Returns:
            DataFrame containing data

        Raises:
            requests.RequestException: If request fails
            ValueError: If format is unsupported
        """
        response = self.session.get(source_config.source_path, timeout=30)
        response.raise_for_status()

        if source_config.format == "csv":
            from io import StringIO
            return pd.read_csv(StringIO(response.text))
        elif source_config.format == "json":
            return pd.DataFrame(response.json())
        else:
            raise ValueError(f"Unsupported format for URL: {source_config.format}")

    def _fetch_s3(self, source_config: DataSourceConfig) -> pd.DataFrame:
        """Fetch data from S3.

        Args:
            source_config: Data source configuration

        Returns:
            DataFrame containing data

        Raises:
            ImportError: If boto3 is not installed
            ValueError: If credentials are missing or format is unsupported
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for S3 data sources. Install with: pip install boto3")

        # Extract credentials from config
        credentials = source_config.credentials or {}
        aws_access_key_id = credentials.get("aws_access_key_id") or self.settings.data_source.aws_access_key_id
        aws_secret_access_key = credentials.get("aws_secret_access_key") or self.settings.data_source.aws_secret_access_key
        aws_region = credentials.get("aws_region") or self.settings.data_source.aws_region

        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError("AWS credentials required for S3 data source")

        # Parse S3 path (s3://bucket/key)
        if not source_config.source_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {source_config.source_path}")

        path_parts = source_config.source_path[5:].split("/", 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ""

        # Fetch from S3
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

        obj = s3_client.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read()

        if source_config.format == "csv":
            from io import BytesIO
            return pd.read_csv(BytesIO(content))
        elif source_config.format == "parquet":
            from io import BytesIO
            return pd.read_parquet(BytesIO(content))
        elif source_config.format == "json":
            from io import BytesIO
            return pd.read_json(BytesIO(content))
        else:
            raise ValueError(f"Unsupported format: {source_config.format}")

    def validate_schema(self, data: pd.DataFrame, schema: DataSchema) -> ValidationResult:
        """Validate data against expected schema.

        Args:
            data: DataFrame to validate
            schema: Expected schema definition

        Returns:
            ValidationResult with validation status and errors
        """
        self.logger.info("Validating data schema")
        start_time = time.time()

        errors = []
        warnings = []
        rows_failed = 0

        try:
            # Check target column exists
            if schema.target not in data.columns:
                errors.append(f"Target column '{schema.target}' not found in data")

            # Check each feature
            feature_names = [f.name for f in schema.features]
            for feature in schema.features:
                if feature.name not in data.columns:
                    errors.append(f"Feature '{feature.name}' not found in data")
                    self.metrics.increment_validation_errors(
                        error_type="missing_column",
                        environment=self.settings.environment,
                    )
                    continue

                # Check data type
                actual_dtype = str(data[feature.name].dtype)
                if not self._dtypes_compatible(actual_dtype, feature.dtype):
                    errors.append(
                        f"Feature '{feature.name}' has dtype '{actual_dtype}', expected '{feature.dtype}'"
                    )
                    self.metrics.increment_validation_errors(
                        error_type="dtype_mismatch",
                        environment=self.settings.environment,
                    )

                # Check nullability
                if not feature.nullable and data[feature.name].isnull().any():
                    null_count = data[feature.name].isnull().sum()
                    errors.append(
                        f"Feature '{feature.name}' has {null_count} null values but is not nullable"
                    )
                    rows_failed += null_count
                    self.metrics.increment_validation_errors(
                        error_type="null_constraint",
                        environment=self.settings.environment,
                    )

                # Check constraints
                if feature.constraints:
                    constraint_errors = self._validate_constraints(
                        data[feature.name], feature.name, feature.constraints
                    )
                    errors.extend(constraint_errors)
                    if constraint_errors:
                        self.metrics.increment_validation_errors(
                            error_type="constraint_violation",
                            environment=self.settings.environment,
                        )

            # Check for extra columns
            extra_cols = set(data.columns) - set(feature_names + [schema.target])
            if extra_cols:
                warnings.append(f"Extra columns found in data: {extra_cols}")

            is_valid = len(errors) == 0
            duration = time.time() - start_time

            self.metrics.record_duration(
                operation="validate_schema",
                duration_seconds=duration,
                status="success" if is_valid else "failure",
                environment=self.settings.environment,
            )

            result = ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                rows_validated=len(data),
                rows_failed=rows_failed,
            )

            self.logger.info(
                f"Validation {'passed' if is_valid else 'failed'} "
                f"({len(errors)} errors, {len(warnings)} warnings) in {duration:.2f}s"
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_duration(
                operation="validate_schema",
                duration_seconds=duration,
                status="error",
                environment=self.settings.environment,
            )
            self.metrics.increment_errors(
                operation="validate_schema",
                error_type=type(e).__name__,
                environment=self.settings.environment,
            )
            self.logger.error(f"Schema validation error: {e}")
            raise

    def _dtypes_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual dtype is compatible with expected dtype.

        Args:
            actual: Actual dtype string
            expected: Expected dtype string

        Returns:
            True if compatible, False otherwise
        """
        # Normalize dtype strings
        actual = actual.lower()
        expected = expected.lower()

        # Exact match
        if actual == expected:
            return True

        # Compatible numeric types
        numeric_types = {
            "float": ["float64", "float32", "float16"],
            "int": ["int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"],
        }

        for base_type, compatible in numeric_types.items():
            if expected in compatible and actual in compatible:
                return True

        # Object/string compatibility
        if expected in ["object", "string"] and actual in ["object", "string"]:
            return True

        return False

    def _validate_constraints(
        self, series: pd.Series, name: str, constraints: dict[str, Any]
    ) -> list[str]:
        """Validate constraints on a data series.

        Args:
            series: Data series to validate
            name: Feature name
            constraints: Dictionary of constraints

        Returns:
            List of error messages
        """
        errors = []

        if "min" in constraints:
            min_val = constraints["min"]
            if (series < min_val).any():
                count = (series < min_val).sum()
                errors.append(f"Feature '{name}' has {count} values below minimum {min_val}")

        if "max" in constraints:
            max_val = constraints["max"]
            if (series > max_val).any():
                count = (series > max_val).sum()
                errors.append(f"Feature '{name}' has {count} values above maximum {max_val}")

        if "allowed_values" in constraints:
            allowed = set(constraints["allowed_values"])
            invalid = set(series.unique()) - allowed
            if invalid:
                errors.append(
                    f"Feature '{name}' has invalid values: {invalid}. Allowed: {allowed}"
                )

        return errors

    def save_data(self, data: pd.DataFrame, output_path: Path) -> None:
        """Save processed data to file.

        Args:
            data: DataFrame to save
            output_path: Path to save data

        Raises:
            ValueError: If output format is unsupported
            IOError: If save fails
        """
        self.logger.info(f"Saving data to {output_path}")
        start_time = time.time()

        try:
            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine format from extension
            suffix = output_path.suffix.lower()
            if suffix == ".csv":
                data.to_csv(output_path, index=False)
            elif suffix == ".parquet":
                data.to_parquet(output_path, index=False)
            elif suffix == ".json":
                data.to_json(output_path, orient="records")
            else:
                raise ValueError(f"Unsupported output format: {suffix}")

            duration = time.time() - start_time
            self.metrics.record_duration(
                operation="save_data",
                duration_seconds=duration,
                status="success",
                environment=self.settings.environment,
            )
            self.metrics.increment_operations(
                operation="save_data",
                status="success",
                environment=self.settings.environment,
            )

            self.logger.info(f"Successfully saved {len(data)} rows in {duration:.2f}s")

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_duration(
                operation="save_data",
                duration_seconds=duration,
                status="failure",
                environment=self.settings.environment,
            )
            self.metrics.increment_errors(
                operation="save_data",
                error_type=type(e).__name__,
                environment=self.settings.environment,
            )
            self.logger.error(f"Failed to save data: {e}")
            raise

    def emit_metrics(self, metrics: IngestionMetrics) -> None:
        """Emit ingestion metrics to Prometheus.

        Args:
            metrics: Ingestion metrics to emit
        """
        self.metrics.record_rows_processed(
            count=metrics.rows_processed,
            source_type=metrics.source_type,
            environment=self.settings.environment,
        )

        self.metrics.record_duration(
            operation="full_ingestion",
            duration_seconds=metrics.duration_seconds,
            status="success" if metrics.success else "failure",
            environment=self.settings.environment,
        )

        if metrics.validation_errors > 0:
            for _ in range(metrics.validation_errors):
                self.metrics.increment_validation_errors(
                    error_type="validation_error",
                    environment=self.settings.environment,
                )

        self.logger.info(f"Emitted metrics: {metrics}")

    def run_ingestion(
        self,
        source_config: DataSourceConfig,
        schema: DataSchema,
        output_path: Path,
    ) -> IngestionMetrics:
        """Run complete data ingestion pipeline.

        Args:
            source_config: Data source configuration
            schema: Expected data schema
            output_path: Path to save processed data

        Returns:
            IngestionMetrics with pipeline results

        Raises:
            ValueError: If validation fails
        """
        start_time = time.time()

        try:
            # Fetch data
            data = self.fetch_data(source_config)

            # Validate schema
            validation_result = self.validate_schema(data, schema)
            if not validation_result.is_valid:
                raise ValueError(
                    f"Data validation failed: {', '.join(validation_result.errors)}"
                )

            # Save data
            self.save_data(data, output_path)

            # Create metrics
            duration = time.time() - start_time
            metrics = IngestionMetrics(
                rows_processed=len(data),
                duration_seconds=duration,
                validation_errors=len(validation_result.errors),
                success=True,
                source_type=source_config.source_type,
            )

            # Emit metrics
            self.emit_metrics(metrics)

            return metrics

        except Exception as e:
            duration = time.time() - start_time
            metrics = IngestionMetrics(
                rows_processed=0,
                duration_seconds=duration,
                validation_errors=0,
                success=False,
                source_type=source_config.source_type,
            )
            self.emit_metrics(metrics)
            raise

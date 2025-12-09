"""Prometheus metrics utilities for ML Pipeline Deployment System.

Provides base classes and utilities for emitting metrics to Prometheus.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class MetricsCollector:
    """Base class for collecting and emitting Prometheus metrics.

    Provides common metrics patterns for pipeline components.
    """

    def __init__(
        self,
        component_name: str,
        registry: CollectorRegistry | None = None,
    ):
        """Initialize metrics collector.

        Args:
            component_name: Name of the component (e.g., 'data_ingestion', 'training')
            registry: Prometheus registry (uses default if not specified)
        """
        self.component_name = component_name
        self.registry = registry or REGISTRY
        self._metrics: dict[str, Any] = {}

        # Common labels for all metrics
        self._common_labels = ["component", "environment"]

        # Initialize common metrics
        self._init_common_metrics()

    def _init_common_metrics(self) -> None:
        """Initialize common metrics shared across components."""
        # Duration histogram
        self._metrics["duration"] = Histogram(
            f"{self.component_name}_duration_seconds",
            f"Duration of {self.component_name} operations in seconds",
            labelnames=self._common_labels + ["operation", "status"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
            registry=self.registry,
        )

        # Success/failure counter
        self._metrics["operations"] = Counter(
            f"{self.component_name}_operations_total",
            f"Total number of {self.component_name} operations",
            labelnames=self._common_labels + ["operation", "status"],
            registry=self.registry,
        )

        # Error counter
        self._metrics["errors"] = Counter(
            f"{self.component_name}_errors_total",
            f"Total number of {self.component_name} errors",
            labelnames=self._common_labels + ["operation", "error_type"],
            registry=self.registry,
        )

    def record_duration(
        self,
        operation: str,
        duration_seconds: float,
        status: str = "success",
        environment: str = "development",
    ) -> None:
        """Record operation duration.

        Args:
            operation: Name of the operation
            duration_seconds: Duration in seconds
            status: Operation status (success/failure)
            environment: Deployment environment
        """
        self._metrics["duration"].labels(
            component=self.component_name,
            environment=environment,
            operation=operation,
            status=status,
        ).observe(duration_seconds)

    def increment_operations(
        self,
        operation: str,
        status: str = "success",
        environment: str = "development",
    ) -> None:
        """Increment operation counter.

        Args:
            operation: Name of the operation
            status: Operation status (success/failure)
            environment: Deployment environment
        """
        self._metrics["operations"].labels(
            component=self.component_name,
            environment=environment,
            operation=operation,
            status=status,
        ).inc()

    def increment_errors(
        self,
        operation: str,
        error_type: str,
        environment: str = "development",
    ) -> None:
        """Increment error counter.

        Args:
            operation: Name of the operation
            error_type: Type of error
            environment: Deployment environment
        """
        self._metrics["errors"].labels(
            component=self.component_name,
            environment=environment,
            operation=operation,
            error_type=error_type,
        ).inc()

    def create_gauge(
        self,
        name: str,
        description: str,
        extra_labels: list[str] | None = None,
    ) -> Gauge:
        """Create a gauge metric.

        Args:
            name: Metric name (will be prefixed with component name)
            description: Metric description
            extra_labels: Additional labels beyond common ones

        Returns:
            Prometheus Gauge instance
        """
        full_name = f"{self.component_name}_{name}"
        labels = self._common_labels + (extra_labels or [])
        gauge = Gauge(full_name, description, labelnames=labels, registry=self.registry)
        self._metrics[name] = gauge
        return gauge

    def create_counter(
        self,
        name: str,
        description: str,
        extra_labels: list[str] | None = None,
    ) -> Counter:
        """Create a counter metric.

        Args:
            name: Metric name (will be prefixed with component name)
            description: Metric description
            extra_labels: Additional labels beyond common ones

        Returns:
            Prometheus Counter instance
        """
        full_name = f"{self.component_name}_{name}"
        labels = self._common_labels + (extra_labels or [])
        counter = Counter(full_name, description, labelnames=labels, registry=self.registry)
        self._metrics[name] = counter
        return counter

    def create_histogram(
        self,
        name: str,
        description: str,
        extra_labels: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        """Create a histogram metric.

        Args:
            name: Metric name (will be prefixed with component name)
            description: Metric description
            extra_labels: Additional labels beyond common ones
            buckets: Histogram buckets

        Returns:
            Prometheus Histogram instance
        """
        full_name = f"{self.component_name}_{name}"
        labels = self._common_labels + (extra_labels or [])
        histogram = Histogram(
            full_name,
            description,
            labelnames=labels,
            buckets=buckets or Histogram.DEFAULT_BUCKETS,
            registry=self.registry,
        )
        self._metrics[name] = histogram
        return histogram

    def get_metrics_output(self) -> bytes:
        """Generate Prometheus metrics output.

        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)


class DataIngestionMetrics(MetricsCollector):
    """Metrics collector for data ingestion service."""

    def __init__(self, registry: CollectorRegistry | None = None):
        super().__init__("data_ingestion", registry)

        # Data-specific metrics
        self._rows_processed = self.create_gauge(
            "rows_processed",
            "Number of rows processed in last ingestion",
            extra_labels=["source_type"],
        )
        self._validation_errors = self.create_counter(
            "validation_errors_total",
            "Total number of validation errors",
            extra_labels=["error_type"],
        )

    def record_rows_processed(
        self,
        count: int,
        source_type: str,
        environment: str = "development",
    ) -> None:
        """Record number of rows processed."""
        self._rows_processed.labels(
            component=self.component_name,
            environment=environment,
            source_type=source_type,
        ).set(count)

    def increment_validation_errors(
        self,
        error_type: str,
        environment: str = "development",
    ) -> None:
        """Increment validation error counter."""
        self._validation_errors.labels(
            component=self.component_name,
            environment=environment,
            error_type=error_type,
        ).inc()


class TrainingMetrics(MetricsCollector):
    """Metrics collector for training service."""

    def __init__(self, registry: CollectorRegistry | None = None):
        super().__init__("training", registry)

        # Training-specific metrics
        self._gpu_utilization = self.create_gauge(
            "gpu_utilization_percent",
            "GPU utilization percentage during training",
            extra_labels=["gpu_model"],
        )
        self._model_accuracy = self.create_gauge(
            "model_accuracy",
            "Model accuracy after training",
            extra_labels=["model_type", "model_version"],
        )
        self._training_loss = self.create_gauge(
            "loss",
            "Training loss",
            extra_labels=["model_type", "epoch"],
        )

    def record_gpu_utilization(
        self,
        utilization: float,
        gpu_model: str,
        environment: str = "development",
    ) -> None:
        """Record GPU utilization."""
        self._gpu_utilization.labels(
            component=self.component_name,
            environment=environment,
            gpu_model=gpu_model,
        ).set(utilization)

    def record_model_accuracy(
        self,
        accuracy: float,
        model_type: str,
        model_version: str,
        environment: str = "development",
    ) -> None:
        """Record model accuracy."""
        self._model_accuracy.labels(
            component=self.component_name,
            environment=environment,
            model_type=model_type,
            model_version=model_version,
        ).set(accuracy)


class InferenceMetrics(MetricsCollector):
    """Metrics collector for inference service."""

    def __init__(self, registry: CollectorRegistry | None = None):
        super().__init__("inference", registry)

        # Inference-specific metrics
        self._requests = self.create_counter(
            "requests_total",
            "Total number of inference requests",
            extra_labels=["model_version", "status_code"],
        )
        self._latency = self.create_histogram(
            "latency_seconds",
            "Inference latency in seconds",
            extra_labels=["model_version"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )
        self._batch_size = self.create_histogram(
            "batch_size",
            "Batch size of inference requests",
            extra_labels=["model_version"],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        )

    def record_request(
        self,
        model_version: str,
        status_code: int,
        environment: str = "development",
    ) -> None:
        """Record inference request."""
        self._requests.labels(
            component=self.component_name,
            environment=environment,
            model_version=model_version,
            status_code=str(status_code),
        ).inc()

    def record_latency(
        self,
        latency_seconds: float,
        model_version: str,
        environment: str = "development",
    ) -> None:
        """Record inference latency."""
        self._latency.labels(
            component=self.component_name,
            environment=environment,
            model_version=model_version,
        ).observe(latency_seconds)

    def record_batch_size(
        self,
        batch_size: int,
        model_version: str,
        environment: str = "development",
    ) -> None:
        """Record batch size."""
        self._batch_size.labels(
            component=self.component_name,
            environment=environment,
            model_version=model_version,
        ).observe(batch_size)

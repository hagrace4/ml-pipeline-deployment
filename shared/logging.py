"""Logging utilities for ML Pipeline Deployment System.

Provides structured logging with consistent formatting across all services.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

# Default log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    format_string: str = LOG_FORMAT,
    date_format: str = DATE_FORMAT,
) -> None:
    """Configure root logger with consistent formatting.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_string: Log message format
        date_format: Timestamp format
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class StructuredLogger:
    """Logger that outputs structured log entries with context."""

    def __init__(self, name: str, context: dict[str, Any] | None = None):
        """Initialize structured logger.

        Args:
            name: Logger name
            context: Default context to include in all log entries
        """
        self._logger = logging.getLogger(name)
        self._context = context or {}

    def _format_message(self, message: str, extra: dict[str, Any] | None = None) -> str:
        """Format message with context."""
        ctx = {**self._context, **(extra or {})}
        if ctx:
            ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items())
            return f"{message} | {ctx_str}"
        return message

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(self._format_message(message, kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(self._format_message(message, kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(self._format_message(message, kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(self._format_message(message, kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(self._format_message(message, kwargs))

    def with_context(self, **kwargs: Any) -> StructuredLogger:
        """Create a new logger with additional context.

        Args:
            **kwargs: Additional context key-value pairs

        Returns:
            New StructuredLogger with merged context
        """
        return StructuredLogger(self._logger.name, {**self._context, **kwargs})

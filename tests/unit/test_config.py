"""Unit tests for configuration management."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import yaml

from shared.config import (
    Settings,
    load_config_file,
    validate_required_config,
)


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self):
        """Test default settings are loaded correctly."""
        settings = Settings()
        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.gpu_enabled is True

    def test_environment_override(self, monkeypatch):
        """Test environment variables override defaults."""
        monkeypatch.setenv("ML_PIPELINE_ENVIRONMENT", "production")
        monkeypatch.setenv("ML_PIPELINE_DEBUG", "true")
        
        settings = Settings()
        assert settings.environment == "production"
        assert settings.debug is True

    def test_nested_settings(self):
        """Test nested settings are accessible."""
        settings = Settings()
        assert settings.data_source.source_type == "local"
        assert settings.model.model_type == "xgboost"
        assert settings.registry.mlflow_tracking_uri == "http://localhost:5000"

    def test_nested_env_override(self, monkeypatch):
        """Test nested settings can be overridden via env vars."""
        monkeypatch.setenv("DATA_SOURCE_SOURCE_TYPE", "s3")
        monkeypatch.setenv("MODEL_MODEL_TYPE", "pytorch")
        
        settings = Settings()
        assert settings.data_source.source_type == "s3"
        assert settings.model.model_type == "pytorch"


class TestLoadConfigFile:
    """Tests for load_config_file function."""

    def test_load_yaml_file(self):
        """Test loading YAML configuration file."""
        config_data = {
            "environment": "staging",
            "debug": True,
        }
        
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            
            result = load_config_file(Path(f.name))
            assert result["environment"] == "staging"
            assert result["debug"] is True
            
            os.unlink(f.name)

    def test_load_json_file(self):
        """Test loading JSON configuration file."""
        import json
        
        config_data = {"environment": "production"}
        
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            
            result = load_config_file(Path(f.name))
            assert result["environment"] == "production"
            
            os.unlink(f.name)

    def test_unsupported_format(self):
        """Test error on unsupported file format."""
        with NamedTemporaryFile(suffix=".txt", delete=False) as f:
            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                load_config_file(Path(f.name))
            os.unlink(f.name)


class TestValidateRequiredConfig:
    """Tests for validate_required_config function."""

    def test_valid_config(self):
        """Test validation passes with all required fields."""
        settings = Settings()
        settings.data_source.source_path = "/data/test"
        
        # Should not raise
        validate_required_config(settings, ["data_source.source_path"])

    def test_missing_required_field(self):
        """Test validation fails with missing required field."""
        settings = Settings()
        settings.data_source.source_path = ""
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            validate_required_config(settings, ["data_source.source_path"])

    def test_multiple_missing_fields(self):
        """Test validation reports all missing fields."""
        settings = Settings()
        settings.data_source.source_path = ""
        
        with pytest.raises(ValueError) as exc_info:
            validate_required_config(
                settings,
                ["data_source.source_path", "nonexistent.field"]
            )
        
        assert "data_source.source_path" in str(exc_info.value)
        assert "nonexistent.field" in str(exc_info.value)

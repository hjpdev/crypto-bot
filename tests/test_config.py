import os
import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

from app.config.config import Config
from app.core.exceptions import ConfigError


@pytest.fixture
def valid_config_dict() -> Dict[str, Any]:
    """Return a valid configuration dictionary for testing."""
    return {
        "app": {
            "name": "Crypto Trading Bot",
            "log_level": "INFO",
            "strategy": "momentum"
        },
        "exchanges": {
            "kraken": {
                "api_key": "your-api-key",
                "api_secret": "your-api-secret",
                "base_url": "https://api.kraken.com",
                "test_mode": True,
                "enabled": True,

            }
        },
        "data_collection": {
            "symbol_match_patterns": ["/USD"],
            "symbol_exclude_patterns": [".d", "BULL", "BEAR", "UP", "DOWN"],
            "enabled": True,
            "interval": 60,
            "timeframes": ["1m", "5m", "1h"]
        },
        "entry_conditions": [
            {
                "indicator": "rsi",
                "value": 30,
                "operator": "<",
                "timeframe": "1h"
            }
        ],
        "exit_conditions": [
            {
                "indicator": "rsi",
                "value": 70,
                "operator": ">",
                "timeframe": "1h"
            }
        ],
        "position_management": {
            "stake_amount": 100,
            "max_drawdown_percent": 20,
            "stop_loss_percent": 10,
            "trailing_stop": True,
            "trailing_stop_percent": 5
        }
    }


@pytest.fixture
def config_file(valid_config_dict):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp:
        yaml.dump(valid_config_dict, temp)
        temp_path = temp.name

    yield temp_path

    os.unlink(temp_path)


@pytest.fixture
def config_instance(config_file = 'tests/test_config.yaml'):
    """Create a Config instance with the test config file."""
    Config._instance = None

    config = Config(config_file)
    return config


class TestConfigClass:
    """Tests for the Config class."""

    def test_singleton_pattern(self, config_file):
        """Test that Config implements the singleton pattern correctly."""
        config1 = Config(config_file)
        config2 = Config(config_file)

        assert config1 is config2
        assert id(config1) == id(config2)

    def test_get_instance(self, config_file):
        """Test the get_instance class method."""
        Config._instance = None
        config1 = Config.get_instance(config_file)
        config2 = Config.get_instance()

        assert config1 is config2
        assert id(config1) == id(config2)

    def test_load_config(self, config_instance, valid_config_dict):
        """Test that configuration is loaded correctly."""
        assert config_instance.config is not None
        assert "app" in config_instance.config
        assert config_instance.config["app"]["name"] == valid_config_dict["app"]["name"]

    def test_get_method(self, config_instance):
        """Test the get method."""
        app_config = config_instance.get("app")
        assert app_config is not None
        assert app_config["name"] == "Crypto Trading Bot"

        non_existent = config_instance.get("non_existent", "default_value")
        assert non_existent == "default_value"

    def test_get_nested(self, config_instance):
        """Test the get_nested method."""
        exchange_enabled = config_instance.get_nested("exchanges.kraken.enabled")
        assert exchange_enabled is True

        api_key = config_instance.get_nested("exchanges.kraken.api_key")
        assert api_key == "YOUR_API_KEY"

        non_existent = config_instance.get_nested("non.existent.path", "default_value")
        assert non_existent == "default_value"

        partially_invalid = config_instance.get_nested("database.non_existent", "default_value")
        assert partially_invalid == "default_value"

class TestConfigValidation:
    """Tests for configuration validation functionality."""

    def test_valid_config(self, config_file):
        """Test that a valid configuration passes validation."""
        Config._instance = None
        config = Config(config_file)

        assert config is not None

    def test_invalid_timeframe(self, valid_config_dict, config_file):
        """Test validation of invalid timeframe."""
        invalid_config = valid_config_dict.copy()
        invalid_config["data_collection"]["timeframes"] = ["1m", "invalid"]

        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        Config._instance = None

        with pytest.raises(ConfigError) as exc_info:
            Config(config_file)

        assert "timeframes" in str(exc_info.value) or "invalid" in str(exc_info.value)

    def test_invalid_data_collection(self, valid_config_dict, config_file):
        """Test validation of invalid data collection config."""
        invalid_config = valid_config_dict.copy()
        invalid_config["data_collection"]["symbols"] = []

        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        Config._instance = None

        with pytest.raises(ConfigError) as exc_info:
            Config(config_file)

        assert "symbols" in str(exc_info.value) or "too short" in str(exc_info.value)

    def test_non_existent_config(self):
        """Test loading a non-existent configuration file."""
        Config._instance = None

        with pytest.raises(FileNotFoundError):
            Config("non_existent_file.yaml")

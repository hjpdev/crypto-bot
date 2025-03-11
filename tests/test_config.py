import os
import pytest
import tempfile
import yaml
from typing import Dict, Any

from app.config.config import Config
from app.core.exceptions import ConfigError


@pytest.fixture
def valid_config_dict() -> Dict[str, Any]:
    """Return a valid configuration dictionary for testing."""
    return {
        "app": {
            "exchange_id": "kraken",
            "strategy_name": "momentum",
            "log_level": "INFO"
        },
        "data_collection": {
            "symbols": ["BTC/USD", "ETH/USD", "XRP/USD", "BTC/USDT", "ETH/USDT"],
            "symbol_match_patterns": ["/USD"],
            "symbol_exclude_patterns": [".d", "BULL", "BEAR", "UP", "DOWN"],
            "timeframes": ["1m", "5m", "1h"],
            "store_ohlcv": False,
            "interval": 60
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
            "account_balance": 10000.0,
            "max_amount": 100,
            "risk_per_trade": 2.0,
            "max_open_positions": 5,
            "stop_loss_percent": 2.5,
            "trailing_stop_percent": 1.0,
            "take_profit": {
                "type": "fixed_percentage",
                "targets": [
                    {"target": 1.5, "percentage": 20},
                    {"target": 2.5, "percentage": 25},
                    {"target": 3.0, "percentage": 30}
                ]
            }
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
def config_instance(config_file='tests/test_config.yaml'):
    """Create a Config instance with the test config file."""
    Config._instance = None

    config = Config(config_file)
    return config


class TestConfigClass:
    def test_singleton_pattern(self, config_file):
        config1 = Config(config_file)
        config2 = Config(config_file)

        assert config1 is config2
        assert id(config1) == id(config2)

    def test_get_instance(self, config_file):
        Config._instance = None
        config1 = Config.get_instance(config_file)
        config2 = Config.get_instance()

        assert config1 is config2
        assert id(config1) == id(config2)

    def test_load_config(self, config_instance, valid_config_dict):
        """Test that configuration is loaded correctly."""
        assert config_instance.config is not None
        assert "app" in config_instance.config

    def test_get_method(self, config_instance):
        """Test the get method."""
        app_config = config_instance.get("app")
        assert app_config is not None

        non_existent = config_instance.get("non_existent", "default_value")
        assert non_existent == "default_value"

    def test_get_nested(self, config_instance):
        """Test the get_nested method."""
        exchange_enabled = config_instance.get_nested("position_management.account_balance")
        assert exchange_enabled == 10000.0

        take_profit_type = config_instance.get_nested("position_management.take_profit.type")
        assert take_profit_type == "fixed_percentage"

        non_existent = config_instance.get_nested("non.existent.path", "default_value")
        assert non_existent == "default_value"

        partially_invalid = config_instance.get_nested("position_management.non_existent", "default_value")
        assert partially_invalid == "default_value"


class TestConfigValidation:
    """Tests for configuration validation functionality."""

    def test_valid_config(self, config_file):
        """Test that a valid configuration passes validation."""
        Config._instance = None
        config = Config(config_file)

        assert config is not None

    def test_invalid_timeframe(self, valid_config_dict, config_file):
        invalid_config = valid_config_dict.copy()
        invalid_config["data_collection"]["timeframes"] = ["1m", "invalid"]

        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        Config._instance = None

        with pytest.raises(ConfigError) as exc_info:
            Config(config_file)

        assert "timeframes" in str(exc_info.value) or "invalid" in str(exc_info.value)

    def test_invalid_data_collection(self, valid_config_dict, config_file):
        invalid_config = valid_config_dict.copy()
        invalid_config["data_collection"]["symbols"] = []

        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        Config._instance = None

        with pytest.raises(ConfigError) as exc_info:
            Config(config_file)

        assert "symbols" in str(exc_info.value) or "too short" in str(exc_info.value)

    def test_non_existent_config(self):
        Config._instance = None

        with pytest.raises(FileNotFoundError):
            Config("non_existent_file.yaml")

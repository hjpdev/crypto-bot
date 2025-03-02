import os
import yaml
import json
import threading
from app.config.validation import validate_config
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from app.core.exceptions import ConfigError
from app.utils.logger import logger

class Config:
    """
    Loads and validates configuration from YAML file.
    Implements the Singleton pattern to ensure only one instance exists.

    Usage:
        config = Config.get_instance()
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, _config_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Config, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._initialized:
            return

        self.config_path = config_path or os.getenv("CONFIG_PATH", "config.yaml")
        self.config: Dict[str, Any] = {}
        self._initialized = True
        self.load_config()

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> 'Config':
        """Get or create the singleton instance of Config."""
        if cls._instance is None:
            return cls(config_path)

        return cls._instance

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from the YAML file and validate it."""
        config_path = Path(self.config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

            validate_config(self.config)

            return self.config

        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            logger.exception(f"Error loading configuration: {e}")
            raise ConfigError(f"Error loading configuration: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def get_nested(self, path: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation."""
        keys = path.split('.')
        value = self.config

        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]

        return value

    def __repr__(self) -> str:
        return str(json.dumps(self.config, indent=4))

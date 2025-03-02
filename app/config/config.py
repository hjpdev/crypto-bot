import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Loads configuration from YAML file.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("CONFIG_PATH", "config.yaml")
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        config_path = Path(self.config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def __repr__(self) -> str:
        return str(json.dumps(self.config, indent=4))

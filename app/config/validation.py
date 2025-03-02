import re
from typing import Dict, Any, List, Optional, Union, Callable
from app.core.exceptions import ConfigError


# Schema types and validators
class ConfigSchemaType:
    """Base class for configuration schema types"""
    def __init__(self, required: bool = True, default: Any = None):
        self.required = required
        self.default = default

    def validate(self, value: Any, path: str) -> Any:
        """
        Validate the value against the schema type.

        Args:
            value: The value to validate
            path: The path to the value for error messages

        Returns:
            Any: The validated value

        Raises:
            ConfigError: If the value is invalid
        """
        if value is None:
            if self.required:
                raise ConfigError(f"Missing required configuration value at '{path}'")
            return self.default
        return value


class StringType(ConfigSchemaType):
    """String configuration type"""
    def __init__(
        self,
        required: bool = True,
        default: Optional[str] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        regex: Optional[str] = None
    ):
        super().__init__(required, default)
        self.min_length = min_length
        self.max_length = max_length
        self.regex = regex and re.compile(regex)

    def validate(self, value: Any, path: str) -> str:
        value = super().validate(value, path)
        if value is None:
            return value

        if not isinstance(value, str):
            raise ConfigError(f"Expected string at '{path}', got {type(value).__name__}")

        if self.min_length is not None and len(value) < self.min_length:
            raise ConfigError(f"String at '{path}' is too short. Min length: {self.min_length}")

        if self.max_length is not None and len(value) > self.max_length:
            raise ConfigError(f"String at '{path}' is too long. Max length: {self.max_length}")

        if self.regex and not self.regex.match(value):
            raise ConfigError(f"String at '{path}' does not match required pattern")

        return value


class IntegerType(ConfigSchemaType):
    """Integer configuration type"""
    def __init__(
        self,
        required: bool = True,
        default: Optional[int] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ):
        super().__init__(required, default)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any, path: str) -> int:
        value = super().validate(value, path)
        if value is None:
            return value

        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"Expected integer at '{path}', got {type(value).__name__}")

        if self.min_value is not None and value < self.min_value:
            raise ConfigError(f"Integer at '{path}' is too small. Min value: {self.min_value}")

        if self.max_value is not None and value > self.max_value:
            raise ConfigError(f"Integer at '{path}' is too large. Max value: {self.max_value}")

        return value


class FloatType(ConfigSchemaType):
    """Float configuration type"""
    def __init__(
        self,
        required: bool = True,
        default: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ):
        super().__init__(required, default)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any, path: str) -> float:
        value = super().validate(value, path)
        if value is None:
            return value

        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise ConfigError(f"Expected float at '{path}', got {type(value).__name__}")

        value = float(value)

        if self.min_value is not None and value < self.min_value:
            raise ConfigError(f"Float at '{path}' is too small. Min value: {self.min_value}")

        if self.max_value is not None and value > self.max_value:
            raise ConfigError(f"Float at '{path}' is too large. Max value: {self.max_value}")

        return value


class BooleanType(ConfigSchemaType):
    """Boolean configuration type"""
    def validate(self, value: Any, path: str) -> bool:
        value = super().validate(value, path)
        if value is None:
            return value

        if not isinstance(value, bool):
            raise ConfigError(f"Expected boolean at '{path}', got {type(value).__name__}")

        return value


class ListType(ConfigSchemaType):
    """List configuration type"""
    def __init__(
        self,
        item_type: ConfigSchemaType,
        required: bool = True,
        default: Optional[List[Any]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ):
        super().__init__(required, default if default is not None else [])
        self.item_type = item_type
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: Any, path: str) -> List[Any]:
        value = super().validate(value, path)
        if value is None:
            return value

        if not isinstance(value, list):
            raise ConfigError(f"Expected list at '{path}', got {type(value).__name__}")

        if self.min_length is not None and len(value) < self.min_length:
            raise ConfigError(f"List at '{path}' is too short. Min length: {self.min_length}")

        if self.max_length is not None and len(value) > self.max_length:
            raise ConfigError(f"List at '{path}' is too long. Max length: {self.max_length}")

        # Validate each item in the list
        validated_items = []
        for i, item in enumerate(value):
            item_path = f"{path}[{i}]"
            validated_items.append(self.item_type.validate(item, item_path))

        return validated_items


class DictType(ConfigSchemaType):
    """Dictionary configuration type"""
    def __init__(
        self,
        schema: Dict[str, ConfigSchemaType],
        required: bool = True,
        default: Optional[Dict[str, Any]] = None,
        additional_properties: bool = False
    ):
        super().__init__(required, default if default is not None else {})
        self.schema = schema
        self.additional_properties = additional_properties

    def validate(self, value: Any, path: str) -> Dict[str, Any]:
        value = super().validate(value, path)
        if value is None:
            return value

        if not isinstance(value, dict):
            raise ConfigError(f"Expected dictionary at '{path}', got {type(value).__name__}")

        validated_dict = {}

        # Check for required properties and validate all provided properties
        for key, schema_type in self.schema.items():
            key_path = f"{path}.{key}" if path else key
            if key in value:
                validated_dict[key] = schema_type.validate(value[key], key_path)
            else:
                # Use default if the key doesn't exist
                validated_val = schema_type.validate(None, key_path)
                if validated_val is not None:
                    validated_dict[key] = validated_val

        # Check for additional properties
        if not self.additional_properties:
            for key in value:
                if key not in self.schema:
                    raise ConfigError(f"Unexpected property '{key}' at '{path}'")
        else:
            # Copy over any additional properties
            for key, val in value.items():
                if key not in self.schema:
                    validated_dict[key] = val

        return validated_dict


class EnumType(ConfigSchemaType):
    """Enumeration configuration type"""
    def __init__(
        self,
        values: List[Any],
        required: bool = True,
        default: Any = None
    ):
        super().__init__(required, default)
        self.values = values

    def validate(self, value: Any, path: str) -> Any:
        value = super().validate(value, path)
        if value is None:
            return value

        if value not in self.values:
            values_str = ", ".join(repr(v) for v in self.values)
            raise ConfigError(f"Value at '{path}' must be one of: {values_str}")

        return value


class AnyType(ConfigSchemaType):
    """Any type configuration type (for values with custom validation)"""
    def __init__(
        self,
        validator: Optional[Callable[[Any, str], Any]] = None,
        required: bool = True,
        default: Any = None
    ):
        super().__init__(required, default)
        self.validator = validator

    def validate(self, value: Any, path: str) -> Any:
        value = super().validate(value, path)
        if value is None:
            return value

        if self.validator:
            try:
                return self.validator(value, path)
            except Exception as e:
                raise ConfigError(f"Custom validation failed for '{path}': {str(e)}")

        return value



timeframe_schema = lambda required=True: EnumType(
    ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
    required=required
)

app_schema = DictType({
    "name": StringType(required=True),
    "log_level": EnumType(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], required=False, default="INFO"),
    "strategy": StringType(required=True),
})

exchanges_schema = DictType({}, additional_properties=True)  # Dynamic keys for exchanges

data_collection_schema = DictType({
    "symbol_match_patterns": ListType(StringType(), required=True, min_length=1),
    "symbol_exclude_patterns": ListType(StringType(), required=True, min_length=1),
    "enabled": BooleanType(required=False, default=True),
    "interval": IntegerType(required=False, default=60, min_value=1),
    "timeframes": ListType(timeframe_schema(), required=True, min_length=1)
})

entry_conditions_schema = ListType(DictType({
    "indicator": StringType(required=True),
    "value": FloatType(required=True),
    "operator": StringType(required=True),
    "timeframe": timeframe_schema(required=False),
}), required=True, min_length=1)

exit_conditions_schema = ListType(DictType({
    "indicator": StringType(required=True),
    "value": FloatType(required=True),
    "operator": StringType(required=True),
    "timeframe": timeframe_schema(required=False),
}), required=True, min_length=1)

position_management_schema = DictType({
    "stake_amount": FloatType(required=True),
    "max_drawdown_percent": FloatType(required=True),
    "stop_loss_percent": FloatType(required=True),
    "trailing_stop": BooleanType(required=True),
    "trailing_stop_percent": FloatType(required=True),
})

root_schema = DictType({
    "app": app_schema,
    "exchanges": exchanges_schema,
    "data_collection": data_collection_schema,
    "entry_conditions": entry_conditions_schema,
    "exit_conditions": exit_conditions_schema,
    "position_management": position_management_schema
}, required=True)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return root_schema.validate(config, "")
    except ConfigError as e:
        raise ConfigError(f"Configuration validation failed: {str(e)}")
    except Exception as e:
        raise ConfigError(f"Unexpected error during configuration validation: {str(e)}")

def validate_app_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return app_schema.validate(config, "app")

def validate_exchanges_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return exchanges_schema.validate(config, "exchanges")

def validate_data_collection_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return data_collection_schema.validate(config, "data_collection")

def validate_entry_conditions_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return entry_conditions_schema.validate(config, "entry_conditions")

def validate_exit_conditions_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return exit_conditions_schema.validate(config, "exit_conditions")

def validate_position_management_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return position_management_schema.validate(config, "position_management")

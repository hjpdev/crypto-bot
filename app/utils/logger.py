import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format string. If None, a default format is used.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_default_logger():
    log_level = os.getenv("LOG_LEVEL", "INFO")

    return setup_logger(
        "crypto_bot",
        log_level=log_level,
    )


logger = get_default_logger()

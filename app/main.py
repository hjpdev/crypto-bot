import argparse
import sys
from pathlib import Path

from app.config import Config
from app.core.database import db
from app.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--setup-db", action="store_true", help="Set up the database and exit")

    return parser.parse_args()


def setup_database():
    logger.info("Setting up database...")
    db.create_database()
    logger.info("Database setup complete.")


def main():
    args = parse_args()

    if args.debug:
        import logging

        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    try:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1

        logger.info(f"Loading configuration from {config_path}")
        config = Config(str(config_path))
        logger.info(f"Configuration loaded: {config.__repr__()}")

        if args.setup_db:
            setup_database()
            return 0

        # Add additional initialization here
        # ...

        logger.info("Crypto Trading Bot started")

        # Add main application loop here
        # ...

        return 0

    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt")
        return 0

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

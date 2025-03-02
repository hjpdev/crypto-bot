import argparse
import asyncio
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


async def run_trading_loop(config):
    """
    Main trading loop that keeps the application running.
    This is where you would implement the core trading logic.
    """
    logger.info("Starting main trading loop...")

    try:
        while True:
            # This is where you would implement your trading logic
            # For example:
            # - Fetch market data
            # - Run strategy calculations
            # - Execute trades if signals are generated
            # - Log performance metrics

            # For now, we just sleep to keep the loop running
            logger.debug("Trading loop cycle...")
            await asyncio.sleep(60)  # Sleep for 60 seconds between cycles
    except asyncio.CancelledError:
        logger.info("Trading loop was cancelled")
    except Exception as e:
        logger.exception(f"Error in trading loop: {e}")
        raise


async def main():
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
        config = Config().get_instance()
        logger.info(f"Configuration loaded: {config.__repr__()}")

        if args.setup_db:
            setup_database()
            return 0

        # Add additional initialization here
        # ...

        logger.info("Crypto Trading Bot started")

        # Run the main application loop and keep it running until interrupted
        trading_task = asyncio.create_task(run_trading_loop(config))

        # Wait for the trading loop to complete (on error or cancellation)
        await trading_task

        return 0

    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt")
        return 0

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    # This ensures we still return the right exit code to the OS
    if exit_code:
        import sys

        sys.exit(exit_code)

import argparse
import asyncio
from pathlib import Path
import time
from typing import Dict, Any
from decimal import Decimal

from app.config import Config
from app.core.database import db
from app.utils.logger import logger
from app.services.exchange_service import ExchangeService
from app.services.market_analyzer import MarketAnalyzer
from app.services.indicator_service import IndicatorService
from app.services.market_filter import MarketFilter
from app.services.portfolio_manager import PortfolioManager
from app.services.risk_manager import RiskManager
from app.strategies.momentum_strategy import MomentumStrategy


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
    parser.add_argument(
        "--dry-run", action="store_true", help="Run in dry-run mode (no actual trades)"
    )

    return parser.parse_args()


def setup_database():
    logger.info("Setting up database...")
    db.create_database()
    logger.info("Database setup complete.")


async def initialize_services(config: Config, dry_run: bool = False) -> Dict[str, Any]:
    """
    Initialize all required services.

    Args:
        config: Application configuration
        dry_run: Whether to run in dry-run mode

    Returns:
        Dictionary containing all initialized services
    """
    logger.info("Initializing services...")

    # Initialize exchange service
    exchange_config = config.get_nested("exchange", {})
    exchange_id = exchange_config.get("id", "kraken")

    # Get API credentials (if available)
    api_key = exchange_config.get("api_key")
    api_secret = exchange_config.get("secret")

    logger.info(f"Initializing exchange service for {exchange_id}")
    exchange_service = ExchangeService(
        exchange_id=exchange_id,
        api_key=api_key,
        secret=api_secret,
        sandbox=dry_run,  # Use sandbox mode for dry runs
        enableRateLimit=True,
    )

    # Initialize indicator service
    indicator_service = IndicatorService()

    # Initialize market analyzer
    market_analyzer = MarketAnalyzer(indicator_service)

    # Initialize market filter
    market_filter = MarketFilter(exchange_service)

    # Initialize risk manager
    risk_manager = RiskManager(config)

    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(
        exchange_service=exchange_service, risk_manager=risk_manager, config=config, dry_run=dry_run
    )

    # Initialize strategy (use momentum strategy by default)
    strategy_config = config.get_nested("strategies.momentum", {})
    strategy = MomentumStrategy(strategy_config)

    # Return all services in a dictionary
    return {
        "exchange_service": exchange_service,
        "indicator_service": indicator_service,
        "market_analyzer": market_analyzer,
        "market_filter": market_filter,
        "risk_manager": risk_manager,
        "portfolio_manager": portfolio_manager,
        "strategy": strategy,
    }


async def run_trading_loop(config: Config, dry_run: bool = False):
    """
    Main trading loop that keeps the application running.

    Args:
        config: Application configuration
        dry_run: Whether to run in dry-run mode (no actual trades)
    """
    logger.info(f"Starting trading loop in {'dry-run' if dry_run else 'live'} mode...")

    # Initialize all required services
    services = await initialize_services(config, dry_run)

    exchange_service = services["exchange_service"]
    market_filter = services["market_filter"]
    market_analyzer = services["market_analyzer"]
    portfolio_manager = services["portfolio_manager"]
    strategy = services["strategy"]

    # Get trading configuration
    trading_config = config.get_nested("trading", {})
    cycle_interval = trading_config.get("cycle_interval", 60)  # Seconds between cycles
    market_filter_config = trading_config.get("market_filter", {})

    # Get symbols to trade from config, or filter using market filter
    symbols_config = trading_config.get("symbols", [])
    if symbols_config:
        symbols = symbols_config
        logger.info(f"Using {len(symbols)} symbols from configuration")
    else:
        # Filter markets according to criteria
        logger.info("No specific symbols configured, applying market filter...")
        all_symbols = exchange_service.exchange.symbols
        symbols = market_filter.apply_all_filters(all_symbols, market_filter_config)
        logger.info(f"Market filter selected {len(symbols)} symbols for trading")

    if not symbols:
        logger.error(
            "No symbols available for trading. Check your configuration or market filter settings."
        )
        return

    # Get timeframes to analyze
    timeframes = trading_config.get("timeframes", ["1h"])
    primary_timeframe = timeframes[0] if timeframes else "1h"

    try:
        while True:
            cycle_start_time = time.time()
            logger.debug(f"Starting trading cycle with {len(symbols)} symbols")

            # Process each symbol
            for symbol in symbols:
                try:
                    logger.debug(f"Processing {symbol}")

                    # Fetch market data for each timeframe
                    market_data = {}
                    for timeframe in timeframes:
                        ohlcv_data = await asyncio.to_thread(
                            exchange_service.fetch_ohlcv, symbol, timeframe, limit=100
                        )
                        market_data[timeframe] = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "ohlcv_data": ohlcv_data,
                        }

                    # Skip if no data available for primary timeframe
                    if (
                        primary_timeframe not in market_data
                        or market_data[primary_timeframe]["ohlcv_data"].empty
                    ):
                        logger.warning(
                            f"No data available for {symbol} on {primary_timeframe} timeframe"
                        )
                        continue

                    # Analyze market data
                    primary_data = market_data[primary_timeframe]
                    regime = market_analyzer.detect_market_regime(primary_data["ohlcv_data"])
                    logger.info(f"{symbol} market regime: {regime}")

                    # Check if we should enter a new position
                    should_enter, entry_info = strategy.should_enter_position(symbol, primary_data)

                    if should_enter:
                        logger.info(f"Strategy suggests entering position for {symbol}")
                        # Calculate account balance (in a real scenario, this would come from the exchange)
                        account_balance = Decimal("10000")  # Example balance

                        # Enter the position
                        success = await portfolio_manager.enter_position(
                            symbol=symbol,
                            strategy=strategy,
                            market_data=primary_data,
                            account_balance=account_balance,
                        )

                        if success:
                            logger.info(f"Successfully entered position for {symbol}")
                        else:
                            logger.warning(f"Failed to enter position for {symbol}")

                    # Check if we should exit any existing positions
                    open_positions = portfolio_manager.get_open_positions(symbol)

                    for position in open_positions:
                        should_exit, exit_info = strategy.should_exit_position(
                            position, primary_data
                        )

                        if should_exit:
                            logger.info(
                                f"Strategy suggests exiting position {position.id} for {symbol}"
                            )

                            success = await portfolio_manager.exit_position(
                                position_id=position.id, market_data=primary_data
                            )

                            if success:
                                logger.info(
                                    f"Successfully exited position {position.id} for {symbol}"
                                )
                            else:
                                logger.warning(
                                    f"Failed to exit position {position.id} for {symbol}"
                                )

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    # Continue with next symbol

            # Calculate sleep time to maintain consistent cycle interval
            elapsed = time.time() - cycle_start_time
            sleep_time = max(0, cycle_interval - elapsed)

            if sleep_time > 0:
                logger.debug(f"Cycle completed in {elapsed:.2f}s, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
            else:
                logger.warning(
                    f"Cycle took {elapsed:.2f}s, exceeding interval of {cycle_interval}s"
                )
                # Small sleep to prevent CPU spinning if cycles consistently take too long
                await asyncio.sleep(1)

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

        # Run the main application loop and keep it running until interrupted
        trading_task = asyncio.create_task(run_trading_loop(config, dry_run=args.dry_run))

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

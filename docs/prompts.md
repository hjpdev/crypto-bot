# Phase 1: Project Setup and Foundation
## Prompt 1: Project Structure and Initial Setup
Create the initial project structure for a cryptocurrency trading bot according to this directory layout:

crypto-bot/
├── alembic/                      # Database migration files
│   └── versions/
├── app/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config_loader.py      # YAML config loader
│   ├── core/
│   │   ├── __init__.py
│   │   ├── database.py           # SQLAlchemy setup
│   │   └── exceptions.py         # Custom exceptions
│   ├── models/
│   │   ├── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   ├── strategies/
│   │   ├── __init__.py
│   ├── utils/
│       ├── __init__.py
│       └── logger.py
├── tests/                        
│   ├── __init__.py
├── scripts/
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md

For each directory, create empty __init__.py files. Create a comprehensive .gitignore file suitable for Python projects, ensuring it excludes virtual environments, cached files, and sensitive configuration.

For requirements.txt, include these dependencies with specific versions:
- python = ">=3.11.5"
- sqlalchemy = "^2.0.0"
- alembic = "^1.12.0"
- ccxt = "^4.1.87"
- pyyaml = "^6.0"
- python-dotenv = "^1.0.0"
- pandas = "^2.2.1"
- pandas-ta = "^0.3.14b0"
- numpy = "^1.24.1"
- pytest = "^7.4.0"
- black = "^23.9.1"
- flake8 = "^6.1.0"
- psycopg2-binary = "^2.9.7"

Create a basic setup.py file that uses setuptools to make the package installable.

Create a simple README.md with the project title "Crypto Trading Bot", a brief description based on the project being a cryptocurrency trading bot that collects market data for training future machine learning models, and basic sections for Installation, Usage, and Features.

Make sure all files are properly structured and imports will work correctly when the project grows.

## Prompt 2: Development Environment Configuration

Now let's create configuration files for code quality tools and set up the development environment:

1. Create a pyproject.toml file with configuration for Black with these settings:
   - line-length = 88
   - target-version = ["py311"]
   - include directories: app, tests, scripts
   - exclude directories: .venv, build, dist

2. Create a .flake8 configuration file with:
   - max-line-length = 88 (to match Black)
   - exclude patterns for standard directories like venv, build, etc.
   - select = E,F,W,C (errors, warnings, complexity)
   - ignore = E203, W503 (to be compatible with Black formatting)

3. Create a simple .env.example file (without actual secrets) with placeholders for:
   - DATABASE_URL=postgresql://username:password@localhost:5432/crypto_bot
   - LOG_LEVEL=INFO
   - EXCHANGE_API_KEY=your_api_key_here
   - EXCHANGE_SECRET=your_secret_here

4. Create a config.yaml.example file based on the specification with key sections:
   - exchange configuration
   - cryptocurrency filtering
   - momentum strategy parameters
   - risk management settings
   - application settings
   - but omit actual sensitive values

Make sure the configuration files are in the appropriate locations and follow best practices for a Python project.

## Prompt 3: Docker Setup for Database - Note, skipped this.

Now, let's create Docker configuration files for the PostgreSQL database service:

1. Create a docker-compose.yml file with:
   - PostgreSQL service:
     - Use postgres:14 image
     - Set environment variables for:
       - POSTGRES_USER=crypto_user
       - POSTGRES_PASSWORD=crypto_password
       - POSTGRES_DB=crypto_bot
     - Map port 5432 to localhost:5432
     - Set up a volume for persistent data storage: ./data/postgres:/var/lib/postgresql/data
     - Configure healthcheck using pg_isready
   - Add proper networking configuration
   - Include comments explaining key configuration options

2. Create a directory scripts/db with:
   - init-db.sql script that sets up initial permissions and extensions like:
     - CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
     - User permissions for the crypto_user
   - A backup-db.sh shell script with functionality to create a timestamped backup of the database

3. Update the .gitignore file to exclude the data/ directory where database files will be stored

Make sure the Docker configuration is properly set up to allow the application to connect to the database and that data persistence is correctly configured.

## Prompt 4: Docker Setup for Application

Now, let's extend our Docker configuration to include the application service:

1. Create a Dockerfile for the application with:
   - Use python:3.11.5-slim as base image
   - Set up a non-root user 'appuser' for security
   - Install system dependencies needed for psycopg2 and other libraries
   - Create and activate virtual environment
   - Copy requirements.txt first (for better caching)
   - Install dependencies
   - Copy the application code
   - Set proper permissions for the application directory
   - Set the PYTHONPATH
   - Configure the entry point to run the application

2. Update the docker-compose.yml file to:
   - Add the application service that:
     - Builds from the Dockerfile
     - Depends on the database service
     - Maps the local codebase as a volume for development
     - Sets environment variables via .env file
     - Configures restart policy appropriately
     - Waits for the database to be healthy before starting

3. Create a docker-compose.dev.yml file with development-specific overrides:
   - Set development-specific environment variables
   - Mount the source code as a volume to allow live code changes
   - Configure any development-specific ports or settings

4. Create a .dockerignore file to exclude unnecessary files from the Docker context:
   - .git directory
   - __pycache__ directories
   - *.pyc files
   - data directory
   - local environment files
   - any other files not needed in the container

Ensure the Docker setup is secure, follows best practices, and is suitable for both development and production environments.

## Prompt 5: Configuration Management Implementation

Let's implement the configuration management system for the application:

1. Implement app/config/config_loader.py with:
   - A ConfigLoader class that:
     - Loads configuration from a YAML file
     - Supports environment variable overrides
     - Validates the configuration against a schema
     - Handles configuration reloading
     - Implements a singleton pattern for global config access
   - Functions for:
     - get_config(): Returns the current configuration
     - reload_config(): Reloads the configuration from disk
     - get_nested_config(path): Gets a nested configuration value using dot notation

2. Create app/config/validation.py with:
   - Functions to validate different sections of the configuration
   - Schema definitions for each configuration section
   - Type checking and validation for critical parameters

3. Create app/config/__init__.py that:
   - Exports the main functions and classes
   - Initializes the configuration system when imported

4. Create tests/test_config.py with:
   - Unit tests for the ConfigLoader class
   - Tests for loading configuration from files
   - Tests for environment variable overrides
   - Tests for validation functionality
   - Tests for handling invalid configurations

5. Create a sample_config.yaml in the tests directory with test configuration data

Use type hints throughout the code and include comprehensive docstrings. Make sure the configuration system properly handles errors and edge cases.

## Prompt 6: Database Connection Management

Let's implement the database connection management using SQLAlchemy:

1. Implement app/core/database.py with:
   - A Database class that:
     - Creates and manages the SQLAlchemy engine
     - Provides session management functionality
     - Handles connection pooling
     - Implements reconnection logic for handling connection issues
   - Functions for:
     - get_engine(): Returns the current SQLAlchemy engine
     - get_session(): Returns a new session
     - create_all_tables(): Creates all tables if they don't exist
     - session_scope(): Context manager for session handling

2. Create app/core/exceptions.py with:
   - Custom exception classes for:
     - DatabaseConnectionError: For connection issues
     - ConfigurationError: For configuration problems
     - ExchangeConnectionError: For exchange API issues
     - ValidationError: For data validation failures

3. Update app/core/__init__.py to export the main functions and classes

4. Create tests/test_database.py with:
   - Unit tests for the Database class
   - Tests for session management
   - Tests for connection error handling
   - Tests that use an in-memory SQLite database for testing

Use dependency injection patterns where appropriate and ensure proper error handling and resource cleanup. Include comprehensive logging for database operations. Use type hints and docstrings throughout.

## Prompt 7: Alembic Migration Setup

Let's set up Alembic for database migrations:

1. Create an alembic.ini file in the project root with:
   - Configuration for PostgreSQL
   - Script location pointing to alembic directory
   - Template configuration
   - Logging configuration

2. Set up the alembic environment:
   - Create alembic/env.py that:
     - Imports SQLAlchemy models
     - Sets up the database connection from configuration
     - Configures the migration context
     - Handles both online and offline migrations
   - Include support for environment variable substitution

3. Create a script file at scripts/create_migration.py that:
   - Takes a migration message as a command-line argument
   - Runs alembic to generate a new migration
   - Sets up proper logging

4. Create a script file at scripts/apply_migrations.py that:
   - Runs all pending migrations
   - Handles errors and provides clear output
   - Supports a --dry-run flag

5. Create tests/test_migrations.py with:
   - Tests to verify migration setup
   - Tests for generating and applying migrations
   - Tests using a temporary SQLite database

Make sure the Alembic setup integrates properly with the application's database configuration and can be used in both development and production environments.

# Phase 2: Core Data Models and Services

## Prompt 8: Base Model Implementation

Let's implement the base SQLAlchemy model and common model utilities:

1. Create app/models/base.py with:
   - A declarative Base class for SQLAlchemy ORM
   - A BaseModel class that inherits from Base and includes:
     - id column as primary key (UUID type with server default)
     - created_at column (timestamp with auto-creation)
     - updated_at column (timestamp with auto-update)
     - Common methods like to_dict(), from_dict(), etc.
     - Any common utility methods needed across models
     - Meta configuration for SQLAlchemy

2. Create app/models/mixins.py with:
   - TimestampMixin: For created_at and updated_at functionality
   - DictMixin: For dictionary conversion methods
   - Any other reusable mixins for model functionality

3. Update app/models/__init__.py to:
   - Export the Base class
   - Export common model utilities
   - Define a function get_all_models() that returns all model classes

4. Create tests/test_base_model.py with:
   - Tests for the BaseModel functionality
   - Tests for mixins
   - Tests for timestamp functionality
   - Tests for dictionary conversion

Ensure proper type hinting throughout the code and comprehensive docstrings. Make the base model flexible and reusable for all the specific models we'll create.

## Prompt 9: Cryptocurrency Model

Let's implement the Cryptocurrency model:

1. Create app/models/cryptocurrency.py with:
   - A Cryptocurrency class that inherits from BaseModel with:
     - symbol: String column (e.g., "BTC/USD") with unique constraint
     - name: String column (e.g., "Bitcoin")
     - is_active: Boolean column with default True
     - market_cap: Numeric column, nullable
     - avg_daily_volume: Numeric column, nullable
     - exchange_specific_id: String column, nullable
     - listing_date: Date column, nullable
     - Relationships to:
       - market_data (one-to-many)
       - simulated_trades (one-to-many)
       - market_snapshots (one-to-many)
     - Class methods for:
       - get_by_symbol(symbol): Fetches cryptocurrency by symbol
       - get_active(): Returns all active cryptocurrencies
       - update_market_data(data): Updates market cap and volume

2. Update app/models/__init__.py to export the Cryptocurrency class

3. Create tests/test_cryptocurrency_model.py with:
   - Tests for creating, retrieving, updating, and deleting cryptocurrencies
   - Tests for relationship functionality
   - Tests for custom methods
   - Tests for constraints (e.g., unique symbol)

Include validation logic for the cryptocurrency fields and proper error handling. Use type hints and comprehensive docstrings. Make sure the relationships are properly defined for future models.

## Prompt 10: Market Data Model

Let's implement the MarketData model for storing OHLCV data:

1. Create app/models/market_data.py with:
   - A MarketData class that inherits from BaseModel with:
     - cryptocurrency_id: Foreign key to Cryptocurrency
     - timestamp: DateTime column with index
     - open: Numeric column
     - high: Numeric column
     - low: Numeric column
     - close: Numeric column
     - volume: Numeric column
     - indicators: JSON column (to store calculated indicators)
     - Relationship to Cryptocurrency (many-to-one)
     - Unique constraint on (cryptocurrency_id, timestamp)
     - Class methods for:
       - get_latest(cryptocurrency_id, limit): Gets most recent data
       - get_range(cryptocurrency_id, start, end): Gets data in time range
       - get_for_indicators(cryptocurrency_id, periods): Gets enough data to calculate indicators

2. Update app/models/__init__.py to export the MarketData class

3. Create tests/test_market_data_model.py with:
   - Tests for creating, retrieving, updating, and deleting market data
   - Tests for relationship with Cryptocurrency
   - Tests for custom methods
   - Tests for constraints and indexes
   - Tests for storing and retrieving indicator data

Include validation for OHLCV data and handle time zone considerations properly. Make sure the JSON storage for indicators is properly implemented with serialization/deserialization methods. Use type hints and comprehensive docstrings.

## Prompt 11: SimulatedTrades and PartialExits

Let's implement the SimulatedTrades and PartialExits models:

1. Create app/models/trades.py with:
   - A SimulatedTrade class that inherits from BaseModel with:
     - cryptocurrency_id: Foreign key to Cryptocurrency
     - entry_timestamp: DateTime column
     - entry_price: Numeric column
     - position_size: Numeric column
     - position_type: Enum column (LONG/SHORT)
     - stop_loss_price: Numeric column
     - take_profit_price: Numeric column
     - status: Enum column (OPEN/PARTIALLY_CLOSED/CLOSED)
     - exit_timestamp: DateTime column, nullable
     - exit_price: Numeric column, nullable
     - profit_loss: Numeric column, nullable
     - profit_loss_percentage: Numeric column, nullable
     - strategy_used: String column
     - notes: Text column, nullable
     - Relationship to Cryptocurrency (many-to-one)
     - Relationship to PartialExits (one-to-many)
     - Methods for:
       - calculate_current_pl(current_price): Calculates current P/L
       - apply_exit(price, timestamp, full_exit=False): Records exit
       - should_exit(current_price): Determines if position should exit

   - A PartialExit class that inherits from BaseModel with:
     - trade_id: Foreign key to SimulatedTrade
     - exit_timestamp: DateTime column
     - exit_price: Numeric column
     - exit_percentage: Numeric column
     - profit_loss: Numeric column
     - trailing_stop_activated: Boolean column
     - Relationship to SimulatedTrade (many-to-one)

2. Update app/models/__init__.py to export both classes

3. Create tests/test_trade_models.py with:
   - Tests for creating and managing trade records
   - Tests for partial exit functionality
   - Tests for P/L calculation
   - Tests for status transitions
   - Tests for relationships between models

Include validation logic for trade data and implement proper error handling. Make sure the relationships between models are correctly defined. Use type hints and comprehensive docstrings throughout.

## Prompt 12: Configuration and Performance Models

Let's implement the ConfigurationHistory and PerformanceMetrics models:

1. Create app/models/system.py with:
   - A ConfigurationHistory class that inherits from BaseModel with:
     - timestamp: DateTime column (when config was saved)
     - configuration: JSON column (storing entire config)
     - run_id: String column (unique identifier for each bot run)
     - notes: Text column, nullable
     - Class methods for:
       - save_current_config(config, run_id, notes): Saves config snapshot
       - get_latest(): Gets most recent configuration
       - get_by_run_id(run_id): Gets all configs for a run

   - A PerformanceMetrics class that inherits from BaseModel with:
     - timestamp: DateTime column
     - run_id: String column (matching ConfigurationHistory)
     - total_trades: Integer column
     - winning_trades: Integer column
     - losing_trades: Integer column
     - win_rate: Numeric column
     - average_profit: Numeric column
     - average_loss: Numeric column
     - profit_factor: Numeric column
     - max_drawdown: Numeric column
     - sharpe_ratio: Numeric column
     - total_profit_loss: Numeric column
     - Class methods for:
       - record_current_performance(metrics_dict, run_id): Saves metrics
       - get_latest(): Gets most recent metrics
       - get_by_run_id(run_id): Gets all metrics for a run

2. Update app/models/__init__.py to export both classes

3. Create tests/test_system_models.py with:
   - Tests for saving and retrieving configurations
   - Tests for performance metrics functionality
   - Tests for JSON serialization/deserialization
   - Tests for run_id based querying

Ensure proper handling of JSON data and implement validation for the metrics. Include methods to calculate performance metrics from trade data. Use type hints and comprehensive docstrings throughout.

## Prompt 13: Market Snapshots Model

Let's implement the MarketSnapshots model for storing periodic market condition data:

1. Create app/models/snapshots.py with:
   - A MarketSnapshot class that inherits from BaseModel with:
     - cryptocurrency_id: Foreign key to Cryptocurrency
     - timestamp: DateTime column with index
     - ohlcv: JSON column (storing OHLCV data for multiple timeframes)
     - indicators: JSON column (storing calculated indicators)
     - order_book: JSON column (storing order book snapshot)
     - trading_volume: Numeric column
     - market_sentiment: Numeric column, nullable
     - correlation_btc: Numeric column, nullable
     - Relationship to Cryptocurrency (many-to-one)
     - Unique constraint on (cryptocurrency_id, timestamp)
     - Class methods for:
       - get_latest(cryptocurrency_id): Gets most recent snapshot
       - get_range(cryptocurrency_id, start, end): Gets snapshots in time range
       - get_with_specific_indicator(cryptocurrency_id, indicator): Filter by indicator presence

2. Update app/models/__init__.py to export the MarketSnapshot class

3. Create tests/test_snapshot_model.py with:
   - Tests for creating and retrieving snapshots
   - Tests for relationship with Cryptocurrency
   - Tests for JSON field functionality
   - Tests for query methods
   - Tests for handling time series data

Implement proper serialization/deserialization for JSON fields and include validation for snapshot data. Handle time zone issues consistently. Use type hints and comprehensive docstrings throughout the code.

## Prompt 14: Exchange Service - Basic Setup

Let's implement the basic exchange service using CCXT:

1. Create app/services/exchange_service.py with:
   - An ExchangeService class that:
     - Initializes and configures CCXT exchange connection
     - Handles authentication if API keys are provided
     - Implements rate limiting and backoff logic
     - Provides basic error handling for API calls
     - Includes methods for:
       - get_exchange(): Returns configured exchange instance
       - fetch_markets(): Gets available markets
       - get_ticker(symbol): Gets current ticker data
       - fetch_ohlcv(symbol, timeframe, since, limit): Gets OHLCV data
       - get_order_book(symbol, limit): Gets order book
     - Implements caching for frequently accessed data

2. Create app/services/exchange_rate_limiter.py with:
   - A RateLimiter class that:
     - Tracks API call frequency
     - Implements exponential backoff for rate limit errors
     - Provides decorators for rate-limited methods

3. Update app/services/__init__.py to export the service classes

4. Create tests/test_exchange_service.py with:
   - Tests for initialization and configuration
   - Tests for API methods with mocked responses
   - Tests for error handling
   - Tests for rate limiting functionality
   - Tests for backoff behavior

Use dependency injection for configuration and make the service configurable. Implement comprehensive error handling and logging. Use type hints and detailed docstrings. Include timeout handling and connection recovery.

## Prompt 15: Exchange Service - Data Fetching

Let's extend the exchange service with comprehensive data fetching capabilities:

1. Extend app/services/exchange_service.py with:
   - New methods in ExchangeService class:
     - fetch_historical_ohlcv(symbol, timeframe, start_time, end_time): Fetches historical data with pagination
     - fetch_multiple_symbols(symbols, timeframe, since, limit): Fetches data for multiple symbols efficiently
     - fetch_order_book_snapshot(symbol, depth): Gets a snapshot of the order book
     - fetch_recent_trades(symbol, limit): Gets recent trade data
     - get_ticker_batch(symbols): Gets ticker data for multiple symbols
     - fetch_funding_rate(symbol): Gets funding rate for perpetual contracts
     - get_symbol_metadata(symbol): Gets details about trading pair
   - A retrying mechanism with configurable parameters
   - Error normalization to handle exchange-specific errors
   - Support for multiple timeframes with conversion utilities

2. Create app/services/data_normalization.py with:
   - Functions to normalize exchange data:
     - normalize_ohlcv(exchange_data): Standardizes OHLCV data
     - normalize_ticker(exchange_data): Standardizes ticker data
     - normalize_order_book(exchange_data): Standardizes order book data
     - standardize_symbol(exchange, symbol): Creates consistent symbol format

3. Update tests/test_exchange_service.py with:
   - Tests for new data fetching methods
   - Tests for data normalization
   - Tests for error handling in complex scenarios
   - Tests for pagination handling

4. Create a simple script in scripts/fetch_market_data.py that:
   - Uses the exchange service to fetch and display market data
   - Demonstrates proper usage of the service
   - Handles common errors gracefully

Ensure proper error handling, particularly for network issues and rate limiting. Implement smart retry logic with exponential backoff. Use proper type hints and comprehensive docstrings.

## Prompt 16: Exchange Service - Cryptocurrency Filtering

Let's implement the cryptocurrency filtering functionality in the exchange service:

1. Create app/services/market_filter.py with:
   - A MarketFilter class that:
     - Filters cryptocurrencies based on configurable criteria
     - Methods for:
       - filter_by_market_cap(symbols, min_cap): Filters by market cap
       - filter_by_volume(symbols, min_volume): Filters by trading volume
       - filter_by_spread(symbols, max_spread): Filters by bid-ask spread
       - filter_by_volatility(symbols, min_vol, max_vol): Filters by volatility
       - filter_by_allowed_quote(symbols, allowed_quotes): Filters by quote currency
       - apply_all_filters(symbols, config): Applies all filters based on config
     - Utility methods to fetch data needed for filtering

2. Update app/services/exchange_service.py to:
   - Add integration with MarketFilter
   - Add a method get_filtered_symbols() that returns symbols passing all filters
   - Implement caching for filter results with configurable timeout
   - Add logging for filtered symbols

3. Create tests/test_market_filter.py with:
   - Tests for individual filter functions
   - Tests for combined filtering
   - Tests with mock market data
   - Tests for edge cases (empty lists, all filtered out, etc.)

4. Create a utility script in scripts/list_filtered_markets.py that:
   - Runs the market filter on current exchange data
   - Outputs filtered symbols with their properties
   - Shows why specific symbols were filtered out

Implement proper error handling for external data sources. Include logging of filtering decisions. Use type hints and comprehensive docstrings throughout the code.

## Prompt 17: Indicator Service - Basic Indicators

Let's implement the basic technical indicators service:

1. Create app/services/indicator_service.py with:
   - An IndicatorService class that:
     - Calculates various technical indicators
     - Has methods for:
       - calculate_rsi(dataframe, period=14): Calculates RSI
       - calculate_macd(dataframe, fast=12, slow=26, signal=9): Calculates MACD
       - calculate_ema(dataframe, period): Calculates Exponential Moving Average
       - calculate_sma(dataframe, period): Calculates Simple Moving Average
       - calculate_roc(dataframe, period): Calculates Rate of Change
       - batch_calculate(dataframe, indicators_config): Calculates multiple indicators
     - Handles both pandas DataFrames and lists/arrays as input
     - Includes validation for input data

2. Create app/services/data_preparation.py with:
   - Functions for:
     - ohlcv_to_dataframe(ohlcv_data): Converts OHLCV data to DataFrame
     - prepare_for_indicators(dataframe): Ensures data is ready for indicators
     - resample_ohlcv(dataframe, timeframe): Resamples to different timeframe
     - validate_ohlcv_data(data): Validates OHLCV data structure

3. Update app/services/__init__.py to export the new classes

4. Create tests/test_indicator_service.py with:
   - Tests for each indicator calculation
   - Tests with known inputs and expected outputs
   - Tests for handling invalid data
   - Tests for batch calculation
   - Tests for different input formats

5. Create tests/test_data_preparation.py with:
   - Tests for data conversion functions
   - Tests for validation functions
   - Tests for resampling functionality

Use pandas-ta efficiently and implement proper error handling. Include validation for input data and parameters. Use type hints and comprehensive docstrings throughout the code.

## Prompt 18: Indicator Service - Advanced Indicators

Let's expand the indicator service with more advanced technical indicators:

1. Enhance app/services/indicator_service.py with additional methods:
   - calculate_bollinger_bands(dataframe, period=20, std_dev=2): Calculates Bollinger Bands
   - calculate_atr(dataframe, period=14): Calculates Average True Range
   - calculate_adx(dataframe, period=14): Calculates Average Directional Index
   - calculate_obv(dataframe): Calculates On-Balance Volume
   - calculate_vwap(dataframe): Calculates Volume-weighted Average Price
   - calculate_support_resistance(dataframe, lookback=14): Identifies support/resistance levels
   - calculate_multi_timeframe(dataframes_dict, indicator_func, **params): Calculates indicator across timeframes
   - detect_divergence(price_data, indicator_data, window=10): Detects bullish/bearish divergence

2. Create app/services/indicator_utils.py with:
   - Helper functions for advanced calculations:
     - find_swing_highs(data, window): Identifies swing high points
     - find_swing_lows(data, window): Identifies swing low points
     - identify_trend(data, window): Identifies trend direction
     - smooth_data(data, method='ema', period=5): Smooths data for better analysis
     - normalize_indicator(data, method='minmax'): Normalizes indicator values

3. Update tests/test_indicator_service.py with:
   - Tests for new indicator calculations
   - Tests for multi-timeframe analysis
   - Tests for divergence detection
   - Tests for support/resistance identification

4. Create tests/test_indicator_utils.py with:
   - Tests for utility functions
   - Tests with known patterns for swing detection
   - Tests for trend identification
   - Tests for data smoothing and normalization

5. Create a script in scripts/calculate_indicators.py that:
   - Fetches market data for a specified symbol
   - Calculates a comprehensive set of indicators
   - Displays or plots the results
   - Demonstrates proper usage of the indicator service

Use vectorized operations for efficiency. Implement proper error handling, especially for edge cases. Use type hints and comprehensive docstrings throughout the code.

## Phase 3: Strategy and Risk Management

## Prompt 19: Strategy Framework

Let's implement the strategy framework for the trading bot:

1. Create app/strategies/base_strategy.py with:
   - An abstract BaseStrategy class that:
     - Defines the interface for all trading strategies
     - Has abstract methods:
       - generate_signals(market_data): Generates buy/sell signals
       - should_enter_position(symbol, market_data): Determines if should enter
       - should_exit_position(position, market_data): Determines if should exit
       - calculate_position_size(symbol, account_balance): Calculates position size
       - get_stop_loss(symbol, entry_price, market_data): Determines stop loss level
       - get_take_profit(symbol, entry_price, market_data): Determines take profit levels
     - Has common utility methods useful for all strategies
     - Includes configuration handling

2. Create app/strategies/strategy_utils.py with:
   - Utility functions for strategy implementation:
     - calculate_risk_reward_ratio(entry, stop_loss, take_profit): Calculates R:R ratio
     - validate_signal(signal, market_data, min_confidence): Validates trading signal
     - calculate_signal_strength(indicators): Quantifies signal strength
     - combine_signal_sources(signals, weights): Combines multiple signal sources

3. Create app/models/signals.py with:
   - A Signal class that:
     - Represents a trading signal
     - Includes signal type (BUY/SELL), symbol, timestamp, confidence, etc.
     - Includes the source of the signal and supporting indicators

4. Update app/strategies/__init__.py to export the classes and functions

5. Create tests/test_base_strategy.py with:
   - Tests for strategy interface
   - Tests for utility methods
   - Tests with mock market data

6. Create tests/test_strategy_utils.py with:
   - Tests for utility functions
   - Tests for signal validation and combination

Use dependency injection for services and configuration. Make the framework flexible and extensible. Use type hints and comprehensive docstrings throughout the code.

## Prompt 20: Momentum Strategy Implementation

Let's implement the momentum strategy:

1. Create app/strategies/momentum_strategy.py with:
   - A MomentumStrategy class that inherits from BaseStrategy:
     - Implements all abstract methods from BaseStrategy
     - Includes methods for:
       - _check_rsi_condition(data): Checks for RSI oversold/overbought
       - _check_macd_condition(data): Checks for MACD crossover/divergence
       - _check_volume_confirmation(data): Confirms with volume
       - _check_trend_alignment(data): Verifies trend direction
       - _calculate_signal_confidence(conditions): Quantifies signal confidence
       - _apply_filters(signal, market_data): Applies additional filters
     - Uses the indicator service for calculations
     - Configurable parameters via config file

2. Update the app/strategies/__init__.py to export the MomentumStrategy class

3. Create tests/test_momentum_strategy.py with:
   - Tests for signal generation
   - Tests for entry/exit conditions
   - Tests for position sizing
   - Tests for stop loss and take profit calculation
   - Tests with different market scenarios
   - Tests with different configuration parameters

4. Create a script in scripts/test_momentum_strategy.py that:
   - Initializes the momentum strategy with sample configuration
   - Fetches historical market data for a few symbols
   - Generates and displays trading signals
   - Visualizes the signals on a price chart
   - Shows performance metrics for the strategy

Implement the strategy with clear, understandable logic. Make parameters configurable. Include extensive logging of decision-making. Use type hints and comprehensive docstrings throughout the code.

## Prompt 21: Risk Management Implementation

Let's implement the risk management system:

1. Create app/services/risk_manager.py with:
   - A RiskManager class that:
     - Handles position sizing based on risk parameters
     - Determines stop loss levels (fixed, volatility-based, or indicator-based)
     - Calculates take profit levels with partial exit points
     - Implements trailing stop logic
     - Enforces maximum position limits
     - Methods for:
       - calculate_position_size(symbol, entry, stop_loss, risk_per_trade, balance): Calculates size
       - calculate_stop_loss(symbol, entry_price, direction, market_data, config): Determines stop price
       - calculate_take_profit_levels(entry_price, direction, stop_loss, config): Sets take profit levels
       - adjust_trailing_stop(position, current_price, config): Updates trailing stop if needed
       - validate_trade(trade_params, portfolio_state): Validates trade against risk limits
       - calculate_portfolio_exposure(positions): Calculates current exposure
       - should_adjust_position_size(current_volatility, baseline_volatility): Adjusts for volatility

2. Create app/services/portfolio_manager.py with:
   - A PortfolioManager class that:
     - Tracks current portfolio state
     - Manages position limits
     - Handles diversification rules
     - Methods for:
       - add_position(position): Adds a new position
       - update_position(position_id, updates): Updates existing position
       - close_position(position_id, exit_price): Closes a position
       - get_current_exposure(): Gets total portfolio exposure
       - get_exposure_per_symbol(symbol): Gets exposure for a specific symbol
       - check_position_limits(new_position): Checks if new position is allowed

3. Update app/services/__init__.py to export the new classes

4. Create tests/test_risk_manager.py with:
   - Tests for position sizing calculations
   - Tests for stop loss determination
   - Tests for take profit calculations
   - Tests for trailing stop adjustments
   - Tests for validation against risk limits

5. Create tests/test_portfolio_manager.py with:
   - Tests for position tracking
   - Tests for exposure calculations
   - Tests for position limits enforcement

Use dependency injection for services and configuration. Implement proper error handling. Make the system configurable via the configuration file. Use type hints and comprehensive docstrings throughout the code.

## Prompt 22: Market Analysis Implementation

Let's implement the market analysis functionality:

1. Create app/services/market_analyzer.py with:
   - A MarketAnalyzer class that:
     - Analyzes market conditions to support trading decisions
     - Identifies market regimes (trending, ranging, volatile)
     - Detects support and resistance levels
     - Analyzes correlations between assets
     - Methods for:
       - detect_market_regime(market_data, timeframe='1h'): Identifies current regime
       - identify_support_resistance(market_data, lookback=20): Finds key levels
       - calculate_volatility(market_data, window=14): Measures market volatility
       - detect_volume_anomalies(market_data, window=20): Identifies unusual volume
       - calculate_correlation_matrix(symbols_data): Calculates correlation between assets
       - is_trend_strong(market_data, timeframe='1h'): Determines trend strength
       - get_market_context(symbol, timeframes=['5m', '15m', '1h']): Multi-timeframe analysis

2. Create app/services/market_sentiment.py with:
   - A MarketSentiment class that:
     - Analyzes market sentiment indicators
     - Aggregates data from various sources
     - Methods for:
       - calculate_internal_indicators(market_data): Uses price/volume for sentiment
       - get_market_breadth(market_data_collection): Analyzes market breadth
       - calculate_buying_selling_pressure(order_book, trades): Analyzes order flow
       - get_overall_sentiment(symbol): Combines all sentiment indicators

3. Update app/services/__init__.py to export the new classes

4. Create tests/test_market_analyzer.py with:
   - Tests for regime detection
   - Tests for support/resistance identification
   - Tests for volatility calculation
   - Tests for trend strength
   - Tests with different market scenarios

5. Create tests/test_market_sentiment.py with:
   - Tests for sentiment indicators
   - Tests for market breadth calculation
   - Tests for pressure analysis
   - Tests for overall sentiment calculation

6. Create a script in scripts/analyze_market.py that:
   - Fetches market data for selected symbols
   - Runs market analysis
   - Displays results in a readable format
   - Visualizes key levels and regimes

Implement algorithms efficiently using vectorized operations where possible. Include visualization helpers for analysis results. Use type hints and comprehensive docstrings throughout the code.

## Phase 4: Application Processes

## Prompt 23: Opportunity Scanner Process

Let's implement the opportunity scanner process:

1. Create app/services/scanner.py with:
   - An OpportunityScanner class that:
     - Scans the market for trading opportunities
     - Runs as a separate process or thread
     - Uses strategies to identify potential trades
     - Methods for:
       - scan_markets(symbols): Scans multiple markets for opportunities
       - evaluate_opportunity(symbol, market_data): Evaluates a single opportunity
       - filter_opportunities(opportunities): Applies additional filters
       - prioritize_opportunities(opportunities): Ranks opportunities by quality
       - record_opportunity(opportunity): Saves opportunity to database
       - get_required_market_data(symbol): Fetches data needed for evaluation

2. Create app/core/process.py with:
   - A BaseProcess class that:
     - Implements common process functionality
     - Handles graceful startup and shutdown
     - Manages process state and health checking
     - Provides error handling and recovery
     - Implements configurable intervals
     - Includes logging and monitoring

3. Create app/processes/scanner_process.py with:
   - Implementation of the OpportunityScanner process
   - Integration with the rest of the application
   - Proper error handling and logging

Ensure the process is efficient and handles all necessary tasks. Make sure it integrates well with the rest of the application. Use type hints and comprehensive docstrings throughout the code.

# Crypto Trading Bot - Comprehensive Developer Specification

## 1. Project Overview

This project involves developing an automated cryptocurrency trading bot that collects market data for training future machine learning models. The bot will simulate trades by recording potential positions in a PostgreSQL database without executing actual trades. It focuses on a momentum-based strategy to capture short-term price increases.

### Objectives:
- Develop a data collection system for cryptocurrency market data
- Implement a momentum-based strategy for identifying trading opportunities
- Create a position management system to simulate entries and exits
- Store all relevant data for future machine learning model training
- Provide a configurable system for strategy parameters and risk management

## 2. Detailed Requirements

### 2.1 Functional Requirements

#### Data Collection
- Connect to Kraken exchange via CCXT library
- Collect 1-minute OHLCV (Open, High, Low, Close, Volume) candles
- Calculate technical indicators for momentum strategy
- Store periodic market snapshots regardless of trades
- Filter cryptocurrencies based on market cap, volume, and other criteria

#### Opportunity Scanner
- Run as an independent process
- Implement momentum strategy indicators
- Scan for trading opportunities based on configurable criteria
- Record potential entry points to database
- Apply filtering rules to cryptocurrencies

#### Position Management
- Run as an independent process
- Simulate entries and exits without actual trading
- Implement configurable risk management rules
- Track partial exits with configurable take-profit levels
- Apply trailing stops with configurable parameters
- Record all position changes to database

#### Configuration Management
- Load configuration from YAML file
- Support for updating configuration without restart
- Record configuration history in database

#### Reporting & Logging
- Log all events and errors to console
- Record performance metrics
- Track portfolio status

### 2.2 Non-Functional Requirements

#### Performance
- Efficient processing of market data
- Minimize API calls to avoid rate limits

#### Reliability
- Implement exponential backoff for API issues
- Graceful handling of network outages
- Data integrity protections

#### Security
- Secure storage of API credentials (for future use)
- No exposure of sensitive data in logs

#### Maintainability
- Comprehensive unit tests
- Clear code organization
- Proper documentation
- Code formatting with black and linting with flake8

## 3. Architecture Design

### 3.1 High-Level Architecture

The system will follow a modular architecture with separate components for different responsibilities:

1. **Core Services Layer** - Responsible for fundamental operations
   - Exchange integration (CCXT)
   - Database access (SQLAlchemy)
   - Configuration management

2. **Domain Layer** - Implements business logic
   - Technical indicator calculation
   - Strategy implementation
   - Risk management

3. **Application Layer** - Coordinates processes
   - Opportunity scanner process
   - Position management process
   - Main application orchestration

4. **Database Layer** - Persists data
   - PostgreSQL database
   - SQLAlchemy ORM
   - Alembic migrations

### 3.2 Component Interactions

- **Main Application** initializes and orchestrates the Opportunity Scanner and Position Manager processes
- **Opportunity Scanner** retrieves market data through Exchange Service, applies momentum strategy, and saves potential trades
- **Position Manager** monitors simulated open positions, applies risk management rules, and records exits
- Both processes share access to the database through SQLAlchemy models

## 4. Database Design

### 4.1 Entity-Relationship Diagram

The database will consist of the following tables:

1. **Cryptocurrencies** - Information about tradable cryptocurrencies
   - id (PK)
   - symbol (e.g., "BTC/USD")
   - name
   - is_active (boolean)
   - market_cap
   - avg_daily_volume
   - created_at
   - updated_at

2. **MarketData** - OHLCV and indicator data
   - id (PK)
   - cryptocurrency_id (FK)
   - timestamp
   - open
   - high
   - low
   - close
   - volume
   - indicators (JSON)

3. **SimulatedTrades** - Record of simulated trades
   - id (PK)
   - cryptocurrency_id (FK)
   - entry_timestamp
   - entry_price
   - size
   - position_type (long/short)
   - stop_loss_price
   - take_profit_price
   - status (open, partially_closed, closed)
   - exit_timestamp
   - exit_price
   - profit_loss
   - profit_loss_percentage
   - strategy_used
   - notes

4. **PartialExits** - Record of partial exit points
   - id (PK)
   - trade_id (FK)
   - exit_timestamp
   - exit_price
   - exit_percentage
   - profit_loss
   - trailing_stop_activated (boolean)

5. **ConfigurationHistory** - History of configurations
   - id (PK)
   - timestamp
   - configuration (JSON)
   - run_id
   - notes

6. **PerformanceMetrics** - Performance tracking
   - id (PK)
   - timestamp
   - run_id
   - total_trades
   - winning_trades
   - losing_trades
   - win_rate
   - average_profit
   - average_loss
   - profit_factor
   - max_drawdown
   - sharpe_ratio
   - total_profit_loss

7. **MarketSnapshots** - Periodic market condition snapshots
   - id (PK)
   - cryptocurrency_id (FK)
   - timestamp
   - ohlcv (JSON)
   - indicators (JSON)
   - order_book (JSON)
   - trading_volume
   - market_sentiment
   - correlation_btc

### 4.2 Database Migration Strategy

- Use Alembic for database migrations
- Create initial migration for schema setup
- Add migrations for any schema changes
- Version control all migrations

## 5. Technical Stack

- **Programming Language**: Python 3.11.5
- **Database**: PostgreSQL
- **ORM**: SQLAlchemy
- **Migration Tool**: Alembic
- **API Library**: CCXT (v4.1.87)
- **Data Processing**: numpy (1.24.1), pandas (2.2.1), pandas-ta (0.3.14b)
- **Code Quality**: flake8, black
- **Containerization**: Docker, Docker Compose
- **Testing**: pytest

## 6. Metrics & Indicators

### 6.1 Price Data
- OHLCV (1-minute candles)
- Previous candle data

### 6.2 Momentum Indicators
- Relative Strength Index (RSI) - 14 period
- Moving Average Convergence Divergence (MACD)
- Rate of Change (ROC) - multiple periods (1, 5, 15 mins)
- Average Directional Index (ADX)

### 6.3 Volume Indicators
- Volume moving average (10 period)
- On-Balance Volume (OBV)
- Volume Rate of Change
- Volume weighted average price (VWAP)

### 6.4 Volatility Metrics
- Bollinger Bands (20,2)
- Average True Range (ATR)

### 6.5 Market Context
- Multi-timeframe analysis
- Order book data
- Support/resistance levels

### 6.6 Market Breadth
- Correlation with Bitcoin
- Market sector performance

## 7. Filtering Criteria

- Minimum market capitalization: $100M (configurable)
- Minimum 24-hour trading volume: $5M (configurable)
- Maximum bid-ask spread percentage (configurable)
- Minimum exchange listings (configurable)
- Maximum volatility threshold (configurable)
- Trading pair requirements (configurable)
- Minimum data history (configurable)

## 8. Risk Management

- Maximum position size as percentage of portfolio (configurable)
- Stop loss levels (configurable)
- Take profit levels with partial exit points (configurable)
- Trailing stop activation and offset (configurable)

## 9. Error Handling Strategy

### 9.1 API Errors
- Implement exponential backoff for rate limit errors
- Retry mechanism for temporary failures
- Circuit breaker pattern for persistent failures

### 9.2 Network Issues
- Exponential backoff for reconnection attempts
- Graceful degradation when service unavailable
- State recovery on reconnection

### 9.3 Data Integrity
- Validation of incoming data
- Database transaction management
- Data consistency checks

## 10. Testing Plan

### 10.1 Unit Testing
- Test all core components in isolation
- Mock external dependencies
- Aim for high code coverage

### 10.2 Integration Testing
- Test interactions between components
- Use mocked exchange responses

### 10.3 Functional Testing
- End-to-end tests for main workflows
- Validation of strategy implementation

### 10.4 Performance Testing
- Measure execution time for critical operations
- Validate database performance

## 11. Deployment

### 11.1 Development Environment
- PostgreSQL in Docker container
- Application running directly on host
- Environment variables in .env file

### 11.2 Production Environment
- Application and database in Docker containers
- Docker Compose for orchestration
- Volume mapping for data persistence

## 12. Project Directory Structure

```
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
│   │   ├── cryptocurrency.py
│   │   ├── market_data.py
│   │   ├── simulated_trade.py    # And other DB models
│   │   └── ...
│   ├── services/
│   │   ├── __init__.py
│   │   ├── exchange_service.py   # CCXT integration
│   │   ├── indicator_service.py  # Technical indicators
│   │   ├── scanner_service.py    # Opportunity scanner
│   │   └── position_service.py   # Position management
│   ├── strategies/
│   │   ├── __init__.py
│   │   └── momentum_strategy.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
├── tests/                        # Comprehensive unit tests
│   ├── __init__.py
│   ├── test_exchange_service.py
│   ├── test_indicator_service.py
│   └── ...
├── scripts/
│   ├── seed_database.py
│   └── run_backtest.py
├── .env                          # Environment variables
├── .env.example                  # Example .env file
├── .gitignore
├── alembic.ini                   # Alembic config
├── config.yaml                   # Configuration file
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── setup.py
└── README.md
```

## 13. Configuration File

```yaml
# Exchange Configuration
exchange:
  name: kraken
  rate_limit_retries: 5
  backoff_factor: 2
  timeout_seconds: 30

# Cryptocurrency Filtering
filtering:
  min_market_cap: 100000000  # $100M
  min_daily_volume: 5000000  # $5M
  max_spread_percentage: 0.5
  min_exchange_listings: 3
  max_volatility: 15
  allowed_quote_currencies: 
    - USD
    - USDT
    - BTC

# Strategy Parameters
momentum_strategy:
  rsi:
    period: 14
    overbought: 70
    oversold: 30
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
  roc:
    periods: [1, 5, 15]
    thresholds: [0.5, 1.0, 1.5]
  bollinger_bands:
    period: 20
    std_dev: 2
  trend_confirmation_required: true
  
# Risk Management
risk_management:
  max_position_size_percentage: 5
  stop_loss_percentage: 2
  take_profit_levels:
    - percentage: 1.5
      position_percentage: 30
    - percentage: 3.0
      position_percentage: 40
    - percentage: 5.0
      position_percentage: 30
  trailing_stop:
    activation_percentage: 2
    offset_percentage: 1
  max_open_positions: 5

# Application Settings
application:
  opportunity_scan_interval_seconds: 60
  position_check_interval_seconds: 30
  logging_level: INFO
  market_data_storage_days: 30
  
# Database
database:
  connection_string: "postgresql://username:password@localhost:5432/crypto_bot"
  pool_size: 10
  max_overflow: 20
```

## 14. Development Workflow

### 14.1 Environment Setup
1. Clone repository
2. Create and activate virtual environment
3. Install dependencies from requirements.txt
4. Create .env file from .env.example
5. Start PostgreSQL via Docker Compose
6. Run Alembic migrations

### 14.2 Implementation Phases

#### Phase 1: Foundation (Days 1-3)
- Setup project structure
- Implement database models
- Create Alembic migrations
- Setup Docker environment

#### Phase 2: Core Services (Days 4-6)
- Implement exchange service with CCXT
- Create indicator calculation service
- Develop configuration loading

#### Phase 3: Strategy Implementation (Days 7-9)
- Implement momentum strategy logic
- Add filtering criteria
- Create risk management rules

#### Phase 4: Process Management (Days 10-12)
- Develop opportunity scanner process
- Implement position management process
- Create main application orchestration

#### Phase 5: Testing and Refinement (Days 13-14)
- Implement comprehensive tests
- Refine error handling
- Optimize performance

## 15. Maintenance Considerations

- Regular backups of the database
- Monitoring for exchange API changes
- Updating dependencies for security patches
- Performance optimization based on collected data

## 16. Future Extensions

- Machine learning model integration with fastai
- Real trading implementation
- Multiple strategy support
- Web dashboard for monitoring
- Alert system for significant events 
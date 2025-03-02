# Crypto Trading Bot

A cryptocurrency trading bot designed to collect market data from various exchanges, execute trading strategies, and prepare datasets for training machine learning models. This project aims to provide a flexible framework for algorithmic trading in the cryptocurrency markets.

## Features

- Connect to multiple cryptocurrency exchanges using CCXT
- Collect and store historical market data
- Implement custom trading strategies
- Backtest strategies against historical data
- Generate datasets for machine learning models
- Execute trades automatically based on signals

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/crypto-bot.git
   cd crypto-bot
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```
   pip install -e .
   ```

5. Set up the database:
   ```
   alembic upgrade head
   ```

## Usage

1. Create a configuration file (see `config/example.yaml` for reference)
2. Run the bot:
   ```
   python -m app.main --config path/to/config.yaml
   ```

## Development

- Run tests: `pytest`
- Format code: `black .`
- Lint code: `flake8` 
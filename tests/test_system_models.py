import copy
import unittest
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from app.models.base_model import Base
from app.models.system import ConfigurationHistory, PerformanceMetrics


class TestSystemModels(unittest.TestCase):
    """Test case for system models (ConfigurationHistory, PerformanceMetrics)."""

    @classmethod
    def setUpClass(cls):
        """Set up the test database and tables."""
        # Patch the engine to use SQLite but with mocked behavior that resembles PostgreSQL
        # SQLite does not handle JSON in the same way as PostgreSQL
        cls.engine_patcher = patch('app.core.database.create_engine')
        cls.mock_create_engine = cls.engine_patcher.start()

        # Create an actual SQLite engine just for test setup
        cls.engine = create_engine('sqlite:///:memory:')
        cls.mock_create_engine.return_value = cls.engine

        # Create the tables
        Base.metadata.create_all(cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        cls.engine_patcher.stop()
        Base.metadata.drop_all(cls.engine)

    def setUp(self):
        """Create a new session for each test."""
        self.session = self.SessionLocal()

        # Generate a unique run_id for each test to ensure isolation
        self.test_run_id = f"test_run_{uuid.uuid4().hex[:8]}"

        # Create test data for configuration history
        self.test_config = {
            "exchange": "binance",
            "api_key": "test_key",
            "api_secret": "test_secret",
            "pairs": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1h",
            "strategy": {
                "name": "test_strategy",
                "parameters": {
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                }
            }
        }

        # Create test data for performance metrics
        self.test_trades = [
            {"profit_loss": 100.0},
            {"profit_loss": -50.0},
            {"profit_loss": 200.0},
            {"profit_loss": 150.0},
            {"profit_loss": -75.0}
        ]

        self.test_metrics = {
            "total_trades": 5,
            "winning_trades": 3,
            "losing_trades": 2,
            "win_rate": 60.0,
            "average_profit": 150.0,
            "average_loss": -62.5,
            "profit_factor": 4.5,
            "max_drawdown": 15.0,
            "sharpe_ratio": 1.2,
            "total_profit_loss": 325.0
        }

        # Create a patcher for the database session
        self.db_patcher = patch('app.models.system.get_db')
        self.mock_get_db = self.db_patcher.start()

        # Create a mock that returns a new session on each call
        def get_session_generator():
            while True:
                yield self.session

        self.mock_get_db.return_value = get_session_generator()

    def tearDown(self):
        """Close the session and stop the patcher."""
        self.session.close()
        self.db_patcher.stop()

    def test_save_and_retrieve_configuration(self):
        """Test saving and retrieving configuration history."""
        # Save configuration directly to test the model without class methods
        config_history = ConfigurationHistory(
            configuration=copy.deepcopy(self.test_config),
            run_id=self.test_run_id,
            notes="Test configuration"
        )
        self.session.add(config_history)
        self.session.commit()
        self.session.refresh(config_history)

        # Retrieve configuration
        config_id = config_history.id
        self.session.expunge_all()  # Detach all objects from session

        saved_config = self.session.query(ConfigurationHistory).filter_by(id=config_id).first()

        # Verify attributes
        self.assertIsNotNone(saved_config)
        self.assertEqual(saved_config.run_id, self.test_run_id)
        self.assertEqual(saved_config.notes, "Test configuration")

        # Verify JSON data
        self.assertEqual(saved_config.configuration["exchange"], "binance")
        self.assertEqual(saved_config.configuration["strategy"]["name"], "test_strategy")
        self.assertEqual(saved_config.configuration["strategy"]["parameters"]["rsi_period"], 14)

    def test_configuration_history_class_methods(self):
        """Test class methods for ConfigurationHistory."""
        # Test save_current_config
        config1 = ConfigurationHistory.save_current_config(
            config=copy.deepcopy(self.test_config),
            run_id=self.test_run_id,
            notes="First config"
        )

        self.assertEqual(config1.notes, "First config")

        # Create second config with modified parameters
        modified_config = copy.deepcopy(self.test_config)
        modified_config["strategy"]["parameters"]["rsi_period"] = 21

        # Wait a moment to ensure different timestamps
        import time
        time.sleep(0.1)

        config2 = ConfigurationHistory.save_current_config(
            config=modified_config,
            run_id=self.test_run_id,
            notes="Second config"
        )

        self.assertEqual(config2.notes, "Second config")
        self.assertEqual(config2.configuration["strategy"]["parameters"]["rsi_period"], 21)

        # Test get_latest
        latest_config = ConfigurationHistory.get_latest()
        self.assertEqual(latest_config.id, config2.id)
        self.assertEqual(latest_config.configuration["strategy"]["parameters"]["rsi_period"], 21)

        # Test get_by_run_id - ensure we only see configs for this test's run_id
        run_configs = ConfigurationHistory.get_by_run_id(self.test_run_id)
        self.assertEqual(len(run_configs), 2)
        config_notes = [c.notes for c in run_configs]
        self.assertIn("First config", config_notes)
        self.assertIn("Second config", config_notes)

    def test_save_and_retrieve_performance_metrics(self):
        """Test saving and retrieving performance metrics."""
        # Save metrics
        metrics = PerformanceMetrics(
            run_id=self.test_run_id,
            total_trades=self.test_metrics["total_trades"],
            winning_trades=self.test_metrics["winning_trades"],
            losing_trades=self.test_metrics["losing_trades"],
            win_rate=self.test_metrics["win_rate"],
            average_profit=self.test_metrics["average_profit"],
            average_loss=self.test_metrics["average_loss"],
            profit_factor=self.test_metrics["profit_factor"],
            max_drawdown=self.test_metrics["max_drawdown"],
            sharpe_ratio=self.test_metrics["sharpe_ratio"],
            total_profit_loss=self.test_metrics["total_profit_loss"]
        )
        self.session.add(metrics)
        self.session.commit()

        saved_metrics = self.session.query(
            PerformanceMetrics
        ).filter_by(run_id=self.test_run_id).first()

        # Verify attributes
        self.assertIsNotNone(saved_metrics)
        self.assertEqual(saved_metrics.run_id, self.test_run_id)
        self.assertEqual(saved_metrics.total_trades, 5)
        self.assertEqual(float(saved_metrics.win_rate), 60.0)
        self.assertEqual(float(saved_metrics.profit_factor), 4.5)

    def test_performance_metrics_class_methods(self):
        """Test class methods for PerformanceMetrics."""
        # Test record_current_performance
        metrics1 = PerformanceMetrics.record_current_performance(
            metrics_dict=self.test_metrics.copy(),
            run_id=self.test_run_id
        )

        self.assertEqual(float(metrics1.win_rate), 60.0)

        # Create second metrics with different values
        modified_metrics = self.test_metrics.copy()
        modified_metrics["win_rate"] = 70.0
        modified_metrics["total_profit_loss"] = 400.0

        # Wait a moment to ensure different timestamps
        import time
        time.sleep(0.1)

        metrics2 = PerformanceMetrics.record_current_performance(
            metrics_dict=modified_metrics,
            run_id=self.test_run_id
        )

        self.assertEqual(float(metrics2.win_rate), 70.0)

        # Test get_latest
        latest_metrics = PerformanceMetrics.get_latest()
        self.assertEqual(latest_metrics.id, metrics2.id)
        self.assertEqual(float(latest_metrics.win_rate), 70.0)

        # Test get_by_run_id
        run_metrics = PerformanceMetrics.get_by_run_id(self.test_run_id)
        self.assertEqual(len(run_metrics), 2)
        profit_losses = [float(m.total_profit_loss) for m in run_metrics]
        self.assertIn(325.0, profit_losses)
        self.assertIn(400.0, profit_losses)

    def test_calculate_metrics_from_trades(self):
        """Test calculating metrics from trade data."""
        calculated_metrics = PerformanceMetrics.calculate_from_trades(self.test_trades)

        # Verify calculations
        self.assertEqual(calculated_metrics["total_trades"], 5)
        self.assertEqual(calculated_metrics["winning_trades"], 3)
        self.assertEqual(calculated_metrics["losing_trades"], 2)
        self.assertEqual(calculated_metrics["win_rate"], 60.0)
        self.assertAlmostEqual(calculated_metrics["average_profit"], 150.0)
        self.assertAlmostEqual(calculated_metrics["average_loss"], -62.5)
        self.assertAlmostEqual(calculated_metrics["total_profit_loss"], 325.0)

        # Test with empty trades list
        empty_metrics = PerformanceMetrics.calculate_from_trades([])
        self.assertEqual(empty_metrics["total_trades"], 0)
        self.assertEqual(empty_metrics["win_rate"], 0)

    def test_run_id_based_querying(self):
        """Test querying based on run_id across both models."""
        # Create unique run IDs for this test to avoid conflicts
        run_ids = [f"unique_run_{i}_{uuid.uuid4().hex[:4]}" for i in range(3)]

        for i, run_id in enumerate(run_ids):
            # Create configuration
            config = copy.deepcopy(self.test_config)
            config["strategy"]["parameters"]["rsi_period"] = 14 + i

            config_history = ConfigurationHistory(
                configuration=config,
                run_id=run_id
            )
            self.session.add(config_history)

            # Create metrics
            metrics = PerformanceMetrics(
                run_id=run_id,
                total_trades=5,
                winning_trades=3,
                losing_trades=2,
                win_rate=50.0 + (i * 10.0),
                average_profit=150.0,
                average_loss=-62.5,
                profit_factor=4.5,
                max_drawdown=15.0,
                sharpe_ratio=1.2,
                total_profit_loss=325.0
            )
            self.session.add(metrics)

        self.session.commit()

        # Query configurations for a specific run
        configs = self.session.query(ConfigurationHistory).filter_by(run_id=run_ids[1]).all()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].configuration["strategy"]["parameters"]["rsi_period"], 15)

        # Query metrics for a specific run
        metrics_list = self.session.query(PerformanceMetrics).filter_by(run_id=run_ids[2]).all()
        self.assertEqual(len(metrics_list), 1)
        self.assertEqual(float(metrics_list[0].win_rate), 70.0)

        # Use the class methods to query
        configs_via_method = ConfigurationHistory.get_by_run_id(run_ids[1])
        metrics_via_method = PerformanceMetrics.get_by_run_id(run_ids[2])

        self.assertEqual(len(configs_via_method), 1)
        self.assertEqual(len(metrics_via_method), 1)
        self.assertEqual(
            configs_via_method[0].configuration["strategy"]["parameters"]["rsi_period"], 15
        )
        self.assertEqual(float(metrics_via_method[0].win_rate), 70.0)

    def test_validation_in_record_current_performance(self):
        """Test validation in record_current_performance method."""
        # Test with missing required fields
        incomplete_metrics = {
            "total_trades": 5,
            "winning_trades": 3
            # Missing required fields
        }

        with self.assertRaises(ValueError):
            PerformanceMetrics.record_current_performance(incomplete_metrics, self.test_run_id)


if __name__ == "__main__":
    unittest.main()

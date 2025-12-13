"""Test configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.config import (
    Config,
    DataConfig,
    ModelConfig,
    CVConfig,
    OutputConfig,
    LoggingConfig,
    VisualizationConfig,
)


class TestConfig:
    """Test cases for configuration management."""

    def test_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "data": {
                "features_path": "test_features.csv",
                "targets_path": "test_targets.csv",
                "test_size": 0.2,
                "random_state": 123,
                "stratify": True,
            },
            "models": {
                "logistic_regression": {"C": 1.0},
                "decision_tree": {"max_depth": 3},
                "random_forest": {"n_estimators": 100},
                "svm": {"kernel": "rbf"},
            },
            "hyperparameter_grids": {
                "logistic_regression": {"C": [0.1, 1.0]},
                "decision_tree": {"max_depth": [3, 5]},
            },
            "cross_validation": {
                "cv_folds": 3,
                "scoring": "accuracy",
                "n_jobs": 1,
                "verbose": 0,
            },
            "output": {
                "results_dir": "test_results",
                "model_results_path": "test_results/models.pkl",
                "save_models": True,
                "save_predictions": False,
            },
            "logging": {
                "level": "DEBUG",
                "format": "%(message)s",
                "file": "test.log",
                "console": False,
            },
            "visualization": {
                "figure_size": [10, 6],
                "dpi": 150,
                "style": "default",
                "color_palette": "viridis",
                "save_format": "jpg",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config.from_yaml(temp_path)

            assert isinstance(config.data, DataConfig)
            assert config.data.features_path == "test_features.csv"
            assert config.data.test_size == 0.2

            assert isinstance(config.models, ModelConfig)
            assert config.models.logistic_regression["C"] == 1.0

            assert isinstance(config.cross_validation, CVConfig)
            assert config.cross_validation.cv_folds == 3

            assert isinstance(config.output, OutputConfig)
            assert config.output.results_dir == "test_results"

            assert isinstance(config.logging, LoggingConfig)
            assert config.logging.level == "DEBUG"

            assert isinstance(config.visualization, VisualizationConfig)
            assert config.visualization.figure_size == [10, 6]

        finally:
            Path(temp_path).unlink()

    def test_config_file_not_found(self):
        """Test error handling when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("nonexistent_config.yaml")

    def test_invalid_yaml(self):
        """Test error handling for invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                Config.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_ensure_directories(self):
        """Test directory creation functionality."""
        config_data = {
            "data": {
                "features_path": "test.csv",
                "targets_path": "test.csv",
                "test_size": 0.3,
                "random_state": 42,
                "stratify": True,
            },
            "models": {
                "logistic_regression": {},
                "decision_tree": {},
                "random_forest": {},
                "svm": {},
            },
            "hyperparameter_grids": {},
            "cross_validation": {
                "cv_folds": 5,
                "scoring": "accuracy",
                "n_jobs": -1,
                "verbose": 1,
            },
            "output": {
                "results_dir": "test_output",
                "model_results_path": "test_output/models/results.pkl",
                "save_models": True,
                "save_predictions": True,
            },
            "logging": {
                "level": "INFO",
                "format": "%(message)s",
                "file": "test_output/logs/test.log",
                "console": True,
            },
            "visualization": {
                "figure_size": [12, 8],
                "dpi": 300,
                "style": "default",
                "color_palette": "default",
                "save_format": "png",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config.from_yaml(temp_path)
            config.ensure_directories()

            assert Path("test_output").exists()
            assert Path("test_output/models").exists()
            assert Path("test_output/logs").exists()

            # Cleanup
            import shutil

            shutil.rmtree("test_output")

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])

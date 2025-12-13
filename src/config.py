"""Configuration management module for ML pipeline."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data configuration parameters."""

    features_path: str
    targets_path: str
    test_size: float
    random_state: int
    stratify: bool


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    logistic_regression: Dict[str, Any]
    decision_tree: Dict[str, Any]
    random_forest: Dict[str, Any]
    svm: Dict[str, Any]


@dataclass
class CVConfig:
    """Cross-validation configuration parameters."""

    cv_folds: int
    scoring: str
    n_jobs: int
    verbose: int


@dataclass
class OutputConfig:
    """Output configuration parameters."""

    results_dir: str
    model_results_path: str
    save_models: bool
    save_predictions: bool


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""

    level: str
    format: str
    file: str
    console: bool


@dataclass
class VisualizationConfig:
    """Visualization configuration parameters."""

    figure_size: list[int]
    dpi: int
    style: str
    color_palette: str
    save_format: str


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig
    models: ModelConfig
    hyperparameter_grids: Dict[str, Dict[str, Any]]
    cross_validation: CVConfig
    output: OutputConfig
    logging: LoggingConfig
    visualization: VisualizationConfig

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file. If None, uses default path.

        Returns:
            Config object with loaded parameters.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid.
        """
        if config_path is None:
            config_path = "config/config.yaml"

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            data=DataConfig(**config_dict["data"]),
            models=ModelConfig(**config_dict["models"]),
            hyperparameter_grids=config_dict["hyperparameter_grids"],
            cross_validation=CVConfig(**config_dict["cross_validation"]),
            output=OutputConfig(**config_dict["output"]),
            logging=LoggingConfig(**config_dict["logging"]),
            visualization=VisualizationConfig(**config_dict["visualization"]),
        )

    def ensure_directories(self) -> None:
        """Create necessary directories based on configuration."""
        directories = [
            self.output.results_dir,
            os.path.dirname(self.output.model_results_path),
            os.path.dirname(self.logging.file),
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Global Config object.
    """
    global _config
    if _config is None:
        _config = Config.from_yaml()
        _config.ensure_directories()
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file.

    Args:
        config_path: Path to config file. If None, uses default path.

    Returns:
        New Config object.
    """
    global _config
    _config = Config.from_yaml(config_path)
    _config.ensure_directories()
    return _config

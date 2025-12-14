"""Logging configuration and utilities for ML pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional
from src.config import get_config


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: Optional[bool] = None,
) -> logging.Logger:
    """Set up structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. If None, uses config.
        console_output: Whether to output to console. If None, uses config.

    Returns:
        Configured logger instance.
    """
    config = get_config()

    # Use provided values or fall back to config
    log_level = level or config.logging.level
    log_file_path = log_file or config.logging.file
    console_enabled = (
        console_output if console_output is not None else config.logging.console
    )

    # Create logger
    logger = logging.getLogger("ml_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.logging.format)

    # File handler
    if log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for a specific module.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"ml_pipeline.{name}")


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls with arguments and return value.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    """

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise

    return wrapper


def log_data_info(data, name: str = "data") -> None:
    """Log information about dataset shape and basic statistics.

    Args:
        data: Dataset to analyze (DataFrame, array, etc.).
        name: Name of the dataset for logging.
    """
    logger = get_logger("data")

    try:
        if hasattr(data, "shape"):
            logger.info(f"{name} shape: {data.shape}")
        if hasattr(data, "describe") and callable(data.describe):
            logger.info(f"{name} basic statistics:\n{data.describe()}")
        if hasattr(data, "isnull") and callable(data.isnull):
            # Handle both DataFrame and Series
            if hasattr(data, "columns"):  # DataFrame
                null_count = data.isnull().sum().sum()
            else:  # Series
                null_count = data.isnull().sum()
            logger.info(f"{name} missing values: {null_count}")
    except Exception as e:
        logger.warning(f"Could not analyze {name}: {str(e)}")


def log_model_performance(model_name: str, metrics: dict) -> None:
    """Log model performance metrics.

    Args:
        model_name: Name of the model.
        metrics: Dictionary of performance metrics.
    """
    logger = get_logger("model")

    logger.info(f"Model: {model_name}")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")


def log_experiment_start(experiment_name: str, params: dict) -> None:
    """Log the start of an experiment with parameters.

    Args:
        experiment_name: Name of the experiment.
        params: Experiment parameters.
    """
    logger = get_logger("experiment")

    logger.info(f"Starting experiment: {experiment_name}")
    logger.info("Parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")


def log_experiment_end(experiment_name: str, results: dict) -> None:
    """Log the end of an experiment with results.

    Args:
        experiment_name: Name of the experiment.
        results: Experiment results.
    """
    logger = get_logger("experiment")

    logger.info(f"Completed experiment: {experiment_name}")
    logger.info("Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")


# Initialize logging when module is imported
try:
    setup_logging()
except Exception:
    # Fallback to basic logging if setup fails
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

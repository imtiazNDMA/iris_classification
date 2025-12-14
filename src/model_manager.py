"""Model management utilities for production deployment."""

import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
import numpy as np

from src.logger import get_logger, log_function_call
from src.config import get_config


class ModelManager:
    """Manages model saving, loading, and versioning for production."""

    def __init__(self):
        """Initialize ModelManager."""
        self.config = get_config()
        self.logger = get_logger(self.__class__.__name__)

    @log_function_call
    def save_production_model(
        self,
        model: BaseEstimator,
        label_encoder: Optional[LabelEncoder] = None,
        model_info: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0",
    ) -> str:
        """Save model in production-ready format.

        Args:
            model: Trained model to save.
            label_encoder: Fitted label encoder.
            model_info: Model metadata.
            version: Model version.

        Returns:
            Path to saved model.
        """
        try:
            # Create models directory if it doesn't exist
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            # Model file path
            model_path = models_dir / f"iris_model_v{version}.joblib"

            # Prepare model package
            model_package = {
                "model": model,
                "label_encoder": label_encoder,
                "model_info": model_info or {},
                "version": version,
                "feature_names": [
                    "sepal_length",
                    "sepal_width",
                    "petal_length",
                    "petal_width",
                ],
                "class_names": ["setosa", "versicolor", "virginica"],
                "metadata": {
                    "framework": "scikit-learn",
                    "problem_type": "classification",
                    "n_features": 4,
                    "n_classes": 3,
                },
            }

            # Save using joblib (better for sklearn objects)
            joblib.dump(model_package, model_path)

            self.logger.info(f"Production model saved to {model_path}")
            return str(model_path)

        except Exception as e:
            self.logger.error(f"Failed to save production model: {str(e)}")
            raise

    @log_function_call
    def load_production_model(self, model_path: str) -> Dict[str, Any]:
        """Load production model from file.

        Args:
            model_path: Path to model file.

        Returns:
            Model package dictionary.
        """
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model package
            model_package = joblib.load(model_path)

            self.logger.info(f"Production model loaded from {model_path}")
            return model_package

        except Exception as e:
            self.logger.error(f"Failed to load production model: {str(e)}")
            raise

    @log_function_call
    def validate_model_package(self, model_package: Dict[str, Any]) -> bool:
        """Validate model package structure.

        Args:
            model_package: Model package to validate.

        Returns:
            True if valid, False otherwise.
        """
        required_keys = ["model", "feature_names", "class_names"]

        for key in required_keys:
            if key not in model_package:
                self.logger.error(f"Missing required key in model package: {key}")
                return False

        # Validate model object
        if not hasattr(model_package["model"], "predict"):
            self.logger.error("Model object does not have predict method")
            return False

        # Validate feature names
        if len(model_package["feature_names"]) != 4:
            self.logger.error("Expected 4 feature names for iris dataset")
            return False

        # Validate class names
        if len(model_package["class_names"]) != 3:
            self.logger.error("Expected 3 class names for iris dataset")
            return False

        self.logger.info("Model package validation passed")
        return True

    @log_function_call
    def get_model_info(self, model_package: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model information from package.

        Args:
            model_package: Model package.

        Returns:
            Model information dictionary.
        """
        model = model_package["model"]

        info = {
            "model_type": type(model).__name__,
            "model_params": model.get_params() if hasattr(model, "get_params") else {},
            "feature_names": model_package["feature_names"],
            "class_names": model_package["class_names"],
            "version": model_package.get("version", "unknown"),
            "metadata": model_package.get("metadata", {}),
        }

        # Add model-specific info if available
        if hasattr(model, "feature_importances_"):
            info["feature_importances"] = dict(
                zip(model_package["feature_names"], model.feature_importances_)
            )

        if hasattr(model, "n_features_in_"):
            info["n_features"] = model.n_features_in_

        if hasattr(model, "classes_"):
            info["classes"] = model.classes_.tolist()

        return info


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance.

    Returns:
        ModelManager instance.
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

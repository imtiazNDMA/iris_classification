"""Data loading and validation module for ML pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder

from src.logger import get_logger, log_data_info, log_function_call
from src.config import get_config


class DataValidationError(Exception):
    """Custom exception for data validation errors."""

    pass


class DataLoader:
    """Handles data loading, validation, and preprocessing."""

    def __init__(self):
        """Initialize DataLoader with configuration."""
        self.config = get_config()
        self.logger = get_logger(self.__class__.__name__)
        self.label_encoder = LabelEncoder()

    @log_function_call
    def load_data(
        self, features_path: Optional[str] = None, targets_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load features and targets from CSV files.

        Args:
            features_path: Path to features CSV file. If None, uses config.
            targets_path: Path to targets CSV file. If None, uses config.

        Returns:
            Tuple of (features DataFrame, targets Series).

        Raises:
            FileNotFoundError: If data files don't exist.
            DataValidationError: If data validation fails.
        """
        features_path = features_path or self.config.data.features_path
        targets_path = targets_path or self.config.data.targets_path

        self.logger.info(f"Loading data from {features_path} and {targets_path}")

        # Check file existence
        if not Path(features_path).exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if not Path(targets_path).exists():
            raise FileNotFoundError(f"Targets file not found: {targets_path}")

        try:
            # Load data
            features_df = pd.read_csv(features_path)
            targets_df = pd.read_csv(targets_path)

            # Convert targets to Series first
            if targets_df.shape[1] == 1:
                targets_series = targets_df.iloc[:, 0]
            else:
                targets_series = targets_df.squeeze()

            # Validate data (now with Series)
            self._validate_data(features_df, targets_series)

            self.logger.info(
                f"Successfully loaded data: {features_df.shape[0]} samples, {features_df.shape[1]} features"
            )
            log_data_info(features_df, "features")
            log_data_info(targets_series, "targets")

            return features_df, targets_series

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise DataValidationError(f"Data loading failed: {str(e)}")

    @log_function_call
    def _validate_data(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Validate loaded data for consistency and quality.

        Args:
            features: Features DataFrame.
            targets: Targets Series.

        Raises:
            DataValidationError: If validation fails.
        """
        # Check shapes
        if len(features) != len(targets):
            raise DataValidationError(
                f"Features and targets length mismatch: {len(features)} vs {len(targets)}"
            )

        # Check for missing values
        if features.isnull().any().any():
            missing_count = features.isnull().sum().sum()
            self.logger.warning(f"Found {missing_count} missing values in features")

        if targets.isnull().any():
            missing_count = targets.isnull().sum()
            raise DataValidationError(
                f"Found {missing_count} missing values in targets"
            )

        # Check feature types
        non_numeric_features = features.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_features) > 0:
            self.logger.warning(
                f"Non-numeric features found: {list(non_numeric_features)}"
            )

        # Check data size
        if len(features) < 10:
            raise DataValidationError(f"Dataset too small: {len(features)} samples")

        # Check number of classes
        n_classes = targets.nunique()
        if n_classes < 2:
            raise DataValidationError(f"Need at least 2 classes, found {n_classes}")
        if n_classes > 10:
            self.logger.warning(f"Many classes found: {n_classes}")

        self.logger.info(
            f"Data validation passed: {len(features)} samples, {features.shape[1]} features, {n_classes} classes"
        )

    @log_function_call
    def encode_targets(self, targets: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
        """Encode categorical targets to numerical labels.

        Args:
            targets: Categorical targets Series.

        Returns:
            Tuple of (encoded targets array, fitted label encoder).
        """
        self.logger.info(f"Encoding {targets.nunique()} unique target classes")

        # Check if targets are already numeric
        if pd.api.types.is_numeric_dtype(targets):
            self.logger.info("Targets are already numeric, skipping encoding")
            return targets.values, None

        # Encode categorical targets
        encoded_targets = self.label_encoder.fit_transform(targets)

        self.logger.info(f"Target classes: {list(self.label_encoder.classes_)}")

        return encoded_targets, self.label_encoder

    @log_function_call
    def get_data_info(
        self, features: pd.DataFrame, targets: pd.Series
    ) -> Dict[str, Any]:
        """Get comprehensive information about the dataset.

        Args:
            features: Features DataFrame.
            targets: Targets Series.

        Returns:
            Dictionary containing dataset information.
        """
        info = {
            "n_samples": len(features),
            "n_features": features.shape[1],
            "feature_names": list(features.columns),
            "n_classes": targets.nunique(),
            "class_names": list(targets.unique())
            if not pd.api.types.is_numeric_dtype(targets)
            else None,
            "class_distribution": targets.value_counts().to_dict(),
            "missing_values": {
                "features": features.isnull().sum().sum(),
                "targets": targets.isnull().sum(),
            },
            "feature_types": features.dtypes.to_dict(),
            "memory_usage": {
                "features_mb": features.memory_usage(deep=True).sum() / 1024**2,
                "targets_mb": targets.memory_usage(deep=True) / 1024**2,
            },
        }

        # Add statistical summary for numeric features
        numeric_features = features.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            info["feature_statistics"] = numeric_features.describe().to_dict()

        return info

    @log_function_call
    def create_data_splits(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: Optional[bool] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create train-test splits with validation.

        Args:
            features: Features DataFrame.
            targets: Targets Series.
            test_size: Proportion of data for testing. If None, uses config.
            random_state: Random state for reproducibility. If None, uses config.
            stratify: Whether to stratify the split. If None, uses config.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        from sklearn.model_selection import train_test_split

        test_size = test_size or self.config.data.test_size
        random_state = random_state or self.config.data.random_state
        stratify = stratify if stratify is not None else self.config.data.stratify

        self.logger.info(
            f"Creating train-test split: test_size={test_size}, random_state={random_state}, stratify={stratify}"
        )

        # Determine stratification
        stratify_param = targets if stratify and targets.nunique() > 1 else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                targets,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_param,
            )

            self.logger.info(
                f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples"
            )

            # Check class distribution in splits
            if stratify_param is not None:
                train_dist = y_train.value_counts(normalize=True)
                test_dist = y_test.value_counts(normalize=True)
                self.logger.info(
                    f"Class distribution - Train: {train_dist.to_dict()}, Test: {test_dist.to_dict()}"
                )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.error(f"Failed to create train-test split: {str(e)}")
            raise DataValidationError(f"Train-test split failed: {str(e)}")


# Global data loader instance
_data_loader: Optional[DataLoader] = None


def get_data_loader() -> DataLoader:
    """Get global data loader instance.

    Returns:
        DataLoader instance.
    """
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader

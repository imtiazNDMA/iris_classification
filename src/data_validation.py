"""Data validation and quality checks module."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import warnings

from src.logger import get_logger


class DataValidationError(Exception):
    """Custom exception for data validation failures."""

    pass


class DataValidator:
    """Comprehensive data validation and quality checks."""

    def __init__(self):
        """Initialize DataValidator."""
        self.logger = get_logger(self.__class__.__name__)
        self.validation_results = {}

    def validate_dataset(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        min_samples: int = 10,
        max_classes: int = 50,
        min_features: int = 1,
        max_features: int = 1000,
    ) -> Dict[str, Any]:
        """Perform comprehensive dataset validation.

        Args:
            features: Features DataFrame.
            targets: Targets Series.
            min_samples: Minimum number of samples required.
            max_classes: Maximum number of classes allowed.
            min_features: Minimum number of features required.
            max_features: Maximum number of features allowed.

        Returns:
            Dictionary containing validation results.

        Raises:
            DataValidationError: If validation fails.
        """
        self.logger.info("Starting comprehensive dataset validation")

        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "info": {},
            "quality_score": 0.0,
        }

        try:
            # Basic shape validation
            self._validate_shapes(features, targets, validation_results)

            # Sample size validation
            self._validate_sample_size(features, min_samples, validation_results)

            # Feature validation
            self._validate_features(
                features, min_features, max_features, validation_results
            )

            # Target validation
            self._validate_targets(targets, max_classes, validation_results)

            # Data quality checks
            self._check_data_quality(features, targets, validation_results)

            # Statistical validation
            self._validate_statistics(features, targets, validation_results)

            # Calculate quality score
            validation_results["quality_score"] = self._calculate_quality_score(
                validation_results
            )

            # Determine overall validity
            validation_results["is_valid"] = len(validation_results["errors"]) == 0

            if not validation_results["is_valid"]:
                error_msg = "Dataset validation failed: " + "; ".join(
                    validation_results["errors"]
                )
                self.logger.error(error_msg)
                raise DataValidationError(error_msg)

            self.logger.info(
                f"Dataset validation passed. Quality score: {validation_results['quality_score']:.2f}"
            )
            return validation_results

        except Exception as e:
            self.logger.error(f"Validation process failed: {str(e)}")
            raise DataValidationError(f"Validation process failed: {str(e)}")

    def _validate_shapes(
        self, features: pd.DataFrame, targets: pd.Series, results: Dict[str, Any]
    ) -> None:
        """Validate data shapes and consistency."""
        self.logger.debug("Validating data shapes")

        # Check if features and targets have same number of samples
        if len(features) != len(targets):
            results["errors"].append(
                f"Shape mismatch: {len(features)} features vs {len(targets)} targets"
            )

        # Check for empty data
        if len(features) == 0:
            results["errors"].append("Empty dataset: no samples found")

        if len(features.columns) == 0:
            results["errors"].append("No features found in dataset")

        # Store shape information
        results["info"]["shape"] = {
            "n_samples": len(features),
            "n_features": len(features.columns),
            "targets_length": len(targets),
        }

    def _validate_sample_size(
        self, features: pd.DataFrame, min_samples: int, results: Dict[str, Any]
    ) -> None:
        """Validate minimum sample size requirements."""
        self.logger.debug("Validating sample size")

        n_samples = len(features)

        if n_samples < min_samples:
            results["errors"].append(
                f"Insufficient samples: {n_samples} < {min_samples}"
            )
        elif n_samples < 50:
            results["warnings"].append(
                f"Small dataset: {n_samples} samples (may affect model performance)"
            )

        # Check for class balance issues
        if n_samples < 100:
            results["warnings"].append("Small dataset may not be representative")

    def _validate_features(
        self,
        features: pd.DataFrame,
        min_features: int,
        max_features: int,
        results: Dict[str, Any],
    ) -> None:
        """Validate feature characteristics."""
        self.logger.debug("Validating features")

        n_features = len(features.columns)

        if n_features < min_features:
            results["errors"].append(
                f"Insufficient features: {n_features} < {min_features}"
            )
        elif n_features > max_features:
            results["warnings"].append(f"High dimensionality: {n_features} features")

        # Check for duplicate features
        duplicate_columns = features.columns.duplicated().sum()
        if duplicate_columns > 0:
            results["errors"].append(
                f"Found {duplicate_columns} duplicate feature names"
            )

        # Check for constant features
        constant_features = []
        for col in features.columns:
            if features[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            results["warnings"].append(
                f"Found {len(constant_features)} constant features: {constant_features}"
            )

        # Check feature types
        numeric_features = features.select_dtypes(include=[np.number]).columns
        categorical_features = features.select_dtypes(exclude=[np.number]).columns

        results["info"]["feature_types"] = {
            "numeric": len(numeric_features),
            "categorical": len(categorical_features),
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features),
        }

        if len(categorical_features) > 0:
            results["warnings"].append(
                f"Found {len(categorical_features)} categorical features that may need encoding"
            )

    def _validate_targets(
        self, targets: pd.Series, max_classes: int, results: Dict[str, Any]
    ) -> None:
        """Validate target characteristics."""
        self.logger.debug("Validating targets")

        # Check for missing values
        missing_targets = targets.isnull().sum()
        if missing_targets > 0:
            results["errors"].append(
                f"Found {missing_targets} missing values in targets"
            )

        # Check number of classes
        n_classes = targets.nunique()

        if n_classes < 2:
            results["errors"].append(f"Need at least 2 classes, found {n_classes}")
        elif n_classes > max_classes:
            results["warnings"].append(
                f"Many classes: {n_classes} (may affect model performance)"
            )

        # Check class balance
        class_counts = targets.value_counts()
        min_class_size = class_counts.min()
        max_class_size = class_counts.max()
        imbalance_ratio = (
            max_class_size / min_class_size if min_class_size > 0 else float("inf")
        )

        if imbalance_ratio > 10:
            results["warnings"].append(
                f"High class imbalance: ratio {imbalance_ratio:.2f}"
            )

        results["info"]["targets"] = {
            "n_classes": n_classes,
            "class_distribution": class_counts.to_dict(),
            "imbalance_ratio": imbalance_ratio,
            "class_names": list(targets.unique())
            if not pd.api.types.is_numeric_dtype(targets)
            else None,
        }

    def _check_data_quality(
        self, features: pd.DataFrame, targets: pd.Series, results: Dict[str, Any]
    ) -> None:
        """Check overall data quality."""
        self.logger.debug("Checking data quality")

        # Missing values in features
        missing_features = features.isnull().sum()
        total_missing = missing_features.sum()

        if total_missing > 0:
            missing_percentage = (
                total_missing / (len(features) * len(features.columns))
            ) * 100
            if missing_percentage > 20:
                results["warnings"].append(
                    f"High missing data: {missing_percentage:.1f}%"
                )
            else:
                results["warnings"].append(f"Missing data: {missing_percentage:.1f}%")

            results["info"]["missing_values"] = {
                "total": total_missing,
                "percentage": missing_percentage,
                "by_feature": missing_features[missing_features > 0].to_dict(),
            }

        # Outlier detection for numeric features
        numeric_features = features.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            outlier_counts = {}
            for col in numeric_features.columns:
                Q1 = numeric_features[col].quantile(0.25)
                Q3 = numeric_features[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (
                    (numeric_features[col] < (Q1 - 1.5 * IQR))
                    | (numeric_features[col] > (Q3 + 1.5 * IQR))
                ).sum()
                if outliers > 0:
                    outlier_counts[col] = outliers

            if outlier_counts:
                total_outliers = sum(outlier_counts.values())
                outlier_percentage = (total_outliers / len(features)) * 100
                if outlier_percentage > 10:
                    results["warnings"].append(
                        f"Many outliers: {outlier_percentage:.1f}%"
                    )

                results["info"]["outliers"] = outlier_counts

        # Duplicate rows
        duplicate_rows = features.duplicated().sum()
        if duplicate_rows > 0:
            duplicate_percentage = (duplicate_rows / len(features)) * 100
            if duplicate_percentage > 5:
                results["warnings"].append(
                    f"Many duplicates: {duplicate_percentage:.1f}%"
                )

            results["info"]["duplicates"] = {
                "count": duplicate_rows,
                "percentage": duplicate_percentage,
            }

    def _validate_statistics(
        self, features: pd.DataFrame, targets: pd.Series, results: Dict[str, Any]
    ) -> None:
        """Validate statistical properties."""
        self.logger.debug("Validating statistical properties")

        numeric_features = features.select_dtypes(include=[np.number])

        if not numeric_features.empty:
            # Check for zero variance features
            zero_var_features = []
            for col in numeric_features.columns:
                if numeric_features[col].var() == 0:
                    zero_var_features.append(col)

            if zero_var_features:
                results["warnings"].append(
                    f"Zero variance features: {zero_var_features}"
                )

            # Check for extreme values
            extreme_values = {}
            for col in numeric_features.columns:
                if abs(numeric_features[col]).max() > 1e6:
                    extreme_values[col] = abs(numeric_features[col]).max()

            if extreme_values:
                results["warnings"].append(f"Extreme values found: {extreme_values}")

            # Correlation analysis
            if len(numeric_features.columns) > 1:
                corr_matrix = numeric_features.corr().abs()
                high_corr_pairs = []

                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.95:
                            high_corr_pairs.append(
                                (
                                    corr_matrix.columns[i],
                                    corr_matrix.columns[j],
                                    corr_matrix.iloc[i, j],
                                )
                            )

                if high_corr_pairs:
                    results["warnings"].append(
                        f"Highly correlated features: {high_corr_pairs}"
                    )
                    results["info"]["high_correlations"] = high_corr_pairs

    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score.

        Args:
            results: Validation results dictionary.

        Returns:
            Quality score between 0 and 100.
        """
        score = 100.0

        # Deduct points for errors
        score -= len(results["errors"]) * 20

        # Deduct points for warnings
        score -= len(results["warnings"]) * 5

        # Bonus points for good characteristics
        info = results.get("info", {})

        # Good sample size
        if info.get("shape", {}).get("n_samples", 0) > 100:
            score += 5

        # Balanced classes
        targets_info = info.get("targets", {})
        if targets_info.get("imbalance_ratio", float("inf")) < 2:
            score += 5

        # Low missing data
        missing_info = info.get("missing_values", {})
        if missing_info.get("percentage", 0) < 5:
            score += 5

        return max(0.0, min(100.0, score))

    def generate_quality_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable quality report.

        Args:
            validation_results: Results from validate_dataset.

        Returns:
            Formatted quality report string.
        """
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY REPORT")
        report.append("=" * 60)

        # Overall assessment
        status = "✅ VALID" if validation_results["is_valid"] else "❌ INVALID"
        score = validation_results["quality_score"]
        report.append(f"Overall Status: {status}")
        report.append(f"Quality Score: {score:.1f}/100")
        report.append("")

        # Dataset info
        info = validation_results.get("info", {})
        if "shape" in info:
            shape = info["shape"]
            report.append("Dataset Information:")
            report.append(f"  Samples: {shape['n_samples']}")
            report.append(f"  Features: {shape['n_features']}")
            report.append("")

        if "targets" in info:
            targets = info["targets"]
            report.append("Target Information:")
            report.append(f"  Classes: {targets['n_classes']}")
            report.append(f"  Imbalance Ratio: {targets['imbalance_ratio']:.2f}")
            report.append("")

        # Errors
        if validation_results["errors"]:
            report.append("ERRORS:")
            for error in validation_results["errors"]:
                report.append(f"  ❌ {error}")
            report.append("")

        # Warnings
        if validation_results["warnings"]:
            report.append("WARNINGS:")
            for warning in validation_results["warnings"]:
                report.append(f"  ⚠️  {warning}")
            report.append("")

        # Quality indicators
        if (
            len(validation_results["errors"]) == 0
            and len(validation_results["warnings"]) == 0
        ):
            report.append("🎉 Excellent data quality! No issues found.")
        elif len(validation_results["errors"]) == 0:
            report.append("✅ Data is valid with minor quality concerns.")

        report.append("=" * 60)

        return "\n".join(report)


# Global validator instance
_validator: Optional[DataValidator] = None


def get_data_validator() -> DataValidator:
    """Get global data validator instance.

    Returns:
        DataValidator instance.
    """
    global _validator
    if _validator is None:
        _validator = DataValidator()
    return _validator

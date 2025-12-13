"""Model training and evaluation module for ML pipeline."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.logger import (
    get_logger,
    log_function_call,
    log_model_performance,
    log_experiment_start,
    log_experiment_end,
)
from src.config import get_config


class ModelTrainer:
    """Handles model training, hyperparameter tuning, and evaluation."""

    def __init__(self):
        """Initialize ModelTrainer with configuration."""
        self.config = get_config()
        self.logger = get_logger(self.__class__.__name__)
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    @log_function_call
    def get_base_models(self) -> Dict[str, BaseEstimator]:
        """Get baseline models with default parameters.

        Returns:
            Dictionary of model name to model instance.
        """
        models = {
            "logistic_regression": LogisticRegression(
                **self.config.models.logistic_regression
            ),
            "decision_tree": DecisionTreeClassifier(**self.config.models.decision_tree),
            "random_forest": RandomForestClassifier(**self.config.models.random_forest),
            "svm": SVC(**self.config.models.svm),
        }

        self.logger.info(f"Created {len(models)} baseline models")
        return models

    @log_function_call
    def train_baseline_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Train baseline models and evaluate performance.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.

        Returns:
            Dictionary of model results.
        """
        models = self.get_base_models()
        baseline_results = {}

        self.logger.info("Training baseline models")
        log_experiment_start(
            "baseline_training", {"n_models": len(models), "n_samples": len(X_train)}
        )

        for name, model in models.items():
            self.logger.info(f"Training {name}")

            try:
                # Scale features for models that need it
                X_train_scaled, X_test_scaled = self._scale_features(
                    X_train, X_test, name
                )

                # Train model
                model.fit(X_train_scaled, y_train)

                # Make predictions
                y_pred = model.predict(X_test_scaled)

                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred)

                # Cross-validation
                cv_scores = cross_val_score(
                    model,
                    X_train_scaled,
                    y_train,
                    cv=self.config.cross_validation.cv_folds,
                    scoring=self.config.cross_validation.scoring,
                    n_jobs=self.config.cross_validation.n_jobs,
                )

                metrics["cv_score"] = cv_scores.mean()
                metrics["cv_std"] = cv_scores.std()

                baseline_results[name] = {
                    "model": model,
                    "metrics": metrics,
                    "predictions": y_pred,
                    "cv_scores": cv_scores,
                }

                log_model_performance(f"{name} (baseline)", metrics)
                self.models[f"{name}_baseline"] = model

            except Exception as e:
                self.logger.error(f"Failed to train {name}: {str(e)}")
                baseline_results[name] = {"error": str(e)}

        log_experiment_end(
            "baseline_training",
            {
                "n_successful": len(
                    [r for r in baseline_results.values() if "error" not in r]
                )
            },
        )
        return baseline_results

    @log_function_call
    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Perform hyperparameter tuning for all models.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.

        Returns:
            Dictionary of tuned model results.
        """
        models = self.get_base_models()
        tuning_results = {}

        self.logger.info("Starting hyperparameter tuning")
        log_experiment_start(
            "hyperparameter_tuning",
            {
                "n_models": len(models),
                "cv_folds": self.config.cross_validation.cv_folds,
            },
        )

        for name, model in models.items():
            if name not in self.config.hyperparameter_grids:
                self.logger.warning(
                    f"No hyperparameter grid found for {name}, skipping"
                )
                continue

            self.logger.info(f"Tuning {name}")

            try:
                # Scale features
                X_train_scaled, X_test_scaled = self._scale_features(
                    X_train, X_test, name
                )

                # Get hyperparameter grid
                param_grid = self.config.hyperparameter_grids[name]

                # Perform grid search
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=self.config.cross_validation.cv_folds,
                    scoring=self.config.cross_validation.scoring,
                    n_jobs=self.config.cross_validation.n_jobs,
                    verbose=self.config.cross_validation.verbose,
                )

                grid_search.fit(X_train_scaled, y_train)

                # Get best model
                best_model = grid_search.best_estimator_

                # Make predictions
                y_pred = best_model.predict(X_test_scaled)

                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred)
                metrics["cv_score"] = grid_search.best_score_
                metrics["best_params"] = grid_search.best_params_

                tuning_results[name] = {
                    "model": best_model,
                    "metrics": metrics,
                    "predictions": y_pred,
                    "grid_search": grid_search,
                    "best_params": grid_search.best_params_,
                    "best_score": grid_search.best_score_,
                }

                log_model_performance(f"{name} (tuned)", metrics)
                self.models[f"{name}_tuned"] = best_model

            except Exception as e:
                self.logger.error(f"Failed to tune {name}: {str(e)}")
                tuning_results[name] = {"error": str(e)}

        log_experiment_end(
            "hyperparameter_tuning",
            {
                "n_successful": len(
                    [r for r in tuning_results.values() if "error" not in r]
                )
            },
        )
        return tuning_results

    @log_function_call
    def _scale_features(
        self, X_train: np.ndarray, X_test: np.ndarray, model_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features for models that require it.

        Args:
            X_train: Training features.
            X_test: Test features.
            model_name: Name of the model.

        Returns:
            Tuple of (scaled X_train, scaled X_test).
        """
        # Tree-based models don't need scaling
        if model_name in ["decision_tree", "random_forest"]:
            return X_train, X_test

        # Scale features for other models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    @log_function_call
    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary of metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        return metrics

    @log_function_call
    def get_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Get confusion matrix for predictions.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Confusion matrix array.
        """
        return confusion_matrix(y_true, y_pred)

    @log_function_call
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Get detailed classification report.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            target_names: Names of target classes.

        Returns:
            Classification report string.
        """
        return classification_report(y_true, y_pred, target_names=target_names)

    @log_function_call
    def compare_models(
        self,
        baseline_results: Dict[str, Dict[str, Any]],
        tuning_results: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Compare model performance across baseline and tuned versions.

        Args:
            baseline_results: Results from baseline models.
            tuning_results: Results from tuned models.

        Returns:
            DataFrame with model comparison.
        """
        comparison_data = []

        # Add baseline results
        for name, result in baseline_results.items():
            if "error" not in result:
                metrics = result["metrics"]
                comparison_data.append(
                    {
                        "model": name,
                        "version": "baseline",
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "cv_score": metrics["cv_score"],
                    }
                )

        # Add tuned results
        for name, result in tuning_results.items():
            if "error" not in result:
                metrics = result["metrics"]
                comparison_data.append(
                    {
                        "model": name,
                        "version": "tuned",
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "cv_score": metrics["cv_score"],
                    }
                )

        comparison_df = pd.DataFrame(comparison_data)

        # Calculate improvement
        if len(comparison_df) > 0:
            baseline_df = comparison_df[
                comparison_df["version"] == "baseline"
            ].set_index("model")
            tuned_df = comparison_df[comparison_df["version"] == "tuned"].set_index(
                "model"
            )

            common_models = set(baseline_df.index) & set(tuned_df.index)
            for model in common_models:
                for metric in ["accuracy", "precision", "recall", "f1_score"]:
                    baseline_val = baseline_df.loc[model, metric]
                    tuned_val = tuned_df.loc[model, metric]
                    improvement = ((tuned_val - baseline_val) / baseline_val) * 100
                    comparison_df.loc[
                        (comparison_df["model"] == model)
                        & (comparison_df["version"] == "tuned"),
                        f"{metric}_improvement",
                    ] = improvement

        self.logger.info(f"Created comparison for {len(comparison_df)} model versions")
        return comparison_df

    @log_function_call
    def get_best_model(
        self, comparison_df: pd.DataFrame, metric: str = "accuracy"
    ) -> Tuple[str, BaseEstimator, Dict[str, Any]]:
        """Get the best performing model based on specified metric.

        Args:
            comparison_df: Model comparison DataFrame.
            metric: Metric to use for ranking.

        Returns:
            Tuple of (model_name, model_instance, model_info).
        """
        best_row = comparison_df.loc[comparison_df[metric].idxmax()]
        model_name = f"{best_row['model']}_{best_row['version']}"

        if model_name in self.models:
            best_model = self.models[model_name]
            model_info = best_row.to_dict()

            self.logger.info(
                f"Best model: {model_name} with {metric}: {best_row[metric]:.4f}"
            )
            return model_name, best_model, model_info
        else:
            raise ValueError(f"Best model {model_name} not found in trained models")


# Global model trainer instance
_model_trainer: Optional[ModelTrainer] = None


def get_model_trainer() -> ModelTrainer:
    """Get global model trainer instance.

    Returns:
        ModelTrainer instance.
    """
    global _model_trainer
    if _model_trainer is None:
        _model_trainer = ModelTrainer()
    return _model_trainer

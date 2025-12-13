"""pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_iris_data():
    """Create sample Iris-like dataset for testing."""
    np.random.seed(42)

    # Generate synthetic Iris data
    n_samples = 150
    n_classes = 3
    n_features = 4

    # Create features with realistic ranges
    features = np.random.rand(n_samples, n_features)
    features[:, 0] = features[:, 0] * 3 + 4  # sepal_length: 4-7
    features[:, 1] = features[:, 1] * 2 + 2  # sepal_width: 2-4
    features[:, 2] = features[:, 2] * 5 + 1  # petal_length: 1-6
    features[:, 3] = features[:, 3] * 2 + 0.1  # petal_width: 0.1-2.1

    # Create balanced targets
    targets = np.repeat(["setosa", "versicolor", "virginica"], n_samples // n_classes)

    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    X = pd.DataFrame(features, columns=feature_names)
    y = pd.Series(targets, name="species")

    return X, y


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        "data": {
            "features_path": "test_features.csv",
            "targets_path": "test_targets.csv",
            "test_size": 0.3,
            "random_state": 42,
            "stratify": True,
        },
        "models": {
            "logistic_regression": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "liblinear",
                "random_state": 42,
            },
            "decision_tree": {
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
            },
            "random_forest": {
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
            },
            "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True},
        },
        "hyperparameter_grids": {
            "logistic_regression": {"penalty": ["l1", "l2"], "C": [0.1, 1.0, 10.0]},
            "decision_tree": {
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, None],
            },
            "random_forest": {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
            "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]},
        },
        "cross_validation": {
            "cv_folds": 5,
            "scoring": "accuracy",
            "n_jobs": -1,
            "verbose": 1,
        },
        "output": {
            "results_dir": "test_results",
            "model_results_path": "test_results/models/test_results.pkl",
            "save_models": True,
            "save_predictions": True,
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "test_results/logs/test.log",
            "console": True,
        },
        "visualization": {
            "figure_size": [12, 8],
            "dpi": 300,
            "style": "seaborn-v0_8",
            "color_palette": "coolwarm",
            "save_format": "png",
        },
    }


@pytest.fixture
def mock_model_results():
    """Mock model results for testing."""
    return {
        "logistic_regression": {
            "baseline": {
                "accuracy": 0.9333,
                "precision": 0.9333,
                "recall": 0.9333,
                "f1_score": 0.9333,
                "cv_score": 0.9500,
            },
            "tuned": {
                "accuracy": 0.9333,
                "precision": 0.9333,
                "recall": 0.9333,
                "f1_score": 0.9333,
                "cv_score": 0.9500,
                "best_params": {"C": 1.0, "penalty": "l2"},
            },
        },
        "decision_tree": {
            "baseline": {
                "accuracy": 0.9111,
                "precision": 0.9111,
                "recall": 0.9111,
                "f1_score": 0.9111,
                "cv_score": 0.9389,
            },
            "tuned": {
                "accuracy": 0.9778,
                "precision": 0.9778,
                "recall": 0.9778,
                "f1_score": 0.9778,
                "cv_score": 0.9524,
                "best_params": {"max_depth": 3, "criterion": "gini"},
            },
        },
    }


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# Custom assertions
def assert_dataframe_equal(df1, df2, check_dtype=True, check_index=True):
    """Custom assertion for DataFrame comparison."""
    pd.testing.assert_frame_equal(
        df1, df2, check_dtype=check_dtype, check_index=check_index
    )


def assert_series_equal(s1, s2, check_dtype=True, check_index=True):
    """Custom assertion for Series comparison."""
    pd.testing.assert_series_equal(
        s1, s2, check_dtype=check_dtype, check_index=check_index
    )


# Test utilities
def create_test_csv_file(data, filepath):
    """Create a test CSV file from DataFrame or dict."""
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    elif isinstance(data, dict):
        pd.DataFrame(data).to_csv(filepath, index=False)
    else:
        raise ValueError("Data must be DataFrame or dict")


def create_test_pickle_file(data, filepath):
    """Create a test pickle file."""
    import pickle

    with open(filepath, "wb") as f:
        pickle.dump(data, f)

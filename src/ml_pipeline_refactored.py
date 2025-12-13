"""Refactored ML pipeline with modular architecture."""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import pickle

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.config import get_config
from src.logger import setup_logging, get_logger
from src.data_loader import get_data_loader
from src.model_trainer import get_model_trainer


class MLPipeline:
    """Main ML pipeline orchestrator."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize ML pipeline.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        # Load configuration
        if config_path:
            from src.config import reload_config

            self.config = reload_config(config_path)
        else:
            self.config = get_config()

        # Setup logging
        self.logger = setup_logging(
            level=self.config.logging.level,
            log_file=self.config.logging.file,
            console_output=self.config.logging.console,
        )

        # Initialize components
        self.data_loader = get_data_loader()
        self.model_trainer = get_model_trainer()

        self.logger.info("ML Pipeline initialized")

    def run(self) -> Dict[str, Any]:
        """Run the complete ML pipeline.

        Returns:
            Dictionary containing all results.
        """
        self.logger.info("=" * 100)
        self.logger.info(" " * 30 + "IRIS CLASSIFICATION PIPELINE")
        self.logger.info(" " * 25 + "Modular Architecture v2.0")
        self.logger.info("=" * 100)

        try:
            # Step 1: Load and prepare data
            self.logger.info("\n" + "-" * 100)
            self.logger.info("STEP 1: Data Loading and Preparation")
            self.logger.info("-" * 100)

            features_df, targets_series = self.data_loader.load_data()
            data_info = self.data_loader.get_data_info(features_df, targets_series)

            # Encode targets
            y_encoded, label_encoder = self.data_loader.encode_targets(targets_series)

            # Create train-test split
            X_train, X_test, y_train, y_test = self.data_loader.create_data_splits(
                features_df, targets_series
            )

            # Convert to numpy arrays for modeling
            X_train_np = X_train.values
            X_test_np = X_test.values
            y_train_np = y_encoded if label_encoder else y_train.values
            y_test_np = y_encoded if label_encoder else y_test.values

            # Step 2: Train baseline models
            self.logger.info("\n" + "-" * 100)
            self.logger.info("STEP 2: Baseline Model Training")
            self.logger.info("-" * 100)

            baseline_results = self.model_trainer.train_baseline_models(
                X_train_np, y_train_np, X_test_np, y_test_np
            )

            # Step 3: Hyperparameter tuning
            self.logger.info("\n" + "-" * 100)
            self.logger.info("STEP 3: Hyperparameter Tuning")
            self.logger.info("-" * 100)

            tuning_results = self.model_trainer.tune_hyperparameters(
                X_train_np, y_train_np, X_test_np, y_test_np
            )

            # Step 4: Model comparison and analysis
            self.logger.info("\n" + "-" * 100)
            self.logger.info("STEP 4: Model Comparison and Analysis")
            self.logger.info("-" * 100)

            comparison_df = self.model_trainer.compare_models(
                baseline_results, tuning_results
            )

            # Get best model
            best_model_name, best_model, best_model_info = (
                self.model_trainer.get_best_model(comparison_df, metric="accuracy")
            )

            # Step 5: Save results
            self.logger.info("\n" + "-" * 100)
            self.logger.info("STEP 5: Saving Results")
            self.logger.info("-" * 100)

            results = {
                "data_info": data_info,
                "baseline_results": baseline_results,
                "tuning_results": tuning_results,
                "comparison": comparison_df.to_dict(),
                "best_model": {
                    "name": best_model_name,
                    "info": best_model_info,
                    "model": best_model,
                },
                "label_encoder": label_encoder,
            }

            if self.config.output.save_models:
                self._save_results(results)

            # Step 6: Generate summary report
            self._generate_summary_report(results)

            self.logger.info("\n" + "=" * 100)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 100)

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save pipeline results to file.

        Args:
            results: Results dictionary to save.
        """
        try:
            # Create directory if it doesn't exist
            results_path = Path(self.config.output.model_results_path)
            results_path.parent.mkdir(parents=True, exist_ok=True)

            # Save results (excluding the actual model objects for pickle compatibility)
            save_results = results.copy()
            if "best_model" in save_results:
                save_results["best_model"] = {
                    "name": save_results["best_model"]["name"],
                    "info": save_results["best_model"]["info"],
                    # Exclude the actual model object
                }

            # Remove model objects from baseline and tuning results
            for result_type in ["baseline_results", "tuning_results"]:
                if result_type in save_results:
                    for model_name in save_results[result_type]:
                        if isinstance(save_results[result_type][model_name], dict):
                            save_results[result_type][model_name] = {
                                k: v
                                for k, v in save_results[result_type][
                                    model_name
                                ].items()
                                if k != "model" and k != "grid_search"
                            }

            with open(results_path, "wb") as f:
                pickle.dump(save_results, f)

            self.logger.info(f"Results saved to {results_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")

    def _generate_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate and display summary report.

        Args:
            results: Pipeline results.
        """
        self.logger.info("\n" + "=" * 50 + " SUMMARY REPORT " + "=" * 50)

        # Data info
        data_info = results["data_info"]
        self.logger.info(
            f"Dataset: {data_info['n_samples']} samples, {data_info['n_features']} features"
        )
        self.logger.info(
            f"Classes: {data_info['n_classes']} ({', '.join(map(str, data_info['class_names'])) if data_info['class_names'] else 'Numeric'})"
        )

        # Best model
        best_model = results["best_model"]
        self.logger.info(f"\nBest Model: {best_model['name']}")
        self.logger.info(f"Accuracy: {best_model['info']['accuracy']:.4f}")
        self.logger.info(f"CV Score: {best_model['info']['cv_score']:.4f}")

        # Model comparison
        comparison = results["comparison"]
        if comparison:
            self.logger.info("\nModel Rankings (by Accuracy):")

            # Sort models by accuracy
            models_by_accuracy = sorted(
                [(k, v) for k, v in comparison.items() if "accuracy" in v],
                key=lambda x: x[1]["accuracy"],
                reverse=True,
            )

            for i, (model_key, metrics) in enumerate(models_by_accuracy[:5], 1):
                emoji = (
                    "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                )
                self.logger.info(f"{emoji} {model_key}: {metrics['accuracy']:.4f}")

        self.logger.info("\n" + "=" * 108)


def main():
    """Main entry point for the ML pipeline."""
    try:
        # Initialize and run pipeline
        pipeline = MLPipeline()
        results = pipeline.run()

        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Results saved to: {get_config().output.model_results_path}")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

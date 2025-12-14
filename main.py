"""Main entry point for the Iris Classification ML Pipeline."""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.config import get_config, reload_config
from src.logger import setup_logging, get_logger
from src.ml_pipeline_refactored import MLPipeline
from src.api import main as api_main


def run_pipeline(config_path: Optional[str] = None) -> None:
    """Run the ML pipeline.

    Args:
        config_path: Path to custom configuration file.
    """
    try:
        # Initialize pipeline
        pipeline = MLPipeline(config_path)

        # Run complete pipeline
        results = pipeline.run()

        print("\nPipeline completed successfully!")
        print(f"Results saved to: {get_config().output.model_results_path}")

        return results

    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"\nPipeline failed: {str(e)}")
        sys.exit(1)


def start_api_server(
    host: str = "0.0.0.0", port: int = 8000, reload: bool = False, workers: int = 1
) -> None:
    """Start the API server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        reload: Enable auto-reload.
        workers: Number of worker processes.
    """
    try:
        # Setup logging
        setup_logging()

        # Import and configure API
        from src.api import app

        import uvicorn

        logger = get_logger(__name__)
        logger.info(f"Starting Iris Classification API on {host}:{port}")

        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info",
        )

    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"API server failed: {str(e)}")
        print(f"\nAPI server failed: {str(e)}")
        sys.exit(1)


def run_tests() -> None:
    """Run the test suite."""
    try:
        import pytest

        print("Running test suite...")
        result = pytest.main(["tests/", "-v", "--tb=short"])

        if result == 0:
            print("\nAll tests passed!")
        else:
            print(f"\nTests failed with exit code: {result}")
            sys.exit(result)

    except ImportError:
        print("pytest not found. Install with: uv add pytest")
        sys.exit(1)
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        sys.exit(1)


def validate_data() -> None:
    """Run data validation on the dataset."""
    try:
        from src.data_loader import get_data_loader
        from src.data_validation import get_data_validator

        print("Running data validation...")

        # Load data
        data_loader = get_data_loader()
        features, targets = data_loader.load_data()

        # Validate data
        validator = get_data_validator()
        results = validator.validate_dataset(features, targets)

        # Generate report
        report = validator.generate_quality_report(results)
        print(report)

        if results["is_valid"]:
            print("\nData validation passed!")
        else:
            print("\nData validation failed!")
            sys.exit(1)

    except Exception as e:
        print(f"Data validation failed: {str(e)}")
        sys.exit(1)

    except Exception as e:
        print(f"Data validation failed: {str(e)}")
        sys.exit(1)


def show_config() -> None:
    """Display current configuration."""
    try:
        config = get_config()

        print("Current Configuration:")
        print("=" * 50)

        print("Data:")
        print(f"  Features: {config.data.features_path}")
        print(f"  Targets: {config.data.targets_path}")
        print(f"  Test Size: {config.data.test_size}")
        print(f"  Random State: {config.data.random_state}")

        print("\nModels:")
        for model_name, params in config.models.__dict__.items():
            print(f"  {model_name}: {params}")

        print("\nCross-Validation:")
        print(f"  Folds: {config.cross_validation.cv_folds}")
        print(f"  Scoring: {config.cross_validation.scoring}")
        print(f"  Jobs: {config.cross_validation.n_jobs}")

        print("\nOutput:")
        print(f"  Results Dir: {config.output.results_dir}")
        print(f"  Model Results: {config.output.model_results_path}")
        print(f"  Save Models: {config.output.save_models}")

        print("\nLogging:")
        print(f"  Level: {config.logging.level}")
        print(f"  File: {config.logging.file}")
        print(f"  Console: {config.logging.console}")

        print("=" * 50)

    except Exception as e:
        print(f"Failed to load configuration: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Iris Classification ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run ML pipeline
  python main.py --api              # Start API server
  python main.py --test             # Run tests
  python main.py --validate         # Validate data
  python main.py --config           # Show configuration
  python main.py --config custom.yaml  # Use custom config
        """,
    )

    # Main command options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pipeline", action="store_true", help="Run the ML pipeline (default)"
    )
    group.add_argument("--api", action="store_true", help="Start the API server")
    group.add_argument("--test", action="store_true", help="Run the test suite")
    group.add_argument("--validate", action="store_true", help="Validate the dataset")
    group.add_argument(
        "--config", action="store_true", help="Show current configuration"
    )

    # Configuration options
    parser.add_argument(
        "--config-file", type=str, help="Path to custom configuration file"
    )

    # API options
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind API server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind API server (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for API server"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for API server (default: 1)",
    )

    # Other options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set default action if none specified
    if not any([args.pipeline, args.api, args.test, args.validate, args.config]):
        args.pipeline = True

    # Setup logging level
    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    # Execute requested action
    try:
        if args.config:
            show_config()
        elif args.test:
            run_tests()
        elif args.validate:
            validate_data()
        elif args.api:
            start_api_server(
                host=args.host, port=args.port, reload=args.reload, workers=args.workers
            )
        else:  # default: run pipeline
            run_pipeline(config_path=args.config_file)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nOperation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

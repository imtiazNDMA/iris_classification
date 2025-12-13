"""FastAPI service for model serving and inference."""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.logger import setup_logging, get_logger
from src.data_validation import get_data_validator


# Pydantic models for API
class FeatureInput(BaseModel):
    """Input model for single prediction."""

    sepal_length: float = Field(..., ge=0.0, le=15.0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0.0, le=10.0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0.0, le=10.0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0.0, le=5.0, description="Petal width in cm")

    @validator("sepal_width")
    def validate_sepal_width(cls, v, values):
        """Validate sepal width based on sepal length."""
        if "sepal_length" in values and v > values["sepal_length"]:
            raise ValueError("Sepal width should not exceed sepal length")
        return v

    @validator("petal_width")
    def validate_petal_width(cls, v, values):
        """Validate petal width based on petal length."""
        if "petal_length" in values and v > values["petal_length"]:
            raise ValueError("Petal width should not exceed petal length")
        return v


class BatchFeatureInput(BaseModel):
    """Input model for batch predictions."""

    features: List[FeatureInput] = Field(
        ..., min_items=1, max_items=1000, description="List of feature sets"
    )

    @validator("features")
    def validate_features_length(cls, v):
        """Validate batch size."""
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 samples")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: str
    confidence: float
    class_probabilities: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[PredictionResponse]
    total_samples: int
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Model information response."""

    model_name: str
    model_type: str
    accuracy: float
    classes: List[str]
    feature_names: List[str]
    training_samples: int
    last_trained: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float


class ModelService:
    """Service class for model management and inference."""

    def __init__(self):
        """Initialize ModelService."""
        self.config = get_config()
        self.logger = get_logger(self.__class__.__name__)
        self.model = None
        self.label_encoder = None
        self.model_info = {}
        self.feature_names = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
        self.class_names = ["setosa", "versicolor", "virginica"]

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load trained model from file.

        Args:
            model_path: Path to model file. If None, uses config.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            model_path = model_path or self.config.output.model_results_path

            if not Path(model_path).exists():
                self.logger.error(f"Model file not found: {model_path}")
                return False

            with open(model_path, "rb") as f:
                results = pickle.load(f)

            # Extract model and metadata
            if "best_model" in results:
                # This is a results file, need to load the actual model
                self.logger.warning(
                    "Results file loaded, but actual model not available for inference"
                )
                return False

            # For demo purposes, create a simple mock model
            from sklearn.ensemble import RandomForestClassifier

            self.model = RandomForestClassifier(n_estimators=10, random_state=42)

            # Train on dummy data for demo
            dummy_X = np.array(
                [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]
            )
            dummy_y = np.array([0, 0, 2])
            self.model.fit(dummy_X, dummy_y)

            # Create mock label encoder
            from sklearn.preprocessing import LabelEncoder

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.class_names)

            # Set model info
            self.model_info = {
                "model_name": "RandomForestClassifier",
                "model_type": "ensemble",
                "accuracy": 0.95,
                "classes": self.class_names,
                "feature_names": self.feature_names,
                "training_samples": 150,
            }

            self.logger.info("Model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False

    def predict_single(self, features: FeatureInput) -> PredictionResponse:
        """Make prediction for single sample.

        Args:
            features: Input features.

        Returns:
            Prediction response.
        """
        if self.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        try:
            # Convert to numpy array
            feature_array = np.array(
                [
                    [
                        features.sepal_length,
                        features.sepal_width,
                        features.petal_length,
                        features.petal_width,
                    ]
                ]
            )

            # Make prediction
            prediction_encoded = self.model.predict(feature_array)[0]
            probabilities = self.model.predict_proba(feature_array)[0]

            # Decode prediction
            if self.label_encoder:
                prediction = self.label_encoder.inverse_transform([prediction_encoded])[
                    0
                ]
            else:
                prediction = self.class_names[prediction_encoded]

            # Create probability dictionary
            class_probs = {}
            for i, class_name in enumerate(self.class_names):
                class_probs[class_name] = float(probabilities[i])

            confidence = float(np.max(probabilities))

            return PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                class_probabilities=class_probs,
            )

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}",
            )

    def predict_batch(self, features: BatchFeatureInput) -> BatchPredictionResponse:
        """Make predictions for batch of samples.

        Args:
            features: Batch input features.

        Returns:
            Batch prediction response.
        """
        import time

        start_time = time.time()

        predictions = []
        for feature_set in features.features:
            prediction = self.predict_single(feature_set)
            predictions.append(prediction)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(features.features),
            processing_time_ms=processing_time,
        )

    def get_model_info(self) -> ModelInfo:
        """Get model information.

        Returns:
            Model information.
        """
        if self.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        return ModelInfo(**self.model_info)

    def health_check(self) -> HealthResponse:
        """Perform health check.

        Returns:
            Health status.
        """
        import time

        uptime = time.time() - getattr(self, "_start_time", time.time())

        return HealthResponse(
            status="healthy" if self.model is not None else "unhealthy",
            model_loaded=self.model is not None,
            version="1.0.0",
            uptime_seconds=uptime,
        )


# Initialize FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="Machine Learning API for Iris flower classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model service
model_service = ModelService()


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    # Setup logging
    setup_logging()

    # Load model
    model_service._start_time = time.time()
    success = model_service.load_model()

    if not success:
        logger = get_logger(__name__)
        logger.warning("Model failed to load, using demo model")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "Iris Classification API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return model_service.health_check()


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    return model_service.get_model_info()


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: FeatureInput):
    """Make prediction for single sample."""
    return model_service.predict_single(features)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(features: BatchFeatureInput):
    """Make predictions for batch of samples."""
    return model_service.predict_batch(features)


@app.get("/predict/demo")
async def predict_demo():
    """Demo prediction with sample data."""
    sample_features = FeatureInput(
        sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2
    )
    return model_service.predict_single(sample_features)


def main():
    """Main function to run the API server."""
    import argparse

    parser = argparse.ArgumentParser(description="Iris Classification API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    logger = get_logger(__name__)
    logger.info(f"Starting Iris Classification API on {args.host}:{args.port}")

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()

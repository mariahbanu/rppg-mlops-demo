"""
FastAPI application for model serving
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.rppg_net import create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="rPPG Heart Rate Estimation API",
    description="Estimate heart rate from RGB signals using deep learning",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total inference requests', ['status'])
INFERENCE_DURATION = Histogram('inference_duration_seconds', 'Inference duration')
PREDICTION_VALUE = Histogram('predicted_heart_rate_bpm', 'Predicted heart rate values')

# Global model
model = None
device = None


class PredictionRequest(BaseModel):
    signal: list


class PredictionResponse(BaseModel):
    heart_rate_bpm: float
    confidence: float
    model_version: str
    processing_time_ms: float


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, device
    
    logger.info("🚀 Starting model server...")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        model = create_model(sequence_length=900)
        model_path = Path('data/models/best_model.pth')
        
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"✅ Model loaded from {model_path}")
        else:
            logger.warning("⚠️ No trained model found. Using untrained model.")
        
        model = model.to(device)
        model.eval()
        
        logger.info("✅ Model server ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "rPPG Heart Rate Estimation API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict heart rate from RGB signal

    Args:
        request: PredictionRequest containing signal as list of RGB values [[r,g,b], [r,g,b], ...]
                 Should be 900 frames (30 fps * 30 seconds)

    Returns:
        Heart rate prediction
    """
    start_time = time.time()

    try:
        signal = request.signal

        # Validate input
        if not isinstance(signal, list):
            raise HTTPException(status_code=400, detail="Signal must be a list")
        
        if len(signal) != 900:
            raise HTTPException(
                status_code=400,
                detail=f"Signal must have 900 frames (got {len(signal)})"
            )
        
        # Convert to tensor
        signal_array = np.array(signal, dtype=np.float32)
        
        if signal_array.shape != (900, 3):
            raise HTTPException(
                status_code=400,
                detail=f"Signal must have shape (900, 3), got {signal_array.shape}"
            )
        
        signal_tensor = torch.FloatTensor(signal_array).unsqueeze(0)  # Add batch dimension
        signal_tensor = signal_tensor.to(device)
        
        # Predict
        with INFERENCE_DURATION.time():
            with torch.no_grad():
                prediction = model(signal_tensor)
        
        heart_rate = float(prediction.item())
        
        # Log metrics
        PREDICTION_VALUE.observe(heart_rate)
        REQUEST_COUNT.labels(status='success').inc()
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return PredictionResponse(
            heart_rate_bpm=round(heart_rate, 1),
            confidence=0.85,  # Placeholder - implement proper uncertainty quantification
            model_version="1.0",
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        REQUEST_COUNT.labels(status='error').inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(status='error').inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict-batch")
async def predict_batch(signals: list):
    """
    Predict heart rate for multiple signals

    Args:
        signals: List of signals [[[r,g,b], ...], [[r,g,b], ...]]

    Returns:
        List of predictions
    """
    predictions = []

    for signal in signals:
        request = PredictionRequest(signal=signal)
        result = await predict(request)
        predictions.append(result)

    return predictions


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

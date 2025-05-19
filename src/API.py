from fastapi import FastAPI, Request
from pydantic import BaseModel
from joblib import load
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# Load the trained SVC model
try:
    model = load(r"C:\MLOPS\MLOps-Course-Labs\src\mlartifacts\819788715109338745\0a84188a7ef24978b9dff6dee6075aba\artifacts\model\model.pkl")
    logging.info("SVC model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

# Input schema
class PredictRequest(BaseModel):
    feature1: float
    feature2: float

# Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = datetime.utcnow().isoformat()
    logging.info(f"Request {request_id} - {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response {request_id} - Status: {response.status_code}")
    return response

@app.get("/")
async def home():
    logging.info("Home endpoint called.")
    return {"message": "Welcome to the SVC Prediction API"}

@app.get("/health")
async def health():
    logging.info("Health check called.")
    model_status = "loaded" if model else "not loaded"
    return {"status": "ok", "model": model_status}

@app.post("/predict")
async def predict(data: PredictRequest):
    logging.info(f"Prediction requested: {data}")
    if model is None:
        logging.error("Prediction failed: Model not loaded")
        return {"error": "Model not loaded"}
    try:
        input_array = np.array([[data.feature1, data.feature2]])
        prediction = model.predict(input_array)[0]
        logging.info(f"Prediction result: {prediction}")
        return {"prediction": int(prediction)}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": "Prediction failed"}

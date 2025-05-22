from fastapi import FastAPI, Request
from pydantic import BaseModel
from joblib import load
import numpy as np
import logging
from datetime import datetime
from starlette.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Gauge, Counter,Histogram

REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP Requests',
    ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 
    'HTTP request latency',
    ['endpoint']
)

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
    model = load("/app/src/model.pkl")
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
async def metrics_middleware(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()

    path = request.url.path
    REQUEST_LATENCY.labels(endpoint=path).observe(process_time)
    REQUEST_COUNT.labels(method=request.method, endpoint=path, http_status=response.status_code).inc()

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

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

psi_metric = Gauge('feature_psi', 'Population Stability Index for feature data drift', ['feature'])

baseline_feature1 = np.random.normal(0, 1, 1000)
baseline_feature2 = np.random.normal(0, 1, 1000)

def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    psi = 0
    for i in range(buckets):
        expected_pct = ((expected >= breakpoints[i]) & (expected < breakpoints[i+1])).mean()
        actual_pct = ((actual >= breakpoints[i]) & (actual < breakpoints[i+1])).mean()
        expected_pct = max(expected_pct, 0.0001)
        actual_pct = max(actual_pct, 0.0001)
        psi += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return psi

@app.get("/update_psi")
def update_psi():
    actual_feature1 = np.random.normal(0, 1, 1000)
    actual_feature2 = np.random.normal(0, 1, 1000)
    psi1 = calculate_psi(baseline_feature1, actual_feature1)
    psi2 = calculate_psi(baseline_feature2, actual_feature2)
    psi_metric.labels(feature="feature1").set(psi1)
    psi_metric.labels(feature="feature2").set(psi2)
    return {"psi_feature1": psi1, "psi_feature2": psi2}

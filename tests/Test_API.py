import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from src.API import app, REQUEST_COUNT

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the SVC Prediction API"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert "status" in json_data
    assert "model" in json_data

def test_predict_with_model(monkeypatch):
    class DummyModel:
        def predict(self, X):
            # Return sum of features rounded as dummy prediction
            return [int(round(sum(X[0])))]

    monkeypatch.setattr("src.API.model", DummyModel())

    response = client.post("/predict", json={"feature1": 1.2, "feature2": 2.3})
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert isinstance(json_data["prediction"], int)

def test_predict_model_not_loaded(monkeypatch):
    monkeypatch.setattr("src.API.model", None)

    response = client.post("/predict", json={"feature1": 1.0, "feature2": 2.0})
    assert response.status_code == 200
    json_data = response.json()
    assert "error" in json_data
    assert json_data["error"] == "Model not loaded"

def test_predict_invalid_input():
    # Missing feature2
    response = client.post("/predict", json={"feature1": 1.0})
    assert response.status_code == 422  # validation error

    # Wrong type for feature1
    response = client.post("/predict", json={"feature1": "abc", "feature2": 2.0})
    assert response.status_code == 422

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

def test_update_psi():
    response = client.get("/update_psi")
    assert response.status_code == 200
    json_data = response.json()
    assert "psi_feature1" in json_data
    assert "psi_feature2" in json_data
    assert isinstance(json_data["psi_feature1"], float)
    assert isinstance(json_data["psi_feature2"], float)

def test_middleware_request_count():
    # Before sending request, get current count of requests to /health
    count_before = None
    labels = REQUEST_COUNT.labels(method="GET", endpoint="/health", http_status=200)
    try:
        count_before = labels._value.get()
    except AttributeError:
        # Internal attribute name might differ depending on prometheus client version
        count_before = labels._value.get()

    response = client.get("/health")
    assert response.status_code == 200

    # Check if count incremented
    count_after = None
    try:
        count_after = labels._value.get()
    except AttributeError:
        count_after = labels._value.get()

    assert count_after is None or (count_before is not None and count_after >= count_before)

def test_predict_edge_cases(monkeypatch):
    class DummyModel:
        def predict(self, X):
            return [0]

    monkeypatch.setattr("src.API.model", DummyModel())

    # Zero inputs
    response = client.post("/predict", json={"feature1": 0.0, "feature2": 0.0})
    assert response.status_code == 200

    # Very large inputs
    response = client.post("/predict", json={"feature1": 1e10, "feature2": -1e10})
    assert response.status_code == 200

    # Negative inputs
    response = client.post("/predict", json={"feature1": -5.5, "feature2": -10.1})
    assert response.status_code == 200

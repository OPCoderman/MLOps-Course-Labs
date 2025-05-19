from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"feature1": 1.0, "feature2": 2.0})
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data or "error" in json_data
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert "status" in json_data and "model" in json_data
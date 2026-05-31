from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_status_endpoint_returns_error_for_fake_model():
  response = client.get("/model-status/999")

  assert response.status_code == 200

  assert response.json() == {"error": "Model ID 999 not found."}
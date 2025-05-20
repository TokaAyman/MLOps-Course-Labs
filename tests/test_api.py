from fastapi.testclient import TestClient
from src.api.main import app  # now it points to src/api/main.py

client = TestClient(app)
...
 # if main.py is inside src
client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


#git remote add origin https://github.com/TokaAyman/MLOps-Course-Labs.git
"""(a) The bearer-token dependency rejects missing/wrong tokens with 401."""

from fastapi.testclient import TestClient

from app.main import app
from app.tests.conftest import TEST_TOKEN

client = TestClient(app)


def test_health_has_no_auth():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_triage_rejects_missing_token():
    response = client.post("/api/v1/triage", json={"symptom_text": "sốt cao", "photo_urls": []})
    assert response.status_code == 401


def test_triage_rejects_wrong_token():
    response = client.post(
        "/api/v1/triage",
        json={"symptom_text": "sốt cao", "photo_urls": []},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


def test_triage_accepts_correct_token():
    response = client.post(
        "/api/v1/triage",
        json={"symptom_text": "sốt cao", "photo_urls": []},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200


def test_chat_rejects_missing_token():
    response = client.post("/api/v1/chat", json={"session_id": None, "message": "xin chào"})
    assert response.status_code == 401


def test_chat_rejects_wrong_token():
    response = client.post(
        "/api/v1/chat",
        json={"session_id": None, "message": "xin chào"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401

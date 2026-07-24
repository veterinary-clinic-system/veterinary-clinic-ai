"""(d) POST /api/v1/chat end-to-end: suggest_booking True for urgent-sounding
messages, False for general questions.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.tests.conftest import TEST_TOKEN

client = TestClient(app)
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}


def test_chat_urgent_message_suggests_booking():
    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": None,
            "message": "Chó nhà em bị co giật liên tục và khó thở nặng, phải làm sao đây?",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggest_booking"] is True
    assert isinstance(body["reply"], str) and body["reply"].strip()
    assert body["session_id"]  # a uuid4 was generated since none was given


def test_chat_general_question_does_not_suggest_booking():
    response = client.post(
        "/api/v1/chat",
        json={"session_id": "existing-session-123", "message": "Phòng khám mở cửa mấy giờ vậy ạ?"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggest_booking"] is False
    assert body["session_id"] == "existing-session-123"  # echoed back, not regenerated
    assert isinstance(body["reply"], str) and body["reply"].strip()


def test_chat_symptom_message_suggests_booking():
    response = client.post(
        "/api/v1/chat",
        json={"session_id": None, "message": "Bé nhà em bị ngứa da mấy hôm nay"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggest_booking"] is True

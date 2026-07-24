"""(c) POST /api/v1/triage end-to-end.

Runs with ENABLE_CV=False (forced by conftest.clean_settings) and no
photo_urls, so this never needs torch/transformers to be installed.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.tests.conftest import TEST_TOKEN

client = TestClient(app)
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}

VALID_COLORS = {"RED", "ORANGE", "YELLOW", "GREEN", "BLUE"}


def test_triage_endpoint_returns_well_formed_response():
    response = client.post(
        "/api/v1/triage",
        json={
            "symptom_text": "Bé nhà em bị sốt cao và nôn nhiều hai ngày nay",
            "photo_urls": [],
            "pet_species": "dog",
            "pet_age_months": 24,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()

    assert body["priority_color"] in VALID_COLORS
    assert body["priority_color"] == "YELLOW"  # "sốt cao" + "nôn nhiều" -> severity 2
    assert body["cv_confidence"] is None  # no photos, CV skipped entirely
    assert isinstance(body["extracted_symptom_keywords"], list)
    assert "sốt cao" in body["extracted_symptom_keywords"]
    assert isinstance(body["suspected_disease_groups"], list)
    for group in body["suspected_disease_groups"]:
        assert set(group.keys()) == {"name", "confidence"}
        assert 0.0 <= group["confidence"] <= 1.0
    assert 0.0 <= body["nlp_confidence"] <= 1.0
    assert 0.0 <= body["overall_confidence"] <= 1.0


def test_triage_endpoint_handles_empty_symptom_text():
    response = client.post(
        "/api/v1/triage",
        json={"symptom_text": "", "photo_urls": []},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["priority_color"] == "BLUE"
    assert body["extracted_symptom_keywords"] == []
    assert body["cv_confidence"] is None

"""Integration tests for the Clinical AI Assessment FastAPI endpoints."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client():
	"""Create a TestClient with the application lifespan enabled."""
	from fastapi.testclient import TestClient
	from src.api.main import app
	with TestClient(app) as c:
		yield c


def test_health_endpoint_returns_200(client):
	"""The health endpoint responds successfully."""
	response = client.get("/api/v1/health")
	assert response.status_code == 200


def test_health_endpoint_returns_healthy_status(client):
	"""Health output includes status and version metadata."""
	response = client.get("/api/v1/health")
	data = response.json()
	assert data["status"] in ["healthy", "degraded"]
	assert "model_version" in data
	assert "timestamp" in data


def test_predict_endpoint_returns_200(client):
	"""A valid clinical note completes the prediction pipeline."""
	payload = {"note_id": "TEST001", "text": "45 year old male with fever. Diagnosed with pneumonia. Started on amoxicillin."}
	response = client.post("/api/v1/predict", json=payload)
	assert response.status_code == 200


def test_predict_returns_extracted_entities(client):
	"""Prediction output contains the structured entity fields."""
	payload = {"note_id": "TEST001", "text": "45 year old male with fever. Diagnosed with pneumonia. Started on amoxicillin."}
	response = client.post("/api/v1/predict", json=payload)
	data = response.json()
	assert "extracted_entities" in data
	assert "diagnosis" in data["extracted_entities"]
	assert "medications" in data["extracted_entities"]


def test_predict_returns_guideline_assessment(client):
	"""Prediction output contains guideline compliance results."""
	payload = {"note_id": "TEST001", "text": "45 year old male with fever. Diagnosed with pneumonia. Started on amoxicillin."}
	response = client.post("/api/v1/predict", json=payload)
	data = response.json()
	assert "guideline_assessment" in data
	assert "overall_status" in data["guideline_assessment"]


def test_predict_returns_explanation(client):
	"""Prediction output contains clinician-facing explanations."""
	payload = {"note_id": "TEST001", "text": "45 year old male with fever. Diagnosed with pneumonia. Started on amoxicillin."}
	response = client.post("/api/v1/predict", json=payload)
	data = response.json()
	assert "explanation" in data
	assert "compliance_badge" in data["explanation"]
	assert "limitations" in data["explanation"]


def test_predict_empty_text_returns_error(client):
	"""API rejects an empty clinical note with a client error."""
	payload = {"note_id": "TEST001", "text": ""}
	response = client.post("/api/v1/predict", json=payload)
	assert response.status_code in [400, 422]


def test_predict_forbidden_drug_flagged(client):
	"""A forbidden medication is surfaced as non-compliant."""
	payload = {"note_id": "TEST002", "text": "25 year old female with dysuria. Diagnosed urinary tract infection. Prescribed ciprofloxacin."}
	response = client.post("/api/v1/predict", json=payload)
	data = response.json()
	assert data["guideline_assessment"]["overall_status"] == "NON_COMPLIANT"
	assert "ciprofloxacin" in data["guideline_assessment"]["forbidden_drugs_present"]


def test_root_endpoint(client):
	"""The root endpoint exposes API navigation links."""
	response = client.get("/")
	assert response.status_code == 200
	data = response.json()
	assert "docs" in data

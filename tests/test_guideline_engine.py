"""Tests for deterministic clinical guideline compliance evaluation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def evaluator():
	"""Load the repository's production guideline rules."""
	guidelines_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "guidelines.json"
	from src.guideline_engine.evaluator import GuidelineEvaluator
	return GuidelineEvaluator(guidelines_path)


def test_pneumonia_amoxicillin_accepted(evaluator):
	"""A recommended pneumonia drug is identified as present."""
	entities = {"diagnosis": "pneumonia", "medications": ["amoxicillin"], "symptoms": [], "age": None, "sex": None}
	result = evaluator.evaluate(entities)
	assert "amoxicillin" in result["recommended_drugs_present"]
	assert result["forbidden_drugs_present"] == []


def test_uti_ciprofloxacin_flagged(evaluator):
	"""A forbidden UTI drug produces a non-compliant assessment."""
	entities = {"diagnosis": "urinary tract infection", "medications": ["ciprofloxacin"], "symptoms": [], "age": None, "sex": None}
	result = evaluator.evaluate(entities)
	assert result["overall_status"] == "NON_COMPLIANT"
	assert "ciprofloxacin" in result["forbidden_drugs_present"]


def test_unknown_diagnosis_returns_unable_to_assess(evaluator):
	"""Unknown conditions are not evaluated against unrelated guidelines."""
	entities = {"diagnosis": "dengue fever", "medications": ["paracetamol"], "symptoms": [], "age": None, "sex": None}
	result = evaluator.evaluate(entities)
	assert result["overall_status"] == "UNABLE_TO_ASSESS"


def test_none_diagnosis_handled_gracefully(evaluator):
	"""Missing diagnoses result in a safe unable-to-assess status."""
	entities = {"diagnosis": None, "medications": ["amoxicillin"], "symptoms": [], "age": None, "sex": None}
	result = evaluator.evaluate(entities)
	assert result["overall_status"] == "UNABLE_TO_ASSESS"


def test_empty_medications_list(evaluator):
	"""No medications means no recommended drugs are present."""
	entities = {"diagnosis": "pneumonia", "medications": [], "symptoms": [], "age": None, "sex": None}
	result = evaluator.evaluate(entities)
	assert result["recommended_drugs_present"] == []


def test_compliant_case_meningitis(evaluator):
	"""A recommended meningitis medication is identified as present."""
	entities = {"diagnosis": "meningitis", "medications": ["ceftriaxone"], "symptoms": [], "age": None, "sex": None}
	result = evaluator.evaluate(entities)
	assert "ceftriaxone" in result["recommended_drugs_present"]
	assert result["forbidden_drugs_present"] == []


def test_reasoning_list_not_empty(evaluator):
	"""A matched guideline produces auditable reasoning."""
	entities = {"diagnosis": "pneumonia", "medications": ["amoxicillin"], "symptoms": [], "age": None, "sex": None}
	result = evaluator.evaluate(entities)
	assert len(result["reasoning"]) > 0


def test_guideline_version_returned(evaluator):
	"""The evaluator returns the loaded guideline version."""
	entities = {"diagnosis": "pneumonia", "medications": ["amoxicillin"], "symptoms": [], "age": None, "sex": None}
	result = evaluator.evaluate(entities)
	assert result["guideline_version"] == "1.0.0"

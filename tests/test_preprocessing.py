"""Tests for clinical text cleaning, tokenization, and rule-based annotation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.text_processor import ClinicalAnnotator, ClinicalTextPreprocessor


def test_clean_text_removes_extra_whitespace():
	"""Cleaning collapses repeated whitespace while preserving text content."""
	preprocessor = ClinicalTextPreprocessor()
	result = preprocessor.clean("45  year   old  male")
	assert result == "45 year old male"


def test_tokenize_splits_correctly():
	"""Tokenization returns clinically meaningful word tokens."""
	preprocessor = ClinicalTextPreprocessor()
	tokens = preprocessor.tokenize("fever and cough.")
	assert "fever" in tokens
	assert "cough" in tokens


def test_annotate_extracts_age():
	"""Annotation extracts the documented age span."""
	annotator = ClinicalAnnotator()
	result = annotator.annotate("N001", "45 year old male with fever.")
	assert result["extracted"]["age"] == "45 year old"


def test_annotate_extracts_sex():
	"""Annotation extracts and normalizes the documented sex."""
	annotator = ClinicalAnnotator()
	result = annotator.annotate("N001", "45 year old male with fever.")
	assert result["extracted"]["sex"] == "male"


def test_annotate_extracts_diagnosis():
	"""Annotation extracts and normalizes a triggered diagnosis."""
	annotator = ClinicalAnnotator()
	result = annotator.annotate("N001", "Diagnosed with pneumonia.")
	assert result["extracted"]["diagnosis"] == "pneumonia"


def test_annotate_extracts_medications():
	"""Annotation extracts medications following prescription triggers."""
	annotator = ClinicalAnnotator()
	result = annotator.annotate("N001", "Started on amoxicillin.")
	assert "amoxicillin" in result["extracted"]["medications"]


def test_annotate_extracts_symptoms():
	"""Annotation extracts recognized symptoms from a clinical note."""
	annotator = ClinicalAnnotator()
	result = annotator.annotate("N001", "Patient has fever and cough.")
	assert "fever" in result["extracted"]["symptoms"]


def test_annotate_empty_note():
	"""Annotation always returns token and BIO-tag collections."""
	annotator = ClinicalAnnotator()
	result = annotator.annotate("N001", "No symptoms reported.")
	assert result["tokens"] is not None
	assert result["ner_tags"] is not None


def test_bio_tags_length_matches_tokens():
	"""Each token receives exactly one BIO tag."""
	annotator = ClinicalAnnotator()
	result = annotator.annotate("N001", "45 year old male with pneumonia.")
	assert len(result["tokens"]) == len(result["ner_tags"])

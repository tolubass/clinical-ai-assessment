"""
Pydantic v2 schemas for FastAPI Clinical AI Assessment API.

Defines request and response models for clinical note processing pipeline:
- ClinicalNoteRequest: Inbound clinical note data
- ExtractedEntities: Structured entity extraction output
- GuidelineAssessment: Guideline evaluation results
- ExplanationResult: Clinician-friendly explanations
- PredictionResponse: Complete pipeline response
- HealthResponse: API health status
- ErrorResponse: Standardized error format
"""

from pydantic import BaseModel, field_validator
from typing import List, Optional
from datetime import datetime


class ClinicalNoteRequest(BaseModel):
    """
    Request schema for clinical note processing.
    
    Attributes:
        note_id: Unique identifier for this clinical note
        text: Raw unstructured clinical note text (minimum 10 characters)
    """
    note_id: str
    text: str
    
    @field_validator('text')
    @classmethod
    def strip_text(cls, v: str) -> str:
        """Strip leading/trailing whitespace from clinical text."""
        if v is not None:
            v = v.strip()
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "note_id": "N001",
                "text": "45 year old male with fever and productive cough. "
                        "Diagnosed with pneumonia. Started on amoxicillin."
            }
        }
    }


class ExtractedEntities(BaseModel):
    """
    Structured clinical entities extracted from raw note text.
    
    Attributes:
        age: Patient age (e.g., "45 year old", "62")
        sex: Patient sex/gender
        symptoms: List of identified clinical symptoms
        diagnosis: Primary clinical diagnosis
        medications: List of prescribed medications
    """
    age: Optional[str] = None
    sex: Optional[str] = None
    symptoms: List[str] = []
    diagnosis: Optional[str] = None
    medications: List[str] = []


class GuidelineAssessment(BaseModel):
    """
    Clinical guideline compliance assessment results.
    
    Attributes:
        condition_matched: Guideline matched to extracted diagnosis
        guideline_version: Version of guideline engine used
        recommended_drugs_present: Medications matching recommendations
        recommended_drugs_missing: Recommended medications not documented
        forbidden_drugs_present: Medications contraindicated for condition
        required_tests: Tests required by guideline for condition
        missing_tests: Required tests not documented
        overall_status: Compliance status (COMPLIANT/PARTIAL/NON_COMPLIANT)
        reasoning: List of specific rule-matching strings
    """
    condition_matched: Optional[str] = None
    guideline_version: str
    recommended_drugs_present: List[str] = []
    recommended_drugs_missing: List[str] = []
    forbidden_drugs_present: List[str] = []
    required_tests: List[str] = []
    missing_tests: List[str] = []
    overall_status: str
    reasoning: List[str] = []


class EntityExplanation(BaseModel):
    """
    Human-readable explanation for a single extracted entity.
    
    Attributes:
        entity_type: Classification (AGE, SEX, SYMPTOM, DIAGNOSIS, MEDICATION, TEST)
        value: Extracted value from text
        confidence: Model confidence score (0.0-1.0)
        explanation: Textual explanation of extraction
    """
    entity_type: str
    value: str
    confidence: Optional[float] = None
    explanation: str


class GuidelineExplanation(BaseModel):
    """
    Human-readable explanation for guideline evaluation result.
    
    Attributes:
        rule_type: Type of rule (FORBIDDEN_PRESENT, TEST_MISSING, RECOMMENDED_MISSING, etc.)
        item: Specific item evaluated (drug name, test name, etc.)
        explanation: Detailed explanation of result
        severity: Clinical severity (CRITICAL, WARNING, INFO)
    """
    rule_type: str
    item: str
    explanation: str
    severity: str


class ExplanationResult(BaseModel):
    """
    Complete explainability output for clinical prediction.
    
    Attributes:
        entity_explanations: Per-entity extraction explanations
        guideline_explanations: Per-rule compliance explanations
        overall_summary: Clinical paragraph summarizing findings
        compliance_badge: Visual compliance status indicator
        extraction_method_summary: Summary of entity extraction method
        model_confidence_summary: Summary of model confidence levels
        limitations: List of 5 key disclaimers/limitations
    """
    entity_explanations: List[EntityExplanation]
    guideline_explanations: List[GuidelineExplanation]
    overall_summary: str
    compliance_badge: str
    extraction_method_summary: str
    model_confidence_summary: str
    limitations: List[str]


class PredictionResponse(BaseModel):
    """
    Complete response from clinical note processing pipeline.
    
    Attributes:
        note_id: Original note identifier
        extracted_entities: Structured entities from NER model
        guideline_assessment: Guideline compliance evaluation
        explanation: Clinician-friendly explanations
        extraction_method: Method used for entity extraction (ML/HYBRID/FALLBACK)
        model_version: Version of NER model
        guideline_version: Version of guideline ruleset
        inference_latency_ms: Total pipeline execution time
        timestamp: ISO8601 timestamp when prediction was generated
    """
    note_id: str
    extracted_entities: ExtractedEntities
    guideline_assessment: GuidelineAssessment
    explanation: ExplanationResult
    extraction_method: str
    model_version: str
    guideline_version: str
    inference_latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """
    API health status response.
    
    Attributes:
        status: Health status (healthy/degraded)
        model_loaded: Whether NER model is loaded
        model_version: Version of NER model
        guideline_version: Version of guideline engine
        api_version: Version of API
        timestamp: ISO8601 timestamp of health check
    """
    status: str
    model_loaded: bool
    model_version: str
    guideline_version: str
    api_version: str
    timestamp: str


class ErrorResponse(BaseModel):
    """
    Standardized error response for all API exceptions.
    
    Attributes:
        error: Short error type/code
        detail: Detailed error description
        timestamp: ISO8601 timestamp when error occurred
    """
    error: str
    detail: str
    timestamp: str

"""
FastAPI routes for Clinical AI Assessment API.

Implements two endpoints:
- GET /health: API health and component status
- POST /predict: Full clinical note processing pipeline
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timezone
from loguru import logger
import time
from src.api.schemas import (
    ClinicalNoteRequest, PredictionResponse, HealthResponse, ErrorResponse,
    ExtractedEntities, GuidelineAssessment, ExplanationResult,
)
from src.api.main import get_predictor, get_evaluator, get_explainer
import src.config as config

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description="Returns health status and component availability",
)
def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Verifies that NER model and guideline engine are loaded and accessible.
    Returns overall API status (healthy/degraded) based on component availability.
    
    Returns:
        HealthResponse: Health status with component versions
        
    Raises:
        HTTPException: If critical components fail to load
    """
    try:
        predictor = get_predictor()
        model_loaded = predictor is not None
    except Exception as e:
        logger.error(f"✗ Predictor unavailable: {e}")
        model_loaded = False
    
    status_value = "healthy" if model_loaded else "degraded"
    
    return HealthResponse(
        status=status_value,
        model_loaded=model_loaded,
        model_version=config.MODEL_VERSION,
        guideline_version=config.GUIDELINE_VERSION,
        api_version=config.API_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Process clinical note",
    description="Extract entities and evaluate treatment against guidelines",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Pipeline execution error"},
    },
)
def predict(request: ClinicalNoteRequest) -> PredictionResponse:
    """
    Clinical note processing endpoint.
    
    Runs complete pipeline:
    1. Clinical NER: Extract entities (age, sex, symptoms, diagnosis, medications)
    2. Guideline Evaluation: Check compliance with evidence-based guidelines
    3. Explainability: Generate clinician-friendly explanations
    
    Args:
        request: Clinical note with ID and raw text
        
    Returns:
        PredictionResponse: Complete prediction with entities, assessment, explanations
        
    Raises:
        HTTPException(400): If input validation fails
        HTTPException(500): If pipeline execution fails
    """
    try:
        # ── Validate input ──────────────────────────────────────
        if not request.text or len(request.text.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Clinical note text must be at least 10 characters",
            )
        
        # ── Load pipeline components ────────────────────────────
        predictor = get_predictor()
        evaluator = get_evaluator()
        explainer = get_explainer()
        
        if not predictor or not evaluator or not explainer:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Required pipeline components not loaded",
            )
        
        # ── Run prediction pipeline ─────────────────────────────
        start_time = time.time()
        
        # Step 1: Extract entities
        logger.info(f"🔍 Extracting entities from note {request.note_id}")
        prediction = predictor.predict(request.text)
        extracted_entities = prediction["extracted"]
        
        # Step 2: Evaluate guideline compliance
        logger.info(f"📋 Evaluating guideline compliance for note {request.note_id}")
        guideline_result = evaluator.evaluate(extracted_entities)
        
        # Step 3: Generate explanations
        logger.info(f"💡 Generating explanations for note {request.note_id}")
        explanation = explainer.explain(prediction, guideline_result)
        
        # Calculate total latency
        inference_latency_ms = (time.time() - start_time) * 1000
        
        # ── Build response ──────────────────────────────────────
        response = PredictionResponse(
            note_id=request.note_id,
            extracted_entities=ExtractedEntities(**extracted_entities),
            guideline_assessment=GuidelineAssessment(
                condition_matched=guideline_result.get("condition_matched"),
                guideline_version=guideline_result.get("guideline_version", config.GUIDELINE_VERSION),
                recommended_drugs_present=guideline_result.get("recommended_drugs_present", []),
                recommended_drugs_missing=guideline_result.get("recommended_drugs_missing", []),
                forbidden_drugs_present=guideline_result.get("forbidden_drugs_present", []),
                required_tests=guideline_result.get("required_tests", []),
                missing_tests=guideline_result.get("missing_tests", []),
                overall_status=guideline_result.get("overall_status", "UNKNOWN"),
                reasoning=guideline_result.get("reasoning", []),
            ),
            explanation=ExplanationResult(**explanation),
            extraction_method=prediction.get("extraction_method", "ML"),
            model_version=prediction.get("model_version", config.MODEL_VERSION),
            guideline_version=config.GUIDELINE_VERSION,
            inference_latency_ms=inference_latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        # Log successful prediction
        logger.info(
            f"✓ Prediction complete | note_id={request.note_id} | "
            f"latency={inference_latency_ms:.0f}ms | "
            f"status={response.guideline_assessment.overall_status}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Pipeline error for note {request.note_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(e)}",
        )

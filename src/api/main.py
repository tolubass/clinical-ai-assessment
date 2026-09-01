"""
FastAPI application factory for Clinical AI Assessment API.

Initializes the FastAPI application with lifespan management for:
- Loading clinical NER predictor (BioBERT fine-tuned model)
- Loading guideline evaluation engine (rule-based compliance checker)
- Loading explainability module (reasoning generator)
- Configuring logging

Serves as the main entry point for uvicorn:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger
from datetime import datetime, timezone
import sys
from pathlib import Path

# ── Add parent directory to path ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import src.config as config
from src.modeling.predictor import ClinicalNERPredictor
from src.guideline_engine.evaluator import GuidelineEvaluator
from src.explainability.explainer import ClinicalExplainer

# ── Module-level components ─────────────────────────────────
_predictor: ClinicalNERPredictor = None
_evaluator: GuidelineEvaluator = None
_explainer: ClinicalExplainer = None


def get_predictor() -> ClinicalNERPredictor:
    """
    Retrieve loaded clinical NER predictor.
    
    Returns:
        ClinicalNERPredictor: BioBERT fine-tuned model for entity extraction
        
    Raises:
        RuntimeError: If predictor not loaded during startup
    """
    global _predictor
    if _predictor is None:
        raise RuntimeError("Predictor not initialized. Check startup logs.")
    return _predictor


def get_evaluator() -> GuidelineEvaluator:
    """
    Retrieve loaded guideline evaluation engine.
    
    Returns:
        GuidelineEvaluator: Rule-based guideline compliance checker
        
    Raises:
        RuntimeError: If evaluator not loaded during startup
    """
    global _evaluator
    if _evaluator is None:
        raise RuntimeError("Evaluator not initialized. Check startup logs.")
    return _evaluator


def get_explainer() -> ClinicalExplainer:
    """
    Retrieve loaded explainability module.
    
    Returns:
        ClinicalExplainer: Explanation generator for predictions
        
    Raises:
        RuntimeError: If explainer not loaded during startup
    """
    global _explainer
    if _explainer is None:
        raise RuntimeError("Explainer not initialized. Check startup logs.")
    return _explainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage FastAPI application lifecycle.
    
    Startup: Load all pipeline components (predictor, evaluator, explainer)
    Shutdown: Clean up resources
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None
    """
    global _predictor, _evaluator, _explainer
    
    # ── Startup ─────────────────────────────────────────────────
    logger.info("🚀 Clinical AI Assessment API starting up...")
    
    try:
        logger.info(f"📦 Loading NER predictor from {config.MODEL_SAVE_DIR}")
        _predictor = ClinicalNERPredictor(config.MODEL_SAVE_DIR)
        logger.info(f"✓ NER predictor loaded successfully (version {config.MODEL_VERSION})")
    except Exception as e:
        logger.error(f"✗ Failed to load NER predictor: {str(e)}", exc_info=True)
        logger.exception(e)
        raise
    
    try:
        logger.info(f"📋 Loading guideline evaluator from {config.GUIDELINES_PATH}")
        _evaluator = GuidelineEvaluator(config.GUIDELINES_PATH)
        logger.info(f"✓ Guideline evaluator loaded successfully (version {config.GUIDELINE_VERSION})")
    except Exception as e:
        logger.error(f"✗ Failed to load guideline evaluator: {str(e)}", exc_info=True)
        logger.exception(e)
        raise
    
    try:
        logger.info("💡 Loading explainability module")
        _explainer = ClinicalExplainer()
        logger.info("✓ Explainability module loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load explainability module: {str(e)}", exc_info=True)
        logger.exception(e)
        raise
    
    if _predictor and _evaluator and _explainer:
        logger.info("=" * 80)
        logger.info("✓ All components loaded. API ready for predictions.")
        logger.info("=" * 80)
    else:
        logger.warning("⚠️  Some components failed to load. API may be degraded.")
    
    yield
    
    # ── Shutdown ────────────────────────────────────────────────
    logger.info("🛑 Clinical AI Assessment API shutting down...")
    _predictor = None
    _evaluator = None
    _explainer = None


# ── Configure logging ───────────────────────────────────────
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    str(config.LOG_FILE),
    rotation=config.LOG_ROTATION,
    retention=config.LOG_RETENTION,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="INFO",
)

# ── Create FastAPI application ──────────────────────────────
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Import and include router ───────────────────────────────
from src.api.routes import router as api_router

app.include_router(api_router, prefix="/api/v1", tags=["Clinical Assessment"])


@app.get(
    "/",
    summary="API root",
    description="Welcome message with links to documentation and endpoints",
)
def root():
    """
    Root endpoint providing API documentation links.
    
    Returns:
        dict: Welcome message and endpoint references
    """
    return {
        "message": "Clinical AI Assessment API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict",
    }

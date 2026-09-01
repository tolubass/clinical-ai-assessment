# Clinical AI Assessment — Entity Extraction & Guideline Evaluation

![Python 3.14](https://img.shields.io/badge/Python-3.14-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-API-009688) ![BioBERT](https://img.shields.io/badge/BioBERT-NER-orange) ![MLflow](https://img.shields.io/badge/MLflow-MLOps-blueviolet) ![Docker](https://img.shields.io/badge/Docker-Container-2496ED) ![Tests](https://img.shields.io/badge/Tests-26%20Passed-brightgreen)

## Problem Statement

Clinical notes in healthcare are unstructured free text. This system extracts structured clinical entities and evaluates documented treatment against evidence-based guidelines to support clinical decision-making in digital health settings.

## Project Objectives

- Extract structured entities (age, sex, symptoms, diagnosis, medications) from clinical free text
- Evaluate prescribed treatment against evidence-based clinical guidelines
- Provide explainable, auditable decision-support output
- Expose the complete pipeline through a production-grade REST API
- Demonstrate MLOps, reproducibility, and clinical safety principles

## System Architecture

```text
Clinical Note (Free Text)
				 |
				 v
Input Validation + Text Preprocessing
				 |
				 v
BioBERT NER Model + Hybrid Fallback
				 |
				 v
Structured Entities: age, sex, diagnosis, medications, symptoms
				 |
				 v
Guideline Evaluation Engine (Deterministic)
				 |
				 v
Explainability Layer
				 |
				 v
API Response (JSON)
```

## Repository Structure

```text
clinical-ai-assessment/
|-- data/
|   |-- raw/                    # Clinical notes and guideline rules
|   |-- processed/              # Reproducible train/validation/test splits
|   `-- annotated/              # BIO-annotated NER dataset
|-- src/
|   |-- preprocessing/          # Cleaning, tokenization, rule-based fallback
|   |-- modeling/               # BioBERT training and prediction
|   |-- guideline_engine/       # Deterministic guideline evaluation
|   |-- explainability/         # Grounded clinical explanations
|   `-- api/                    # FastAPI application, routes, and schemas
|-- models/ner_model/           # Trained model artifacts and metadata
|-- notebooks/                  # EDA and model evaluation notebooks
|-- tests/                      # Unit and API tests
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## Data Description

- 50 synthetic clinical notes
- 10 distinct diagnoses
- Average ~20 words per note
- 100% guideline coverage
- No duplicates

## Machine Learning Approach

**Model:** `dmis-lab/biobert-base-cased-v1.2`

**Reason:** Pre-trained on PubMed and PMC clinical text. Transfer learning appropriate for 50-note dataset.

The NER model uses a BIO tagging scheme with 13 labels. Data is split at note level using a 70/15/15 train/validation/test split to prevent leakage. A fixed random seed of 42 provides reproducible data preparation and training.

The extraction pipeline is hybrid: BioBERT is the primary extractor, while the rule-based fallback is used when diagnosis confidence is low or the model cannot safely recover a diagnosis. This fallback supports predictable behavior for multi-word clinical conditions.

## Model Performance

| Entity     | Precision | Recall | F1     |
| ---------- | --------- | ------ | ------ |
| AGE        | 1.0000    | 1.0000 | 1.0000 |
| SEX        | 1.0000    | 1.0000 | 1.0000 |
| DIAGNOSIS  | 1.0000    | 0.7143 | 0.8333 |
| MEDICATION | 1.0000    | 0.7778 | 0.8750 |
| SYMPTOM    | 0.7778    | 0.9333 | 0.8485 |
| Overall    | 0.7222    | 0.6782 | 0.6946 |

**Note:** Overall F1 dragged down by I-tag weakness on small dataset. Entity-level performance is strong.

## Guideline Evaluation Engine

The guideline evaluator is a deterministic rule engine that loads explicit rules from `guidelines.json`. It normalizes drug names, checks recommended and forbidden drugs, identifies missing required tests, and returns auditable reasoning for each decision.

The engine never uses an LLM for clinical decisions. Explanations are grounded in the extracted entities and the exact guideline assessment output; drug and test names are not invented by the system.

## API Documentation

### Endpoints

| Method | Path              | Purpose                                  |
| ------ | ----------------- | ---------------------------------------- |
| `GET`  | `/api/v1/health`  | Component health and version status      |
| `POST` | `/api/v1/predict` | Extract entities and evaluate guidelines |
| `GET`  | `/docs`           | Swagger UI documentation                 |
| `GET`  | `/redoc`          | ReDoc documentation                      |

### Sample Request

```json
{
  "note_id": "N001",
  "text": "45 year old male with fever and cough. Diagnosed with pneumonia. Started on amoxicillin."
}
```

### Sample Response

```json
{
  "note_id": "N001",
  "extracted_entities": {
    "age": "45 year old",
    "sex": "male",
    "symptoms": ["fever", "cough"],
    "diagnosis": "pneumonia",
    "medications": ["amoxicillin"]
  },
  "guideline_assessment": {
    "condition_matched": "pneumonia",
    "guideline_version": "1.0.0",
    "recommended_drugs_present": ["amoxicillin"],
    "recommended_drugs_missing": ["azithromycin"],
    "forbidden_drugs_present": [],
    "required_tests": ["chest_xray"],
    "missing_tests": ["chest_xray"],
    "overall_status": "PARTIAL_COMPLIANCE",
    "reasoning": [
      "Matched guideline for condition: pneumonia",
      "Drug amoxicillin is recommended for pneumonia and is present in prescription. ACCEPTED."
    ]
  },
  "explanation": {
    "entity_explanations": [],
    "guideline_explanations": [],
    "overall_summary": "The documented treatment partially complies with the matched guideline.",
    "compliance_badge": "PARTIAL_COMPLIANCE",
    "extraction_method_summary": "Entities extracted using the clinical NER pipeline.",
    "model_confidence_summary": "Confidence values are reported per extracted entity.",
    "limitations": ["Clinical decision-support prototype only."]
  },
  "extraction_method": "hybrid",
  "model_version": "1.0.0",
  "guideline_version": "1.0.0",
  "inference_latency_ms": 42.5,
  "timestamp": "2026-09-01T12:00:00+00:00"
}
```

## Installation

```bash
git clone https://github.com/tolubass/clinical-ai-assessment.git
cd clinical-ai-assessment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ **Model Artifact Required**
> The trained model file (~411MB) exceeds GitHub's file size
> limit and is not stored in this repository.
> You must run training before starting the API:
>
> ```bash
> python -m src.preprocessing.text_processor
> python -m src.modeling.trainer
> ```
>
> Training takes approximately 5-10 minutes on CPU.
> The API will not start without the trained model.

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
python -m pytest tests/ -v
```

After startup, open the interactive API documentation at `http://localhost:8000/docs`.

## Docker

```bash
docker build -t clinical-ai-assessment:1.0.0 .
docker run -p 8000:8000 clinical-ai-assessment:1.0.0
```

The image uses Python 3.11 for package compatibility with PyTorch and Transformers. Trained model artifacts and guideline data are included in the image; tests, notebooks, logs, and local experiment output are excluded by `.dockerignore`.

## MLOps

- MLflow experiment tracking with SQLite backend
- Model versioning with `model_metadata.json`
- Structured prediction logging with Loguru
- Inference latency tracked per request
- Reproducible training with fixed random seed 42
- Dependency pinning in `requirements.txt`

## Error Analysis

| Error Type                  | Cause                        | Severity | Mitigation         |
| --------------------------- | ---------------------------- | -------- | ------------------ |
| Multi-word diagnosis miss   | I-tag weakness               | Critical | Hybrid fallback    |
| Low symptom confidence      | Small dataset                | Medium   | More training data |
| Medication stopword leakage | Prepositions near drug names | Low      | Stopword filter    |
| I-tag prediction weakness   | Class imbalance              | Medium   | Weighted loss      |

## Limitations

- Trained on 50 synthetic notes only
- Multi-word diagnoses use rule-based fallback
- Required tests not extractable from notes
- Clinical decision-support prototype only — does not replace clinical judgment

## Ethical and Safety Considerations

- No autonomous clinical decisions
- All outputs presented as decision support
- Guideline engine is deterministic — no hallucination in clinical rules
- Synthetic de-identified data only
- Prediction logging avoids full clinical note content

## Future Improvements

- Expand to 500+ real clinical notes
- FHIR-compatible output format
- ICD-10 diagnosis code mapping
- Multilingual support for local languages
- Cloud deployment with auto-scaling

## AI Tools Disclosure

Claude (Anthropic) used for architecture brainstorming, code assistance, debugging, and documentation. All engineering decisions, validation, and final responsibility remain with the developer.

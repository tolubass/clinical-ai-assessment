"""
Explainability module for clinical NER predictions and guideline assessments.

Provides human-readable, clinically meaningful explanations that clinicians
can understand, audit, and trust. Produces explanations at multiple levels:
entity-level confidence, guideline compliance reasoning, and summary assessments.
"""

from typing import Dict, List, Any, Optional
from loguru import logger


class ClinicalExplainer:
    """
    Generates human-readable explanations for clinical NLP predictions
    and guideline compliance assessments.
    
    Provides clinically meaningful explanations suitable for clinician
    review and audit trails, including confidence scores, extraction methods,
    and compliance reasoning.
    """
    
    def __init__(self) -> None:
        """
        Initialize the clinical explainer.
        
        Sets up logging for audit trail and explanation tracking.
        """
        logger.info("ClinicalExplainer initialized")
    
    def explain(
        self,
        prediction: Dict[str, Any],
        guideline_assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanations for prediction and assessment.
        
        Combines entity-level explanations, guideline compliance reasoning,
        and clinical summaries into a single structured explanation dict
        suitable for clinician review.
        
        Args:
            prediction: Full prediction dict from ClinicalNERPredictor.predict()
                containing tokens, ner_tags, extracted entities, confidence scores,
                model_version, extraction_method, etc.
            guideline_assessment: Full assessment dict from GuidelineEvaluator.evaluate()
                containing extracted entities, compliance status, reasoning, etc.
                
        Returns:
            Dictionary with keys:
                - entity_explanations (list): Per-entity explanations with confidence
                - guideline_explanations (list): Per-rule guideline compliance explanations
                - overall_summary (str): Paragraph summary for clinician
                - compliance_badge (str): Visual compliance status badge
                - extraction_method_summary (str): Explanation of extraction approach
                - model_confidence_summary (str): Summary of model confidence
                - limitations (list): System limitations and disclaimers
        """
        logger.info("Generating comprehensive clinical explanations")

        extracted = prediction.get("extracted", {})

        extraction_method = prediction.get("extraction_method", "unknown")
        confidence_scores = prediction.get("confidence_scores", {})
        guideline_status = guideline_assessment.get("overall_status", "UNKNOWN")
        guideline_reasoning = guideline_assessment.get("reasoning", [])
        
        entity_explanations = self._explain_entities(prediction)

        guideline_explanations = self._explain_guidelines(guideline_assessment)
        overall_summary = self._build_summary(extracted, guideline_assessment)
        compliance_badge = self._build_compliance_badge(guideline_status)
        extraction_method_summary = self._build_extraction_method_summary(
            extraction_method, entity_explanations
        )
        model_confidence_summary = self._build_confidence_summary(confidence_scores)
        limitations = self._build_limitations()
        
        result = {
            "entity_explanations": entity_explanations,
            "guideline_explanations": guideline_explanations,
            "overall_summary": overall_summary,
            "compliance_badge": compliance_badge,
            "extraction_method_summary": extraction_method_summary,
            "model_confidence_summary": model_confidence_summary,
            "limitations": limitations,
        }
        
        logger.info(f"Explanations generated. Compliance: {guideline_status}")
        
        return result
    
    def _explain_entities(self, prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build explanations for each extracted entity.
        
        Provides entity-level explanations including confidence scores,
        extraction method, and source text.
        
        Args:
            prediction: Prediction dict from ClinicalNERPredictor.predict()
            
        Returns:
            List of entity explanation dicts with type, value, confidence, method.
        """
        entity_explanations = []
        extracted = prediction.get("extracted", {})
        confidence_scores = prediction.get("confidence_scores", {})
        extraction_method = prediction.get("extraction_method", "model")
        
        if extracted.get("age"):
            confidence = confidence_scores.get("age")
            confidence_str = f"{confidence:.4f}" if confidence else "N/A"
            entity_explanations.append({
                "entity_type": "AGE",
                "value": extracted["age"],
                "extraction_method": extraction_method,
                "confidence": confidence,
                "source_text": extracted["age"],
                "explanation": (
                    f"Age '{extracted['age']}' extracted by BioBERT model "
                    f"with confidence {confidence_str}."
                ),
            })
        
        if extracted.get("sex"):
            confidence = confidence_scores.get("sex")
            confidence_str = f"{confidence:.4f}" if confidence else "N/A"
            entity_explanations.append({
                "entity_type": "SEX",
                "value": extracted["sex"],
                "extraction_method": extraction_method,
                "confidence": confidence,
                "source_text": extracted["sex"],
                "explanation": (
                    f"Sex '{extracted['sex']}' extracted by BioBERT model "
                    f"with confidence {confidence_str}."
                ),
            })
        
        for symptom in extracted.get("symptoms", []):
            confidence = confidence_scores.get("symptom")
            confidence_str = f"{confidence:.4f}" if confidence else "N/A"
            entity_explanations.append({
                "entity_type": "SYMPTOM",
                "value": symptom,
                "extraction_method": extraction_method,
                "confidence": confidence,
                "source_text": symptom,
                "explanation": (
                    f"Symptom '{symptom}' extracted by BioBERT model "
                    f"with confidence {confidence_str}."
                ),
            })
        
        if extracted.get("diagnosis"):
            confidence = confidence_scores.get("diagnosis")
            confidence_str = f"{confidence:.4f}" if confidence else "N/A"
            method_note = (
                "using hybrid rule-based fallback" if extraction_method == "hybrid"
                else "by BioBERT model"
            )
            entity_explanations.append({
                "entity_type": "DIAGNOSIS",
                "value": extracted["diagnosis"],
                "extraction_method": extraction_method,
                "confidence": confidence,
                "source_text": extracted["diagnosis"],
                "explanation": (
                    f"Diagnosis '{extracted['diagnosis']}' extracted {method_note} "
                    f"with confidence {confidence_str}."
                ),
            })
        
        for medication in extracted.get("medications", []):
            confidence = confidence_scores.get("medication")
            confidence_str = f"{confidence:.4f}" if confidence else "N/A"
            entity_explanations.append({
                "entity_type": "MEDICATION",
                "value": medication,
                "extraction_method": extraction_method,
                "confidence": confidence,
                "source_text": medication,
                "explanation": (
                    f"Medication '{medication}' extracted by BioBERT model "
                    f"with confidence {confidence_str}."
                ),
            })
        
        return entity_explanations
    
    def _explain_guidelines(
        self,
        guideline_assessment: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Build explanations for guideline compliance rules applied.
        
        Uses exact reasoning strings from GuidelineEvaluator to ensure
        100% grounding in actual guideline data. Never generates or
        hallucintates drug/test names. All explanations are directly
        from the guideline_assessment dict.
        
        Args:
            guideline_assessment: Assessment dict from GuidelineEvaluator.evaluate()
            
        Returns:
            List of guideline explanation dicts with rule type, item, explanation, severity.
        """
        guideline_explanations = []
        reasoning = guideline_assessment.get("reasoning", [])
        condition = guideline_assessment.get("condition_matched")
        
        for drug in guideline_assessment.get("recommended_drugs_present", []):
            # Find the exact reasoning string for this drug
            explanation_text = self._find_reasoning_for_item(drug, "present", reasoning)
            if not explanation_text:
                # Fallback only if reasoning not found, but never invent drug names
                explanation_text = f"Drug '{drug}' is present in the prescription."
            
            guideline_explanations.append({
                "rule_type": "RECOMMENDED_PRESENT",
                "item": drug,
                "explanation": explanation_text,
                "severity": "INFO",
            })
        
        for drug in guideline_assessment.get("recommended_drugs_missing", []):
            # Find the exact reasoning string for this drug
            explanation_text = self._find_reasoning_for_item(drug, "missing", reasoning)
            if not explanation_text:
                # Fallback only if reasoning not found, but never invent drug names
                explanation_text = f"Drug '{drug}' is missing from the prescription."
            
            guideline_explanations.append({
                "rule_type": "RECOMMENDED_MISSING",
                "item": drug,
                "explanation": explanation_text,
                "severity": "WARNING",
            })
        
        for drug in guideline_assessment.get("forbidden_drugs_present", []):
            # Find the exact reasoning string for this drug
            explanation_text = self._find_reasoning_for_item(drug, "forbidden", reasoning)
            if not explanation_text:
                # Fallback only if reasoning not found, but never invent drug names
                explanation_text = f"Drug '{drug}' is forbidden for this condition."
            
            guideline_explanations.append({
                "rule_type": "FORBIDDEN_PRESENT",
                "item": drug,
                "explanation": explanation_text,
                "severity": "CRITICAL",
            })
        
        for test in guideline_assessment.get("missing_tests", []):
            # Find the exact reasoning string for this test
            explanation_text = self._find_reasoning_for_item(test, "test", reasoning)
            if not explanation_text:
                # Fallback only if reasoning not found, but never invent test names
                explanation_text = f"Test '{test}' is missing from the clinical notes."
            
            guideline_explanations.append({
                "rule_type": "TEST_MISSING",
                "item": test,
                "explanation": explanation_text,
                "severity": "WARNING",
            })
        
        if guideline_assessment.get("overall_status") == "UNABLE_TO_ASSESS":
            # Find the reasoning string about condition not found
            explanation_text = "The extracted diagnosis is not found in the guideline database."
            for reason in reasoning:
                if "not found" in reason.lower() and "guideline" in reason.lower():
                    explanation_text = reason
                    break
            
            guideline_explanations.append({
                "rule_type": "CONDITION_NOT_FOUND",
                "item": guideline_assessment.get("condition_matched") or "Unknown",
                "explanation": explanation_text,
                "severity": "WARNING",
            })
        
        return guideline_explanations
    
    def _find_reasoning_for_item(
        self,
        item: str,
        item_type: str,
        reasoning: List[str],
    ) -> Optional[str]:
        """
        Find the exact reasoning string for a given drug or test.
        
        Searches the reasoning list from GuidelineEvaluator for the
        reasoning string that explains the compliance decision for
        the given item.
        
        Args:
            item: Drug or test name to search for
            item_type: Type of item ("present", "missing", "forbidden", "test")
            reasoning: List of reasoning strings from guideline_assessment
            
        Returns:
            Exact reasoning string if found, None otherwise
        """
        item_lower = item.lower()
        
        for reason in reasoning:
            reason_lower = reason.lower()
            # Check if this reasoning string mentions the item
            if item_lower in reason_lower:
                return reason
        
        return None
    
    def _build_summary(
        self,
        extracted: Dict[str, Any],
        guideline_assessment: Dict[str, Any],
    ) -> str:
        """
        Build a clinical summary paragraph.
        
        Writes a natural language summary suitable for clinician review,
        integrating extracted entities and guideline assessment.
        
        Args:
            extracted: Extracted entities dict
            guideline_assessment: Guideline assessment dict
            
        Returns:
            Plain English summary paragraph.
        """
        age = extracted.get("age") or "Unknown age"
        sex = extracted.get("sex") or "unknown sex"
        symptoms = extracted.get("symptoms", [])
        diagnosis = extracted.get("diagnosis") or "Unknown diagnosis"
        medications = extracted.get("medications", [])
        status = guideline_assessment.get("overall_status", "UNKNOWN")
        
        symptoms_str = ", ".join(symptoms) if symptoms else "not documented"
        medications_str = ", ".join(medications) if medications else "not documented"
        
        status_description = {
            "COMPLIANT": "compliant with clinical guidelines",
            "PARTIAL_COMPLIANCE": "partially compliant with clinical guidelines",
            "NON_COMPLIANT": "NON-COMPLIANT with clinical guidelines",
            "UNABLE_TO_ASSESS": "unable to assess against guidelines",
        }.get(status, "unknown compliance status")
        
        summary = (
            f"Clinical assessment for {age} {sex} presenting with {symptoms_str}. "
            f"Diagnosis: {diagnosis}. Current medications: {medications_str}. "
            f"Treatment assessment shows {status_description}."
        )
        
        return summary
    
    def _build_compliance_badge(self, status: str) -> str:
        """
        Build a visual compliance status badge.
        
        Args:
            status: Compliance status string
            
        Returns:
            Badge string with visual indicator
        """
        badges = {
            "COMPLIANT": "COMPLIANT ✓",
            "PARTIAL_COMPLIANCE": "PARTIAL COMPLIANCE ⚠",
            "NON_COMPLIANT": "NON-COMPLIANT ✗",
            "UNABLE_TO_ASSESS": "UNABLE TO ASSESS ?",
        }
        return badges.get(status, "UNKNOWN STATUS")
    
    def _build_extraction_method_summary(
        self,
        extraction_method: str,
        entity_explanations: List[Dict[str, Any]],
    ) -> str:
        """
        Build extraction method summary.
        
        Explains whether model or hybrid extraction was used and why.
        
        Args:
            extraction_method: Extraction method ("model" or "hybrid")
            entity_explanations: List of entity explanations
            
        Returns:
            Summary string explaining extraction approach
        """
        if extraction_method == "hybrid":
            diagnosis_found = any(
                e["entity_type"] == "DIAGNOSIS" for e in entity_explanations
            )
            if diagnosis_found:
                return (
                    "Entity extraction used hybrid approach: BioBERT model predictions "
                    "combined with rule-based fallback. Model-based NER failed to extract "
                    "diagnosis, so clinically-validated rule-based extraction was activated "
                    "as a safety mechanism to recover critical entities."
                )
            else:
                return (
                    "Entity extraction used hybrid approach combining model and rule-based methods. "
                    "Fallback extraction was activated due to low model confidence, though some "
                    "entities could not be recovered."
                )
        else:
            return (
                "Entity extraction used BioBERT deep learning model exclusively. "
                "All entities were predicted by the fine-tuned token classification model "
                "without rule-based fallback."
            )
    
    def _build_confidence_summary(self, confidence_scores: Dict[str, float]) -> str:
        """
        Build model confidence summary.
        
        Summarizes confidence scores across all entity types.
        
        Args:
            confidence_scores: Dict mapping entity type to confidence float
            
        Returns:
            Summary string of model confidence levels
        """
        if not confidence_scores:
            return "Model confidence scores not available."
        
        summary_parts = []
        for entity_type, confidence in confidence_scores.items():
            confidence_pct = confidence * 100
            if confidence >= 0.9:
                level = "high"
            elif confidence >= 0.75:
                level = "moderate"
            else:
                level = "low"
            summary_parts.append(
                f"{entity_type}: {confidence:.4f} ({level})"
            )
        
        confidence_text = "; ".join(summary_parts)
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        
        return (
            f"Model confidence across entities: {confidence_text}. "
            f"Average confidence: {avg_confidence:.4f}."
        )
    
    def _build_limitations(self) -> List[str]:
        """
        Build list of system limitations and disclaimers.
        
        Returns:
            List of limitation strings for clinician awareness
        """
        return [
            "This system is a clinical decision-support prototype only.",
            "Extracted entities may contain errors. Clinical verification required.",
            "Guideline compliance is based on documented medications only. "
            "Undocumented treatments are not assessed.",
            "Required tests assessment is limited as test results are not "
            "present in the provided clinical notes.",
            "This system does not replace clinical judgment.",
        ]


def load_explainer() -> ClinicalExplainer:
    """
    Factory function to create and return a ClinicalExplainer instance.
    
    Returns:
        Initialized ClinicalExplainer instance.
    """
    return ClinicalExplainer()


if __name__ == "__main__":
    """
    Test the clinical explainer with real end-to-end pipeline.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from src.modeling.predictor import ClinicalNERPredictor
    from src.guideline_engine.evaluator import GuidelineEvaluator
    import src.config as config
    
    test_notes = [
        {
            "label": "Pneumonia with Amoxicillin",
            "text": "45 year old male with fever and productive cough. Diagnosed with pneumonia. Started on amoxicillin."
        },
        {
            "label": "UTI with Ciprofloxacin (Forbidden Drug)",
            "text": "25 year old female with dysuria. Diagnosed urinary tract infection. Prescribed ciprofloxacin."
        },
        {
            "label": "Meningitis with Ceftriaxone",
            "text": "30 year old female with neck stiffness and fever. Impression meningitis. Given ceftriaxone."
        }
    ]
    
    print("=" * 80)
    print("Clinical Explainability Module - End-to-End Pipeline Test")
    print("=" * 80)
    
    try:
        predictor = ClinicalNERPredictor(config.MODEL_SAVE_DIR)
        evaluator = GuidelineEvaluator(config.GUIDELINES_PATH)
        explainer = ClinicalExplainer()
        print("All pipeline components loaded successfully\n")
    except Exception as e:
        print(f"Failed to load pipeline components: {e}")
        exit(1)
    
    for note in test_notes:
        print("-" * 80)
        print(f"Test: {note['label']}")
        print("-" * 80)
        print(f"Clinical Note: {note['text']}\n")
        
        try:
            prediction = predictor.predict(note["text"])
            guideline_result = evaluator.evaluate(prediction["extracted"])
            explanation = explainer.explain(prediction, guideline_result)
            
            print(f"Compliance Badge: {explanation['compliance_badge']}")
            print(f"\nOverall Summary:\n{explanation['overall_summary']}")
            print(f"\nExtraction Method:\n{explanation['extraction_method_summary']}")
            print(f"\nEntity Explanations:")
            for e in explanation["entity_explanations"]:
                print(f"  [{e['entity_type']}] {e['explanation']}")
            print(f"\nGuideline Explanations:")
            for g in explanation["guideline_explanations"]:
                print(f"  [{g['severity']}] {g['explanation']}")
            print(f"\nLimitations:")
            for l in explanation["limitations"]:
                print(f"  • {l}")
        except Exception as e:
            print(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 80)
    print("End-to-end pipeline test completed")
    print("=" * 80)

"""
Deterministic clinical guideline evaluation engine.

This module provides rule-based evaluation of clinical guidelines against
extracted entities from clinical notes. No AI or LLM inference is used.
All decisions are based on explicit guideline rules and entity matching.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger


class GuidelineEvaluator:
    """
    Evaluates clinical guidelines against extracted entities from clinical notes.
    
    Uses deterministic, rule-based logic to check medication compliance,
    required tests, and other guideline criteria.
    
    Attributes:
        guidelines (dict): Loaded guideline rules indexed by condition name.
        guideline_version (str): Version identifier for the loaded guidelines.
    """
    
    def __init__(self, guidelines_path: Path) -> None:
        """
        Initialize the guideline evaluator with guidelines from a JSON file.
        
        Args:
            guidelines_path: Path object pointing to guidelines.json file.
            
        Raises:
            FileNotFoundError: If guidelines file does not exist.
            json.JSONDecodeError: If guidelines file is not valid JSON.
        """
        if not guidelines_path.exists():
            raise FileNotFoundError(f"Guidelines file not found: {guidelines_path}")
        
        with open(guidelines_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "guidelines" in data:
            self.guidelines = data.get("guidelines", {})
            self.guideline_version = data.get("version", "1.0.0")
        else:
            self.guidelines = data
            self.guideline_version = "1.0.0"
    
    def evaluate(self, extracted_entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate extracted clinical entities against loaded guidelines.
        
        Performs deterministic checks on diagnosis, medications, and tests
        against explicit guideline rules.
        
        Args:
            extracted_entities: Dictionary with keys:
                - age (int or None): Patient age
                - sex (str or None): Patient sex
                - symptoms (list): List of symptom strings
                - diagnosis (str or None): Primary diagnosis
                - medications (list): List of prescribed medication names
                
        Returns:
            Dictionary with keys:
                - condition_matched (str or None): Matched guideline condition name
                - guideline_version (str): Version of guidelines used
                - recommended_drugs_present (list): Recommended drugs that are prescribed
                - recommended_drugs_missing (list): Recommended drugs that are missing
                - forbidden_drugs_present (list): Forbidden drugs that are prescribed
                - required_tests (list): Required tests per guideline
                - missing_tests (list): Required tests not mentioned in notes
                - overall_status (str): COMPLIANT, PARTIAL_COMPLIANCE, NON_COMPLIANT, UNABLE_TO_ASSESS
                - reasoning (list): List of decision explanations
                - raw_guideline (dict): Complete guideline rule for matched condition
        """
        reasoning: List[str] = []
        
        diagnosis = extracted_entities.get("diagnosis")
        medications = extracted_entities.get("medications", [])
        
        normalized_diagnosis = self._normalize(diagnosis) if diagnosis else None
        normalized_medications = [self._normalize(med) for med in medications]
        
        result: Dict[str, Any] = {
            "condition_matched": None,
            "guideline_version": self.guideline_version,
            "recommended_drugs_present": [],
            "recommended_drugs_missing": [],
            "forbidden_drugs_present": [],
            "required_tests": [],
            "missing_tests": [],
            "overall_status": "UNABLE_TO_ASSESS",
            "reasoning": reasoning,
            "raw_guideline": {},
        }
        
        if not normalized_diagnosis:
            reasoning.append("No diagnosis found in extracted entities.")
            result["overall_status"] = "UNABLE_TO_ASSESS"
            return result
        
        guideline = self.get_guideline(normalized_diagnosis)
        
        if not guideline:
            reasoning.append(
                f"Diagnosis '{diagnosis}' not found in guidelines database. "
                f"Unable to evaluate compliance."
            )
            result["overall_status"] = "UNABLE_TO_ASSESS"
            result["condition_matched"] = None
            return result
        
        result["condition_matched"] = normalized_diagnosis
        result["raw_guideline"] = guideline
        reasoning.append(f"Matched guideline for condition: {normalized_diagnosis}")
        
        recommended_drugs = [
            self._normalize(drug) 
            for drug in guideline.get("recommended_drugs", [])
        ]
        forbidden_drugs = [
            self._normalize(drug)
            for drug in guideline.get("avoid_drugs", [])
        ]
        required_tests = guideline.get("required_tests", [])

        logger.debug(
            "Comparing normalized medications={} against recommended_drugs={} and avoid_drugs={}",
            normalized_medications,
            recommended_drugs,
            forbidden_drugs,
        )
        
        result["required_tests"] = required_tests
        
        for med in recommended_drugs:
            if med in normalized_medications:
                result["recommended_drugs_present"].append(med)
                reasoning.append(
                    f"Drug {med} is recommended for {normalized_diagnosis} and is present in prescription. ACCEPTED."
                )
            else:
                result["recommended_drugs_missing"].append(med)
                reasoning.append(
                    f"Drug {med} is recommended for {normalized_diagnosis} but not found in prescription. MISSING."
                )
        
        for med in forbidden_drugs:
            if med in normalized_medications:
                result["forbidden_drugs_present"].append(med)
                reasoning.append(
                    f"Drug {med} is explicitly FORBIDDEN for {normalized_diagnosis} by guideline. FLAGGED."
                )
        
        # Check required tests (not extracted from notes, so all are missing for now)
        result["missing_tests"] = required_tests.copy()
        for test in required_tests:
            reasoning.append(
                f"Test {test} is required for {normalized_diagnosis} but not documented. MISSING."
            )
        
        has_forbidden = len(result["forbidden_drugs_present"]) > 0
        all_recommended_present = len(result["recommended_drugs_missing"]) == 0
        all_tests_done = len(result["missing_tests"]) == 0
        
        if has_forbidden:
            result["overall_status"] = "NON_COMPLIANT"
        elif all_recommended_present and all_tests_done:
            result["overall_status"] = "COMPLIANT"
        else:
            result["overall_status"] = "PARTIAL_COMPLIANCE"
        
        return result
    
    def _normalize(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Converts to lowercase and strips leading/trailing whitespace.
        
        Args:
            text: Text string to normalize.
            
        Returns:
            Normalized text string.
        """
        if not isinstance(text, str):
            return str(text).lower().strip()
        return text.lower().strip()
    
    def get_guideline(self, condition: str) -> Dict[str, Any]:
        """
        Retrieve the complete guideline rule for a condition.
        
        Args:
            condition: Normalized condition name (lowercase).
            
        Returns:
            Dictionary containing guideline rule, or empty dict if not found.
        """
        normalized_condition = self._normalize(condition)
        
        if normalized_condition in self.guidelines:
            return self.guidelines[normalized_condition]
        
        return {}


def load_evaluator(guidelines_path: Path) -> GuidelineEvaluator:
    """
    Factory function to create and return a GuidelineEvaluator instance.
    
    Args:
        guidelines_path: Path object to guidelines.json file.
        
    Returns:
        Initialized GuidelineEvaluator instance.
        
    Raises:
        FileNotFoundError: If guidelines file does not exist.
        json.JSONDecodeError: If guidelines file is not valid JSON.
    """
    return GuidelineEvaluator(guidelines_path)


if __name__ == "__main__":
    """
    Test the guideline evaluator with sample clinical cases.
    """
    from src.config import GUIDELINES_PATH
    
    print("=" * 80)
    print("Clinical Guideline Evaluation Engine - Test Suite")
    print("=" * 80)
    
    try:
        evaluator = load_evaluator(GUIDELINES_PATH)
        print(f"Loaded guidelines from: {GUIDELINES_PATH}")
        print(f"Guideline version: {evaluator.guideline_version}")
    except FileNotFoundError as e:
        print(f"Error loading guidelines: {e}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing guidelines JSON: {e}")
        exit(1)
    
    print("\n" + "-" * 80)
    
    print("\nTest Case 1: Pneumonia with appropriate medication")
    print("-" * 80)
    case_1 = {
        "age": 52,
        "sex": "M",
        "symptoms": ["cough", "fever", "dyspnea"],
        "diagnosis": "pneumonia",
        "medications": ["amoxicillin"],
    }
    result_1 = evaluator.evaluate(case_1)
    print(f"Diagnosis: {case_1['diagnosis']}")
    print(f"Prescribed: {case_1['medications']}")
    print(f"Overall Status: {result_1['overall_status']}")
    print("\nReasoning:")
    for reason in result_1["reasoning"]:
        print(f"  • {reason}")
    
    print("\n" + "-" * 80)
    
    print("\nTest Case 2: Urinary Tract Infection with appropriate medication")
    print("-" * 80)
    case_2 = {
        "age": 35,
        "sex": "F",
        "symptoms": ["dysuria", "frequency"],
        "diagnosis": "urinary tract infection",
        "medications": ["ciprofloxacin"],
    }
    result_2 = evaluator.evaluate(case_2)
    print(f"Diagnosis: {case_2['diagnosis']}")
    print(f"Prescribed: {case_2['medications']}")
    print(f"Overall Status: {result_2['overall_status']}")
    print("\nReasoning:")
    for reason in result_2["reasoning"]:
        print(f"  • {reason}")
    
    print("\n" + "-" * 80)
    
    print("\nTest Case 3: Unknown condition not in guidelines")
    print("-" * 80)
    case_3 = {
        "age": 28,
        "sex": "M",
        "symptoms": ["fever", "rash"],
        "diagnosis": "dengue fever",
        "medications": ["paracetamol"],
    }
    result_3 = evaluator.evaluate(case_3)
    print(f"Diagnosis: {case_3['diagnosis']}")
    print(f"Prescribed: {case_3['medications']}")
    print(f"Overall Status: {result_3['overall_status']}")
    print("\nReasoning:")
    for reason in result_3["reasoning"]:
        print(f"  • {reason}")
    
    print("\n" + "=" * 80)
    print("Test suite completed")
    print("=" * 80)

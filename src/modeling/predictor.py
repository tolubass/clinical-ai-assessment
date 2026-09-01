"""
Clinical NER inference engine using fine-tuned BioBERT model.

This module loads a trained BioBERT token classification model and runs
inference on raw clinical notes to extract structured clinical entities
(age, sex, symptoms, diagnosis, medications).

All inference is deterministic and does not use any LLM or generative AI.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from loguru import logger

from src.preprocessing.text_processor import ClinicalTextPreprocessor, ClinicalAnnotator
import src.config as config


class ClinicalNERPredictor:
    """
    Production inference engine for clinical NER using fine-tuned BioBERT.
    
    Loads a trained token classification model and runs inference on clinical
    notes to extract structured entities: age, sex, symptoms, diagnosis,
    medications.
    
    Attributes:
        tokenizer: Loaded tokenizer from model directory.
        model: Loaded token classification model.
        device: Torch device (cuda or cpu).
        metadata: Model metadata dict.
        preprocessor: ClinicalTextPreprocessor instance.
    """
    
    def __init__(self, model_dir: Path, device: Optional[str] = None) -> None:
        """
        Initialize the clinical NER predictor.
        
        Args:
            model_dir: Path to directory containing model files, tokenizer, config.
            device: Device to load model on ('cuda' or 'cpu'). Auto-detects if None.
            
        Raises:
            FileNotFoundError: If required model files not found.
            RuntimeError: If model fails to load.
        """
        self.model_dir = Path(model_dir)
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading BioBERT model from: {self.model_dir}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForTokenClassification.from_pretrained(
                str(self.model_dir)
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Load metadata
            metadata_path = self.model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {"version": config.MODEL_VERSION}
            
            logger.info(
                f"Model loaded successfully. Version: {self.metadata.get('version')}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
        
        # Initialize preprocessor
        self.preprocessor = ClinicalTextPreprocessor()
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on clinical note text to extract entities.
        
        Cleans text, tokenizes, runs model inference, aligns predictions
        to word level, and extracts structured entities.
        
        Args:
            text: Raw clinical note text.
            
        Returns:
            Dictionary with keys:
                - tokens (list): Word-level tokens
                - ner_tags (list): Predicted BIO tags for each token
                - extracted (dict): Structured entities (age, sex, symptoms, diagnosis, medications)
                - confidence_scores (dict): Average confidence per entity type
                - model_version (str): Model version used
                - inference_latency_ms (float): Inference time in milliseconds
        """
        start_time = time.time()
        
        # Handle edge case: empty text
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for prediction")
            return {
                "tokens": [],
                "ner_tags": [],
                "extracted": {
                    "age": None,
                    "sex": None,
                    "symptoms": [],
                    "diagnosis": None,
                    "medications": [],
                },
                "confidence_scores": {},
                "model_version": self.metadata.get("version", config.MODEL_VERSION),
                "inference_latency_ms": 0.0,
            }
        
        # Clean text
        cleaned_text = self.preprocessor.clean(text)
        logger.debug(f"Cleaned text: {cleaned_text}")
        
        # Tokenize at word level
        word_tokens = self.preprocessor.tokenize(cleaned_text)
        if not word_tokens:
            logger.warning("No tokens after preprocessing")
            return {
                "tokens": [],
                "ner_tags": [],
                "extracted": {
                    "age": None,
                    "sex": None,
                    "symptoms": [],
                    "diagnosis": None,
                    "medications": [],
                },
                "confidence_scores": {},
                "model_version": self.metadata.get("version", config.MODEL_VERSION),
                "inference_latency_ms": 0.0,
            }
        
        # Tokenize for model (includes subword tokens)
        encodings = self.tokenizer(
            word_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        )
        
        # Move to device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
        
        # Get predictions and probabilities
        predictions = torch.argmax(logits, dim=2)
        probabilities = torch.softmax(logits, dim=2)
        
        # Align subword predictions back to word level
        word_ids = encodings.word_ids()
        word_level_predictions = []
        word_level_probabilities = []
        
        for word_idx in range(len(word_tokens)):
            # Get the first subword token for this word
            subword_idx = None
            for idx, wid in enumerate(word_ids):
                if wid == word_idx:
                    subword_idx = idx
                    break
            
            if subword_idx is not None:
                pred_id = predictions[0, subword_idx].item()
                prob = probabilities[0, subword_idx].max().item()
                word_level_predictions.append(pred_id)
                word_level_probabilities.append(prob)
        
        # Convert prediction IDs to label strings
        id2label = self.model.config.id2label
        ner_tags = [id2label[pred_id] for pred_id in word_level_predictions]
        
        logger.debug(f"Tokens: {word_tokens}")
        logger.debug(f"NER Tags: {ner_tags}")
        
        # Extract structured entities
        extracted = self._extract_entities(word_tokens, ner_tags, word_level_probabilities)
        
        # Post-processing: merge consecutive single-word symptoms
        extracted["symptoms"] = self._merge_consecutive_symptoms(
            extracted["symptoms"], word_tokens
        )
        
        extraction_method = "model"
        
        # Hybrid fallback: if diagnosis is None, use rule-based extraction
        if extracted["diagnosis"] is None:
            logger.warning(
                "Model failed to extract diagnosis. Activating hybrid fallback with "
                "rule-based annotation."
            )
            try:
                annotator = ClinicalAnnotator()
                fallback_result = annotator.annotate("FALLBACK", cleaned_text)
                fallback_extracted = fallback_result.get("extracted", {})
                
                if extracted["diagnosis"] is None:
                    extracted["diagnosis"] = fallback_extracted.get("diagnosis")
                
                if extracted["age"] is None:
                    extracted["age"] = fallback_extracted.get("age")
                
                if not extracted["medications"]:
                    extracted["medications"] = fallback_extracted.get("medications", [])
                
                if not extracted["symptoms"]:
                    extracted["symptoms"] = fallback_extracted.get("symptoms", [])
                
                extraction_method = "hybrid"
                
                if extracted["diagnosis"]:
                    logger.info(f"Hybrid fallback: extracted diagnosis='{extracted['diagnosis']}'")
                    
            except Exception as e:
                logger.error(f"Hybrid fallback failed: {e}")
        
        # Compute confidence scores
        confidence_scores = self._compute_confidence(ner_tags, word_level_probabilities)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            "tokens": word_tokens,
            "ner_tags": ner_tags,
            "extracted": extracted,
            "confidence_scores": confidence_scores,
            "model_version": self.metadata.get("version", config.MODEL_VERSION),
            "inference_latency_ms": round(latency_ms, 2),
            "extraction_method": extraction_method,
        }
        
        logger.info(f"Prediction completed in {latency_ms:.2f}ms using {extraction_method} extraction")
        
        return result
    
    def _extract_entities(
        self,
        tokens: List[str],
        tags: List[str],
        probabilities: List[float],
    ) -> Dict[str, Any]:
        """
        Extract structured entities from BIO-tagged tokens.
        
        Properly groups consecutive B- and I- tags into complete multi-word spans.
        For example: tokens=['45','year','old'] + tags=['B-AGE','I-AGE','I-AGE']
        produces age="45 year old" (not None).
        
        Args:
            tokens: Word-level token strings.
            tags: BIO tag strings for each token.
            probabilities: Confidence scores for each prediction.
            
        Returns:
            Dictionary with extracted entities:
                - age (str or None): Joined age span
                - sex (str or None): First sex entity
                - symptoms (list): All symptom spans (each is joined tokens)
                - diagnosis (str or None): Joined diagnosis span
                - medications (list): All medication spans (each is joined tokens)
        """
        entities = {
            "age": None,
            "sex": None,
            "symptoms": [],
            "diagnosis": None,
            "medications": [],
        }
        
        # Parse BIO tags into entity spans
        current_entity_type = None
        current_tokens = []
        
        for token, tag in zip(tokens, tags):
            if tag == "O":
                # End current span if entity is active
                if current_entity_type is not None and current_tokens:
                    entity_text = " ".join(current_tokens)
                    self._add_entity(entities, current_entity_type, entity_text)
                    logger.debug(
                        f"Extracted {current_entity_type}: '{entity_text}'"
                    )
                current_entity_type = None
                current_tokens = []
                
            elif tag.startswith("B-"):
                # New entity begins - save previous span first
                if current_entity_type is not None and current_tokens:
                    entity_text = " ".join(current_tokens)
                    self._add_entity(entities, current_entity_type, entity_text)
                    logger.debug(
                        f"Extracted {current_entity_type}: '{entity_text}'"
                    )
                
                # Start new span
                current_entity_type = tag[2:]  # Remove "B-" prefix
                current_tokens = [token]
                
            elif tag.startswith("I-"):
                # Continue current entity
                entity_type = tag[2:]  # Remove "I-" prefix
                
                if current_entity_type == entity_type and current_tokens:
                    # Continuation of same entity - add token
                    current_tokens.append(token)
                else:
                    # Malformed sequence (I- without matching B- or different type)
                    # Save previous if exists
                    if current_entity_type is not None and current_tokens:
                        entity_text = " ".join(current_tokens)
                        self._add_entity(entities, current_entity_type, entity_text)
                        logger.debug(
                            f"Extracted {current_entity_type}: '{entity_text}'"
                        )
                    # Treat I- as beginning of new entity
                    current_entity_type = entity_type
                    current_tokens = [token]
        
        # Handle last active span at end of tokens
        if current_entity_type is not None and current_tokens:
            entity_text = " ".join(current_tokens)
            self._add_entity(entities, current_entity_type, entity_text)
            logger.debug(f"Extracted {current_entity_type}: '{entity_text}'")
        
        return entities
    
    def _add_entity(self, entities: Dict, entity_type: str, entity_text: str) -> None:
        """
        Add extracted entity to the entities dictionary.
        
        Args:
            entities: Dictionary to add entity to.
            entity_type: Entity type (AGE, SEX, SYMPTOM, DIAGNOSIS, MEDICATION).
            entity_text: Entity text span.
        """
        entity_type_lower = entity_type.lower()
        
        if entity_type_lower == "age":
            if entities["age"] is None:
                entities["age"] = entity_text
        elif entity_type_lower == "sex":
            if entities["sex"] is None:
                entities["sex"] = entity_text
        elif entity_type_lower == "symptom":
            if entity_text not in entities["symptoms"]:
                entities["symptoms"].append(entity_text)
        elif entity_type_lower == "diagnosis":
            if entities["diagnosis"] is None:
                entities["diagnosis"] = entity_text
        elif entity_type_lower == "medication":
            if entity_text not in entities["medications"]:
                entities["medications"].append(entity_text)
    
    def _merge_consecutive_symptoms(
        self,
        symptoms: List[str],
        word_tokens: List[str],
    ) -> List[str]:
        """
        Merge consecutive single-word symptoms that appear as adjacent tokens.
        
        Handles cases where the model predicts B-SYMPTOM B-SYMPTOM instead of
        B-SYMPTOM I-SYMPTOM for multi-word symptoms, causing splits like
        ['productive', 'cough'] instead of ['productive cough'].
        
        Args:
            symptoms: List of extracted symptom strings.
            word_tokens: List of original word-level tokens.
            
        Returns:
            List of symptom strings with consecutive adjacent symptoms merged.
        """
        if not symptoms or len(symptoms) < 2:
            return symptoms
        
        # Create a mapping of symptom text to token positions
        symptom_positions = {}
        for symptom in symptoms:
            symptom_words = symptom.lower().split()
            # Find this symptom in the token list
            for i in range(len(word_tokens) - len(symptom_words) + 1):
                tokens_match = True
                for j, word in enumerate(symptom_words):
                    if word_tokens[i + j].lower() != word:
                        tokens_match = False
                        break
                if tokens_match:
                    symptom_positions[symptom] = (i, i + len(symptom_words) - 1)
                    break
        
        # Merge symptoms that are adjacent in token list
        merged = []
        skip_indices = set()
        
        for i, symptom in enumerate(symptoms):
            if i in skip_indices:
                continue
            
            # Check if next symptom is adjacent
            if i + 1 < len(symptoms):
                curr_symptom = symptom
                next_symptom = symptoms[i + 1]
                
                if curr_symptom in symptom_positions and next_symptom in symptom_positions:
                    curr_start, curr_end = symptom_positions[curr_symptom]
                    next_start, next_end = symptom_positions[next_symptom]
                    
                    # If current symptom ends where next symptom begins (adjacent)
                    if curr_end + 1 == next_start:
                        merged_symptom = " ".join(
                            word_tokens[curr_start : next_end + 1]
                        )
                        merged.append(merged_symptom)
                        skip_indices.add(i + 1)
                        logger.debug(
                            f"Merged consecutive symptoms: '{curr_symptom}' + "
                            f"'{next_symptom}' -> '{merged_symptom}'"
                        )
                        continue
            
            merged.append(symptom)
        
        return merged
    
    def _compute_confidence(
        self,
        tags: List[str],
        probabilities: List[float],
    ) -> Dict[str, float]:
        """
        Compute average confidence score per entity type.
        
        Args:
            tags: BIO tag strings for each token.
            probabilities: Confidence scores for each prediction.
            
        Returns:
            Dictionary mapping entity type to average confidence (0-1).
        """
        confidence_scores: Dict[str, List[float]] = {}
        
        for tag, prob in zip(tags, probabilities):
            if tag == "O":
                continue
            
            # Extract entity type (remove B- or I- prefix)
            if tag.startswith("B-") or tag.startswith("I-"):
                entity_type = tag[2:].lower()
                if entity_type not in confidence_scores:
                    confidence_scores[entity_type] = []
                confidence_scores[entity_type].append(prob)
        
        # Compute averages
        result = {}
        for entity_type, probs in confidence_scores.items():
            if probs:
                avg_confidence = sum(probs) / len(probs)
                result[entity_type] = round(avg_confidence, 4)
        
        return result


def load_predictor(model_dir: Path) -> ClinicalNERPredictor:
    """
    Factory function to create and return a ClinicalNERPredictor instance.
    
    Args:
        model_dir: Path to directory containing trained model files.
        
    Returns:
        Initialized ClinicalNERPredictor instance.
        
    Raises:
        FileNotFoundError: If model directory or required files not found.
        RuntimeError: If model fails to load.
    """
    return ClinicalNERPredictor(model_dir)


if __name__ == "__main__":
    """
    Test the clinical NER predictor with sample clinical notes.
    """
    print("=" * 80)
    print("Clinical NER Predictor - Inference Test Suite")
    print("=" * 80)
    
    # Load predictor
    try:
        predictor = load_predictor(config.MODEL_SAVE_DIR)
        print(f"✓ Model loaded from: {config.MODEL_SAVE_DIR}")
    except FileNotFoundError as e:
        print(f"✗ Model not found: {e}")
        print(f"  Expected location: {config.MODEL_SAVE_DIR}")
        exit(1)
    except RuntimeError as e:
        print(f"✗ Model loading error: {e}")
        exit(1)
    
    print("\n" + "-" * 80)
    
    # Test cases
    test_notes = [
        (
            "Pneumonia Case",
            "45 year old male with fever and productive cough. "
            "Diagnosed with pneumonia. Started on amoxicillin.",
        ),
        (
            "Meningitis Case",
            "30 year old female with neck stiffness and fever. "
            "Impression meningitis. Given ceftriaxone.",
        ),
        (
            "UTI Case",
            "25 year old female with dysuria. Diagnosed urinary tract infection. "
            "Prescribed ciprofloxacin.",
        ),
    ]
    
    for case_name, clinical_note in test_notes:
        print(f"\nTest Case: {case_name}")
        print("-" * 80)
        print(f"Clinical Note: {clinical_note}")
        print()
        
        # Run prediction
        result = predictor.predict(clinical_note)
        
        # Print results
        print(f"Tokens ({len(result['tokens'])}): {result['tokens']}")
        print(f"NER Tags: {result['ner_tags']}")
        print()
        
        print("Extracted Entities:")
        extracted = result["extracted"]
        print(f"  Age: {extracted['age']}")
        print(f"  Sex: {extracted['sex']}")
        print(f"  Symptoms: {extracted['symptoms']}")
        print(f"  Diagnosis: {extracted['diagnosis']}")
        print(f"  Medications: {extracted['medications']}")
        print()
        
        print("Confidence Scores:")
        for entity_type, confidence in result["confidence_scores"].items():
            print(f"  {entity_type}: {confidence:.4f}")
        print()
        
        print(f"Model Version: {result['model_version']}")
        print(f"Inference Latency: {result['inference_latency_ms']:.2f} ms")
        print()
    
    print("=" * 80)
    print("Test suite completed")
    print("=" * 80)

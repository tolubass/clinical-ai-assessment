"""
Clinical text preprocessing and BIO annotation pipeline.
Converts raw clinical notes into NER training data.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# ── Clinical vocabulary ──────────────────────────────────────

SYMPTOM_TERMS = [
    "fever", "cough", "productive cough", "headache", "neck stiffness",
    "chest pain", "shortness of breath", "wheezing", "dysuria",
    "lower abdominal pain", "weight loss", "night sweats", "polyuria",
    "polydipsia", "blurred vision", "severe headache", "sore throat",
    "vomiting", "diarrhea", "breathlessness", "chest tightness",
    "chest discomfort", "burning urination", "chronic cough", "fatigue",
    "frequency", "radiating to left arm", "radiating to arm",
]

DIAGNOSIS_TRIGGERS = [
    "diagnosed with", "diagnosed", "impression", "suspected",
    "impression:", "diagnosis:", "suggest", "suggests",
]

MEDICATION_TRIGGERS = [
    "started on", "started", "given", "prescribed",
    "treatment with", "initiated", "commenced",
]

MEDICATION_STOPWORDS = {
    "on", "with", "a", "an", "the", "and", "or"
}

SEX_TERMS = {
    "male": "male",
    "female": "female",
    "man": "male",
    "woman": "female",
    "boy": "male",
    "girl": "female",
}


class ClinicalTextPreprocessor:
    """
    Cleans and normalises clinical free-text notes.
    Designed to preserve clinically meaningful tokens
    such as numbers, units, drug names and abbreviations.
    """

    def clean(self, text: str) -> str:
        """Light-touch cleaning that preserves clinical meaning."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Word-level tokenization preserving clinical structure.
        Splits on whitespace and punctuation boundaries
        while keeping hyphenated drug names intact.
        """
        tokens = re.findall(r"[\w\-]+|[^\w\s]", text)
        return tokens


class ClinicalAnnotator:
    """
    Rule-based BIO annotator for clinical NER.

    Produces BIO-tagged token sequences from raw clinical notes.
    This annotation forms the supervised training data for
    the downstream BioBERT fine-tuning pipeline.

    Entities annotated:
        AGE        — patient age
        SEX        — patient sex
        SYMPTOM    — presenting symptoms
        DIAGNOSIS  — documented diagnosis
        MEDICATION — prescribed medications
    """

    def __init__(self):
        self.preprocessor = ClinicalTextPreprocessor()

    def _find_age(
        self, text: str
    ) -> Optional[Tuple[int, int, str]]:
        """Extract age span from text."""
        pattern = r"(\d{1,3})\s*year\s*old"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.start(), match.end(), match.group().strip()
        return None

    def _find_sex(
        self, text: str
    ) -> Optional[Tuple[int, int, str]]:
        """Extract sex span from text."""
        pattern = r"\b(male|female|man|woman|boy|girl)\b"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.start(), match.end(), match.group().strip()
        return None

    def _find_symptoms(
        self, text: str
    ) -> List[Tuple[int, int, str]]:
        """Extract all symptom spans."""
        spans = []
        for symptom in sorted(SYMPTOM_TERMS, key=len, reverse=True):
            pattern = r"\b" + re.escape(symptom) + r"\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                overlap = any(
                    match.start() < e and match.end() > s
                    for s, e, _ in spans
                )
                if not overlap:
                    spans.append(
                        (match.start(), match.end(), match.group().strip())
                    )
        return sorted(spans, key=lambda x: x[0])

    def _find_diagnosis(
        self, text: str
    ) -> Optional[Tuple[int, int, str]]:
        """Extract diagnosis span following trigger words."""
        for trigger in DIAGNOSIS_TRIGGERS:
            pattern = (
                r"\b"
                + re.escape(trigger)
                + r"\s+([a-zA-Z\s\-]+?)(?=\.|,|\band\b|$)"
            )
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                diag = match.group(1).strip()
                start = match.start(1)
                end = match.end(1)
                return start, end, diag
        return None

    def _find_medications(
        self, text: str
    ) -> List[Tuple[int, int, str]]:
        """Extract medication spans following trigger words."""
        spans = []
        for trigger in MEDICATION_TRIGGERS:
            pattern = (
                r"\b"
                + re.escape(trigger)
                + r"\s+([\w\-]+(?:\s+(?:and|,)\s+[\w\-]+)*)"
            )
            for match in re.finditer(pattern, text, re.IGNORECASE):
                meds_text = match.group(1)
                meds = re.split(r"\s+and\s+|,\s*", meds_text)
                pos = match.start(1)
                for med in meds:
                    med = med.strip()
                    # Skip stopwords and prepositions
                    if not med or med.lower() in MEDICATION_STOPWORDS:
                        pos += len(med) + 1
                        continue
                    med_match = re.search(
                        re.escape(med), text[pos:], re.IGNORECASE
                    )
                    if med_match:
                        abs_start = pos + med_match.start()
                        abs_end = pos + med_match.end()
                        overlap = any(
                            abs_start < e and abs_end > s
                            for s, e, _ in spans
                        )
                        if not overlap:
                            spans.append(
                                (abs_start, abs_end, med)
                            )
                        pos = abs_end
        return sorted(spans, key=lambda x: x[0])

    def _build_char_label_map(
        self,
        text: str,
        entity_spans: Dict[str, List[Tuple[int, int, str]]],
    ) -> List[str]:
        """
        Build a character-level label array for the full text.
        Each character position maps to an entity type or O.
        """
        char_labels = ["O"] * len(text)

        def mark(start, end, entity_type):
            for i in range(start, end):
                if i < len(char_labels):
                    char_labels[i] = entity_type

        if entity_spans.get("AGE"):
            for s, e, _ in entity_spans["AGE"]:
                mark(s, e, "AGE")

        if entity_spans.get("SEX"):
            for s, e, _ in entity_spans["SEX"]:
                mark(s, e, "SEX")

        for s, e, _ in entity_spans.get("SYMPTOM", []):
            mark(s, e, "SYMPTOM")

        if entity_spans.get("DIAGNOSIS"):
            for s, e, _ in entity_spans["DIAGNOSIS"]:
                mark(s, e, "DIAGNOSIS")

        for s, e, _ in entity_spans.get("MEDICATION", []):
            mark(s, e, "MEDICATION")

        return char_labels

    def annotate(
        self, note_id: str, text: str
    ) -> Dict:
        """
        Full annotation pipeline for a single clinical note.
        Returns a dict with tokens and BIO tags.
        """
        cleaned = self.preprocessor.clean(text)
        tokens = self.preprocessor.tokenize(cleaned)

        age_span = self._find_age(cleaned)
        sex_span = self._find_sex(cleaned)
        symptom_spans = self._find_symptoms(cleaned)
        diag_span = self._find_diagnosis(cleaned)
        med_spans = self._find_medications(cleaned)

        entity_spans = {
            "AGE": [age_span] if age_span else [],
            "SEX": [sex_span] if sex_span else [],
            "SYMPTOM": symptom_spans,
            "DIAGNOSIS": [diag_span] if diag_span else [],
            "MEDICATION": med_spans,
        }

        char_labels = self._build_char_label_map(cleaned, entity_spans)

        bio_tags = []
        pos = 0
        for token in tokens:
            token_start = cleaned.find(token, pos)
            if token_start == -1:
                bio_tags.append("O")
                continue
            token_end = token_start + len(token)

            token_char_labels = char_labels[token_start:token_end]
            entity_type = "O"
            for cl in token_char_labels:
                if cl != "O":
                    entity_type = cl
                    break

            if entity_type == "O":
                bio_tags.append("O")
            else:
                if (
                    bio_tags
                    and bio_tags[-1] in (
                        f"B-{entity_type}", f"I-{entity_type}"
                    )
                ):
                    bio_tags.append(f"I-{entity_type}")
                else:
                    bio_tags.append(f"B-{entity_type}")

            pos = token_end

        extracted = {
            "age": age_span[2] if age_span else None,
            "sex": sex_span[2].lower() if sex_span else None,
            "symptoms": [s[2] for s in symptom_spans],
            "diagnosis": diag_span[2].strip().lower() if diag_span else None,
            "medications": [
                m[2].lower() for m in med_spans
                if m[2].lower() not in MEDICATION_STOPWORDS
            ],
        }

        return {
            "note_id": note_id,
            "text": cleaned,
            "tokens": tokens,
            "ner_tags": bio_tags,
            "extracted": extracted,
        }


def annotate_dataset(
    notes_path: Path,
    output_path: Path,
) -> List[Dict]:
    """
    Annotate all clinical notes and save to disk.
    """
    annotator = ClinicalAnnotator()

    with open(notes_path, "r", encoding="utf-8") as f:
        notes = json.load(f)

    annotated = []
    for note in notes:
        result = annotator.annotate(note["note_id"], note["text"])
        annotated.append(result)
        logger.info(
            f"Annotated {note['note_id']}: "
            f"diagnosis={result['extracted']['diagnosis']}, "
            f"meds={result['extracted']['medications']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2)

    logger.info(
        f"Annotation complete. {len(annotated)} notes saved to {output_path}"
    )
    return annotated


if __name__ == "__main__":
    import sys
    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent.parent)
    )
    from src.config import CLINICAL_NOTES_PATH, ANNOTATED_DATA_PATH

    logging.basicConfig(level=logging.INFO)
    data = annotate_dataset(CLINICAL_NOTES_PATH, ANNOTATED_DATA_PATH)

    print(f"\nAnnotated {len(data)} notes")
    print("\nSample annotation (N001):")
    sample = data[0]
    print(f"  Text: {sample['text']}")
    print(f"  Extracted: {sample['extracted']}")
    print(f"\nToken{'':15s} | Tag")
    print("-" * 35)
    for token, tag in zip(sample["tokens"], sample["ner_tags"]):
        if tag != "O":
            print(f"  {token:20s} | {tag}")
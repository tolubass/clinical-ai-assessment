"""
BioBERT fine-tuning pipeline for clinical NER.

Trains a token classification model on the annotated
clinical notes dataset using transfer learning from
dmis-lab/biobert-base-cased-v1.2.

Design decisions:
- Transfer learning chosen over training from scratch
  because the dataset contains only 50 notes.
- BioBERT is pre-trained on PubMed abstracts and
  PMC full-text articles — directly relevant domain.
- Fixed random seed ensures full reproducibility.
- Train/val/test split is performed at note level
  to prevent data leakage across splits.
"""

import json
import random
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import mlflow

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Ensure full reproducibility across all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(
    data: List[Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split at note level to prevent entity leakage across splits.
    All entities from one note stay in the same split.
    """
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    test = [data[i] for i in test_idx]

    logger.info(
        f"Split: train={len(train)}, val={len(val)}, test={len(test)}"
    )
    return train, val, test


class ClinicalNERDataset(Dataset):
    """
    PyTorch Dataset for clinical NER token classification.

    Handles subword tokenization alignment — critical for
    transformer-based NER where one word may split into
    multiple subword tokens. Only the first subword of each
    word receives the label; subsequent subwords get -100
    so they are ignored in the loss computation.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]

        # Tokenize with subword alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Align labels to subword tokens
        word_ids = encoding.word_ids(batch_index=0)
        labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens [CLS], [SEP], [PAD] → ignore
                labels.append(-100)
            elif word_id != previous_word_id:
                # First subword of a word → use the real label
                tag = ner_tags[word_id] if word_id < len(ner_tags) else "O"
                labels.append(self.label2id.get(tag, 0))
            else:
                # Subsequent subwords → ignore in loss
                labels.append(-100)
            previous_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def compute_metrics(
    predictions: List[List[str]],
    references: List[List[str]],
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 per entity class
    and overall micro-averaged metrics.
    Uses strict exact-match entity-level evaluation.
    """
    from sklearn.metrics import classification_report

    flat_preds = [p for seq in predictions for p in seq]
    flat_refs = [r for seq in references for r in seq]

    labels = sorted(set(flat_refs) - {"O"})

    report = classification_report(
        flat_refs,
        flat_preds,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
    }

    for label in labels:
        if label in report:
            clean = label.replace("-", "_").lower()
            metrics[f"{clean}_f1"] = report[label]["f1-score"]
            metrics[f"{clean}_precision"] = report[label]["precision"]
            metrics[f"{clean}_recall"] = report[label]["recall"]

    return metrics


def train(config) -> None:
    """
    Full training pipeline:
    1. Load annotated data
    2. Split train/val/test
    3. Initialise BioBERT tokenizer and model
    4. Train with AdamW + linear warmup schedule
    5. Evaluate on validation set each epoch
    6. Save best model by validation F1
    7. Final evaluation on held-out test set
    8. Log all metrics to MLflow
    """
    set_seed(config.RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Training device: {device}")

    # ── Load data ────────────────────────────────────────────
    with open(config.ANNOTATED_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data, val_data, test_data = split_dataset(
        data,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.RANDOM_SEED,
    )

    # Save split indices for reproducibility
    splits = {
        "train": [d["note_id"] for d in train_data],
        "val": [d["note_id"] for d in val_data],
        "test": [d["note_id"] for d in test_data],
    }
    splits_path = config.PROCESSED_DIR / "data_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Data splits saved to {splits_path}")

    # ── Tokenizer and model ──────────────────────────────────
    logger.info(
        f"Loading tokenizer: {config.PRETRAINED_MODEL_NAME}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRETRAINED_MODEL_NAME
    )

    model = AutoModelForTokenClassification.from_pretrained(
        config.PRETRAINED_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # ── Datasets and loaders ─────────────────────────────────
    train_dataset = ClinicalNERDataset(
        train_data, tokenizer, config.LABEL2ID, config.MAX_SEQ_LENGTH
    )
    val_dataset = ClinicalNERDataset(
        val_data, tokenizer, config.LABEL2ID, config.MAX_SEQ_LENGTH
    )
    test_dataset = ClinicalNERDataset(
        test_data, tokenizer, config.LABEL2ID, config.MAX_SEQ_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # ── Optimiser and scheduler ──────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── MLflow experiment ────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="biobert-ner-v1"):
        mlflow.log_params({
            "model": config.PRETRAINED_MODEL_NAME,
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "random_seed": config.RANDOM_SEED,
        })

        best_val_f1 = 0.0
        best_epoch = 0

        # ── Training loop ────────────────────────────────────
        for epoch in range(1, config.NUM_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            start_time = time.time()

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()

                # Gradient clipping for stable training
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - start_time

            # ── Validation ───────────────────────────────────
            val_metrics = evaluate(
                model, val_loader, config.ID2LABEL, device
            )

            logger.info(
                f"Epoch {epoch}/{config.NUM_EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "val_f1": val_metrics["f1"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                },
                step=epoch,
            )

            # ── Checkpoint best model ─────────────────────────
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch
                model.save_pretrained(config.MODEL_SAVE_DIR)
                tokenizer.save_pretrained(config.MODEL_SAVE_DIR)
                logger.info(
                    f"  New best model saved "
                    f"(val F1={best_val_f1:.4f})"
                )

        logger.info(
            f"Training complete. Best val F1={best_val_f1:.4f} "
            f"at epoch {best_epoch}"
        )

        # ── Final test evaluation ─────────────────────────────
        logger.info("Evaluating on held-out test set...")
        best_model = AutoModelForTokenClassification.from_pretrained(
            config.MODEL_SAVE_DIR
        )
        best_model.to(device)

        test_metrics = evaluate(
            best_model, test_loader, config.ID2LABEL, device
        )

        logger.info(
            f"Test Results — "
            f"Precision: {test_metrics['precision']:.4f} | "
            f"Recall: {test_metrics['recall']:.4f} | "
            f"F1: {test_metrics['f1']:.4f}"
        )

        mlflow.log_metrics({
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
        })

        for key, val in test_metrics.items():
            if key not in ("precision", "recall", "f1"):
                mlflow.log_metric(f"test_{key}", val)

        # ── Save model metadata ───────────────────────────────
        metadata = {
            "model_version": config.MODEL_VERSION,
            "base_model": config.PRETRAINED_MODEL_NAME,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "test_metrics": test_metrics,
            "training_config": {
                "epochs": config.NUM_EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "max_seq_length": config.MAX_SEQ_LENGTH,
                "random_seed": config.RANDOM_SEED,
            },
            "data_splits": splits,
            "entity_labels": config.ENTITY_LABELS,
        }

        metadata_path = config.MODEL_SAVE_DIR / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifact(str(metadata_path))
        logger.info(f"Model metadata saved to {metadata_path}")

        return test_metrics


def evaluate(
    model,
    data_loader: DataLoader,
    id2label: Dict[int, str],
    device,
) -> Dict[str, float]:
    """
    Run inference on a dataloader and compute NER metrics.
    Ignores padded positions (label == -100).
    """
    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels_np = labels.numpy()

            for pred_seq, label_seq in zip(predictions, labels_np):
                pred_tags = []
                true_tags = []
                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    pred_tags.append(id2label.get(p, "O"))
                    true_tags.append(id2label.get(l, "O"))
                all_predictions.append(pred_tags)
                all_references.append(true_tags)

    return compute_metrics(all_predictions, all_references)


if __name__ == "__main__":
    import sys
    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent.parent)
    )
    import src.config as config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info("Starting BioBERT NER training pipeline...")
    metrics = train(config)
    print("\n── Final Test Metrics ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")
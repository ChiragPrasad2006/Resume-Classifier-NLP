"""Fine-tune Longformer and select checkpoints using resume-level validation.

Run from the repository root with PDFs under data/data/data/<CATEGORY>/*.pdf.
The source checkpoint and prior experiments are left untouched. The existing
80/20 seed-42 split is reproduced so comparisons remain aligned with prior runs.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import ClassLabel, Dataset, Features, Value
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DATA_DIR = Path("data/data/data")
RESUME_CHECKPOINT = Path("longformer_model_further_training/checkpoint-epoch-2")
OUTPUT_DIR = Path("longformer_model_accuracy_tuning")
MAX_LENGTH = 2048
CHUNK_OVERLAP = 256
RANDOM_SEED = 42
LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 3
PDF_WORKERS = 16


def load_dataset_from_pdfs(categories, label2id):
    tasks = [
        (str(pdf), label2id[category])
        for category in categories
        for pdf in (DATA_DIR / category).glob("*.pdf")
    ]
    print(f"Found {len(tasks)} PDFs across {len(categories)} categories.")
    workers = min(PDF_WORKERS, len(tasks))
    worker_script = Path(__file__).with_name("pdf_text_worker.py")
    if not worker_script.is_file():
        raise FileNotFoundError(f"PDF worker script not found: {worker_script.resolve()}")

    print(f"Extracting PDFs with {workers} separate CPU worker processes.")
    with tempfile.TemporaryDirectory(prefix="resume_pdf_extraction_") as temp_name:
        temp_dir = Path(temp_name)
        processes = []
        for worker_index in range(workers):
            input_path = temp_dir / f"worker-{worker_index}-input.jsonl"
            output_path = temp_dir / f"worker-{worker_index}-output.jsonl"
            with input_path.open("w", encoding="utf-8") as input_file:
                for task_index in range(worker_index, len(tasks), workers):
                    json.dump({"index": task_index, "path": tasks[task_index][0]}, input_file)
                    input_file.write("\n")
            processes.append((subprocess.Popen([
                sys.executable, str(worker_script),
                "--input", str(input_path), "--output", str(output_path),
            ]), output_path))

        texts = [None] * len(tasks)
        for process, output_path in tqdm(processes, desc="Waiting for PDF workers"):
            if process.wait() != 0:
                raise RuntimeError("A PDF extraction worker exited unexpectedly.")
            with output_path.open("r", encoding="utf-8") as output_file:
                for line in output_file:
                    result = json.loads(line)
                    texts[result["index"]] = result["text"]

    rows = [(text, label) for (_, label), text in zip(tasks, texts) if text]
    if not rows:
        raise ValueError(f"No readable PDF resumes found under {DATA_DIR.resolve()}")
    features = Features({"text": Value("string"), "label": ClassLabel(names=categories)})
    dataset = Dataset.from_dict(
        {"text": [row[0] for row in rows], "label": [row[1] for row in rows]},
        features=features,
    )
    # Keep the same deterministic held-out resumes used by the prior script.
    return dataset.train_test_split(
        test_size=0.2, seed=RANDOM_SEED, stratify_by_column="label"
    )


def aggregate_document_logits(logits, chunk_counts):
    """Average a resume's chunk logits into one prediction per PDF."""
    logits = np.asarray(logits)
    predictions = []
    offset = 0
    for count in chunk_counts:
        if count < 1:
            raise ValueError("Every readable resume must produce at least one token chunk.")
        predictions.append(logits[offset:offset + count].mean(axis=0))
        offset += count
    if offset != len(logits):
        raise ValueError(f"Expected {offset} chunk predictions, got {len(logits)}.")
    return np.stack(predictions)


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR.resolve()}")
    if not RESUME_CHECKPOINT.is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_CHECKPOINT.resolve()}")

    categories = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
    model = AutoModelForSequenceClassification.from_pretrained(RESUME_CHECKPOINT)
    label2id = model.config.label2id
    id2label = {int(index): label for index, label in model.config.id2label.items()}
    expected_categories = [id2label[index] for index in range(model.config.num_labels)]
    if categories != expected_categories:
        raise ValueError(
            "Dataset category folders do not match the checkpoint label mapping.\n"
            f"Checkpoint: {expected_categories}\nDataset:    {categories}"
        )

    raw = load_dataset_from_pdfs(categories, label2id)
    tokenizer = AutoTokenizer.from_pretrained(RESUME_CHECKPOINT)

    def tokenize(batch, indices):
        encoded = tokenizer(
            batch["text"], truncation=True, max_length=MAX_LENGTH,
            stride=CHUNK_OVERLAP, return_overflowing_tokens=True,
        )
        source_rows = np.asarray(encoded.pop("overflow_to_sample_mapping"), dtype=np.int64)
        counts = np.bincount(source_rows, minlength=len(batch["label"]))
        encoded["labels"] = [batch["label"][row] for row in source_rows]
        encoded["chunk_weight"] = [1.0 / counts[row] for row in source_rows]
        encoded["document_id"] = (source_rows + int(indices[0])).tolist()
        return encoded

    tokenized = raw.map(
        tokenize, batched=True, batch_size=8, with_indices=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing resumes into overlapping chunks",
    )
    validation_chunk_counts = np.bincount(
        tokenized["test"]["document_id"], minlength=len(raw["test"])
    ).tolist()
    validation_labels = np.asarray(raw["test"]["label"])

    def compute_metrics(eval_pred):
        chunk_logits, _ = eval_pred
        document_logits = aggregate_document_logits(chunk_logits, validation_chunk_counts)
        document_predictions = np.argmax(document_logits, axis=-1)
        return {
            "doc_accuracy": float(np.mean(document_predictions == validation_labels)),
            "doc_macro_f1": float(f1_score(
                validation_labels, document_predictions, average="macro", zero_division=0
            )),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda" and not use_bf16:
        raise RuntimeError("CUDA is available but BF16 is unsupported; update precision settings.")
    print(f"Training on {device}; BF16 {'enabled' if use_bf16 else 'disabled'}.")
    print("Using unweighted loss for accuracy optimization and equal total loss per resume.")

    class DocumentBalancedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            chunk_weights = inputs.pop("chunk_weight", None)
            inputs.pop("document_id", None)
            outputs = model(**inputs)
            per_chunk_loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, reduction="none"
            )
            if chunk_weights is None:
                loss = per_chunk_loss.mean()
            else:
                chunk_weights = chunk_weights.to(per_chunk_loss.dtype)
                loss = (per_chunk_loss * chunk_weights).sum() / chunk_weights.sum().clamp_min(1e-8)
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs.pop("chunk_weight", None)
            inputs.pop("document_id", None)
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "trainer_state"),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_doc_accuracy",
        greater_is_better=True,
        logging_steps=25,
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adafactor",
        report_to="none",
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        remove_unused_columns=False,
    )
    trainer = DocumentBalancedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(
        f"Starting {NUM_TRAIN_EPOCHS} epochs from {RESUME_CHECKPOINT}. "
        "Each epoch is saved; the best resume-level validation accuracy is restored."
    )
    trainer.train()

    final_dir = OUTPUT_DIR / "best"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    metrics = trainer.evaluate()
    print("Best-checkpoint validation metrics:", metrics)

    predicted = trainer.predict(tokenized["test"])
    document_logits = aggregate_document_logits(
        predicted.predictions, validation_chunk_counts
    )
    document_predictions = np.argmax(document_logits, axis=-1)
    print("\nResume-level validation report:\n")
    print(classification_report(
        validation_labels, document_predictions,
        labels=list(range(len(categories))), target_names=categories, zero_division=0,
    ))
    print("Resume-level validation confusion matrix:")
    print(confusion_matrix(validation_labels, document_predictions))
    print(f"Best model saved to {final_dir.resolve()}")
    print(
        "Note: this split has been evaluated in prior runs, so it is a tuning/validation "
        "set; its score is not an unbiased final test estimate."
    )


if __name__ == "__main__":
    main()

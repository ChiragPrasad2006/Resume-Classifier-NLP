import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from pypdf import PdfReader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset, Features, Value, ClassLabel
import evaluate
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

DATA_DIR = Path("data/data/data")
MODEL_SAVE_PATH = "./models/longformer-base-4096_finetuned/final"
MODEL_ID = "allenai/longformer-base-4096"


def extract_text_from_pdf(pdf_path_str: str) -> str:
    try:
        reader = PdfReader(pdf_path_str)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text).strip()
    except Exception as e:
        print(f"Error reading {pdf_path_str}: {e}")
        return ""

def load_pdf_task(args):
    pdf_path_str, category, cat_id = args
    content = extract_text_from_pdf(pdf_path_str)
    if content:
        return content, cat_id, Path(pdf_path_str).stem
    return None

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found at: {DATA_DIR.resolve()}\n"
            f"Please place your categorized folders inside this path or update DATA_DIR."
        )

    # Dynamically locate categories
    categories = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    label2id = {cat: idx for idx, cat in enumerate(categories)}
    id2label = {idx: cat for cat, idx in label2id.items()}

    tasks = []
    for category in categories:
        cat_dir = DATA_DIR / category
        pdf_files = list(cat_dir.glob("*.pdf"))
        cat_id = label2id[category]
        for pdf_file in pdf_files:
            tasks.append((str(pdf_file), category, cat_id))

    print(f"Found {len(tasks)} PDFs across {len(categories)} categories.")
    print("Extracting PDFs to memory (using CPU multiprocessing for speed)...")
    
    # Multi-processing setup optimized for local CPU cores
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(load_pdf_task, tasks, chunksize=10),
            total=len(tasks),
            desc="Parsing PDFs"
        ))

    loaded_texts = []
    loaded_labels = []
    loaded_file_ids = []

    for res in results:
        if res is not None:
            text, label, f_id = res
            loaded_texts.append(text)
            loaded_labels.append(label)
            loaded_file_ids.append(f_id)

    print(f"Successfully parsed and loaded {len(loaded_texts)} PDFs into memory.\n")

    class_features = Features({
        "text": Value("string"),
        "label": ClassLabel(names=categories)
    })

    raw_dataset = Dataset.from_dict({
        "text": loaded_texts,
        "label": loaded_labels
    }, features=class_features)

    dataset_split = raw_dataset.train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column="label"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=2048,
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset_split.map(
        tokenize_fn,
        batched=True,
        batch_size=32,
        remove_columns=["text"]
    )

    train_labels = tokenized_datasets["train"]["label"]
    classes = np.unique(train_labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(train_labels)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Imbalance detected. Balanced Class Weights sent to {device}.")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(categories),
        id2label=id2label,
        label2id=label2id
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    per_device_batch_size = 2
    gradient_accumulation_steps = 8
    num_train_epochs = 6

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=1,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adafactor",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    print("Starting local training loop...")
    trainer.train()

    print("Running final evaluation cycle...")
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    predictions_output = trainer.predict(tokenized_datasets["test"])
    y_pred = np.argmax(predictions_output.predictions, axis=1)
    y_true = predictions_output.label_ids
    target_names = [id2label[i] for i in range(len(categories))]

    print("\nDetailed Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Save local outputs
    print(f"Saving trained architecture locally to {MODEL_SAVE_PATH}...")
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("Saving completed!")

if __name__ == "__main__":
    main()
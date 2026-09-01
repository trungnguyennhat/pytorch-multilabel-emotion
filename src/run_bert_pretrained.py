from pathlib import Path
from time import perf_counter
import json

import datasets as hf_datasets
import numpy as np
import sklearn
import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from run_mean_pooling_mlp import calculate_metrics, load_clean_dataset
from utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bert_pretrained"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pt"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.npz"
RUN_PATH = OUTPUT_DIR / "run.json"
RESULTS_PATH = OUTPUT_DIR / "results.txt"

MODEL_NAME = "google-bert/bert-base-cased"
MAX_LENGTH = 50
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
THRESHOLD = 0.5
SEED = 42


class TokenizedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            **{key: value[index] for key, value in self.encodings.items()},
            "labels": self.labels[index],
        }


def encode_targets(split, num_labels):
    targets = np.zeros((len(split), num_labels), dtype=np.int8)
    for row, label_ids in enumerate(split["labels"]):
        targets[row, label_ids] = 1
    return targets


def evaluate(model, dataloader, device, label_names):
    model.eval()
    losses, targets, probabilities = [], [], []
    criterion = nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            losses.append(criterion(logits, labels).item())
            targets.append(labels.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())

    targets = np.concatenate(targets).astype(np.int8)
    probabilities = np.concatenate(probabilities)
    predictions = (probabilities >= THRESHOLD).astype(np.int8)
    return {
        "loss": float(np.mean(losses)),
        "targets": targets,
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": calculate_metrics(targets, predictions, label_names),
    }


def main():
    package_start = perf_counter()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract, clean_dataset = load_clean_dataset()
    label_names = contract["schema"]["label_names"]
    num_labels = len(label_names)
    targets = {
        split: encode_targets(clean_dataset[split], num_labels)
        for split in ("train", "validation", "test")
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    datasets = {
        split: TokenizedDataset(
            list(clean_dataset[split]["text"]), targets[split], tokenizer
        )
        for split in targets
    }
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for split in datasets
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={index: label for index, label in enumerate(label_names)},
        label2id={label: index for index, label in enumerate(label_names)},
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = EPOCHS * len(dataloaders["train"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_validation_macro_f1 = -1.0
    training_start = perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for batch in dataloaders["train"]:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                loss = model(**inputs, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(loss.item())

        validation = evaluate(
            model, dataloaders["validation"], device, label_names
        )
        validation_macro_f1 = validation["metrics"]["macro"]["f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation["loss"],
                "validation_macro_f1": validation_macro_f1,
                "validation_micro_f1": validation["metrics"]["micro"]["f1"],
            }
        )
        print(
            f"Epoch {epoch}/{EPOCHS} | train loss {np.mean(train_losses):.4f} | "
            f"val macro-F1 {validation_macro_f1:.4f} | "
            f"val micro-F1 {validation['metrics']['micro']['f1']:.4f}"
        )
        if validation_macro_f1 > best_validation_macro_f1:
            best_validation_macro_f1 = validation_macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": epoch,
                    "label_names": label_names,
                    "model_name": MODEL_NAME,
                },
                CHECKPOINT_PATH,
            )

    training_seconds = perf_counter() - training_start
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = evaluate(model, dataloaders["validation"], device, label_names)
    test = evaluate(model, dataloaders["test"], device, label_names)

    np.savez_compressed(
        PREDICTIONS_PATH,
        validation_targets=validation["targets"],
        validation_probabilities=validation["probabilities"],
        validation_predictions=validation["predictions"],
        test_targets=test["targets"],
        test_probabilities=test["probabilities"],
        test_predictions=test["predictions"],
    )
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    run = {
        "system": "bert_pretrained",
        "reference": "GoEmotions original pretrained baseline",
        "seed": SEED,
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": hf_datasets.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "dataset": {
            "source": contract["dataset"]["source"],
            "config": contract["dataset"]["config"],
            "evaluation_policy": contract["evaluation_policy"]["name"],
            "clean_split_sizes": {split: len(clean_dataset[split]) for split in targets},
        },
        "label_names": label_names,
        "preprocessing": {
            "tokenizer": MODEL_NAME,
            "cased": True,
            "max_length": MAX_LENGTH,
            "padding": "max_length",
            "truncation": True,
        },
        "model_config": {
            "pretrained_model": MODEL_NAME,
            "num_labels": num_labels,
            "trainable_parameter_count": parameter_count,
        },
        "training_config": {
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "loss": "BCEWithLogitsLoss",
            "selection_metric": "validation_macro_f1",
            "threshold": THRESHOLD,
        },
        "selection": {
            "best_epoch": checkpoint["best_epoch"],
            "best_validation_macro_f1": best_validation_macro_f1,
            "history": history,
        },
        "results": {
            "validation": {"loss": validation["loss"], **validation["metrics"]},
            "test": {"loss": test["loss"], **test["metrics"]},
        },
        "efficiency": {
            "training_seconds": training_seconds,
            "full_package_seconds": perf_counter() - package_start,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        },
        "artifacts": {
            "checkpoint": "outputs/bert_pretrained/checkpoint.pt",
            "predictions": "outputs/bert_pretrained/predictions.npz",
            "results": "outputs/bert_pretrained/results.txt",
        },
    }
    RUN_PATH.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    text = (
        "Pretrained BERT GoEmotions Reference Baseline\n\n"
        f"Model: {MODEL_NAME}\nBest epoch: {checkpoint['best_epoch']}\n"
        f"Threshold: {THRESHOLD}\n"
        f"Validation macro/micro-F1: {validation['metrics']['macro']['f1']:.4f}/"
        f"{validation['metrics']['micro']['f1']:.4f}\n"
        f"Test macro/micro-F1: {test['metrics']['macro']['f1']:.4f}/"
        f"{test['metrics']['micro']['f1']:.4f}\n"
    )
    RESULTS_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

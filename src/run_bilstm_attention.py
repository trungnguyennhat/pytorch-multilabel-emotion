from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)
from run_mean_pooling_mlp import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    LOG_EVERY_EPOCHS,
    MAX_EPOCHS,
    THRESHOLD,
    TOKEN_PATTERN,
    WEIGHT_DECAY,
    build_dataloaders,
    build_vocabulary,
    choost_max_length,
    evaluate,
    format_results,
    load_clean_dataset,
    train_one_epoch,
)
from utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bilstm_attention"

CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pt"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.npz"
RUN_PATH = OUTPUT_DIR / "run.json"
RESULTS_PATH = OUTPUT_DIR / "results.txt"

SPLIT_NAMES = ("train", "validation", "test")

SEED = 42
EMBEDDING_DIM = 128
LSTM_HIDDEN_DIM = 128
ATTENTION_HIDDEN_DIM = 128
DROPOUT = 0.2

class BiLSTMAttention(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        num_labels,
        embedding_dim,
        lstm_hidden_dim,
        attention_hidden_dim,
        dropout,
        padding_id,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_id,
        )
        
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        bilstm_output_dim = lstm_hidden_dim * 2
        
        self.attention = nn.Sequential(
            nn.Linear(
                bilstm_output_dim,
                attention_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                attention_hidden_dim,
                1,
                bias=False,
            )
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(
            bilstm_output_dim,
            num_labels,
        )
    def forward(
        self,
        input_ids, 
        attention_mask,
    ):
        # bien token id thanh embedding vector
        embeddings = self.embedding(input_ids)
        
        # tinh sequence length de xem 1 cau co bao nhieu la token that
        sequence_lengths = (
            attention_mask
            .sum(dim=1)
            .cpu()
        )
        
        # cho biet LSTM moi cau xu ly den dau (bo phan pad)
        # bien embedding tensor thanh PackedSequence
        # LSTM xu ly token that, bo PAD
        packed_embeddings = pack_padded_sequence(
            embeddings, 
            lengths=sequence_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        
        # dua packed embeddings qua bilstm
        # packed_outputs chua hidden state cua tung token
        packed_outputs, _ = self.bilstm(
            packed_embeddings
        )
        
        # chuyen packed sequence thanh tensor 
        bilstm_outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=input_ids.shape[1],
        )
        
        # chay tren hidden state cua tung token
        attention_scores = (
            self.attention(bilstm_outputs)
            .squeeze(-1)
        )
        
        # thay score tai vi tri pad = - vo cung
        attention_scores = attention_scores.masked_fill(
            ~attention_mask,
            torch.finfo(
                attention_scores.dtype
            ).min,
        )
        
        # bien score thanh weight qua softmax
        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )
        
        # tinh feature cho moi cau
        sentence_features = torch.bmm(
            attention_weights.unsqueeze(1),
            bilstm_outputs,
        ).squeeze(1)
        
        # ap dropout de tranh overfit
        sentence_features = self.dropout(
            sentence_features
        )

        # chuyen so chieu 2H sang 28 label
        logits = self.classifier(
            sentence_features
        )
        
        return logits
        
def save_checkpoint(
    model,
    vocabulary,
    label_names,
    max_length,
    best_epoch,
    best_validation_macro_f1,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocabulary_size": len(vocabulary),
            "num_labels": len(label_names),
            "embedding_dim": EMBEDDING_DIM,
            "lstm_hidden_dim": LSTM_HIDDEN_DIM,
            "attention_hidden_dim": (
                ATTENTION_HIDDEN_DIM
            ),
            "dropout": DROPOUT,
            "padding_id": 0,
        },
        "vocabulary": vocabulary,
        "tokenizer_config": {
            "name": (
                "lowercase_regex_words_and_punctuation"
            ),
            "pattern": TOKEN_PATTERN.pattern,
            "minimum_frequency": 2,
            "padding_id": 0,
            "unknown_id": 1,
            "max_length": max_length,
        },
        "label_names": label_names,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(
            best_validation_macro_f1
        ),
        "threshold": THRESHOLD,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )

def train_and_select(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    vocabulary,
    label_names,
    max_length,
):
    best_validation_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    
    training_start = perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = perf_counter()

        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        
        validation_output = evaluate(
            model=model,
            dataloader=dataloaders["validation"],
            criterion=criterion,
            device=device,
            label_names=label_names,
        )
        
        validation_macro_f1 = validation_output["metrics"]["macro"]["f1"]
        
        validation_micro_f1 = validation_output["metrics"]["micro"]["f1"]
        
        epoch_seconds = perf_counter() - epoch_start
        
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(
                    validation_output["loss"]
                ),
                "validation_macro_f1": float(
                    validation_macro_f1
                ),
                "validation_micro_f1": float(
                    validation_micro_f1
                ),
                "runtime_seconds": float(
                    epoch_seconds
                ),
            }
        )
        
        should_log = (
            epoch % LOG_EVERY_EPOCHS == 0
            or epoch == MAX_EPOCHS
        )

        if should_log:
            print(
                f"Epoch {epoch:02d} | "
                f"train loss {train_loss:.4f} | "
                f"validation loss "
                f"{validation_output['loss']:.4f} | "
                f"macro-F1 "
                f"{validation_macro_f1:.4f} | "
                f"micro-F1 "
                f"{validation_micro_f1:.4f}"
            )
        
        if (
            validation_macro_f1
            > best_validation_macro_f1
        ):
            best_validation_macro_f1 = (
                validation_macro_f1
            )
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                vocabulary=vocabulary,
                label_names=label_names,
                max_length=max_length,
                best_epoch=best_epoch,
                best_validation_macro_f1=(
                    best_validation_macro_f1
                ),
            )
        else:
            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                if not should_log:
                    print(
                        f"Epoch {epoch:02d} | "
                        f"train loss {train_loss:.4f} | "
                        f"validation loss "
                        f"{validation_output['loss']:.4f} | "
                        f"macro-F1 "
                        f"{validation_macro_f1:.4f} | "
                        f"micro-F1 "
                        f"{validation_micro_f1:.4f}"
                    )

                print(
                    "Early stopping after "
                    f"{EARLY_STOPPING_PATIENCE} "
                    "epochs without improvement. "
                    f"Best epoch: {best_epoch}."
                )
                break

    training_seconds = (
        perf_counter() - training_start
    )

    return {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(
            best_validation_macro_f1
        ),
        "history": history,
        "training_seconds": float(
            training_seconds
        ),
    }

def load_selected_checkpoint(
    model,
    device,
):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    package_start = perf_counter()
    
    contract, clean_dataset = load_clean_dataset()
    
    label_names = contract["schema"]["label_names"]
    num_labels = len(label_names)
    
    train_texts = clean_dataset["train"]["text"]

    vocabulary = build_vocabulary(train_texts)
    
    max_length, length_summary = (choost_max_length(train_texts))
    
    _, dataloaders = build_dataloaders(
        clean_dataset=clean_dataset,
        vocabulary=vocabulary,
        max_length=max_length,
        num_labels=num_labels,
        device=device,
    )
    
    model = BiLSTMAttention(
        vocabulary_size=len(vocabulary),
        num_labels=num_labels,
        embedding_dim=EMBEDDING_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        attention_hidden_dim=ATTENTION_HIDDEN_DIM,
        dropout=DROPOUT,
        padding_id=0,
    ).to(device)
    
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    
    training_summary = train_and_select(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        vocabulary=vocabulary,
        label_names=label_names,
        max_length=max_length,
    )
    
    checkpoint = load_selected_checkpoint(
        model=model,
        device=device,
    )

    validation_output = evaluate(
        model=model,
        dataloader=dataloaders["validation"],
        criterion=criterion,
        device=device,
        label_names=label_names,
    )
    
    test_start = perf_counter()

    test_output = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
        label_names=label_names,
    )
    
    test_evaluation_seconds = (
        perf_counter() - test_start
    )

    np.savez_compressed(
        PREDICTIONS_PATH,
        validation_targets=(
            validation_output["targets"]
        ),
        validation_probabilities=(
            validation_output["probabilities"]
        ),
        validation_predictions=(
            validation_output["predictions"]
        ),
        test_targets=test_output["targets"],
        test_probabilities=(
            test_output["probabilities"]
        ),
        test_predictions=(
            test_output["predictions"]
        ),
    )
    
    if device.type == "cuda":
        torch.cuda.synchronize(device)

        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )
    else:
        peak_gpu_memory_mb = 0.0

    full_package_seconds = (
        perf_counter() - package_start
    )
    
    run_record = {
        "system": "bilstm_attention",
        "seed": SEED,
        "dataset": {
            "source": contract["dataset"]["source"],
            "config": contract["dataset"]["config"],
            "evaluation_policy": contract["evaluation_policy"]["name"],
            "official_fingerprints": {
                split_name: contract["official_splits"][split_name]["fingerprint"]
                for split_name in SPLIT_NAMES
            },
            "clean_split_sizes": {
                split_name: len(
                    clean_dataset[split_name]
                )
                for split_name in SPLIT_NAMES
            },
        },
        "label_names": label_names,
        "preprocessing": {
            "tokenizer": (
                "lowercase_regex_words_and_punctuation"
            ),
            "token_pattern": TOKEN_PATTERN.pattern,
            "minimum_frequency": 2,
            "vocabulary_size": len(vocabulary),
            "max_length": max_length,
            "train_token_length_summary": (
                length_summary
            ),
            "padding_id": 0,
            "unknown_id": 1,
        },
        "model_config": checkpoint[
            "model_config"
        ],
        "training_config": {
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "BCEWithLogitsLoss",
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
            "gradient_clip_norm": (
                GRADIENT_CLIP_NORM
            ),
            "selection_metric": (
                "validation_macro_f1"
            ),
            "threshold": THRESHOLD,
        },
        "selection": {
            "best_epoch": training_summary[
                "best_epoch"
            ],
            "best_validation_macro_f1": (
                training_summary[
                    "best_validation_macro_f1"
                ]
            ),
            "history": training_summary[
                "history"
            ],
        },
        "results": {
            "validation": {
                "loss": float(
                    validation_output["loss"]
                ),
                **validation_output["metrics"],
            },
            "test": {
                "loss": float(
                    test_output["loss"]
                ),
                **test_output["metrics"],
            },
        },
        "efficiency": {
            "trainable_parameter_count": (
                parameter_count
            ),
            "training_seconds": (
                training_summary["training_seconds"]
            ),
            "test_evaluation_seconds": float(
                test_evaluation_seconds
            ),
            "full_package_seconds": float(
                full_package_seconds
            ),
            "peak_gpu_memory_mb": float(
                peak_gpu_memory_mb
            ),
        },
        "artifacts": {
            "checkpoint": str(
                CHECKPOINT_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "predictions": str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "results": str(
                RESULTS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    RUN_PATH.write_text(
        json.dumps(
            run_record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    RESULTS_PATH.write_text(
        format_results(
            best_epoch=training_summary[
                "best_epoch"
            ],
            validation_output=validation_output,
            test_output=test_output,
        ).replace(
            "Mean Pooling MLP Results",
            "BiLSTM + Attention Results",
            1,
        ),
        encoding="utf-8",
    )

    print("\nSELECTED RESULTS")
    print(
        "Validation macro-F1: "
        f"{validation_output['metrics']['macro']['f1']:.4f}"
    )
    print(
        "Validation micro-F1: "
        f"{validation_output['metrics']['micro']['f1']:.4f}"
    )
    print(
        "Test macro-F1: "
        f"{test_output['metrics']['macro']['f1']:.4f}"
    )
    print(
        "Test micro-F1: "
        f"{test_output['metrics']['micro']['f1']:.4f}"
    )
    print(
        f"Saved checkpoint: {CHECKPOINT_PATH}"
    )
    print(
        f"Saved predictions: {PREDICTIONS_PATH}"
    )
    print(
        f"Saved run record: {RUN_PATH}"
    )
    print(
        f"Saved concise results: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
    
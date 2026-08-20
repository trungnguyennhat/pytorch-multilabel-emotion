from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch
from torch import nn

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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "transformer_encoder"

CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pt"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.npz"
RUN_PATH = OUTPUT_DIR / "run.json"
RESULTS_PATH = OUTPUT_DIR / "results.txt"

SPLIT_NAMES = ("train", "validation", "test")

SEED = 42

D_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
FEEDFORWARD_DIM = 256
DROPOUT = 0.2
PADDING_ID = 0

class TransformerEmotionClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        num_labels,
        max_length,
        d_model,
        num_heads,
        num_encoder_layers,
        feedforward_dim,
        dropout,
        padding_id,
    ):
        super().__init__()
        
        self.max_length = max_length
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=d_model,
            padding_idx=padding_id,
        )
        
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=d_model,
        )
        
        self.embedding_dropout = nn.Dropout(
            dropout
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        
        self.transformer_encoder = (
            nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_encoder_layers,
            )
        )
        
        self.output_norm = nn.LayerNorm(
            d_model
        )
        
        self.output_dropout = nn.Dropout(
            dropout
        )
        
        self.classifier = nn.Linear(
            d_model,
            num_labels,
        )
        
    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        batch_size, sequence_length = (input_ids.shape)
        
        # expand position id de khop voi token embedding (chiều là B, L, 128 (có thể để là L, 128))
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0)
        
        position_ids = position_ids.expand(
            batch_size,
            sequence_length,
        )
        
        token_embeddings = (
            self.token_embedding(input_ids)
        )
        
        position_embeddings = (
            self.position_embedding(position_ids)
        )  

        encoder_inputs = (
            token_embeddings
            + position_embeddings
        )

        # dropout khi embedding tranh phu thuoc vao 1 feature cu the
        encoder_inputs = (
            self.embedding_dropout(
                encoder_inputs
            )
        )
        
        padding_mask = ~attention_mask
        
        encoder_outputs = (
            self.transformer_encoder(
                encoder_inputs,
                src_key_padding_mask=padding_mask,
            )
        )
        
        encoder_outputs = self.output_norm(
            encoder_outputs
        )
        
        float_mask = (
            attention_mask
            .unsqueeze(-1)
            .to(dtype=encoder_outputs.dtype)
        )
        
        masked_outputs = (
            encoder_outputs * float_mask
        )
        
        summed_outputs = (
            masked_outputs.sum(dim=1)
        )
        real_token_counts = (
            float_mask
            .sum(dim=1)
            .clamp(min=1.0)
        )

        pooled_features = (
            summed_outputs
            / real_token_counts
        )

        pooled_features = self.output_dropout(
            pooled_features
        )

        logits = self.classifier(
            pooled_features
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
            "max_length": max_length,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_encoder_layers": (
                NUM_ENCODER_LAYERS
            ),
            "feedforward_dim": FEEDFORWARD_DIM,
            "dropout": DROPOUT,
            "padding_id": PADDING_ID,
            "activation": "gelu",
            "norm_first": True,
            "pooling": "masked_mean",
        },
        "vocabulary": vocabulary,
        "tokenizer_config": {
            "name": (
                "lowercase_regex_words_and_punctuation"
            ),
            "pattern": TOKEN_PATTERN.pattern,
            "minimum_frequency": 2,
            "padding_id": PADDING_ID,
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
            dataloader=dataloaders[
                "validation"
            ],
            criterion=criterion,
            device=device,
            label_names=label_names,
        )

        validation_macro_f1 = (
            validation_output["metrics"]["macro"]["f1"]
        )
        
        validation_micro_f1 = (
            validation_output["metrics"]["micro"]["f1"]
        )
        
        epoch_seconds = (
            perf_counter() - epoch_start
        )
        
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    train_loss
                ),
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
                        f"train loss "
                        f"{train_loss:.4f} | "
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
        torch.cuda.reset_peak_memory_stats(
            device
        )
        torch.cuda.synchronize(device)

    package_start = perf_counter()

    contract, clean_dataset = (
        load_clean_dataset()
    )
        
    label_names = contract["schema"]["label_names"]
    num_labels = len(label_names)
    
    train_texts = clean_dataset["train"]["text"]
    
    vocabulary = build_vocabulary(train_texts)
    
    max_length, length_summary = (
        choost_max_length(train_texts)
    )
    
    _, dataloaders = build_dataloaders(
        clean_dataset=clean_dataset,
        vocabulary=vocabulary,
        max_length=max_length,
        num_labels=num_labels,
        device=device,
    )
    
    model = TransformerEmotionClassifier(
        vocabulary_size=len(vocabulary),
        num_labels=num_labels,
        max_length=max_length,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=(
            NUM_ENCODER_LAYERS
        ),
        feedforward_dim=FEEDFORWARD_DIM,
        dropout=DROPOUT,
        padding_id=PADDING_ID,
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
            validation_output[
                "probabilities"
            ]
        ),
        validation_predictions=(
            validation_output[
                "predictions"
            ]
        ),
        test_targets=(
            test_output["targets"]
        ),
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
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 2)
        )
    else:
        peak_gpu_memory_mb = 0.0

    full_package_seconds = (
        perf_counter() - package_start
    )
    
    run_record = {
        "system": "transformer_encoder",
        "seed": SEED,
        "dataset": {
            "source": contract["dataset"]["source"],
            "config": contract["dataset"]["config"],
            "evaluation_policy": contract["evaluation_policy"]["name"],
            "official_fingerprints": {
                split_name: contract[
                    "official_splits"
                ][split_name]["fingerprint"]
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
            "token_pattern": (
                TOKEN_PATTERN.pattern
            ),
            "minimum_frequency": 2,
            "vocabulary_size": len(
                vocabulary
            ),
            "max_length": max_length,
            "train_token_length_summary": (
                length_summary
            ),
            "padding_id": PADDING_ID,
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
                training_summary[
                    "training_seconds"
                ]
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
            validation_output=(
                validation_output
            ),
            test_output=test_output,
        ).replace(
            "Mean Pooling MLP Results",
            "Transformer Encoder Results",
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
        f"Saved checkpoint: "
        f"{CHECKPOINT_PATH}"
    )
    print(
        f"Saved predictions: "
        f"{PREDICTIONS_PATH}"
    )
    print(
        f"Saved run record: {RUN_PATH}"
    )
    print(
        f"Saved concise results: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
    
    
        
        
        
        
        
    
    

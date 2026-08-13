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
        
        packed_outputs, _ = self.bilstm(
            packed_embeddings
        )
        
        bilstm_outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=input_ids.shape[1],
        )
        
        attention_scores = (
            self.attention(bilstm_outputs)
            .squeeze(-1)
        )
        
        attention_scores = attention_scores.masked_fill(
            ~attention_mask,
            torch.finfo(
                attention_scores.dtype
            ).min,
        )
        
        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )
        
        sentence_features = torch.bmm(
            attention_weights.unsqueeze(1),
            bilstm_outputs,
        ).squeeze(1)
        
        sentence_features = self.dropout(
            sentence_features
        )

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
        
        
    }
        
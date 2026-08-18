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
        
        
        
    
    

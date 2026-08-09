from collections import Counter
from pathlib import Path
from time import perf_counter
import json
import math
import re

import datasets as hf_datasets
import numpy as np
import sklearn
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "dataset_contract.json"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mean_pooling_mlp"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pt"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.npz"
RUN_PATH = OUTPUT_DIR / "run.json"

SPLIT_NAMES = ("train", "validation", "test")

SEED = 42
MIN_FREQUENCY = 2
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.2

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
GRADIENT_CLIP_NORM = 1.0
THRESHOLD = 0.5

def load_clean_dataset():
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset contract not found: {CONTRACT_PATH}"
        )

    contract = json.loads(
        CONTRACT_PATH.read_text(encoding="utf-8")
    )

    if contract["validation"]["status"] != "passed":
        raise ValueError(
            "Dataset contract has not passed validation"
        )

    evaluation_policy = contract["evaluation_policy"]

    if not evaluation_policy[
        "primary_evaluation_uses_clean_views"
    ]:
        raise ValueError(
            "Contract does not enable clean evaluation views"
        )

    official_dataset = load_dataset(
        contract["dataset"]["source"],
        contract["dataset"]["config"],
    )

    clean_splits = {}

    for split_name in SPLIT_NAMES:
        official_split = official_dataset[split_name]

        expected_fingerprint = contract[
            "official_splits"
        ][split_name]["fingerprint"]

        if official_split._fingerprint != expected_fingerprint:
            raise ValueError(
                f"{split_name} fingerprint changed: "
                f"{official_split._fingerprint} != "
                f"{expected_fingerprint}"
            )

        split_policy = evaluation_policy["splits"][split_name]
        excluded_ids = set(split_policy["excluded_ids"])
        official_ids = set(official_split["id"])

        missing_excluded_ids = excluded_ids - official_ids

        if missing_excluded_ids:
            raise ValueError(
                f"{split_name} is missing excluded IDs: "
                f"{sorted(missing_excluded_ids)}"
            )

        keep_indices = [
            index
            for index, row_id in enumerate(official_split["id"])
            if row_id not in excluded_ids
        ]

        clean_split = official_split.select(keep_indices)
        expected_size = split_policy["clean_num_rows"]

        if len(clean_split) != expected_size:
            raise ValueError(
                f"{split_name} clean size changed: "
                f"{len(clean_split)} != {expected_size}"
            )

        clean_splits[split_name] = clean_split

    return contract, DatasetDict(clean_splits)

# phần 1: cho phép các ký tự, chữ số, _ và cả dấu ' (ví dụ don't)
# phần 2: cho phép lấy cả các ký hiệu, dấu câu, dấu ngoặc
TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)? | [^\w\s]", re.UNICODE)

def tokenize(text):
    """
    Lowercase và tách từ/dấu câu.

    Ví dụ:
        "I'm happy!"
        -> ["i'm", "happy", "!"]
    """
    return TOKEN_PATTERN.findall(text.lower())

def build_vocabulary(train_texts):



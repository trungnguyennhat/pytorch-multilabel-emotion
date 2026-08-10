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
    token_counts = Counter()
    
    # chi lay nhung token nao xuat hien it nhat 2 lan
    for text in train_texts:
        token_counts.update(tokenize(text))
        
    retained_tokens = [
        token 
        for token, frequency in token_counts.items()
        if frequency >= MIN_FREQUENCY
    ]
    
    # xep tu cao den thap, neu bang nhau thi xep theo token
    retained_tokens.sort(
        key=lambda token: (-token_counts[token], token)
    )
    
    vocabulary = {
        PAD_TOKEN: PAD_ID, 
        UNK_TOKEN: UNK_ID,
    }
    
    # gan ID cho tung token trong retained_tokens, su dung do dai vocab -> moi token la duy nhat
    for token in retained_tokens:
        vocabulary[token] = len(vocabulary)
    
    return vocabulary 

def choost_max_length(train_texts):
    # tong hop do dai token cua toan bo text trong train
    token_lengths = np.asarray(
        [max(len(tokenize(text)), 1) for text in train_texts],
        dtype=np.int32 
    )
    
    # lay do dai token chiem 99%, de tranh outlier qua lon
    p99_token_length = int(
        np.ceil(np.percentile(token_lengths, 99))
    )
    
    # lam tron len boi so cua 8 -> do hardware cho optimize
    max_length = int(
        math.ceil(p99_token_length / 8) * 8
    )
    
    length_summary = {
        "minimum": int(token_lengths.min()),
        "median": float(np.median(token_lengths)),
        "p99": p99_token_length,
        "maximum": int(token_lengths.max()),
        "selected_max_length": max_length,
    }

    return max_length, length_summary

# encode text sẽ tokenize text trước
# sau đó sẽ duyệt toàn bộ token trong text và chuyển nó thành các token id có trong vocabulary
# những phần nào có giá trị sẽ có attention_mask = True
# sau đó thêm padding và thêm attention_mask = False cho phần padding đó
def encode_text(text, vocabulary, max_length):
    tokens = tokenize(text)
    
    if not tokens:
        tokens = [UNK_TOKEN]
    
    # neu token co trong vocab -> lay id
    # neu khong co -> lay id cua <UNK>
    token_ids = [
        vocabulary.get(token, UNK_ID)
        for token in tokens[:max_length]
    ]
    
    # nhung token_ids co gia tri -> attention mask
    attention_mask = [True] * len(token_ids)
    # them padding = max length - len token id da co
    padding_length = max_length - len(token_ids)
    
    # them vao cac vi tri sau do cac padding
    token_ids.extend([PAD_ID] * padding_length)
    
    # them cac gia tri false vao attention mask giong voi token id o tren
    attention_mask.extend([False] * padding_length)
    
    return token_ids, attention_mask

# class kế thừa class Dataset
class GoEmotionsTorchDataset(Dataset):
    """
    Mỗi item:
        input_ids:      [max_length], int64
        attention_mask: [max_length], bool
        labels:         [num_labels], float32
    """
    
    # constructor
    def __init__(
        self,
        hf_split,
        vocabulary,
        max_length,
        num_labels,
    ):
        self.hf_split = hf_split
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.num_labels = num_labels
    
    # dunder method, có thể dùng như: len(object) = object.__len__() = len(object.hf_split)
    def __len__(self):
        return len(self.hf_split)
    
    # lấy sample thứ index
    def __getitem__(self, index):
        row = self.hf_split[index]
        
        # encode text từ sample
        input_ids, attention_mask = encode_text(
            row["text"],
            self.vocabulary,
            self.max_length,
        )
        
        # tạo labels là tensor ([0., 0., 0., ...])
        labels = torch.zeros(
            self.num_labels,
            dtype=torch.float32,
        )
        
        # chỉ những label nào có giá trị tại row label thì đặt = 1.0
        # đây là multi-hot encoding
        labels[row["labels"]] = 1.0
        
        # trả về 3 dạng tensor
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long,),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool,),
            "labels": labels,
        }
        
def build_dataloaders(
    clean_dataset,
    vocabulary, 
    max_length,
    num_labels,
    device,
):
    # với mỗi split thì ta tạo 1 dataset riêng chứa các trường đã return ở trên
    # nó chưa tạo luôn mà chỉ tạo 3 object ví dụ 1 cái là torch_datasets["train"]
    # khi nào cần lấy 1 sample thì có thể getitem
    # torch_datasets["train"][10] = torch_datasets["train"].__getitem__(10) = lấy sample thứ 10
    torch_datasets = {
        split_name: GoEmotionsTorchDataset(
            hf_split=clean_dataset[split_name],
            vocabulary=vocabulary,
            max_length=max_length,
            num_labels=num_labels,
        )
        for split_name in SPLIT_NAMES
    }
    
    # dataloader sẽ gọi getitem nhiều lần và gom các sample đã gọi thành 1 batch để iterate
    # for batch in dataloaders["train"]
    dataloaders = {
        "train": DataLoader(
            torch_datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        "validation": DataLoader(
            torch_datasets["validation"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            torch_datasets["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
    }
    return torch_datasets, dataloaders
        
def inspect_batch(
    train_dataset,
    clean_train_split,
    vocabulary,
    num_labels,
):
    ""
        
    
    



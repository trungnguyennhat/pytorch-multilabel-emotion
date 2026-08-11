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
        
class MeanPoolingMLP(nn.Module):
    """
    input_ids:      [batch_size, sequence_length]
    attention_mask: [batch_size, sequence_length]

    Output:
        logits: [batch_size, num_labels]
    """
    
    def __init__(
        self, 
        vocabulary_size,
        num_labels,
        embedding_dim,
        hidden_dim,
        dropout,
    ):
        # goi constructor cua class cha, khac voi java la tu dong goi
        super().__init__()
        
        # tao 1 object thuoc class nn.Embedding
        self.embedding = nn.Embedding(
            # moi token trong vocab deu can 1 vector embedding 
            num_embeddings=vocabulary_size,
            # so chieu cua vecto embedding
            embedding_dim=embedding_dim,
            padding_idx=PAD_ID
        )
        # sau khi embedding cho tung token o tren, se lay mean pooling cac token de tao ra 1 vecto dai dien cho ca cau [embedding_dim]
        # gia su: "a" -> [128], "b" -> [128],...-> mean pooling -> [128] cho 1 cau
        
        # sau do moi tao classifier de du lieu chay lan luot qua cac layer theo dung thu tu
        self.classifier = nn.Sequential(
            # sau khi da co vecto cua cau ta cho qua tang linear de bieu dien thanh hidden_dim
            nn.Linear(embedding_dim, hidden_dim),
            # ReLU de them non linear
            nn.ReLU(),
            # khi train se ngau nhien dropout de giam overfit
            nn.Dropout(dropout),
            # bien hidden vector thanh vecto 28 chieu = so emotion
            nn.Linear(hidden_dim, num_labels),
        )
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        # doi dang attention tu true/false thanh float 1.0/0.0
        float_mask = attention_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
        
        # -> embedding cua token * 1, embedding cua pad * 0
        masked_embeddings = embeddings * float_mask
        
        # cong embedding cua tat ca token trong moi cau 
        summed_embeddings = masked_embeddings.sum(dim=1)
        
        # chia cho min = 1, tranh de 0
        real_token_counts = float_mask.sum(dim=1).clamp(min=1.0)
        
        # day chinh la mean pooling, ket qua cua toan bo embedding trong cau chia cho so token
        pooled_features = summed_embeddings / real_token_counts 
        
        # sau khi co embedding cua ca cau ta cho qua classifier, thu dc logits 
        logits = self.classifier(pooled_features)
        
        return logits
    
def calculate_metrics(
    y_true,
    y_pred,
    label_names,
):
    metrics = {}
    
    for average in ("macro", "micro"):
        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true, 
                y_pred, 
                average=average,
                zero_division=0,
            )
        )
        
        metrics[average] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    (
        per_label_precision,
        per_label_recall,
        per_label_f1,
        per_label_support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    
    metrics["per_label"] = [
        {
            "label_id": label_id,
            "label": label_name,
            "support": int(per_label_support[label_id]),
            "precision": float(
                per_label_precision[label_id]
            ),
            "recall": float(
                per_label_recall[label_id]
            ),
            "f1": float(per_label_f1[label_id]),
        }
        for label_id, label_name in enumerate(label_names)
    ]

    return metrics

def train_one_epoch(
    model, 
    dataloader, 
    criterion,
    optimizer,
    device,
):
    # chuyen model sang training mode, de su dung dropout
    model.train()
    
    # cong don loss va tinh so sample da xu ly
    total_loss = 0.0
    total_samples = 0
    
    # wrapper hien progress bar
    progress = tqdm(
        dataloader, 
        desc="Train",
        leave=False,
    )
    
    # khi for batch, dataloader se chon cac index, chon dataset[index], lay cac truong cua tung sample, gom cac sample thanh tung batch va tra ve batch
    for batch in progress:
        # batch["input_ids"] la tensor dang o cpu, to device se chuyen sang cuda
        input_ids = batch["input_ids"].to(
            device,
            non_blocking=True,
        )
        
        attention_mask = batch["attention_mask"].to(
            device, 
            non_blocking=True,
        )
        
        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )
        
        optimizer.zero_grad(set_to_none=True)
        
        # model la nn.Module, se goi nn.Module.__call__(), khi do se goi forward() = model(...)
        logits = model(input_ids, attention_mask)
        
        if logits.shape != labels.shape:
            raise ValueError(
                f"logits shape {tuple(logits.shape)} "
                f"does not match labels shape "
                f"{tuple(labels.shape)}"
            )
        
        # tinh loss, criterion la ham loss (BCE)
        # criterion la 1 object loss func, cung la 1 nn.Module
        # bien logit thanh prob bang sigmoid roi so voi label bang BCE
        # sau khi tinh loss thi pytorch se lay mean cuoi cung -> scalar tensor
        loss = criterion(logits, labels)
        
        # tinh gradient cho tung parameter, chua cap nhat weight
        # pytorch o day se xay 1 computation graph, moi parameter se duoc cap nhat dua tren chain rule
        # tat ca cac parameter co requires_grad=True thi se duoc cap nhat gradient - chi update weight, chi tinh grad
        loss.backward()
        
        # gioi han do lon cua gradient
        clip_grad_norm_(
            # lay tat ca parameter trainable cua model, moi parameter co gradient rieng
            model.parameters(),
            
            # pytorch coi tat ca gradient nhu 1 vecto roi tinh L2, neu qua lon so voi max norm se scale xuong 1 ti le de gradnorm/maxnorm
            max_norm=GRADIENT_CLIP_NORM,
        )
        
        # cap nhat cac weight cua tung parameter
        optimizer.step()
        
        batch_size = labels.shape[0]
        # loss item chuyen tensor scalar thanh python float, tinh tong so loss cua cac sample, loss.item o day la trung binh cua 1 batch
        total_loss += loss.item() * batch_size 
        total_samples += batch_size
        
        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / total_samples

def evaluate(
    model,
    dataloader,
    criterion,
    device, 
    label_names,
):
    # thay đổi mode của model
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    probability_batches = []
    target_batches = []
    
    # pytorch không xây computation graph để tính gradient
    # khi train thì sẽ cần graph trên để lưu thông tin các layer nhưng với evaluate thì sẽ không cần
    with torch.inference_mode():
        for batch in tqdm(
            dataloader, 
            desc="Evaluate",
            leave=False,
        ):
            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )
            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )
            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )
            
            # input_ids -> embedding -> mean pooling -> classifier -> logits
            logits = model(input_ids, attention_mask)
            if logits.shape != labels.shape:
                raise ValueError(
                    f"logits shape {tuple(logits.shape)} "
                    f"does not match labels shape "
                    f"{tuple(labels.shape)}"
                )
            
            # ở đây logits sẽ không bị thay đổi mặc dù trong criterion sẽ phải tính sigmoid từ logits để ra loss
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)
            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # lưu probability: tách khỏi computation graph, chuyển sang cpu để làm việc với numpy, thêm prediction của batch hiện tại vào list
            probability_batches.append(
                probabilities.detach().cpu().numpy()
            )
            # lưu label thật vào list
            target_batches.append(
                labels.detach().cpu().numpy()
            )

    y_probability = np.concatenate(
        probability_batches,
        axis=0,
    ).astype(np.float32)
    
    
    
        
    
    



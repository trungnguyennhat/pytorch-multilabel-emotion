from itertools import combinations
from pathlib import Path
from time import perf_counter
import json

import datasets as hf_datasets
import numpy as np
import sklearn
from datasets import DatasetDict, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "dataset_contract.json"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tfidf_logreg"
RUN_PATH = OUTPUT_DIR / "run.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.npz"

SPLIT_NAMES = ("train", "validation", "test")
THRESHOLD = 0.5

CANDIDATES = (
    {
        "name": "unigram", # lay unigram = 1 tu
        "ngram_range": (1, 1),
        "min_df": 2, # chi lay nhung tu nao xuat hien it nhat trong 2 cau
    },
    {
        "name": "unigram_bigram", # lay bigram = 2 tu
        "ngram_range": (1, 2),
        "min_df": 2,
    },
)

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
    
    # check fingerprint
    for split_name in SPLIT_NAMES:
        expected_fingerprint = contract["official_splits"][split_name]["fingerprint"]
        actual_fingerprint = official_dataset[split_name]._fingerprint
        
        if actual_fingerprint != expected_fingerprint:
            raise ValueError(
                f"{split_name} fingerprint changed: "
                f"{actual_fingerprint} != "
                f"{expected_fingerprint}"
            )
            
        split_policy = evaluation_policy["splits"][split_name]
        excluded_ids = set(split_policy["excluded_ids"])
        official_ids = set(official_dataset[split_name]["id"])

        missing_excluded_ids = (excluded_ids - official_ids)

        if missing_excluded_ids:
            raise ValueError(
                f"{split_name} is missing excluded IDs: "
                f"{sorted(missing_excluded_ids)}"
            )
            
        keep_indices = [
            index
            for index, row_id in enumerate(
                official_dataset[split_name]["id"]
            )
            if row_id not in excluded_ids
        ]
        
        clean_split = official_dataset[split_name].select(keep_indices)
        expected_rows = split_policy["clean_num_rows"]
        
        if len(clean_split) != expected_rows:
            raise ValueError(
                f"{split_name} clean size changed: "
                f"{len(clean_split)} != {expected_rows}"
            )
        clean_splits[split_name] = clean_split
    
    clean_dataset = DatasetDict(clean_splits)
    clean_text_sets = {
        split_name: set(clean_dataset[split_name]["text"])
        for split_name in SPLIT_NAMES
    }
    for left_split, right_split in combinations(
        SPLIT_NAMES,
        2,
    ):
        shared_text_count = len(
            clean_text_sets[left_split]
            & clean_text_sets[right_split]
        )

        if shared_text_count:
            raise ValueError(
                f"{left_split} and {right_split} share "
                f"{shared_text_count} exact texts"
            )

    return contract, clean_dataset


def encode_targets(clean_dataset, label_names):
    num_labels = len(label_names)
    # tao object cua class va khoi tao voi list la num_labels(0 -> 27)
    encoder = MultiLabelBinarizer(classes=list(range(num_labels)))
    targets = {
        # fit: tao ra 1 array co dinh duoc lay tu train, moi vi tri chi 1 label_id, nhung do da khoi tao classes o tren nen vi tri 0->27 se duoc map la id 0->27 
        # transform: sau khi co vi tri cung voi id tai vi tri do, ta se chuyen cac list chua label_id thanh cac vector multi-hot. So chieu = array cua fit (o day la 28) va list label_id dang chua label nao thi cot mang id do se = 1, con lai = 0 vd: [1 0 0 1 ....] 
        # tao thanh mang 2 chieu
        "train": encoder.fit_transform(clean_dataset["train"]["labels"]).astype(np.int8),
        "validation": encoder.transform(clean_dataset["validation"]["labels"]).astype(np.int8),
        "test": encoder.transform(clean_dataset["test"]["labels"]).astype(np.int8),
    }
    
    expected_classes = list(range(num_labels))
    actual_classes = list(encoder.classes_)
    if actual_classes != expected_classes:
        raise ValueError(
            "Target columns differ from contract label order"
        )
    for split_name in SPLIT_NAMES:
        expected_shape = (
            len(clean_dataset[split_name]),
            num_labels,
        )

        if targets[split_name].shape != expected_shape:
            raise ValueError(
                f"{split_name} target shape is "
                f"{targets[split_name].shape}; "
                f"expected {expected_shape}"
            )

    return targets
    
def calculate_metrics(y_true, y_pred, label_names):
    metrics = {}
    
    # cac diem nay tinh voi tung label, vi du voi label 0 se tinh tu tat ca cac vecto, TN, FP,...
    # macro se tinh cho tat ca cac label truoc roi lay trung binh
    # micro se tinh tong TN, FP,... cua tat ca cac label roi moi ap dung cong thuc
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
    
    # de average none la tinh cac diem cho tung label va khong gop lai
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
            "support": int(
                per_label_support[label_id]
            ),
            "precision": float(
                per_label_precision[label_id]
            ),
            "recall": float(
                per_label_recall[label_id]
            ),
            "f1": float(
                per_label_f1[label_id]
            ),
        }
        for label_id, label_name in enumerate(
            label_names
        )
    ]
    
    return metrics
        
def fit_candidate(
    candidate,
    train_text,
    validation_texts,
    y_train,
    y_validation,
    label_names,
):
    print(f"Training candidate: {candidate['name']}")

from collections import Counter
from itertools import combinations
from numbers import Integral
from pathlib import Path
import json

import torch
from datasets import DatasetDict, load_dataset

DATASET_NAME = "google-research-datasets/go_emotions"
DATASET_CONFIG = "simplified"

EXPECTED_SPLITS = ("train", "validation", "test")
DOCUMENTED_SPLIT_SIZES = {
    "train": 43_410,
    "validation": 5_426,
    "test": 5_427,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

def load_and_validate_dataset():
    # dataset la 1 dict dang "train": dataset{}, "valid": dataset{}
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected dataset to be a DatasetDict, but got {type(dataset)}")
    
    # lay set tu cac split trong dataset
    actual_splits = set(dataset.keys())
    expected_splits = set(EXPECTED_SPLITS)

    if actual_splits != expected_splits:
        raise ValueError(
            f"Unexpected splits: {sorted(actual_splits)}; "
            f"expected {sorted(expected_splits)}"
        )
    
    label_names = None
    
    # duyet moi split
    for split_name in actual_splits:

        split = dataset[split_name]
        required_columns = {"text", "labels"}
        missing_columns = required_columns - set(split.column_names)
        
        if missing_columns:
            raise ValueError(
                f"{split_name} is missing columns: {sorted(missing_columns)}"
            )
        
        # split o tren tra ve 1 doi tuong kieu Dataset giong df, co cac column la text va labels, no la du lieu cua ca bang, moi row la 1 dict
        # nhung khi split.features thi no se tra ve thong tin va kieu du lieu cua column nhu o duoi 
        # labels_feature la 1 doi tuong kieu Sequence, sau do lay ra mo ta cot, nhu la kieu du lieu cua 1 column, phai dung .feature de truy cap thong tin ve cac nhan ben trong no
        # cot co 1 kieu du lieu la Sequence
        # nhan co 1 kieu du lieu la ClassLabel
        # de cot truy cap duoc cac nhan ben trong no, ta dung .feature de lay ra 1 doi tuong kieu ClassLabel, sau do dung .names de lay ra ten cac nhan
        """
        Sequence(
            feature=ClassLabel(
                names=["politics", "sports", "technology"]
            )
        )
        """
        # sequence va Classlabel o day deu la class, () tao 1 object tu cac class
        labels_feature = split.features["labels"] 
        
        # getattr(.."feature"...) tuong duong voi viec check .feature
        # neu co thi lay ra, neu khong co tra ve None
        # .feature tra ve 1 doi tuong kieu ClassLabel va ten cac nhan cua column "labels"
        class_label = getattr(labels_feature, "feature", None)
        split_label_names = getattr(class_label, "names", None)
        
        if split_label_names is None:
            raise TypeError(
                f"{split_name}.features['labels'] is not a sequence "
                "containing ClassLabel"
            )
        
        # dam bao chuyen thanh list
        split_label_names = list(split_label_names)
        
        if label_names is None:
            label_names = split_label_names
        elif split_label_names != label_names:
            raise ValueError(
                f"Label order in {split_name} differs from train"
            )
        
    if len(label_names) != 28:
        raise ValueError(
            f"Expected 28 labels, but found {len(label_names)}"
        )
    
    return dataset, label_names
        
def audit_dataset(dataset, num_labels):
    split_statistics = {}
    text_sets = {} # khac voi khoi tao set, a = set()
    blocking_issues = []
    
    blocking_fields = (
        "missing_text_count",
        "empty_text_count",
        "invalid_text_type_count",
        "missing_labels_count",
        "empty_labels_count",
        "invalid_labels_container_count",
        "invalid_label_id_count",
        "repeated_label_id_row_count",
    )
    
    for split_name in EXPECTED_SPLITS:
        split = dataset[split_name]
        
        # tao object cua class Counter, text_counter co kieu Counter (con cua dict -> hd nhu dict)
        text_counter = Counter()
        
        counts = {
            "missing_text_count": 0,
            "empty_text_count": 0,
            "invalid_text_type_count": 0,
            "missing_labels_count": 0,
            "empty_labels_count": 0,
            "invalid_labels_container_count": 0,
            "invalid_label_id_count": 0,
            "repeated_label_id_row_count": 0,
        }
        
        # tung row trong split la 1 dict, {"text": "text", "labels": [0,...]}
        for row in split:
            text = row["text"]
            labels = row["labels"]
            
            # check neu text nao bi loi khong, neu khong thi dem so lan xuat hien cua text do
            if text is None:
                counts["missing_text_count"] += 1
            elif not isinstance(text, str):
                counts["invalid_text_type_count"] += 1
            elif not text.strip():
                counts["empty_text_count"] += 1
            else:
                text_counter[text] += 1
            
            # tiep tuc check cho labels
            if labels is None:
                counts["missing_labels_count"] += 1
                continue

            if not isinstance(labels, list):
                counts["invalid_labels_container_count"] += 1
                continue
            
            if not labels:
                counts["empty_labels_count"] += 1
                continue
            
            # lay ra list valid id tu so luong label id cua ca dataset
            valid_ids = [
                int(label_id)
                for label_id in labels
                if (
                    isinstance(label_id, Integral)
                    and  not isinstance(label_id, bool)
                    and 0 <= int(label_id) < num_labels
                )
            ]
            
            counts["invalid_label_id_count"] += len(labels) - len(valid_ids)

            if len(valid_ids) != len(set(valid_ids)):
                counts["repeated_label_id_row_count"] += 1

        actual_size = len(split)
        documented_size = DOCUMENTED_SPLIT_SIZES[split_name]
        
        if actual_size != documented_size:
            blocking_issues.append(
                f"{split_name}: loaded {actual_size} rows, "
                f"documented value is {documented_size}"
            )
        
        for field in blocking_fields:
            if counts[field] > 0:
                blocking_issues.append(
                    f"{split_name}: {field} = {counts[field]}"
                )
        
        split_statistics[split_name] = {
            "num_rows": actual_size,
            "documented_num_rows": documented_size,
            "columns": list(split.column_names),
            "fingerprint": split._fingerprint,
            **counts,
            "duplicate_text_value_count": sum(
                frequency > 1 for frequency in text_counter.values()
            ),
            "duplicate_text_row_count": sum(
                frequency - 1
                for frequency in text_counter.values()
                if frequency > 1
            ),
        }
        
        text_sets[split_name] = set(text_counter)
        
    cross_split_duplicates = {}
    
    # ghep tung split tung doi
    for left_split, right_split in combinations(EXPECTED_SPLITS, 2):
        # dem so cau trung o trong 2 split
        overlap_count = len(
            text_sets[left_split] & text_sets[right_split]
        )
        # tao ten va luu ket qua count so cau lap
        pair_name = f"{left_split}_vs_{right_split}"
        cross_split_duplicates[pair_name] = overlap_count
        # neu lap thi se log vao blocking_issues
        if overlap_count > 0:
            blocking_issues.append(
                f"{pair_name}: {overlap_count} exact duplicate texts"
            )
    
    return split_statistics, cross_split_duplicates, blocking_issues

    
        

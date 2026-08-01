from collections import Counter
from itertools import combinations
from numbers import Integral
from pathlib import Path
import json

from datasets import DatasetDict, load_dataset


DATASET_NAME = "google-research-datasets/go_emotions"
DATASET_CONFIG = "simplified"
EXPECTED_SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = {"id", "text", "labels"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "data" / "artifacts" / "dataset_contract.json"

def load_contract():
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict, received {type(dataset).__name__}"
        )

    actual_splits = set(dataset.keys())
    expected_splits = set(EXPECTED_SPLITS)

    if actual_splits != expected_splits:
        raise ValueError(
            f"Unexpected splits: {sorted(actual_splits)}; "
            f"expected: {sorted(expected_splits)}"
        )

    label_names = None

    for split_name in EXPECTED_SPLITS:
        split = dataset[split_name]

        missing_columns = REQUIRED_COLUMNS - set(split.column_names)
        if missing_columns:
            raise ValueError(
                f"{split_name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        labels_feature = split.features["labels"]
        class_label = getattr(labels_feature, "feature", None)
        split_label_names = getattr(class_label, "names", None)

        if split_label_names is None:
            raise TypeError(
                f"{split_name}.features['labels'] does not contain "
                "ClassLabel metadata"
            )

        split_label_names = list(split_label_names)

        if label_names is None:
            label_names = split_label_names
        elif split_label_names != label_names:
            raise ValueError(
                f"Label order in {split_name} differs from train"
            )

    if not label_names:
        raise ValueError("Dataset contains no label names")

    return dataset, label_names

def audit_rows_and_splits(dataset, label_names):
    num_labels = len(label_names)
    split_summaries = {}
    split_text_sets = {}
    blocking_issues = []

    blocking_count_names = (
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

        for row in split:
            text = row["text"]
            labels = row["labels"]

            if text is None:
                counts["missing_text_count"] += 1
            elif not isinstance(text, str):
                counts["invalid_text_type_count"] += 1
            elif not text.strip():
                counts["empty_text_count"] += 1
            else:
                text_counter[text] += 1

            if labels is None:
                counts["missing_labels_count"] += 1
                continue

            if not isinstance(labels, list):
                counts["invalid_labels_container_count"] += 1
                continue

            if not labels:
                counts["empty_labels_count"] += 1
                continue

            valid_label_ids = []

            for label_id in labels:
                is_integer = (
                    isinstance(label_id, Integral)
                    and not isinstance(label_id, bool)
                )

                if is_integer and 0 <= int(label_id) < num_labels:
                    valid_label_ids.append(int(label_id))
                else:
                    counts["invalid_label_id_count"] += 1

            if len(valid_label_ids) != len(set(valid_label_ids)):
                counts["repeated_label_id_row_count"] += 1

        for count_name in blocking_count_names:
            if counts[count_name] > 0:
                blocking_issues.append(
                    f"{split_name}: {count_name} = {counts[count_name]}"
                )

        split_summaries[split_name] = {
            "num_rows": len(split),
            "fingerprint": split._fingerprint,
            "columns": list(split.column_names),
            "data_quality": counts,
            "within_split_duplicates": {
                "duplicated_text_value_count": sum(
                    frequency > 1
                    for frequency in text_counter.values()
                ),
                "extra_duplicate_row_count": sum(
                    frequency - 1
                    for frequency in text_counter.values()
                    if frequency > 1
                ),
            },
        }

        split_text_sets[split_name] = set(text_counter.keys())

    cross_split_duplicates = {}

    for left_split, right_split in combinations(EXPECTED_SPLITS, 2):
        shared_text_count = len(
            split_text_sets[left_split]
            & split_text_sets[right_split]
        )

        pair_name = f"{left_split}_vs_{right_split}"
        cross_split_duplicates[pair_name] = {
            "shared_unique_text_count": shared_text_count
        }

    return split_summaries, cross_split_duplicates, blocking_issues

def build_clean_evaluation_views(dataset):
    all_ids = set()
    
    # lay toan bo cac id cua tung split va check xem cac split sau co bi trung ko
    for split_name in EXPECTED_SPLITS:
        split_ids = list(dataset[split_name]["id"])
        split_id_set = set(split_ids)

        if len(split_ids) != len(split_id_set):
            raise ValueError(
                f"{split_name} contains duplicate row IDs"
            )

        
        shared_ids = all_ids & split_id_set
        if shared_ids:
            raise ValueError(
                f"{split_name} shares {len(shared_ids)} row IDs "
                "with an earlier split"
            )

        all_ids.update(split_id_set)

    # lay toan bo text trong tap train
    train_texts = set(dataset["train"]["text"])

    # loai nhung row nao da xuat hien o trong train, chia thanh 2 list loai va giu
    validation_keep_indices = []
    validation_excluded_ids = []

    for index, (row_id, text) in enumerate(
        zip(
            dataset["validation"]["id"],
            dataset["validation"]["text"],
        )
    ):
        if text in train_texts:
            validation_excluded_ids.append(row_id)
        else:
            validation_keep_indices.append(index)

    validation_clean = dataset["validation"].select(
        validation_keep_indices
    )
    validation_clean_texts = set(validation_clean["text"])

    # tiep tuc xu ly voi tap test
    protected_test_texts = train_texts | validation_clean_texts
    test_keep_indices = []
    test_excluded_ids = []

    for index, (row_id, text) in enumerate(
        zip(
            dataset["test"]["id"],
            dataset["test"]["text"],
        )
    ):
        if text in protected_test_texts:
            test_excluded_ids.append(row_id)
        else:
            test_keep_indices.append(index)

    test_clean = dataset["test"].select(test_keep_indices)

    # clean dataset chi luu nhung thu da loc o tren
    clean_dataset = DatasetDict(
        {
            "train": dataset["train"],
            "validation": validation_clean,
            "test": test_clean,
        }
    )

    clean_text_sets = {
        split_name: set(clean_dataset[split_name]["text"])
        for split_name in EXPECTED_SPLITS
    }

    for left_split, right_split in combinations(
        EXPECTED_SPLITS,
        2,
    ):
        shared_texts = (
            clean_text_sets[left_split]
            & clean_text_sets[right_split]
        )

        if shared_texts:
            raise AssertionError(
                f"Clean views still contain {len(shared_texts)} "
                f"shared texts between {left_split} and {right_split}"
            )

    evaluation_policy = {
        "name": "exact_text_disjoint_evaluation",
        "official_splits_preserved": True,
        "primary_evaluation_uses_clean_views": True,
        "comparison": {
            "field": "text",
            "normalization": "none",
            "case_sensitive": True,
        },
        "splits": {
            "train": {
                "source_split": "train",
                "official_num_rows": len(dataset["train"]),
                "clean_num_rows": len(clean_dataset["train"]),
                "excluded_ids": [],
            },
            "validation": {
                "source_split": "validation",
                "excluded_if_text_appears_in": ["train"],
                "official_num_rows": len(dataset["validation"]),
                "clean_num_rows": len(validation_clean),
                "excluded_ids": validation_excluded_ids,
            },
            "test": {
                "source_split": "test",
                "excluded_if_text_appears_in": [
                    "train",
                    "validation_clean",
                ],
                "official_num_rows": len(dataset["test"]),
                "clean_num_rows": len(test_clean),
                "excluded_ids": test_excluded_ids,
            },
        },
    }

    return clean_dataset, evaluation_policy
    

def write_artifact(
    label_names,
    split_summaries,
    cross_split_duplicates,
    evaluation_policy,
    blocking_issues,
):
    validation_status = (
        "passed" if not blocking_issues else "failed"
    )

    artifact = {
        "dataset": {
            "source": DATASET_NAME,
            "config": DATASET_CONFIG,
        },
        "schema": {
            "required_columns": sorted(REQUIRED_COLUMNS),
            "num_labels": len(label_names),
            "label_names": label_names,
        },
        "official_splits": split_summaries,
        "cross_split_duplicates": cross_split_duplicates,
        "evaluation_policy": evaluation_policy,
        "validation": {
            "status": validation_status,
            "blocking_issues": blocking_issues,
        },
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def main():
    dataset, label_names = load_contract()

    (
        split_summaries,
        cross_split_duplicates,
        blocking_issues,
    ) = audit_rows_and_splits(dataset, label_names)

    if blocking_issues:
        write_artifact(
            label_names=label_names,
            split_summaries=split_summaries,
            cross_split_duplicates=cross_split_duplicates,
            evaluation_policy=None,
            blocking_issues=blocking_issues,
        )

        print(f"Artifact written to: {ARTIFACT_PATH}")
        print("AUDIT FAILED")

        for issue in blocking_issues:
            print(f"- {issue}")

        raise SystemExit(1)

    clean_dataset, evaluation_policy = (
        build_clean_evaluation_views(dataset)
    )

    write_artifact(
        label_names=label_names,
        split_summaries=split_summaries,
        cross_split_duplicates=cross_split_duplicates,
        evaluation_policy=evaluation_policy,
        blocking_issues=blocking_issues,
    )

    print(f"Artifact written to: {ARTIFACT_PATH}")

    for split_name in EXPECTED_SPLITS:
        official_size = len(dataset[split_name])
        clean_size = len(clean_dataset[split_name])
        removed_count = official_size - clean_size

        print(
            f"{split_name}: official={official_size}, "
            f"clean={clean_size}, removed={removed_count}"
        )

    print("AUDIT PASSED")


if __name__ == "__main__":
    main()

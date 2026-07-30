import csv
import json
from collections import Counter
from itertools import combinations
from numbers import Integral
from pathlib import Path

import datasets
import torch
from datasets import load_dataset


DATASET_ID = "google-research-datasets/go_emotions"
CONFIG_NAME = "simplified"
EXPECTED_SPLITS = ("train", "validation", "test")
ARTIFACT_DIR = Path("data/artifacts")


def audit_split(split_name, split, num_labels):
    stats = {
        "split": split_name,
        "num_rows": len(split),
        "missing_text_rows": 0,
        "empty_text_rows": 0,
        "non_string_text_rows": 0,
        "missing_labels_rows": 0,
        "empty_label_rows": 0,
        "invalid_label_container_rows": 0,
        "invalid_label_rows": 0,
        "duplicate_label_id_rows": 0,
    }
    valid_texts = []

    for row in split:
        text = row["text"]
        labels = row["labels"]

        if text is None:
            stats["missing_text_rows"] += 1
        elif not isinstance(text, str):
            stats["non_string_text_rows"] += 1
        elif not text.strip():
            stats["empty_text_rows"] += 1
        else:
            valid_texts.append(text)

        if labels is None:
            stats["missing_labels_rows"] += 1
            continue

        if not isinstance(labels, (list, tuple)):
            stats["invalid_label_container_rows"] += 1
            continue

        if len(labels) == 0:
            stats["empty_label_rows"] += 1
            continue

        valid_label_ids = []
        row_has_invalid_label = False

        for label_id in labels:
            is_valid = (
                isinstance(label_id, Integral)
                and not isinstance(label_id, bool)
                and 0 <= int(label_id) < num_labels
            )

            if is_valid:
                valid_label_ids.append(int(label_id))
            else:
                row_has_invalid_label = True

        if row_has_invalid_label:
            stats["invalid_label_rows"] += 1

        if len(valid_label_ids) != len(set(valid_label_ids)):
            stats["duplicate_label_id_rows"] += 1

    text_counts = Counter(valid_texts)
    stats["unique_nonempty_texts"] = len(text_counts)
    stats["duplicate_text_rows_beyond_first"] = sum(
        count - 1 for count in text_counts.values() if count > 1
    )
    stats["fingerprint"] = getattr(split, "_fingerprint", None)

    return stats, set(valid_texts)


def main():
    dataset = load_dataset(DATASET_ID, CONFIG_NAME)

    actual_splits = tuple(dataset.keys())
    if set(actual_splits) != set(EXPECTED_SPLITS):
        raise RuntimeError(
            f"Expected splits {EXPECTED_SPLITS}, but received {actual_splits}"
        )

    reference_features = dataset["train"].features
    required_columns = {"text", "labels"}

    for split_name in EXPECTED_SPLITS:
        split = dataset[split_name]

        missing_columns = required_columns - set(split.column_names)
        if missing_columns:
            raise RuntimeError(
                f"{split_name} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if split.features != reference_features:
            raise RuntimeError(
                f"Feature schema in {split_name} differs from train."
            )

    labels_container = reference_features["labels"]
    label_feature = getattr(labels_container, "feature", None)
    label_names = getattr(label_feature, "names", None)

    if not label_names:
        raise RuntimeError(
            "Could not obtain label names from the labels ClassLabel feature."
        )

    label_names = list(label_names)

    if len(label_names) != len(set(label_names)):
        raise RuntimeError("Label names are not unique.")

    num_labels = len(label_names)
    label_to_id = {
        label_name: label_id
        for label_id, label_name in enumerate(label_names)
    }
    id_to_label = {
        str(label_id): label_name
        for label_id, label_name in enumerate(label_names)
    }

    split_summaries = []
    text_sets = {}

    for split_name in EXPECTED_SPLITS:
        stats, texts = audit_split(
            split_name,
            dataset[split_name],
            num_labels,
        )
        split_summaries.append(stats)
        text_sets[split_name] = texts

    cross_split_overlaps = {}

    for left_split, right_split in combinations(EXPECTED_SPLITS, 2):
        overlap = text_sets[left_split] & text_sets[right_split]
        pair_name = f"{left_split}__{right_split}"

        cross_split_overlaps[pair_name] = {
            "shared_unique_exact_texts": len(overlap)
        }

    hard_failure_columns = (
        "missing_text_rows",
        "empty_text_rows",
        "non_string_text_rows",
        "missing_labels_rows",
        "empty_label_rows",
        "invalid_label_container_rows",
        "invalid_label_rows",
        "duplicate_label_id_rows",
    )
    hard_failures = []

    for split_stats in split_summaries:
        for column in hard_failure_columns:
            if split_stats[column] > 0:
                hard_failures.append(
                    f"{split_stats['split']}: "
                    f"{column}={split_stats[column]}"
                )

    multi_hot_example = None

    if not hard_failures:
        sample_index = 0
        sample_labels = [
            int(label_id)
            for label_id in dataset["train"][sample_index]["labels"]
        ]

        target = torch.zeros(num_labels, dtype=torch.float32)
        target[sample_labels] = 1.0

        assert target.shape == (num_labels,)
        assert target.dtype == torch.float32
        assert int(target.sum().item()) == len(set(sample_labels))

        multi_hot_example = {
            "split": "train",
            "sample_index": sample_index,
            "label_ids": sample_labels,
            "label_names": [
                id_to_label[str(label_id)]
                for label_id in sample_labels
            ],
            "shape": list(target.shape),
            "dtype": str(target.dtype),
            "values": target.tolist(),
        }

    summary = {
        "dataset_id": DATASET_ID,
        "config_name": CONFIG_NAME,
        "datasets_version": datasets.__version__,
        "actual_splits": list(actual_splits),
        "schema": reference_features.to_dict(),
        "num_labels": num_labels,
        "label_order": label_names,
        "split_summaries": split_summaries,
        "cross_split_exact_text_overlaps": cross_split_overlaps,
        "duplicate_policy": (
            "Report exact duplicate texts; keep official splits unchanged."
        ),
        "multi_hot_example": multi_hot_example,
        "validation": {
            "status": "passed" if not hard_failures else "failed",
            "hard_failures": hard_failures,
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    json_artifacts = {
        ARTIFACT_DIR / "dataset_summary.json": summary,
        ARTIFACT_DIR / "label_to_id.json": label_to_id,
        ARTIFACT_DIR / "id_to_label.json": id_to_label,
    }

    for path, payload in json_artifacts.items():
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    split_table_path = ARTIFACT_DIR / "split_summary.csv"

    with split_table_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(split_summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(split_summaries)

    print(f"Dataset: {DATASET_ID} ({CONFIG_NAME})")
    print(f"Schema: {reference_features}")
    print(f"Number of labels: {num_labels}")
    print(f"Label order: {label_names}")

    for stats in split_summaries:
        print(
            f"{stats['split']}: "
            f"rows={stats['num_rows']}, "
            f"duplicate_rows="
            f"{stats['duplicate_text_rows_beyond_first']}"
        )

    for pair_name, result in cross_split_overlaps.items():
        print(
            f"{pair_name}: "
            f"shared_exact_texts="
            f"{result['shared_unique_exact_texts']}"
        )

    print(
        "Multi-hot example: "
        f"shape={multi_hot_example['shape'] if multi_hot_example else None}, "
        f"dtype={multi_hot_example['dtype'] if multi_hot_example else None}"
    )

    print(f"Artifacts written to: {ARTIFACT_DIR.resolve()}")

    if hard_failures:
        raise SystemExit(
            "AUDIT FAILED:\n- " + "\n- ".join(hard_failures)
        )

    print("AUDIT PASSED")


if __name__ == "__main__":
    main()


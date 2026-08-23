from pathlib import Path
import json

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
)

from run_mean_pooling_mlp import (
    calculate_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STANDARD_RUN_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "transformer_encoder"
    / "run.json"
)
STANDARD_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "transformer_encoder"
    / "predictions.npz"
)

WEIGHTED_RUN_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "transformer_pos_weight"
    / "run.json"
)

WEIGHTED_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "transformer_pos_weight"
    / "predictions.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "imbalance_threshold"
)
EXPERIMENTS_PATH = (
    OUTPUT_DIR / "experiments.json"
)
RESULTS_PATH = OUTPUT_DIR / "results.txt"

FIXED_THRESHOLD = 0.5

THRESHOLD_GRID = np.round(
    np.arange(
        0.05,
        1.00,
        0.05,
    ),
    decimals=2,
)

STRATEGY_ORDER = (
    "standard_fixed",
    "standard_global",
    "standard_per_label",
    "weighted_fixed",
    "weighted_global",
    "weighted_per_label",
)

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing run artifact: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )

# load 4 array, targets là label gốc, probabilities là dự đoán
def load_prediction_artifact(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction artifact: {path}"
        )

    with np.load(path) as artifact:
        required_keys = {
            "validation_targets",
            "validation_probabilities",
            "test_targets",
            "test_probabilities",
        }

        missing_keys = (
            required_keys - set(artifact.files)
        )

        if missing_keys:
            raise ValueError(
                f"{path} is missing arrays: "
                f"{sorted(missing_keys)}"
            )

        return {
            key: artifact[key].copy()
            for key in required_keys
        }

def choose_tied_threshold(
    thresholds,
    scores,
):
    """
    Chọn threshold có F1 cao nhất.

    Tie-breaker:
    1. Gần 0.5 nhất.
    2. Nếu vẫn bằng nhau, chọn threshold cao hơn.
    """
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )
    thresholds = np.asarray(
        thresholds,
        dtype=np.float64,
    )

    best_score = scores.max()

    tied_indices = np.where(
        np.isclose(
            scores,
            best_score,
            rtol=0.0,
            atol=1e-12,
        )
    )[0]

    selected_index = min(
        tied_indices,
        key=lambda index: (
            abs(
                thresholds[index]
                - FIXED_THRESHOLD
            ),
            -thresholds[index],
        ),
    )

    return (
        float(thresholds[selected_index]),
        float(scores[selected_index]),
    )

# hàm để apply threshold vào array và trả về prediction
def apply_thresholds(
    probabilities,
    thresholds,
):
    threshold_array = np.asarray(
        thresholds,
        dtype=np.float32,
    )

    predictions = (
        probabilities >= threshold_array
    ).astype(np.int8)

    return predictions

# thử từng threshold trong khoảng 0.05 đến 1 
# rồi tính metric f1,...cho prediction của từng giá trị
# lưu threshold và f1 score cho từng giá trị threshold, sau đó chọn ra f1 score tốt nhất
# nếu có f1score bằng nhau ta chọn theo threshold gần với 0,5 nhất
def tune_global_threshold(
    validation_targets,
    validation_probabilities,
    label_names,
):
    candidate_rows = []

    for threshold in THRESHOLD_GRID:
        predictions = apply_thresholds(
            validation_probabilities,
            threshold,
        )

        metrics = calculate_metrics(
            y_true=validation_targets,
            y_pred=predictions,
            label_names=label_names,
        )

        candidate_rows.append(
            {
                "threshold": float(
                    threshold
                ),
                "validation_macro_f1": (
                    metrics["macro"]["f1"]
                ),
            }
        )

    selected_threshold, best_f1 = (
        choose_tied_threshold(
            thresholds=[
                row["threshold"]
                for row in candidate_rows
            ],
            scores=[
                row["validation_macro_f1"]
                for row in candidate_rows
            ],
        )
    )

    return {
        "selected_threshold": (
            selected_threshold
        ),
        "best_validation_macro_f1": (
            best_f1
        ),
        "candidates": candidate_rows,
    }

# đối với tune threshold cho từng label, ta sẽ tách các label và thử với từng threshold và chọn ra f1 score tốt nhất tương tự như trên
def tune_per_label_thresholds(
    validation_targets,
    validation_probabilities,
    label_names,
):
    selected_thresholds = []
    tuning_rows = []

    for label_id, label_name in enumerate(
        label_names
    ):
        label_targets = (
            validation_targets[:, label_id]
        )
        label_probabilities = (
            validation_probabilities[
                :,
                label_id,
            ]
        )

        candidate_f1_scores = []

        for threshold in THRESHOLD_GRID:
            label_predictions = (
                label_probabilities
                >= threshold
            ).astype(np.int8)

            _, _, f1, _ = (
                precision_recall_fscore_support(
                    label_targets,
                    label_predictions,
                    average="binary",
                    zero_division=0,
                )
            )

            candidate_f1_scores.append(
                float(f1)
            )

        selected_threshold, best_f1 = (
            choose_tied_threshold(
                thresholds=THRESHOLD_GRID,
                scores=candidate_f1_scores,
            )
        )

        selected_thresholds.append(
            selected_threshold
        )

        tuning_rows.append(
            {
                "label_id": label_id,
                "label": label_name,
                "selected_threshold": (
                    selected_threshold
                ),
                "best_validation_f1": (
                    best_f1
                ),
                "candidates": [
                    {
                        "threshold": float(
                            threshold
                        ),
                        "validation_f1": (
                            candidate_f1_scores[
                                candidate_index
                            ]
                        ),
                    }
                    for candidate_index, threshold
                    in enumerate(
                        THRESHOLD_GRID
                    )
                ],
            }
        )

    return {
        "selected_thresholds": (
            selected_thresholds
        ),
        "per_label_tuning": tuning_rows,
    }


def attach_threshold_to_per_label(
    metrics,
    thresholds,
):
    threshold_array = np.asarray(
        thresholds,
        dtype=np.float64,
    )

    if threshold_array.ndim == 0:
        threshold_array = np.full(
            len(metrics["per_label"]),
            float(threshold_array),
            dtype=np.float64,
        )

    per_label_with_thresholds = []

    for row, threshold in zip(
        metrics["per_label"],
        threshold_array,
    ):
        per_label_with_thresholds.append(
            {
                **row,
                "threshold": float(
                    threshold
                ),
            }
        )

    return {
        "macro": metrics["macro"],
        "micro": metrics["micro"],
        "per_label": (
            per_label_with_thresholds
        ),
    }

# đóng băng threshold rồi áp dụng rồi tính metric trên valid và test
def evaluate_strategy(
    name,
    loss_strategy,
    threshold_strategy,
    thresholds,
    prediction_artifact,
    label_names,
    tuning_evidence,
):
    validation_predictions = (
        apply_thresholds(
            prediction_artifact[
                "validation_probabilities"
            ],
            thresholds,
        )
    )

    test_predictions = apply_thresholds(
        prediction_artifact[
            "test_probabilities"
        ],
        thresholds,
    )

    validation_metrics = calculate_metrics(
        y_true=prediction_artifact[
            "validation_targets"
        ],
        y_pred=validation_predictions,
        label_names=label_names,
    )

    test_metrics = calculate_metrics(
        y_true=prediction_artifact[
            "test_targets"
        ],
        y_pred=test_predictions,
        label_names=label_names,
    )

    return {
        "name": name,
        "loss_strategy": loss_strategy,
        "threshold_strategy": (
            threshold_strategy
        ),
        "thresholds": (
            [float(value) for value in thresholds]
            if np.asarray(thresholds).ndim > 0
            else float(thresholds)
        ),
        "tuning_evidence": tuning_evidence,
        "validation": (
            attach_threshold_to_per_label(
                validation_metrics,
                thresholds,
            )
        ),
        "test": (
            attach_threshold_to_per_label(
                test_metrics,
                thresholds,
            )
        ),
    }

# format để ghi kết quả cho 6 strategy
# 6 strategy gồm: 
# - standard fixed/global/per_label: standard dùng standard prediction (không pos weight) và fixed: 0.5 global: tune threshold trên toàn bộ sample per_label: tune cho từng label
# - weight fixed/global/per_label: weight dùng pos weight prediction và ....
def format_results(
    experiment_rows,
    selected_strategy,
):
    lines = [
        "Transformer Imbalance and Threshold Results",
        "",
        (
            "Selected by validation macro-F1: "
            f"{selected_strategy}"
        ),
        "",
        (
            "Strategy | Val Macro P/R/F1 | "
            "Val Micro P/R/F1 | "
            "Test Macro P/R/F1 | "
            "Test Micro P/R/F1"
        ),
        "-" * 150,
    ]

    for row in experiment_rows:
        validation_macro = row["validation"]["macro"]
        validation_micro = row["validation"]["micro"]
        test_macro = row["test"]["macro"]
        test_micro = row["test"]["micro"]

        lines.append(
            f"{row['name']} | "
            f"{validation_macro['precision']:.4f}/"
            f"{validation_macro['recall']:.4f}/"
            f"{validation_macro['f1']:.4f} | "
            f"{validation_micro['precision']:.4f}/"
            f"{validation_micro['recall']:.4f}/"
            f"{validation_micro['f1']:.4f} | "
            f"{test_macro['precision']:.4f}/"
            f"{test_macro['recall']:.4f}/"
            f"{test_macro['f1']:.4f} | "
            f"{test_micro['precision']:.4f}/"
            f"{test_micro['recall']:.4f}/"
            f"{test_micro['f1']:.4f}"
        )

    lines.append("")
    lines.append("Selected thresholds")

    for row in experiment_rows:
        if row["threshold_strategy"] == "fixed":
            threshold_text = (
                f"{row['thresholds']:.2f}"
            )
        elif (
            row["threshold_strategy"] == "global"
        ):
            threshold_text = (
                f"{row['thresholds']:.2f}"
            )
        else:
            threshold_text = ", ".join(
                f"{value:.2f}"
                for value in row["thresholds"]
            )

        lines.append(
            f"{row['name']}: {threshold_text}"
        )

    lines.append("")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    standard_run = load_json(
        STANDARD_RUN_PATH
    )
    weighted_run = load_json(
        WEIGHTED_RUN_PATH
    )

    standard_predictions = (
        load_prediction_artifact(
            STANDARD_PREDICTIONS_PATH
        )
    )
    weighted_predictions = (
        load_prediction_artifact(
            WEIGHTED_PREDICTIONS_PATH
        )
    )

    label_names = standard_run[
        "label_names"
    ]

    if weighted_run["label_names"] != label_names:
        raise ValueError(
            "Standard and weighted label orders differ"
        )

    if (
        standard_run["preprocessing"]
        != weighted_run["preprocessing"]
    ):
        raise ValueError(
            "Standard and weighted preprocessing "
            "contracts differ"
        )

    for split_name in (
        "validation",
        "test",
    ):
        target_key = f"{split_name}_targets"

        if not np.array_equal(
            standard_predictions[target_key],
            weighted_predictions[target_key],
        ):
            raise ValueError(
                f"Standard and weighted "
                f"{split_name} targets differ"
            )

    standard_global = (
        tune_global_threshold(
            validation_targets=(
                standard_predictions[
                    "validation_targets"
                ]
            ),
            validation_probabilities=(
                standard_predictions[
                    "validation_probabilities"
                ]
            ),
            label_names=label_names,
        )
    )

    standard_per_label = (
        tune_per_label_thresholds(
            validation_targets=(
                standard_predictions[
                    "validation_targets"
                ]
            ),
            validation_probabilities=(
                standard_predictions[
                    "validation_probabilities"
                ]
            ),
            label_names=label_names,
        )
    )

    weighted_global = (
        tune_global_threshold(
            validation_targets=(
                weighted_predictions[
                    "validation_targets"
                ]
            ),
            validation_probabilities=(
                weighted_predictions[
                    "validation_probabilities"
                ]
            ),
            label_names=label_names,
        )
    )

    weighted_per_label = (
        tune_per_label_thresholds(
            validation_targets=(
                weighted_predictions[
                    "validation_targets"
                ]
            ),
            validation_probabilities=(
                weighted_predictions[
                    "validation_probabilities"
                ]
            ),
            label_names=label_names,
        )
    )

    experiment_rows = [
        evaluate_strategy(
            name="standard_fixed",
            loss_strategy="standard_bce",
            threshold_strategy="fixed",
            thresholds=FIXED_THRESHOLD,
            prediction_artifact=(
                standard_predictions
            ),
            label_names=label_names,
            tuning_evidence=None,
        ),
        evaluate_strategy(
            name="standard_global",
            loss_strategy="standard_bce",
            threshold_strategy="global",
            thresholds=standard_global[
                "selected_threshold"
            ],
            prediction_artifact=(
                standard_predictions
            ),
            label_names=label_names,
            tuning_evidence=standard_global,
        ),
        evaluate_strategy(
            name="standard_per_label",
            loss_strategy="standard_bce",
            threshold_strategy="per_label",
            thresholds=standard_per_label[
                "selected_thresholds"
            ],
            prediction_artifact=(
                standard_predictions
            ),
            label_names=label_names,
            tuning_evidence=(
                standard_per_label
            ),
        ),
        evaluate_strategy(
            name="weighted_fixed",
            loss_strategy=(
                "capped_pos_weight_bce"
            ),
            threshold_strategy="fixed",
            thresholds=FIXED_THRESHOLD,
            prediction_artifact=(
                weighted_predictions
            ),
            label_names=label_names,
            tuning_evidence=None,
        ),
        evaluate_strategy(
            name="weighted_global",
            loss_strategy=(
                "capped_pos_weight_bce"
            ),
            threshold_strategy="global",
            thresholds=weighted_global[
                "selected_threshold"
            ],
            prediction_artifact=(
                weighted_predictions
            ),
            label_names=label_names,
            tuning_evidence=weighted_global,
        ),
        evaluate_strategy(
            name="weighted_per_label",
            loss_strategy=(
                "capped_pos_weight_bce"
            ),
            threshold_strategy="per_label",
            thresholds=weighted_per_label[
                "selected_thresholds"
            ],
            prediction_artifact=(
                weighted_predictions
            ),
            label_names=label_names,
            tuning_evidence=(
                weighted_per_label
            ),
        ),
    ]

    strategy_order_index = {
        name: index
        for index, name
        in enumerate(STRATEGY_ORDER)
    }

    selected_row = max(
        experiment_rows,
        key=lambda row: (
            row["validation"][
                "macro"
            ]["f1"],
            -strategy_order_index[
                row["name"]
            ],
        ),
    )

    experiments_record = {
        "selection_policy": {
            "metric": (
                "validation_macro_f1"
            ),
            "strategy_tie_breaker": (
                "first strategy in declared order"
            ),
            "threshold_grid": [
                float(value)
                for value in THRESHOLD_GRID
            ],
            "threshold_tie_breaker": [
                "closest_to_0.5",
                "higher_threshold",
            ],
            "declared_strategy_order": list(
                STRATEGY_ORDER
            ),
        },
        "sources": {
            "standard_run": str(
                STANDARD_RUN_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "standard_predictions": str(
                STANDARD_PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "weighted_run": str(
                WEIGHTED_RUN_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "weighted_predictions": str(
                WEIGHTED_PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
        "label_names": label_names,
        "selected_strategy": (
            selected_row["name"]
        ),
        "experiments": experiment_rows,
    }

    EXPERIMENTS_PATH.write_text(
        json.dumps(
            experiments_record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    RESULTS_PATH.write_text(
        format_results(
            experiment_rows=experiment_rows,
            selected_strategy=(
                selected_row["name"]
            ),
        ),
        encoding="utf-8",
    )

    print(
        "Selected strategy: "
        f"{selected_row['name']}"
    )
    print(
        "Validation macro-F1: "
        f"{selected_row['validation']['macro']['f1']:.4f}"
    )
    print(
        "Test macro-F1: "
        f"{selected_row['test']['macro']['f1']:.4f}"
    )
    print(
        f"Saved experiments: "
        f"{EXPERIMENTS_PATH}"
    )
    print(
        f"Saved concise results: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
    

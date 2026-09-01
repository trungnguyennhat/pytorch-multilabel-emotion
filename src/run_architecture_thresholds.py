from pathlib import Path
import json

import numpy as np

from run_threshold_experiments import (
    FIXED_THRESHOLD,
    THRESHOLD_GRID,
    evaluate_strategy,
    load_json,
    load_prediction_artifact,
    tune_global_threshold,
    tune_per_label_thresholds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "architecture_thresholds"

ARCHITECTURES = {
    "tfidf_logreg": "tfidf_logreg",
    "mean_pooling_mlp": "mean_pooling_mlp",
    "bilstm_attention": "bilstm_attention",
    "transformer_encoder": "transformer_encoder",
    "bert_pretrained": "bert_pretrained",
}

STRATEGIES = ("fixed", "global", "per_label")


def format_results(rows, selected_by_architecture):
    lines = [
        "Threshold Tuning for All Architectures",
        "",
        "Architecture | Strategy | Val Macro F1 | Val Micro F1 | Test Macro F1 | Test Micro F1",
        "-" * 105,
    ]
    for row in rows:
        marker = " *" if row["name"] == selected_by_architecture[row["architecture"]] else ""
        lines.append(
            f"{row['architecture']} | {row['threshold_strategy']}{marker} | "
            f"{row['validation']['macro']['f1']:.4f} | "
            f"{row['validation']['micro']['f1']:.4f} | "
            f"{row['test']['macro']['f1']:.4f} | "
            f"{row['test']['micro']['f1']:.4f}"
        )
    lines.extend(["", "* selected by validation macro-F1", ""])
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    reference_labels = None
    reference_targets = None

    for architecture, artifact_dir in ARCHITECTURES.items():
        artifact_root = PROJECT_ROOT / "outputs" / artifact_dir
        run = load_json(artifact_root / "run.json")
        predictions = load_prediction_artifact(artifact_root / "predictions.npz")
        label_names = run["label_names"]

        if reference_labels is None:
            reference_labels = label_names
            reference_targets = {
                split: predictions[f"{split}_targets"]
                for split in ("validation", "test")
            }
        elif label_names != reference_labels:
            raise ValueError(f"{architecture} label order differs")
        else:
            for split in ("validation", "test"):
                if not np.array_equal(reference_targets[split], predictions[f"{split}_targets"]):
                    raise ValueError(f"{architecture} {split} targets differ")

        global_tuning = tune_global_threshold(
            predictions["validation_targets"],
            predictions["validation_probabilities"],
            label_names,
        )
        per_label_tuning = tune_per_label_thresholds(
            predictions["validation_targets"],
            predictions["validation_probabilities"],
            label_names,
        )
        thresholds = {
            "fixed": FIXED_THRESHOLD,
            "global": global_tuning["selected_threshold"],
            "per_label": per_label_tuning["selected_thresholds"],
        }
        evidence = {
            "fixed": None,
            "global": global_tuning,
            "per_label": per_label_tuning,
        }

        for strategy in STRATEGIES:
            row = evaluate_strategy(
                name=f"{architecture}_{strategy}",
                loss_strategy="not_applicable" if architecture == "tfidf_logreg" else "standard_bce",
                threshold_strategy=strategy,
                thresholds=thresholds[strategy],
                prediction_artifact=predictions,
                label_names=label_names,
                tuning_evidence=evidence[strategy],
            )
            row["architecture"] = architecture
            rows.append(row)

    selected_by_architecture = {}
    for architecture in ARCHITECTURES:
        candidates = [row for row in rows if row["architecture"] == architecture]
        selected_by_architecture[architecture] = max(
            candidates,
            key=lambda row: (
                row["validation"]["macro"]["f1"],
                -STRATEGIES.index(row["threshold_strategy"]),
            ),
        )["name"]

    record = {
        "selection_policy": {
            "metric": "validation_macro_f1",
            "threshold_grid": [float(value) for value in THRESHOLD_GRID],
            "threshold_tie_breaker": ["closest_to_0.5", "higher_threshold"],
            "strategy_tie_breaker": "first strategy in declared order",
            "declared_strategy_order": list(STRATEGIES),
        },
        "sources": {
            architecture: f"outputs/{artifact_dir}/predictions.npz"
            for architecture, artifact_dir in ARCHITECTURES.items()
        },
        "label_names": reference_labels,
        "selected_by_architecture": selected_by_architecture,
        "experiments": rows,
    }
    (OUTPUT_DIR / "experiments.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "results.txt").write_text(
        format_results(rows, selected_by_architecture), encoding="utf-8"
    )
    print(format_results(rows, selected_by_architecture))


if __name__ == "__main__":
    main()

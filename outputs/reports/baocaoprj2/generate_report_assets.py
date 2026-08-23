from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = REPORT_DIR / "hinh"
CONTRACT_PATH = ROOT / "data" / "artifacts" / "dataset_contract.json"
MODEL_DIRS = {
    "TF-IDF + LR": ROOT / "outputs" / "tfidf_logreg",
    "Mean Pooling MLP": ROOT / "outputs" / "mean_pooling_mlp",
    "BiLSTM + Attention": ROOT / "outputs" / "bilstm_attention",
    "Transformer Encoder": ROOT / "outputs" / "transformer_encoder",
}
NEURAL_MODELS = ["Mean Pooling MLP", "BiLSTM + Attention", "Transformer Encoder"]
THRESHOLD_PATH = ROOT / "outputs" / "imbalance_threshold" / "experiments.json"
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def clean_splits(contract: dict):
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / "google-research-datasets___go_emotions"
        / "simplified"
    )
    cached_versions = sorted(cache_root.glob("*/*"), key=lambda path: path.stat().st_mtime)
    if cached_versions and all(
        (cached_versions[-1] / f"go_emotions-{split}.arrow").exists()
        for split in ("train", "validation", "test")
    ):
        cache_dir = cached_versions[-1]
        dataset = DatasetDict(
            {
                split: Dataset.from_file(str(cache_dir / f"go_emotions-{split}.arrow"))
                for split in ("train", "validation", "test")
            }
        )
        loaded_from_arrow_cache = True
    else:
        dataset = load_dataset(
            contract["dataset"]["source"],
            contract["dataset"]["config"],
        )
        loaded_from_arrow_cache = False
    clean = {}
    for split_name in ("train", "validation", "test"):
        expected = contract["official_splits"][split_name]["fingerprint"]
        if not loaded_from_arrow_cache and dataset[split_name]._fingerprint != expected:
            raise RuntimeError(f"Fingerprint changed for {split_name}")
        if len(dataset[split_name]) != contract["official_splits"][split_name]["num_rows"]:
            raise RuntimeError(f"Unexpected official size for {split_name}")
        if dataset[split_name].column_names != contract["official_splits"][split_name]["columns"]:
            raise RuntimeError(f"Unexpected columns for {split_name}")
        excluded = set(
            contract["evaluation_policy"]["splits"][split_name]["excluded_ids"]
        )
        indices = [
            index
            for index, row_id in enumerate(dataset[split_name]["id"])
            if row_id not in excluded
        ]
        clean[split_name] = dataset[split_name].select(indices)
        expected_rows = contract["evaluation_policy"]["splits"][split_name][
            "clean_num_rows"
        ]
        if len(clean[split_name]) != expected_rows:
            raise RuntimeError(f"Unexpected clean size for {split_name}")
    return clean


def load_artifacts(contract: dict):
    label_names = contract["schema"]["label_names"]
    runs = {name: load_json(path / "run.json") for name, path in MODEL_DIRS.items()}
    predictions = {
        name: np.load(path / "predictions.npz") for name, path in MODEL_DIRS.items()
    }
    expected_shape = (5385, len(label_names))
    neural_targets = []
    for name in NEURAL_MODELS:
        bundle = predictions[name]
        if bundle["test_predictions"].shape != expected_shape:
            raise RuntimeError(f"Unexpected test prediction shape for {name}")
        if bundle["test_targets"].shape != expected_shape:
            raise RuntimeError(f"Unexpected test target shape for {name}")
        neural_targets.append(bundle["test_targets"].astype(np.int8))
        if runs[name]["label_names"] != label_names:
            raise RuntimeError(f"Label order differs for {name}")
    if not all(np.array_equal(neural_targets[0], item) for item in neural_targets[1:]):
        raise RuntimeError("Neural test targets differ")
    thresholds = load_json(THRESHOLD_PATH)
    if thresholds["label_names"] != label_names:
        raise RuntimeError("Threshold label order differs")
    return runs, predictions, thresholds


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig, filename: str) -> None:
    fig.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_label_distribution(train_split, label_names: list[str]) -> None:
    supports = np.zeros(len(label_names), dtype=int)
    for labels in train_split["labels"]:
        supports[labels] += 1
    order = np.argsort(supports)
    median = float(np.median(supports))
    fig, ax = plt.subplots(figsize=(8.4, 8.2))
    ax.barh(np.array(label_names)[order], supports[order], color="#4C78A8")
    ax.axvline(median, color="#E45756", linestyle="--", linewidth=1.6,
               label=f"Trung vị = {median:.0f}")
    ax.set_xlabel("Số mẫu dương trong tập train")
    ax.set_title("Phân bố support của 28 nhãn cảm xúc")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "label_distribution.png")


def plot_model_comparison(runs: dict) -> None:
    metrics = ["Macro Precision", "Macro Recall", "Macro F1", "Micro F1"]
    values = []
    for run in runs.values():
        test = run["results"]["test"]
        values.append(
            [test["macro"]["precision"], test["macro"]["recall"],
             test["macro"]["f1"], test["micro"]["f1"]]
        )
    values = np.asarray(values)
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    for index, (name, scores) in enumerate(zip(runs, values)):
        bars = ax.bar(x + (index - 1.5) * width, scores, width,
                      label=name, color=COLORS[index])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Điểm số trên test")
    ax.set_title("So sánh bốn kiến trúc tại threshold 0.5")
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "model_comparison.png")


def plot_threshold_study(threshold_data: dict) -> None:
    experiments = threshold_data["experiments"]
    display = {
        "standard_fixed": "Standard\n0.5",
        "standard_global": "Standard\nglobal",
        "standard_per_label": "Standard\nper-label",
        "weighted_fixed": "Weighted\n0.5",
        "weighted_global": "Weighted\nglobal",
        "weighted_per_label": "Weighted\nper-label",
    }
    names = [display[item["name"]] for item in experiments]
    metric_names = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    x = np.arange(len(experiments))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    for offset, ((key, label), color) in enumerate(zip(metric_names, COLORS[:3])):
        scores = [item["test"]["macro"][key] for item in experiments]
        bars = ax.bar(x + (offset - 1) * width, scores, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)
    selected_index = [item["name"] for item in experiments].index(
        threshold_data["selected_strategy"]
    )
    ax.axvspan(selected_index - 0.48, selected_index + 0.48,
               color="#72B7B2", alpha=0.14, label="Được chọn bằng validation")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Macro score trên test")
    ax.set_title("Ảnh hưởng của loss weighting và threshold")
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    save_figure(fig, "threshold_study.png")


def plot_per_label_f1(threshold_data: dict) -> None:
    by_name = {item["name"]: item for item in threshold_data["experiments"]}
    fixed = by_name["standard_fixed"]["test"]["per_label"]
    tuned = by_name["standard_per_label"]["test"]["per_label"]
    fixed_f1 = np.array([item["f1"] for item in fixed])
    tuned_f1 = np.array([item["f1"] for item in tuned])
    support = np.array([item["support"] for item in fixed])
    labels = np.array([item["label"] for item in fixed])
    order = np.argsort(tuned_f1 - fixed_f1)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 9.5))
    ax.hlines(y, fixed_f1[order], tuned_f1[order], color="#B9C0C7", linewidth=2)
    ax.scatter(fixed_f1[order], y, color="#4C78A8", s=28, label="Fixed 0.5", zorder=3)
    ax.scatter(tuned_f1[order], y, color="#E45756", s=28, label="Per-label", zorder=3)
    tick_labels = [f"{label} (n={count})" for label, count in zip(labels[order], support[order])]
    ax.set_yticks(y, tick_labels)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("F1 trên test")
    ax.set_title("Per-label F1 trước và sau threshold tuning")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "per_label_f1.png")


def plot_slice_performance(test_split, predictions: dict) -> dict:
    texts = test_split["text"]
    token_counts = np.array([len(text.split()) for text in texts])
    targets = predictions[NEURAL_MODELS[0]]["test_targets"].astype(np.int8)
    cardinality = targets.sum(axis=1)
    length_groups = [
        ("1--10", token_counts <= 10),
        ("11--20", (token_counts >= 11) & (token_counts <= 20)),
        (">20", token_counts > 20),
    ]
    cardinality_groups = [
        ("1", cardinality == 1),
        ("2", cardinality == 2),
        ("≥3", cardinality >= 3),
    ]
    result = {"length": {}, "cardinality": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    for axis, groups, result_key, title, xlabel in [
        (axes[0], length_groups, "length", "Theo độ dài văn bản", "Số token"),
        (axes[1], cardinality_groups, "cardinality", "Theo số nhãn thật", "Label cardinality"),
    ]:
        x = np.arange(len(groups))
        width = 0.25
        for model_index, (model_name, color) in enumerate(zip(NEURAL_MODELS, COLORS[:3])):
            model_predictions = predictions[model_name]["test_predictions"].astype(np.int8)
            scores = []
            for group_name, mask in groups:
                score = f1_score(
                    targets[mask], model_predictions[mask], average="samples", zero_division=0
                )
                scores.append(score)
                result[result_key].setdefault(group_name, {})[model_name] = score
            axis.plot(x, scores, marker="o", linewidth=2, color=color, label=model_name)
        labels = [f"{name}\n(n={int(mask.sum())})" for name, mask in groups]
        axis.set_xticks(x, labels)
        axis.set_ylim(0, 0.68)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Sample-level F1")
        axis.set_title(title)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Hiệu năng theo đặc điểm của test sample", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "slice_performance.png")
    return result


def format_labels(row: np.ndarray, label_names: list[str]) -> str:
    return ", ".join(label_names[index] for index in np.flatnonzero(row)) or "none"


def choose_error_examples(test_split, targets, tuned_predictions, label_names):
    texts = test_split["text"]
    negation = re.compile(r"\b(no|not|never|n't|nothing|nobody|neither)\b", re.I)
    supports = targets.sum(axis=0)
    candidates = []
    for index, text in enumerate(texts):
        true = targets[index]
        pred = tuned_predictions[index]
        if np.array_equal(true, pred):
            continue
        false_negative_ids = np.flatnonzero((true == 1) & (pred == 0))
        if negation.search(text) and 60 <= len(text) <= 180:
            category = "Negation"
            priority = 0
            secondary = len(text)
        elif len(false_negative_ids) and supports[false_negative_ids].min() < 50:
            category = "Bỏ sót nhãn hiếm"
            priority = 1
            secondary = index
        elif true.sum() >= 3:
            category = "Nhiều nhãn"
            priority = 2
            secondary = index
        else:
            category = "Mơ hồ/nhãn gần nghĩa"
            priority = 3
            secondary = index
        candidates.append((priority, secondary, index, category))
    selected = []
    used_categories = set()
    for priority, secondary, index, category in sorted(candidates):
        if category not in used_categories:
            selected.append(
                {
                    "text": texts[index],
                    "true": format_labels(targets[index], label_names),
                    "predicted": format_labels(tuned_predictions[index], label_names),
                    "category": category,
                }
            )
            used_categories.add(category)
        if len(selected) == 4:
            break
    return selected


def write_tables(runs: dict, thresholds: dict, examples: list[dict]) -> None:
    lines = []
    lines.extend([
        r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"\textbf{Mô hình} & \textbf{Macro P} & \textbf{Macro R} & \textbf{Macro F1} & \textbf{Micro P} & \textbf{Micro R} & \textbf{Micro F1} \\",
        r"\midrule",
    ])
    for name, run in runs.items():
        test = run["results"]["test"]
        lines.append(
            f"{latex_escape(name)} & {test['macro']['precision']:.4f} & {test['macro']['recall']:.4f} & "
            f"{test['macro']['f1']:.4f} & {test['micro']['precision']:.4f} & "
            f"{test['micro']['recall']:.4f} & {test['micro']['f1']:.4f} \\\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Kết quả test của bốn kiến trúc tại threshold 0.5}",
        r"\label{tab:model-results}", r"\end{table}", "",
        r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"\textbf{Chiến lược} & \textbf{Macro P} & \textbf{Macro R} & \textbf{Macro F1} & \textbf{Micro P} & \textbf{Micro R} & \textbf{Micro F1} \\",
        r"\midrule",
    ])
    for experiment in thresholds["experiments"]:
        test = experiment["test"]
        marker = r"\textbf{" if experiment["name"] == thresholds["selected_strategy"] else ""
        close = "}" if marker else ""
        lines.append(
            f"{marker}{latex_escape(experiment['name'])}{close} & {test['macro']['precision']:.4f} & "
            f"{test['macro']['recall']:.4f} & {test['macro']['f1']:.4f} & "
            f"{test['micro']['precision']:.4f} & {test['micro']['recall']:.4f} & "
            f"{test['micro']['f1']:.4f} \\\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Kết quả test của sáu chiến lược loss và threshold}",
        r"\label{tab:threshold-results}", r"\end{table}", "",
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"\textbf{Mô hình} & \textbf{Tham số} & \textbf{Best epoch} & \textbf{Train (s)} & \textbf{Peak GPU (MB)} \\",
        r"\midrule",
    ])
    for name in NEURAL_MODELS:
        run = runs[name]
        efficiency = run["efficiency"]
        lines.append(
            f"{latex_escape(name)} & {efficiency['trainable_parameter_count']:,} & "
            f"{run['selection']['best_epoch']} & {efficiency['training_seconds']:.1f} & "
            f"{efficiency['peak_gpu_memory_mb']:.1f} \\\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Chi phí huấn luyện của ba kiến trúc neural trên cùng thiết bị}",
        r"\label{tab:efficiency}", r"\end{table}", "",
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\begin{tabular}{p{5.2cm}p{3.0cm}p{3.0cm}p{2.2cm}}", r"\toprule",
        r"\textbf{Văn bản} & \textbf{Nhãn thật} & \textbf{Dự đoán} & \textbf{Nhóm lỗi} \\",
        r"\midrule",
    ])
    for item in examples:
        text = item["text"]
        if len(text) > 145:
            text = text[:142].rstrip() + "..."
        lines.append(
            f"{latex_escape(text)} & {latex_escape(item['true'])} & "
            f"{latex_escape(item['predicted'])} & {latex_escape(item['category'])} \\\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Một số lỗi của Transformer với per-label thresholds}",
        r"\label{tab:error-examples}", r"\end{table}", "",
    ])
    (REPORT_DIR / "generated_tables.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    contract = load_json(CONTRACT_PATH)
    clean = clean_splits(contract)
    runs, predictions, thresholds = load_artifacts(contract)
    plot_label_distribution(clean["train"], contract["schema"]["label_names"])
    plot_model_comparison(runs)
    plot_threshold_study(thresholds)
    plot_per_label_f1(thresholds)
    plot_slice_performance(clean["test"], predictions)
    selected = next(
        item for item in thresholds["experiments"]
        if item["name"] == thresholds["selected_strategy"]
    )
    selected_thresholds = np.asarray(selected["thresholds"], dtype=np.float32)
    transformer = predictions["Transformer Encoder"]
    tuned_predictions = (
        transformer["test_probabilities"] >= selected_thresholds.reshape(1, -1)
    ).astype(np.int8)
    targets = transformer["test_targets"].astype(np.int8)
    examples = choose_error_examples(
        clean["test"], targets, tuned_predictions, contract["schema"]["label_names"]
    )
    write_tables(runs, thresholds, examples)
    print(f"Created 5 figures and generated_tables.tex in {REPORT_DIR}")
    # datasets/multiprocess 0.70 emits a harmless ResourceTracker shutdown
    # traceback on this Python 3.12 environment. All artifacts are closed here.
    import sys

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

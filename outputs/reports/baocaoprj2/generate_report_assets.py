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
    "BERT pretrained": ROOT / "outputs" / "bert_pretrained",
}
NEURAL_MODELS = ["Mean Pooling MLP", "BiLSTM + Attention", "Transformer Encoder"]
THRESHOLD_PATH = ROOT / "outputs" / "architecture_thresholds" / "experiments.json"
IMBALANCE_THRESHOLD_PATH = ROOT / "outputs" / "imbalance_threshold" / "experiments.json"
IMBALANCE_THRESHOLD_PATH = ROOT / "outputs" / "imbalance_threshold" / "experiments.json"
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
ARCHITECTURE_KEYS = {
    "TF-IDF + LR": "tfidf_logreg",
    "Mean Pooling MLP": "mean_pooling_mlp",
    "BiLSTM + Attention": "bilstm_attention",
    "Transformer Encoder": "transformer_encoder",
    "BERT pretrained": "bert_pretrained",
}


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


def write_method_tables(contract: dict, runs: dict) -> None:
    split_rows = []
    for split_name, display_name in (
        ("train", "Train"),
        ("validation", "Validation"),
        ("test", "Test"),
    ):
        policy = contract["evaluation_policy"]["splits"][split_name]
        official = policy["official_num_rows"]
        clean = policy["clean_num_rows"]
        split_rows.append(
            f"{display_name} & {official:,} & {official - clean:,} & {clean:,} \\\\"
        )
    split_table = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"\textbf{Split} & \textbf{Official} & \textbf{Loại do trùng xuyên split} & \textbf{Clean} \\",
        r"\midrule", *split_rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{Số mẫu trước và sau khi tạo text-disjoint evaluation views}",
        r"\label{tab:clean-splits}", r"\end{table}", "",
    ]
    (REPORT_DIR / "generated_clean_split_table.tex").write_text(
        "\n".join(split_table), encoding="utf-8"
    )

    mlp = runs["Mean Pooling MLP"]
    bilstm = runs["BiLSTM + Attention"]
    transformer = runs["Transformer Encoder"]
    bert = runs["BERT pretrained"]
    rows = [
        (
            "MLP", mlp["model_config"]["embedding_dim"],
            mlp["model_config"]["hidden_dim"], "1 MLP", "--",
            mlp["model_config"]["dropout"], mlp["training_config"],
            mlp["preprocessing"]["max_length"], "--",
        ),
        (
            "BiLSTM", bilstm["model_config"]["embedding_dim"],
            f"{bilstm['model_config']['lstm_hidden_dim']}/hướng", "1 BiLSTM",
            "--", bilstm["model_config"]["dropout"], bilstm["training_config"],
            bilstm["preprocessing"]["max_length"], "--",
        ),
        (
            "Transformer", transformer["model_config"]["d_model"],
            transformer["model_config"]["d_model"],
            f"{transformer['model_config']['num_encoder_layers']} encoder",
            transformer["model_config"]["num_heads"],
            transformer["model_config"]["dropout"], transformer["training_config"],
            transformer["preprocessing"]["max_length"], "--",
        ),
        (
            "BERT-base-cased", 768, 768, "12 encoder", 12, 0.1,
            bert["training_config"], bert["preprocessing"]["max_length"], "10\\%",
        ),
    ]
    table_rows = []
    for name, embedding, hidden, layers, heads, dropout, training, max_length, warmup in rows:
        epochs = training.get("max_epochs", training.get("maximum_epochs", training.get("epochs")))
        lr = training["learning_rate"]
        lr_text = r"$10^{-3}$" if lr == 0.001 else r"$5\times10^{-5}$"
        table_rows.append(
            f"{name} & {embedding} & {hidden} & {layers} & {heads} & {dropout} & "
            f"{training['optimizer']} & {lr_text} & {training['batch_size']} & "
            f"{epochs} & {max_length} & {warmup} \\\\"
        )
    hyper_table = [
        r"\begin{table}[H]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrlrrrrr}", r"\toprule",
        r"\textbf{Model} & \textbf{Emb.} & \textbf{Hidden} & \textbf{Layers} & \textbf{Heads} & \textbf{Dropout} & \textbf{Optimizer} & \textbf{LR} & \textbf{Batch} & \textbf{Epochs} & \textbf{Max len.} & \textbf{Warmup} \\",
        r"\midrule", *table_rows, r"\bottomrule", r"\end{tabular}%", r"}",
        r"\caption{Hyperparameter của bốn neural models}",
        r"\label{tab:neural-hyperparameters}", r"\end{table}", "",
    ]
    (REPORT_DIR / "generated_hyperparameters_table.tex").write_text(
        "\n".join(hyper_table), encoding="utf-8"
    )


def write_pos_weight_table() -> None:
    experiments = load_json(IMBALANCE_THRESHOLD_PATH)["experiments"]
    by_name = {item["name"]: item for item in experiments}
    configurations = (
        ("BCE thường", "Fixed 0.5", "standard_fixed"),
        ("BCE thường", "Per-label tuned", "standard_per_label"),
        (r"BCE + \texttt{pos\_weight}", "Fixed 0.5", "weighted_fixed"),
        (r"BCE + \texttt{pos\_weight}", "Per-label tuned", "weighted_per_label"),
    )
    rows = []
    for loss, threshold, key in configurations:
        macro = by_name[key]["test"]["macro"]
        rows.append(
            f"{loss} & {threshold} & {macro['precision']:.4f} & "
            f"{macro['recall']:.4f} & {macro['f1']:.4f} \\\\"
        )
    table = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"\textbf{Loss} & \textbf{Threshold} & \textbf{Macro P} & \textbf{Macro R} & \textbf{Macro F1} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{Ảnh hưởng của \texttt{pos\_weight} và threshold trên Transformer}",
        r"\label{tab:pos-weight-results}", r"\end{table}", "",
    ]
    (REPORT_DIR / "generated_pos_weight_table.tex").write_text(
        "\n".join(table), encoding="utf-8"
    )


def write_pos_weight_table() -> None:
    experiments = load_json(IMBALANCE_THRESHOLD_PATH)["experiments"]
    by_name = {item["name"]: item for item in experiments}
    configurations = (
        ("BCE thường", "Fixed 0.5", "standard_fixed"),
        ("BCE thường", "Per-label tuned", "standard_per_label"),
        (r"BCE + \texttt{pos\_weight}", "Fixed 0.5", "weighted_fixed"),
        (r"BCE + \texttt{pos\_weight}", "Per-label tuned", "weighted_per_label"),
    )
    rows = []
    for loss, threshold, key in configurations:
        macro = by_name[key]["test"]["macro"]
        rows.append(
            f"{loss} & {threshold} & {macro['precision']:.4f} & "
            f"{macro['recall']:.4f} & {macro['f1']:.4f} \\\\"
        )
    table = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"\textbf{Loss} & \textbf{Threshold} & \textbf{Macro P} & \textbf{Macro R} & \textbf{Macro F1} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{Ảnh hưởng của \texttt{pos\_weight} và threshold trên Transformer}",
        r"\label{tab:pos-weight-results}", r"\end{table}", "",
    ]
    (REPORT_DIR / "generated_pos_weight_table.tex").write_text(
        "\n".join(table), encoding="utf-8"
    )


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
    expected_shapes = {
        "validation": (5383, len(label_names)),
        "test": (5385, len(label_names)),
    }
    reference_targets = {}
    for name in MODEL_DIRS:
        bundle = predictions[name]
        for split, expected_shape in expected_shapes.items():
            if bundle[f"{split}_probabilities"].shape != expected_shape:
                raise RuntimeError(f"Unexpected {split} probability shape for {name}")
            if bundle[f"{split}_targets"].shape != expected_shape:
                raise RuntimeError(f"Unexpected {split} target shape for {name}")
            targets = bundle[f"{split}_targets"].astype(np.int8)
            if split not in reference_targets:
                reference_targets[split] = targets
            elif not np.array_equal(reference_targets[split], targets):
                raise RuntimeError(f"{split.title()} targets differ for {name}")
        if runs[name]["label_names"] != label_names:
            raise RuntimeError(f"Label order differs for {name}")
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
    width = 0.16
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    for index, (name, scores) in enumerate(zip(runs, values)):
        bars = ax.bar(x + (index - 2) * width, scores, width,
                      label=name, color=COLORS[index])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Điểm số trên test")
    ax.set_title("So sánh năm hệ thống tại threshold 0.5")
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "model_comparison.png")


def plot_threshold_study(threshold_data: dict) -> None:
    by_name = {item["name"]: item for item in threshold_data["experiments"]}
    names = list(ARCHITECTURE_KEYS)
    fixed = [by_name[f"{key}_fixed"]["test"]["macro"]["f1"] for key in ARCHITECTURE_KEYS.values()]
    tuned = [
        by_name[threshold_data["selected_by_architecture"][key]]["test"]["macro"]["f1"]
        for key in ARCHITECTURE_KEYS.values()
    ]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    first = ax.bar(x - width / 2, fixed, width, label="Fixed 0.5", color="#4C78A8")
    second = ax.bar(x + width / 2, tuned, width, label="Threshold chọn trên validation", color="#E45756")
    ax.bar_label(first, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(second, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, [name.replace(" ", "\n", 1) for name in names])
    ax.set_ylim(0, 0.58)
    ax.set_ylabel("Macro-F1 trên test")
    ax.set_title("Macro-F1 trước và sau threshold tuning")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, "threshold_study.png")


def plot_per_label_f1(threshold_data: dict) -> None:
    by_name = {item["name"]: item for item in threshold_data["experiments"]}
    fixed = by_name["transformer_encoder_fixed"]["test"]["per_label"]
    tuned = by_name["transformer_encoder_per_label"]["test"]["per_label"]
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
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"\textbf{Mô hình} & \textbf{Macro P} & \textbf{Macro R} & \textbf{Macro F1} & \textbf{Micro P} & \textbf{Micro R} & \textbf{Micro F1} \\",
        r"\midrule",
    ]
    for name, run in runs.items():
        test = run["results"]["test"]
        lines.append(
            f"{latex_escape(name)} & {test['macro']['precision']:.4f} & {test['macro']['recall']:.4f} & "
            f"{test['macro']['f1']:.4f} & {test['micro']['precision']:.4f} & "
            f"{test['micro']['recall']:.4f} & {test['micro']['f1']:.4f} \\\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Kết quả test của năm hệ thống tại threshold 0.5}",
        r"\label{tab:model-results}", r"\end{table}", "",
        r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrl}", r"\toprule",
        r"\textbf{Hệ thống} & \textbf{Fixed F1} & \textbf{Tuned F1} & \textbf{$\Delta$ F1} & \textbf{Micro F1} & \textbf{Chọn} \\",
        r"\midrule",
    ])
    by_name = {item["name"]: item for item in thresholds["experiments"]}
    for display_name, key in ARCHITECTURE_KEYS.items():
        fixed = by_name[f"{key}_fixed"]
        selected = by_name[thresholds["selected_by_architecture"][key]]
        fixed_f1 = fixed["test"]["macro"]["f1"]
        tuned_f1 = selected["test"]["macro"]["f1"]
        lines.append(
            f"{latex_escape(display_name)} & {fixed_f1:.4f} & {tuned_f1:.4f} & "
            f"{tuned_f1 - fixed_f1:+.4f} & {selected['test']['micro']['f1']:.4f} & "
            f"{latex_escape(selected['threshold_strategy'].replace('_', '-'))} \\\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Fixed 0.5 và threshold được chọn bằng validation macro-F1}",
        r"\label{tab:threshold-results}", r"\end{table}", "",
    ])
    (REPORT_DIR / "generated_tables.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\begin{tabular}{p{5.2cm}p{3.0cm}p{3.0cm}p{2.2cm}}", r"\toprule",
        r"\textbf{Văn bản} & \textbf{Nhãn thật} & \textbf{Dự đoán} & \textbf{Nhóm lỗi} \\",
        r"\midrule",
    ]
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
    (REPORT_DIR / "generated_error_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    contract = load_json(CONTRACT_PATH)
    clean = clean_splits(contract)
    runs, predictions, thresholds = load_artifacts(contract)
    write_method_tables(contract, runs)
    write_pos_weight_table()
    write_pos_weight_table()
    plot_label_distribution(clean["train"], contract["schema"]["label_names"])
    plot_model_comparison(runs)
    plot_threshold_study(thresholds)
    plot_per_label_f1(thresholds)
    plot_slice_performance(clean["test"], predictions)
    selected_name = thresholds["selected_by_architecture"]["transformer_encoder"]
    selected = next(item for item in thresholds["experiments"] if item["name"] == selected_name)
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
    print(f"Created 5 figures and generated report tables in {REPORT_DIR}")
    # datasets/multiprocess 0.70 emits a harmless ResourceTracker shutdown
    # traceback on this Python 3.12 environment. All artifacts are closed here.
    import sys

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

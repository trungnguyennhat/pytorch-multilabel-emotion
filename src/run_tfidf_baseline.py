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
RESULTS_PATH = OUTPUT_DIR / "results.txt"
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
    train_texts,
    validation_texts,
    y_train,
    y_validation,
    label_names,
):
    print(f"Training candidate: {candidate['name']}")
    candidate_start = perf_counter()
    
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=candidate["ngram_range"],
        min_df=candidate["min_df"],
        max_df=1.0,
    )
    
    vectorization_start = perf_counter()
    
    """
    # tai buoc fit: bien text thanh so
    1. chuan hoa theo cac config cua Tfidf o tren va giu lai cac token hop le
    2. sau do tao duoc vocab tu cac tu do (map co dinh feature -> vi tri cot, co the hieu nhu 1 dict)
    3. tinh idf cho tung feature dang co trong vocab
    - tinh IDF cho 1 feature = feature do hiem den dau trong toan bo train? 
    -> log(1 + N_train / (1 + df(t))) + 1, 
    # df: so sample trong toan bo train chua feature do
    # N_train: toan bo so sample trong train
    -> token/ngram cang hiem thi IDF cang cao
    sau khi fit thi da biet duoc feature do nam o cot nao trong vocab va co idf = bnh
    
    # tai buoc transform:
    1. voi 1 sample, se tinh tf cho tung feature, feature trong 1 sample se phai lay dua tren vocabulary da duoc fit
    tinh TF(t, d): so lan feature t xuat hien trong cau d
    2. nhân tf với idf của từng feature tương ứng
    3. L2-normalize toan bo vector tf-idf cua sample
    4. thu duoc 1 vecto co cung so chieu voi vocabulary
    
    chu y:
    - ta cần vocab vì để khi biểu diễn sample theo vector thì phải đối chiếu với vocab xem token có trong sample ở vị trí nào trong vocab.
    -> tức là 1 sample khi biểu diễn theo vector thì chỉ những token nào có trong vocab thì mới được biểu diễn ở sample đó và vị trí trong vector được biểu diễn trùng với trong vocab
    """
    X_train = vectorizer.fit_transform(train_texts)
    # khi valid transform thì đang lấy vocab từ train
    X_validation = vectorizer.transform(validation_texts)
    
    vectorization_seconds = (
        perf_counter() - vectorization_start
    )
    
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            class_weight=None,
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        ),
        n_jobs=1,
    )
    fit_start = perf_counter()
    classifier.fit(X_train, y_train)
    fit_seconds = perf_counter() - fit_start

    train_probabilities = classifier.predict_proba(X_train)
    train_predictions = (train_probabilities >= THRESHOLD).astype(np.int8)

    train_metrics = calculate_metrics(
        y_true=y_train,
        y_pred=train_predictions,
        label_names=label_names,
    )
    
    prediction_start = perf_counter()
    
    validation_probabilities = (classifier.predict_proba(X_validation))
    validation_predictions = (perf_counter() - prediction_start)
    
    validation_predictions = (validation_probabilities >= THRESHOLD).astype(np.int8)
    
    prediction_seconds = (perf_counter() - prediction_start)
    
    validation_metrics = calculate_metrics(
        y_true=y_validation,
        y_pred=validation_predictions,
        label_names=label_names,
    )

    config_record = {
        "name": candidate["name"],
        "ngram_range": list(
            candidate["ngram_range"]
        ),
        "min_df": candidate["min_df"],
        "lowercase": True,
        "stop_words": None,
        "C": 1.0,
        "class_weight": None,
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": 42,
        "threshold": THRESHOLD,
    }

    summary = {
        "config": config_record,
        "vocabulary_size": len(
            vectorizer.vocabulary_
        ),
        "matrix_shapes": {
            "train": list(X_train.shape),
            "validation": list(
                X_validation.shape
            ),
        },
        "runtime_seconds": {
            "vectorization": float(
                vectorization_seconds
            ),
            "classifier_fit": float(fit_seconds),
            "validation_prediction": float(
                prediction_seconds
            ),
            "total": float(
                perf_counter() - candidate_start
            ),
        },
        "validation_metrics": {
            "macro": validation_metrics["macro"],
            "micro": validation_metrics["micro"],
        },
        "train_metrics": {
            "macro": train_metrics["macro"],
            "micro": train_metrics["micro"],
        },
    }

    print(
        f"Validation macro-F1: "
        f"{validation_metrics['macro']['f1']:.4f}"
    )
    print(
        f"Validation micro-F1: "
        f"{validation_metrics['micro']['f1']:.4f}"
    )

    # tra ve 1 dict
    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "validation_probabilities": (
            validation_probabilities
        ),
        "validation_predictions": (
            validation_predictions
        ),
        "validation_metrics": validation_metrics,
        "summary": summary,
    }
    
def evaluate_model(
    candidate,
    texts,
    targets,
    label_names,
):
    evaluation_start = perf_counter()

    X = candidate["vectorizer"].transform(texts)

    probabilities = candidate["classifier"].predict_proba(X)

    predictions = (probabilities >= THRESHOLD).astype(np.int8)
    
    metrics = calculate_metrics(
        y_true=targets,
        y_pred=predictions,
        label_names=label_names,
    )

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": metrics,
        "matrix_shape": list(X.shape),
        "runtime_seconds": float(
            perf_counter() - evaluation_start
        ),
    }

def write_outputs(
    contract,
    clean_dataset,
    targets,
    label_names,
    candidate_summaries,
    selected_candidate,
    test_result,
    total_runtime_seconds,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        PREDICTIONS_PATH,
        label_names=np.asarray(
            label_names,
            dtype=str,
        ),
        validation_ids=np.asarray(
            clean_dataset["validation"]["id"],
            dtype=str,
        ),
        validation_targets=targets[
            "validation"
        ].astype(np.int8),
        validation_probabilities=selected_candidate[
            "validation_probabilities"
        ].astype(np.float32),
        validation_predictions=selected_candidate[
            "validation_predictions"
        ].astype(np.int8),
        test_ids=np.asarray(
            clean_dataset["test"]["id"],
            dtype=str,
        ),
        test_targets=targets[
            "test"
        ].astype(np.int8),
        test_probabilities=test_result[
            "probabilities"
        ].astype(np.float32),
        test_predictions=test_result[
            "predictions"
        ].astype(np.int8),
    )

    run_record = {
        "system": "tfidf_logreg",
        "software": {
            "datasets": hf_datasets.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "dataset": {
            "source": contract["dataset"]["source"],
            "config": contract["dataset"]["config"],
            "official_fingerprints": {
                split_name: contract[
                    "official_splits"
                ][split_name]["fingerprint"]
                for split_name in SPLIT_NAMES
            },
            "evaluation_policy": contract[
                "evaluation_policy"
            ]["name"],
            "clean_split_sizes": {
                split_name: len(
                    clean_dataset[split_name]
                )
                for split_name in SPLIT_NAMES
            },
        },
        "label_names": label_names,
        "selection": {
            "metric": "validation_macro_f1",
            "threshold": THRESHOLD,
            "tie_breaker": (
                "first candidate in declared order"
            ),
            "candidates": candidate_summaries,
            "selected_config": selected_candidate[
                "summary"
            ]["config"],
        },
        "results": {
            "validation": selected_candidate[
                "validation_metrics"
            ],
            "test": test_result["metrics"],
        },
        "selected_model": {
            "vocabulary_size": selected_candidate[
                "summary"
            ]["vocabulary_size"],
            "matrix_shapes": {
                "train": selected_candidate[
                    "summary"
                ]["matrix_shapes"]["train"],
                "validation": selected_candidate[
                    "summary"
                ]["matrix_shapes"]["validation"],
                "test": test_result["matrix_shape"],
            },
        },
        "runtime_seconds": {
            "test_evaluation": test_result[
                "runtime_seconds"
            ],
            "full_package": float(
                total_runtime_seconds
            ),
        },
        "artifacts": {
            "predictions": (
                PREDICTIONS_PATH
                .relative_to(PROJECT_ROOT)
                .as_posix()
            ),
        },
    }

    RUN_PATH.write_text(
        json.dumps(
            run_record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result_lines = [
        "TF-IDF + One-vs-Rest Logistic Regression Results",
        "",
        "Train and validation results by candidate",
        "-----------------------------------------",
    ]

    for candidate_summary in candidate_summaries:
        candidate_config = candidate_summary["config"]
        train_metrics = candidate_summary["train_metrics"]
        candidate_metrics = candidate_summary["validation_metrics"]

        result_lines.extend(
            [
                f"Candidate: {candidate_config['name']}",
                f"  N-gram range: {tuple(candidate_config['ngram_range'])}",
                f"  Minimum document frequency: {candidate_config['min_df']}",
                f"  Vocabulary size: {candidate_summary['vocabulary_size']}",
                "  Train",
                (
                    "    Macro — "
                    f"precision: {train_metrics['macro']['precision']:.4f}, "
                    f"recall: {train_metrics['macro']['recall']:.4f}, "
                    f"F1: {train_metrics['macro']['f1']:.4f}"
                ),
                (
                    "    Micro — "
                    f"precision: {train_metrics['micro']['precision']:.4f}, "
                    f"recall: {train_metrics['micro']['recall']:.4f}, "
                    f"F1: {train_metrics['micro']['f1']:.4f}"
                ),
                "  Validation",
                (
                    "    Macro — "
                    f"precision: {candidate_metrics['macro']['precision']:.4f}, "
                    f"recall: {candidate_metrics['macro']['recall']:.4f}, "
                    f"F1: {candidate_metrics['macro']['f1']:.4f}"
                ),
                (
                    "    Micro — "
                    f"precision: {candidate_metrics['micro']['precision']:.4f}, "
                    f"recall: {candidate_metrics['micro']['recall']:.4f}, "
                    f"F1: {candidate_metrics['micro']['f1']:.4f}"
                ),
                "",
            ]
        )

    selected_config = selected_candidate["summary"]["config"]
    test_metrics = test_result["metrics"]

    result_lines.extend(
        [
            "Final test results",
            "------------------",
            (
                "Candidate selected by validation: "
                f"{selected_config['name']}"
            ),
            "Test",
            (
                "  Macro — "
                f"precision: {test_metrics['macro']['precision']:.4f}, "
                f"recall: {test_metrics['macro']['recall']:.4f}, "
                f"F1: {test_metrics['macro']['f1']:.4f}"
            ),
            (
                "  Micro — "
                f"precision: {test_metrics['micro']['precision']:.4f}, "
                f"recall: {test_metrics['micro']['recall']:.4f}, "
                f"F1: {test_metrics['micro']['f1']:.4f}"
            ),
        ]
    )

    RESULTS_PATH.write_text(
        "\n".join(result_lines) + "\n",
        encoding="utf-8",
    )
    
def main():
    package_start = perf_counter()

    contract, clean_dataset = load_clean_dataset()
    label_names = contract["schema"]["label_names"]

    targets = encode_targets(
        clean_dataset=clean_dataset,
        label_names=label_names,
    )

    train_texts = list(
        clean_dataset["train"]["text"]
    )
    validation_texts = list(
        clean_dataset["validation"]["text"]
    )
    test_texts = list(
        clean_dataset["test"]["text"]
    )

    candidate_summaries = []
    selected_candidate = None
    selected_macro_f1 = None

    for candidate in CANDIDATES:
        fitted_candidate = fit_candidate(
            candidate=candidate,
            train_texts=train_texts,
            validation_texts=validation_texts,
            y_train=targets["train"],
            y_validation=targets["validation"],
            label_names=label_names,
        )

        candidate_summaries.append(
            fitted_candidate["summary"]
        )

        candidate_macro_f1 = fitted_candidate[
            "validation_metrics"
        ]["macro"]["f1"]

        if (
            selected_candidate is None
            or candidate_macro_f1
            > selected_macro_f1
        ):
            selected_candidate = fitted_candidate
            selected_macro_f1 = candidate_macro_f1

    selected_name = selected_candidate[
        "summary"
    ]["config"]["name"]

    print(
        f"Selected candidate: {selected_name} "
        f"(validation macro-F1="
        f"{selected_macro_f1:.4f})"
    )
    print("Evaluating validation-selected candidate on clean test")

    test_result = evaluate_model(
        candidate=selected_candidate,
        texts=test_texts,
        targets=targets["test"],
        label_names=label_names,
    )

    total_runtime_seconds = (
        perf_counter() - package_start
    )

    write_outputs(
        contract=contract,
        clean_dataset=clean_dataset,
        targets=targets,
        label_names=label_names,
        candidate_summaries=candidate_summaries,
        selected_candidate=selected_candidate,
        test_result=test_result,
        total_runtime_seconds=total_runtime_seconds,
    )

    print(
        f"Test macro-F1: "
        f"{test_result['metrics']['macro']['f1']:.4f}"
    )
    print(
        f"Test micro-F1: "
        f"{test_result['metrics']['micro']['f1']:.4f}"
    )
    print(f"Run record: {RUN_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print("TF-IDF BASELINE COMPLETED")


if __name__ == "__main__":
    main()

    
    

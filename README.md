# Comparative Study of PyTorch Neural Architectures for Multi-label Emotion Classification

Project nghiên cứu bài toán **multi-label emotion classification** trên GoEmotions. Trọng tâm là tự xây dựng và huấn luyện các neural architecture bằng PyTorch, sau đó thực hiện controlled experiments, metric analysis và error analysis để tạo báo cáo đồ án có thể tái lập.

```text
raw text and labels
→ data audit and EDA
→ preprocessing and vocabulary
→ Dataset/DataLoader
→ neural architectures and custom training loop
→ threshold and imbalance experiments
→ controlled comparison and error analysis
→ final report
```

## Current Status

> Agent chỉ được chỉnh sửa nội dung nằm giữa hai marker dưới đây nếu người dùng không yêu cầu sửa file khác.

<!-- CURRENT_STATUS_START -->

- **Project state:** In progress
- **Current stage:** Stage 1 — GoEmotions dataset audit and problem formulation
- **Current work package:** Work Package 1 — Load, audit and validate GoEmotions; freeze splits and label mapping
- **Reusable work from previous project:** Người học đã hoàn thành project MT vs LLM, có kinh nghiệm thiết kế evaluation set, chạy experiment có schema thống nhất, so sánh model và viết final technical report; không tái sử dụng source code trực tiếp
- **Next action:** Brainstorm và chốt thiết kế package audit GoEmotions trước khi viết code: dataset configuration, schema, official splits, data-quality checks, label order, multi-hot example và artifact outputs
- **Evidence required to complete current package:** Dataset config/schema và split sizes từ loader thực tế; missing/empty/duplicate statistics; label names/order; một multi-hot target đúng shape và `float32`; dataset summary cùng label mapping artifacts được lưu
- **Blockers:** None
- **Last updated:** 2026-07-30

### Stage progress

- [x] Stage 0 — Environment, repository and experiment contract
- [ ] Stage 1 — GoEmotions dataset audit and problem formulation
- [ ] Stage 2 — Exploratory data analysis and preprocessing design
- [ ] Stage 3 — TF-IDF and Logistic Regression baseline
- [ ] Stage 4 — PyTorch data pipeline
- [ ] Stage 5 — Mean Pooling MLP neural baseline
- [ ] Stage 6 — BiLSTM with Attention
- [ ] Stage 7 — Transformer Encoder
- [ ] Stage 8 — Imbalance, threshold and ablation experiments
- [ ] Stage 9 — Reproducibility, efficiency and controlled comparison
- [ ] Stage 10 — Error analysis and final report

<!-- CURRENT_STATUS_END -->

## 1. Project Goals

Xây dựng một experimental pipeline để:

- Hiểu sâu PyTorch: tensor, `Dataset`, `DataLoader`, `nn.Module`, autograd, custom training loop và checkpoint.
- So sánh classical baseline với ba neural architecture train từ đầu.
- Đánh giá ảnh hưởng của architecture, class imbalance và threshold strategy.
- So sánh quality với parameter count, runtime và GPU memory.
- Tạo error analysis và báo cáo có thể truy vết đến từng experiment.

Project không chỉ tối ưu điểm số; mục tiêu chính là hiểu và chứng minh được toàn bộ vòng đời của một NLP experiment.

## 2. Research Questions

1. Neural models có cải thiện so với TF-IDF + Logistic Regression không?
2. Mean Pooling MLP, BiLSTM + Attention và Transformer Encoder khác nhau thế nào về macro-F1 và micro-F1?
3. Sequence models có lợi thế trên câu dài, negation hoặc sample có nhiều emotion labels không?
4. `pos_weight` ảnh hưởng thế nào đến precision, recall và F1 của label hiếm?
5. Global hoặc per-label threshold có tốt hơn threshold `0.5` không?
6. Model phức tạp hơn có đáng đổi lấy chi phí train, số parameter và GPU memory lớn hơn không?
7. Các failure mode chính có liên quan đến sarcasm, negation, ambiguity hay label correlation không?

Các câu hỏi này là hypothesis-driven nhưng kết quả có thể bác bỏ dự đoán ban đầu.

## 3. Task and Dataset

### Dataset

- GoEmotions, ưu tiên official simplified configuration nếu loader hiện tại hỗ trợ.
- Dùng official train/validation/test split.
- Stage đầu phải audit schema thực tế thay vì dựa vào tutorial.

### Input and output

Input là một câu tiếng Anh. Output là một tập emotion labels; một sample có thể có nhiều positive labels.

Đây là bài toán **multi-label**, không phải multi-class:

- Mỗi label là một binary target độc lập.
- Target là multi-hot `float32` với shape `[num_labels]`.
- Model trả logits với shape `[batch_size, num_labels]`.
- Sigmoid chuyển logits thành probabilities.
- Threshold chuyển probabilities thành binary predictions.

Loss mặc định:

```python
torch.nn.BCEWithLogitsLoss()
```

Không đặt sigmoid trong `forward()` khi train bằng `BCEWithLogitsLoss`.

## 4. Systems Compared

| System | Pipeline | Vai trò |
|---|---|---|
| `tfidf_logreg` | TF-IDF → One-vs-Rest Logistic Regression | Classical baseline |
| `mean_pooling_mlp` | Embedding → masked mean pooling → MLP | Neural baseline đơn giản |
| `bilstm_attention` | Embedding → BiLSTM → attention pooling | Sequence model |
| `transformer_encoder` | Embedding + positional encoding → Transformer Encoder → masked pooling | Self-attention model |
| `distilbert_finetuned` | Pretrained DistilBERT → classifier | Optional extension |

DistilBERT chỉ được thực hiện sau khi bốn system chính và report sơ bộ đã hoàn thành.

## 5. Experimental Contract

### Data and leakage

- Không gộp validation hoặc test vào train.
- Vocabulary, TF-IDF và `pos_weight` chỉ được fit/tính từ train.
- Hyperparameter, checkpoint và threshold chỉ được chọn bằng validation.
- Test chỉ được dùng cho final evaluation sau khi config đã chốt.
- Kiểm tra missing text, invalid labels và duplicate text trong/giữa splits.

### Label representation

- Giữ một label order cố định cho toàn project.
- Lưu `label_to_id.json` và `id_to_label.json`.
- Mọi target, logits, prediction, metric và threshold artifact phải dùng cùng label order.

### Model selection and thresholds

- Primary selection metric: validation macro-F1.
- Luôn báo cáo micro-F1 song song.
- So sánh threshold cố định `0.5`, tuned global threshold và tuned per-label thresholds.
- Threshold chỉ được tune trên validation probabilities và đóng băng trước test evaluation.

### Seeds and comparison

Stability seeds mặc định:

```text
13, 42, 2026
```

Chỉ chạy nhiều seeds sau khi config chính đã ổn định. Nếu kết luận architecture nào tốt hơn, final comparison phải có nhiều seeds cho từng neural architecture được so sánh; single-run result chỉ được xem là exploratory.

Khi so sánh architectures, giữ cố định khi có thể:

- Data splits, label mapping và vocabulary/tokenizer.
- Maximum sequence length.
- Loss và threshold strategy của architecture experiment chính.
- Evaluation code, hardware và early-stopping policy.

Batch size có thể khác do memory nhưng phải được báo cáo. Kết quả là practical system comparison, không phải causal architecture ablation nếu model capacity hoặc optimization khác nhau.

### Reproducibility metadata

Mỗi run cần lưu tối thiểu:

```text
run_id, model_name, seed, config
dataset name/config/fingerprint
git commit
start/end time, best epoch, best validation metric
checkpoint path, parameter count, runtime, peak GPU memory
```

Không chọn seed tốt nhất làm kết quả chính và không đổi config giữa một multi-seed experiment.

## 6. Evaluation

### Overall metrics

- Micro precision, recall và F1.
- Macro precision, recall và F1.
- Weighted F1 và samples-F1.
- Hamming loss và exact-match ratio.
- Micro PR-AUC và macro PR-AUC khi hợp lệ.

### Per-label metrics

- Support, precision, recall, F1 và selected threshold.
- PR-AUC khi label có đủ positive và negative samples.

### Slice analysis

- Label frequency.
- Text length.
- Label cardinality.
- Representative error categories.

### Efficiency

- Trainable parameter count.
- Time per epoch và total training time.
- Peak GPU memory.
- Inference latency hoặc throughput trên cùng setup.

## 7. Implementation Roadmap

| Stage | Mục tiêu | Artifact chính |
|---|---|---|
| 0 | Environment và reproducibility setup | Environment summary, dependency file, seed utility |
| 1 | Audit GoEmotions và đóng băng experiment contract | Dataset summary, split table, label mappings |
| 2 | EDA và preprocessing decisions | Figures, label/text statistics, preprocessing note |
| 3 | TF-IDF + Logistic Regression | Baseline model, predictions và metrics |
| 4 | Vocabulary, `Dataset`, `collate_fn`, `DataLoader` | Vocabulary artifact và inspected batch |
| 5 | Mean Pooling MLP và custom training loop | Checkpoint, curves, validation metrics |
| 6 | BiLSTM + Attention | Checkpoint, attention validation, comparison |
| 7 | Transformer Encoder | Checkpoint, mask validation, comparison |
| 8 | `pos_weight`, threshold và controlled ablations | Loss/threshold tables và artifacts |
| 9 | Multiple seeds và efficiency benchmark | Mean ± std, runtime/GPU comparison |
| 10 | Error analysis và final report | Error taxonomy, figures, final report |

Mỗi stage chỉ tạo file khi thực sự cần. Không scaffold toàn bộ source tree từ đầu.

## 8. Project Structure

```text
pytorch-multilabel-emotion/
├── AGENTS.md
├── README.md
├── requirements.txt
├── check_env.py
├── configs/
├── data/
│   └── artifacts/
├── notebooks/
├── src/
└── outputs/
    ├── checkpoints/
    ├── metrics/
    ├── figures/
    └── reports/
```

- Notebook dùng cho exploration, visualization và narrative analysis.
- `src/` dùng cho logic cần chạy lại.
- Không commit raw data hoặc checkpoint lớn.
- Metrics, config và report phải truy ngược được đến run.

## 9. Environment

Stack chính:

- Python 3.12.
- PyTorch.
- pandas, NumPy và scikit-learn.
- Hugging Face `datasets`.
- Matplotlib, tqdm và Jupyter.

Setup trên Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_env.py
```

Environment hiện tại dùng PyTorch `2.13.0+cu132` với RTX 5060 Ti 16GB. Khi tái tạo trên máy khác, CUDA wheel phải phù hợp với driver/GPU của máy đó.

Không bắt buộc dùng PyTorch Lightning, Hugging Face `Trainer`, W&B, Hydra hoặc Optuna trong minimum project.

## 10. Definition of Done

Project chỉ hoàn thành khi có bằng chứng cho tất cả mục sau:

- Dataset audit, EDA report và label mappings cố định.
- TF-IDF + Logistic Regression baseline.
- Custom PyTorch `Dataset`, dynamic-padding `collate_fn` và `DataLoader`.
- Mean Pooling MLP, BiLSTM + Attention và Transformer Encoder được train bằng custom loop.
- Standard BCE và `pos_weight` experiment.
- Fixed/global/per-label threshold comparison.
- Overall, per-label, learning-curve và efficiency results.
- Multiple-seed results cho final neural comparison.
- Slice analysis và representative error taxonomy.
- Final report trả lời research questions, methodology, results, limitations và conclusion.
- `Current Status` ghi `Project completed`.

Quy tắc làm việc với agent, cách review code và cách cập nhật trạng thái nằm trong [`AGENTS.md`](AGENTS.md).

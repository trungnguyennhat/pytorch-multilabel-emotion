# Comparative Study of PyTorch Neural Architectures for Multi-label Emotion Classification

Project nghiên cứu bài toán **multi-label emotion classification** trên GoEmotions. Trọng tâm là so sánh có kiểm soát một classical baseline với ba neural architecture train từ đầu bằng PyTorch, sau đó phân tích chất lượng, chi phí và failure modes.

Nguyên tắc xuyên suốt là **research-first, minimum implementation**: chỉ viết code cần để bảo vệ tính hợp lệ của experiment hoặc trả lời trực tiếp một research question. Không biến mỗi phép kiểm tra, metric hoặc artifact thành một helper/file riêng.

```text
GoEmotions contract và data analysis
→ TF-IDF baseline
→ shared PyTorch data/training/evaluation pipeline
→ MLP, BiLSTM, Transformer experiments
→ imbalance và threshold experiments
→ controlled comparison, error analysis và final report
```

## Current Status

> Agent chỉ được chỉnh sửa nội dung nằm giữa hai marker dưới đây nếu người dùng không yêu cầu sửa file khác.

<!-- CURRENT_STATUS_START -->

- **Project state:** In progress
- **Current stage:** Stage 5 — Imbalance and threshold experiments
- **Completed with current evidence:** Stage 0–1 environment/data contract đã pass với 28 ordered labels và clean views 43,410 / 5,383 / 5,385. Stage 2 TF-IDF đạt clean test macro/micro-F1 0.2315/0.4171. Stage 3 Mean Pooling MLP đạt 0.3541/0.4788. Stage 4 đã hoàn thành cả hai model: BiLSTM + Attention đạt 0.3880/0.4948; Transformer Encoder selected ở epoch 27 đạt validation macro/micro-F1 0.4508/0.5349 và clean test 0.4366/0.5336. Transformer checkpoint/run/vocabulary/28-label contract nhất quán; validation/test targets, probabilities và predictions đúng shape `(5383, 28)` / `(5385, 28)` và range hợp lệ.
- **Current work package:** Thiết kế Stage 5 controlled experiments cho train-only `pos_weight` và fixed/global/per-label thresholds, dùng Transformer Encoder làm neural candidate mạnh nhất hiện tại
- **Reusable work from previous project:** Người học đã có kinh nghiệm thiết kế evaluation set, chạy experiment theo schema thống nhất, so sánh model và viết technical report; không tái sử dụng source code trực tiếp
- **Next action:** Chốt thiết kế Stage 5 trước implementation: baseline không weight ở threshold 0.5, Transformer train lại với `pos_weight` chỉ tính từ train labels, và threshold global/per-label chỉ tune từ validation probabilities
- **Evidence required to complete current package:** Bảng controlled experiments tách ảnh hưởng của loss weighting và threshold; mọi threshold được chọn bằng validation; test chỉ evaluate sau khi strategy đóng băng; probabilities được tái sử dụng thay vì forward lại cho từng threshold
- **Current evidence gap:** Chưa xác định công thức/clipping policy cho `pos_weight`, search space và tie-breaker cho global/per-label threshold, hay model runs tối thiểu cần train lại
- **Blockers:** None
- **Last updated:** 2026-08-21

### Stage progress

- [x] Stage 0 — Environment and repository setup
- [x] Stage 1 — Data contract and analysis
- [x] Stage 2 — TF-IDF + Logistic Regression baseline
- [x] Stage 3 — Shared PyTorch pipeline + Mean Pooling MLP
- [x] Stage 4 — BiLSTM + Attention and Transformer Encoder
- [ ] Stage 5 — Imbalance and threshold experiments
- [ ] Stage 6 — Final controlled comparison, error analysis and report

<!-- CURRENT_STATUS_END -->

## 1. Research Goals

Project phải trả lời các câu hỏi sau:

1. Neural models có cải thiện so với TF-IDF + Logistic Regression không?
2. Mean Pooling MLP, BiLSTM + Attention và Transformer Encoder khác nhau thế nào về macro-F1 và micro-F1?
3. Sequence models có lợi thế trên câu dài, negation hoặc sample có nhiều emotion labels không?
4. `pos_weight` ảnh hưởng thế nào đến precision, recall và F1 của label hiếm?
5. Global hoặc per-label threshold có tốt hơn threshold `0.5` không?
6. Model phức tạp hơn có đáng đổi lấy parameter count, training time và GPU memory lớn hơn không?
7. Những failure mode chính có liên quan đến sarcasm, negation, ambiguity hoặc label correlation không?

Kết quả có thể bác bỏ giả thuyết ban đầu. Không thêm model, metric hoặc ablation nếu chúng không giúp trả lời một trong các câu hỏi trên.

## 2. Task and Systems

### Problem contract

- Input: một câu tiếng Anh.
- Output: một tập emotion labels; một sample có thể có nhiều positive labels.
- Đây là bài toán **multi-label**, không phải multi-class.
- Target có dtype `float32` và shape `[num_labels]`.
- Model trả logits `[batch_size, num_labels]`.
- `sigmoid(logits)` tạo probabilities; threshold tạo binary predictions.
- Loss neural mặc định là `torch.nn.BCEWithLogitsLoss()`; không đặt sigmoid trong `forward()`.

### Systems bắt buộc

| System | Pipeline | Vai trò nghiên cứu |
|---|---|---|
| `tfidf_logreg` | TF-IDF → One-vs-Rest Logistic Regression | Classical baseline |
| `mean_pooling_mlp` | Embedding → masked mean pooling → MLP | Neural baseline đơn giản |
| `bilstm_attention` | Embedding → BiLSTM → attention pooling | Sequence model |
| `transformer_encoder` | Embedding + positional encoding → Transformer Encoder → masked pooling | Self-attention model |

Pretrained models và các architecture khác nằm ngoài minimum scope.

## 3. Experimental Contract

### Data contract

- Dùng GoEmotions official simplified configuration và official train/validation/test splits.
- Audit schema từ loader thực tế; không hard-code row count hoặc label count từ tutorial làm điều kiện pass.
- Ghi dataset fingerprint, split sizes, columns và ordered label names vào một artifact duy nhất: `data/artifacts/dataset_contract.json`.
- `label_names[index]` là mapping chuẩn từ ID sang tên; mọi target, logit, prediction, threshold và metric dùng cùng thứ tự này. Không lưu hai mapping JSON ngược nhau.
- Không sửa, loại hoặc di chuyển sample trong Stage 1. Nếu cần thay đổi dữ liệu sau audit, phải ghi thành một quyết định nghiên cứu riêng.

Audit tối thiểu chỉ kiểm tra các rủi ro có thể làm experiment sai:

- **Dừng:** thiếu official split/column, schema labels không nhất quán, missing/empty text hoặc labels, label ID không hợp lệ, label ID lặp trong một row, exact duplicate text giữa các splits.
- **Chỉ báo cáo:** exact duplicate text trong cùng split và các thống kê phân phối phục vụ EDA.

Không tạo multi-hot example ở Stage 1. Shape, dtype và label mapping của multi-hot target được kiểm tra một lần trên inspected batch khi xây PyTorch `Dataset`/`DataLoader` ở Stage 3.

### Leakage and selection

- Không gộp validation hoặc test vào train.
- Vocabulary, TF-IDF và `pos_weight` chỉ được fit/tính từ train.
- Hyperparameter, checkpoint và threshold chỉ được chọn bằng validation.
- Test chỉ được dùng sau khi config và threshold đã đóng băng.
- Primary selection metric là validation macro-F1; luôn báo cáo micro-F1 song song.
- Không chọn seed tốt nhất làm kết quả chính.

### Controlled comparison

Giữ cố định giữa các neural architectures khi hợp lý:

- Data splits, label order, vocabulary/tokenization và maximum sequence length.
- Evaluation code, loss/threshold của comparison chính, hardware và early-stopping policy.
- Batch size có thể khác vì memory nhưng phải được báo cáo.

Kết quả là practical system comparison, không được diễn giải như causal architecture ablation nếu model capacity hoặc optimization khác nhau.

Chỉ chạy các seed `13, 42, 2026` sau khi mỗi architecture đã có một config ổn định. Exploratory runs không cần chạy nhiều seed.

## 4. Minimum Evaluation

Chỉ tính các kết quả cần cho research questions:

- **Model selection:** validation macro-F1.
- **Overall quality:** macro precision/recall/F1 và micro precision/recall/F1.
- **Rare-label analysis:** per-label support, precision, recall và F1.
- **Threshold study:** so sánh fixed `0.5`, tuned global và tuned per-label thresholds; chỉ tune trên validation.
- **Research slices:** text length, label cardinality và các nhóm negation/ambiguity có đủ evidence.
- **Efficiency:** trainable parameter count, total training time và peak GPU memory trên cùng setup.
- **Final stability:** mean và standard deviation qua ba seeds cho các neural architectures.

Không mặc định tính weighted-F1, samples-F1, Hamming loss, exact match, PR-AUC hoặc inference benchmark. Chỉ bổ sung khi final analysis cho thấy một metric đó cần để giải thích kết quả.

## 5. Minimum Implementation Rules

### Một trách nhiệm chỉ có một nguồn sự thật

- Dataset audit chỉ chạy đầy đủ ở Stage 1. Các stage sau đọc contract đã đóng băng và chỉ kiểm tra nhanh boundary mà chúng trực tiếp phụ thuộc.
- Multi-hot shape/dtype/mapping chỉ smoke-check ở Stage 3, không tạo example artifact riêng và không lặp lại ở từng model.
- Tất cả neural models dùng chung data pipeline, training loop, evaluation code và checkpoint format.
- Tất cả threshold experiments dùng cùng validation probabilities đã lưu; không forward model lại cho từng threshold.
- Final tables và figures được tạo từ run outputs, không nhập tay lại kết quả.

### Chỉ tách code khi có boundary thật

- Một work-package script mặc định nằm trong một file.
- `main()` điều phối; function riêng chỉ dành cho một pha có input/output hoặc invariant độc lập.
- Không tạo helper để bọc một thao tác, rút ngắn code block hoặc kiểm tra lại output vừa được tạo.
- Không tạo class nếu function và plain dictionary đã đủ.
- Không thêm registry, factory, dataclass, config framework, logging framework, caching, CLI nhiều tầng hoặc report generator trong minimum scope.
- Một file mới chỉ hợp lý khi logic được dùng lại giữa nhiều experiments, là implementation độc lập của một model, hoặc là artifact bắt buộc.

Với Stage 1, một script gồm các pha `load contract → audit rows/splits → write artifact → main` là đủ. Có thể gộp các pha nếu data flow vẫn rõ; không cần function riêng cho từng counter, mapping, assertion hoặc example.

### Artifact policy

- Lưu artifact cần để tái lập kết quả hoặc thực hiện stage sau; không lưu output chỉ để chứng minh một dòng code hoạt động.
- Debug và exploratory runs có thể bị ghi đè. Chỉ giữ checkpoint/output của candidate được chọn và final multi-seed runs.
- Mỗi final run cần một record duy nhất chứa: model, seed, config, dataset fingerprint, best epoch, validation metric, parameter count, runtime, peak GPU memory và checkpoint/prediction paths.
- Checkpoint neural phải đủ để load model với vocabulary và ordered label names tương ứng.

## 6. Implementation Roadmap

| Stage | Câu hỏi cần giải quyết | Minimum output |
|---|---|---|
| 0 | Environment có chạy đúng không? | Dependency file và một environment smoke check |
| 1 | Dữ liệu thực tế có đủ tin cậy để nghiên cứu không? | Một audit script, `dataset_contract.json` và data-analysis notebook/note |
| 2 | Classical baseline đạt mức nào? | Validation/test predictions và metrics của TF-IDF + Logistic Regression |
| 3 | Shared PyTorch pipeline có đúng và neural baseline học được không? | Inspected batch, shared training loop, MLP checkpoint và metrics |
| 4 | Sequence và self-attention models khác MLP thế nào? | BiLSTM/Transformer checkpoints, predictions và metrics |
| 5 | Imbalance handling và threshold strategy thay đổi kết quả thế nào? | Models/predictions cần thiết và một bảng controlled experiments; threshold được chọn từ validation predictions |
| 6 | Kết luận cuối có ổn định và trả lời research questions không? | Multi-seed comparison, efficiency/error analysis và final report |

Stage là research milestone, không phải yêu cầu tạo package hoặc folder riêng. Không scaffold toàn bộ source tree từ đầu.

## 7. Suggested Minimal Structure

Chỉ tạo path khi stage tương ứng cần đến:

```text
pytorch-multilabel-emotion/
├── AGENTS.md
├── README.md
├── requirements.txt
├── check_env.py
├── data/
│   ├── audit_goemotions.py
│   └── artifacts/
│       └── dataset_contract.json
├── notebooks/                 # data analysis hoặc narrative exploration
├── src/                       # shared code sau khi Stage 2/3 thực sự cần
└── outputs/                   # selected/final run artifacts và final report
```

Không bắt buộc tạo sẵn `configs/`, nhiều output subfolders hoặc một file cho mỗi metric/model. Cấu trúc được mở rộng khi có artifact thật, không mở rộng để dự phòng.

## 8. Environment

Stack chính: Python 3.12, PyTorch, Hugging Face `datasets`, pandas, NumPy, scikit-learn, Matplotlib, tqdm và Jupyter.

Setup trên Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_env.py
```

Environment hiện tại dùng PyTorch `2.13.0+cu132` với RTX 5060 Ti 16GB. CUDA wheel phải phù hợp với driver/GPU của máy chạy.

Neural models dùng custom PyTorch training/validation loop. Không dùng Hugging Face `Trainer`, PyTorch Lightning, W&B, Hydra, Optuna hoặc AutoML trong minimum project.

## 9. Definition of Done

Project hoàn thành khi có evidence cho các kết quả sau, không phụ thuộc vào số file hoặc số dòng code:

- Dataset contract và data-analysis decisions đã đóng băng.
- TF-IDF + Logistic Regression baseline đã được đánh giá.
- Shared PyTorch data/training/evaluation pipeline chạy đúng.
- Mean Pooling MLP, BiLSTM + Attention và Transformer Encoder đã được train và so sánh.
- Standard BCE so với `pos_weight`, cùng fixed/global/per-label threshold, đã được kiểm tra có kiểm soát.
- Final neural comparison có ba seeds, quality metrics và efficiency measurements.
- Error analysis và final report trả lời các research questions, nêu methodology, results, limitations và conclusion đúng mức evidence.
- `Current Status` ghi `Project completed`.

Quy tắc làm việc với agent, cách review code và cách cập nhật trạng thái nằm trong [`AGENTS.md`](AGENTS.md).

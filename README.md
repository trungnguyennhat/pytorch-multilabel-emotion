# Multi-label Emotion Classification on GoEmotions

An end-to-end comparison of classical, train-from-scratch neural, and pretrained approaches for 28-label emotion classification. The project covers data auditing, leakage-safe preprocessing, custom PyTorch training loops, threshold optimization, imbalance handling, and error analysis.

[Read the full Project II report (PDF)](outputs/reports/baocaoprj2/main.pdf)

## Key results

All thresholds are selected using validation macro-F1 and then frozen for test evaluation.

| System | Test macro-F1 @ 0.5 | Tuned macro-F1 | Tuned micro-F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.2315 | 0.4301 | 0.5346 |
| Mean Pooling MLP | 0.3541 | 0.3983 | 0.4895 |
| BiLSTM + Attention | 0.3880 | 0.4095 | 0.4953 |
| Transformer Encoder | 0.4366 | 0.4679 | 0.5374 |
| BERT-base-cased (pretrained reference) | **0.4703** | **0.4976** | **0.5826** |

Main findings:

- Threshold choice materially changes system ranking. TF-IDF rises from 0.2315 to 0.4301 macro-F1 and outperforms the MLP and BiLSTM after tuning.
- The train-from-scratch Transformer is the strongest custom neural model, reaching 0.4679 macro-F1.
- BERT remains the best tuned system at 0.4976 macro-F1, but it is an external pretrained reference rather than a data-equivalent comparison.
- `pos_weight` increases Transformer recall but introduces too many false positives: tuned macro-F1 is 0.4360 versus 0.4679 with standard BCE.

## Experimental design

### Data

The project uses the `simplified` configuration of [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions), containing 27 emotion labels plus `neutral`.

| Split | Official | Removed cross-split duplicates | Clean |
|---|---:|---:|---:|
| Train | 43,410 | 0 | 43,410 |
| Validation | 5,426 | 43 | 5,383 |
| Test | 5,427 | 42 | 5,385 |

Exact-text duplicates are removed only across evaluation splits. Vocabulary, IDF, `pos_weight`, model selection, and threshold tuning never use test data.

### Systems

- **TF-IDF + Logistic Regression:** unigram sparse features with 28 one-vs-rest classifiers.
- **Mean Pooling MLP:** learned token embeddings, masked mean pooling, and an MLP head.
- **BiLSTM + Attention:** bidirectional sequence encoding with learned attention pooling.
- **Transformer Encoder:** two train-from-scratch encoder layers with four attention heads.
- **BERT-base-cased:** pretrained WordPiece encoder fine-tuned as the reference baseline from the original GoEmotions work.

Neural models use `BCEWithLogitsLoss`. Evaluation reports macro/micro precision, recall, and F1. Threshold experiments compare fixed 0.5, one validation-tuned global threshold, and 28 validation-tuned per-label thresholds.

## Repository structure

```text
data/
  audit_goemotions.py                 # dataset audit and clean-split contract
  artifacts/dataset_contract.json
src/
  run_tfidf_baseline.py
  run_mean_pooling_mlp.py
  run_bilstm_attention.py
  run_transformer_encoder.py
  run_bert_pretrained.py
  run_transformer_pos_weight.py
  run_architecture_thresholds.py      # common threshold evaluation
outputs/
  <system>/run.json                   # configuration and metrics
  <system>/predictions.npz            # targets, probabilities, predictions
  architecture_thresholds/            # fixed/global/per-label comparison
  reports/baocaoprj2/main.pdf         # final report
```

## Setup

The project was developed with Python 3.12, PyTorch, Hugging Face Datasets/Transformers, and scikit-learn.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_env.py
```

The recorded experiments used CUDA, but the scripts automatically fall back to CPU where supported. Fine-tuning BERT requires substantially more memory and time than the train-from-scratch baselines.

## Reproduce the experiments

Run commands from the repository root:

```powershell
python data\audit_goemotions.py
python src\run_tfidf_baseline.py
python src\run_mean_pooling_mlp.py
python src\run_bilstm_attention.py
python src\run_transformer_encoder.py
python src\run_bert_pretrained.py
python src\run_architecture_thresholds.py
```

Optional imbalance experiment:

```powershell
python src\run_transformer_pos_weight.py
python src\run_threshold_experiments.py
```

To regenerate report assets and compile the LaTeX report:

```powershell
venv\Scripts\python.exe outputs\reports\baocaoprj2\generate_report_assets.py
cd outputs\reports\baocaoprj2
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The 413 MB BERT checkpoint is intentionally excluded from Git because it exceeds GitHub's file-size limit. The tracked `run.json`, predictions, metrics, and report contain the evidence needed to inspect the recorded result.

## Limitations

- Neural results use one fixed seed (`42`); multi-seed variance has not been measured.
- Per-label thresholds have 28 degrees of freedom and may overfit rare validation labels.
- BERT benefits from external pretraining and is not a strictly controlled architecture-only comparison.
- Very rare labels such as `grief` and `relief` have unstable per-label test metrics.

## Tech stack

Python · PyTorch · Hugging Face Datasets/Transformers · scikit-learn · NumPy · Matplotlib · LaTeX

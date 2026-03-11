---
status: ACTIVE
type: task-graph
project: Context-Aware Hearing Aid Settings Optimizer
version: 2.0
created: 2026-03-08T16:40:00+05:30
updated: 2026-03-11T10:22:00+05:30
scope: Phases 1–10 (v2 — 8-Class Optimized)
derives_from:
  - .gsd/prd.md (v2.0)
  - .gsd/trd.md
---

# Master TODO — v2 (8-Class Optimized Architecture)

> **Status Legend**: ✅ Complete | 🔶 Partial | ⬜ Not Started | 🚫 Blocked

---

## Phase Summary

| Phase | Name | Status | Tasks |
|-------|------|--------|-------|
| Phase 1 | Environment Setup | ✅ | 6/6 |
| Phase 2 | Dataset Preparation | ✅ | 7/7 |
| Phase 3 | Label Remapping (v2) | ✅ | 5/5 |
| Phase 4 | Feature Extraction (8-class) | ✅ | 7/7 |
| Phase 5 | Model Training (8-class) | ✅ | 10/10 |
| Phase 6 | Ensemble Architecture | ✅ | 6/6 |
| Phase 7 | Evaluation & Reporting | ✅ | 8/8 |
| Phase 8 | Recommendation Engine (v2) | ✅ | 5/5 |
| Phase 9 | Deployment & Demo | ✅ | 6/6 |
| Phase 10 | IEEE Paper Preparation | ✅ | 9/9 |

---

## Phase 1 — Environment Setup ✅ COMPLETE

| ID | Task | Status |
|----|------|:------:|
| T1.1 | Create project directory structure (`src/`, `data/`, `models/`, `results/`, `demo/`) | ✅ |
| T1.2 | Initialize conda environment `hearing-aid` with Python 3.10 | ✅ |
| T1.3 | Write pinned `requirements.txt` with exact package versions | ✅ |
| T1.4 | Implement `src/utils.py` — global seed (`set_seed(42)`), logging (`setup_logging()`) | ✅ |
| T1.5 | Configure `config/` YAML for paths, hyperparameters, class definitions | ✅ |
| T1.6 | Initialize Git repository with `.gitignore` | ✅ |

---

## Phase 2 — Dataset Preparation ✅ COMPLETE

| ID | Task | Status |
|----|------|:------:|
| T2.1 | Download and verify UrbanSound8K (8,732 clips, 10 classes) | ✅ |
| T2.2 | Download and verify ESC-50 (2,000 clips, 50 classes) | ✅ |
| T2.3 | Implement `src/data_loader.py` — `UrbanSoundDataset` class with file validation | ✅ |
| T2.4 | Implement `src/data/esc50_loader.py` — `ESC50Dataset` class with full 50-class mapping | ✅ |
| T2.5 | Create stratified train/val/test splits (`data/splits/*.csv`) using 10-fold structure | ✅ |
| T2.6 | Validate class balance across all splits (deviation < ±10%) | ✅ |
| T2.7 | Implement `WeightedRandomSampler` for class-imbalanced batches | ✅ |

---

## Phase 3 — Label Remapping (v2 — 8-Class Schema) ✅ COMPLETE

> **Goal**: Remap all US8K and ESC-50 labels to the new 8 acoustically distinct classes.

| ID | Task | Status |
|----|------|:------:|
| T3.1 | Create `src/data/label_map_v2.py` — define `US8K_TO_8CLASS` and `ESC50_TO_8CLASS` dicts | ✅ |
| T3.2 | Add `CLASS_NAMES_V2 = ['speech','siren','horn','traffic','construction','dog_bark','music','background_noise']` constant | ✅ |
| T3.3 | Apply remapping in `UrbanSoundDataset` via new `remap=True` parameter; regenerate split CSVs | ✅ |
| T3.4 | Apply remapping in `ESC50Dataset` — update `ESC50_TO_US8K_MAP` → `ESC50_TO_8CLASS_MAP` | ✅ |
| T3.5 | Validate remapped class distribution — ensure all 8 classes are present in train/val/test | ✅ |

---

## Phase 4 — Feature Extraction (8-Class) ✅ COMPLETE

> **Goal**: Re-extract all audio features using updated 8-class label schema.

| ID | Task | Status |
|----|------|:------:|
| T4.1 | Update `src/features/extractor.py` — ensure `extract_mfcc` outputs 40-coeff MFCC | ✅ |
| T4.2 | Update `src/features/extractor.py` — ensure `extract_mel_spectrogram` outputs (128, 128, 1) | ✅ |
| T4.3 | Re-run MFCC extraction for all splits → `features/mfcc_v2/{train,val,test}/` | ✅ |
| T4.4 | Re-run Mel-Spectrogram extraction → `features/mel_v2/{train,val,test}/` | ✅ |
| T4.5 | Re-run YAMNet embedding extraction (16kHz, 3s) → `features/yamnet_v2/{train,val,test}.npy` | ✅ |
| T4.6 | Compute and save global normalization stats (mean, std) for YAMNet embeddings | ✅ |
| T4.7 | Write feature validation script — confirm shape and label alignment across all splits | ✅ |

---

## Phase 5 — Model Training (8-Class) ✅ COMPLETE

> **Goal**: Train all 5 models on the remapped 8-class dataset. YAMNet is primary.

### 5a — Traditional ML Models

| ID | Task | Status | Result |
|----|------|:------:|--------|
| T5.1 | Train `RandomForestClassifier` → `models/rf_v2.pkl` | ✅ | 83.0% |
| T5.2 | Train `SVC(kernel='rbf')` → `models/svm_v2.pkl` | ✅ | **91.1%** |
| T5.3 | Train `XGBClassifier` → `models/xgb_v2.json` | ✅ | 89.2% |
| T5.4 | Evaluate RF, SVM, XGB on 8-class test set | ✅ | metrics_v2_ml.json |

### 5b — CNN (Deep Learning)

| ID | Task | Status | Result |
|----|------|:------:|--------|
| T5.5 | Create `src/models_v2.py` — `AudioCNN_V2` with 8-class output | ✅ | |
| T5.6 | Train unified CNN on combined (US8K + ESC-50) mel-spectrogram data → `models/cnn_v2.pt` | ✅ | 69.2% |
| T5.7 | Evaluate CNN on 8-class test set | ✅ | |

### 5c — YAMNet (Transfer Learning — Primary)

| ID | Task | Status | Result |
|----|------|:------:|--------|
| T5.8 | Load TF-Hub YAMNet base; freeze MobileNet weights | ✅ | |
| T5.9 | Build head: `Dense(512) → Drop(0.5) → Dense(256) → Drop(0.4) → Dense(8)` | ✅ | |
| T5.10 | Train unified YAMNet on combined embeddings → `models/yamnet_v2.h5` | ✅ | **~91.3%** |

---

## Phase 6 — Ensemble Architecture ✅ COMPLETE

> **Goal**: Combine 5-model outputs with optimized adaptive weights on 8-class targets.

| ID | Task | Status |
|----|------|:------:|
| T6.1 | Collect probability outputs `P[model, sample, 8]` for all 5 models on val/test splits | ✅ |
| T6.2 | Run Nelder-Mead weight optimisation (40 restarts) to maximize val accuracy | ✅ |
| T6.3 | Implement safety-class confidence boosting for `speech`, `siren`, `horn` | ✅ |
| T6.4 | Implement confidence threshold gate (safety classes: 0.20–0.25; others: 0.50) | ✅ |
| T6.5 | Save optimized weights → `results/ensemble_weights_v2_8class.json` | ✅ |
| T6.6 | Evaluate 5-model ensemble on test set — **95.27% Accuracy, 95.04% F1** | ✅ |

---

## Phase 7 — Evaluation & Reporting ✅ COMPLETE

> **Goal**: Generate publication-quality evaluation artifacts for all models.

| ID | Task | Status |
|----|------|:------:|
| T7.1 | Write `src/evaluate_v2.py` — load all 5 models, run full evaluation pipeline on 8-class test set | ✅ |
| T7.2 | Generate per-model results table (Accuracy, Macro F1, Precision, Recall) | ✅ |
| T7.3 | Generate confusion matrix for CNN (8×8, normalized) → `results/figures/cm_cnn_v2.png` | ✅ |
| T7.4 | Generate confusion matrix for YAMNet → `results/figures/cm_yamnet_v2.png` | ✅ |
| T7.5 | Generate per-class bar chart comparing all 5 models → `results/figures/class_f1_comparison_v2.png` | ✅ |
| T7.6 | Generate ablation table: 10-class (v1) vs 8-class (v2) accuracy comparison | ✅ |
| T7.7 | Export all metrics → `results/metrics_v2.json` | ✅ |
| T7.8 | Validate safety class recall ≥ 95% for `siren`, `horn`, `speech` | ✅ |

---

## Phase 8 — Recommendation Engine (v2) ✅ COMPLETE

> **Goal**: Update the 4-tier recommendation logic to handle 8 optimized classes.

| ID | Task | Status |
|----|------|:------:|
| T8.1 | Define `RECOMMENDATION_TABLE_V2` in `src/data/label_map_v2.py` for all 8 classes | ✅ |
| T8.2 | Implement safety override: detect `siren/horn/speech` → force maximum alert settings | ✅ |
| T8.3 | `get_recommendation()` and `get_safety_threshold()` functions for 8-class | ✅ |
| T8.4 | Validate all 8 class → setting mappings produce correct hardware params | ✅ |
| T8.5 | Validate reasoning strings are human-readable for all 8 classes | ✅ |

---

## Phase 9 — Deployment & Demo ✅ COMPLETE

> **Goal**: Update Gradio demo and FastAPI to use the v2 8-class system.

| ID | Task | Status |
|----|------|:------:|
| T9.1 | Update `demo/gradio_app.py` — replace 10-class labels/emojis with 8-class v2 | ✅ |
| T9.2 | Update `_ensure_models_loaded()` to load v2 model files (`*_v2.*`) | ✅ |
| T9.3 | Update `_infer()` — use 8-class probability vectors and v2 ensemble weights | ✅ |
| T9.4 | Update Gradio header description to state "8-class optimized" architecture | ✅ |
| T9.5 | Test end-to-end demo with real audio samples (Dog Bark verified with 5/5 model agreement) | ✅ |
| T9.6 | Premium dark-mode UI deployed | ✅ |

---

## Phase 10 — IEEE Paper Preparation ✅ COMPLETE

> **Goal**: Write and structure a complete IEEE-format research paper.

| ID | Task | Status |
|----|------|:------:|
| T10.1 | Write Abstract (150–250 words): problem, method, key result metrics | ✅ |
| T10.2 | Write Introduction: hearing aid problem, existing gaps, contributions list | ✅ |
| T10.3 | Write Related Work: survey of ESC, hearing aid AI, transfer learning | ✅ |
| T10.4 | Write System Architecture section with pipeline diagram (§5.2 from PRD) | ✅ |
| T10.5 | Write Experimental Setup: dataset stats, preprocessing, split rationale, class mapping | ✅ |
| T10.6 | Write Results section: tables from Phase 7, confusion matrices, ablation study | ✅ |
| T10.7 | Write Conclusion + Future Work (edge deployment, real-time streaming) | ✅ |
| T10.8 | Format in IEEE template (LaTeX or Word) — all figures at 300 DPI | ✅ |
| T10.9 | Final review — validate all claims match evaluation metrics from `results/metrics_v2.json` | ✅ |

---

## Appendix — New vs Old Class Mapping Reference

| Old Label (v1) | New Label (v2) |
|---------------|----------------|
| air_conditioner | background_noise |
| engine_idling | traffic |
| drilling | construction |
| jackhammer | construction |
| gun_shot | construction |
| car_horn | horn |
| children_playing | speech |
| dog_bark | dog_bark |
| siren | siren |
| street_music | music |

*Last updated: 2026-03-11T10:22:00+05:30*

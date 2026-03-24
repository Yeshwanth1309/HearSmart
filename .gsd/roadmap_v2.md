---
status: ACTIVE
type: technical-roadmap
project: Context-Aware Hearing Aid Settings Optimizer
version: 2.0
created: 2026-03-11T10:22:00+05:30
scope: Phase 3 → Phase 10 (v2 — 8-Class Optimized Implementation)
derives_from:
  - .gsd/prd.md (v2.0)
  - .gsd/todo.md (v2.0)
---

# Training & Implementation Roadmap — v2

> This roadmap defines the exact execution sequence for re-implementing the
> hearing aid system on the new **8-class acoustically distinct schema**.
> Phases 1 and 2 are already complete. Execution begins at Phase 3.

---

## Execution Order

```
Phase 3 → Phase 4 → Phase 5a → Phase 5b → Phase 5c → Phase 6 → Phase 7 → Phase 8 → Phase 9 → Phase 10
```

---

## Phase 3 — Label Remapping

**Script**: `src/data/label_map_v2.py` (already created)

**Steps**:
1. Load `data/splits/train.csv`, `val.csv`, `test.csv`
2. Apply `US8K_TO_8CLASS` mapping to `class` column → new column `class_v2`
3. Save remapped CSVs to `data/splits_v2/train.csv`, `val.csv`, `test.csv`
4. Apply `ESC50_TO_8CLASS` to `esc50_loader.py` → column `us8k_label` → `class_v2`
5. Validate: `value_counts()` on all splits — confirm all 8 classes present

**Output**: `data/splits_v2/*.csv`

---

## Phase 4 — Feature Extraction

**Script**: `src/features/extract_features_v2.py`

**Steps**:
1. Load `data/splits_v2/*.csv`
2. For each sample, run:
   - `extract_mfcc(path)` → save to `features/mfcc_v2/{split}/{stem}.npy`
   - `extract_mel_spectrogram(path)` → save to `features/mel_v2/{split}/{stem}.npy`
3. Run YAMNet embedding extraction (16 kHz, 3-second window):
   - Load YAMNet from TF-Hub
   - Extract embeddings from all train/val/test files
   - Mean-pool frames → `features/yamnet_v2/{split}.npy`
4. Compute and save normalization stats:
   - `models/yamnet_v2_mu.npy`, `models/yamnet_v2_sig.npy` (from train set)

**Output**: `features/mfcc_v2/`, `features/mel_v2/`, `features/yamnet_v2/`

---

## Phase 5a — Traditional ML Training (RF, SVM, XGB)

**Script**: `src/train_traditional_v2.py`

**Architecture**:
- Input: MFCC features from `features/mfcc_v2/train/`
- Labels: 8-class indices from `data/splits_v2/train.csv`
- Class weights: inverse-frequency from training set

**Steps**:
1. Load all train MFCC features + labels
2. Train `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)`
3. Train `SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced')`
4. Train `XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05)`
5. Evaluate each on val set → save metrics to `results/metrics_v2_ml.json`
6. Save models: `models/rf_v2.pkl`, `models/svm_v2.pkl`, `models/xgb_v2.json`

---

## Phase 5b — CNN Training

**Script**: `src/train_cnn_v2.py`

**Architecture** (updated for 8 classes):
```
Conv2D(32) → BN → MaxPool
Conv2D(64) → BN → MaxPool
Conv2D(128) → BN → GlobalAvgPool
Dense(256, relu) → Dropout(0.5) → Dense(8, softmax)
```

**Steps**:
1. Build `UnifiedCNNDataset` loading from `features/mel_v2/train/` (US8K) + ESC-50 on-the-fly
2. Apply `WeightedRandomSampler` using 8-class inverse-frequency weights
3. Train 60 epochs with `AdamW(lr=1e-3)` + `CosineAnnealingLR`
4. Save best checkpoint → `models/cnn_v2.pt`
5. Evaluate on `features/mel_v2/test/` → log accuracy, F1, confusion matrix

---

## Phase 5c — YAMNet Training (Primary Model)

**Script**: `src/train_yamnet_v2.py`

**Architecture**:
```
YAMNet MobileNet (frozen) → Mean Pool → [1024]
Dense(512, relu) → Dropout(0.5)
Dense(256, relu) → Dropout(0.4)
Dense(8, softmax)
```

**Steps**:
1. Load `features/yamnet_v2/train.npy` + ESC-50 extracted embeddings
2. Normalize with global mu/sig from Phase 4
3. Compute 8-class inverse-frequency weights from merged label array
4. Train 80 epochs with `Adam(lr=1e-3)` + `ReduceLROnPlateau`
5. Save best (`monitor=val_accuracy`) → `models/yamnet_v2.h5`
6. Evaluate on normalized val/test embeddings → log full classification report

---

## Phase 6 — Ensemble Optimization

**Script**: `src/optimize_ensemble_v2.py`

**Steps**:
1. Load all 5 models: RF, SVM, XGB, CNN, YAMNet
2. Collect `P[model, N_val, 8]` probability matrices for validation set
3. Run Nelder-Mead weight optimization (40 restarts):
   - Minimize `-accuracy_score(y_val, weighted_ensemble.argmax(1))`
4. Apply safety-class boosting rules:
   - If YAMNet confidence for `siren/horn/speech` > 0.20–0.25 → boost YAMNet weight
5. Evaluate final ensemble on test set
6. Save → `results/ensemble_weights_v2_8class.json`

---

## Phase 7 — Evaluation & Figures

**Script**: `src/evaluate_v2.py`

**Outputs**:
- `results/metrics_v2.json` — per-model and ensemble metrics
- `results/figures/cm_cnn_v2.png` — 8×8 confusion matrix (CNN)
- `results/figures/cm_yamnet_v2.png` — 8×8 confusion matrix (YAMNet)
- `results/figures/cm_ensemble_v2.png` — 8×8 confusion matrix (Ensemble)
- `results/figures/class_f1_comparison_v2.png` — bar chart: 5 models × 8 classes
- `results/figures/ablation_v1_v2.png` — accuracy comparison: 10-class vs 8-class

**Safety Validation**:
- Recall for `siren`, `horn`, `speech` must be ≥ 0.95 for YAMNet and Ensemble

---

## Phase 8 — Recommendation Engine Update

**Script**: update `src/recommendations.py`

**Steps**:
1. Import `RECOMMENDATION_TABLE_V2` from `src/data/label_map_v2.py`
2. Update `recommend_from_probs(probs, class_names)` to accept 8-class input
3. Add safety override logic for `siren/horn/speech` detection
4. Validate all 8 mappings produce correct `volume`, `noise_reduction`, `directionality`, `speech_enhancement`

---

## Phase 9 — Demo Update

**Script**: `demo/gradio_app.py`

**Steps**:
1. Replace `CLASS_NAMES` (10-class) with `CLASS_NAMES_V2` (8-class)
2. Replace `CLASS_EMOJIS` with `CLASS_EMOJIS_V2`
3. Update `_ensure_models_loaded()` to load `rf_v2`, `svm_v2`, `xgb_v2`, `cnn_v2`, `yamnet_v2`
4. Update `_infer()` to output 8-class probabilities
5. Update recommendation display to render 8-class settings
6. Test with each of the 8 class types using real audio examples

---

## Phase 10 — IEEE Paper

**Structure** (IEEE double-column format):
1. Title + Authors + Abstract
2. Introduction (problem, gap, contribution)
3. Related Work (3–5 citations per area)
4. System Architecture (pipeline figure from PRD §5.2)
5. Dataset & Preprocessing (US8K + ESC-50 stats, remapping rationale)
6. Experiments (5 models, 8 classes, metrics table)
7. Results & Discussion (figures from Phase 7)
8. Ablation: v1 (10-class) vs v2 (8-class)
9. Conclusion + Future Work
10. References

---

## File Output Summary

| File | Phase | Status |
|------|-------|--------|
| `src/data/label_map_v2.py` | 3 | ✅ Created |
| `data/splits_v2/*.csv` | 3 | ⬜ |
| `features/mfcc_v2/` | 4 | ⬜ |
| `features/mel_v2/` | 4 | ⬜ |
| `features/yamnet_v2/*.npy` | 4 | ⬜ |
| `models/rf_v2.pkl` | 5a | ⬜ |
| `models/svm_v2.pkl` | 5a | ⬜ |
| `models/xgb_v2.json` | 5a | ⬜ |
| `models/cnn_v2.pt` | 5b | ⬜ |
| `models/yamnet_v2.h5` | 5c | ⬜ |
| `results/ensemble_weights_v2_8class.json` | 6 | ⬜ |
| `results/metrics_v2.json` | 7 | ⬜ |
| `results/figures/cm_*.png` | 7 | ⬜ |

*Last updated: 2026-03-11T10:22:00+05:30*

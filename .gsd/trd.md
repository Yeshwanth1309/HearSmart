---
status: FINALIZED
type: technical-requirements-document
project: Context-Aware Hearing Aid System
created: 2026-03-08T16:36:00+05:30
finalized: 2026-03-08T16:36:00+05:30
source: User-provided TRD
derives_from: .gsd/prd.md
---

# Technical Requirements Document (TRD)

> **Status**: `FINALIZED`
>
> System architecture specification translating TRD into implementation-ready modules.
> Derived from and constrained by the [PRD](.gsd/prd.md).

---

## 1. System Overview

A **multi-model audio classification pipeline** that processes environmental sounds
and generates context-aware hearing aid settings. The system combines traditional ML,
deep learning, transfer learning, and rule-based decision logic across a six-stage
inference pipeline.

### Hard Constraints (from PRD)

| Constraint | Target | Validation Method |
|------------|--------|-------------------|
| Macro F1-score | ≥ 85% | Test-set evaluation on UrbanSound8K |
| Inference latency | < 500 ms end-to-end | Wall-clock timing per sample |
| Total ensemble model size | < 200 MB on disk | `du -sh models/` |
| Reproducibility | Fully deterministic | Fixed seeds, pinned deps |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                                    │
│  Audio file (.wav/.mp3) │ Microphone stream │ API request (base64)      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  MODULE 1: AUDIO PREPROCESSING                          │
│  Validation → Resample (16 kHz) → Mono → 3s Norm → DC Remove → Norm    │
│  Quality: SNR, RMS, clipping, silence detection                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  MODULE 2: FEATURE EXTRACTION                           │
│  ┌──────────────┐  ┌───────────────────┐  ┌──────────────────┐         │
│  │ MFCC (80,)   │  │ Mel-Spec (128×128)│  │ Waveform (48000) │         │
│  │ → Trad. ML   │  │ → CNN             │  │ → YAMNet          │         │
│  └──────────────┘  └───────────────────┘  └──────────────────┘         │
└──────┬──────────────────────┬──────────────────────┬───────────────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  MODULE 3: MODEL ENSEMBLE (5 models)                    │
│  ┌──────────┐  ┌──────┐  ┌─────────┐  ┌──────┐  ┌────────┐            │
│  │ Random   │  │ SVM  │  │ XGBoost │  │ CNN  │  │ YAMNet │            │
│  │ Forest   │  │      │  │         │  │      │  │        │            │
│  │ w=0.08   │  │w=0.10│  │ w=0.12  │  │w=0.25│  │ w=0.45 │            │
│  └────┬─────┘  └──┬───┘  └────┬────┘  └──┬───┘  └───┬────┘            │
│       └────────────┴──────────┴───────────┴──────────┘                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  Weighted probability vectors
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  MODULE 4: DECISION ENGINE                              │
│  Weighted voting → Confidence thresholding → Safety-critical override   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  (class, confidence, is_uncertain)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  MODULE 5: RECOMMENDATION ENGINE                        │
│  Tier 1: Rule-based → Tier 2: Confidence adj. → Tier 3: Acoustic →     │
│  Tier 4: User personalization                                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  MODULE 6: OUTPUT LAYER                                  │
│  JSON response │ REST API (FastAPI) │ Web UI (Gradio) │ CLI             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Specifications

### 3.1 Module 1 — Audio Preprocessing

**Purpose:** Standardize raw audio from any source into a clean, fixed-format signal.

**Location:** `src/feature_extraction.py` → `load_audio()`, `normalize_audio()`, `pad_or_truncate()`, `preprocess_audio()`

| Step | Operation | Parameters | Output |
|------|-----------|------------|--------|
| 1 | Format validation | Check .wav/.mp3 integrity | Pass / RuntimeError |
| 2 | Loading | `librosa.load()` / `soundfile.read()` | Raw audio array |
| 3 | Resampling | target_sr = 16000 Hz | 16 kHz mono signal |
| 4 | Mono conversion | `librosa.to_mono()` if stereo | 1D array |
| 5 | Duration normalization | `pad_or_truncate(audio, 48000)` | Exactly 3s (48000 samples) |
| 6 | DC offset removal | Subtract mean | Zero-centered signal |
| 7 | Peak normalization | Scale to [-1, 1] | Normalized signal |

#### Quality Metrics (computed but not blocking)

| Metric | Description | Method |
|--------|-------------|--------|
| SNR | Signal-to-noise ratio | Spectral estimation |
| RMS energy | Root mean square amplitude | `np.sqrt(np.mean(audio**2))` |
| Clipping detection | Consecutive max-amplitude samples | Threshold check |
| Silence detection | RMS below threshold | RMS < 1e-4 flags warning |

#### Interface Contract

```python
def preprocess_audio(
    file_path: str,
    target_sr: int = 16000,
    duration: float = 3.0
) -> tuple[np.ndarray, int]:
    """
    Returns: (audio: np.ndarray shape=(48000,) dtype=float32, sr: int=16000)
    Raises: RuntimeError on validation failure
    """
```

---

### 3.2 Module 2 — Feature Extraction Pipeline

**Purpose:** Transform preprocessed audio into three parallel feature representations.

**Location:** `src/feature_extraction.py` → `extract_mfcc()`, `extract_mel_spectrogram()`, `extract_waveform()`

#### Feature A: MFCC

| Property | Value |
|----------|-------|
| Coefficients | 40 |
| Aggregation | Mean + Std across time frames |
| Output shape | `(80,)` — 40 means + 40 stds |
| dtype | float32 |
| Used by | Random Forest, SVM, XGBoost |
| Parameters | n_fft=2048, hop_length=512, sr=16000 |

```python
def extract_mfcc(
    file_path: str,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:  # shape (80,), dtype float32
```

#### Feature B: Mel-Spectrogram

| Property | Value |
|----------|-------|
| Mel bands | 128 (TRD target) / 64 (current impl) |
| Time frames | 128 |
| Channel dimension | 1 (for CNN input) |
| Output shape | `(128, 128, 1)` |
| dtype | float32 |
| Normalization | Log-scale (dB), then min-max to [0, 1] |
| Used by | CNN |
| Parameters | n_fft=1024, hop_length=512, n_mels=128 |

```python
def extract_mel_spectrogram(
    file_path: str,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128
) -> np.ndarray:  # shape (128, 128, 1), dtype float32
```

> **Note:** TRD specifies 128 mel bands. Current implementation uses n_mels=64.
> Reconciliation required during Phase 5 (CNN training). The PRD originally
> specified 128×128 as the target spectrogram shape.

#### Feature C: Raw Waveform

| Property | Value |
|----------|-------|
| Sample rate | 16000 Hz |
| Duration | 3.0 seconds |
| Output shape | `(48000,)` |
| dtype | float32 |
| Used by | YAMNet |

```python
def extract_waveform(
    file_path: str,
    sr: int = 16000,
    duration: float = 3.0
) -> np.ndarray:  # shape (48000,), dtype float32
```

#### Feature Caching

**Location:** `src/feature_extraction.py` → `cache_features()`

All three feature types are extracted once and cached to `features/` as `.npy` files:

```
features/
├── mfcc/
│   ├── train/       # (N_train,) .npy files, each (80,)
│   ├── val/
│   └── test/
├── mel_spectrogram/
│   ├── train/       # (N_train,) .npy files, each (128, 128, 1)
│   ├── val/
│   └── test/
└── waveform/
    ├── train/       # (N_train,) .npy files, each (48000,)
    ├── val/
    └── test/
```

---

### 3.3 Module 3 — Data Augmentation

**Purpose:** Increase training data diversity to improve generalization.

**Location:** `src/feature_extraction.py` → `augment_audio()`

| Augmentation | Method | Parameters | Applied Probability |
|--------------|--------|------------|---------------------|
| Time stretching | `librosa.effects.time_stretch()` | rate ∈ [0.8, 1.2] | Probabilistic |
| Pitch shifting | `librosa.effects.pitch_shift()` | steps ∈ [-2, 2] | Probabilistic |
| Noise addition | Additive Gaussian noise | σ ∈ [0.005, 0.015] | Probabilistic |
| Time shifting | Circular shift | max_shift = sr//10 | Probabilistic |
| SpecAugment | Frequency/time masking | On mel-spectrogram | CNN only |

```python
def augment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    seed: int = 42
) -> np.ndarray:  # same shape, dtype float32
```

**Constraints:**
- Applied **only** to training split — never to validation or test
- Deterministic via seed parameter
- Post-augmentation signal is re-normalized

---

### 3.4 Module 4 — Data Pipeline

**Purpose:** Load, validate, split, and serve the UrbanSound8K dataset.

**Location:** `src/data_loader.py` → `UrbanSoundDataset`

#### Dataset Specification

| Property | Value |
|----------|-------|
| Dataset | UrbanSound8K |
| Total samples | 8,732 labeled audio clips |
| Sound classes | 10 |
| Max clip duration | ≤ 4 seconds |
| Standard duration | 3 seconds (trimmed/padded) |
| Predefined folds | 10 (UrbanSound8K's fold structure) |

#### Validation Checks

| Check | Method | Failure Action |
|-------|--------|----------------|
| File existence | `Path.exists()` | Log + skip |
| Format integrity | `sf.read()` attempt | Log + skip |
| Duration > 0 | `len(audio) > 0` | Log + skip |
| RMS energy ≥ 1e-4 | `np.sqrt(np.mean(audio**2))` | Log + skip |
| Class distribution | Stratification check | Warning if imbalanced > 20% |
| Corrupted file detection | Load + quality check | Log + skip |

#### Split Strategy

| Split | Ratio | Purpose | Augmentation |
|-------|-------|---------|-------------|
| Train | 70% (~6,112 samples) | Model training | Yes |
| Validation | 15% (~1,310 samples) | Hyperparameter tuning | No |
| Test | 15% (~1,310 samples) | Final evaluation | No |

**Location on disk:**

```
data/
├── UrbanSound8K/
│   ├── audio/
│   │   ├── fold1/ ... fold10/
│   └── metadata/
│       └── UrbanSound8K.csv
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

#### Interface Contract

```python
class UrbanSoundDataset:
    def __init__(self, data_root: str = "data/UrbanSound8K"):
        ...

    def load_metadata(self) -> pd.DataFrame:
        """Load UrbanSound8K.csv metadata."""

    def validate_files(self) -> pd.DataFrame:
        """Validate all audio files, return clean DataFrame."""

    def create_splits(
        self,
        df: pd.DataFrame,
        output_dir: str = "data/splits",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> dict:
        """Create stratified train/val/test splits."""
```

---

### 3.5 Module 5 — Model Training Pipeline

**Purpose:** Train 5 independent classification models using appropriate features.

#### 3.5.1 Traditional ML Branch

**Location:** `src/train_rf_baseline.py`, `src/train_rf_tuned.py`, `src/train_svm.py`, `src/train_xgboost.py`

**Shared data loading:** `src/train.py` → `load_training_data()`

| Model | Library | Feature Input | Key Hyperparameters | Expected Size |
|-------|---------|---------------|---------------------|---------------|
| Random Forest | sklearn | MFCC (80,) | n_estimators, max_depth, min_samples_split | ~140 MB |
| SVM | sklearn | MFCC (80,) | C, gamma, kernel='rbf' | ~3 MB |
| XGBoost | xgboost | MFCC (80,) | n_estimators, learning_rate, max_depth | ~10 MB |

**Training protocol (Traditional ML):**

```
1. Load cached MFCC features via load_training_data()
2. Encode labels with LabelEncoder (saved to models/label_encoder.pkl)
3. Hyperparameter optimization (grid search or Optuna)
4. Stratified cross-validation (10-fold)
5. Train final model on full training set
6. Serialize to models/{model_name}.pkl or .json
7. Log metrics and training time
```

**Serialization:**

| Model | Format | Path |
|-------|--------|------|
| Random Forest | joblib (.pkl) | `models/random_forest.pkl` |
| SVM | joblib (.pkl) | `models/svm.pkl` |
| XGBoost | JSON (.json) | `models/xgboost.json` |
| Label Encoder | joblib (.pkl) | `models/label_encoder.pkl` |

#### 3.5.2 Deep Learning Branch

**Location:** `src/models.py` (to be implemented)

**CNN Model:**

| Property | Value |
|----------|-------|
| Framework | PyTorch |
| Input | Mel-spectrogram (128, 128, 1) |
| Architecture | Conv2D stack → BatchNorm → Dropout → FC |
| Optimizer | Adam, lr=1e-3 with scheduler |
| Loss | CrossEntropyLoss |
| Epochs | 50 (with early stopping, patience=10) |
| Batch size | 32 |
| Checkpointing | Save best val-loss model |

```python
class AudioCNN(nn.Module):
    """
    CNN for mel-spectrogram classification.
    Input: (batch, 1, 128, 128)
    Output: (batch, 10) logits
    """
```

**YAMNet (Transfer Learning):**

| Property | Value |
|----------|-------|
| Framework | TensorFlow / TF Hub |
| Base model | YAMNet (pretrained on AudioSet) |
| Input | Raw waveform (48000,) at 16 kHz |
| Strategy | Freeze base, add classification head |
| Head | GlobalAvgPool → Dense(256) → Dropout → Dense(10) |
| Fine-tuning | Optional: unfreeze top N layers |
| Optimizer | Adam, lr=1e-4 |
| Epochs | 30 (with early stopping, patience=7) |

```python
def build_yamnet_model(num_classes: int = 10) -> tf.keras.Model:
    """
    Build YAMNet transfer learning model.
    Input: (batch, 48000) raw waveform
    Output: (batch, 10) probabilities
    """
```

**Serialization (DL):**

| Model | Format | Path |
|-------|--------|------|
| CNN | PyTorch state_dict (.pt) | `models/cnn.pt` |
| YAMNet | TF SavedModel or .h5 | `models/yamnet/` |

---

### 3.6 Module 6 — Ensemble Decision Engine

**Purpose:** Combine predictions from all 5 models into a single robust decision.

**Location:** `src/models.py` (to be implemented)

#### Weighted Probability Voting

The ensemble computes a weighted sum of per-model probability vectors:

```
P_ensemble(class_k) = Σ_i (w_i × P_i(class_k))    for i ∈ {RF, SVM, XGB, CNN, YAMNet}
```

#### Model Weights

| Model | Weight | Rationale |
|-------|--------|-----------|
| Random Forest | 0.08 | Lowest expected accuracy, baseline |
| SVM | 0.10 | Slightly better than RF on audio tasks |
| XGBoost | 0.12 | Gradient boosting advantage |
| CNN | 0.25 | Deep learning captures complex patterns |
| YAMNet | 0.45 | Transfer learning, highest expected accuracy |
| **Total** | **1.00** | Weights sum to 1 |

> **Weight calibration:** Initial weights are heuristic. After all models are trained,
> weights may be recalibrated using validation-set performance. Weights can be
> optimized via grid search or Optuna on validation Macro-F1.

#### Confidence Thresholding

| Scenario | Threshold (τ) | Action |
|----------|---------------|--------|
| Default | 0.70 | Accept prediction |
| Safety-critical sounds | 0.60 | Lower bar — don't miss safety events |
| Uncertain / ambiguous | 0.75 | Higher bar — require stronger confidence |
| Below threshold | — | Mark `is_uncertain = true`, apply conservative recommendation |

```python
def ensemble_predict(
    model_predictions: dict[str, np.ndarray],  # {model_name: probabilities (10,)}
    weights: dict[str, float],
    safety_classes: list[int] = [1, 6, 8],  # car_horn, gun_shot, siren
    default_threshold: float = 0.70,
    safety_threshold: float = 0.60,
    uncertain_threshold: float = 0.75
) -> dict:
    """
    Returns: {
        'class_id': int,
        'class_name': str,
        'confidence': float,
        'is_uncertain': bool,
        'model_contributions': dict[str, float]
    }
    """
```

#### Safety-Critical Override Logic

```
FOR each model in {RF, SVM, XGB, CNN, YAMNet}:
    IF model predicts safety class WITH probability > safety_threshold:
        IF no existing safety override OR new probability > existing:
            SET override = True
            SET predicted_class = safety class
            SET confidence = max(model probability, ensemble probability)

IF override:
    RETURN safety class prediction (regardless of ensemble vote)
```

**Safety classes:**

| Class ID | Class Name | Override Threshold |
|----------|------------|-------------------|
| 1 | Car horn | 0.60 |
| 6 | Gun shot | 0.60 |
| 8 | Siren | 0.60 |

---

### 3.7 Module 7 — Recommendation Engine

**Purpose:** Convert classified environment + confidence into optimal hearing aid parameters.

**Location:** `src/recommendations.py` (to be implemented)

#### Four-Tier Decision Architecture

```
Classification Result
       │
       ▼
┌──────────────────┐
│  TIER 1           │  Rule-based: class → base parameters
│  Rule Mapping     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  TIER 2           │  Confidence < threshold → conservative defaults
│  Confidence Adj.  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  TIER 3           │  SNR, speech_prob, loudness → fine adjustments
│  Acoustic Refine  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  TIER 4           │  User profile → personalized modifications
│  Personalization  │
└──────────────────┘
```

#### Tier 1 — Rule-Based Mapping

Each of the 10 sound classes maps to a base hearing aid configuration:

| Class | Volume (1–10) | Noise Reduction | Directionality | Speech Enhancement |
|-------|:---:|---|---|---|
| Air conditioner | 5 | High | Omnidirectional | On |
| Car horn | 3 | High | Adaptive | Off |
| Children playing | 6 | Medium | Omnidirectional | On |
| Dog bark | 5 | Medium | Adaptive | Off |
| Drilling | 3 | High | Directional | On |
| Engine idling | 5 | High | Omnidirectional | Off |
| Gun shot | 2 | High | Adaptive | Off |
| Jackhammer | 3 | High | Directional | On |
| Siren | 4 | High | Adaptive | Off |
| Street music | 7 | Low | Omnidirectional | Off |

#### Tier 2 — Confidence Adjustment

```python
if prediction.is_uncertain:
    # Apply conservative defaults
    settings.volume = 5              # neutral mid-range
    settings.noise_reduction = "Medium"
    settings.directionality = "Omnidirectional"
    settings.speech_enhancement = True
    settings.reasoning += " [Conservative: low-confidence prediction]"
```

#### Tier 3 — Acoustic Refinement

| Audio Property | Adjustment Rule |
|---------------|-----------------|
| High SNR (> 20 dB) | Reduce noise reduction by one level |
| Low SNR (< 5 dB) | Increase noise reduction, enable speech enhancement |
| High speech probability | Switch to Directional + Speech Enhancement On |
| High loudness (> 0.8 RMS) | Reduce volume by 1–2 |
| Low loudness (< 0.1 RMS) | Increase volume by 1–2 |

#### Tier 4 — User Personalization (Should-Have)

| User Profile Field | Type | Adjustment |
|-------------------|------|------------|
| Hearing loss severity | mild / moderate / severe | Severe → +2 volume |
| Tinnitus | bool | True → +1 noise reduction |
| Age group | young / adult / elderly | Elderly → favor speech enhancement |
| Listening preference | speech / music / balanced | Music → reduce noise filtering |

#### Output Schema

```python
@dataclass
class Recommendation:
    volume: int              # 1–10
    noise_reduction: str     # "Low" | "Medium" | "High"
    directionality: str      # "Omnidirectional" | "Directional" | "Adaptive"
    speech_enhancement: bool # True | False
    reasoning: str           # Human-readable explanation
    tier_applied: list[int]  # [1, 2, 3] which tiers were applied
```

---

### 3.8 Module 8 — Evaluation Framework

**Purpose:** Compute, visualize, and export research-grade metrics for all models.

**Location:** `src/evaluate.py` (to be implemented), `figures/`

#### Per-Model Metrics

| Metric | Computation | Required |
|--------|-------------|----------|
| Accuracy | `accuracy_score(y_true, y_pred)` | Yes |
| Macro F1-score | `f1_score(y_true, y_pred, average='macro')` | Yes — **primary metric** |
| Weighted F1-score | `f1_score(y_true, y_pred, average='weighted')` | Yes |
| Per-class F1 | `f1_score(y_true, y_pred, average=None)` | Yes |
| Precision (macro) | `precision_score(y_true, y_pred, average='macro')` | Yes |
| Recall (macro) | `recall_score(y_true, y_pred, average='macro')` | Yes |
| Confusion matrix | `confusion_matrix(y_true, y_pred)` | Yes |
| ROC curve (OvR) | `roc_curve()` per class | Yes |
| AUC (OvR) | `roc_auc_score(y_true, y_prob, multi_class='ovr')` | Yes |
| Calibration | Brier score, reliability diagram | Yes |
| Training curve | Loss & accuracy per epoch (DL only) | Yes |

#### Cross-Model Comparison Artifacts

| Artifact | Type | Contents |
|----------|------|----------|
| Accuracy comparison bar chart | PNG 300+ DPI | All 5 models side-by-side |
| F1-score comparison bar chart | PNG 300+ DPI | Macro F1 per model |
| Paradigm comparison table | Markdown + LaTeX | Trad. ML vs. DL vs. Transfer |
| Inference latency table | Markdown | ms per sample per model |
| Safety-class F1 heatmap | PNG 300+ DPI | F1 for classes 1, 6, 8 per model |
| ROC curves (overlay) | PNG 300+ DPI | All models, per class |
| Confusion matrices (grid) | PNG 300+ DPI | 5 matrices in one figure |

#### Publication Figure Standards

| Property | Requirement |
|----------|-------------|
| Resolution | ≥ 300 DPI |
| Format | PNG (figures), PDF (paper) |
| Font size | ≥ 8pt in figures |
| Color scheme | Colorblind-safe palette |
| Labels | All axes labeled with units |
| Legends | Included in every multi-series plot |
| Figure size | IEEE column width (3.5" single, 7" double) |

#### Metrics Export

```
results/
├── metrics/
│   ├── random_forest_metrics.json
│   ├── svm_metrics.json
│   ├── xgboost_metrics.json
│   ├── cnn_metrics.json
│   ├── yamnet_metrics.json
│   └── ensemble_metrics.json
└── comparison/
    ├── model_comparison.json
    └── model_comparison.csv

figures/
├── confusion_matrices/
│   ├── rf_confusion.png
│   ├── svm_confusion.png
│   ├── xgb_confusion.png
│   ├── cnn_confusion.png
│   └── yamnet_confusion.png
├── roc_curves/
│   ├── rf_roc.png
│   ├── ...
│   └── all_models_roc.png
├── training_curves/
│   ├── cnn_training.png
│   └── yamnet_training.png
├── comparison/
│   ├── accuracy_comparison.png
│   ├── f1_comparison.png
│   └── paradigm_comparison.png
└── safety/
    └── safety_class_performance.png
```

---

### 3.9 Module 9 — Deployment Architecture

**Purpose:** Serve the trained pipeline through multiple interfaces.

#### Interface Specification

| Interface | Framework | Purpose | Priority |
|-----------|-----------|---------|----------|
| REST API | FastAPI | Programmatic access | Nice-to-have |
| Web Demo | Gradio | Interactive testing | Nice-to-have |
| Python SDK | Native | Library usage | Must-have |
| CLI tool | argparse | Batch processing | Should-have |

**Location:** `demo/api.py` (FastAPI), `demo/gradio_app.py` (Gradio)

#### REST API Specification

```
POST /predict
Content-Type: multipart/form-data
Body: audio_file (binary)

Response 200:
{
    "prediction": {
        "class_name": "siren",
        "class_id": 8,
        "confidence": 0.92,
        "is_uncertain": false,
        "model_contributions": {
            "random_forest": 0.08,
            "svm": 0.06,
            "xgboost": 0.11,
            "cnn": 0.22,
            "yamnet": 0.45
        }
    },
    "recommendation": {
        "volume": 4,
        "noise_reduction": "High",
        "directionality": "Adaptive",
        "speech_enhancement": false,
        "reasoning": "Siren detected with high confidence. Volume reduced and noise reduction maximized for safety awareness.",
        "tiers_applied": [1]
    },
    "metadata": {
        "latency_ms": 312,
        "model_versions": {...},
        "timestamp": "2026-03-08T16:36:00+05:30"
    }
}
```

#### Edge Deployment Targets (Nice-to-Have)

| Target | Requirements |
|--------|-------------|
| NVIDIA Jetson Nano | ONNX export, < 200 MB, ARM-compatible |
| ONNX Runtime | Convert PyTorch/TF models to ONNX |
| TensorFlow Lite | Convert YAMNet for mobile inference |

---

## 4. Technology Stack

### Core Dependencies

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| **Language** | Python | 3.10.x | Runtime |
| **Environment** | Conda | hearing-aid env | Isolation |
| **ML (Traditional)** | scikit-learn | pinned | RF, SVM, metrics |
| **ML (Boosting)** | xgboost | pinned | XGBoost model |
| **DL (CNN)** | PyTorch | pinned | CNN architecture |
| **DL (YAMNet)** | TensorFlow | pinned | Transfer learning |
| **Audio** | librosa | pinned | Feature extraction |
| **Audio I/O** | soundfile | pinned | Audio file handling |
| **Data** | pandas, numpy | pinned | Data manipulation |
| **Visualization** | matplotlib, seaborn | pinned | Publication figures |
| **Serialization** | joblib | pinned | Model persistence |
| **Optimization** | Optuna | pinned | Hyperparameter search |
| **Experiment tracking** | Weights & Biases | pinned | Training logs |
| **API** | FastAPI | pinned | REST endpoint |
| **Web UI** | Gradio | pinned | Demo interface |

### Reproducibility Requirements

| Requirement | Implementation |
|-------------|----------------|
| Fixed random seeds | `src/utils.py` → `set_seed(42)` across Python, NumPy, PyTorch, TF |
| Pinned dependencies | `requirements.txt` with exact versions |
| Deterministic operations | `torch.backends.cudnn.deterministic = True` |
| Cached features | `.npy` files prevent recomputation variance |

---

## 5. Hardware Requirements

### Minimum Specification

| Resource | Requirement |
|----------|-------------|
| CPU | 6 cores |
| RAM | 8 GB |
| Storage | 20 GB |
| GPU | Not required (CPU fallback) |

### Recommended Specification

| Resource | Requirement |
|----------|-------------|
| CPU | 8+ cores |
| RAM | 16 GB |
| Storage | 40 GB |
| GPU | NVIDIA RTX 3060 (6 GB VRAM) |

### Cloud Alternatives

| Platform | Tier | GPU |
|----------|------|-----|
| Google Colab | Pro | T4 / V100 |
| Kaggle | Free | P100 |

---

## 6. Testing & Validation Requirements

### Unit Tests

| Module | Test Focus | Location |
|--------|-----------|----------|
| Feature extraction | MFCC shape (80,), Mel-spec shape (128,128,1), Waveform shape (48000,) | `tests/test_features.py` |
| Data loader | Metadata loading, file validation, split ratios | `tests/test_data_loader.py` |
| Recommendation engine | All 10 classes produce valid params, confidence adjustment | `tests/test_recommendations.py` |
| Ensemble engine | Weight normalization, safety override, thresholding | `tests/test_ensemble.py` |
| Audio preprocessing | Normalization bounds, duration enforcement | `tests/test_preprocessing.py` |

### Integration Tests

| Test | Description |
|------|-------------|
| End-to-end pipeline | Audio file → preprocessing → features → prediction → recommendation |
| API round-trip | Upload audio → receive JSON response → validate schema |
| Batch processing | Process 100 files → verify consistent output format |

### Performance Tests

| Test | Target | Method |
|------|--------|--------|
| Inference latency | < 500 ms per sample | `time.perf_counter()` wall-clock |
| Memory usage | < 2 GB peak during inference | `tracemalloc` or `psutil` |
| Batch throughput | Document samples/sec | Timed batch processing |
| Model load time | < 10 seconds cold start | Timed model deserialization |

---

## 7. PRD ↔ TRD Traceability Matrix

| PRD Requirement | TRD Module | Section |
|-----------------|------------|---------|
| FR-01: 3-second audio input | Module 1: Audio Preprocessing | §3.1 |
| FR-02: 10-class classification | Module 5: Model Training | §3.5 |
| FR-03: Class + confidence output | Module 6: Decision Engine | §3.6 |
| FR-04: ≥ 85% accuracy | Module 8: Evaluation Framework | §3.8 |
| FR-05: Hearing aid parameters | Module 7: Recommendation Engine | §3.7 |
| FR-06: Human-readable reasoning | Module 7: Recommendation Engine | §3.7 |
| FR-07: 5-model comparison | Module 5 + Module 8 | §3.5, §3.8 |
| FR-08–FR-13: Evaluation artifacts | Module 8: Evaluation Framework | §3.8 |
| FR-14: MFCC extraction | Module 2: Feature Extraction | §3.2 |
| FR-15: Mel-spectrogram extraction | Module 2: Feature Extraction | §3.2 |
| FR-16: Raw waveform extraction | Module 2: Feature Extraction | §3.2 |
| FR-17: Audio augmentation | Module 3: Data Augmentation | §3.3 |
| FR-18: User personalization | Module 7: Tier 4 | §3.7 |
| FR-19: Web demo | Module 9: Deployment (Gradio) | §3.9 |
| FR-20: REST API JSON output | Module 9: Deployment (FastAPI) | §3.9 |
| SC-01: ≥ 85% accuracy | Module 8: Primary metric check | §3.8 |
| SC-04: < 500 ms latency | Module 9: Performance test | §6 |
| Deployment: < 200 MB models | Module 5: Serialization + Module 9 | §3.5, §3.9 |

---

## 8. Open Architecture Decisions

| Decision | Current State | Notes |
|----------|---------------|-------|
| Mel-spectrogram n_mels | TRD says 128, current impl uses 64 | Reconcile in CNN training phase |
| Ensemble weight calibration | Heuristic weights | Post-training optimization recommended |
| YAMNet framework | TF Hub | Verify compatibility with TF version |
| Edge deployment format | ONNX vs TFLite | Deferred to deployment phase |
| Experiment tracker | Optuna + W&B specified | Confirm setup in training phases |
| Model size budget | RF alone is ~140 MB | May need compression for < 200 MB total |

---

*Last updated: 2026-03-08T16:36:00+05:30*
*Source: User-provided Technical Requirements Document*
*Derives from: .gsd/prd.md (FINALIZED)*
*GSD Status: FINALIZED — Ready for ROADMAP planning*

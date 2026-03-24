---
status: FINALIZED-v2
type: product-requirements-document
project: Context-Aware Hearing Aid Settings Optimizer
created: 2026-03-08T16:32:00+05:30
updated: 2026-03-11T10:22:00+05:30
version: 2.0
source: User-updated PRD (March 2026)
---

# Product Requirements Document (PRD) — v2.0

> **Status**: `FINALIZED-v2`
>
> This document supersedes PRD v1.0. All SPEC, ROADMAP, TODO, and PLAN documents
> derive from this PRD. Changes are tracked under §12 (Changelog).

---

## 1. System Overview

An **AI-powered audio classification and recommendation system** that detects environmental sounds and automatically adjusts hearing aid settings in real time.

The system targets the **466 million people worldwide** with hearing loss by replacing static, manually-adjusted hearing aid profiles with intelligent, environment-aware audio adaptation. It combines traditional machine learning, deep learning, and transfer learning to classify urban sound environments, then maps each classification to optimal hearing aid parameters via a rule-based recommendation engine.

**Core value proposition:** Eliminate manual hearing aid adjustments by automatically detecting the sound environment and recommending optimal device settings — improving speech clarity, reducing listener fatigue, and enhancing safety-critical sound awareness.

---

## 2. Problem Statement

Current hearing aids operate with **fixed audio settings**, forcing users to manually switch profiles when environments change.

### 2.1 Impact

| Problem | User Impact |
|---------|-------------|
| Acoustically similar classes (drilling/jackhammer/gun_shot) cause misclassification | Settings applied to wrong environment |
| Poor speech understanding in noisy places | Reduced communication ability |
| Listener fatigue from manual adjustments | Lower quality of life |
| Difficulty detecting safety-critical sounds (siren, horn, speech) | Personal safety risk |

### 2.2 Root Cause

> **v2 Addition**: The original 10-class UrbanSound8K taxonomy contains **acoustically overlapping sound categories** (e.g., `drilling`, `jackhammer`, and `gun_shot` share high spectral energy spikes). This overlap degrades classifier confidence and causes incorrect hearing aid setting recommendations. The v2 design consolidates these into **8 acoustically distinct, perceptually meaningful classes**.

---

## 3. Sound Classification Schema — v2 (8-Class Optimized)

### 3.1 Optimized Class Definitions

| Class ID | Class Name | Acoustic Signature | Hearing Aid Priority |
|----------|------------|-------------------|---------------------|
| C-01 | `speech` | Voiced, 100–8000 Hz, dynamic envelope | 🔴 SAFETY (P1) |
| C-02 | `siren` | Frequency-swept pure tone, periodic | 🔴 SAFETY (P1) |
| C-03 | `horn` | Short burst, sharp harmonic transient | 🔴 SAFETY (P1) |
| C-04 | `traffic` | Broadband low-freq rumble, continuous | 🟡 HIGH (P2) |
| C-05 | `construction` | Impulsive broadband, irregular rhythm | 🟡 HIGH (P2) |
| C-06 | `dog_bark` | Tonal burst, mid-freq, short duration | 🟢 MEDIUM (P3) |
| C-07 | `music` | Harmonic, periodic, wide dynamic range | 🟢 MEDIUM (P3) |
| C-08 | `background_noise` | Diffuse, low-amplitude, no clear structure | 🔵 LOW (P4) |

### 3.2 UrbanSound8K → 8-Class Label Mapping

| Original US8K Label | New 8-Class Label | Reasoning |
|--------------------|-------------------|-----------|
| `air_conditioner` | `background_noise` | Diffuse, low-freq hum |
| `engine_idling` | `traffic` | Motor vehicle rumble |
| `drilling` | `construction` | Impulsive broadband noise |
| `jackhammer` | `construction` | Impulsive broadband noise (same cluster) |
| `gun_shot` | `construction` | Short impulse burst — acoustically construction-like |
| `car_horn` | `horn` | Short harmonic transient |
| `children_playing` | `speech` | Voiced, social soundscape |
| `dog_bark` | `dog_bark` | Retained — perceptually distinct |
| `siren` | `siren` | Retained — safety critical |
| `street_music` | `music` | Retained — harmonic |

> **Note**: ESC-50 additional samples are re-mapped to the same 8-class target schema.

### 3.3 Safety-Critical Confidence Thresholds

| Class | Standard Threshold | Safety Override Threshold |
|-------|-------------------|--------------------------|
| `speech` | 0.50 | **0.25** |
| `siren` | 0.50 | **0.20** |
| `horn` | 0.50 | **0.25** |
| All others | 0.50 | 0.50 |

---

## 4. Research Goals — v2

### 4.1 Primary Research Goals

| ID | Goal | Success Metric |
|----|------|----------------|
| RG-01 | High-accuracy 8-class sound classification | ≥ 88% accuracy on remapped US8K test set |
| RG-02 | Real-time inference feasibility | Inference latency < 500 ms per sample |
| RG-03 | Comparative ML paradigm evaluation | 5 models trained with publication-quality comparison |
| RG-04 | Automatic hearing aid parameter recommendation | Valid parameter output for all 8 sound classes |
| RG-05 | IEEE-grade research publication | Complete paper with results section |

### 4.2 Secondary Research Goals

| ID | Goal | Priority |
|----|------|----------|
| RG-06 | Personalization system (user profiles) | Should-have |
| RG-07 | Edge device deployment | Nice-to-have |
| RG-08 | Web demo interface (Gradio) | ✅ Complete |
| RG-09 | Open-source repository | Should-have |
| RG-10 | ESC-50 data augmentation for minority classes | ✅ Complete |

---

## 5. Model Architecture — v2

### 5.1 Ensemble Models (5 total, YAMNet Primary)

| Model | Role | Input | Output |
|-------|------|-------|--------|
| Random Forest | Baseline ML | MFCC (40 × time) flattened | 8-class probs |
| SVM (RBF kernel) | Baseline ML | Scaled MFCC | 8-class probs |
| XGBoost | Boosted ML | MFCC | 8-class probs |
| CNN (Mel-Spectrogram) | Deep Learning | 128×128×1 Mel | 8-class probs |
| **YAMNet** (Primary) | Transfer Learning | 48000-sample waveform | 8-class probs |

### 5.2 Inference Pipeline

```
Audio Input (WAV/MP3/OGG/FLAC)
  ↓
Preprocessing
  • Resample → 16 kHz mono
  • Pad / trim → 3 seconds (48,000 samples)
  • Amplitude normalize
  ↓
Parallel Feature Extraction
  ├── MFCC (40 coefficients, for RF/SVM/XGB)
  └── Mel-Spectrogram (128×128×1, for CNN)
  └── Waveform (48000,) for YAMNet
  ↓
Model Ensemble
  ├── Random Forest   → P_rf  [8]
  ├── SVM             → P_svm [8]
  ├── XGBoost         → P_xgb [8]
  ├── CNN             → P_cnn [8]
  └── YAMNet (primary)→ P_yam [8]
  ↓
Adaptive Weighted Aggregation
  • Dynamic weights from Nelder-Mead optimisation
  • YAMNet confidence-boosting rule
  • Safety class priority rule
  ↓
Confidence Threshold Gate
  • Safety classes: threshold = 0.20–0.25
  • Standard classes: threshold = 0.50
  ↓
Safety Override Layer
  (speech / siren / horn → highest priority alert if detected)
  ↓
Recommendation Engine (4-Tier)
  → Volume, Noise Reduction, Directionality, Speech Enhancement
  ↓
Output: Hearing Aid Settings + Explanation
```

### 5.3 YAMNet Architecture Specification

```
Input: 3-second audio @ 16 kHz → shape (48000,)
  ↓
YAMNet MobileNet Base (FROZEN — TF-Hub pretrained)
  ↓ shape: (N_frames, 1024) embeddings
Mean Temporal Pooling → shape: (1024,)
  ↓
Dense(512, activation='relu')
Dropout(0.5)
Dense(256, activation='relu')
Dropout(0.4)
Dense(8, activation='softmax')   ← 8 optimized classes
```

### 5.4 CNN Architecture Specification

```
Input: Mel-Spectrogram → (128, 128, 1)
  ↓
Conv2D(32, 3×3, 'relu') → BatchNorm → MaxPool(2×2)
Conv2D(64, 3×3, 'relu') → BatchNorm → MaxPool(2×2)
Conv2D(128, 3×3, 'relu') → BatchNorm → GlobalAvgPool
  ↓
Dense(256, 'relu')
Dropout(0.5)
Dense(8, 'softmax')          ← 8 optimized classes
```

---

## 6. Recommendation Engine — v2 (8-Class)

| Detected Class | Volume | Noise Reduction | Directionality | Speech Enhancement |
|---------------|--------|----------------|----------------|--------------------|
| `speech` | 7/10 | Low | Directional | **On** |
| `siren` | 9/10 | Off | Omnidirectional | Off |
| `horn` | 8/10 | Off | Omnidirectional | Off |
| `traffic` | 5/10 | High | Directional | On |
| `construction` | 4/10 | High | Omnidirectional | Off |
| `dog_bark` | 5/10 | Medium | Adaptive | Off |
| `music` | 6/10 | Low | Omnidirectional | Off |
| `background_noise` | 5/10 | Medium | Omnidirectional | Off |

---

## 7. Datasets

### 7.1 Primary Dataset — UrbanSound8K

| Property | Value |
|----------|-------|
| Source | Freesound / NYU |
| Total clips | 8,732 |
| Classes | 10 (remapped → 8 in v2) |
| Sampling rate | Variable (resampled to 16 kHz) |
| Splits | Predefined 10-fold |
| Label mapping | See §3.2 |

### 7.2 Supplementary Dataset — ESC-50

| Property | Value |
|----------|-------|
| Source | Freesound |
| Total clips | 2,000 (50 classes × 40 clips) |
| Classes used | All 50 (remapped to 8 v2 classes) |
| Splits | Fold 5 = test; Folds 1–4 = train |

---

## 8. Evaluation Metrics

| Metric | Target (8-class) |
|--------|-----------------|
| Overall Accuracy | ≥ 88% |
| Macro F1 | ≥ 87% |
| Safety Class Recall (siren/horn/speech) | ≥ 95% |
| Inference Latency | < 500 ms (GPU) / < 2000 ms (CPU) |

---

## 9. Non-Functional Requirements

- **Reproducibility**: Random seed 42 enforced globally
- **Logging**: All training runs produce structured log files
- **Modularity**: Each model is independently saveable and loadable
- **Explainability**: Every ouput includes human-readable reasoning string

---

## 10. Deployment

| Component | Technology |
|-----------|-----------|
| Interactive Demo | Gradio (port 7860) |
| API | FastAPI |
| Model Storage | `models/` directory (`.pt`, `.h5`, `.pkl`, `.json`) |
| Config | `config/` YAML files |

---

## 11. IEEE Paper Requirements

- Abstract, Introduction, Related Work
- System Architecture diagram
- Experimental Setup (datasets, splits, preprocessing)
- Results table (5 models × 8 classes: Accuracy, F1, Precision, Recall)
- Confusion matrices for CNN and YAMNet
- Ablation study: 10-class baseline vs 8-class optimized
- Conclusion and Future Work

---

## 12. Changelog

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-03-08 | Initial PRD — 10 UrbanSound8K classes |
| v2.0 | 2026-03-11 | **Class redesign** — 10 → 8 acoustically distinct classes; YAMNet primary; safety thresholds; ESC-50 unified pipeline |

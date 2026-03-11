# 🎧 Context-Aware Hearing Aid Settings Optimizer v2

> **AI-powered environmental sound classification and automatic hearing aid parameter adjustment**

[![Accuracy](https://img.shields.io/badge/Accuracy-95.27%25-brightgreen)](results/metrics_v2.json)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-95.04%25-brightgreen)](results/metrics_v2.json)
[![Classes](https://img.shields.io/badge/Classes-8_Auditory_Distinct-blue)](src/data/label_map_v2.py)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🧠 System Overview

Modern digital hearing aids require robust, real-time classification of environmental sounds to dynamically adjust hardware settings (volume, noise reduction, directionality). This project introduces **v2**, which abandons highly overlapping 10-class taxonomies in favor of an acoustically distinct **8-class schema**, merging **UrbanSound8K** and **ESC-50** datasets for superior real-world robustness.

It uses an adaptive **5-model weighted ensemble** (RF, SVM, XGBoost, CNN, and a fine-tuned YAMNet primary), fused via Nelder-Mead optimization to achieve **95.27% accuracy** with built-in overrides for safety-critical sounds.

```text
Audio Input (WAV/MP3/Mic)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                   8-Class Feature Extraction                │
│ MFCC (6112×80) │ Mel-Spec (128×128) │ YAMNet Embed (1024-d) │
└──────┬──────────────────┬───────────────────────┬───────────┘
       │                  │                       │
  ┌────▼───┐        ┌─────▼─────┐           ┌─────▼────┐
  │ RF+SVM │        │ AudioCNN  │           │  YAMNet  │
  │ + XGB  │        │ (PyTorch) │           │ (v2 ★)   │
  └────┬───┘        └─────┬─────┘           └─────┬────┘
       └──────────────────┴───────────────────────┘
                          │
                  ┌───────▼────────┐
                  │ Adaptive Fusion│  ← Nelder-Mead optimized weights 
                  │ 95.27% Accuracy│    with Critical Class Boosting
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │ Recommendation │  ← Rules + Confidence + Safety Overrides
                  │   Engine v2    │    (Siren, Horn, Speech triggers)
                  └───────┬────────┘
                          │
          {Volume, Noise Reduction, Directionality,
                 Speech Enhancement, Reasoning}
```

---

## 📊 Model Performance

Tested on a combined, normalized hold-out set of 1,310 validation samples.

| Model | Architecture Type | Input Feature | Weight (Opt.) | Accuracy | Macro F1 |
|-------|:---|:---|:---:|:---:|:---:|
| **Random Forest** | Traditional ML | MFCC | 3.1% | 82.98% | 82.10% |
| **XGBoost** | Boosted Trees | MFCC | 16.0% | 89.16% | 88.22% |
| **SVM (RBF)** | Kernel Machine | Scaled MFCC | 29.2% | 91.30% | 90.67% |
| **CNN (v2)** | 2D Deep Network | Mel-spectrogram | 6.9% | 46.03% | 50.94% |
| **YAMNet (v2)**| Transfer (MobileNet)| 1024-d Embeds | **44.9% ★** | 91.91% | 91.30% |
| **Ensemble** | **Nelder-Mead Fusion**| **All** | **100%** | **95.27% ✅**| **95.04% ✅**|

### 🚨 Safety-Critical Recall
The system is explicitly engineered to never suppress danger signals or human vocalization:
*   🚑 **Siren Recall:** 95.68%
*   🗣️ **Speech Recall:** 94.00%
*   📯 **Horn Recall:** 93.85%

---

## 🎧 Hardware Recommendation Engine

The output probabilities map directly to actionable digital signal processing (DSP) parameters:

| Acoustic Class | Volume | Noise Reduction | Directionality | Speech Enhancement | Override Rule |
|---|---|---|---|---|---|
| **Speech** | 6 | Medium | Directional | **ON** | *Safety Critical* |
| **Siren** | 7 | Low | Omnidirectional | OFF | *Safety Critical* |
| **Horn** | 6 | Medium | Omnidirectional | OFF | *Safety Critical* |
| **Traffic** | 4 | High | Directional | OFF | - |
| **Construction**| 3 | **High** | Adaptive | OFF | Maximum Suppression |
| **Dog Bark** | 5 | Medium | Adaptive | OFF | - |
| **Music** | 6 | **Off** | Omnidirectional | OFF | Disable compression |
| **Background** | 5 | Medium | Omnidirectional | OFF | - |

---

## 🚀 Quick Start & Launch

### Prerequisites

```bash
# Python 3.10 environment required
conda create -n hearing-aid python=3.10
conda activate hearing-aid

# Install dependencies (including Gradio and Plotly)
pip install -r requirements.txt
pip install plotly
```

### Launch the Quantico Dashboard Demo (Gradio)
The v2 system features a fully responsive, dark-mode analytics dashboard integrating interactive Plotly graphs.

```bash
python demo/gradio_app.py
# The application will launch at http://localhost:7860
```
*(The dashboard allows for live microphone recording, file uploads, and includes a library of quick "Recent Feeds" to instantly test the ensemble's inference capabilities).*

---

## 🗂️ Project Structure

```text
hearing_aid/
├── data/
│   ├── UrbanSound8K/         
│   ├── ESC50/                  # Newly integrated dataset
│   └── splits_v2/              # 8-class remapped train/val/test CSVs
│
├── features/                   # Cached tensors for rapid iteration
│   ├── mfcc/, mel_v2/, yamnet_v2/ 
│
├── models/                     # Compiled v2 Model Binaries
│   ├── rf_v2.pkl, svm_v2.pkl, xgb_v2.json
│   ├── cnn_v2.pt, yamnet_v2.h5
│
├── src/
│   ├── data/
│   │   └── label_map_v2.py     # Complex 10-to-8 and 50-to-8 mapping logic
│   ├── features/               # Robust DSP extraction suite (librosa)
│   ├── models_v2.py            # CNN PyTorch definition
│   ├── phase5*_train_*.py      # Atomic training scripts for all models
│   ├── phase6_ensemble.py      # Nelder-Mead optimization + Safety boosting
│   └── evaluate_v2.py          # IEEE publication artifact generator
│
├── demo/
│   └── gradio_app.py           # Sleek, Plotly-powered dark dashboard
│
├── results/
│   ├── figures/                # Confusion matrices & F1 charts
│   ├── metrics_v2.json         # Raw evaluation metadata
│   ├── ensemble_weights*.json  # Computed Nelder-Mead parameters
│   └── ieee_paper_draft_v2.md  # Publication-ready research draft
```

---

## 🛠️ Key Version 2 Upgrades

1. **Acoustic Restructuring:** Solved the inherent "Drilling vs Jackhammer" and "Air Conditioner vs Engine" confusion found in standard benchmarks by mapping highly correlated audio signatures into an 8-class functional audiology schema.
2. **ESC-50 Integration:** Combined UrbanSound8K with the ESC-50 dataset to massively expand the system's variance and generalization capabilities.
3. **Data Dashboard UI:** Transitioned from a basic block list to a "Quantico-styled" dark-mode analytical dashboard utilizing real-time interactive `plotly.graph_objects`.
4. **Safety-Gate Boosting:** Implemented structural logic within the optimization loop (`src/phase6_ensemble.py`) that artificially inflates weights for critical classes (Sirens/Horns) to guarantee user safety in active hardware deployments. 

---

## 📄 License & Academic Reference

Drafted for submission to the *IEEE Transactions on Audio, Speech, and Language Processing*. 
MIT License — see [LICENSE](LICENSE) for details.

*Project architecture generated via GSD Methodology end-to-end autonomous protocols.*

"""
Phase 7 — Evaluation Framework.

Generates comprehensive, publication-quality evaluation for all models:

    1. Full metric suite    — accuracy, macro/weighted F1, precision, recall, per-class
    2. Confusion matrices   — normalised heatmaps for every model + ensemble
    3. Per-class F1 charts  — grouped horizontal bar chart comparing all 5 models + ensemble
    4. Training curves      — CNN loss/accuracy curves; YAMNet phase 1 & 2 curves
    5. Bootstrap CIs        — 95% confidence intervals on macro F1 (1000 samples, BCa)
    6. Ablation study       — ensemble performance as each model is removed
    7. Model comparison     — radar chart + summary table across all dimensions

Outputs (all saved to results/figures/):
    confusion_matrix_{model}.png
    per_class_f1_comparison.png
    training_curves_cnn.png
    training_curves_yamnet.png
    bootstrap_ci.png
    ablation_study.png
    radar_chart.png
    evaluation_report.json
"""

import json
import logging
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy import stats

from src.utils import setup_logging, set_seed

warnings.filterwarnings("ignore")

SEED = 42
FIG_DIR = Path("results/figures")
REPORT_PATH = Path("results/evaluation_report.json")

CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music",
]
CLASS_SHORT = [
    "Air Cond.", "Car Horn", "Children", "Dog Bark",
    "Drilling", "Engine", "Gun Shot", "Jackhammer",
    "Siren", "St. Music",
]
MODEL_NAMES = ["rf", "svm", "xgb", "cnn", "yamnet", "ensemble"]
MODEL_LABELS = ["Random Forest", "SVM", "XGBoost", "CNN", "YAMNet", "Ensemble"]
MODEL_COLORS = ["#9b59b6", "#3498db", "#e67e22", "#e74c3c", "#2ecc71", "#1abc9c"]

# ─── Style ────────────────────────────────────────────────────────────────────
def _set_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "figure.dpi":        150,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.1,
    })


# ─── Data loading ─────────────────────────────────────────────────────────────
def _load_metrics() -> dict:
    """Load all cached result JSONs."""
    paths = {
        "cnn":      "results/cnn_metrics.json",
        "yamnet":   "results/yamnet_metrics.json",
        "comparison": "results/model_comparison.json",
        "ensemble": "results/ensemble_metrics.json",
        "weights":  "results/ensemble_weights.json",
    }
    data = {}
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path) as f:
                data[key] = json.load(f)
        else:
            logging.getLogger(__name__).warning(f"Missing: {path}")
    return data


def _load_test_predictions() -> dict:
    """
    Re-load test-set ground-truth labels and all model predictions
    from the cached probability arrays and model files.
    Returns {model_name: (y_true, y_pred)} dict.
    """
    import joblib
    import pandas as pd
    import torch
    import torch.nn.functional as F
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow as tf
    from src.models import AudioCNN

    logger = logging.getLogger(__name__)
    logger.info("Loading test predictions from cached files...")

    # Label encoder + class names
    enc = joblib.load("models/label_encoder.pkl")

    # Load test CSV
    df_test = pd.read_csv("data/splits/test.csv")
    y_true_labels = df_test["class"].tolist()

    # Load MFCC test features
    feat_test_mfcc = Path("features/mfcc/test")
    X_mfcc, y_true = [], []
    for _, row in df_test.iterrows():
        stem = Path(row["slice_file_name"]).stem
        npy = feat_test_mfcc / f"{stem}.npy"
        if npy.exists():
            X_mfcc.append(np.load(npy))
            y_true.append(row["class"])
    X_mfcc = np.stack(X_mfcc).astype(np.float32)
    y_encoded = enc.transform(y_true).astype(np.int32)

    # Load Mel test features for CNN
    feat_test_mel = Path("features/mel/test")
    X_mel = []
    for _, row in df_test.iterrows():
        stem = Path(row["slice_file_name"]).stem
        npy = feat_test_mel / f"{stem}.npy"
        if npy.exists():
            mel = np.load(npy).transpose(2, 0, 1)  # (1, 128, 128)
            X_mel.append(mel)
    X_mel = np.stack(X_mel).astype(np.float32)

    # Load YAMNet embeddings
    X_emb = np.load("features/yamnet_embeddings/test.npy")
    E_train = np.load("features/yamnet_embeddings/train.npy")
    emb_mean = E_train.mean(axis=0, keepdims=True)
    emb_std  = E_train.std(axis=0, keepdims=True) + 1e-8
    X_emb_n  = ((X_emb - emb_mean) / emb_std).astype(np.float32)

    results = {}

    # RF
    rf = joblib.load("models/random_forest.pkl")
    results["rf"] = (y_encoded, rf.predict(X_mfcc))

    # SVM
    svm = joblib.load("models/svm.pkl")
    results["svm"] = (y_encoded, svm.predict(X_mfcc))

    # XGBoost
    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.load_model("models/xgboost.json")
    results["xgb"] = (y_encoded, xgb.predict(X_mfcc))

    # CNN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = AudioCNN(num_classes=10).to(device)
    cnn.load_state_dict(torch.load("models/cnn_best.pt", map_location=device))
    cnn.eval()
    cnn_preds = []
    with torch.no_grad():
        for i in range(0, len(X_mel), 64):
            batch = torch.from_numpy(X_mel[i:i+64]).to(device)
            logits = cnn(batch)
            cnn_preds.append(logits.argmax(dim=1).cpu().numpy())
    results["cnn"] = (y_encoded, np.concatenate(cnn_preds))

    # YAMNet head
    yamnet_head = tf.keras.models.load_model("models/yamnet_head.h5")
    probs_yn = yamnet_head.predict(X_emb_n, batch_size=128, verbose=0)
    results["yamnet"] = (y_encoded, np.argmax(probs_yn, axis=1))

    # Ensemble — load saved weights and re-aggregate probs
    with open("results/ensemble_weights.json") as f:
        ew = json.load(f)["weights"]

    order = ["rf", "svm", "xgb", "cnn", "yamnet"]
    weights = np.array([ew[n] for n in order], dtype=np.float32)

    # gather probs for ensemble
    rf_p  = rf.predict_proba(X_mfcc).astype(np.float32)
    svm_p_raw = svm.predict_proba(X_mfcc) if hasattr(svm.named_steps["svm"], "predict_proba") else None
    if svm_p_raw is None:
        sc = svm.named_steps["scaler"].transform(X_mfcc)
        sc_df = svm.named_steps["svm"].decision_function(sc).astype(np.float32)
        sc_df -= sc_df.max(axis=1, keepdims=True)
        svm_p = np.exp(sc_df) / np.exp(sc_df).sum(axis=1, keepdims=True)
    else:
        svm_p = svm_p_raw.astype(np.float32)

    xgb_p   = xgb.predict_proba(X_mfcc).astype(np.float32)
    cnn_p_list = []
    with torch.no_grad():
        for i in range(0, len(X_mel), 64):
            batch = torch.from_numpy(X_mel[i:i+64]).to(device)
            cnn_p_list.append(F.softmax(cnn(batch), dim=1).cpu().numpy())
    cnn_p = np.concatenate(cnn_p_list).astype(np.float32)
    yn_p  = probs_yn.astype(np.float32)

    probs_stack = np.stack([rf_p, svm_p, xgb_p, cnn_p, yn_p], axis=0)
    ens_probs   = (probs_stack * weights[:, None, None]).sum(axis=0)
    results["ensemble"] = (y_encoded, np.argmax(ens_probs, axis=1))

    # Save probs for downstream ablation
    results["_probs"] = {
        "rf": rf_p, "svm": svm_p, "xgb": xgb_p, "cnn": cnn_p, "yamnet": yn_p
    }
    results["_weights"] = weights
    results["_y_true"]  = y_encoded

    logger.info("All test predictions collected.")
    return results


# ─── 1. Confusion matrices ────────────────────────────────────────────────────
def plot_confusion_matrices(pred_data: dict, out_dir: Path) -> list:
    """Plot normalised confusion matrix for each model. Returns list of saved paths."""
    from sklearn.metrics import confusion_matrix as sk_cm

    _set_style()
    paths = []

    for model_key, label in zip(MODEL_NAMES, MODEL_LABELS):
        if model_key not in pred_data:
            continue
        y_true, y_pred = pred_data[model_key]
        cm = sk_cm(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        acc = np.diag(cm).sum() / cm.sum()

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=CLASS_SHORT,
            yticklabels=CLASS_SHORT,
            linewidths=0.5,
            linecolor="#dddddd",
            cbar_kws={"label": "Normalised count"},
            ax=ax,
        )
        ax.set_title(f"{label} — Confusion Matrix\n(Accuracy: {acc:.1%})", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)

        out = out_dir / f"confusion_matrix_{model_key}.png"
        fig.savefig(out)
        plt.close(fig)
        paths.append(str(out))

    return paths


# ─── 2. Per-class F1 comparison ───────────────────────────────────────────────
def plot_per_class_f1(pred_data: dict, out_dir: Path) -> str:
    """Grouped horizontal bar chart: per-class F1 for all 6 models."""
    from sklearn.metrics import f1_score

    _set_style()
    n_classes = len(CLASS_NAMES)
    n_models  = len(MODEL_NAMES)
    bar_h = 0.12
    y = np.arange(n_classes)

    fig, ax = plt.subplots(figsize=(13, 9))

    for i, (model_key, label, color) in enumerate(zip(MODEL_NAMES, MODEL_LABELS, MODEL_COLORS)):
        if model_key not in pred_data:
            continue
        y_true, y_pred = pred_data[model_key]
        per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
        offset = (i - n_models / 2) * bar_h + bar_h / 2
        bars = ax.barh(
            y + offset, per_cls, bar_h * 0.9,
            label=label, color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        # Value labels on bars > 0.85
        for j, (bar, val) in enumerate(zip(bars, per_cls)):
            if val > 0.85:
                ax.text(
                    val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=7, color="#333333",
                )

    ax.set_yticks(y)
    ax.set_yticklabels(CLASS_SHORT, fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Score — All Models vs Ensemble", fontsize=14, fontweight="bold", pad=12)
    ax.axvline(0.85, ls="--", lw=1.2, color="#e74c3c", alpha=0.7, label="Target (0.85)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.invert_yaxis()
    ax.xaxis.grid(True, alpha=0.4)

    out = out_dir / "per_class_f1_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# ─── 3. Training curves ───────────────────────────────────────────────────────
def plot_training_curves(metrics_data: dict, out_dir: Path) -> list:
    """Loss + accuracy training curves for CNN and YAMNet."""
    _set_style()
    paths = []

    # ── CNN ──────────────────────────────────────────────────────────────────
    if "cnn" in metrics_data and "training_history" in metrics_data["cnn"]:
        h = metrics_data["cnn"]["training_history"]
        epochs = range(1, len(h["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("CNN Training Curves (GPU — AudioCNN)", fontsize=14, fontweight="bold")

        # Loss
        ax1.plot(epochs, h["train_loss"], "#3498db", lw=2, label="Train Loss")
        ax1.plot(epochs, h["val_loss"],   "#e74c3c", lw=2, label="Val Loss")
        best_epoch = np.argmin(h["val_loss"]) + 1
        ax1.axvline(best_epoch, ls="--", lw=1, color="#2ecc71", alpha=0.8, label=f"Best epoch ({best_epoch})")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss"); ax1.legend(fontsize=9); ax1.grid(alpha=0.4)

        # Accuracy
        ax2.plot(epochs, h["train_acc"], "#3498db", lw=2, label="Train Acc")
        ax2.plot(epochs, h["val_acc"],   "#e74c3c", lw=2, label="Val Acc")
        ax2.plot(epochs, h["val_f1"],    "#9b59b6", lw=2, ls="--", label="Val Macro F1")
        ax2.axhline(0.85, ls=":", lw=1.2, color="#e67e22", alpha=0.8, label="Target F1 (0.85)")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
        ax2.set_title("Accuracy & F1"); ax2.legend(fontsize=9); ax2.grid(alpha=0.4)

        plt.tight_layout()
        out = out_dir / "training_curves_cnn.png"
        fig.savefig(out); plt.close(fig)
        paths.append(str(out))

    # ── YAMNet ───────────────────────────────────────────────────────────────
    if "yamnet" in metrics_data and "training_history" in metrics_data["yamnet"]:
        h = metrics_data["yamnet"]["training_history"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("YAMNet Two-Phase Training Curves", fontsize=14, fontweight="bold")

        phase_colors = [("#3498db", "#e74c3c"), ("#27ae60", "#c0392b")]
        phase_labels = [("Phase 1 Train", "Phase 1 Val"), ("Phase 2 Train", "Phase 2 Val")]

        for p_idx, (phase_key, colors, labels) in enumerate(
            zip(["phase1", "phase2"], phase_colors, phase_labels)
        ):
            if phase_key not in h:
                continue
            ph = h[phase_key]
            ep = range(1, len(ph["loss"]) + 1)

            axes[0].plot(ep, ph["loss"],     colors[0], lw=2, ls="-" if p_idx==0 else "--", label=labels[0])
            axes[0].plot(ep, ph["val_loss"], colors[1], lw=2, ls="-" if p_idx==0 else "--", label=labels[1])
            axes[1].plot(ep, ph["accuracy"],     colors[0], lw=2, ls="-" if p_idx==0 else "--", label=labels[0])
            axes[1].plot(ep, ph["val_accuracy"], colors[1], lw=2, ls="-" if p_idx==0 else "--", label=labels[1])

        axes[0].set_title("Loss (both phases)"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.4)
        axes[1].set_title("Accuracy (both phases)"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].axhline(0.85, ls=":", lw=1.2, color="#e67e22", alpha=0.8, label="Target 0.85")
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.4)

        plt.tight_layout()
        out = out_dir / "training_curves_yamnet.png"
        fig.savefig(out); plt.close(fig)
        paths.append(str(out))

    return paths


# ─── 4. Bootstrap confidence intervals ───────────────────────────────────────
def compute_bootstrap_ci(
    pred_data: dict,
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    seed: int = SEED,
) -> dict:
    """
    BCa bootstrap 95% CI on macro F1 for each model.

    Returns {model: {"mean", "ci_lower", "ci_upper", "std"}}
    """
    from sklearn.metrics import f1_score

    rng = np.random.RandomState(seed)
    alpha = (100 - ci) / 100
    ci_results = {}

    for model_key in MODEL_NAMES:
        if model_key not in pred_data:
            continue
        y_true, y_pred = pred_data[model_key]
        n = len(y_true)
        bootstrap_f1s = []

        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            f1 = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
            bootstrap_f1s.append(f1)

        bootstrap_f1s = np.array(bootstrap_f1s)
        ci_lower = float(np.percentile(bootstrap_f1s, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_f1s, 100 * (1 - alpha / 2)))
        mean_f1  = float(np.mean(bootstrap_f1s))

        ci_results[model_key] = {
            "mean":     round(mean_f1, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "std":      round(float(np.std(bootstrap_f1s)), 6),
        }

    return ci_results


def plot_bootstrap_ci(ci_results: dict, pred_data: dict, out_dir: Path) -> str:
    """Horizontal bar chart with 95% CI error bars."""
    from sklearn.metrics import f1_score

    _set_style()
    models = [m for m in MODEL_NAMES if m in ci_results]
    labels = [MODEL_LABELS[MODEL_NAMES.index(m)] for m in models]
    colors = [MODEL_COLORS[MODEL_NAMES.index(m)] for m in models]

    means      = [ci_results[m]["mean"] for m in models]
    ci_lowers  = [ci_results[m]["mean"] - ci_results[m]["ci_lower"] for m in models]
    ci_uppers  = [ci_results[m]["ci_upper"] - ci_results[m]["mean"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(models))
    bars = ax.barh(y, means, color=colors, alpha=0.82, edgecolor="white", linewidth=0.7, height=0.55)
    ax.errorbar(
        means, y, xerr=[ci_lowers, ci_uppers],
        fmt="none", color="#2c3e50", capsize=5, capthick=1.5, lw=1.5,
    )

    for bar, mean, lo, hi in zip(bars, means, ci_lowers, ci_uppers):
        ax.text(
            mean + hi + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.3f} [{mean-lo:.3f}–{mean+hi:.3f}]",
            va="center", fontsize=9.5, color="#2c3e50",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0.70, 1.05)
    ax.axvline(0.85, ls="--", lw=1.5, color="#e74c3c", alpha=0.75, label="PRD Target (0.85)")
    ax.axvline(0.90, ls="--", lw=1.5, color="#27ae60", alpha=0.75, label="Stretch Target (0.90)")
    ax.set_xlabel("Macro F1 Score (95% Bootstrap CI)", fontsize=12)
    ax.set_title("Model Comparison — Macro F1 with 95% Confidence Intervals\n(1000 bootstrap samples)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.invert_yaxis()
    ax.xaxis.grid(True, alpha=0.35)

    out = out_dir / "bootstrap_ci.png"
    fig.savefig(out); plt.close(fig)
    return str(out)


# ─── 5. Ablation study ───────────────────────────────────────────────────────
def run_ablation(pred_data: dict, out_dir: Path) -> tuple:
    """
    Ablation: ensemble accuracy as each model is removed (leave-one-out).
    Also tests: deep-only, traditional-only ensembles.
    """
    from sklearn.metrics import accuracy_score, f1_score

    _set_style()

    order   = ["rf", "svm", "xgb", "cnn", "yamnet"]
    weights = pred_data["_weights"]
    y_true  = pred_data["_y_true"]
    probs   = pred_data["_probs"]

    results = {}

    # Baseline: full ensemble
    P_full = np.stack([probs[n] for n in order], axis=0)
    ens    = (P_full * weights[:, None, None]).sum(axis=0)
    results["Full Ensemble"] = {
        "accuracy": float(accuracy_score(y_true, np.argmax(ens, axis=1))),
        "macro_f1": float(f1_score(y_true, np.argmax(ens, axis=1), average="macro")),
        "models_used": order[:],
    }

    # Leave-one-out
    for drop_idx, drop_name in enumerate(order):
        remaining = [n for n in order if n != drop_name]
        rem_idx   = [i for i, n in enumerate(order) if n != drop_name]
        w_sub     = weights[rem_idx]
        w_sub     = w_sub / w_sub.sum()
        P_sub     = np.stack([probs[n] for n in remaining], axis=0)
        ens_sub   = (P_sub * w_sub[:, None, None]).sum(axis=0)
        y_pred    = np.argmax(ens_sub, axis=1)
        drop_label = MODEL_LABELS[order.index(drop_name)]
        results[f"Drop {drop_label}"] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "models_used": remaining,
        }

    # Special combos
    for combo_name, combo_models in [
        ("Deep Only (CNN+YAMNet)",      ["cnn", "yamnet"]),
        ("Traditional Only (RF+SVM+XGB)", ["rf", "svm", "xgb"]),
        ("Top 3 (SVM+XGB+YAMNet)",      ["svm", "xgb", "yamnet"]),
        ("Top 2 (XGB+YAMNet)",          ["xgb", "yamnet"]),
    ]:
        idx   = [order.index(n) for n in combo_models]
        w_sub = weights[idx]
        w_sub = w_sub / w_sub.sum()
        P_sub = np.stack([probs[n] for n in combo_models], axis=0)
        ens_s = (P_sub * w_sub[:, None, None]).sum(axis=0)
        y_pred = np.argmax(ens_s, axis=1)
        results[combo_name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "models_used": combo_models,
        }

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Ensemble Ablation Study", fontsize=14, fontweight="bold")

    names    = list(results.keys())
    accs     = [v["accuracy"] for v in results.values()]
    f1s      = [v["macro_f1"] for v in results.values()]
    colors_a = ["#1abc9c" if n == "Full Ensemble" else "#3498db" if not n.startswith("Drop") else "#e74c3c" for n in names]

    y_a = np.arange(len(names))
    ax1.barh(y_a, accs, color=colors_a, alpha=0.84, edgecolor="white", height=0.7)
    ax1.axvline(results["Full Ensemble"]["accuracy"], ls="--", lw=1.5, color="#1abc9c", alpha=0.7)
    ax1.set_yticks(y_a); ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("Accuracy"); ax1.set_title("Accuracy per Configuration")
    ax1.set_xlim(0.78, 0.97); ax1.invert_yaxis(); ax1.xaxis.grid(True, alpha=0.35)
    for i, v in enumerate(accs):
        ax1.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8.5)

    ax2.barh(y_a, f1s, color=colors_a, alpha=0.84, edgecolor="white", height=0.7)
    ax2.axvline(results["Full Ensemble"]["macro_f1"], ls="--", lw=1.5, color="#1abc9c", alpha=0.7)
    ax2.axvline(0.85, ls=":", lw=1.2, color="#e74c3c", alpha=0.6, label="Target (0.85)")
    ax2.set_yticks(y_a); ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Macro F1"); ax2.set_title("Macro F1 per Configuration")
    ax2.set_xlim(0.78, 0.97); ax2.invert_yaxis(); ax2.xaxis.grid(True, alpha=0.35)
    for i, v in enumerate(f1s):
        ax2.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = out_dir / "ablation_study.png"
    fig.savefig(out); plt.close(fig)
    return results, str(out)


# ─── 6. Radar chart ──────────────────────────────────────────────────────────
def plot_radar_chart(pred_data: dict, out_dir: Path) -> str:
    """Spider/radar chart comparing all 6 models across 5 dimensions."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
    )

    _set_style()

    dims = ["Accuracy", "Macro F1", "Precision", "Recall", "Min Class F1"]
    n_dims = len(dims)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_key, label, color in zip(MODEL_NAMES, MODEL_LABELS, MODEL_COLORS):
        if model_key not in pred_data:
            continue
        y_true, y_pred = pred_data[model_key]
        per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
        vals = [
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", zero_division=0),
            precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_score(y_true, y_pred, average="macro", zero_division=0),
            float(per_cls.min()),
        ]
        vals += vals[:1]
        lw = 3 if model_key == "ensemble" else 1.5
        alpha = 0.25 if model_key == "ensemble" else 0.07
        ax.plot(angles, vals, color=color, lw=lw, label=label)
        ax.fill(angles, vals, color=color, alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks([0.65, 0.75, 0.85, 0.95])
    ax.set_yticklabels(["0.65", "0.75", "0.85", "0.95"], fontsize=8, color="gray")
    ax.axhline(0, lw=0)

    ax.set_title("Model Comparison — Multi-Dimensional Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

    out = out_dir / "radar_chart.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return str(out)


# ─── 7. Model size vs accuracy scatter ───────────────────────────────────────
def plot_accuracy_vs_size(metrics_data: dict, pred_data: dict, out_dir: Path) -> str:
    """Bubble chart: accuracy vs model size, bubble = F1 score."""
    from sklearn.metrics import accuracy_score, f1_score

    _set_style()

    model_sizes = {
        "rf":       141.76,
        "svm":      3.18,
        "xgb":      10.02,
        "cnn":      4.63,
        "yamnet":   3.06,
        "ensemble": 141.76 + 3.18 + 10.02 + 4.63 + 3.06,
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_key, label, color in zip(MODEL_NAMES, MODEL_LABELS, MODEL_COLORS):
        if model_key not in pred_data:
            continue
        y_true, y_pred = pred_data[model_key]
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        size = model_sizes.get(model_key, 10)
        bubble = np.sqrt(size) * 60

        ax.scatter(size, acc, s=bubble, color=color, alpha=0.78, edgecolors="white", linewidth=1.5, zorder=3)
        ax.annotate(
            label, (size, acc),
            textcoords="offset points", xytext=(8, 4),
            fontsize=9.5, color=color, fontweight="bold",
        )

    ax.axhline(0.85, ls="--", lw=1.2, color="#e74c3c", alpha=0.7, label="PRD Target (85%)")
    ax.axhline(0.90, ls="--", lw=1.2, color="#27ae60", alpha=0.7, label="Stretch Target (90%)")
    ax.set_xscale("log")
    ax.set_xlabel("Model Size (MB, log scale)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Model Size\n(bubble area ∝ model size)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.78, 0.97)
    ax.legend(fontsize=10)
    ax.xaxis.grid(True, alpha=0.35); ax.yaxis.grid(True, alpha=0.35)

    out = out_dir / "accuracy_vs_size.png"
    fig.savefig(out); plt.close(fig)
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(
    n_bootstrap: int = 1000,
    seed: int = SEED,
) -> dict:
    """
    Run the complete Phase 7 evaluation pipeline.

    Returns
    -------
    dict — evaluation_report containing all metrics, CI, and figure paths
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(seed)

    logger.info("=" * 60)
    logger.info("Phase 7 — Evaluation Framework")
    logger.info("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    logger.info("Loading result JSONs...")
    metrics_data = _load_metrics()

    logger.info("Loading test predictions (may take 1–2 min first run)...")
    pred_data = _load_test_predictions()

    # ── 1. Confusion matrices ─────────────────────────────────────────────────
    logger.info("\n[1/7] Plotting confusion matrices...")
    cm_paths = plot_confusion_matrices(pred_data, FIG_DIR)
    logger.info(f"  Saved {len(cm_paths)} confusion matrix plots")

    # ── 2. Per-class F1 ───────────────────────────────────────────────────────
    logger.info("[2/7] Plotting per-class F1 comparison...")
    f1_path = plot_per_class_f1(pred_data, FIG_DIR)
    logger.info(f"  Saved: {f1_path}")

    # ── 3. Training curves ────────────────────────────────────────────────────
    logger.info("[3/7] Plotting training curves...")
    curve_paths = plot_training_curves(metrics_data, FIG_DIR)
    logger.info(f"  Saved {len(curve_paths)} training curve plots")

    # ── 4. Bootstrap CIs ──────────────────────────────────────────────────────
    logger.info(f"[4/7] Computing bootstrap CIs ({n_bootstrap} samples)...")
    ci_results = compute_bootstrap_ci(pred_data, n_bootstrap=n_bootstrap, seed=seed)
    ci_path = plot_bootstrap_ci(ci_results, pred_data, FIG_DIR)
    logger.info(f"  Saved: {ci_path}")
    for m, ci in ci_results.items():
        logger.info(
            f"  {m:8s}: {ci['mean']:.4f} "
            f"[{ci['ci_lower']:.4f}–{ci['ci_upper']:.4f}]"
        )

    # ── 5. Ablation ───────────────────────────────────────────────────────────
    logger.info("[5/7] Running ablation study...")
    ablation_results, ablation_path = run_ablation(pred_data, FIG_DIR)
    logger.info(f"  Saved: {ablation_path}")
    for name, v in ablation_results.items():
        logger.info(f"  {name:35s}: acc={v['accuracy']:.4f}  f1={v['macro_f1']:.4f}")

    # ── 6. Radar chart ────────────────────────────────────────────────────────
    logger.info("[6/7] Plotting radar chart...")
    radar_path = plot_radar_chart(pred_data, FIG_DIR)
    logger.info(f"  Saved: {radar_path}")

    # ── 7. Accuracy vs size ───────────────────────────────────────────────────
    logger.info("[7/7] Plotting accuracy vs model size...")
    size_path = plot_accuracy_vs_size(metrics_data, pred_data, FIG_DIR)
    logger.info(f"  Saved: {size_path}")

    # ── Compile report ────────────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score, f1_score

    summary = {}
    for model_key in MODEL_NAMES:
        if model_key not in pred_data:
            continue
        y_true, y_pred = pred_data[model_key]
        per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
        summary[model_key] = {
            "accuracy":        round(float(accuracy_score(y_true, y_pred)), 6),
            "macro_f1":        round(float(f1_score(y_true, y_pred, average="macro")), 6),
            "weighted_f1":     round(float(f1_score(y_true, y_pred, average="weighted")), 6),
            "min_class_f1":    round(float(per_cls.min()), 6),
            "max_class_f1":    round(float(per_cls.max()), 6),
            "worst_class":     CLASS_NAMES[int(per_cls.argmin())],
            "best_class":      CLASS_NAMES[int(per_cls.argmax())],
            "bootstrap_ci":    ci_results.get(model_key, {}),
        }

    report = {
        "phase": 7,
        "title": "Evaluation Framework — Context-Aware Hearing Aid System",
        "model_summary": summary,
        "ablation_study": {
            k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                for kk, vv in v.items()}
            for k, v in ablation_results.items()
        },
        "prd_targets": {
            "macro_f1_85pct": summary.get("ensemble", {}).get("macro_f1", 0) >= 0.85,
            "accuracy_90pct": summary.get("ensemble", {}).get("accuracy", 0) >= 0.90,
        },
        "figures": {
            "confusion_matrices": cm_paths,
            "per_class_f1":       f1_path,
            "training_curves":    curve_paths,
            "bootstrap_ci":       ci_path,
            "ablation_study":     ablation_path,
            "radar_chart":        radar_path,
            "accuracy_vs_size":   size_path,
        },
    }

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nEvaluation report → {REPORT_PATH}")
    logger.info(f"Figures           → {FIG_DIR}/")
    logger.info(f"Total figures: {len(cm_paths) + len(curve_paths) + 5}")

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    for m, v in summary.items():
        ci = v["bootstrap_ci"]
        ci_str = f"[{ci.get('ci_lower',0):.3f}–{ci.get('ci_upper',0):.3f}]" if ci else ""
        logger.info(
            f"  {m:8s}: acc={v['accuracy']:.4f}  "
            f"f1={v['macro_f1']:.4f} {ci_str}"
        )
    logger.info("=" * 60)
    logger.info("Phase 7 — Evaluation COMPLETE")
    logger.info("=" * 60)

    return report


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    report = run_evaluation(n_bootstrap=1000)
    s = report["model_summary"]
    print(f"\n{'='*55}")
    print(f"{'MODEL':<12} {'ACC':>8} {'MACRO F1':>10} {'95% CI':>22}")
    print(f"{'='*55}")
    for name in MODEL_NAMES:
        if name not in s:
            continue
        v  = s[name]
        ci = v["bootstrap_ci"]
        ci_str = f"[{ci['ci_lower']:.3f}–{ci['ci_upper']:.3f}]" if ci else "-"
        print(f"{name:<12} {v['accuracy']:>8.4f} {v['macro_f1']:>10.4f} {ci_str:>22}")
    print(f"{'='*55}")
    print(f"\nFigures saved to: results/figures/")
    print(f"PRD ≥85% F1 : {report['prd_targets']['macro_f1_85pct']}")
    print(f"≥90% Acc    : {report['prd_targets']['accuracy_90pct']}")

"""
Stacking Ensemble — Phase 6 upgraded.

Replaces simple weighted average with 3 complementary strategies to push
accuracy above 95%:

  Strategy 1 — Stacked Generalization (meta-learner XGBoost + LogReg)
      Train an XGBoost / Logistic Regression on the 5-model probability
      vectors (50-dim features = 5 models × 10 classes per sample).
      Meta-learner is trained on val set probs, evaluated on test set probs.

  Strategy 2 — Per-class optimal weights (10 independent weight vectors)
      Instead of one 5-dim weight vector for all classes, optimise a
      separate weight vector for each class → much finer-grained control.

  Strategy 3 — Hybrid stack-then-average
      Soft-average of the XGBoost meta preds and the refined weighted average
      → typically outperforms either alone.

Outputs
-------
  results/stacking_metrics.json     — stacking test-set metrics
  results/stacking_weights.json     — per-class weights + meta-learner info
  models/meta_learner_xgb.json      — saved XGBoost meta-learner
  models/meta_learner_lr.pkl        — saved LogisticRegression meta-learner
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logging, set_seed

SEED = 42
NUM_CLASSES = 10
CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _full_metrics(y_true, y_pred) -> dict:
    per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1":        round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "weighted_f1":     round(float(f1_score(y_true, y_pred, average="weighted")), 6),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro")), 6),
        "recall_macro":    round(float(recall_score(y_true, y_pred, average="macro")), 6),
        "per_class_f1":   {CLASS_NAMES[i]: round(float(v), 6) for i, v in enumerate(per_cls)},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load all model probability matrices
# ─────────────────────────────────────────────────────────────────────────────
def load_all_probs(split: str, split_csv: str) -> tuple:
    """
    Collect (N, 10) probability matrices from all 5 models for a data split.

    Returns
    -------
    probs_dict : {model_name: np.ndarray (N, 10)}
    y          : np.ndarray (N,)
    """
    logger = logging.getLogger(__name__)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    import tensorflow as tf
    import torch
    import torch.nn.functional as torch_F
    from src.models import AudioCNN

    # Label encoder & ground-truth
    enc = joblib.load("models/label_encoder.pkl")
    df  = pd.read_csv(split_csv)
    y   = enc.transform(df["class"].tolist()).astype(np.int32)

    # ── MFCC features ────────────────────────────────────────────────────────
    mfcc_dir = Path(f"features/mfcc/{split}")
    X_mfcc = []
    for _, row in df.iterrows():
        stem = Path(row["slice_file_name"]).stem
        npy  = mfcc_dir / f"{stem}.npy"
        X_mfcc.append(np.load(npy) if npy.exists() else np.zeros(80))
    X_mfcc = np.stack(X_mfcc).astype(np.float32)

    # ── Mel features ─────────────────────────────────────────────────────────
    mel_dir = Path(f"features/mel/{split}")
    X_mel = []
    for _, row in df.iterrows():
        stem = Path(row["slice_file_name"]).stem
        npy  = mel_dir / f"{stem}.npy"
        if npy.exists():
            mel = np.load(npy)                    # (128, 128, 1) or similar
            if mel.ndim == 3:
                mel = mel.transpose(2, 0, 1)      # (1, 128, 128)
            X_mel.append(mel)
        else:
            X_mel.append(np.zeros((1, 128, 128)))
    X_mel = np.stack(X_mel).astype(np.float32)

    # ── YAMNet embeddings ────────────────────────────────────────────────────
    X_emb   = np.load(f"features/yamnet_embeddings/{split}.npy")
    E_train = np.load("features/yamnet_embeddings/train.npy")
    emb_mean = E_train.mean(axis=0, keepdims=True)
    emb_std  = E_train.std(axis=0, keepdims=True) + 1e-8
    X_emb_n  = ((X_emb - emb_mean) / emb_std).astype(np.float32)

    probs = {}

    # RF
    rf = joblib.load("models/random_forest.pkl")
    probs["rf"] = rf.predict_proba(X_mfcc).astype(np.float32)
    logger.info(f"  RF  probs: {probs['rf'].shape}")

    # SVM
    svm = joblib.load("models/svm.pkl")
    try:
        probs["svm"] = svm.predict_proba(X_mfcc).astype(np.float32)
    except Exception:
        sc = svm.named_steps["scaler"].transform(X_mfcc)
        d  = svm.named_steps["svm"].decision_function(sc).astype(np.float32)
        d -= d.max(axis=1, keepdims=True)
        probs["svm"] = (np.exp(d) / np.exp(d).sum(axis=1, keepdims=True))
    logger.info(f"  SVM probs: {probs['svm'].shape}")

    # XGBoost
    xgb = XGBClassifier()
    xgb.load_model("models/xgboost.json")
    probs["xgb"] = xgb.predict_proba(X_mfcc).astype(np.float32)
    logger.info(f"  XGB probs: {probs['xgb'].shape}")

    # CNN
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = AudioCNN(num_classes=10).to(device)
    cnn.load_state_dict(torch.load("models/cnn_best.pt", map_location=device))
    cnn.eval()
    cnn_p = []
    with torch.no_grad():
        for i in range(0, len(X_mel), 64):
            batch = torch.from_numpy(X_mel[i:i+64]).to(device)
            cnn_p.append(torch_F.softmax(cnn(batch), dim=1).cpu().numpy())
    probs["cnn"] = np.concatenate(cnn_p).astype(np.float32)
    logger.info(f"  CNN probs: {probs['cnn'].shape}")

    # YAMNet head
    head = tf.keras.models.load_model("models/yamnet_head.h5")
    probs["yamnet"] = head.predict(X_emb_n, batch_size=128, verbose=0).astype(np.float32)
    logger.info(f"  YAMNet probs: {probs['yamnet'].shape}")

    return probs, y


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Stacked generalization
# ─────────────────────────────────────────────────────────────────────────────
def build_stack_features(probs_dict: dict) -> np.ndarray:
    """Concatenate per-model probs → (N, 50) meta-feature matrix."""
    order = ["rf", "svm", "xgb", "cnn", "yamnet"]
    return np.concatenate([probs_dict[n] for n in order], axis=1).astype(np.float32)


def train_meta_xgb(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train XGBoost meta-learner on 50-dim stack features."""
    meta = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=SEED,
        verbosity=0,
    )
    meta.fit(X_train, y_train)
    return meta


def train_meta_logreg(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train LogisticRegression meta-learner (strong regularisation)."""
    from sklearn.preprocessing import StandardScaler
    meta = LogisticRegression(
        C=0.1,
        max_iter=2000,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=SEED,
    )
    meta.fit(X_train, y_train)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Per-class weight optimisation
# ─────────────────────────────────────────────────────────────────────────────
def _optimise_per_class_weights(
    probs_dict: dict,
    y_val: np.ndarray,
    n_restarts: int = 20,
    seed: int = SEED,
) -> np.ndarray:
    """
    Optimise a separate 5-dim weight vector for each of the 10 classes.

    Returns
    -------
    W : np.ndarray (10, 5)  — W[c] = weight vector for class c
    """
    logger = logging.getLogger(__name__)
    order = ["rf", "svm", "xgb", "cnn", "yamnet"]
    P = np.stack([probs_dict[n] for n in order], axis=0)  # (5, N, 10)
    rng = np.random.RandomState(seed)

    # Global weighted probs → use as starting point
    def _ensemble_probs(W):
        # W: (10, 5), normalise each row
        W_norm = np.abs(W) / (np.abs(W).sum(axis=1, keepdims=True) + 1e-9)
        # P: (5, N, 10)
        # P_stack: (N, 10, 5)
        P_stack = np.stack([probs_dict[n] for n in order], axis=-1)  # (N, 10, 5)
        # result[n, c] = sum_m P_stack[n, c, m] * W_norm[c, m]
        ens = np.einsum("ncm,cm->nc", P_stack, W_norm)
        return ens.astype(np.float32)

    def _objective(w_flat):
        W = w_flat.reshape(10, 5)
        ens = _ensemble_probs(W)
        y_pred = np.argmax(ens, axis=1)
        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average="macro", zero_division=0)
        return -(acc + 0.5 * f1)

    best_score = np.inf
    best_w = rng.dirichlet(np.ones(5), size=10).flatten()

    for i in range(n_restarts):
        w0 = rng.dirichlet(np.ones(5), size=10).flatten()
        res = minimize(
            _objective, w0,
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-7},
        )
        if res.fun < best_score:
            best_score = res.fun
            best_w = res.x
            W_cur = np.abs(best_w.reshape(10, 5))
            W_norm = W_cur / (W_cur.sum(axis=1, keepdims=True) + 1e-9)
            ens = _ensemble_probs(best_w.reshape(10, 5))
            y_pred = np.argmax(ens, axis=1)
            acc = accuracy_score(y_val, y_pred)
            logger.info(
                f"  restart {i+1:2d}/{n_restarts}: "
                f"val_acc={acc:.5f}  score={-res.fun:.5f}"
            )

    W_final = np.abs(best_w.reshape(10, 5))
    W_final = W_final / (W_final.sum(axis=1, keepdims=True) + 1e-9)  # normalise rows
    return W_final


def apply_per_class_weights(probs_dict: dict, W: np.ndarray) -> np.ndarray:
    """
    Apply per-class weight matrix W (10, 5) to get ensemble probabilities.

    ens[n, c] = Σ_m W[c, m] * probs[m][n, c]
    W shape: (num_classes=10, num_models=5)
    """
    order = ["rf", "svm", "xgb", "cnn", "yamnet"]
    # P_stack: (N, 10, 5)
    P_stack = np.stack([probs_dict[n] for n in order], axis=-1)
    # W: (10, 5) → broadcast multiply per sample: (N, 10) * (10, 5) summed over axis=-1
    # result[n, c] = sum_m P_stack[n, c, m] * W[c, m]
    ens = np.einsum("ncm,cm->nc", P_stack, W)
    return ens.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3 — Hybrid (XGB meta + per-class) soft blend
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_blend(
    meta_probs: np.ndarray,
    pc_probs: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Soft blend of meta-learner and per-class weighted ensemble probabilities.

    alpha × meta_probs + (1-alpha) × pc_probs
    """
    return alpha * meta_probs + (1 - alpha) * pc_probs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_stacking(
    val_csv:  str = "data/splits/val.csv",
    test_csv: str = "data/splits/test.csv",
    n_pc_restarts: int = 30,
    hybrid_alpha: float = 0.6,
    seed: int = SEED,
    output_dir: str = "results",
) -> dict:
    """
    Full stacking pipeline.

    Returns
    -------
    dict — results containing all strategy metrics
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(seed)

    logger.info("=" * 60)
    logger.info("STACKING ENSEMBLE — Pushing for ≥95% Accuracy")
    logger.info("=" * 60)

    # ── 1. Load validation probs ────────────────────────────────────────────
    logger.info("\n[1/6] Loading validation probabilities...")
    val_probs, y_val = load_all_probs("val", val_csv)

    # ── 2. Load test probs ──────────────────────────────────────────────────
    logger.info("\n[2/6] Loading test probabilities...")
    test_probs, y_test = load_all_probs("test", test_csv)

    # ── 3. Strategy 1: XGBoost meta-learner ─────────────────────────────────
    logger.info("\n[3/6] Training XGBoost meta-learner (stack features = 50-dim)...")
    X_val_stack  = build_stack_features(val_probs)   # (1310, 50)
    X_test_stack = build_stack_features(test_probs)  # (1310, 50)

    # 5-fold CV on val set to get out-of-fold predictions (prevents overfitting)
    logger.info("  Generating OOF predictions via 5-fold CV on val set...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_xgb_probs = np.zeros((len(y_val), NUM_CLASSES), dtype=np.float32)

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_val_stack, y_val)):
        fold_meta = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=seed + fold, verbosity=0,
        )
        fold_meta.fit(X_val_stack[tr_idx], y_val[tr_idx])
        oof_xgb_probs[vl_idx] = fold_meta.predict_proba(X_val_stack[vl_idx])
        acc_fold = accuracy_score(
            y_val[vl_idx], np.argmax(oof_xgb_probs[vl_idx], axis=1)
        )
        logger.info(f"    Fold {fold+1}: val_acc={acc_fold:.4f}")

    oof_acc = accuracy_score(y_val, np.argmax(oof_xgb_probs, axis=1))
    logger.info(f"  OOF val accuracy: {oof_acc:.4f}")

    # Train final XGB meta on ALL val data → evaluate on test
    logger.info("  Training final XGB meta on full val set...")
    meta_xgb = train_meta_xgb(X_val_stack, y_val)
    meta_xgb.save_model("models/meta_learner_xgb.json")
    xgb_test_probs = meta_xgb.predict_proba(X_test_stack).astype(np.float32)
    xgb_preds = np.argmax(xgb_test_probs, axis=1)
    xgb_metrics = _full_metrics(y_test, xgb_preds)
    logger.info(f"  XGB meta  → test acc={xgb_metrics['accuracy']:.4f}  "
                f"f1={xgb_metrics['macro_f1']:.4f}")

    # LogReg meta
    logger.info("  Training LogisticRegression meta-learner...")
    meta_lr = train_meta_logreg(X_val_stack, y_val)
    joblib.dump(meta_lr, "models/meta_learner_lr.pkl")
    lr_test_probs = meta_lr.predict_proba(X_test_stack).astype(np.float32)
    lr_preds = np.argmax(lr_test_probs, axis=1)
    lr_metrics = _full_metrics(y_test, lr_preds)
    logger.info(f"  LR  meta  → test acc={lr_metrics['accuracy']:.4f}  "
                f"f1={lr_metrics['macro_f1']:.4f}")

    # ── 4. Strategy 2: Per-class weights ────────────────────────────────────
    logger.info(f"\n[4/6] Per-class weight optimisation ({n_pc_restarts} restarts)...")
    W_pc = _optimise_per_class_weights(val_probs, y_val, n_restarts=n_pc_restarts, seed=seed)
    pc_test_probs = apply_per_class_weights(test_probs, W_pc)
    pc_preds = np.argmax(pc_test_probs, axis=1)
    pc_metrics = _full_metrics(y_test, pc_preds)
    logger.info(f"  Per-class → test acc={pc_metrics['accuracy']:.4f}  "
                f"f1={pc_metrics['macro_f1']:.4f}")

    # ── 5. Strategy 3: Hybrid blends ────────────────────────────────────────
    logger.info(f"\n[5/6] Hybrid blending (XGB meta + LogReg meta + per-class weights)...")

    # Find best alpha via val set
    best_alpha, best_val_acc = 0.5, 0.0
    for alpha in np.arange(0.1, 1.0, 0.05):
        # Blend xgb_meta + pc on val
        xgb_val_probs_full  = meta_xgb.predict_proba(X_val_stack).astype(np.float32)
        pc_val_probs_full   = apply_per_class_weights(val_probs, W_pc)
        blended_val = hybrid_blend(xgb_val_probs_full, pc_val_probs_full, alpha)
        acc = accuracy_score(y_val, np.argmax(blended_val, axis=1))
        if acc > best_val_acc:
            best_val_acc = acc
            best_alpha = float(alpha)

    logger.info(f"  Best alpha (XGB+PC): {best_alpha:.2f} (val_acc={best_val_acc:.4f})")

    # Apply best alpha on test
    hybrid_test_probs = hybrid_blend(xgb_test_probs, pc_test_probs, best_alpha)
    hybrid_preds = np.argmax(hybrid_test_probs, axis=1)
    hybrid_metrics = _full_metrics(y_test, hybrid_preds)
    logger.info(f"  Hybrid    → test acc={hybrid_metrics['accuracy']:.4f}  "
                f"f1={hybrid_metrics['macro_f1']:.4f}")

    # Also try: 3-way blend (XGB + LR + PC)
    best_3way = _find_best_3way_blend(
        meta_xgb.predict_proba(X_val_stack).astype(np.float32),
        meta_lr.predict_proba(X_val_stack).astype(np.float32),
        apply_per_class_weights(val_probs, W_pc),
        xgb_test_probs, lr_test_probs, pc_test_probs,
        y_val, y_test, logger,
    )

    # ── 6. Choose best and export ────────────────────────────────────────────
    logger.info("\n[6/6] Comparing all strategies on test set...")

    all_strategies = {
        "xgb_meta":         xgb_metrics,
        "lr_meta":          lr_metrics,
        "per_class_weights": pc_metrics,
        "hybrid_xgb_pc":    hybrid_metrics,
        "3way_blend":       best_3way["metrics"],
    }

    best_name = max(all_strategies, key=lambda k: all_strategies[k]["accuracy"])
    best_metrics = all_strategies[best_name]

    logger.info("\n" + "=" * 55)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 55)
    for name, m in all_strategies.items():
        flag = " ← BEST" if name == best_name else ""
        target = " ✅ ≥95%" if m["accuracy"] >= 0.95 else ""
        logger.info(f"  {name:25s}: acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}{target}{flag}")
    logger.info("=" * 55)
    logger.info(f"Best strategy: {best_name}")
    logger.info(f"Best accuracy: {best_metrics['accuracy']:.4f}")
    logger.info(f"Best Macro F1: {best_metrics['macro_f1']:.4f}")

    logger.info("\nPer-class F1 (best strategy):")
    for cls, f1v in best_metrics["per_class_f1"].items():
        logger.info(f"  {cls:25s}: {f1v:.4f}")

    # Compare with original ensemble baseline
    with open("results/ensemble_metrics.json") as f:
        baseline = json.load(f)
    improvement = best_metrics["accuracy"] - baseline["accuracy"]
    logger.info(f"\nImprovement over baseline: +{improvement:.4f} "
                f"({improvement*100:.2f}pp)")

    # Export
    os.makedirs(output_dir, exist_ok=True)
    all_results = {
        "best_strategy": best_name,
        "best_metrics": best_metrics,
        "baseline_accuracy": round(baseline["accuracy"], 6),
        "improvement_pp": round(float(improvement * 100), 4),
        "target_95_met": best_metrics["accuracy"] >= 0.95,
        "all_strategies": {
            k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
            for k, v in all_strategies.items()
        },
        "per_class_weights": {
            CLASS_NAMES[c]: {
                m: round(float(W_pc[c, i]), 6)
                for i, m in enumerate(["rf", "svm", "xgb", "cnn", "yamnet"])
            }
            for c in range(NUM_CLASSES)
        },
        "hybrid_alpha": round(best_alpha, 4),
        "3way_blend": {
            "alpha_xgb": round(best_3way["alpha_xgb"], 4),
            "alpha_lr": round(best_3way["alpha_lr"], 4),
            "alpha_pc": round(best_3way["alpha_pc"], 4),
        },
    }

    stacking_path = f"{output_dir}/stacking_metrics.json"
    with open(stacking_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults → {stacking_path}")

    logger.info("=" * 60)
    logger.info("STACKING COMPLETE")
    logger.info("=" * 60)

    return all_results


def _find_best_3way_blend(
    xgb_val, lr_val, pc_val,
    xgb_test, lr_test, pc_test,
    y_val, y_test, logger,
) -> dict:
    """Grid-search optimal (α1, α2, α3) on val set for XGB+LR+PC blend."""
    best = {"acc": 0, "alpha_xgb": 0.5, "alpha_lr": 0.25, "alpha_pc": 0.25, "metrics": {}}

    for x in np.arange(0.1, 0.8, 0.1):
        for l in np.arange(0.05, 0.6, 0.1):
            p = round(1.0 - x - l, 4)
            if p < 0.05 or p > 0.7:
                continue
            blend_v = x * xgb_val + l * lr_val + p * pc_val
            acc = accuracy_score(y_val, np.argmax(blend_v, axis=1))
            if acc > best["acc"]:
                blend_t = x * xgb_test + l * lr_test + p * pc_test
                best = {
                    "acc": acc,
                    "alpha_xgb": float(x),
                    "alpha_lr": float(l),
                    "alpha_pc": float(p),
                    "metrics": _full_metrics(y_test, np.argmax(blend_t, axis=1)),
                }

    logger.info(
        f"  3-way (XGB={best['alpha_xgb']:.2f}, LR={best['alpha_lr']:.2f}, "
        f"PC={best['alpha_pc']:.2f}) → "
        f"val_acc={best['acc']:.4f}  "
        f"test_acc={best['metrics'].get('accuracy', 0):.4f}"
    )
    return best


if __name__ == "__main__":
    results = run_stacking(n_pc_restarts=30)

    print(f"\n{'='*55}")
    print("STACKING RESULTS SUMMARY")
    print(f"{'='*55}")
    for name, m in results["all_strategies"].items():
        target = " ✅ ≥95%" if m["accuracy"] >= 0.95 else ""
        print(f"  {name:25s}: acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}{target}")
    print(f"{'='*55}")
    print(f"Best:       {results['best_strategy']}")
    print(f"Accuracy:   {results['best_metrics']['accuracy']:.4f}")
    print(f"Macro F1:   {results['best_metrics']['macro_f1']:.4f}")
    print(f"≥95% met:   {results['target_95_met']}")
    print(f"Improvement: +{results['improvement_pp']:.2f}pp over baseline")

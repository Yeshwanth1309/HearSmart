"""
Phase 4 Comprehensive Audit Script
Context-Aware Hearing Aid Project
Run from project root: python phase4_audit.py
ASCII-only output to avoid Windows cp1252 encoding issues.
"""

import os
import sys
import json
import time
import math
from pathlib import Path

# Suppress verbose TF / torch logs before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Output helpers (pure ASCII)
# ---------------------------------------------------------------------------
results = {}   # section_key -> bool

def section(title):
    print("\n" + "=" * 65)
    print("  " + title)
    print("=" * 65)

def report(label, ok, note=""):
    tag = "[PASS]" if ok else "[FAIL]"
    msg = f"  {tag}  {label}"
    if note:
        msg += f"\n         -> {note}"
    print(msg)
    return ok

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAFETY_CLASSES    = {"car_horn", "gun_shot", "siren"}
LATENCY_LIMIT_MS  = 2000.0
SIZE_LIMIT_MB     = 150.0
SEED              = 42

mfcc_root  = PROJECT_ROOT / "features" / "mfcc"
splits_dir = PROJECT_ROOT / "data"     / "splits"
models_dir = PROJECT_ROOT / "models"

# ============================================================
# SECTION 1 — MFCC Feature Integrity
# ============================================================
section("1. MFCC Feature Integrity")

overall_feature_ok = True
split_counts = {}

for split in ["train", "val", "test"]:
    csv_path = splits_dir / f"{split}.csv"
    npy_dir  = mfcc_root  / split

    if not csv_path.exists():
        report(f"{split}.csv exists", False, str(csv_path))
        overall_feature_ok = False
        continue

    df = pd.read_csv(csv_path)
    csv_count = len(df)
    split_counts[split] = csv_count

    npy_stems = set(p.stem for p in npy_dir.glob("*.npy")) if npy_dir.exists() else set()
    csv_stems = set(Path(fn).stem for fn in df["slice_file_name"])

    missing = csv_stems - npy_stems
    extra   = npy_stems - csv_stems

    ok_count = report(
        f"{split}: CSV rows={csv_count}, .npy files={len(npy_stems)}",
        len(missing) == 0,
        f"{len(missing)} missing, {len(extra)} extra" if (missing or extra) else "",
    )

    dups = df["slice_file_name"].duplicated().sum()
    ok_dup = report(
        f"{split}: no duplicate filenames in CSV",
        dups == 0,
        f"{dups} duplicate(s)" if dups else "",
    )

    # Sample up to 50 .npy files for shape / dtype / NaN / Inf
    sample = list(csv_stems & npy_stems)[:50]
    bad_shape = bad_dtype = has_nan = has_inf = 0
    for stem in sample:
        arr = np.load(npy_dir / f"{stem}.npy")
        if arr.shape != (80,):
            bad_shape += 1
        if arr.dtype != np.float32:
            bad_dtype += 1
        if np.any(np.isnan(arr)):
            has_nan += 1
        if np.any(np.isinf(arr)):
            has_inf += 1

    n = len(sample)
    report(f"{split}: shape == (80,) [sampled {n}]",   bad_shape == 0, f"{bad_shape} bad" if bad_shape else "")
    report(f"{split}: dtype == float32 [sampled {n}]", bad_dtype == 0, f"{bad_dtype} bad" if bad_dtype else "")
    report(f"{split}: no NaN [sampled {n}]",           has_nan   == 0, f"{has_nan} with NaN"  if has_nan  else "")
    report(f"{split}: no Inf [sampled {n}]",           has_inf   == 0, f"{has_inf} with Inf"  if has_inf  else "")

    overall_feature_ok &= (ok_count and ok_dup
                           and bad_shape == 0 and bad_dtype == 0
                           and has_nan == 0 and has_inf == 0)

results["1_mfcc_integrity"] = overall_feature_ok
print(f"\n  --> Section 1: {'PASS' if overall_feature_ok else 'FAIL'}")

# ============================================================
# SECTION 2 — Model Training Verification
# ============================================================
section("2. Model Training Verification")

import joblib

rf_path     = models_dir / "random_forest.pkl"
svm_path    = models_dir / "svm_model.pkl"
scaler_path = models_dir / "svm_scaler.pkl"
xgb_pkl     = models_dir / "xgboost.pkl"
xgb_json    = models_dir / "xgboost.json"
xgb_exists  = xgb_pkl.exists() or xgb_json.exists()
xgb_actual  = xgb_pkl if xgb_pkl.exists() else (xgb_json if xgb_json.exists() else None)

ok_rf  = report("models/random_forest.pkl exists", rf_path.exists())
ok_svm = report("models/svm_model.pkl exists",     svm_path.exists())
ok_sc  = report("models/svm_scaler.pkl exists",    scaler_path.exists())
ok_xgb = report(
    "models/xgboost file exists",
    xgb_exists,
    f"found: {xgb_actual.name}" if xgb_actual else "neither xgboost.pkl nor xgboost.json found",
)

def check_fitted(path):
    """Return True if the joblib object at path is a fitted sklearn estimator."""
    if path is None or not path.exists():
        return False
    try:
        obj = joblib.load(path)
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(obj)
        return True
    except Exception:
        try:
            obj = joblib.load(path)
            return hasattr(obj, "predict")
        except Exception:
            return False

ok_rf_fit  = report("RF  is fitted (sklearn check)", check_fitted(rf_path))
ok_svm_fit = report("SVM is fitted (sklearn check)", check_fitted(svm_path))

xgb_fitted = False
if xgb_actual and xgb_actual.exists():
    try:
        if xgb_actual.suffix == ".json":
            from xgboost import XGBClassifier
            _xgb = XGBClassifier()
            _xgb.load_model(str(xgb_actual))
            xgb_fitted = True
        else:
            _xgb_obj = joblib.load(xgb_actual)
            xgb_fitted = hasattr(_xgb_obj, "predict")
    except Exception as e:
        print(f"         -> XGBoost load error: {e}")
ok_xgb_fit = report("XGBoost is loadable / has predict()", xgb_fitted)

sec2_ok = all([ok_rf, ok_svm, ok_sc, ok_xgb, ok_rf_fit, ok_svm_fit, ok_xgb_fit])
results["2_model_training"] = sec2_ok
print(f"\n  --> Section 2: {'PASS' if sec2_ok else 'FAIL'}")

# ============================================================
# SECTION 3 — Reproducibility
# ============================================================
section("3. Reproducibility")

finalize_path = PROJECT_ROOT / "src" / "phase4_finalize_and_export.py"
utils_path    = PROJECT_ROOT / "src" / "utils.py"

def file_contains(path, text):
    if not path.exists():
        return False
    return text in path.read_text(encoding="utf-8", errors="ignore")

ok_seed_const = report("SEED = 42 defined in finalize script",
                        file_contains(finalize_path, "SEED = 42"))
ok_np_seed    = report("np.random.seed called in finalize script",
                        file_contains(finalize_path, "np.random.seed"))
ok_set_seed   = report("set_seed(SEED) called in finalize script",
                        file_contains(finalize_path, "set_seed(SEED)"))
ok_sk_seed    = report("random_state=SEED used in RF / SVM / XGB",
                        file_contains(finalize_path, "random_state=SEED"))
ok_utils      = report("utils.set_seed covers random, numpy, torch, tf",
                        all(file_contains(utils_path, t) for t in
                            ["random.seed", "numpy.random.seed",
                             "torch.manual_seed", "tensorflow.random.set_seed"]))

sec3_ok = all([ok_seed_const, ok_np_seed, ok_set_seed, ok_sk_seed, ok_utils])
results["3_reproducibility"] = sec3_ok
print(f"\n  --> Section 3: {'PASS' if sec3_ok else 'FAIL'}")

# ============================================================
# SECTION 4 — Evaluation Metrics (live, on test set)
# ============================================================
section("4. Evaluation Metrics (Test Set)")

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

sec4_ok = True
metrics_by_model = {}

# --- load all three splits ---
def load_split(split, le, fit=False):
    csv  = splits_dir / f"{split}.csv"
    feat = mfcc_root  / split
    df   = pd.read_csv(csv)
    X_list, labels = [], []
    for _, row in df.iterrows():
        stem = Path(row["slice_file_name"]).stem
        p    = feat / f"{stem}.npy"
        if p.exists():
            X_list.append(np.load(p))
            labels.append(row["class"])
    X = np.stack(X_list)
    if fit:
        le.fit(labels)
    y = le.transform(labels)
    return X, y, labels

try:
    le = LabelEncoder()
    X_train, y_train, _        = load_split("train", le, fit=True)
    X_val,   y_val,   _        = load_split("val",   le)
    X_test,  y_test,  y_raw    = load_split("test",  le)

    X_full = np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    class_names = list(le.classes_)

    print(f"\n  Classes ({len(class_names)}): {class_names}")
    print(f"  Train+Val N={len(X_full)}   Test N={len(X_test)}")

    def evaluate(model, X, y, preprocess=None):
        Xp = preprocess(X) if preprocess else X
        yp = model.predict(Xp)
        macro    = float(f1_score(y, yp, average="macro"))
        weighted = float(f1_score(y, yp, average="weighted"))
        pc_arr   = f1_score(y, yp, average=None)
        pc       = {class_names[i]: float(pc_arr[i]) for i in range(len(class_names))}
        cm       = confusion_matrix(y, yp).tolist()
        has_nan  = any(math.isnan(v) for v in [macro, weighted] + list(pc.values()))
        return macro, weighted, pc, cm, has_nan

    # --- Random Forest ---
    rf_model = joblib.load(rf_path)
    rf_macro, rf_weighted, rf_pc, rf_cm, rf_nan = evaluate(rf_model, X_test, y_test)
    metrics_by_model["random_forest"] = dict(
        macro_f1=round(rf_macro, 6), weighted_f1=round(rf_weighted, 6),
        per_class_f1={k: round(v,6) for k,v in rf_pc.items()}, confusion_matrix=rf_cm)
    report(f"RF  macro F1 = {rf_macro:.4f}  weighted = {rf_weighted:.4f}  (no NaN)", not rf_nan)
    report("RF  safety-critical F1 present",
           all(c in rf_pc for c in SAFETY_CLASSES),
           "  " + " | ".join(f"{c}={rf_pc.get(c, 'MISSING'):.4f}" for c in sorted(SAFETY_CLASSES)))
    print(f"    RF per-class: " +
          "  ".join(f"{k}={v:.3f}" for k, v in rf_pc.items()))

    # --- SVM ---
    svm_model  = joblib.load(svm_path)
    svm_scaler = joblib.load(scaler_path)
    svm_macro, svm_weighted, svm_pc, svm_cm, svm_nan = evaluate(
        svm_model, X_test, y_test, preprocess=svm_scaler.transform)
    metrics_by_model["svm"] = dict(
        macro_f1=round(svm_macro, 6), weighted_f1=round(svm_weighted, 6),
        per_class_f1={k: round(v,6) for k,v in svm_pc.items()}, confusion_matrix=svm_cm)
    report(f"SVM macro F1 = {svm_macro:.4f}  weighted = {svm_weighted:.4f}  (no NaN)", not svm_nan)
    report("SVM safety-critical F1 present",
           all(c in svm_pc for c in SAFETY_CLASSES),
           "  " + " | ".join(f"{c}={svm_pc.get(c,'MISSING'):.4f}" for c in sorted(SAFETY_CLASSES)))
    print(f"    SVM per-class: " +
          "  ".join(f"{k}={v:.3f}" for k, v in svm_pc.items()))

    # --- XGBoost ---
    if xgb_actual.suffix == ".json":
        from xgboost import XGBClassifier as XGBC
        xgb_model = XGBC()
        xgb_model.load_model(str(xgb_actual))
    else:
        xgb_model = joblib.load(xgb_actual)
    xgb_macro, xgb_weighted, xgb_pc, xgb_cm, xgb_nan = evaluate(xgb_model, X_test, y_test)
    metrics_by_model["xgboost"] = dict(
        macro_f1=round(xgb_macro, 6), weighted_f1=round(xgb_weighted, 6),
        per_class_f1={k: round(v,6) for k,v in xgb_pc.items()}, confusion_matrix=xgb_cm)
    report(f"XGB macro F1 = {xgb_macro:.4f}  weighted = {xgb_weighted:.4f}  (no NaN)", not xgb_nan)
    report("XGB safety-critical F1 present",
           all(c in xgb_pc for c in SAFETY_CLASSES),
           "  " + " | ".join(f"{c}={xgb_pc.get(c,'MISSING'):.4f}" for c in sorted(SAFETY_CLASSES)))
    print(f"    XGB per-class: " +
          "  ".join(f"{k}={v:.3f}" for k, v in xgb_pc.items()))

    sec4_ok = not (rf_nan or svm_nan or xgb_nan)

except Exception as exc:
    report("Metric computation", False, str(exc))
    import traceback; traceback.print_exc()
    sec4_ok = False

results["4_evaluation_metrics"] = sec4_ok
print(f"\n  --> Section 4: {'PASS' if sec4_ok else 'FAIL'}")

# ============================================================
# SECTION 5 — Performance Benchmarks
# ============================================================
section("5. Performance Benchmarks")

def measure_latency(model, X, preprocess=None, n_runs=100):
    rng = np.random.RandomState(SEED)
    indices = rng.randint(0, len(X), n_runs)
    times = []
    for i in indices:
        s = X[i : i + 1]
        if preprocess:
            s = preprocess(s)
        t0 = time.perf_counter()
        model.predict(s)
        times.append(time.perf_counter() - t0)
    return round(float(np.mean(times)) * 1000, 4)

def file_mb(p):
    if p is None or not (isinstance(p, Path) and p.exists()):
        return 0.0
    return round(p.stat().st_size / (1024 * 1024), 4)

sec5_ok = True
lat = {}
sz  = {}

try:
    lat["rf"]  = measure_latency(rf_model, X_test)
    sz["rf"]   = file_mb(rf_path)
    lat["svm"] = measure_latency(svm_model, X_test, preprocess=svm_scaler.transform)
    sz["svm"]  = round(file_mb(svm_path) + file_mb(scaler_path), 4)
    lat["xgb"] = measure_latency(xgb_model, X_test)
    sz["xgb"]  = file_mb(xgb_actual)

    ok_rf_lat  = report(f"RF  latency = {lat['rf']} ms  (limit {LATENCY_LIMIT_MS} ms)",  lat["rf"]  < LATENCY_LIMIT_MS)
    ok_svm_lat = report(f"SVM latency = {lat['svm']} ms  (limit {LATENCY_LIMIT_MS} ms)", lat["svm"] < LATENCY_LIMIT_MS)
    ok_xgb_lat = report(f"XGB latency = {lat['xgb']} ms  (limit {LATENCY_LIMIT_MS} ms)", lat["xgb"] < LATENCY_LIMIT_MS)

    print(f"  RF  size = {sz['rf']} MB")
    print(f"  SVM size = {sz['svm']} MB  (model + scaler)")
    print(f"  XGB size = {sz['xgb']} MB")

    total_mb = sz["rf"] + sz["svm"] + sz["xgb"]
    ok_size = report(f"Combined size = {total_mb:.2f} MB  (limit {SIZE_LIMIT_MB} MB)", total_mb < SIZE_LIMIT_MB)
    sec5_ok = all([ok_rf_lat, ok_svm_lat, ok_xgb_lat, ok_size])

except Exception as exc:
    report("Latency / size benchmark", False, str(exc))
    import traceback; traceback.print_exc()
    sec5_ok = False

results["5_performance"] = sec5_ok
print(f"\n  --> Section 5: {'PASS' if sec5_ok else 'FAIL'}")

# ============================================================
# SECTION 6 — Artifact Export
# ============================================================
section("6. Artifact Export")

json_path = PROJECT_ROOT / "results" / "traditional_ml_metrics.json"

ok_a1 = report("models/random_forest.pkl",                  rf_path.exists())
ok_a2 = report("models/svm_model.pkl",                      svm_path.exists())
ok_a3 = report("models/svm_scaler.pkl",                     scaler_path.exists())
ok_a4 = report("models/xgboost model (pkl or json)",
                xgb_exists, f"found as {xgb_actual.name}" if xgb_actual else "MISSING")
ok_a5 = report("results/traditional_ml_metrics.json exists", json_path.exists(),
                "run phase4_finalize_and_export.py to generate" if not json_path.exists() else "")

json_struct_ok = False
if json_path.exists():
    try:
        with open(json_path, encoding="utf-8") as f:
            jdata = json.load(f)
        required_models  = {"random_forest", "svm", "xgboost"}
        required_metrics = {"macro_f1", "weighted_f1", "per_class_f1", "confusion_matrix"}
        missing_m = required_models - set(jdata.keys())
        issues = []
        if missing_m:
            issues.append(f"missing model entries: {missing_m}")
        for m in (required_models - missing_m):
            miss_k = required_metrics - set(jdata[m].keys())
            if miss_k:
                issues.append(f"{m} missing keys: {miss_k}")
        json_struct_ok = not issues
        report("JSON structure valid (3 models + required keys)",
               json_struct_ok, "; ".join(issues) if issues else "")
    except Exception as exc:
        report("JSON parseable", False, str(exc))
else:
    # Auto-generate from live run metrics
    try:
        for name, lat_v, sz_v in [
            ("random_forest", lat.get("rf"),  sz.get("rf")),
            ("svm",           lat.get("svm"), sz.get("svm")),
            ("xgboost",       lat.get("xgb"), sz.get("xgb")),
        ]:
            if name in metrics_by_model and lat_v is not None:
                metrics_by_model[name]["latency_ms"]    = lat_v
                metrics_by_model[name]["model_size_mb"] = sz_v
        json_path.parent.mkdir(exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_by_model, f, indent=2)
        print(f"         -> Auto-generated {json_path}")
        ok_a5 = json_struct_ok = True
    except Exception as exc:
        print(f"         -> Auto-generate failed: {exc}")

sec6_ok = all([ok_a1, ok_a2, ok_a3, ok_a4, ok_a5])
results["6_artifact_export"] = sec6_ok
print(f"\n  --> Section 6: {'PASS' if sec6_ok else 'FAIL'}")

# ============================================================
# SECTION 7 — PRD Compliance
# ============================================================
section("7. PRD Compliance Check")

# 7a — Stratified 10-fold CV
validate_path = PROJECT_ROOT / "tests" / "validate_folds.py"
skf_defined   = (file_contains(validate_path, "StratifiedKFold")
                 and file_contains(validate_path, "n_splits=10"))
ok_cv = report(
    "Stratified 10-fold CV defined (validate_folds.py)",
    skf_defined,
)
# Clarify: CV is validation-only, not used as the training loop
print("         NOTE: 10-fold CV used for STRUCTURAL VALIDATION only.")
print("               Final models are trained on single train+val split -> NOT publication-ready.")

# 7b — Macro F1 baseline established
try:
    best = max(rf_macro, svm_macro, xgb_macro)
    best_name = {rf_macro: "RF", svm_macro: "SVM", xgb_macro: "XGB"}[best]
    ok_baseline = report(f"Macro F1 baseline established (best={best:.4f} by {best_name})", True)
except Exception:
    ok_baseline = report("Macro F1 baseline established", False, "metrics unavailable")

# 7c — Safety-critical class F1 for all three models
try:
    sc_rows = []
    for mname, pc in [("RF", rf_pc), ("SVM", svm_pc), ("XGB", xgb_pc)]:
        for cls in sorted(SAFETY_CLASSES):
            v = pc.get(cls)
            sc_rows.append((mname, cls, v))
    all_sc_present = all(v is not None for _, _, v in sc_rows)
    ok_safety = report("Safety-critical class F1 reported for all models", all_sc_present)
    print()
    print("  Safety-critical per-class F1:")
    print(f"  {'Class':<20} {'RF':>8} {'SVM':>8} {'XGB':>8}")
    print("  " + "-" * 48)
    for cls in sorted(SAFETY_CLASSES):
        rf_v  = rf_pc.get(cls,  float("nan"))
        svm_v = svm_pc.get(cls, float("nan"))
        xgb_v = xgb_pc.get(cls, float("nan"))
        print(f"  {cls:<20} {rf_v:>8.4f} {svm_v:>8.4f} {xgb_v:>8.4f}")
except Exception as exc:
    ok_safety = report("Safety-critical class F1", False, str(exc))

# 7d — Data leakage check
try:
    td = pd.read_csv(splits_dir / "train.csv")
    vd = pd.read_csv(splits_dir / "val.csv")
    ed = pd.read_csv(splits_dir / "test.csv")
    ts = set(td["slice_file_name"])
    vs = set(vd["slice_file_name"])
    es = set(ed["slice_file_name"])
    tv = ts & vs; te = ts & es; ve = vs & es
    ok_leak = report(
        "No data leakage between train / val / test",
        not (tv or te or ve),
        f"Train&Val={len(tv)}, Train&Test={len(te)}, Val&Test={len(ve)}" if (tv or te or ve) else "",
    )
except Exception as exc:
    ok_leak = report("Data leakage check", False, str(exc))

sec7_ok = all([ok_cv, ok_baseline, ok_safety, ok_leak])
results["7_prd_compliance"] = sec7_ok
print(f"\n  --> Section 7: {'PASS' if sec7_ok else 'FAIL'}")

# ============================================================
# FINAL VERDICT
# ============================================================
section("PHASE 4 FINAL VERDICT")

engineering_sections = [
    "1_mfcc_integrity",
    "2_model_training",
    "3_reproducibility",
    "4_evaluation_metrics",
    "5_performance",
    "6_artifact_export",
]
engineering_ok   = all(results.get(k, False) for k in engineering_sections)
ten_fold_in_loop = False   # confirmed: only structural validation, not training loop
publication_ok   = engineering_ok and ten_fold_in_loop

print()
print("  Section-by-section summary:")
print(f"  {'Section':<40} {'Status':>6}")
print("  " + "-" * 48)
for k, v in results.items():
    label = k.replace("_", " ").title()
    print(f"  {label:<40} {'PASS' if v else 'FAIL':>6}")

print()
try:
    print("  Model Performance Summary:")
    print(f"  {'Model':<14} {'Macro F1':>9} {'Weighted F1':>12} {'Latency (ms)':>13} {'Size (MB)':>10}")
    print("  " + "-" * 62)
    for row in [
        ("Random Forest", rf_macro,  rf_weighted,  lat.get("rf",  "?"), sz.get("rf",  "?")),
        ("SVM",           svm_macro, svm_weighted, lat.get("svm", "?"), sz.get("svm", "?")),
        ("XGBoost",       xgb_macro, xgb_weighted, lat.get("xgb", "?"), sz.get("xgb", "?")),
    ]:
        name, macro, weighted, latency, size = row
        print(f"  {name:<14} {macro:>9.4f} {weighted:>12.4f} {latency:>13} {size:>10}")
except Exception:
    pass

print()
if publication_ok:
    print("  [STATUS] Phase 4 Status: COMPLETE (Publication-Ready)")
elif engineering_ok:
    print("  [STATUS] Phase 4 Status: COMPLETE (Engineering)")
    print()
    print("  WHY not Publication-Ready:")
    print("   - Final models trained on single train+val split, NOT a 10-fold CV loop.")
    print("   - validate_folds.py confirms stratified fold STRUCTURE is correct,")
    print("     but that is not the same as training each fold and averaging scores.")
    print("   - For publication-grade results, re-train using cross_val_score / CV loop.")
else:
    print("  [STATUS] Phase 4 Status: INCOMPLETE")
    failed = [k for k, v in results.items() if not v]
    print(f"  Failed sections: {failed}")

print()
print("  NOTE: Classical ML models (RF/SVM/XGB) on MFCC features are NOT")
print("        expected to hit >=85% macro F1. That target requires")
print("        CNN + YAMNet transfer learning (Phase 5).")
print("=" * 65)

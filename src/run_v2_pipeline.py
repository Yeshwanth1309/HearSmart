"""
Master orchestrator — runs Phases 3→6 in one go.
Usage: python src/run_v2_pipeline.py
"""
import subprocess, sys, time, os

PYTHON = sys.executable
ENV = {**os.environ, "PYTHONPATH": "c:/Users/gyesh/OneDrive/Desktop/hearing_aid",
       "TF_CPP_MIN_LOG_LEVEL": "3", "TF_ENABLE_ONEDNN_OPTS": "0"}
CWD = "c:/Users/gyesh/OneDrive/Desktop/hearing_aid"

PHASES = [
    ("Phase 3 — Label Remapping",      "src/phase3_remap.py"),
    ("Phase 4 — Feature Extraction",   "src/phase4_features.py"),
    ("Phase 5a — Traditional ML",      "src/phase5a_train_ml.py"),
    ("Phase 5b — Unified CNN",         "src/phase5b_train_cnn.py"),
    ("Phase 5c — Unified YAMNet",      "src/phase5c_train_yamnet.py"),
    ("Phase 6 — Ensemble Optimize",    "src/phase6_ensemble.py"),
]

t_total = time.time()
for name, script in PHASES:
    print(f"\n{'='*64}")
    print(f"  ▶  {name}")
    print(f"{'='*64}")
    t0 = time.time()
    result = subprocess.run([PYTHON, script], env=ENV, cwd=CWD)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ❌ {name} FAILED (exit {result.returncode}) — stopping.")
        sys.exit(result.returncode)
    print(f"\n  ✅ {name} done in {elapsed/60:.1f} min")

print(f"\n{'='*64}")
print(f"  🎉 ALL PHASES COMPLETE in {(time.time()-t_total)/60:.1f} min")
print(f"{'='*64}")

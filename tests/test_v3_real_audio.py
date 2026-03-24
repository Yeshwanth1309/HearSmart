"""
HearSmart V3 — Real Audio Validation (Task 2)
Uses REAL audio from ESC-50 + UrbanSound8K demo samples.
Builds full confusion matrix on real-world data.
"""
import os, sys, io, time, collections
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np

CLASS_NAMES = [
    "speech", "siren", "horn", "traffic",
    "construction", "dog_bark", "music", "background_noise"
]

# ESC-50 category -> our 8-class mapping
ESC50_MAP = {
    # SPEECH
    "crying_baby": "speech", "clapping": "speech", "laughing": "speech",
    "coughing": "speech", "breathing": "speech", "sneezing": "speech",
    "footsteps": "speech",
    # SIREN
    "siren": "siren", "clock_alarm": "siren",
    # HORN
    "car_horn": "horn",
    # TRAFFIC
    "helicopter": "traffic", "engine": "traffic", "train": "traffic",
    "airplane": "traffic",
    # CONSTRUCTION
    "chainsaw": "construction", "hand_saw": "construction",
    "door_wood_knock": "construction", "glass_breaking": "construction",
    "fireworks": "construction",
    # DOG BARK
    "dog": "dog_bark",
    # MUSIC
    "church_bells": "music",
    # BACKGROUND
    "rain": "background_noise", "sea_waves": "background_noise",
    "wind": "background_noise", "crickets": "background_noise",
    "chirping_birds": "background_noise", "crackling_fire": "background_noise",
    "water_drops": "background_noise", "thunderstorm": "background_noise",
    "vacuum_cleaner": "background_noise", "clock_tick": "background_noise",
    "rooster": "background_noise", "pig": "background_noise",
    "cow": "background_noise", "frog": "background_noise",
    "cat": "background_noise", "hen": "background_noise",
    "insects": "background_noise", "sheep": "background_noise",
    "crow": "background_noise",
}

# Minimum samples per class for validation
MIN_SAMPLES = 3
TARGET_SAMPLES = {
    "speech": 3, "siren": 4, "horn": 3, "traffic": 3,
    "construction": 3, "dog_bark": 3, "music": 3, "background_noise": 3,
}

SAFETY_CLASSES = {"siren", "horn"}


def find_real_audio_samples():
    """Find real audio samples from ESC-50 and demo/samples."""
    esc50_meta = Path("hearing_aid/data/ESC50/meta/esc50.csv")
    esc50_audio = Path("hearing_aid/data/ESC50/audio")

    samples_by_class = {c: [] for c in CLASS_NAMES}

    # 1. ESC-50 real audio
    if esc50_meta.exists():
        df = pd.read_csv(str(esc50_meta))
        for _, row in df.iterrows():
            cat = row["category"]
            mapped = ESC50_MAP.get(cat)
            if not mapped:
                continue

            fpath = esc50_audio / row["filename"]
            if fpath.exists():
                target = TARGET_SAMPLES.get(mapped, 3)
                if len(samples_by_class[mapped]) < target:
                    samples_by_class[mapped].append({
                        "path": str(fpath),
                        "source": "ESC-50",
                        "original_cat": cat,
                        "desc": f"{cat} ({row['filename']})",
                    })

    # 2. UrbanSound8K demo samples (dog bark)
    demo_dir = Path("demo/samples")
    for f in sorted(demo_dir.glob("101415-*.wav")):
        if len(samples_by_class["dog_bark"]) < TARGET_SAMPLES["dog_bark"]:
            samples_by_class["dog_bark"].append({
                "path": str(f),
                "source": "UrbanSound8K",
                "original_cat": "dog_bark",
                "desc": f"dog_bark ({f.name})",
            })

    return samples_by_class


def run_validation():
    from src.pipeline_v3 import get_engine
    import src.pipeline_v3 as pv3
    pv3._engine = None

    print("=" * 75)
    print("  HearSmart V3 — REAL AUDIO Validation")
    print("=" * 75)

    # Find real samples
    samples_by_class = find_real_audio_samples()

    print("\n  Audio samples found:")
    total_samples = 0
    for cls in CLASS_NAMES:
        n = len(samples_by_class[cls])
        total_samples += n
        status = "OK" if n >= MIN_SAMPLES else "LOW"
        print(f"    {cls:20s}: {n} samples [{status}]")

    if total_samples == 0:
        print("\n  [FATAL] No real audio samples found.")
        return

    # Load engine
    print(f"\n  Loading engine...")
    engine = get_engine()
    engine._ensure_loaded()
    print(f"  Engine ready.\n")

    # Run inference on all samples
    confusion = {c: collections.Counter() for c in CLASS_NAMES}
    per_class_correct = {c: 0 for c in CLASS_NAMES}
    per_class_total = {c: 0 for c in CLASS_NAMES}
    safety_tp = 0
    safety_fn = 0
    safety_fp = 0
    safety_tn = 0
    all_latencies = []
    missed_safety = []
    false_safety = []

    for true_class in CLASS_NAMES:
        samples = samples_by_class[true_class]
        if not samples:
            print(f"  --- {true_class.upper()} --- (no samples)")
            continue

        print(f"\n  --- {true_class.upper()} ({len(samples)} samples) ---")

        for sample in samples:
            engine.reset_buffers()
            fpath = sample["path"]

            try:
                # Run twice for temporal confirmation
                engine.infer(fpath)
                result = engine.infer(fpath)
                pred = result.environment
                all_latencies.append(result.latency_ms)

                confusion[true_class][pred] += 1
                per_class_total[true_class] += 1
                if pred == true_class:
                    per_class_correct[true_class] += 1

                match = "OK  " if pred == true_class else "MISS"

                # Safety analysis
                is_true_safety = true_class in SAFETY_CLASSES
                is_pred_safety = result.safety_verdict

                if is_true_safety and is_pred_safety:
                    safety_tp += 1
                elif is_true_safety and not is_pred_safety:
                    safety_fn += 1
                    missed_safety.append((sample["desc"], true_class, result))
                elif not is_true_safety and is_pred_safety:
                    safety_fp += 1
                    false_safety.append((sample["desc"], true_class, result))
                else:
                    safety_tn += 1

                safety_flag = ""
                if is_true_safety and not is_pred_safety:
                    safety_flag = " [SAFETY MISSED!]"
                elif not is_true_safety and is_pred_safety:
                    safety_flag = " [FALSE ALARM!]"

                print(f"    [{match}] {sample['desc'][:45]:45s}"
                      f" -> {pred:18s} conf={result.confidence:.1%}"
                      f" safety={result.safety_probability:.3f}"
                      f" {result.latency_ms:.0f}ms{safety_flag}")

                # Print YAMNet/CNN detail for misclassifications
                if pred != true_class:
                    y_name, y_conf, _ = result.per_model["YAMNet"]
                    c_name, c_conf, _ = result.per_model["CNN"]
                    print(f"          YAMNet={y_name} ({y_conf:.1%})  "
                          f"CNN={c_name} ({c_conf:.1%})  "
                          f"src={result.safety_source}")

            except Exception as e:
                print(f"    [ERR] {sample['desc'][:45]}: {e}")

    # ─── CONFUSION MATRIX ────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  CONFUSION MATRIX (rows=true, cols=predicted)")
    print("=" * 75)

    SHORT = {c: c[:7] for c in CLASS_NAMES}
    lbl = "TRUE\\PRED"
    header = f"  {lbl:14s} | " + " ".join(f"{SHORT[c]:>8s}" for c in CLASS_NAMES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for true_c in CLASS_NAMES:
        row = f"  {true_c:14s} | "
        for pred_c in CLASS_NAMES:
            cnt = confusion[true_c][pred_c]
            if cnt > 0:
                mark = f"[{cnt}]" if true_c == pred_c else f" {cnt} "
                row += f"{mark:>8s}"
            else:
                row += f"{'  .':>8s}"
        print(row)

    # ─── ACCURACY ─────────────────────────────────────────────────────────
    total_correct = sum(per_class_correct.values())
    total_tested = sum(per_class_total.values())
    overall_acc = total_correct / total_tested if total_tested else 0

    print(f"\n  Overall Accuracy: {total_correct}/{total_tested} = {overall_acc*100:.1f}%")
    print(f"\n  Per-class accuracy:")
    for cls in CLASS_NAMES:
        n = per_class_total[cls]
        c = per_class_correct[cls]
        acc = c / n * 100 if n > 0 else 0
        bar = "#" * int(acc / 10)
        print(f"    {cls:20s}: {c}/{n} = {acc:5.1f}% [{bar:<10s}]")

    # ─── SAFETY METRICS ──────────────────────────────────────────────────
    print(f"\n  Safety Metrics:")
    safety_total = safety_tp + safety_fn
    recall = safety_tp / safety_total * 100 if safety_total > 0 else 0
    nonsafety_total = safety_fp + safety_tn
    fp_rate = safety_fp / nonsafety_total * 100 if nonsafety_total > 0 else 0

    print(f"    True Positives  (safety detected):   {safety_tp}")
    print(f"    False Negatives (safety missed):      {safety_fn}")
    print(f"    False Positives (non-safety flagged): {safety_fp}")
    print(f"    True Negatives  (non-safety clear):   {safety_tn}")
    print(f"\n    Safety Recall:     {safety_tp}/{safety_total} = {recall:.1f}%"
          f"  {'[PASS >= 95%]' if recall >= 95 else '[FAIL < 95%]'}")
    print(f"    False Positive Rate: {safety_fp}/{nonsafety_total} = {fp_rate:.1f}%"
          f"  {'[PASS < 30%]' if fp_rate < 30 else '[FAIL >= 30%]'}")

    if missed_safety:
        print(f"\n    MISSED SAFETY SOUNDS ({len(missed_safety)}):")
        for desc, tc, r in missed_safety:
            print(f"      {desc} (true={tc}, pred={r.environment}, "
                  f"prob={r.safety_probability:.3f})")

    if false_safety:
        print(f"\n    FALSE ALARMS ({len(false_safety)}):")
        for desc, tc, r in false_safety:
            print(f"      {desc} (true={tc}, pred={r.environment}, "
                  f"prob={r.safety_probability:.3f}, src={r.safety_source})")

    # ─── LATENCY ─────────────────────────────────────────────────────────
    if all_latencies:
        warm = all_latencies[1:] if len(all_latencies) > 1 else all_latencies
        print(f"\n  Latency (warm):")
        print(f"    Average: {sum(warm)/len(warm):.1f}ms")
        print(f"    Max:     {max(warm):.1f}ms")
        print(f"    Min:     {min(warm):.1f}ms")

    # ─── THRESHOLD RECOMMENDATION ────────────────────────────────────────
    print(f"\n  Threshold Status:")
    from src.safety_config import (YAMNET_SIREN_THRESHOLD, YAMNET_HORN_THRESHOLD,
                                    CNN_SIREN_THRESHOLD, CNN_HORN_THRESHOLD,
                                    SAFETY_FRAMES_REQUIRED)
    need_adjust = False
    if fp_rate >= 30:
        print(f"    FP rate {fp_rate:.1f}% >= 30% — consider raising thresholds")
        need_adjust = True
    if recall < 95:
        print(f"    Recall {recall:.1f}% < 95% — consider lowering thresholds")
        need_adjust = True

    if not need_adjust:
        print(f"    All targets met — LOCKING current thresholds:")
    else:
        print(f"    Current thresholds (may need adjustment):")

    print(f"      YAMNET_SIREN_THRESHOLD  = {YAMNET_SIREN_THRESHOLD}")
    print(f"      YAMNET_HORN_THRESHOLD   = {YAMNET_HORN_THRESHOLD}")
    print(f"      CNN_SIREN_THRESHOLD     = {CNN_SIREN_THRESHOLD}")
    print(f"      CNN_HORN_THRESHOLD      = {CNN_HORN_THRESHOLD}")
    print(f"      SAFETY_FRAMES_REQUIRED  = {SAFETY_FRAMES_REQUIRED}")

    print("\n" + "=" * 75)

    return overall_acc, recall, fp_rate


if __name__ == "__main__":
    run_validation()

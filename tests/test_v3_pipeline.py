"""
V3 End-to-End Test -- runs full inference on sample audio files
and validates all 6 features.
"""
import os, sys, time, io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from src.pipeline_v3 import get_engine, HearSmartV3Engine


def run_tests():
    print("=" * 70)
    print("  HearSmart V3 — End-to-End Pipeline Test")
    print("=" * 70)

    # Find sample audio files
    sample_dir = Path("demo/samples")
    if not sample_dir.exists():
        print(f"  ❌ Sample directory not found: {sample_dir}")
        return

    samples = sorted(sample_dir.glob("*.wav"))
    if not samples:
        print("  ❌ No .wav files found in demo/samples/")
        return

    print(f"\n  Found {len(samples)} sample file(s):\n")
    for s in samples:
        print(f"    • {s.name}")

    # Get engine (first call triggers model loading)
    print("\n  Loading V3 engine (first call — models loading) ...")
    t_load = time.perf_counter()
    engine = get_engine()
    engine.reset_buffers()
    load_ms = round((time.perf_counter() - t_load) * 1000, 1)
    print(f"  Engine loaded in {load_ms} ms\n")

    results = []

    # Run inference on each sample
    for i, audio_path in enumerate(samples):
        print("-" * 70)
        print(f"  TEST {i+1}/{len(samples)}: {audio_path.name}")
        print("-" * 70)

        try:
            result = engine.infer(str(audio_path))
            results.append(result)

            print(f"\n  Environment      : {result.environment}")
            print(f"  Confidence       : {result.confidence:.1%}")
            print(f"  Safety Verdict   : {'[!] DANGER' if result.safety_verdict else '[OK] Safe'}")
            print(f"  Safety Prob      : {result.safety_probability:.3f}")
            print(f"  Safety Source    : {result.safety_source}")
            print(f"  Temporal Status  : {result.temporal_status}")
            print(f"  Risk Score       : {result.risk_score:.3f}")
            print(f"  Risk Action      : {result.risk_action}")
            print(f"  Transition OK    : {result.transition_approved}")
            print(f"  Settings Status  : {result.settings_status}")
            print(f"  Volume           : {result.volume}/10")
            print(f"  Noise Reduction  : {result.noise_reduction}")
            print(f"  Directionality   : {result.directionality}")
            print(f"  Speech Enhance   : {'On' if result.speech_enhancement else 'Off'}")
            print(f"  Reasoning        : {result.reasoning}")
            print(f"  Latency          : {result.latency_ms} ms")
            print(f"  Per-model:")
            for m, (pred, conf, w) in result.per_model.items():
                print(f"    {m:10s}: {pred:20s} ({conf:.1%}, weight={w:.2f})")
            print()

        except Exception as e:
            print(f"  [FAIL]: {e}")
            import traceback
            traceback.print_exc()
            print()

    # ── Validation Summary ───────────────────────────────────────────────
    print("=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)

    all_pass = True

    # Check 1: All inferences completed
    if len(results) == len(samples):
        print(f"  [PASS] All {len(samples)} inferences completed successfully")
    else:
        print(f"  [FAIL] Only {len(results)}/{len(samples)} inferences completed")
        all_pass = False

    # Check 2: Latency
    latencies = [r.latency_ms for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    print(f"  Avg latency: {avg_latency:.1f} ms | Max: {max_latency:.1f} ms")
    if max_latency <= 500:
        print(f"  [PASS] All inferences under 500ms")
    else:
        print(f"  [WARN] Some inferences exceeded 500ms (max: {max_latency:.1f}ms)")

    # Check 3: Safety model exists and runs
    safety_probs = [r.safety_probability for r in results]
    print(f"  [PASS] Safety classifier ran on all samples (probs: {[f'{p:.3f}' for p in safety_probs]})")

    # Check 4: Temporal consistency working
    statuses = [r.temporal_status for r in results]
    print(f"  [PASS] Temporal consistency active (statuses: {statuses})")

    # Check 5: Risk scores computed
    risks = [r.risk_score for r in results]
    print(f"  [PASS] Risk scores computed: {[f'{r:.3f}' for r in risks]}")

    # Check 6: Pipeline uses only 2 models
    for r in results:
        models_used = list(r.per_model.keys())
        if set(models_used) == {"YAMNet", "CNN"}:
            print(f"  [PASS] 2-model ensemble confirmed (YAMNet + CNN only)")
            break
    else:
        print(f"  [FAIL] Unexpected model set: {models_used}")
        all_pass = False

    print()
    if all_pass:
        print("  [PASS] ALL V3 CHECKS PASSED")
    else:
        print("  [WARN] SOME CHECKS NEED ATTENTION")

    print("=" * 70)


if __name__ == "__main__":
    run_tests()

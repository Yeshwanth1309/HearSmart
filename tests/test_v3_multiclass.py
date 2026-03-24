"""
V3 Multi-Class Pipeline Test — tests all 8 sound classes
including safety-critical sounds (siren, horn, gunshot).
"""
import os, sys, time, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from src.pipeline_v3 import get_engine


def run_multi_test():
    print("=" * 75)
    print("  HearSmart V3 -- Multi-Class Pipeline Test (All 8 Classes)")
    print("=" * 75)

    # Test files mapped to expected class categories
    test_files = {
        "siren_test.wav":        {"expected": "siren",       "safety": True},
        "horn_test.wav":         {"expected": "horn",        "safety": True},
        "gunshot_test.wav":      {"expected": "construction","safety": False},
        "traffic_test.wav":      {"expected": "traffic",     "safety": False},
        "construction_test.wav": {"expected": "construction","safety": False},
        "music_test.wav":        {"expected": "music",       "safety": False},
        "speech_test.wav":       {"expected": "speech",      "safety": False},
        "background_test.wav":   {"expected": "background_noise", "safety": False},
        "101415-3-0-2.wav":      {"expected": "dog_bark",    "safety": False},
    }

    sample_dir = Path("demo/samples")
    available = {}
    for name, info in test_files.items():
        fpath = sample_dir / name
        if fpath.exists():
            available[name] = info
        else:
            print(f"  [SKIP] {name} not found")

    print(f"\n  Testing {len(available)} audio files\n")

    # Load engine
    print("  Loading V3 engine...")
    engine = get_engine()
    engine.reset_buffers()
    print("  Engine ready.\n")

    results = []
    safety_checks = []

    for name, info in available.items():
        # Reset buffers between unrelated tests
        engine.reset_buffers()

        fpath = str(sample_dir / name)
        print("-" * 75)
        print(f"  FILE: {name}")
        print(f"  Expected class: {info['expected']}  |  Safety expected: {info['safety']}")
        print("-" * 75)

        try:
            result = engine.infer(fpath)
            results.append((name, info, result))

            # Classification result
            match = "MATCH" if result.environment == info["expected"] else "MISMATCH"
            print(f"\n  Classification : {result.environment:20s} [{match}]")
            print(f"  Confidence     : {result.confidence:.1%}")

            # Safety analysis
            safety_match = ""
            if info["safety"]:
                if result.safety_verdict:
                    safety_match = "[CORRECT]"
                    safety_checks.append(True)
                else:
                    safety_match = "[MISSED!]"
                    safety_checks.append(False)
            else:
                if result.safety_verdict:
                    safety_match = "[FALSE ALARM]"
                    safety_checks.append(False)
                else:
                    safety_match = "[CORRECT]"
                    safety_checks.append(True)

            safety_icon = "[!] DANGER" if result.safety_verdict else "[OK] Safe"
            print(f"  Safety Verdict : {safety_icon}  {safety_match}")
            print(f"  Safety Prob    : {result.safety_probability:.3f}")
            print(f"  Safety Source  : {result.safety_source}")

            # V3 features
            print(f"  Temporal       : {result.temporal_status}")
            print(f"  Risk Score     : {result.risk_score:.3f} ({result.risk_action})")
            print(f"  Settings       : {result.settings_status}")

            # Applied settings
            print(f"  Volume         : {result.volume}/10")
            print(f"  Noise Red.     : {result.noise_reduction}")
            print(f"  Directionality : {result.directionality}")
            print(f"  Speech Enh.    : {'On' if result.speech_enhancement else 'Off'}")

            # Per-model breakdown
            print(f"  Latency        : {result.latency_ms:.1f} ms")
            for m, (pred, conf, w) in result.per_model.items():
                print(f"    {m:8s}: {pred:20s} conf={conf:.1%}  w={w:.2f}")
            print()

        except Exception as e:
            print(f"  [FAIL]: {e}\n")
            import traceback
            traceback.print_exc()

    # ── Summary Report ───────────────────────────────────────────────────
    print("=" * 75)
    print("  MULTI-CLASS TEST SUMMARY")
    print("=" * 75)

    total = len(results)
    correct_class = sum(1 for _, info, r in results if r.environment == info["expected"])
    print(f"\n  Classification Accuracy: {correct_class}/{total} "
          f"({correct_class/total*100:.0f}%)")

    print(f"\n  {'File':<25s} {'Expected':<18s} {'Predicted':<18s} {'Match':<8s} "
          f"{'Safety':<10s} {'Risk':<6s} {'Latency':<8s}")
    print("  " + "-" * 93)

    for name, info, r in results:
        match = "OK" if r.environment == info["expected"] else "MISS"
        safety = "OVERRIDE" if r.safety_verdict else "safe"
        print(f"  {name:<25s} {info['expected']:<18s} {r.environment:<18s} "
              f"{match:<8s} {safety:<10s} {r.risk_score:<6.3f} {r.latency_ms:<8.1f}")

    # Safety-specific analysis
    print(f"\n  --- SAFETY ANALYSIS ---")
    safety_files = [(n, i, r) for n, i, r in results if i["safety"]]
    nonsafety_files = [(n, i, r) for n, i, r in results if not i["safety"]]

    if safety_files:
        detected = sum(1 for _, _, r in safety_files if r.safety_verdict)
        print(f"  Safety sounds detected:  {detected}/{len(safety_files)}")
        for name, info, r in safety_files:
            status = "DETECTED" if r.safety_verdict else "MISSED"
            print(f"    {name:<25s} -> {status}  "
                  f"(prob={r.safety_probability:.3f}, src={r.safety_source})")

    if nonsafety_files:
        false_alarms = sum(1 for _, _, r in nonsafety_files if r.safety_verdict)
        print(f"  False alarms:            {false_alarms}/{len(nonsafety_files)}")
        if false_alarms:
            for name, info, r in nonsafety_files:
                if r.safety_verdict:
                    print(f"    [FALSE ALARM] {name} -> prob={r.safety_probability:.3f}")

    # Latency analysis
    latencies = [r.latency_ms for _, _, r in results]
    # Exclude first call (cold start)
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
    print(f"\n  --- LATENCY ANALYSIS ---")
    print(f"  First call (cold):  {latencies[0]:.1f} ms")
    if warm_latencies:
        print(f"  Warm avg:           {sum(warm_latencies)/len(warm_latencies):.1f} ms")
        print(f"  Warm max:           {max(warm_latencies):.1f} ms")
        print(f"  Warm min:           {min(warm_latencies):.1f} ms")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    run_multi_test()

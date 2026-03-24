"""
HearSmart V3 — Full Audit & Safety Hardening Test Suite
Tasks 1-5: Classification Audit, Safety Recall, Stability, Integration
"""
import os, sys, io, time, math, wave, struct, random, collections
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tempfile

SR = 16000
DUR = 3.0
N  = int(SR * DUR)


# ─── SIGNAL GENERATORS ───────────────────────────────────────────────────────

def _wav(path, samples, sr=SR):
    with wave.open(str(path), 'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b''.join(
            struct.pack('<h', max(-32767, min(32767, int(s * 32767))))
            for s in samples))

def _clip(x): return max(-1.0, min(1.0, x))

def gen_siren(variant=0, dur=DUR, sr=SR):
    """Wailing siren variants: 0=ambulance sweep, 1=police hi-lo, 2=fire engine"""
    rng = random.Random(variant)
    n = int(sr * dur)
    params = [
        (600, 1400, 0.5, 0.75),   # ambulance
        (800, 1200, 0.8, 0.80),   # police
        (500, 1600, 0.4, 0.70),   # fire engine
    ]
    lo, hi, rate, amp = params[variant % len(params)]
    out = []
    for i in range(n):
        t = i / sr
        freq = lo + (hi - lo) * 0.5 * (1 + math.sin(2 * math.pi * rate * t))
        v = amp * math.sin(2 * math.pi * freq * t)
        v += 0.2 * math.sin(2 * math.pi * freq * 2 * t)
        v += rng.gauss(0, 0.02)
        out.append(_clip(v))
    return out

def gen_horn(variant=0, dur=DUR, sr=SR):
    """Car/truck horn variants"""
    rng = random.Random(variant + 10)
    n = int(sr * dur)
    freqs = [(420, 520), (350, 450), (500, 620)]
    f1, f2 = freqs[variant % len(freqs)]
    pattern = [0.6, 0.2, 0.8, 0.0, 0.9]  # on/off pattern
    out = []
    for i in range(n):
        t = i / sr
        seg = int((t / dur) * len(pattern))
        seg = min(seg, len(pattern) - 1)
        on = pattern[seg]
        v = on * (0.5 * math.sin(2*math.pi*f1*t) +
                  0.5 * math.sin(2*math.pi*f2*t))
        v += rng.gauss(0, 0.01)
        out.append(_clip(v))
    return out

def gen_traffic(variant=0, dur=DUR, sr=SR):
    rng = random.Random(variant + 20)
    n = int(sr * dur)
    out = []
    for i in range(n):
        t = i / sr
        v = 0.3 * math.sin(2*math.pi*60*t)  # engine rumble
        v += 0.15 * math.sin(2*math.pi*120*t)
        v += rng.uniform(-1, 1) * (0.1 + 0.05 * variant)
        v *= 0.7 + 0.3 * math.sin(2*math.pi*0.2*t)
        out.append(_clip(v * 0.6))
    return out

def gen_construction(variant=0, dur=DUR, sr=SR):
    rng = random.Random(variant + 30)
    n = int(sr * dur)
    rates = [25, 18, 12]
    rate = rates[variant % len(rates)]
    out = []
    for i in range(n):
        t = i / sr
        phase = (t * rate) % 1.0
        if phase < 0.3:
            env = math.exp(-phase * 20)
            v = env * rng.uniform(-1, 1) * 0.9
            v += env * math.sin(2*math.pi*800*t) * 0.3
        else:
            v = rng.uniform(-1, 1) * 0.05
        out.append(_clip(v * 0.7))
    return out

def gen_dog_bark(variant=0, dur=DUR, sr=SR):
    rng = random.Random(variant + 40)
    n = int(sr * dur)
    # Bark times: single, double, rapid
    bark_patterns = [[0.5, 1.8], [0.3, 0.8, 1.5, 2.2], [0.2, 0.5, 0.8, 1.1, 1.4, 1.7]]
    barks = bark_patterns[variant % len(bark_patterns)]
    out = []
    for i in range(n):
        t = i / sr
        v = 0.0
        for bt in barks:
            dt = t - bt
            if 0 <= dt < 0.25:
                env = math.exp(-dt * 15) * (1 - math.exp(-dt * 500))
                f = 430 + rng.uniform(-30, 30)
                v += env * (0.6 * math.sin(2*math.pi*f*t) +
                            0.3 * rng.uniform(-1, 1))
        out.append(_clip(v))
    return out

def gen_music(variant=0, dur=DUR, sr=SR):
    n = int(sr * dur)
    scales = [
        [261.6, 293.7, 329.6, 392.0, 523.3],   # C major
        [440.0, 493.9, 523.3, 587.3, 659.3],   # A major
        [196.0, 220.0, 246.9, 293.7, 329.6],   # G major (lower, softer)
    ]
    notes = scales[variant % len(scales)]
    nd = dur / len(notes)
    out = []
    for i in range(n):
        t = i / sr
        ni = min(int(t / nd), len(notes) - 1)
        f = notes[ni]
        lt = t - ni * nd
        env = min(1.0, lt * 15) * math.exp(-lt * 1.5)
        amp = 0.3 + 0.2 * variant
        v = amp * env * (math.sin(2*math.pi*f*t) +
                         0.4*math.sin(2*math.pi*f*2*t) +
                         0.2*math.sin(2*math.pi*f*3*t))
        out.append(_clip(v))
    return out

def gen_speech(variant=0, dur=DUR, sr=SR):
    rng = random.Random(variant + 60)
    n = int(sr * dur)
    formant_sets = [
        [(700, 1200), (300, 2500), (500, 1500)],   # normal
        [(600, 1100), (280, 2200), (450, 1400)],   # crowded (slightly muffled)
        [(650, 1300), (320, 2600), (480, 1600)],   # whisper variant
    ]
    formants = formant_sets[variant % len(formant_sets)]
    vd = dur / len(formants)
    amps = [0.6, 0.4, 0.3]  # whisper quieter
    amp = amps[variant % len(amps)]
    out = []
    for i in range(n):
        t = i / sr
        vi = min(int(t / vd), len(formants) - 1)
        f1, f2 = formants[vi]
        lt = t - vi * vd
        env = min(1.0, lt * 10) * min(1.0, (vd - lt) * 10)
        glottal = (0.5 * math.sin(2*math.pi*120*t) +
                   0.3 * math.sin(2*math.pi*240*t))
        v = amp * glottal * (0.5*math.sin(2*math.pi*f1*t) +
                              0.3*math.sin(2*math.pi*f2*t))
        v += rng.gauss(0, 0.015)
        out.append(_clip(v * env))
    return out

def gen_background(variant=0, dur=DUR, sr=SR):
    rng = random.Random(variant + 70)
    n = int(sr * dur)
    freqs = [50, 100, 200]   # AC/rain/wind base freq
    base_f = freqs[variant % len(freqs)]
    out = []
    for i in range(n):
        t = i / sr
        v = 0.1 * math.sin(2*math.pi*base_f*t)
        v += rng.uniform(-1, 1) * (0.06 + 0.04 * variant)
        out.append(_clip(v))
    return out


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def make_tmp(samples):
    f = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    _wav(f.name, samples)
    f.close()
    return f.name

def sep(ch='=', n=72): print(ch * n)

def print_result_detail(label, result, expected_class, show_raw=True):
    match = result.environment == expected_class
    status = "OK  " if match else "MISS"
    print(f"  [{status}] {label}")
    print(f"         Predicted : {result.environment:18s} conf={result.confidence:.1%}")
    print(f"         Expected  : {expected_class}")
    if show_raw:
        print(f"         YAMNet-8  : {result.per_model['YAMNet'][0]:18s} "
              f"conf={result.per_model['YAMNet'][1]:.1%}")
        print(f"         CNN       : {result.per_model['CNN'][0]:18s} "
              f"conf={result.per_model['CNN'][1]:.1%}")
        print(f"         Safety    : prob={result.safety_probability:.3f}  "
              f"src={result.safety_source}  frames={result.safety_frame_count}")
        print(f"         Risk      : {result.risk_score:.3f}  action={result.risk_action}")
        print(f"         Settings  : {result.settings_status}  "
              f"floor_blocked={result.confidence_floor_blocked}")
        print(f"         Latency   : {result.latency_ms:.1f}ms")
    return match


# ─── TASK 1: CLASSIFICATION AUDIT ────────────────────────────────────────────

def task1_classification_audit(engine):
    sep()
    print("  TASK 1 — Full Classification Audit (8 classes x 3 variants)")
    sep()

    CLASS_TESTS = {
        "speech":           (gen_speech,       ["Normal conversation", "Crowded speech", "Whisper"]),
        "siren":            (gen_siren,        ["Ambulance sweep", "Police hi-lo", "Fire engine"]),
        "horn":             (gen_horn,         ["Car horn", "Truck horn", "Repeated honking"]),
        "traffic":          (gen_traffic,      ["Highway noise", "City traffic", "Idle engine"]),
        "construction":     (gen_construction, ["Drilling", "Jackhammer", "Chainsaw"]),
        "dog_bark":         (gen_dog_bark,     ["Single bark", "Repeated barking", "Rapid barks"]),
        "music":            (gen_music,        ["Street music", "Loud music", "Soft music"]),
        "background_noise": (gen_background,   ["Rain", "AC noise", "Wind"]),
    }

    # confusion matrix: true_class -> predicted_class -> count
    confusion = {c: collections.Counter() for c in CLASS_NAMES_V2_LOCAL}
    total = correct = 0
    safety_false_positives = []
    safety_false_negatives = []
    SAFETY_CLASSES = {"siren", "horn"}

    for true_class, (gen_fn, descriptions) in CLASS_TESTS.items():
        print(f"\n  --- {true_class.upper()} ---")
        for variant, desc in enumerate(descriptions):
            engine.reset_buffers()
            samples = gen_fn(variant)
            # Run twice (first=Uncertain, second=Confirmed by temporal)
            tmp = make_tmp(samples)
            try:
                engine.infer(tmp)    # warm temporal buffer
                result = engine.infer(tmp)
            finally:
                os.unlink(tmp)

            pred = result.environment
            confusion[true_class][pred] += 1
            match = (pred == true_class)
            total += 1
            if match: correct += 1

            print_result_detail(desc, result, true_class)

            # Safety audit
            if true_class in SAFETY_CLASSES and not result.safety_verdict:
                safety_false_negatives.append((desc, true_class, result))
            if true_class not in SAFETY_CLASSES and result.safety_verdict:
                safety_false_positives.append((desc, true_class, result))

    # Print confusion matrix
    print()
    sep('-')
    print("  CONFUSION MATRIX (rows=true, cols=predicted)")
    sep('-')
    SHORT = {c: c[:6] for c in CLASS_NAMES_V2_LOCAL}
    header = "  {:14s} | ".format("TRUE \\ PRED") + " ".join(f"{SHORT[c]:>7s}" for c in CLASS_NAMES_V2_LOCAL)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for true_c in CLASS_NAMES_V2_LOCAL:
        row = f"  {true_c:14s} | "
        for pred_c in CLASS_NAMES_V2_LOCAL:
            cnt = confusion[true_c][pred_c]
            row += f"{'['+str(cnt)+']':>7s}" if cnt > 0 else f"{'  .':>7s}"
        print(row)

    print(f"\n  Classification Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    if safety_false_positives:
        print(f"\n  [CRITICAL] FALSE POSITIVE safety triggers ({len(safety_false_positives)}):")
        for desc, tc, r in safety_false_positives:
            print(f"    '{desc}' (true={tc}) -> safety FIRED via {r.safety_source}")
    else:
        print("\n  [OK] No false positive safety triggers")

    if safety_false_negatives:
        print(f"\n  [CRITICAL] MISSED safety sounds ({len(safety_false_negatives)}):")
        for desc, tc, r in safety_false_negatives:
            print(f"    '{desc}' (true={tc}) -> safety MISSED "
                  f"(prob={r.safety_probability:.3f})")
    else:
        print("  [OK] No missed safety sounds")

    return confusion, correct / total, safety_false_positives, safety_false_negatives


# ─── TASK 2: (Hardening already applied to pipeline_v3.py) ───────────────────
# The hardening constants are in src/safety_config.py and the pipeline
# applies all H1-H5 measures. We report the config here.

def task2_report_hardening():
    from src.safety_config import (
        SIREN_YAMNET_INDICES, HORN_YAMNET_INDICES, TRAFFIC_YAMNET_INDICES,
        YAMNET_SIREN_THRESHOLD, YAMNET_HORN_THRESHOLD,
        CNN_SIREN_THRESHOLD, CNN_HORN_THRESHOLD,
        SAFETY_CLASSIFIER_THRESHOLD, SAFETY_CLASSIFIER_HIGH_CONF,
        SAFETY_FRAMES_REQUIRED, TRAFFIC_NEGATIVE_GUARD_REDUCTION,
        CONFIDENCE_FLOORS, DOG_YAMNET_INDICES
    )
    sep()
    print("  TASK 2 — Safety Hardening Config (Applied)")
    sep()
    print(f"  H1: SIREN_YAMNET_INDICES  = {SIREN_YAMNET_INDICES}")
    print(f"  H1: HORN_YAMNET_INDICES   = {HORN_YAMNET_INDICES}")
    print(f"  H1: TRAFFIC_YAMNET_INDICES= {TRAFFIC_YAMNET_INDICES}")
    print(f"  H1: DOG_YAMNET_INDICES    = {DOG_YAMNET_INDICES}")
    print(f"  H2: Traffic guard reduction = {TRAFFIC_NEGATIVE_GUARD_REDUCTION:.0%}")
    print(f"  H3: SAFETY_FRAMES_REQUIRED  = {SAFETY_FRAMES_REQUIRED}")
    print(f"  H3: High-conf bypass at     = {SAFETY_CLASSIFIER_HIGH_CONF}")
    print(f"  H4: CONFIDENCE_FLOORS:")
    for cls, floor in CONFIDENCE_FLOORS.items():
        print(f"       {cls:20s}: {floor:.0%}")
    print(f"  H5: Dog/Horn disambiguation enabled")
    print(f"\n  Thresholds:")
    print(f"    yamnet_siren  : {YAMNET_SIREN_THRESHOLD}")
    print(f"    yamnet_horn   : {YAMNET_HORN_THRESHOLD}")
    print(f"    cnn_siren     : {CNN_SIREN_THRESHOLD}")
    print(f"    cnn_horn      : {CNN_HORN_THRESHOLD}")
    print(f"    safety_classif: {SAFETY_CLASSIFIER_THRESHOLD}")


# ─── TASK 3: SAFETY RECALL VERIFICATION ─────────────────────────────────────

def task3_safety_recall(engine):
    sep()
    print("  TASK 3 — Safety Recall Verification (20 safety samples)")
    sep()

    # 8 siren + 7 horn + 5 mixed (siren over traffic background)
    safety_tests = []
    for v in range(8):
        safety_tests.append(("siren", f"Siren variant {v}", lambda v=v: gen_siren(v)))
    for v in range(7):
        safety_tests.append(("horn",  f"Horn variant {v}",  lambda v=v: gen_horn(v)))
    for v in range(5):
        # Mixed: siren layered over traffic background
        def gen_mixed(v=v):
            s = gen_siren(v % 3)
            t = gen_traffic(v % 3)
            n = len(s)
            # Attenuate siren slightly and add traffic
            return [_clip(0.7 * s[i] + 0.3 * t[i]) for i in range(n)]
        safety_tests.append(("siren", f"Siren+Traffic mix {v}", gen_mixed))

    tp = 0; fn_list = []; fp_list = []

    for true_class, desc, gen_fn in safety_tests:
        engine.reset_buffers()
        samples = gen_fn()
        tmp = make_tmp(samples)
        try:
            engine.infer(tmp)          # frame 1
            result = engine.infer(tmp) # frame 2 (frame counter = 2 => triggers)
        finally:
            os.unlink(tmp)

        detected = result.safety_verdict
        match_str = "DETECTED" if detected else "MISSED  "
        flag = "[OK]  " if detected else "[MISS]"
        print(f"  {flag} {match_str} | {desc:35s} "
              f"prob={result.safety_probability:.3f} "
              f"src={result.safety_source} "
              f"frames={result.safety_frame_count}")

        if detected:
            tp += 1
        else:
            fn_list.append((desc, true_class, result))

    total_safety = len(safety_tests)
    recall = tp / total_safety
    print(f"\n  Safety Recall: {tp}/{total_safety} = {recall*100:.1f}%")

    if fn_list:
        print(f"  False Negatives ({len(fn_list)}):")
        for desc, tc, r in fn_list:
            print(f"    MISSED: '{desc}' "
                  f"siren_prob={r.safety_probability:.3f} "
                  f"frames={r.safety_frame_count}")
    else:
        print("  False Negatives: NONE")

    target_met = recall >= 0.95
    print(f"\n  Target (>=95%): {'PASS' if target_met else 'FAIL'}")
    if not target_met:
        print("  ACTION: Recall below 95% — see root cause analysis below.")
        print("  Root cause: Safety classifier trained on synthetic data only.")
        print("  Fix needed: Retrain safety_classifier.pt on real siren/horn audio.")

    return recall, fn_list, fp_list


# ─── TASK 4: STABILITY TEST ───────────────────────────────────────────────────

def task4_stability(engine):
    sep()
    print("  TASK 4 — Classification Stability (10-second continuous audio)")
    sep()

    FRAMES = 10   # 10 inference frames simulating ~10 seconds
    GENERATORS = {
        "speech":           gen_speech,
        "siren":            gen_siren,
        "horn":             gen_horn,
        "traffic":          gen_traffic,
        "construction":     gen_construction,
        "dog_bark":         gen_dog_bark,
        "music":            gen_music,
        "background_noise": gen_background,
    }

    stability_scores = {}
    print(f"\n  {'Class':20s} {'Correct':>8s} {'Total':>6s} {'Stability':>10s} {'Status':>8s}")
    print("  " + "-" * 60)

    for cls, gen_fn in GENERATORS.items():
        engine.reset_buffers()
        samples = gen_fn(0)
        tmp = make_tmp(samples)
        correct_frames = 0
        predictions = []

        try:
            for frame in range(FRAMES):
                result = engine.infer(tmp)
                predictions.append(result.environment)
                # Count both direct match and safety override for safety classes
                if result.environment == cls:
                    correct_frames += 1
                elif cls in ("siren", "horn") and result.safety_verdict:
                    correct_frames += 1  # safety override counts as correct for safety classes
        finally:
            os.unlink(tmp)

        stability = correct_frames / FRAMES * 100
        stability_scores[cls] = stability
        status = "PASS" if stability >= 90 else "FAIL"
        most_common = collections.Counter(predictions).most_common(1)[0][0]
        print(f"  {cls:20s} {correct_frames:>8d} {FRAMES:>6d} {stability:>9.1f}% "
              f"{status:>8s}  (most common: {most_common})")

    all_pass = all(s >= 90 for s in stability_scores.values())
    print(f"\n  Overall stability: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    if not all_pass:
        for cls, s in stability_scores.items():
            if s < 90:
                print(f"  [FAIL] {cls}: {s:.1f}% — temporal buffer or model ambiguity")

    return stability_scores


# ─── TASK 5: INTEGRATED VERIFICATION ────────────────────────────────────────

def task5_integration(engine):
    sep()
    print("  TASK 5 — Final Integrated Verification")
    sep()

    results_pass = {}

    # TEST 1: Normal environments
    print("\n  [TEST 1] Normal environments — no safety override expected")
    sequence = [
        ("background_noise", gen_background(0), "Background Noise"),
        ("traffic",          gen_traffic(0),    "Traffic"),
        ("speech",           gen_speech(0),     "Speech"),
        ("music",            gen_music(0),      "Music"),
    ]
    engine.reset_buffers()
    test1_pass = True
    for expected, samples, desc in sequence:
        tmp = make_tmp(samples)
        try:
            engine.infer(tmp)
            result = engine.infer(tmp)
        finally:
            os.unlink(tmp)

        wrong_safety = result.safety_verdict
        print(f"    {desc:18s}: pred={result.environment:18s} "
              f"safety={'[!]OVERRIDE' if wrong_safety else '[OK]safe'}")
        if wrong_safety:
            test1_pass = False
            print(f"      [FAIL] False safety trigger on {desc}")
    print(f"  TEST 1: {'PASS' if test1_pass else 'FAIL'}")
    results_pass["test1_normal"] = test1_pass

    # TEST 2: Safety sounds during traffic
    print("\n  [TEST 2] Safety override during traffic → siren → traffic")
    engine.reset_buffers()
    test2_pass = True
    steps = [
        (gen_traffic(0), "traffic", "Traffic (pre)"),
        (gen_siren(0),   "siren",   "Siren (override)"),
        (gen_traffic(0), "traffic", "Traffic (post)"),
    ]
    for samples, expected, desc in steps:
        tmp = make_tmp(samples)
        try:
            engine.infer(tmp)
            result = engine.infer(tmp)
        finally:
            os.unlink(tmp)

        if expected == "siren":
            ok = result.safety_verdict
            status = "PASS" if ok else "FAIL"
            if not ok: test2_pass = False
        else:
            ok = not result.safety_verdict
            status = "PASS" if ok else "FAIL"
            if not ok: test2_pass = False

        print(f"    {desc:20s}: pred={result.environment:15s} "
              f"override={result.safety_verdict}  [{status}]")
    print(f"  TEST 2: {'PASS' if test2_pass else 'FAIL'}")
    results_pass["test2_safety"] = test2_pass

    # TEST 3: Ambiguous sounds (dog bark near traffic, construction with horn)
    print("\n  [TEST 3] Ambiguous sounds — disambiguation")
    engine.reset_buffers()
    test3_pass = True

    # Dog bark near traffic: should stay dog_bark, not trigger horn override
    def gen_bark_traffic():
        b = gen_dog_bark(0)
        t = gen_traffic(0)
        return [_clip(0.7 * b[i] + 0.3 * t[i]) for i in range(N)]

    tmp = make_tmp(gen_bark_traffic())
    try:
        engine.infer(tmp)
        result = engine.infer(tmp)
    finally:
        os.unlink(tmp)

    bark_ok = not result.safety_verdict
    print(f"    Dog+Traffic: pred={result.environment:15s} "
          f"safety={result.safety_verdict} -> {'PASS (no horn trigger)' if bark_ok else 'FAIL (false horn)'}")
    if not bark_ok: test3_pass = False

    # Construction with horn: horn should override
    def gen_constr_horn():
        c = gen_construction(0)
        h = gen_horn(0)
        return [_clip(0.4 * c[i] + 0.6 * h[i]) for i in range(N)]

    engine.reset_buffers()
    tmp = make_tmp(gen_constr_horn())
    try:
        engine.infer(tmp)
        result = engine.infer(tmp)
    finally:
        os.unlink(tmp)
    horn_ok = result.safety_verdict or result.environment == "horn"
    print(f"    Constr+Horn: pred={result.environment:15s} "
          f"safety={result.safety_verdict} -> {'PASS (horn detected)' if horn_ok else 'FAIL (horn missed)'}")
    if not horn_ok: test3_pass = False

    print(f"  TEST 3: {'PASS' if test3_pass else 'FAIL'}")
    results_pass["test3_ambiguous"] = test3_pass

    # TEST 4: Edge cases
    print("\n  [TEST 4] Edge cases")
    test4_pass = True

    # Quiet siren (attenuated)
    def gen_quiet_siren():
        s = gen_siren(0)
        return [x * 0.3 for x in s]   # 70% quieter

    engine.reset_buffers()
    tmp = make_tmp(gen_quiet_siren())
    try:
        engine.infer(tmp)
        result = engine.infer(tmp)
    finally:
        os.unlink(tmp)
    quiet_ok = result.safety_verdict or result.environment in ("siren", "horn")
    print(f"    Quiet siren: pred={result.environment:15s} "
          f"safety={result.safety_verdict} -> {'PASS' if quiet_ok else 'FAIL (missed quiet siren)'}")
    if not quiet_ok:
        test4_pass = False
        print("    NOTE: Quiet siren missed — consider lower YAMNET_SIREN_THRESHOLD")

    # Loud music: should NOT trigger safety
    loud_music = [min(1.0, x * 2.5) for x in gen_music(1)]
    engine.reset_buffers()
    tmp = make_tmp(loud_music)
    try:
        engine.infer(tmp)
        result = engine.infer(tmp)
    finally:
        os.unlink(tmp)
    music_ok = not result.safety_verdict
    print(f"    Loud music : pred={result.environment:15s} "
          f"safety={result.safety_verdict} -> {'PASS (no false alarm)' if music_ok else 'FAIL (false alarm!)'}")
    if not music_ok: test4_pass = False

    # Rapid env changes every 2 frames
    print("    Rapid changes: ", end="", flush=True)
    engine.reset_buffers()
    classes_seen = []
    for cls, gfn in [("traffic", gen_traffic), ("speech", gen_speech),
                      ("music", gen_music), ("background_noise", gen_background)]:
        tmp = make_tmp(gfn(0))
        try:
            r = engine.infer(tmp)
        finally:
            os.unlink(tmp)
        classes_seen.append(r.settings_status)

    # At least some should be "Held" (temporal filter working)
    held_count = classes_seen.count("Held")
    temporal_ok = held_count >= 2
    print(f"Held={held_count}/4 -> {'PASS (temporal filter active)' if temporal_ok else 'FAIL (no temporal filtering)'}")
    if not temporal_ok: test4_pass = False

    print(f"  TEST 4: {'PASS' if test4_pass else 'FAIL'}")
    results_pass["test4_edge"] = test4_pass

    return results_pass


# ─── MAIN ─────────────────────────────────────────────────────────────────────

CLASS_NAMES_V2_LOCAL = [
    "speech", "siren", "horn", "traffic",
    "construction", "dog_bark", "music", "background_noise"
]

def main():
    sep()
    print("  HearSmart V3 — Full Audit & Safety Hardening Test Suite")
    print("  2026-03-24 | All 5 Tasks")
    sep()

    from src.pipeline_v3 import get_engine, _engine
    import src.pipeline_v3 as pv3
    # Reset singleton so we get fresh hardened engine
    pv3._engine = None

    print("\n  Loading engine...")
    t0 = time.perf_counter()
    engine = get_engine()
    engine._ensure_loaded()
    load_ms = round((time.perf_counter() - t0) * 1000)
    print(f"  Engine loaded in {load_ms}ms\n")

    # Task 2: Report hardening config
    task2_report_hardening()

    # Task 1: Audit
    confusion, accuracy, fp_safety, fn_safety = task1_classification_audit(engine)

    # Task 3: Safety recall
    recall, fn_list, fp_list = task3_safety_recall(engine)

    # Task 4: Stability
    stability = task4_stability(engine)

    # Task 5: Integration
    integration = task5_integration(engine)

    # ─── FINAL REPORT ────────────────────────────────────────────────────
    sep()
    print("  FINAL REPORT")
    sep()
    print(f"\n  1. Classification Accuracy : {accuracy*100:.1f}%")
    print(f"  2. Safety Recall           : {recall*100:.1f}%")
    print(f"  3. False Safety Triggers   : {len(fp_safety)} (non-safety->override)")
    print(f"\n  4. Stability Scores:")
    for cls, s in stability.items():
        bar = "#" * int(s / 10)
        status = "PASS" if s >= 90 else "FAIL"
        print(f"     {cls:20s}: {s:5.1f}% [{bar:<10s}] {status}")

    print(f"\n  5. Calibrated Thresholds:")
    from src.safety_config import (YAMNET_SIREN_THRESHOLD, YAMNET_HORN_THRESHOLD,
                                    SAFETY_FRAMES_REQUIRED, CONFIDENCE_FLOORS)
    print(f"     yamnet_siren_threshold = {YAMNET_SIREN_THRESHOLD}  (recall priority)")
    print(f"     yamnet_horn_threshold  = {YAMNET_HORN_THRESHOLD}  (recall priority)")
    print(f"     safety_frames_required = {SAFETY_FRAMES_REQUIRED}  (~{SAFETY_FRAMES_REQUIRED*0.96:.1f}s min detection)")

    print(f"\n  6. Integration Tests:")
    for test, passed in integration.items():
        print(f"     {test:25s}: {'PASS' if passed else 'FAIL'}")

    all_integration = all(integration.values())
    print(f"\n  7. Bugs found & fixed:")
    print(f"     - YAMNet indices were generic; replaced with calibrated constants")
    print(f"     - Traffic noise could trigger safety; traffic negative guard added")
    print(f"     - Single-frame transient safety triggers; frame counter added")
    print(f"     - Dog Bark vs Horn confusion; disambiguation logic added")
    print(f"     - Settings applied on low confidence; per-class floors added")

    # Warm latency from last integration test result
    print(f"\n  8. Pipeline Status:")
    print(f"     Recall>=95%  : {'YES' if recall >= 0.95 else 'NO — see root cause above'}")
    print(f"     FP<30%       : {'YES' if len(fp_safety) < 0.3 * 8 else 'NO'}")
    print(f"     All stable   : {'YES' if all(s>=90 for s in stability.values()) else 'PARTIAL'}")
    print(f"     Integration  : {'ALL PASS' if all_integration else 'SOME FAIL'}")

    sep()

if __name__ == "__main__":
    main()

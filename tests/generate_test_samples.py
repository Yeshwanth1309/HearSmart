"""
Download diverse test audio samples for V3 pipeline testing.
Uses free audio sources to get samples of different sound classes.
"""
import os, sys, io, urllib.request, struct, wave, math, random
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SAMPLE_DIR = Path("demo/samples")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
DURATION = 3.0
N_SAMPLES = int(SR * DURATION)


def write_wav(filepath, samples, sr=16000):
    """Write mono 16-bit WAV file."""
    with wave.open(str(filepath), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        data = b''
        for s in samples:
            s = max(-1.0, min(1.0, s))
            data += struct.pack('<h', int(s * 32767))
        wf.writeframes(data)


def generate_siren(duration=3.0, sr=16000):
    """Generate realistic emergency siren sound (wailing)."""
    samples = []
    for i in range(int(sr * duration)):
        t = i / sr
        # Wailing siren: frequency sweeps between 600-1600 Hz
        sweep = 600 + 500 * math.sin(2 * math.pi * 0.5 * t)  # 0.5 Hz modulation
        val = 0.7 * math.sin(2 * math.pi * sweep * t)
        # Add harmonics
        val += 0.3 * math.sin(2 * math.pi * sweep * 2 * t)
        val += 0.1 * math.sin(2 * math.pi * sweep * 3 * t)
        samples.append(val * 0.8)
    return samples


def generate_car_horn(duration=3.0, sr=16000):
    """Generate car horn sound (dual-tone)."""
    samples = []
    for i in range(int(sr * duration)):
        t = i / sr
        # Car horn: two fundamental frequencies
        f1, f2 = 420, 520
        val = 0.5 * math.sin(2 * math.pi * f1 * t)
        val += 0.5 * math.sin(2 * math.pi * f2 * t)
        # Slight amplitude modulation
        val *= 0.7 + 0.3 * math.sin(2 * math.pi * 5 * t)
        # On-off pattern (honking)
        if (t % 0.8) > 0.5:
            val *= 0.05
        samples.append(val * 0.8)
    return samples


def generate_gunshot(duration=3.0, sr=16000):
    """Generate gunshot-like impulse sound."""
    random.seed(42)
    samples = []
    shot_times = [0.3, 1.2, 2.1]  # Three shots
    for i in range(int(sr * duration)):
        t = i / sr
        val = 0.0
        for st in shot_times:
            dt = t - st
            if 0 <= dt < 0.15:
                # Sharp attack, fast decay
                envelope = math.exp(-dt * 40) * (1 - math.exp(-dt * 2000))
                # Broadband noise burst
                noise = random.uniform(-1, 1)
                val += envelope * noise * 0.9
                # Low frequency thump
                val += envelope * math.sin(2 * math.pi * 80 * dt) * 0.5
        samples.append(max(-1, min(1, val)))
    return samples


def generate_traffic(duration=3.0, sr=16000):
    """Generate traffic/engine idling sound."""
    random.seed(123)
    samples = []
    for i in range(int(sr * duration)):
        t = i / sr
        # Low-frequency engine rumble
        val = 0.3 * math.sin(2 * math.pi * 60 * t)
        val += 0.2 * math.sin(2 * math.pi * 120 * t)
        val += 0.1 * math.sin(2 * math.pi * 180 * t)
        # Road noise (filtered random)
        noise = random.uniform(-1, 1) * 0.15
        val += noise
        # Slight variation
        val *= 0.8 + 0.2 * math.sin(2 * math.pi * 0.3 * t)
        samples.append(val * 0.6)
    return samples


def generate_construction(duration=3.0, sr=16000):
    """Generate drilling/jackhammer sound."""
    random.seed(456)
    samples = []
    drill_freq = 25  # impacts per second
    for i in range(int(sr * duration)):
        t = i / sr
        # Repetitive impact pattern
        phase = (t * drill_freq) % 1.0
        if phase < 0.3:
            impact = math.exp(-phase * 20)
            val = impact * (random.uniform(-1, 1) * 0.8)
            val += impact * math.sin(2 * math.pi * 800 * t) * 0.4
        else:
            val = random.uniform(-1, 1) * 0.05
        samples.append(val * 0.7)
    return samples


def generate_music(duration=3.0, sr=16000):
    """Generate simple musical tones (melodic pattern)."""
    samples = []
    # Simple melody: C-E-G-C ascending
    notes = [261.6, 329.6, 392.0, 523.3, 392.0, 329.6]
    note_dur = duration / len(notes)
    for i in range(int(sr * duration)):
        t = i / sr
        note_idx = min(int(t / note_dur), len(notes) - 1)
        freq = notes[note_idx]
        local_t = t - note_idx * note_dur
        # Envelope
        env = min(1.0, local_t * 20) * math.exp(-local_t * 2)
        val = env * 0.5 * math.sin(2 * math.pi * freq * t)
        val += env * 0.2 * math.sin(2 * math.pi * freq * 2 * t)
        val += env * 0.1 * math.sin(2 * math.pi * freq * 3 * t)
        samples.append(val * 0.8)
    return samples


def generate_speech(duration=3.0, sr=16000):
    """Generate speech-like formant patterns."""
    random.seed(789)
    samples = []
    # Vowel formants (approximate)
    formants = [(700, 1200), (300, 2500), (500, 1500)]  # a, i, o
    vowel_dur = duration / len(formants)
    for i in range(int(sr * duration)):
        t = i / sr
        vowel_idx = min(int(t / vowel_dur), len(formants) - 1)
        f1, f2 = formants[vowel_idx]
        local_t = t - vowel_idx * vowel_dur
        # Glottal pulse (fundamental ~120 Hz male voice)
        glottal = 0.5 * math.sin(2 * math.pi * 120 * t)
        glottal += 0.3 * math.sin(2 * math.pi * 240 * t)
        # Formant resonances
        val = glottal * (0.4 * math.sin(2 * math.pi * f1 * t) +
                         0.3 * math.sin(2 * math.pi * f2 * t))
        # Slight noise (aspiration)
        val += random.uniform(-1, 1) * 0.02
        # Envelope for natural phrasing
        env = min(1.0, local_t * 10) * min(1.0, (vowel_dur - local_t) * 10)
        samples.append(val * env * 0.7)
    return samples


def generate_background(duration=3.0, sr=16000):
    """Generate ambient background noise (wind/AC hum)."""
    random.seed(321)
    samples = []
    for i in range(int(sr * duration)):
        t = i / sr
        # Low frequency hum (AC)
        val = 0.1 * math.sin(2 * math.pi * 50 * t)
        val += 0.05 * math.sin(2 * math.pi * 100 * t)
        # Broadband ambient noise
        val += random.uniform(-1, 1) * 0.08
        # Slow amplitude variation
        val *= 0.8 + 0.2 * math.sin(2 * math.pi * 0.1 * t)
        samples.append(val)
    return samples


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating diverse test audio samples")
    print("=" * 60)

    generators = {
        "siren_test": ("Siren (wailing emergency)", generate_siren),
        "horn_test": ("Car Horn (honking)", generate_car_horn),
        "gunshot_test": ("Gunshot (impulse bursts)", generate_gunshot),
        "traffic_test": ("Traffic (engine idle)", generate_traffic),
        "construction_test": ("Construction (drilling)", generate_construction),
        "music_test": ("Music (melodic tones)", generate_music),
        "speech_test": ("Speech (vocal formants)", generate_speech),
        "background_test": ("Background (ambient)", generate_background),
    }

    for name, (desc, gen_fn) in generators.items():
        filepath = SAMPLE_DIR / f"{name}.wav"
        samples = gen_fn(DURATION, SR)
        write_wav(filepath, samples, SR)
        print(f"  [OK] {desc:35s} -> {filepath}")

    print(f"\n  Generated {len(generators)} test files in {SAMPLE_DIR}")
    print("=" * 60)

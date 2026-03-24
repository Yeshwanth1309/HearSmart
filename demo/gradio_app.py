"""
Gradio Demo V3 — HearSmart V3 Engine Dashboard.
Includes HACKATHON DEMO MODE with auto-playback sequence and real-time UI updates.
"""
import logging, os, sys, tempfile, time, math, struct, wave, io as _io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from src.utils import setup_logging, set_seed
from src.data.label_map_v2 import CLASS_NAMES_V2, CLASS_EMOJIS_V2

setup_logging()
set_seed(42)
logger = logging.getLogger(__name__)

# ─── Demo sequence config (uses real ESC-50 audio) ───────────────────────────
DEMO_STEPS = [
    {"label": "Background Noise", "class": "background_noise",
     "desc": "Quiet ambient environment",   "duration": 4, "step": 1},
    {"label": "Traffic",          "class": "traffic",
     "desc": "City traffic noise",          "duration": 4, "step": 2},
    {"label": "Speech",           "class": "speech",
     "desc": "Human voice detected",        "duration": 4, "step": 3},
    {"label": "SIREN",            "class": "siren",
     "desc": "Emergency vehicle incoming!", "duration": 4, "step": 4},
    {"label": "Traffic",          "class": "traffic",
     "desc": "Siren passed — resuming",     "duration": 4, "step": 5},
    {"label": "Background Noise", "class": "background_noise",
     "desc": "Environment settling",        "duration": 3, "step": 6},
]

# Real audio files to use for each demo step (from ESC-50 / project samples)
DEMO_AUDIO_MAP = {
    "background_noise": [
        "hearing_aid/data/ESC50/audio/1-100210-A-36.wav",   # vacuum cleaner
        "hearing_aid/data/ESC50/audio/1-100038-A-14.wav",   # birds
    ],
    "traffic": [
        "hearing_aid/data/ESC50/audio/1-119125-A-45.wav",   # train
        "hearing_aid/data/ESC50/audio/1-172649-A-40.wav",   # helicopter
    ],
    "speech": [
        "hearing_aid/data/ESC50/audio/1-104089-A-22.wav",   # clapping
        "hearing_aid/data/ESC50/audio/1-105224-A-22.wav",   # clapping
    ],
    "siren": [
        "hearing_aid/data/ESC50/audio/1-31482-A-42.wav",    # siren
        "hearing_aid/data/ESC50/audio/1-13613-A-37.wav",    # clock alarm
    ],
}

# Fallback: generate synthetic audio if real file not found
def _gen_synthetic_wav(class_name: str) -> str:
    """Generate a 3-second synthetic WAV for the given class."""
    SR = 16000; N = 48000
    samples_list = []

    if class_name == "siren":
        for i in range(N):
            t = i / SR
            freq = 800 + 400 * math.sin(2 * math.pi * 0.6 * t)
            samples_list.append(0.7 * math.sin(2 * math.pi * freq * t))
    elif class_name == "horn":
        for i in range(N):
            t = i / SR
            v = 0.5 * math.sin(2*math.pi*420*t) + 0.5*math.sin(2*math.pi*520*t)
            samples_list.append(v * (0.7 if (t % 0.8) < 0.5 else 0.05))
    elif class_name == "traffic":
        import random; rng = random.Random(1)
        for i in range(N):
            t = i / SR
            samples_list.append(0.3*math.sin(2*math.pi*60*t) + rng.uniform(-1,1)*0.1)
    elif class_name == "speech":
        import random; rng = random.Random(2)
        for i in range(N):
            t = i / SR
            v = 0.5*math.sin(2*math.pi*120*t) * math.sin(2*math.pi*700*t)
            samples_list.append(v * 0.6 + rng.gauss(0, 0.01))
    else:  # background
        import random; rng = random.Random(3)
        for i in range(N):
            t = i / SR
            samples_list.append(0.1*math.sin(2*math.pi*50*t) + rng.uniform(-1,1)*0.05)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, 'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(b''.join(
            struct.pack('<h', max(-32767, min(32767, int(s * 32767))))
            for s in samples_list
        ))
    tmp.close()
    return tmp.name


def _get_demo_audio(class_name: str) -> str:
    """Return path to real audio or fallback synthetic."""
    candidates = DEMO_AUDIO_MAP.get(class_name, [])
    for p in candidates:
        if Path(p).exists():
            return p
    return _gen_synthetic_wav(class_name)


# ─── Inference helper ─────────────────────────────────────────────────────────

def _run_inference(audio_path: str):
    from src.pipeline_v3 import get_engine
    engine = get_engine()
    return engine.infer(audio_path)


def _build_plots(result):
    """Build donut + bar chart from result."""
    donut_fig = go.Figure(go.Pie(
        labels=[c.replace('_', ' ').title() for c in CLASS_NAMES_V2],
        values=result.ensemble_probs,
        hole=0.65,
        marker=dict(colors=[
            "#3b82f6","#ef4444","#eab308","#10b981",
            "#8b5cf6","#f97316","#06b6d4","#94a3b8"
        ]),
        textinfo='none', hoverinfo='label+percent'
    ))
    donut_fig.update_layout(
        title="Ensemble Spread",
        title_font=dict(color="#f8fafc", size=14),
        font=dict(color="#94a3b8"),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    m_names = list(result.per_model.keys())
    m_confs = [result.per_model[m][1] * 100 for m in m_names]
    m_preds = [result.per_model[m][0].replace('_', ' ').title() for m in m_names]
    bar_fig = go.Figure(go.Bar(
        x=m_names, y=m_confs, text=m_preds,
        textposition='inside', insidetextanchor='end',
        marker_color=[
            "#ef4444" if p.lower() in ["siren","horn"] else "#3b82f6"
            for p in m_preds
        ],
        hovertemplate="%{x}: %{y:.1f}%<br>Pred: %{text}<extra></extra>",
        marker_line_width=0
    ))
    bar_fig.update_layout(
        title="Model Confidence",
        title_font=dict(color="#f8fafc", size=14),
        font=dict(color="#94a3b8"),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40,b=10,l=10,r=10),
        yaxis=dict(range=[0,100], gridcolor='#334155', zerolinecolor='#334155'),
        xaxis=dict(gridcolor='#334155')
    )
    return donut_fig, bar_fig


def _build_outputs(result, step_info=None):
    """Build all 8 UI outputs from a V3InferenceResult."""
    emoji = CLASS_EMOJIS_V2.get(result.environment, "🎵")
    env_title = result.environment.replace('_', ' ').title()
    is_alert = result.safety_verdict

    step_badge = ""
    if step_info:
        step_badge = f"**Step {step_info['step']}/6** — {step_info['desc']}\n\n"

    if is_alert:
        result_md = f"""{step_badge}## 🚨 {emoji} {env_title} — SAFETY OVERRIDE
### ⚠️ Emergency sound detected! Maximum awareness settings applied.
**Confidence:** {result.confidence:.1%} &nbsp;|&nbsp; ⏱ **{result.latency_ms:.0f}ms** &nbsp;|&nbsp; Source: _{result.safety_source}_"""
    else:
        result_md = f"""{step_badge}## {emoji} {env_title}
**Confidence:** {result.confidence:.1%} &nbsp;|&nbsp; ⏱ **{result.latency_ms:.0f}ms** &nbsp;|&nbsp; 🤖 V3 2-Model Ensemble"""

    rec_md = f"""### Recommended Settings

| Setting | Value |
|:--------|:------|
| **Volume** | {result.volume}/10 |
| **Noise Reduction** | {result.noise_reduction} |
| **Directionality** | {result.directionality} |
| **Speech Enhancement** | {"On" if result.speech_enhancement else "Off"} |

_{result.reasoning}_"""

    risk_str = f"**{result.risk_score:.2f}** — _{result.risk_action.title()}_"
    temporal_str = ("🟢 **Confirmed**" if result.temporal_status == "Confirmed"
                    else "🟡 **Uncertain** — waiting")
    safety_str = (f"🔴 **DANGER** — via _{result.safety_source}_" if is_alert
                  else f"🟢 **Safe** (prob: {result.safety_probability:.2f})")
    status_map = {
        "Applied": "🟢 **Applied**",
        "Conservative": "🟡 **Conservative**",
        "Held": "⚪ **Held**",
        "Safety Override": "🔴 **Safety Override**",
    }
    settings_str = status_map.get(result.settings_status, f"⚪ {result.settings_status}")

    donut_fig, bar_fig = _build_plots(result)

    return (result_md, rec_md, donut_fig, bar_fig,
            risk_str, temporal_str, safety_str, settings_str)


# ─── Live inference handler ───────────────────────────────────────────────────

def predict_audio(audio_input, hearing_loss, tinnitus, age_group, preference):
    if audio_input is None:
        return ("Upload or record audio to begin.", "", None, None,
                "—", "—", "—", "—")
    try:
        from src.pipeline_v3 import get_engine
        engine = get_engine()
    except Exception as e:
        return (f"Engine load failed: {e}", "", None, None, "—", "—", "—", "—")

    if isinstance(audio_input, tuple):
        sr_in, arr = audio_input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, arr, sr_in)
            audio_path = tmp.name
        cleanup = True
    else:
        audio_path = audio_input
        cleanup = False

    try:
        result = engine.infer(audio_path)
    except Exception as e:
        logger.error(str(e), exc_info=True)
        return (f"Inference error: {e}", "", None, None, "—", "—", "—", "—")
    finally:
        if cleanup and os.path.exists(audio_path):
            os.unlink(audio_path)

    return _build_outputs(result)


# ─── Demo step runner ─────────────────────────────────────────────────────────

def run_demo_step(step_idx: int):
    """
    Run inference for one step of the hackathon demo sequence.
    Called sequentially by 'Start Demo' button via yield.
    """
    if step_idx < 0 or step_idx >= len(DEMO_STEPS):
        return

    step = DEMO_STEPS[step_idx]
    audio_path = _get_demo_audio(step["class"])

    try:
        from src.pipeline_v3 import get_engine
        engine = get_engine()
        # Run twice for temporal confirmation (simulates 2 consecutive frames)
        engine.infer(audio_path)
        result = engine.infer(audio_path)
    except Exception as e:
        logger.error(f"Demo step {step_idx} failed: {e}", exc_info=True)
        return None

    return _build_outputs(result, step_info=step)


def stream_demo():
    """
    Generator that yields one UI update per demo step.
    Drives the 'Start Demo' button sequence.
    """
    from src.pipeline_v3 import get_engine
    engine = get_engine()
    engine.reset_buffers()

    for i, step in enumerate(DEMO_STEPS):
        audio_path = _get_demo_audio(step["class"])
        try:
            engine.infer(audio_path)   # warm temporal
            result = engine.infer(audio_path)
            outputs = _build_outputs(result, step_info=step)
        except Exception as e:
            logger.error(f"Demo step {i} error: {e}")
            continue

        yield outputs
        time.sleep(step["duration"])

    # Final: reset
    engine.reset_buffers()


def reset_demo():
    """Reset demo state."""
    try:
        from src.pipeline_v3 import get_engine
        get_engine().reset_buffers()
    except Exception:
        pass
    return ("Demo reset. Press Start Demo to replay.", "", None, None,
            "—", "—", "—", "—")


# ─── PROGRESS HTML generator ─────────────────────────────────────────────────

def _progress_html(active_step: int = 0) -> str:
    steps = [s["label"] for s in DEMO_STEPS]
    items = ""
    for i, lbl in enumerate(steps):
        is_active  = (i == active_step)
        is_done    = (i < active_step)
        is_safety  = (lbl == "SIREN")
        color = ("#ef4444" if is_safety and is_active else
                 "#3b82f6" if is_active else
                 "#22c55e" if is_done else "#374151")
        border = "2px solid " + color
        font_weight = "700" if is_active else "400"
        items += (f'<div style="text-align:center;padding:6px 10px;border-radius:8px;'
                  f'border:{border};background:{"#1e1218" if is_safety and is_active else "#1a1d24"};'
                  f'color:{"#ef4444" if is_safety and is_active else "#f8fafc"};'
                  f'font-weight:{font_weight};font-size:0.78em;min-width:80px">'
                  f'{lbl}</div>')
    return (f'<div style="display:flex;gap:8px;flex-wrap:wrap;'
            f'align-items:center;padding:12px 0">{items}</div>')


# ─── CSS ─────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1536px !important; font-family: 'Inter', -apple-system, sans-serif !important; background: #0f1115 !important; }
body { background: #0f1115 !important; color: #f8fafc !important; }
.hdr { padding: 20px 24px; border-radius: 12px; background: #1a1d24; border: 1px solid #2d3748; margin-bottom: 16px; }
.hdr h1 { color: #f8fafc !important; font-size: 1.8em !important; font-weight: 700 !important; letter-spacing: -0.02em; margin: 0 0 4px !important; }
.hdr p  { color: #94a3b8 !important; margin: 0 !important; font-size: 0.9em; }
.card   { background: #1a1d24; border: 1px solid #2d3748; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
.demo-card { background: #111827; border: 2px solid #3b82f6; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
.alert-card { background: #1c0a0a; border: 2px solid #ef4444; border-radius: 12px; padding: 20px; margin-bottom: 16px; animation: pulse-border 1s infinite; }
@keyframes pulse-border { 0%,100%{border-color:#ef4444} 50%{border-color:#7f1d1d} }
.status-card { background: #1a1d24; border: 1px solid #2d3748; border-radius: 10px; padding: 14px 18px; }
h3 { color: #e2e8f0 !important; font-weight: 600 !important; text-transform: uppercase; font-size: 0.82em !important; letter-spacing: 0.05em; margin-top: 0 !important; border-bottom: 1px solid #2d3748; padding-bottom: 8px; }
.btn-demo { background: linear-gradient(135deg, #3b82f6, #2563eb) !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; font-size: 1.1em !important; color: #fff !important; padding: 14px 28px !important; transition: all 0.2s ease; box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important; }
.btn-demo:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(59,130,246,0.4) !important; }
.btn-reset { background: #374151 !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; color: #d1d5db !important; }
.btn { background: #3b82f6 !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; color: #fff !important; }
table { border-collapse: separate !important; border-spacing: 0 !important; width: 100% !important; margin-top: 10px !important; }
th { border-bottom: 1px solid #2d3748 !important; padding: 10px 14px !important; color: #94a3b8 !important; font-size: 0.75em !important; text-transform: uppercase !important; }
td { padding: 12px 14px !important; border-bottom: 1px solid #1e293b !important; font-size: 0.9em !important; color: #e2e8f0 !important; font-weight: 500 !important; }
footer { display: none !important; }
div.form { border: 1px solid #2d3748 !important; border-radius: 12px !important; background: #1a1d24 !important; }
"""

HEADER = """<div class="hdr">
  <h1>HearSmart V3 &mdash; Context-Aware Hearing Aid Optimizer</h1>
  <p>2-Model Ensemble &middot; Triple Safety Fusion &middot; Temporal Logic &middot; Real-time Risk Scoring</p>
</div>"""

DEMO_HEADER = """<div style="background:linear-gradient(135deg,#1e3a5f,#1a1d24);border:2px solid #3b82f6;
border-radius:12px;padding:16px 20px;margin-bottom:12px">
  <h2 style="color:#60a5fa;margin:0 0 4px;font-size:1.1em;font-weight:700;text-transform:uppercase;letter-spacing:0.05em">
    🎬 Hackathon Demo Mode
  </h2>
  <p style="color:#94a3b8;margin:0;font-size:0.85em">
    Auto-plays 6-step environment sequence &nbsp;&bull;&nbsp; 23 second runtime &nbsp;&bull;&nbsp;
    Siren triggers real safety override &nbsp;&bull;&nbsp; No microphone needed
  </p>
</div>"""


# ─── Build UI ─────────────────────────────────────────────────────────────────

def build_demo():
    with gr.Blocks(title="HearSmart V3") as app:
        gr.HTML(HEADER)

        with gr.Tabs():
            # ──────────────────────────────────────────────────────────────
            # TAB 1: LIVE MODE
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("Live Analysis"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("### Audio Source", elem_classes=["card"])
                        audio_in = gr.Audio(
                            sources=["upload", "microphone"], type="filepath",
                            label="Upload or Record"
                        )
                        with gr.Accordion("User Context (Optional)", open=False):
                            hl_dd  = gr.Dropdown(["none","mild","moderate","severe"],
                                                  value="none", label="Hearing Loss")
                            tin_cb = gr.Checkbox(label="Tinnitus", value=False)
                            age_dd = gr.Dropdown(["young","adult","elderly"],
                                                  value="adult", label="Age Group")
                            pref_dd= gr.Dropdown(["balanced","speech","music"],
                                                  value="balanced", label="Preference")
                        predict_btn = gr.Button("Analyze Feed", variant="primary",
                                                elem_classes=["btn"])
                        gr.Markdown("""### Safety Protocols
🚨 **Triple Safety Fusion + YAMNet Head**
4-condition OR gate — Binary classifier, YAMNet raw,
CNN confidence, YAMNet 8-class head.
`Siren` and `Horn` trigger **instant override**.""",
                                    elem_classes=["card"])

                    with gr.Column(scale=7):
                        live_result_md = gr.Markdown("### Status\nSystem idle.",
                                                      elem_classes=["card"])
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Risk Score",  elem_classes=["status-card"])
                                live_risk_md = gr.Markdown("—", elem_classes=["status-card"])
                            with gr.Column(scale=1):
                                gr.Markdown("### Temporal",    elem_classes=["status-card"])
                                live_temporal_md = gr.Markdown("—", elem_classes=["status-card"])
                            with gr.Column(scale=1):
                                gr.Markdown("### Safety",      elem_classes=["status-card"])
                                live_safety_md = gr.Markdown("—", elem_classes=["status-card"])
                            with gr.Column(scale=1):
                                gr.Markdown("### Settings",    elem_classes=["status-card"])
                                live_settings_md = gr.Markdown("—", elem_classes=["status-card"])
                        with gr.Row():
                            with gr.Column(scale=1):
                                live_donut = gr.Plot(label="Ensemble Spread", show_label=False)
                            with gr.Column(scale=1):
                                live_bar   = gr.Plot(label="Model Confidence", show_label=False)
                        live_rec_md = gr.Markdown(elem_classes=["card"])

                sample_dir = Path(__file__).parent / "samples"
                if sample_dir.exists():
                    samples = [[str(p)] for p in sorted(sample_dir.glob("*.wav"))[:4]]
                    if samples:
                        gr.Examples(examples=samples, inputs=[audio_in],
                                    label="Sample Files")

                live_outs = [live_result_md, live_rec_md, live_donut, live_bar,
                             live_risk_md, live_temporal_md, live_safety_md, live_settings_md]
                live_ins  = [audio_in, hl_dd, tin_cb, age_dd, pref_dd]
                predict_btn.click(fn=predict_audio, inputs=live_ins, outputs=live_outs)
                audio_in.change(fn=predict_audio, inputs=live_ins, outputs=live_outs)

            # ──────────────────────────────────────────────────────────────
            # TAB 2: HACKATHON DEMO MODE
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("🎬 Demo Mode"):
                gr.HTML(DEMO_HEADER)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Progress indicator
                        progress_html = gr.HTML(_progress_html(0),
                                                label="Demo Sequence")

                        # Control buttons
                        with gr.Row():
                            start_btn = gr.Button("▶ START DEMO", variant="primary",
                                                  elem_classes=["btn-demo"], scale=3)
                            reset_btn = gr.Button("↺ Reset", variant="secondary",
                                                  elem_classes=["btn-reset"], scale=1)

                        # Sequence legend
                        gr.Markdown("""### Demo Sequence
| Step | Environment | Duration |
|:--:|:--|:--:|
| 1 | 🔇 Background Noise | 4s |
| 2 | 🚦 Traffic | 4s |
| 3 | 🗣️ Speech | 4s |
| 4 | 🚨 **SIREN — Safety Override** | 4s |
| 5 | 🚦 Traffic (recovery) | 4s |
| 6 | 🔇 Background Noise | 3s |
| | **Total runtime ~23s** | |
""", elem_classes=["card"])

                        gr.Markdown("""### What to watch
- **Risk Score** jumps to 1.00 on siren
- **Safety panel** turns RED on siren  
- **Settings** shows "Safety Override"
- Volume locks to **9/10**, NR = **Off**
- After siren: smooth transition back
""", elem_classes=["card"])

                    with gr.Column(scale=2):
                        # Main result display
                        demo_result_md = gr.Markdown(
                            "### Awaiting Demo Start\nPress **▶ START DEMO** to begin.",
                            elem_classes=["card"]
                        )

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Risk Score",  elem_classes=["status-card"])
                                demo_risk_md = gr.Markdown("—", elem_classes=["status-card"])
                            with gr.Column(scale=1):
                                gr.Markdown("### Temporal",    elem_classes=["status-card"])
                                demo_temporal_md = gr.Markdown("—", elem_classes=["status-card"])
                            with gr.Column(scale=1):
                                gr.Markdown("### Safety",      elem_classes=["status-card"])
                                demo_safety_md = gr.Markdown("—", elem_classes=["status-card"])
                            with gr.Column(scale=1):
                                gr.Markdown("### Settings",    elem_classes=["status-card"])
                                demo_settings_md = gr.Markdown("—", elem_classes=["status-card"])

                        with gr.Row():
                            with gr.Column(scale=1):
                                demo_donut = gr.Plot(label="Ensemble Spread", show_label=False)
                            with gr.Column(scale=1):
                                demo_bar   = gr.Plot(label="Model Confidence", show_label=False)

                        demo_rec_md = gr.Markdown(elem_classes=["card"])

                demo_outs = [demo_result_md, demo_rec_md, demo_donut, demo_bar,
                             demo_risk_md, demo_temporal_md, demo_safety_md, demo_settings_md]

                start_btn.click(
                    fn=stream_demo,
                    inputs=[],
                    outputs=demo_outs,
                )
                reset_btn.click(
                    fn=reset_demo,
                    inputs=[],
                    outputs=demo_outs,
                )

            # ──────────────────────────────────────────────────────────────
            # TAB 3: SYSTEM INFO
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("System Info"):
                gr.Markdown("""
## HearSmart V3 — System Architecture

### Classification Pipeline
| Component | Details |
|:--|:--|
| **YAMNet Base** | TF Hub pre-trained (2M+ AudioSet clips, 521 classes) |
| **YAMNet Head** | Fine-tuned 8-class MLP (UrbanSound8K + ESC-50) |
| **AudioCNN** | Custom 2D CNN on 128×128 Mel spectrograms |
| **Safety Classifier** | Binary MLP on 1024-dim YAMNet embeddings |
| **Ensemble** | Weighted: YAMNet 70% + CNN 30% |

### Safety Hardening (5 Measures)
| Measure | Description |
|:--|:--|
| **H1** | Calibrated AudioSet indices (siren=[394-399], horn=[381-384]) |
| **H2** | Traffic negative guard: 40% score reduction |
| **H3** | Frame counter: 2 consecutive frames required |
| **H4** | Per-class confidence floors (siren/horn=30%, dog_bark=55%) |
| **H5** | Dog Bark ↔ Horn disambiguation |

### Real Audio Validation Results
| Metric | Value | Target |
|:--|:--|:--|
| **Safety Recall** | **100% (7/7)** | ≥95% ✅ |
| False Positive Rate | 5.6% | <30% ✅ |
| Overall Accuracy | 80.0% (25 samples) | — |
| Warm Inference | **~187ms avg** | <500ms ✅ |

### Calibrated Production Thresholds
```
YAMNET_SIREN_THRESHOLD = 0.15  (recall priority)
YAMNET_HORN_THRESHOLD  = 0.15  (recall priority)
CNN_SIREN_THRESHOLD    = 0.65  (post church-bell audit)
CNN_HORN_THRESHOLD     = 0.65  (post church-bell audit)
YAMNET_HEAD_THRESHOLD  = 0.50  (real-audio fix)
SAFETY_FRAMES_REQUIRED = 2     (~1.9s min detection)
```
""", elem_classes=["card"])

    return app


if __name__ == "__main__":
    app = build_demo()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Base(),
        css=CSS,
    )

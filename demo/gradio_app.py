"""
Gradio Demo V3 — HearSmart V3 Engine Dashboard.

2-model adaptive ensemble: AudioCNN + Fine-tuned YAMNet
+ Safety Binary Classifier + Triple Safety Fusion
+ Temporal Consistency + Risk Scoring + Transition Smoothing

New dashboard elements:
  - Risk Score gauge (0.0–1.0)
  - Temporal Status: Confirmed (green) / Uncertain (amber)
  - Safety Model: Safe (green) / Danger (red)
  - Settings Status: Applied / Held / Safety Override / Conservative
"""
import logging, os, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from src.utils import setup_logging, set_seed
from src.data.label_map_v2 import CLASS_NAMES_V2, CLASS_EMOJIS_V2

setup_logging(); set_seed(42)
logger = logging.getLogger(__name__)


def predict_audio(audio_input, hearing_loss, tinnitus, age_group, preference):
    if audio_input is None:
        return ("⚠️ Upload or record audio to begin.", "", None, None,
                "—", "—", "—", "—")

    # Lazy-load the V3 engine
    try:
        from src.pipeline_v3 import get_engine
        engine = get_engine()
    except Exception as e:
        return (f"❌ Engine load failed: {e}", "", None, None,
                "—", "—", "—", "—")

    # Handle microphone tuple input
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
        return (f"❌ Inference error: {e}", "", None, None,
                "—", "—", "—", "—")
    finally:
        if cleanup and os.path.exists(audio_path):
            os.unlink(audio_path)

    # ── Build result markdown ────────────────────────────────────────────
    emoji = CLASS_EMOJIS_V2.get(result.environment, "🎵")
    env_title = result.environment.replace('_', ' ').title()

    safety_tag = ""
    if result.safety_verdict:
        safety_tag = "🚨 **SAFETY OVERRIDE ACTIVE**"

    device_str = "CUDA" if "cuda" in str(next(iter([]), "cpu")) else "CPU"

    result_md = f"""## {emoji} {env_title}

{safety_tag}

**Confidence:** {result.confidence:.1%}

⏱ **{result.latency_ms} ms** | 🤖 **V3 2-Model Ensemble**"""

    # ── Recommendation table ─────────────────────────────────────────────
    rec_md = f"""### Recommended Settings

| Setting | Value |
|:--------|:------|
| **Volume** | {result.volume}/10 |
| **Noise Reduction** | {result.noise_reduction} |
| **Directionality** | {result.directionality} |
| **Speech Enhancement** | {"On" if result.speech_enhancement else "Off"} |

_{result.reasoning}_"""

    # ── Donut chart (ensemble spread) ────────────────────────────────────
    donut_fig = go.Figure(go.Pie(
        labels=[c.replace('_', ' ').title() for c in CLASS_NAMES_V2],
        values=result.ensemble_probs,
        hole=0.65,
        marker=dict(colors=[
            "#3b82f6", "#ef4444", "#eab308", "#10b981",
            "#8b5cf6", "#f97316", "#06b6d4", "#94a3b8"
        ]),
        textinfo='none',
        hoverinfo='label+percent'
    ))
    donut_fig.update_layout(
        title="Ensemble Spread",
        title_font=dict(color="#f8fafc", size=14),
        font=dict(color="#94a3b8"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                    xanchor="center", x=0.5)
    )

    # ── Bar chart (per-model confidence) ─────────────────────────────────
    m_names = list(result.per_model.keys())
    m_confs = [result.per_model[m][1] * 100 for m in m_names]
    m_preds = [result.per_model[m][0].replace('_', ' ').title() for m in m_names]

    bar_fig = go.Figure(go.Bar(
        x=m_names,
        y=m_confs,
        text=m_preds,
        textposition='inside',
        insidetextanchor='end',
        marker_color=[
            "#ef4444" if p.lower() in ["siren", "horn"] else "#3b82f6"
            for p in m_preds
        ],
        hovertemplate="%{x}: %{y:.1f}%<br>Pred: %{text}<extra></extra>",
        marker_line_width=0
    ))
    bar_fig.update_layout(
        title="Model Confidence Profile",
        title_font=dict(color="#f8fafc", size=14),
        font=dict(color="#94a3b8"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=10, l=10, r=10),
        yaxis=dict(range=[0, 100], gridcolor='#334155',
                   zerolinecolor='#334155'),
        xaxis=dict(gridcolor='#334155')
    )

    # ── New V3 status indicators ─────────────────────────────────────────
    risk_str = f"**{result.risk_score:.2f}** — _{result.risk_action.title()}_"

    if result.temporal_status == "Confirmed":
        temporal_str = "🟢 **Confirmed**"
    else:
        temporal_str = "🟡 **Uncertain** — waiting for consistency"

    if result.safety_verdict:
        safety_str = f"🔴 **DANGER** — triggered via _{result.safety_source}_"
    else:
        safety_str = f"🟢 **Safe** (prob: {result.safety_probability:.2f})"

    status_colors = {
        "Applied": "🟢 **Applied**",
        "Conservative": "🟡 **Conservative** — halved volume adjustments",
        "Held": "⚪ **Held** — no changes applied",
        "Safety Override": "🔴 **Safety Override** — max awareness",
    }
    settings_str = status_colors.get(result.settings_status,
                                      f"⚪ {result.settings_status}")

    return (result_md, rec_md, donut_fig, bar_fig,
            risk_str, temporal_str, safety_str, settings_str)


# ─── CSS ─────────────────────────────────────────────────────────────────────
CSS = """
/* Quantico Dark Dashboard Theme Styles — V3 */
.gradio-container { max-width: 1536px !important; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; background: #0f1115 !important; }
body { background: #0f1115 !important; color: #f8fafc !important; }
.hdr { padding: 20px 24px; border-radius: 12px; background: #1a1d24; border: 1px solid #2d3748; margin-bottom: 24px; text-align: left; }
.hdr h1 { color: #f8fafc !important; font-size: 1.8em !important; font-weight: 700 !important; letter-spacing: -0.02em; margin: 0 0 6px !important; }
.hdr p  { color: #94a3b8 !important; margin: 0 !important; font-size: 0.9em; font-weight: 400; }
.card   { background: #1a1d24; border: 1px solid #2d3748; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }
.status-card { background: #1a1d24; border: 1px solid #2d3748; border-radius: 10px; padding: 16px 20px; }
h3 { color: #e2e8f0 !important; font-weight: 600 !important; text-transform: uppercase; font-size: 0.85em !important; letter-spacing: 0.05em; margin-top: 0 !important; border-bottom: 1px solid #2d3748; padding-bottom: 10px;}
.btn { background: #3b82f6 !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 1em !important; color: #ffffff !important; transition: all 0.2s ease;}
.btn:hover { background: #2563eb !important; }
table { border-collapse: separate !important; border-spacing: 0 !important; width: 100% !important; margin-top: 10px !important;}
th { border-bottom: 1px solid #2d3748 !important; padding: 10px 14px !important; text-align: left !important; color: #94a3b8 !important; font-weight: 600 !important; font-size: 0.75em !important; text-transform: uppercase !important; letter-spacing: 0.05em !important;}
td { padding: 12px 14px !important; border-bottom: 1px solid #1e293b !important; font-size: 0.9em !important; color: #e2e8f0 !important; font-weight: 500 !important;}
footer { display: none !important; }
div.form { border: 1px solid #2d3748 !important; box-shadow: none !important; border-radius: 12px !important; background: #1a1d24 !important;}
.gradio-row { display: flex; flex-wrap: wrap; gap: 24px; }
"""

HEADER = """<div class="hdr">
  <h1>HearSmart V3 &mdash; Live Feed</h1>
  <p>2-Model Ensemble &middot; Safety Classifier &middot; Risk Scoring &middot; Temporal Logic</p>
</div>"""


def build_demo():
    with gr.Blocks(theme=gr.themes.Base(), css=CSS,
                   title="HearSmart V3") as demo:
        gr.HTML(HEADER)

        with gr.Row(equal_height=False):
            # ── Left Sidebar ─────────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Source Feed")
                audio_in = gr.Audio(
                    sources=["upload", "microphone"], type="filepath",
                    label="Upload or Record"
                )

                with gr.Accordion("Context (Optional)", open=False):
                    hl_dd = gr.Dropdown(
                        ["none", "mild", "moderate", "severe"],
                        value="none", label="Hearing Loss"
                    )
                    tin_cb = gr.Checkbox(label="Tinnitus", value=False)
                    age_dd = gr.Dropdown(
                        ["young", "adult", "elderly"],
                        value="adult", label="Age Group"
                    )
                    pref_dd = gr.Dropdown(
                        ["balanced", "speech", "music"],
                        value="balanced", label="Preference"
                    )

                predict_btn = gr.Button(
                    "Analyze Feed", variant="primary", elem_classes=["btn"]
                )

                gr.Markdown("""### Safety Protocols
🚨 **Triple Safety Fusion Active**
Binary classifier + YAMNet raw scores + CNN confirmation.
`Siren` and `Horn` trigger **instant override**.""",
                            elem_classes=["card"])

            # ── Main Dashboard ───────────────────────────────────────────
            with gr.Column(scale=7):
                # Top result
                result_md = gr.Markdown(
                    "### Status\nSystem idles.", elem_classes=["card"]
                )

                # V3 Status Indicators Row
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Risk Score", elem_classes=["status-card"])
                        risk_md = gr.Markdown("—", elem_classes=["status-card"])
                    with gr.Column(scale=1):
                        gr.Markdown("### Temporal", elem_classes=["status-card"])
                        temporal_md = gr.Markdown("—", elem_classes=["status-card"])
                    with gr.Column(scale=1):
                        gr.Markdown("### Safety Model", elem_classes=["status-card"])
                        safety_md = gr.Markdown("—", elem_classes=["status-card"])
                    with gr.Column(scale=1):
                        gr.Markdown("### Settings", elem_classes=["status-card"])
                        settings_md = gr.Markdown("—", elem_classes=["status-card"])

                # Chart Row
                with gr.Row():
                    with gr.Column(scale=1):
                        donut_plot = gr.Plot(
                            label="Ensemble Spread", show_label=False
                        )
                    with gr.Column(scale=1):
                        bar_plot = gr.Plot(
                            label="Model Confidence", show_label=False
                        )

                # Settings Data Table
                rec_md = gr.Markdown(elem_classes=["card"])

        # ── Sample library ───────────────────────────────────────────────
        sample_dir = Path(__file__).parent / "samples"
        if sample_dir.exists():
            samples = [[str(p)] for p in sorted(sample_dir.glob("*.wav"))[:3]]
            if samples:
                gr.Examples(
                    examples=samples, inputs=[audio_in], label="Recent Feeds"
                )

        ins = [audio_in, hl_dd, tin_cb, age_dd, pref_dd]
        outs = [result_md, rec_md, donut_plot, bar_plot,
                risk_md, temporal_md, safety_md, settings_md]

        predict_btn.click(fn=predict_audio, inputs=ins, outputs=outs)
        audio_in.change(fn=predict_audio, inputs=ins, outputs=outs)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0", server_port=7860,
        share=False, show_error=True, inbrowser=True
    )

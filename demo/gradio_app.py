"""
Gradio Demo v3 — 8-Class Optimized (Phase 9).
5-model adaptive ensemble: RF + SVM + XGBoost + CNN(v2) + YAMNet(v2)
"""
import logging, os, sys, tempfile, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gradio as gr
import joblib
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as torch_F

from src.features.extractor import extract_mel_spectrogram, extract_mfcc, extract_waveform
from src.utils import setup_logging, set_seed
from src.data.label_map_v2 import (
    CLASS_NAMES_V2, CLASS_EMOJIS_V2,
    SAFETY_CLASSES_V2, SAFETY_THRESHOLD_V2, DEFAULT_THRESHOLD_V2,
    get_recommendation, get_safety_threshold,
)

setup_logging(); set_seed(42)
logger = logging.getLogger(__name__)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODELS: dict = {}
_WEIGHTS: dict = {}


def _load_cnn():
    """Lazy import CNN class."""
    from src.models_v2 import AudioCNN_V2
    return AudioCNN_V2


def _ensure_models_loaded():
    global _WEIGHTS
    if _MODELS: return

    import tensorflow as tf, tensorflow_hub as hub
    for g in tf.config.list_physical_devices("GPU"): tf.config.experimental.set_memory_growth(g, True)

    from xgboost import XGBClassifier
    logger.info("Loading 5-model v2 ensemble …")

    # Traditional ML
    _MODELS["rf"]  = joblib.load("models/rf_v2.pkl")
    _MODELS["svm"] = joblib.load("models/svm_v2.pkl")
    xgb = XGBClassifier(); xgb.load_model("models/xgb_v2.json"); _MODELS["xgb"] = xgb

    # CNN v2
    CNN = _load_cnn()
    cnn = CNN(8).to(torch.device(_DEVICE))
    cnn.load_state_dict(torch.load("models/cnn_v2.pt", map_location=_DEVICE))
    cnn.eval(); _MODELS["cnn"] = cnn

    # YAMNet v2
    _MODELS["yamnet_base"] = hub.load("https://tfhub.dev/google/yamnet/1")
    _MODELS["yamnet_head"] = tf.keras.models.load_model("models/yamnet_v2.h5")
    _MODELS["yamnet_mu"]   = np.load("features/yamnet_v2/mu.npy")
    _MODELS["yamnet_sig"]  = np.load("features/yamnet_v2/sig.npy")

    # Weights
    wfile = "results/ensemble_weights_v2_8class.json"
    if os.path.exists(wfile):
        with open(wfile) as f: _WEIGHTS = json.load(f)["weights"]
    else:
        _WEIGHTS = {"rf": 0.10, "svm": 0.15, "xgb": 0.15, "cnn": 0.25, "yamnet": 0.35}
    logger.info("Models ready.")


def _infer(audio_path):
    import tensorflow as tf
    t0 = time.perf_counter()

    mfcc  = extract_mfcc(audio_path).reshape(1, -1)
    wave  = extract_waveform(audio_path)
    mel   = extract_mel_spectrogram(audio_path)
    mel_t = torch.from_numpy(mel.transpose(2,0,1)[None].astype(np.float32)).to(_DEVICE)

    pp = {}
    pp["rf"]  = _MODELS["rf"].predict_proba(mfcc).astype(np.float32)
    try: pp["svm"] = _MODELS["svm"].predict_proba(mfcc).astype(np.float32)
    except:
        sc = _MODELS["svm"].named_steps["scaler"].transform(mfcc)
        d  = _MODELS["svm"].named_steps["svm"].decision_function(sc).astype(np.float32)
        d -= d.max(1, keepdims=True); pp["svm"] = np.exp(d)/np.exp(d).sum(1, keepdims=True)
    pp["xgb"] = _MODELS["xgb"].predict_proba(mfcc).astype(np.float32)
    with torch.no_grad():
        pp["cnn"] = torch_F.softmax(_MODELS["cnn"](mel_t), dim=1).cpu().numpy().astype(np.float32)

    wav_tf = tf.constant(wave.flatten(), dtype=tf.float32)
    _, emb, _ = _MODELS["yamnet_base"](wav_tf)
    emb_n = ((tf.reduce_mean(emb, 0, keepdims=True).numpy() - _MODELS["yamnet_mu"]) / _MODELS["yamnet_sig"]).astype(np.float32)
    pp["yamnet"] = _MODELS["yamnet_head"].predict(emb_n, verbose=0).astype(np.float32)

    ORDER = ["rf", "svm", "xgb", "cnn", "yamnet"]
    w = np.array([_WEIGHTS.get(m, 0.2) for m in ORDER], dtype=np.float32); w /= w.sum()

    yp = pp["yamnet"][0]; cp = pp["cnn"][0]
    ym, yc = float(yp.max()), int(yp.argmax())
    cm, cc = float(cp.max()), int(cp.argmax())

    # Adaptive boosting — YAMNet primary
    if ym > 0.35:
        boost = min(0.35, (ym - 0.35) * 1.8)
        w[4] += boost; w[3] += boost * 0.4
        w[0] *= 0.2; w[1] *= 0.3; w[2] *= 0.3
    if cc == yc: w[0] *= 0.2; w[1] *= 0.2; w[2] *= 0.2

    # Safety class boosting
    cls_name_yam = CLASS_NAMES_V2[yc]
    cls_name_cnn = CLASS_NAMES_V2[cc]
    if cls_name_yam in SAFETY_CLASSES_V2 and ym > SAFETY_THRESHOLD_V2.get(cls_name_yam, 0.25): w[4] += 0.25
    if cls_name_cnn in SAFETY_CLASSES_V2 and cm > 0.30: w[3] += 0.15
    w /= w.sum()

    P   = np.stack([pp[m][0] for m in ORDER])
    ens = (P * w[:, None]).sum(0)

    pred_idx   = int(ens.argmax())
    pred_conf  = float(ens[pred_idx])
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    per_model  = {m: (CLASS_NAMES_V2[int(pp[m][0].argmax())], float(pp[m][0].max()), float(wi))
                  for m, wi in zip(ORDER, w)}

    return CLASS_NAMES_V2[pred_idx], pred_conf, ens, latency_ms, per_model


def predict_audio(audio_input, hearing_loss, tinnitus, age_group, preference):
    if audio_input is None:
        return "⚠️ Upload or record audio to begin.", "", ""

    try: _ensure_models_loaded()
    except Exception as e: return f"❌ Model load failed: {e}", "", ""

    if isinstance(audio_input, tuple):
        sr_in, arr = audio_input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf; sf.write(tmp.name, arr, sr_in); audio_path = tmp.name
        cleanup = True
    else:
        audio_path = audio_input; cleanup = False

    try:
        class_name, confidence, ens_probs, latency_ms, per_model = _infer(audio_path)
    except Exception as e:
        logger.error(str(e), exc_info=True); return f"❌ Inference error: {e}", "", ""
    finally:
        if cleanup and os.path.exists(audio_path): os.unlink(audio_path)

    rec  = get_recommendation(class_name)
    thresh = get_safety_threshold(class_name)
    emoji  = CLASS_EMOJIS_V2.get(class_name, "🎵")
    is_safe = class_name in SAFETY_CLASSES_V2
    safe_tag = "� **SAFETY ALERT**" if is_safe and confidence >= thresh else ""

    result_md = f"""## {class_name.replace('_',' ').title()}

{safe_tag}

**Confidence:** {confidence:.1%}  {"✓" if confidence >= thresh else "⚠ Low confidence — conservative settings applied"}

⏱ **{latency_ms} ms** | 💻 **{_DEVICE.upper()}** | 🤖 **5 Model Ensemble**"""

    vol_val = f"{rec['volume']}/10"
    nr_icons = {"Low": "Low", "Medium": "Medium", "High": "High", "Off": "Off"}
    di_icons = {"Omnidirectional": "Omnidirectional", "Directional": "Directional", "Adaptive": "Adaptive"}
    rec_md = f"""### Recommended Settings

| Setting | Value |
|:--------|:------|
| **Volume** | {vol_val} |
| **Noise Reduction** | {nr_icons.get(rec['noise_reduction'], '')} |
| **Directionality** | {di_icons.get(rec['directionality'], '')} |
| **Speech Enhancement** | {"On" if rec['speech_enhancement'] else "Off"} |

_{rec['reasoning']}_"""

    donut_fig = go.Figure(go.Pie(
        labels=[c.replace('_',' ').title() for c in CLASS_NAMES_V2],
        values=ens_probs,
        hole=0.65,
        marker=dict(colors=["#3b82f6","#ef4444","#eab308","#10b981","#8b5cf6","#f97316","#06b6d4","#94a3b8"]),
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
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    m_names = ["RF", "SVM", "XGB", "CNN", "YAMNet"]
    m_confs = [per_model[m][1] * 100 for m in per_model.keys()]
    m_preds = [per_model[m][0].replace('_',' ').title() for m in per_model.keys()]
    
    bar_fig = go.Figure(go.Bar(
        x=m_names,
        y=m_confs,
        text=m_preds,
        textposition='inside',
        insidetextanchor='end',
        marker_color=["#3b82f6" if p == class_name else "#ef4444" if p in ["Siren", "Horn"] else "#334155" for p in m_preds],
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
        yaxis=dict(range=[0, 100], gridcolor='#334155', zerolinecolor='#334155'),
        xaxis=dict(gridcolor='#334155')
    )

    return result_md, rec_md, donut_fig, bar_fig


# ─── CSS ─────────────────────────────────────────────────────────────────────
# ─── CSS ─────────────────────────────────────────────────────────────────────
CSS = """
/* Quantico Dark Dashboard Theme Styles */
.gradio-container { max-width: 1536px !important; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; background: #0f1115 !important; }
body { background: #0f1115 !important; color: #f8fafc !important; }
.hdr { padding: 20px 24px; border-radius: 12px; background: #1a1d24; border: 1px solid #2d3748; margin-bottom: 24px; text-align: left; }
.hdr h1 { color: #f8fafc !important; font-size: 1.8em !important; font-weight: 700 !important; letter-spacing: -0.02em; margin: 0 0 6px !important; }
.hdr p  { color: #94a3b8 !important; margin: 0 !important; font-size: 0.9em; font-weight: 400; }
.card   { background: #1a1d24; border: 1px solid #2d3748; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }
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
  <h1>Hearing Aid Optimizer Engine &mdash; Live Feed</h1>
  <p>Dashboard &middot; Analytics</p>
  
</div>"""


def build_demo():
    with gr.Blocks(theme=gr.themes.Base(), css=CSS, title="Hearing Aid AI v2") as demo:
        gr.HTML(HEADER)

        with gr.Row(equal_height=False):
            # Left Sidebar Streamlined Input
            with gr.Column(scale=2):
                gr.Markdown("### Source Feed")
                audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                    label="Upload or Record")

                with gr.Accordion("Context (Optional)", open=False):
                    hl_dd   = gr.Dropdown(["none","mild","moderate","severe"], value="none", label="Hearing Loss")
                    tin_cb  = gr.Checkbox(label="Tinnitus", value=False)
                    age_dd  = gr.Dropdown(["young","adult","elderly"], value="adult", label="Age Group")
                    pref_dd = gr.Dropdown(["balanced","speech","music"], value="balanced", label="Preference")

                predict_btn = gr.Button("Analyze Feed", variant="primary", elem_classes=["btn"])

                gr.Markdown("""### Safety Protocols
🚨 **Critical Overrides Active**
`Siren`, `Horn`, `Speech` thresholds adjusted for maximum awareness.""", elem_classes=["card"])

            # Main Dashboard Area
            with gr.Column(scale=7):
                # Top Metrics
                result_md = gr.Markdown("### Status\nSystem idles.", elem_classes=["card"])
                
                # Chart Row
                with gr.Row():
                    with gr.Column(scale=1):
                        donut_plot = gr.Plot(label="Ensemble Spread", show_label=False)
                    with gr.Column(scale=1):
                        bar_plot = gr.Plot(label="Model Confidence", show_label=False)
                
                # Settings Data Table
                rec_md    = gr.Markdown(elem_classes=["card"])

        sample_dir = Path(__file__).parent / "samples"
        if sample_dir.exists():
            samples = [[str(p)] for p in sorted(sample_dir.glob("*.wav"))[:3]]
            if samples:
                gr.Examples(examples=samples, inputs=[audio_in], label="Recent Feeds")

        ins  = [audio_in, hl_dd, tin_cb, age_dd, pref_dd]
        outs = [result_md, rec_md, donut_plot, bar_plot]
        predict_btn.click(fn=predict_audio, inputs=ins, outputs=outs)
        audio_in.change(fn=predict_audio, inputs=ins, outputs=outs)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, inbrowser=True)

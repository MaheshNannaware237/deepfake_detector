"""
Streamlit app: Deepfake & AI Image Detector

Run with:
    streamlit run app/app.py
"""

import os
import sys

# allow `from src import ...` when running via `streamlit run app/app.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
from PIL import Image

from src import config
from src.inference import (
    available_checkpoints, predict_image, generate_gradcam_overlay, generate_fft_visual
)

st.set_page_config(
    page_title="Deepfake & AI Image Detector",
    page_icon="◈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system — "forensic instrument" identity.
#
# Palette:   #0a0c10 base / #15191f panel / #2a3038 hairline / #e8eaed text
#            #7dd3c0 verified (real)      / #e8654a flagged (fake)
#            #8b95a1 muted secondary text
# Type:      IBM Plex Mono for every measurement/label/score (this is an
#            instrument reading out data, not a marketing page) +
#            Inter for prose/explanatory copy.
# Signature: verdict rendered as an instrument readout — large tabular
#            mono percentage, a thin signal-strength bar, hairline rules
#            instead of cards/shadows, uppercase micro-labels as if
#            printed on lab equipment.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0a0c10;
    --panel: #15191f;
    --line: #2a3038;
    --text: #e8eaed;
    --muted: #8b95a1;
    --real: #7dd3c0;
    --fake: #e8654a;
}

.stApp { background-color: var(--bg); }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* kill default streamlit top padding so the masthead sits high */
.block-container { padding-top: 2rem; max-width: 1200px; }

/* ---------- masthead ---------- */
.masthead {
    display: flex; align-items: baseline; gap: 0.85rem;
    border-bottom: 1px solid var(--line);
    padding-bottom: 1.1rem; margin-bottom: 0.3rem;
}
.masthead-mark {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem;
    color: var(--real); letter-spacing: -0.02em;
}
.masthead-title {
    font-family: 'Inter', sans-serif; font-weight: 600; font-size: 1.55rem;
    color: var(--text); letter-spacing: -0.01em;
}
.masthead-tag {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase;
    margin-left: auto; padding-top: 0.3rem;
}
.subhead {
    color: var(--muted); font-size: 0.95rem; margin: 0.9rem 0 1.6rem 0;
    max-width: 62ch; line-height: 1.55;
}
.subhead b { color: var(--text); font-weight: 500; }

/* ---------- micro labels (printed-on-equipment look) ---------- */
.mono-label {
    font-family: 'IBM Plex Mono', monospace; text-transform: uppercase;
    letter-spacing: 0.12em; font-size: 0.68rem; color: var(--muted);
    margin-bottom: 0.5rem; display: block;
}

/* ---------- verdict readout ---------- */
.readout {
    border: 1px solid var(--line); background: var(--panel);
    padding: 1.4rem 1.6rem; border-radius: 2px;
}
.readout-verdict {
    font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    font-size: 1.85rem; letter-spacing: 0.01em; line-height: 1.1;
    margin: 0.3rem 0 0.6rem 0;
}
.readout-verdict.real { color: var(--real); }
.readout-verdict.fake { color: var(--fake); }
.readout-confidence {
    font-family: 'IBM Plex Mono', monospace; color: var(--muted);
    font-size: 0.85rem; margin-bottom: 0.9rem;
}
.readout-confidence .num { color: var(--text); font-weight: 500; }

/* signal-strength bar (replaces a generic progress bar) */
.signal-track {
    width: 100%; height: 6px; background: var(--line);
    border-radius: 0; position: relative; overflow: hidden; margin-top: 0.2rem;
}
.signal-fill { height: 100%; position: absolute; left: 0; top: 0; }
.signal-fill.real { background: var(--real); }
.signal-fill.fake { background: var(--fake); }
.signal-ticks {
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem;
    color: var(--muted); margin-top: 0.35rem;
}

/* ---------- panel section ---------- */
.panel-caption {
    color: var(--muted); font-size: 0.82rem; line-height: 1.5;
    margin-top: 0.6rem; border-left: 2px solid var(--line); padding-left: 0.7rem;
}

hr { border-color: var(--line) !important; }

/* sidebar */
section[data-testid="stSidebar"] { background-color: var(--panel); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .mono-label { color: var(--muted); }

/* dataframe / metric tweaks toward mono */
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }

.footer-tag {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    color: var(--muted); letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)


def verdict_block(label: str, confidence: float):
    """Renders the signature instrument-readout verdict block."""
    is_fake = (label == "fake")
    css = "fake" if is_fake else "real"
    display = "AI-GENERATED" if is_fake else "AUTHENTIC"
    pct = confidence * 100

    st.markdown(f"""
    <div class="readout">
        <span class="mono-label">Verdict</span>
        <div class="readout-verdict {css}">{display}</div>
        <div class="readout-confidence">confidence <span class="num">{pct:.1f}%</span></div>
        <div class="signal-track">
            <div class="signal-fill {css}" style="width:{pct:.1f}%;"></div>
        </div>
        <div class="signal-ticks"><span>0</span><span>50</span><span>100</span></div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown('<span class="mono-label">Configuration</span>', unsafe_allow_html=True)

models_ready = available_checkpoints()

if not models_ready:
    st.sidebar.error(
        "No trained checkpoints found in outputs/checkpoints/.\n\n"
        "Train one first:\n`python -m src.train --model baseline`"
    )
else:
    st.sidebar.markdown(
        f'<div class="panel-caption">Loaded: <b style="color:var(--text)">'
        f'{", ".join(models_ready)}</b></div>', unsafe_allow_html=True
    )

st.sidebar.markdown("<br>", unsafe_allow_html=True)
model_choice = st.sidebar.selectbox(
    "Model",
    options=models_ready if models_ready else ["baseline"],
    help="Train models with src/train.py — see README.md",
)

show_gradcam = st.sidebar.checkbox("Grad-CAM explanation", value=True)
show_fft = st.sidebar.checkbox("Frequency spectrum (FFT)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="panel-caption">Flags statistical patterns typical of GAN/diffusion '
    'synthesis. Research/educational tool — treat results as one signal, not '
    'forensic proof.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
st.markdown("""
<div class="masthead">
    <span class="masthead-mark">◈</span>
    <span class="masthead-title">Deepfake &amp; AI Image Detector</span>
    <span class="masthead-tag">CNN · Grad-CAM · FFT</span>
</div>
<p class="subhead">
    Upload a face photo to check for statistical signatures consistent with
    <b>AI-generated synthesis</b> — GAN or diffusion-model output — rather than
    an authentic camera-captured photo.
</p>
""", unsafe_allow_html=True)

tab_single, tab_batch, tab_compare = st.tabs(["Single Image", "Batch Analysis", "Model Comparison"])


# ---------------------------------------------------------------------------
# TAB 1: Single image analysis
# ---------------------------------------------------------------------------
with tab_single:
    uploaded = st.file_uploader(
        "Upload a face image", type=["jpg", "jpeg", "png"], key="single_upload",
        label_visibility="collapsed",
    )

    if uploaded is not None and models_ready:
        pil_img = Image.open(uploaded)

        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.markdown('<span class="mono-label">Input</span>', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)

        with col_result:
            with st.spinner("Analyzing..."):
                label, confidence, probs, _ = predict_image(pil_img, model_choice)

            verdict_block(label, confidence)

            st.markdown('<br><span class="mono-label">Class probabilities</span>',
                        unsafe_allow_html=True)
            prob_df = pd.DataFrame({
                "class": list(probs.keys()),
                "probability": list(probs.values()),
            }).set_index("class")
            st.bar_chart(prob_df, color="#7dd3c0")

        st.markdown("<hr>", unsafe_allow_html=True)

        col_gc, col_fft = st.columns(2)

        if show_gradcam:
            with col_gc:
                st.markdown('<span class="mono-label">Grad-CAM — where the model looked</span>',
                            unsafe_allow_html=True)
                with st.spinner("Generating Grad-CAM..."):
                    try:
                        overlay, _ = generate_gradcam_overlay(pil_img, model_choice)
                        st.image(overlay, use_container_width=True)
                        st.markdown(
                            '<div class="panel-caption">Warmer regions contributed most to '
                            'the prediction. Classic GAN tells often cluster around eyes, '
                            'hairline, and ears.</div>', unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.warning(f"Grad-CAM unavailable for this model: {e}")

        if show_fft:
            with col_fft:
                st.markdown('<span class="mono-label">Frequency spectrum — FFT</span>',
                            unsafe_allow_html=True)
                fft_arr = generate_fft_visual(pil_img)
                st.image(fft_arr, use_container_width=True, clamp=True)
                st.markdown(
                    '<div class="panel-caption">Log-magnitude 2D Fourier spectrum. GAN '
                    'upsampling layers can leave faint periodic spikes here, invisible '
                    'in the raw image.</div>', unsafe_allow_html=True,
                )

    elif not models_ready:
        st.info("Train a model first, then return to this tab. See sidebar for the command.")
    else:
        st.info("Upload an image above to begin analysis.")


# ---------------------------------------------------------------------------
# TAB 2: Batch analysis
# ---------------------------------------------------------------------------
with tab_batch:
    st.markdown('<span class="mono-label">Batch upload</span>', unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "Upload multiple face images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="batch_upload", label_visibility="collapsed",
    )

    if batch_files and models_ready:
        if st.button(f"Analyze {len(batch_files)} images", type="primary"):
            rows = []
            progress = st.progress(0.0)
            for i, f in enumerate(batch_files):
                img = Image.open(f)
                label, confidence, probs, _ = predict_image(img, model_choice)
                rows.append({
                    "filename": f.name,
                    "verdict": "AI-GENERATED" if label == "fake" else "AUTHENTIC",
                    "confidence": round(confidence * 100, 1),
                    "prob_real": round(probs.get("real", 0) * 100, 1),
                    "prob_fake": round(probs.get("fake", 0) * 100, 1),
                })
                progress.progress((i + 1) / len(batch_files))

            results_df = pd.DataFrame(rows)
            st.dataframe(results_df, use_container_width=True)

            n_fake = (results_df["verdict"] == "AI-GENERATED").sum()
            n_real = (results_df["verdict"] == "AUTHENTIC").sum()
            c1, c2 = st.columns(2)
            c1.metric("Flagged AI-generated", n_fake)
            c2.metric("Flagged authentic", n_real)

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", csv, "batch_results.csv", "text/csv")
    elif not models_ready:
        st.info("Train a model first. See sidebar for the command.")


# ---------------------------------------------------------------------------
# TAB 3: Model comparison
# ---------------------------------------------------------------------------
with tab_compare:
    st.markdown(
        '<p class="subhead">Run the same image through every trained model side by side — '
        'useful for presenting your baseline-vs-transfer-learning comparison live.</p>',
        unsafe_allow_html=True,
    )
    compare_upload = st.file_uploader(
        "Upload an image to compare across models", type=["jpg", "jpeg", "png"],
        key="compare_upload", label_visibility="collapsed",
    )

    if compare_upload and len(models_ready) > 0:
        pil_img = Image.open(compare_upload)
        st.image(pil_img, width=220)

        cols = st.columns(len(models_ready))
        for col, model_name in zip(cols, models_ready):
            with col:
                st.markdown(f'<span class="mono-label">{model_name}</span>', unsafe_allow_html=True)
                label, confidence, probs, _ = predict_image(pil_img, model_name)
                is_fake = (label == "fake")
                css = "fake" if is_fake else "real"
                verdict = "AI-GENERATED" if is_fake else "AUTHENTIC"
                st.markdown(
                    f'<div class="readout-verdict {css}" style="font-size:1.1rem;">{verdict}</div>'
                    f'<div class="readout-confidence">confidence '
                    f'<span class="num">{confidence*100:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )

        log_path = os.path.join(config.LOG_DIR, "model_comparison.csv")
        if os.path.exists(log_path):
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<span class="mono-label">Saved test-set comparison</span>',
                        unsafe_allow_html=True)
            st.dataframe(pd.read_csv(log_path), use_container_width=True)
    elif len(models_ready) == 0:
        st.info("Train at least one model first.")
    else:
        st.info("Upload an image above to compare across all trained models.")


st.markdown("<hr style='margin-top:2.5rem;'>", unsafe_allow_html=True)
st.markdown(
    '<span class="footer-tag">DEEPFAKE &amp; AI IMAGE DETECTOR · educational project · '
    'trained on 140K Real and Fake Faces (FFHQ vs StyleGAN) · '
    'not for legal, journalistic, or high-stakes verification use</span>',
    unsafe_allow_html=True,
)

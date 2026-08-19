"""
SignalDesk-style internal tool: voice emotion recognition demo + eval.

Tab 1 lets a teammate upload or pick a sample clip and see the trained
model's prediction next to a transparent heuristic baseline (pitch + energy
only). Tab 2 shows how the two were actually evaluated on speakers the
model never trained on, so the "does it work" claim is checkable, not just
asserted.

Run: streamlit run streamlit_app.py
Needs trained_model/model.joblib, trained_model/heuristic.joblib, trained_model/eval_results.json -
from unzipping trained_model.zip (downloaded from running voice_emotion_recognition.ipynb in
Colab) into trained_model/.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import EMOTION_MAP, extract_features  # noqa: E402

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "trained_model" / "model.joblib"
HEURISTIC_PATH = ROOT / "trained_model" / "heuristic.joblib"
EVAL_PATH = ROOT / "trained_model" / "eval_results.json"
SAMPLE_DIR = ROOT / "sample-data"

st.set_page_config(page_title="Voice Emotion Recognition", layout="wide")
st.title("Voice Emotion Recognition")
st.caption(
    "Small classifier vs. a pitch/energy heuristic, trained and evaluated on RAVDESS "
    "(24 actors, 8 emotions, speaker-independent test split)."
)

if not (MODEL_PATH.exists() and HEURISTIC_PATH.exists() and EVAL_PATH.exists()):
    st.error(
        "No trained model found in trained_model/. Run voice_emotion_recognition.ipynb "
        "in Colab and unzip the downloaded trained_model.zip into trained_model/ "
        "(see README.md)."
    )
    st.stop()

model_bundle = joblib.load(MODEL_PATH)
heuristic_bundle = joblib.load(HEURISTIC_PATH)
eval_results = json.loads(EVAL_PATH.read_text())

tab_try, tab_eval = st.tabs(["Try it", "Model eval"])

# ---------------------------------------------------------------------------
# Tab 1: try it on a clip
# ---------------------------------------------------------------------------
with tab_try:
    st.subheader("Predict emotion from a voice clip")

    sample_files = sorted(SAMPLE_DIR.glob("*.wav")) if SAMPLE_DIR.exists() else []
    sample_labels = {
        f"{EMOTION_MAP[f.stem.split('-')[2]].title()} (sample)": f for f in sample_files
    }

    col1, col2 = st.columns([1, 1])
    with col1:
        choice = st.selectbox("Use a sample clip", ["-- none --"] + list(sample_labels.keys()))
    with col2:
        uploaded = st.file_uploader("...or upload your own .wav", type=["wav"])

    audio_path = None
    if uploaded is not None:
        tmp_path = ROOT / "trained_model" / "_uploaded.wav"
        tmp_path.write_bytes(uploaded.read())
        audio_path = tmp_path
    elif choice != "-- none --":
        audio_path = sample_labels[choice]

    if audio_path is not None:
        st.audio(str(audio_path))
        with st.spinner("Extracting features..."):
            feats = extract_features(audio_path)

        # trained model prediction
        X_model = np.array([[feats[f] for f in model_bundle["features"]]])
        X_model_scaled = model_bundle["scaler"].transform(X_model)
        probs = model_bundle["clf"].predict_proba(X_model_scaled)[0]
        classes = model_bundle["clf"].classes_
        model_pred = classes[probs.argmax()]

        # heuristic prediction (nearest centroid on pitch + energy)
        X_h = np.array([[feats[f] for f in heuristic_bundle["features"]]])
        X_h_scaled = heuristic_bundle["scaler"].transform(X_h)
        centroids = heuristic_bundle["centroids"]
        labels = list(centroids.keys())
        dists = {lbl: float(np.linalg.norm(X_h_scaled[0] - centroids[lbl])) for lbl in labels}
        heuristic_pred = min(dists, key=dists.get)

        c1, c2 = st.columns(2)
        c1.metric("Model prediction", model_pred.title())
        c2.metric("Heuristic prediction (pitch + energy only)", heuristic_pred.title())

        st.bar_chart(pd.Series(probs, index=classes, name="probability"))
        with st.expander("Raw features used"):
            st.write({
                "mean_pitch_hz": round(feats["mean_pitch"], 1),
                "pitch_std_hz": round(feats["pitch_std"], 1),
                "rms_energy": round(feats["rms_energy"], 4),
                "zero_crossing_rate": round(feats["zcr"], 4),
            })
    else:
        st.info("Pick a sample clip or upload a .wav to see a prediction.")

# ---------------------------------------------------------------------------
# Tab 2: how good is it, actually
# ---------------------------------------------------------------------------
with tab_eval:
    st.subheader("Held-out evaluation")
    st.caption(
        f"Test set = actors {eval_results['test_actors']} "
        f"({eval_results['n_test']} clips), never seen during training. "
        f"Train set = the other 20 actors ({eval_results['n_train']} clips)."
    )

    n_classes = len(eval_results["labels"])
    chance = 1 / n_classes
    c1, c2, c3 = st.columns(3)
    c1.metric("Model accuracy", f"{eval_results['model_accuracy']:.1%}")
    c2.metric("Heuristic accuracy", f"{eval_results['heuristic_accuracy']:.1%}")
    c3.metric("Random-guess baseline", f"{chance:.1%}")

    st.markdown(
        "The heuristic (nearest-centroid on mean pitch + energy alone) already beats "
        "chance — pitch and loudness carry real signal — but the model, using MFCCs on "
        "top of pitch and energy, does meaningfully better. That gap is the actual case "
        "for the added complexity; without it, a two-feature rule would be the more "
        "honest thing to ship."
    )

    st.markdown("#### Confusion matrix (model)")
    cm = np.array(eval_results["model_confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(eval_results["labels"], rotation=45, ha="right")
    ax.set_yticklabels(eval_results["labels"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig)

    st.markdown("#### Per-emotion precision / recall (model)")
    report = eval_results["model_report"]
    rows = [
        {"emotion": k, "precision": v["precision"], "recall": v["recall"], "f1": v["f1-score"], "n": v["support"]}
        for k, v in report.items()
        if k in eval_results["labels"]
    ]
    st.dataframe(pd.DataFrame(rows).set_index("emotion").round(2), use_container_width=True)

    st.markdown("#### Where the model and heuristic disagree")
    preds_df = pd.DataFrame(eval_results["test_predictions"])
    disagree = preds_df[preds_df["model_pred"] != preds_df["heuristic_pred"]]
    model_right_only = disagree[
        (disagree["model_pred"] == disagree["true"]) & (disagree["heuristic_pred"] != disagree["true"])
    ]
    heuristic_right_only = disagree[
        (disagree["heuristic_pred"] == disagree["true"]) & (disagree["model_pred"] != disagree["true"])
    ]
    d1, d2 = st.columns(2)
    d1.metric("Model correct, heuristic wrong", len(model_right_only))
    d2.metric("Heuristic correct, model wrong", len(heuristic_right_only))
    st.dataframe(disagree[["file", "gender", "true", "model_pred", "heuristic_pred"]], use_container_width=True)

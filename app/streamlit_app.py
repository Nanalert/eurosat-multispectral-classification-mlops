"""Streamlit interface for EuroSAT RGB classification."""

import hashlib
import sys
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from python_script.inference import get_model_details, load_model


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


st.set_page_config(
    page_title="EuroSAT RGB Classifier",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_resource(show_spinner="Loading production model")
def get_predictor():
    return load_model()


@st.cache_data(show_spinner=False)
def get_deployment_details():
    return get_model_details()


def probability_chart(result):
    frame = pd.DataFrame(
        {
            "Class": list(result["probabilities"]),
            "Probability": list(result["probabilities"].values()),
        }
    ).sort_values("Probability", ascending=False)
    return (
        alt.Chart(frame)
        .mark_bar(color="#2f6f5e")
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Class:N", sort="-x", title=None),
            tooltip=["Class:N", alt.Tooltip("Probability:Q", format=".2%")],
        )
        .properties(height=320)
    )


def render_result(result):
    st.subheader(result["label"])
    st.metric("Confidence", f"{result['confidence']:.2%}")
    st.progress(result["confidence"])

    ranked = pd.DataFrame(result["top_predictions"])[["label", "confidence"]]
    ranked.columns = ["Class", "Confidence"]
    ranked["Confidence"] = ranked["Confidence"].map(lambda value: f"{value:.2%}")
    st.dataframe(ranked, hide_index=True, width="stretch")
    st.altair_chart(probability_chart(result), width="stretch")


st.title("EuroSAT RGB Classifier")

with st.sidebar:
    st.subheader("Prediction settings")
    top_k = st.slider("Top classes", min_value=1, max_value=10, value=3)
    deployment = get_deployment_details()
    st.caption("Production ResNet18 · 64 × 64 RGB")
    st.caption(f"Version: {deployment['model_version']}")

uploaded_file = st.file_uploader(
    "Satellite image",
    type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image_digest = hashlib.sha256(image_bytes).hexdigest()
    if st.session_state.get("image_digest") != image_digest:
        st.session_state["image_digest"] = image_digest
        st.session_state.pop("prediction", None)

    image_column, result_column = st.columns([1, 1], gap="large")
    with image_column:
        if not image_bytes:
            st.error("The uploaded image is empty")
            preview = None
        elif len(image_bytes) > MAX_UPLOAD_BYTES:
            st.error("The uploaded image exceeds the 10 MB limit")
            preview = None
        else:
            try:
                preview = Image.open(BytesIO(image_bytes)).convert("RGB")
                st.image(preview, caption=uploaded_file.name, width="stretch")
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("The supplied file is not a readable image")
                preview = None

        classify = st.button(
            "Classify image",
            type="primary",
            icon=":material/analytics:",
            disabled=preview is None,
            width="stretch",
        )

    if classify:
        try:
            with st.spinner("Classifying image"):
                st.session_state["prediction"] = get_predictor().predict(
                    image_bytes,
                    top_k=top_k,
                )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
            st.error(str(error))

    with result_column:
        if "prediction" in st.session_state:
            render_result(st.session_state["prediction"])

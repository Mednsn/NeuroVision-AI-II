"""Streamlit app:  streamlit run app/app.py   (from the project root)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import pandas as pd
import streamlit as st

from src import config
from src import predict as P
from src import training as T

import streamlit as st
st.set_page_config(page_title="NeuroVision", page_icon="🧠", layout="wide")
st.title("🧠 NeuroVision – Brain tumor MRI classifier")
st.caption("Educational demo – not a medical device.")

tab_pred, tab_runs = st.tabs(["Predict", "MLflow experiments"])


@st.cache_resource
def cached_local(path):
    return P.load_model(path)



with tab_pred:
    model, names = None, None
    files = P.list_local_models()
    if files:
        choice = st.selectbox("Model", files, format_func=lambda p: p.name)
        model, names = cached_local(choice), P.load_class_names()
    else:
        st.warning("No model in models/. run `python main.py`.")

    up = st.file_uploader("MRI image", type=["jpg", "jpeg", "png", "bmp"])
    if up and model is not None:
        try:
            img = P.preprocess_bytes(up.getvalue())
            label, score, scores = P.predict_image(model, img, names)
            c1, c2 = st.columns(2)
            c1.image(img, caption="Preprocessed (224×224)")
            c2.metric("Prediction", label, f"{score:.1%}")
            c2.bar_chart(pd.Series(scores, name="probability"))
        except Exception as e:
            st.error(str(e))

with tab_runs:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    try:
        runs = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT],
                                  order_by=["start_time DESC"])
    except Exception:
        runs = pd.DataFrame()
    if runs.empty:
        st.info("No runs yet.")
    else:
        cols = [c for c in runs.columns if c in ("run_id", "tags.mlflow.runName", "start_time")
                or c.startswith(("params.model", "metrics.test_", "metrics.training", "metrics.ft_minutes"))]
        st.dataframe(runs[cols], use_container_width=True)
        st.caption("Full UI: `mlflow ui --backend-store-uri " + config.MLFLOW_TRACKING_URI + "`")

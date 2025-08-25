import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd
import plotly.express as px # pyright: ignore[reportMissingImports]
import torch
import joblib
import os

from modules import KPIGPTDataset, KPIGPTModel  
from pipeline import prepare_kpi_data, predict_future_kpi  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = {
    "emb_dim": 256,
    "context_length": 50,
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "n_features": 7,
    #"n_sites": 515,
    "n_cells": 1000,
}
model= KPIGPTModel(cfg)
# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
base_dir = st.sidebar.text_input("Dataset folder", "per_cell_grouped_output/")
file_start = st.sidebar.number_input("Start file index", min_value=0, value=110)
file_end   = st.sidebar.number_input("End file index", min_value=0, value=111)
seq_len    = st.sidebar.number_input("Sequence length", min_value=50, value=50)

# --- Main ---
st.title("📊 Telecom KPI Forecasting")

# Upload trained model
uploaded_model = st.file_uploader("Upload trained model (.pt)", type=["pth"])
if uploaded_model:
    state_dict  = torch.load(uploaded_model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    st.success("✅ Model loaded")

    if st.button("Run Preprocessing + Prediction"):
        # Step 1: Preprocess
        with st.spinner("Preparing data..."):
            all_seq = prepare_kpi_data(base_dir, file_slice=(file_start, file_end), seq_len=seq_len)
        
        st.write(f"Prepared {len(all_seq)} sequences")

        # Step 2: Run predictions
        with st.spinner("Running predictions..."):
            preds, true_vals = predict_future_kpi(model, all_seq)

        # Step 3: Visualization
        df_vis = pd.DataFrame({"Actual": true_vals.flatten(), "Predicted": preds.flatten()})
        df_vis["Index"] = range(len(df_vis))

        fig = px.line(df_vis, x="Index", y=["Actual", "Predicted"], title="Actual vs Predicted KPI")
        st.plotly_chart(fig, use_container_width=True)

        # Step 4: Download
        csv = df_vis.to_csv(index=False).encode()
        st.download_button("📥 Download predictions", csv, "predictions.csv", "text/csv")
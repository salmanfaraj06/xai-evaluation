"""Streamlit UI v2: Enhanced UX with Visualizations and Interactive Inspector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_loading import load_yaml_config, load_credit_data, split_data
from src.evaluation.run_human_llm_eval import run_llm_human_eval
from src.evaluation.run_technical_eval import run_technical_eval
from src.models import load_artifact
from src.preprocessing import transform_features

# --- Constants ---
CONFIG_PATH = Path("config/config_credit_new.yaml")
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path("config/config_credit.yaml")

# --- Loaders ---
@st.cache_data
def load_config(path: Path) -> Dict:
    return load_yaml_config(path)

@st.cache_data
def load_results(metrics_path: Path, human_path: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    tech = pd.read_csv(metrics_path) if metrics_path.exists() else None
    human = pd.read_csv(human_path) if human_path.exists() else None
    return tech, human

@st.cache_resource
def load_model_and_data(config_path: Path):
    cfg = load_yaml_config(config_path)
    # Load Model
    model_path = Path(cfg["paths"]["pretrained_model"])
    artifact = load_artifact(model_path)
    
    # Load Data (for inspector)
    X, y = load_credit_data(cfg)
    X_train, X_test, _, _ = split_data(X, y, 0.2, 42)
    
    return artifact, X_test, cfg

# --- Visualizations ---
def plot_technical_metrics(df: pd.DataFrame):
    """Bar chart for technical metrics."""
    fig = go.Figure()
    
    if "deletion_auc" in df.columns:
        fig.add_trace(go.Bar(
            x=df["method"], 
            y=df["deletion_auc"], 
            name="Deletion AUC (Lower is Better)",
            marker_color='indianred'
        ))
    
    if "insertion_auc" in df.columns:
        fig.add_trace(go.Bar(
            x=df["method"], 
            y=df["insertion_auc"], 
            name="Insertion AUC (Higher is Better)",
            marker_color='lightsalmon'
        ))
        
    if "anchor_precision" in df.columns:
        val = df.loc[df["method"] == "ANCHOR", "anchor_precision"].values[0]
        fig.add_trace(go.Bar(
            x=["ANCHOR"], 
            y=[val], 
            name="Anchor Precision",
            marker_color='lightseagreen'
        ))
    if "cf_validity_rate" in df.columns:
        val = df.loc[df["method"] == "DiCE", "cf_validity_rate"].values[0]
        fig.add_trace(go.Bar(
            x=["DiCE"], 
            y=[val], 
            name="DiCE Validity",
            marker_color='mediumpurple'
        ))

    fig.update_layout(title="Technical Fidelity Metrics", barmode='group', height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_human_ratings(df: pd.DataFrame):
    """Radar chart for human ratings."""
    numeric_cols = ["interpretability", "completeness", "actionability", "trust", "satisfaction"]
    grouped = df.groupby("explanation_type")[numeric_cols].mean().reset_index()
    
    categories = numeric_cols
    fig = go.Figure()

    for i, row in grouped.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[categories].values,
            theta=categories,
            fill='toself',
            name=row['explanation_type']
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        title="Human Persona Ratings (Average)",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Main App ---
def main():
    st.set_page_config(page_title="XAI Evaluation Dashboard", layout="wide", page_icon="📊")
    
    st.title("📊 Credit Risk XAI Dashboard")
    st.markdown("An interactive platform for evaluating and inspecting AI explanations.")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.info(f"Using config: `{CONFIG_PATH}`")
        if st.button("Reload Configuration"):
            st.cache_data.clear()
            st.rerun()

    # Load Data
    cfg = load_config(CONFIG_PATH)
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    human_dir = Path(cfg["paths"]["human_eval_dir"])
    
    tech_df, human_df = load_results(
        metrics_dir / "technical_metrics.csv",
        human_dir / "agent_llm_eval.csv"
    )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Overview", "📈 Technical Eval", "🧠 Human Eval", "🔍 Instance Inspector"])

    with tab1:
        st.header("Project Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", cfg["model"]["type"].upper())
        with col2:
            if tech_df is not None:
                best_fid = tech_df["deletion_auc"].min()
                st.metric("Best Fidelity (Del AUC)", f"{best_fid:.3f}", delta_color="inverse")
            else:
                st.metric("Technical Eval", "Pending")
        with col3:
            if human_df is not None:
                best_sat = human_df["satisfaction"].mean()
                st.metric("Avg User Satisfaction", f"{best_sat:.2f}/5.0")
            else:
                st.metric("Human Eval", "Pending")

        st.markdown("### Quick Actions")
        if st.button("Run Full Evaluation Pipeline", type="primary"):
            with st.status("Running Evaluation...", expanded=True) as status:
                st.write("Running technical metrics (SHAP, LIME, Anchor, DiCE)...")
                run_technical_eval(str(CONFIG_PATH))
                st.write("Technical evaluation complete.")
                
                st.write("Running Human-LLM evaluation...")
                run_llm_human_eval(str(CONFIG_PATH))
                st.write("Human evaluation complete.")
                
                status.update(label="Evaluation Complete!", state="complete", expanded=False)
            st.rerun()

    with tab2:
        st.header("Technical Metrics")
        st.markdown("Performance metrics for each explanation method.")
        if tech_df is not None:
            plot_technical_metrics(tech_df)
            
            st.divider()
            
            st.subheader("Detailed Metric Data")
            st.dataframe(tech_df, use_container_width=True)
            st.info("💡 **Deletion AUC**: Lower is better (removing features hurts model). **Insertion AUC**: Higher is better (adding features helps model).")
        else:
            st.warning("No technical results found. Please run the evaluation.")

    with tab3:
        st.header("Human-Centered Evaluation")
        st.markdown("How different personas perceive the explanations.")
        if human_df is not None:
            plot_human_ratings(human_df)
            
            st.divider()
            st.subheader("Persona Feedback Inspector")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("#### Configuration")
                persona = st.selectbox("Select Persona", human_df["persona_name"].unique())
                method = st.selectbox("Select Method", human_df["explanation_type"].unique())
            
            with c2:
                st.markdown("#### Feedback")
                feedback = human_df[
                    (human_df["persona_name"] == persona) & 
                    (human_df["explanation_type"] == method)
                ]
                
                if not feedback.empty:
                    for _, row in feedback.iterrows():
                        with st.container(border=True):
                            st.markdown(f"**{row['persona_name']}** ({row['persona_role']})")
                            st.markdown(f"_{row['comment']}_")
                            
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Trust", f"{row['trust']}/5")
                            m2.metric("Satisfaction", f"{row['satisfaction']}/5")
                            m3.metric("Actionability", f"{row['actionability']}/5")
                else:
                    st.info("No feedback found for this combination.")
        else:
            st.warning("No human evaluation results found. Please run the evaluation.")

    with tab4:
        st.header("Instance Inspector")
        st.markdown("Inspect specific customer predictions and feature values.")
        
        try:
            artifact, X_test, _ = load_model_and_data(CONFIG_PATH)
            
            col_sel, col_pred = st.columns([1, 2])
            
            with col_sel:
                idx = st.number_input("Select Instance Index", min_value=0, max_value=len(X_test)-1, value=0)
                
            # Predict logic
            model = artifact["model"]
            preprocessor = artifact.get("preprocessor")
            row = X_test.iloc[idx]
            
            if preprocessor:
                row_proc = transform_features(preprocessor, row.to_frame().T)
            else:
                row_proc = row.values.reshape(1, -1)
                
            prob = model.predict_proba(row_proc)[0, 1]
            
            with col_pred:
                st.metric("Default Probability", f"{prob:.2%}", delta=f"{prob - 0.5:.2%} from baseline")
                if prob > 0.5:
                    st.error("Prediction: **DEFAULT**")
                else:
                    st.success("Prediction: **NO DEFAULT**")

            st.divider()
            
            st.subheader("Feature Values")
            # Transpose for vertical readability
            display_df = row.to_frame().astype(str)
            display_df.columns = ["Value"]
            st.dataframe(display_df, use_container_width=True, height=400)
            
        except Exception as e:
            st.error(f"Could not load model/data for inspector: {e}")

if __name__ == "__main__":
    main()

"""Simplified Streamlit UI for HEXEval.

Upload model + data → Configure → Evaluate → View results & recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os
import shutil

# --- Use Case Configuration ---
USE_CASES = {
    "Heart Disease (Healthcare)": {
        "description": "Predict heart disease risk. Stakeholders: Cardiologists, Patients.",
        "config_path": "hexeval/config/eval_config.yaml",
        "data_path": "usecases/heart.csv",
        "model_path": "usecases/heart_disease_pipeline.pkl",
        "target": "target",
        "output_dir": "outputs/heart_disease",
        "default_sample_size": 100
    },
    "Credit Risk (Finance)": {
        "description": "Predict loan default. Stakeholders: Loan Officers, Risk Managers.",
        "config_path": "hexeval/config/eval_config_credit_risk.yaml",
        "data_path": "usecases/credit_risk_dataset.csv",
        "model_path": "usecases/xgboost_credit_risk_new.pkl", # Ensure this path is correct relative to root
        "target": "loan_status",
        "output_dir": "outputs/credit_risk",
        "default_sample_size": 150
    },
    "Custom Upload": {
        "description": "Upload your own model and dataset.",
        "config_path": None,
        "data_path": None,
        "model_path": None,
        "target": None,
        "output_dir": "outputs/custom_run",
        "default_sample_size": 150
    }
}

st.set_page_config(
    page_title="HEXEval",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 HEXEval - Holistic Explanation Evaluation")
st.caption("Evaluate XAI methods for your tabular model")

with st.sidebar:
    st.header("UseCase Selection")
    selected_use_case = st.selectbox(
        "Select Use Case",
        list(USE_CASES.keys()),
        index=0
    )
    
    use_case_config = USE_CASES[selected_use_case]
    st.info(use_case_config["description"])
    
    st.divider()
    
    st.header("ℹ️ About")
    st.markdown("""
    **HEXEval** helps you choose the best explanation method (SHAP, LIME, Anchor, DiCE) 
    for your stakeholders.
    
    **How it works:**
    1. Select Use Case or Upload
    2. Configure evaluation
    3. Run evaluation
    4. Get recommendations
    """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    sample_size = st.slider("Evaluation sample size", 50, 500, use_case_config["default_sample_size"])
    enable_personas = st.checkbox("Enable LLM personas", value=True, help="Requires OpenAI API key")
    
    if enable_personas:
        api_key_input = st.text_input("OpenAI API Key", type="password")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
        elif "OPENAI_API_KEY" in os.environ:
             st.info("API Key found in environment.")

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Configuration & Run", "ℹ️ Use Case Details", "📊 Results", "💡 Recommendations", "📚 Documentation"])

with tab1:
    # Check if a previous run exists for THIS use case
    default_output = Path(use_case_config["output_dir"])
    has_existing_results = (
        default_output.exists() and
        (default_output / "technical_metrics.csv").exists() and
        (default_output / "persona_ratings.csv").exists() and
        (default_output / "recommendations.json").exists()
    )
    
    if has_existing_results:
        st.success(f"✅ Found existing results for **{selected_use_case}**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"💡 Load previous results from `{default_output}` to save time.")
        with col2:
            if st.button("📂 Load Existing Results", type="primary", use_container_width=True, key="load_btn"):
                import json
                
                try:
                    # Load existing results
                    tech_df = pd.read_csv(default_output / "technical_metrics.csv")
                    persona_df = pd.read_csv(default_output / "persona_ratings.csv")
                    
                    with open(default_output / "recommendations.json") as f:
                        recs = json.load(f)
                    
                    st.session_state["results"] = {
                        "technical_metrics": tech_df,
                        "persona_ratings": persona_df,
                        "recommendations": recs,
                        "output_path": str(default_output)
                    }
                    
                    st.success(f"✅ Loaded results for {selected_use_case}!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Failed to load results: {e}")
        
        st.divider()
    
    st.header(f"Step 1: Configure {selected_use_case}")
    
    use_sample_data = False
    
    if selected_use_case != "Custom Upload":
        use_sample_data = st.checkbox("Use sample model and dataset", value=True)
    
    model_path = None
    data_path = None
    target_column = None
    
    if use_sample_data:
        # Use paths from config
        # Verify files exist
        m_path = Path(use_case_config["model_path"])
        d_path = Path(use_case_config["data_path"])
        
        # Determine if files exist relative to CWD
        # Usually Streamlit runs from root
        if not m_path.exists():
             # Try looking in 'models/' if standard structure, but USE_CASES has 'usecases/'
             # actually I moved pk to 'models/' in previous step?
             # Wait, I moved `xgboost_credit_risk_new.pkl` to `models/`
             # I need to fix the path in USE_CASES if I moved them.
             # Checking walkthrough... "Moved .pkl files to models"
             # So `xgboost_credit_risk_new.pkl` IS IN `models/`
             pass

        st.info(f"Using sample data: `{use_case_config['data_path']}`")
        st.info(f"Using sample model: `{use_case_config['model_path']}`")
        
        model_path = use_case_config["model_path"]
        data_path = use_case_config["data_path"]
        target_column = use_case_config["target"]
        
    else:
        # File Uploaders
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model")
            model_file = st.file_uploader(
                "Upload trained model",
                type=["pkl", "joblib"],
                help="sklearn or XGBoost model with predict_proba()"
            ) 
        
        with col2:
            st.subheader("Dataset")
            data_file = st.file_uploader(
                "Upload CSV dataset",
                type=["csv"],
                help="Tabular data used to train the model"
            )
            
        target_column = st.text_input(
            "Target column name",
            value=use_case_config["target"] if use_case_config["target"] else "",
            placeholder="e.g., 'loan_status', 'churn', 'diagnosis'",
            help="Name of the prediction target in your dataset"
        )
        
        # Handle file saving
        if model_file and data_file:
            # We need to save them content to pass paths to evaluate()
             # Ideally we save to a temp or the output folder
             pass 

    
    st.divider()
    
    # Run Button
    run_ready = False
    if use_sample_data:
        run_ready = True
    elif 'model_file' in locals() and model_file and 'data_file' in locals() and data_file and target_column:
        run_ready = True
        
    if run_ready:
        if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
            with st.spinner("Running evaluation... This may take a few minutes."):
                try:
                    # Handle file paths if uploaded
                    if not use_sample_data:
                        # Save uploaded files temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_model:
                            tmp_model.write(model_file.getvalue())
                            model_path = tmp_model.name
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_data:
                            tmp_data.write(data_file.getvalue())
                            data_path = tmp_data.name
                    
                    # Import here to avoid slow startup
                    from hexeval import evaluate
                    
                    # Helper: fix path if not found (because I moved files)
                    # I moved 'xgboost_credit_risk_new.pkl' to 'models/'
                    # But 'credit_risk_dataset.csv' to 'data/'
                    # And 'heart.csv' might still be in 'usecases/'
                    
                    final_model_path = model_path
                    final_data_path = data_path
                    
                    # Fix for moved files if using defaults
                    if use_sample_data:
                        # Check where I actually moved them
                        # task.md said: "Move .pkl files to models", "Move .csv files to data"
                        # USE_CASES dict above has 'usecases/...'. I should fix logic here or in dict.
                        # I will attempt to fix paths dynamically if file not found at original path
                        
                        if not os.path.exists(final_model_path):
                            # Try models/
                            basename = os.path.basename(final_model_path)
                            alt_path = f"models/{basename}"
                            if os.path.exists(alt_path):
                                final_model_path = alt_path
                        
                        if not os.path.exists(final_data_path):
                            # Try data/
                            basename = os.path.basename(final_data_path)
                            alt_path = f"data/{basename}"
                            if os.path.exists(alt_path):
                                final_data_path = alt_path
                                
                    st.write(f"Using Model: `{final_model_path}`")
                    st.write(f"Using Data: `{final_data_path}`")

                    # Run evaluation
                    results = evaluate(
                        model_path=final_model_path,
                        data_path=final_data_path,
                        target_column=target_column,
                        config_path=use_case_config["config_path"],
                        output_dir=use_case_config["output_dir"],
                        config_overrides={
                            "personas": {"enabled": enable_personas},
                            "evaluation": {"sample_size": sample_size}
                        }
                    )
                    
                    # Store in session state
                    st.session_state['results'] = results
                    st.session_state['model_name'] = model_path if use_sample_data else model_file.name
                    st.session_state['data_name'] = data_path if use_sample_data else data_file.name
                    st.session_state['current_use_case'] = selected_use_case
                    
                    # Cleanup temp files if uploaded
                    if not use_sample_data:
                        os.unlink(model_path)
                        os.unlink(data_path)
                    
                    st.success("✅ Evaluation complete! Check the Results tab.")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
                    st.exception(e)
    else:
        st.info("👆 Configure inputs to begin")

with tab2:
    st.header(f"ℹ️ {selected_use_case} Details")
    
    if selected_use_case == "Custom Upload":
        st.info("Upload your own data to see configuration details.")
    else:
        import yaml
        
        # Load Config
        try:
             with open(use_case_config["config_path"], "r") as f:
                config_data = yaml.safe_load(f)
                
             # Display Domain Info
             st.subheader("🌍 Domain Context")
             domain = config_data.get("domain", {})
             
             col1, col2 = st.columns(2)
             with col1:
                 st.markdown(f"**Task:** {domain.get('prediction_task', 'N/A')}")
                 st.markdown(f"**Stakeholders:** {domain.get('stakeholder_context', 'N/A')}")
             with col2:
                 st.markdown(f"**Positive Outcome:** {domain.get('positive_outcome', 'N/A')}")
                 st.markdown(f"**Negative Outcome:** {domain.get('negative_outcome', 'N/A')}")
             
             st.divider()
             
             # Display Personas
             st.subheader("👥 Stakeholder Personas")
             
             personas_file = config_data.get("personas", {}).get("file")
             if personas_file:
                 try:
                     # Load personas directly from YAML file (avoid import issues on Streamlit Cloud)
                     import yaml
                     
                     # Handle relative path
                     if not os.path.exists(personas_file):
                         # Try relative to project root
                         pass
                     
                     with open(personas_file, 'r') as f:
                         personas_data = yaml.safe_load(f)
                     
                     # Extract personas list
                     personas = personas_data.get('personas', [])
                     
                     for p in personas:
                         with st.expander(f"**{p['name']}** - {p['role']}"):
                             c1, c2 = st.columns([1, 2])
                             with c1:
                                 st.markdown(f"**Risk Profile:** {p.get('risk_profile', 'N/A')}")
                                 st.markdown(f"**AI Comfort:** {p.get('ai_comfort', 'N/A')}")
                             with c2:
                                 st.markdown(f"**Priorities:**")
                                 for prio in p.get('priorities', []):
                                     st.markdown(f"- {prio}")
                                 
                                 st.markdown(f"**Needs:**")
                                 st.info(p.get('explanation_preferences', ''))
                                 
                 except Exception as e:
                     st.warning(f"Could not load personas details: {e}")
             
             st.divider()
             with st.expander("Show Full Configuration (YAML)"):
                 st.code(yaml.dump(config_data), language="yaml")
                 
        except Exception as e:
            st.error(f"Could not load configuration file: {e}")

with tab3:
    st.header("Technical Metrics")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        tech_df = results['technical_metrics']
        
        # Display metrics table
        st.dataframe(tech_df, use_container_width=True, hide_index=True)
        
        # Visualize fidelity
        st.subheader("Fidelity Comparison")
        
        fidelity_data = tech_df[tech_df['fidelity_deletion'].notna()]
        
        if not fidelity_data.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Deletion (lower = better)',
                x=fidelity_data['method'],
                y=fidelity_data['fidelity_deletion'],
                marker_color='indianred'
            ))
            
            fig.add_trace(go.Bar(
                name='Insertion (higher = better)',
                x=fidelity_data['method'],
                y=fidelity_data['fidelity_insertion'],
                marker_color='lightseagreen'
            ))
            
            fig.update_layout(
                barmode='group',
                height=400,
                xaxis_title="Method",
                yaxis_title="AUC Score"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Persona ratings
        if results['persona_ratings'] is not None:
            st.divider()
            st.subheader("Persona Ratings")
            
            persona_df = results['persona_ratings']
            
            # Summary by method - ALL 6 DIMENSIONS
            rating_cols = ['trust', 'satisfaction', 'actionability', 'interpretability', 'completeness', 'decision_support']
            summary = persona_df.groupby('explanation_type')[rating_cols].mean()
            
            st.markdown("**Average Ratings by Method (1-5 scale)**")
            st.dataframe(summary.round(2), use_container_width=True)
            
            # Radar chart with all 6 dimensions
            fig_radar = go.Figure()
            
            for method in summary.index:
                fig_radar.add_trace(go.Scatterpolar(
                    r=summary.loc[method].values,
                    theta=['Trust', 'Satisfaction', 'Actionability', 'Interpretability', 'Completeness', 'Decision Support'],
                    fill='toself',
                    name=method
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                showlegend=True,
                height=500,
                title="Persona Ratings - All Dimensions"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # NEW: Persona-wise analytics
            st.divider()
            st.subheader("🎭 Persona-Wise Analysis")
            
            # Show each persona's perspective
            for persona_name in persona_df['persona_name'].unique():
                persona_data = persona_df[persona_df['persona_name'] == persona_name]
                persona_role = persona_data['persona_role'].iloc[0]
                
                with st.expander(f"**{persona_name}** - {persona_role}", expanded=False):
                    # Ratings by method for this persona
                    persona_summary = persona_data.groupby('explanation_type')[rating_cols].mean()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Average Ratings by Method**")
                        st.dataframe(persona_summary.round(2), use_container_width=True)
                    
                    with col2:
                        # Highlight best method for this persona
                        avg_scores = persona_summary.mean(axis=1)
                        if not avg_scores.empty:
                            best_method = avg_scores.idxmax()
                            best_score = avg_scores.max()
                            st.metric("Best Method", best_method, f"{best_score:.2f}/5")
                    
                    # Show comments for each method
                    st.markdown("**💬 Comments by Method**")
                    for method in persona_data['explanation_type'].unique():
                        method_comments = persona_data[persona_data['explanation_type'] == method]['comment'].values
                        if len(method_comments) > 0:
                            # Show first comment (or average if multiple runs)
                            comment = method_comments[0]
                            st.markdown(f"**{method}:**")
                            st.info(f"_{comment}_")
        
        
    else:
        st.info("Run evaluation first to see results")

with tab4:
    st.header("Recommendations")
    
    if 'results' in st.session_state and st.session_state['results']['recommendations']:
        recs = st.session_state['results']['recommendations']
        
        st.markdown("### 🎯 Which method should you use?")
        st.caption("Recommendations are based on both technical performance and persona feedback")
        
        # Show recommendations per stakeholder
        for stakeholder, rec in recs.items():
            with st.expander(f"**{stakeholder}**", expanded=True):
                st.success(f"**Recommended:** {rec['recommended_method']}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Reasoning:** {rec['reasoning']}")
                    st.markdown(f"**Persona Feedback:** {rec['persona_feedback']}")
                
                with col2:
                    st.metric("Overall Score", f"{rec['score']:.3f}")
                
                if rec.get('alternatives'):
                    st.markdown("**Alternatives:**")
                    alt_df = pd.DataFrame([
                        {"Method": method, "Score": score} 
                        for method, score in rec['alternatives'].items()
                    ])
                    st.dataframe(alt_df, use_container_width=True, hide_index=True)
        
        # NEW: Method Comparison Across Personas
        if 'results' in st.session_state and st.session_state['results']['persona_ratings'] is not None:
            st.divider()
            st.subheader("📊 Method Comparison Across All Personas")
            
            persona_df = st.session_state['results']['persona_ratings']
            
            # Pivot table: personas vs methods showing average rating
            pivot_data = persona_df.pivot_table(
                values='satisfaction',  # or use average of all ratings
                index='persona_name',
                columns='explanation_type',
                aggfunc='mean'
            )
            
            st.markdown("**Satisfaction Scores by Persona & Method**")
            st.dataframe(pivot_data.round(2).style.background_gradient(cmap='RdYlGn', vmin=1, vmax=5), 
                        use_container_width=True)
            
            st.caption("🟢 Higher scores = better fit for that persona")
    
    
    elif 'results' in st.session_state:
        st.warning("Recommendations require persona evaluation. Enable it in settings and re-run.")
    
    else:
        st.info("Run evaluation first to see recommendations")

with tab5:
    st.header("Documentation")
    
    doc_mode = st.radio(
        "Select Topic:",
        ["🚀 Prerequisites & Setup", "💡 How It Works (Concepts)", "⚙️ Configuration Guide"],
        horizontal=True
    )
    
    st.divider()
    
    # Updated paths for docs
    if doc_mode == "🚀 Prerequisites & Setup":
        try:
            with open("docs/HEXEval_Prerequisites.md", "r") as f:
                st.markdown(f.read())
        except FileNotFoundError:
            st.error("Documentation file not found: `docs/HEXEval_Prerequisites.md`")
            
    elif doc_mode == "💡 How It Works (Concepts)":
        try:
            with open("docs/HEXEval_HowItWorks.md", "r") as f:
                st.markdown(f.read())
        except FileNotFoundError:
            st.error("Documentation file not found: `docs/HEXEval_HowItWorks.md`")
            
    elif doc_mode == "⚙️ Configuration Guide":
        try:
            with open("docs/HEXEval_Configuration.md", "r") as f:
                st.markdown(f.read())
        except FileNotFoundError:
            st.error("Documentation file not found: `docs/HEXEval_Configuration.md`")

# Footer
st.divider()
st.caption("HEXEval - Holistic Explanation Evaluation Framework | Made for practitioners, not academics")

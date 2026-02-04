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
        "model_path": "usecases/xgboost_credit_risk_new.pkl", 
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
    
    st.subheader("OpenAI API Key")
    st.caption("Required for persona-based evaluation")
    
    api_key_input = st.text_input(
        "Enter your OpenAI API key",
        type="password",
        help="Your API key is used to evaluate explanations from different stakeholder perspectives"
    )
    
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.success("✓ API key configured")
    elif "OPENAI_API_KEY" in os.environ:
        st.info("✓ API key found in environment")

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
        st.success(f"Found existing results for **{selected_use_case}**")
        
        # Show the Load Existing Results button nicely
        if st.button("Load Existing Results", type="primary", use_container_width=True, key="load_btn"):
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
                
                st.success(f"Loaded results for {selected_use_case}!")
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
        
        m_path = Path(use_case_config["model_path"])
        d_path = Path(use_case_config["data_path"])
       
        if not m_path.exists():
            
             pass

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
          
             pass 

    
    st.divider()
    
    # Run Button
    run_ready = False
    if use_sample_data:
        run_ready = True
    elif 'model_file' in locals() and model_file and 'data_file' in locals() and data_file and target_column:
        run_ready = True
        
    if run_ready:
        if st.button("Run Evaluation", type="primary", use_container_width=True):
            # Create progress tracking UI
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress callback function
            def update_progress(percent: float, message: str):
                progress_bar.progress(int(percent) / 100)
                status_text.text(f"⏳ {message}")
            
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
                
                
                from hexeval import evaluate
                
                
                
                final_model_path = model_path
                final_data_path = data_path
                
                if use_sample_data:
                    
                    if not os.path.exists(final_model_path):
                       
                        basename = os.path.basename(final_model_path)
                        alt_path = f"models/{basename}"
                        if os.path.exists(alt_path):
                            final_model_path = alt_path
                    
                    if not os.path.exists(final_data_path):
                        
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
                        "personas": {"enabled": True}, 
                        "evaluation": {"sample_size": use_case_config["default_sample_size"]}
                    },
                    progress_callback=update_progress
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
                
                # Clear progress indicators and show success
                progress_bar.empty()
                status_text.empty()
                st.success("Evaluation complete! Check the Results tab.")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Evaluation failed: {str(e)}")
                st.exception(e)
    else:
        st.info("👆 Configure inputs to begin")

with tab2:
    st.header(f"{selected_use_case} Details")
    
    if selected_use_case == "Custom Upload":
        st.info("Upload your own data to see configuration details.")
    else:
        import yaml
        
        # Load Config
        try:
             with open(use_case_config["config_path"], "r") as f:
                config_data = yaml.safe_load(f)
                
             # Display Domain Info
             st.subheader("Domain Context")
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
             st.subheader("Stakeholder Personas")
             
             personas_file = config_data.get("personas", {}).get("file")
             if personas_file:
                 try:
                     # Load personas directly from YAML file 
                     import yaml
                     
                     # Handle relative path
                     if not os.path.exists(personas_file):
                         pass
                     
                     with open(personas_file, 'r') as f:
                         personas_data = yaml.safe_load(f)
                     
                     # Extract personas list (handle both dict and list structures)
                     if isinstance(personas_data, list):
                         personas = personas_data
                     else:
                         personas = personas_data.get('personas', [])
                     
                     for idx, p in enumerate(personas):
                         # Expand first persona by default to reduce empty space
                         with st.expander(f"**{p['name']}** - {p['role']}", expanded=(idx == 0)):
                             # More compact layout without excessive columns
                             st.markdown(f"**Risk Profile:** {p.get('risk_profile', 'N/A')} | **AI Comfort:** {p.get('ai_comfort', 'N/A')}")
                             
                             st.markdown("**Priorities:**")
                             for prio in p.get('priorities', []):
                                 st.markdown(f"• {prio}")
                             
                             st.markdown("**Explanation Preferences:**")
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
        
        st.divider()
        
        # ===== 1. QUALITY METRICS (Does it work?) =====
        st.subheader("Quality Metrics")
        st.caption("Measures how well each method performs: Fidelity (SHAP/LIME), Precision & Coverage (Anchor), Validity (DiCE)")
        
        fig_quality = go.Figure()
        
        # SHAP/LIME: Fidelity (Deletion & Insertion)
        fidelity_data = tech_df[tech_df['fidelity_deletion'].notna()]
        if not fidelity_data.empty:
            fig_quality.add_trace(go.Bar(
                name='Deletion AUC ↓',
                x=fidelity_data['method'],
                y=fidelity_data['fidelity_deletion'],
                marker_color='#e74c3c',
                text=fidelity_data['fidelity_deletion'].round(3),
                textposition='outside'
            ))
            fig_quality.add_trace(go.Bar(
                name='Insertion AUC ↑',
                x=fidelity_data['method'],
                y=fidelity_data['fidelity_insertion'],
                marker_color='#2ecc71',
                text=fidelity_data['fidelity_insertion'].round(3),
                textposition='outside'
            ))
        
        # Anchor: Precision & Coverage
        anchor_data = tech_df[tech_df['rule_accuracy'].notna()]
        if not anchor_data.empty:
            fig_quality.add_trace(go.Bar(
                name='Precision ↑',
                x=anchor_data['method'],
                y=anchor_data['rule_accuracy'],
                marker_color='#3498db',
                text=anchor_data['rule_accuracy'].round(3),
                textposition='outside'
            ))
            fig_quality.add_trace(go.Bar(
                name='Coverage ↑',
                x=anchor_data['method'],
                y=anchor_data['rule_applicability'],
                marker_color='#9b59b6',
                text=anchor_data['rule_applicability'].round(3),
                textposition='outside'
            ))
        
        # DiCE: Validity/Success Rate
        dice_data = tech_df[tech_df['counterfactual_success'].notna()]
        if not dice_data.empty:
            fig_quality.add_trace(go.Bar(
                name='Validity ↑',
                x=dice_data['method'],
                y=dice_data['counterfactual_success'],
                marker_color='#1abc9c',
                text=dice_data['counterfactual_success'].round(3),
                textposition='outside'
            ))
        
        fig_quality.update_layout(
            barmode='group',
            height=450,
            xaxis_title="Method",
            yaxis_title="Score",
            yaxis_range=[0, 1.1],
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_quality, use_container_width=True)
        
        st.divider()
        
        # ===== 2. SIMPLICITY METRICS =====
        st.subheader("Parsimony Comparison")
        st.caption("Number of features or conditions used by each method (lower is better for simpler explanations)")
        
        fig_parsimony = go.Figure()
        
        parsimony_methods = []
        parsimony_values = []
        parsimony_labels = []
        
        # SHAP/LIME: Sparsity (num features)
        for _, row in tech_df.iterrows():
            if pd.notna(row.get('num_important_features')):
                parsimony_methods.append(row['method'])
                parsimony_values.append(row['num_important_features'])
                parsimony_labels.append(f"{row['num_important_features']:.1f} features")
        
        # Anchor: Rule Length (num conditions)
        anchor_data = tech_df[tech_df['rule_length'].notna()]
        if not anchor_data.empty:
            for _, row in anchor_data.iterrows():
                parsimony_methods.append(row['method'])
                parsimony_values.append(row['rule_length'])
                parsimony_labels.append(f"{row['rule_length']:.1f} conditions")
        
        # DiCE: Counterfactual Sparsity (num features changed)
        dice_data = tech_df[tech_df['counterfactual_sparsity'].notna()]
        if not dice_data.empty:
            for _, row in dice_data.iterrows():
                parsimony_methods.append(row['method'])
                parsimony_values.append(row['counterfactual_sparsity'])
                parsimony_labels.append(f"{row['counterfactual_sparsity']:.1f} features")
        
        if parsimony_methods:
            fig_parsimony.add_trace(go.Bar(
                x=parsimony_methods,
                y=parsimony_values,
                marker_color='#f39c12',
                text=parsimony_labels,
                textposition='outside'
            ))
            
            fig_parsimony.update_layout(
                height=400,
                xaxis_title="Method",
                yaxis_title="Parsimony Score (lower = better)",
                showlegend=False
            )
            
            st.plotly_chart(fig_parsimony, use_container_width=True)
        else:
            st.info("No parsimony metrics available")
        
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
            
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            
            for idx, method in enumerate(summary.index):
                fig_radar.add_trace(go.Scatterpolar(
                    r=summary.loc[method].values,
                    theta=['Trust', 'Satisfaction', 'Actionability', 'Interpretability', 'Completeness', 'Decision Support'],
                    fill='toself',
                    name=method,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=6)
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
                    st.markdown("**Comments by Method**")
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
        
        st.markdown("### Which method should you use?")
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
            st.subheader("Method Comparison Across All Personas")
            
            persona_df = st.session_state['results']['persona_ratings']
            
            # Pivot table: personas vs methods showing average rating
            pivot_data = persona_df.pivot_table(
                values='satisfaction', 
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
    
    # Simple dropdown navigation
    doc_options = {
       
        "💡 How It Works": "how_it_works",
        "📋 Prerequisites & Setup": "prerequisites",
        "⚙️ Configuration Guide": "configuration"
    }
    
    selected_doc = st.selectbox(
        "Select Topic:",
        options=list(doc_options.keys()),
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Load and display content
    doc_files = {
        
        'how_it_works': 'docs/HEXEval_HowItWorks.md',
        'prerequisites': 'docs/HEXEval_Prerequisites.md',
        'configuration': 'docs/HEXEval_Configuration.md'
    }
    
    doc_key = doc_options[selected_doc]
    doc_file = doc_files.get(doc_key)
    
    if doc_file:
        try:
            with open(doc_file, "r") as f:
                st.markdown(f.read())
        except FileNotFoundError:
            st.error(f"Documentation file not found: `{doc_file}`")

# Footer
st.divider()
st.caption("HEXEval - Holistic Explanation Evaluation Framework")

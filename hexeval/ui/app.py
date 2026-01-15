"""Simplified Streamlit UI for HEXEval.

Upload model + data → Configure → Evaluate → View results & recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="HEXEval",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 HEXEval - Holistic Explanation Evaluation")
st.caption("Evaluate XAI methods for your tabular model")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **HEXEval** helps you choose the best explanation method (SHAP, LIME, Anchor, DiCE) 
    for your stakeholders.
    
    **How it works:**
    1. Upload your model + data
    2. Configure evaluation
    3. Run evaluation
    4. Get recommendations
    """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    sample_size = st.slider("Evaluation sample size", 50, 300, 150)
    enable_personas = st.checkbox("Enable LLM personas", value=False, help="Requires OpenAI API key")
    
    if enable_personas:
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Run", "📊 Results", "💡 Recommendations"])

with tab1:
    st.header("Step 1: Upload Your Model & Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model")
        model_file = st.file_uploader(
            "Upload trained model",
            type=["pkl", "joblib"],
            help="sklearn or XGBoost model with predict_proba()"
        )
        
        if model_file:
            st.success(f"✓ Uploaded: {model_file.name}")
    
    with col2:
        st.subheader("Dataset")
        data_file = st.file_uploader(
            "Upload CSV dataset",
            type=["csv"],
            help="Tabular data used to train the model"
        )
        
        if data_file:
            st.success(f"✓ Uploaded: {data_file.name}")
    
    target_column = st.text_input(
        "Target column name",
        placeholder="e.g., 'loan_status', 'churn', 'diagnosis'",
        help="Name of the prediction target in your dataset"
    )
    
    st.divider()
    
    if model_file and data_file and target_column:
        if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
            with st.spinner("Running evaluation... This may take a few minutes."):
                try:
                    # Save uploaded files temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_model:
                        tmp_model.write(model_file.getvalue())
                        model_path = tmp_model.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_data:
                        tmp_data.write(data_file.getvalue())
                        data_path = tmp_data.name
                    
                    # Import here to avoid slow startup
                    from hexeval import evaluate
                    
                    # Run evaluation
                    results = evaluate(
                        model_path=model_path,
                        data_path=data_path,
                        target_column=target_column,
                    )
                    
                    # Store in session state
                    st.session_state['results'] = results
                    st.session_state['model_name'] = model_file.name
                    st.session_state['data_name'] = data_file.name
                    
                    # Cleanup temp files
                    os.unlink(model_path)
                    os.unlink(data_path)
                    
                    st.success("✅ Evaluation complete! Check the Results tab.")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
                    st.exception(e)
    else:
        st.info("👆 Upload model, data, and specify target column to begin")

with tab2:
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

with tab3:
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

# Footer
st.divider()
st.caption("HEXEval - Holistic Explanation Evaluation Framework | Made for practitioners, not academics")

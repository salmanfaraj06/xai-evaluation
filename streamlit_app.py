"""Streamlit UI for the VXAI-guided credit risk evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from src.data_loading import load_yaml_config
from src.evaluation.run_human_llm_eval import run_llm_human_eval
from src.evaluation.run_technical_eval import run_technical_eval
from src.evaluation.vxai_eval_plan import VXAI_EVAL_PLAN
from src.models import load_artifact

CONFIG_PATH = Path("config/config_credit.yaml")


@st.cache_data(show_spinner=False)
def load_config(path: Path) -> Dict:
    return load_yaml_config(path)


@st.cache_data(show_spinner=False)
def load_metrics(metrics_path: Path) -> pd.DataFrame | None:
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None


@st.cache_data(show_spinner=False)
def load_human_template(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


@st.cache_data(show_spinner=False)
def load_human_llm_results(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_human_llm_summary(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_resource(show_spinner=False)
def load_model_info(artifact_path: Path) -> Dict:
    art = load_artifact(artifact_path)
    return {
        "default_threshold": float(art.get("default_threshold", 0.5)),
        "feature_count": len(art.get("feature_names", [])),
        "has_preprocessor": "preprocessor" in art and art["preprocessor"] is not None,
        "keys": list(art.keys()),
    }


def render_vxai_plan():
    plan_rows = []
    for method, cfg in VXAI_EVAL_PLAN.items():
        plan_rows.append(
            {
                "Method": method,
                "Type": cfg["explanation_type"],
                "Contextuality": cfg["vxai_contextuality_level"],
                "Desiderata": ", ".join(cfg["desiderata"]),
                "Metrics": ", ".join(cfg["metrics"]),
            }
        )
    df = pd.DataFrame(plan_rows)
    st.subheader("Explanation plan (methods → desiderata)")
    st.dataframe(df, width="stretch", hide_index=True)


def render_metrics(metrics_df: pd.DataFrame):
    st.subheader("Technical metrics")
    st.dataframe(metrics_df, width="stretch")


def render_human_template(md_text: str):
    st.subheader("Human-centred evaluation template")
    st.markdown(md_text)


def main():
    st.set_page_config(
        page_title="Dual-Lens XAI Evaluation",
        layout="wide",
    )
    st.title("Dual-Lens XAI Evaluation")
    st.caption(
        "Inspect the explanation plan. Run the evaluation to see technical metrics and persona (LLM) scores."
    )

    cfg = load_config(CONFIG_PATH)
    paths_cfg = cfg.get("paths", {})
    metrics_path = Path(paths_cfg.get("metrics_dir", "outputs/vxai_metrics")) / "technical_metrics.csv"
    artifact_path = Path(paths_cfg.get("pretrained_model", "xgboost_loan_default_research.pkl"))
    human_llm_path = Path(paths_cfg.get("human_eval_dir", "outputs/human_eval_templates")) / "agent_llm_eval.csv"
    human_llm_summary_path = Path(paths_cfg.get("human_eval_dir", "outputs/human_eval_templates")) / "agent_llm_eval_summary.csv"

    with st.sidebar:
        st.header("Artifacts")
        st.write(f"Config: `{CONFIG_PATH}`")
        st.write(f"Model: `{artifact_path}`")
        st.write(f"Metrics: `{metrics_path}`")
        st.write(f"Human LLM eval: `{human_llm_path}`")
        st.divider()
        model_info = None
        try:
            model_info = load_model_info(artifact_path)
            st.success(
                f"Loaded model artifact. Features: {model_info['feature_count']}, "
                f"Default threshold: {model_info['default_threshold']:.2f}"
            )
        except Exception as exc:  # pragma: no cover - UI path
            st.error(f"Failed to load model artifact: {exc}")

    render_vxai_plan()

    st.markdown("### Run end-to-end evaluation")
    metrics_df = None
    llm_df = None
    llm_summary = None

    if st.button("Run XAI technical + persona (LLM) evaluation", type="primary"):
        with st.spinner("Running technical metrics..."):
            try:
                tech_res = run_technical_eval(str(CONFIG_PATH), model_path=str(artifact_path))
                metrics_df = tech_res["metrics"]
                st.success("Technical evaluation completed.")
            except Exception as exc:  # pragma: no cover - runtime path
                st.error(f"Technical evaluation failed: {exc}")
                metrics_df = None
        with st.spinner("Running LLM persona evaluation..."):
            try:
                llm_res = run_llm_human_eval(str(CONFIG_PATH))
                st.success("LLM persona evaluation completed.")
            except Exception as exc:  # pragma: no cover - runtime path
                st.error(f"LLM persona evaluation failed: {exc}")

        # refresh caches after writing outputs
        load_metrics.clear()
        load_human_llm_results.clear()
        load_human_llm_summary.clear()

    # If nothing from local run, try cached loaders
    if metrics_df is None:
        metrics_df = load_metrics(metrics_path)
    if llm_df is None:
        llm_df = load_human_llm_results(human_llm_path)
    if llm_summary is None:
        llm_summary = load_human_llm_summary(human_llm_summary_path)

    if metrics_df is not None or llm_df is not None or llm_summary is not None:
        st.markdown("### Technical evaluation results")
        if metrics_df is not None:
            render_metrics(metrics_df)
            # Key headline metrics if present
            col_a, col_b, col_c, col_d = st.columns(4)
            if "deletion_auc" in metrics_df.columns and metrics_df["deletion_auc"].notna().any():
                col_a.metric("Best deletion AUC", f"{metrics_df['deletion_auc'].max():.3f}")
            if "insertion_auc" in metrics_df.columns and metrics_df["insertion_auc"].notna().any():
                col_b.metric("Best insertion AUC", f"{metrics_df['insertion_auc'].max():.3f}")
            if "anchor_precision" in metrics_df.columns and metrics_df["anchor_precision"].notna().any():
                col_c.metric("Anchor precision", f"{metrics_df['anchor_precision'].max():.3f}")
            if "cf_validity_rate" in metrics_df.columns and metrics_df["cf_validity_rate"].notna().any():
                col_d.metric("DiCE validity", f"{metrics_df['cf_validity_rate'].max():.3f}")
        else:
            st.info("No metrics found yet. Run the evaluation above.")

        st.markdown("### LLM-based persona ratings")
        if llm_df is not None:
            st.dataframe(llm_df, width="stretch")
        else:
            st.info("No LLM persona ratings yet.")
        if llm_summary is not None:
            st.subheader("Persona ratings (mean by explanation type)")
            styled = llm_summary.style.highlight_max(
                subset=[c for c in llm_summary.columns if c != "explanation_type"],
                axis=1,
                color="#2a9d8f",
            )
            st.dataframe(styled, width="stretch", hide_index=True)
        elif llm_df is None:
            st.info("Run the LLM persona evaluation to see aggregated scores.")

        # Correlation between human means and technical metrics
        if llm_summary is not None and metrics_df is not None:
            human_cols = [c for c in llm_summary.columns if c != "explanation_type"]
            tech_cols = [
                c for c in metrics_df.columns
                if c not in ["method"] and metrics_df[c].notna().any()
            ]
            # align on method name
            human_aligned = llm_summary.rename(columns={"explanation_type": "method"})
            merged = pd.merge(metrics_df, human_aligned, on="method", how="inner")
            if not merged.empty and human_cols and tech_cols:
                st.subheader("Correlation: technical vs human metrics")
                st.markdown("Merged view (methods with both technical and human scores):")
                st.dataframe(merged[["method"] + tech_cols + human_cols], width="stretch")

                corr = merged[tech_cols + human_cols].corr()
                corr_view = corr.loc[tech_cols, human_cols]
                styled_corr = corr_view.style.background_gradient(cmap="Blues", axis=None)
                st.markdown("Correlation matrix (technical rows vs human columns):")
                st.dataframe(styled_corr, width="stretch")

    st.markdown("### Maintenance")
    if st.button("Clear all outputs", type="secondary"):
        try:
            if metrics_path.exists():
                metrics_path.unlink()
            if human_llm_path.exists():
                human_llm_path.unlink()
            if human_llm_summary_path.exists():
                human_llm_summary_path.unlink()
            load_metrics.clear()
            load_human_llm_results.clear()
            load_human_llm_summary.clear()
            st.success("All outputs cleared. View reset to default.")
        except Exception as exc:
            st.error(f"Failed to clear outputs: {exc}")


if __name__ == "__main__":
    main()

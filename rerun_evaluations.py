#!/usr/bin/env python3
"""Re-run evaluations for both use cases to update stored results with sparsity metrics."""

import os
from hexeval import evaluate

# Set your OpenAI API key if needed for personas
# os.environ["OPENAI_API_KEY"] = "your-key-here"

def main():
    print("=" * 60)
    print("Re-running HEXEval Evaluations")
    print("=" * 60)
    
    # Use Case 1: Healthcare (Heart Disease)
    print("\n[1/2] Running Healthcare Use Case...")
    print("-" * 60)
    
    try:
        healthcare_results = evaluate(
            model_path="usecases/heart_disease_pipeline.pkl",
            data_path="usecases/heart.csv",
            target_column="target",
            config_path="hexeval/config/eval_config.yaml",
            output_dir="outputs/heart_disease",
        )
        print("✅ Healthcare evaluation complete!")
        print(f"   Results saved to: outputs/heart_disease/")
    except Exception as e:
        print(f"❌ Healthcare evaluation failed: {e}")
    
    # Use Case 2: Credit Risk (Finance)
    print("\n[2/2] Running Credit Risk Use Case...")
    print("-" * 60)
    
    try:
        credit_results = evaluate(
            model_path="usecases/xgboost_credit_risk_new.pkl",
            data_path="usecases/credit_risk_dataset.csv",
            target_column="loan_status",
            config_path="hexeval/config/eval_config_credit_risk.yaml",
            output_dir="outputs/credit_risk",
        )
        print("✅ Credit Risk evaluation complete!")
        print(f"   Results saved to: outputs/credit_risk/")
    except Exception as e:
        print(f"❌ Credit Risk evaluation failed: {e}")
    
    print("\n" + "=" * 60)
    print("All evaluations complete!")
    print("=" * 60)
    print("\nYou can now:")
    print("1. View results in the Streamlit UI")
    print("2. Check the CSV files in outputs/heart_disease/ and outputs/credit_risk/")

if __name__ == "__main__":
    main()

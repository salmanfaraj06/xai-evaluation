"""
HEXEval CLI - Command-line interface for holistic XAI evaluation.

Usage:
    hexeval evaluate MODEL_PATH DATA_PATH --target TARGET_COLUMN
    hexeval validate MODEL_PATH DATA_PATH
"""

import argparse
import logging
import sys

from hexeval import evaluate
from hexeval.core import load_model, load_data, validate_model_data_compatibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    parser = argparse.ArgumentParser(
        description="HEXEval - Holistic Explanation Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run full evaluation")
    eval_parser.add_argument("model", help="Path to model file (.pkl or .joblib)")
    eval_parser.add_argument("data", help="Path to CSV dataset")
    eval_parser.add_argument("--target", help="Target column name", required=True)
    eval_parser.add_argument("--config", help="Path to config YAML", default=None)
    eval_parser.add_argument("--output", help="Output directory", default=None)
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate model and data compatibility")
    val_parser.add_argument("model", help="Path to model file")
    val_parser.add_argument("data", help="Path to CSV dataset")
    val_parser.add_argument("--target", help="Target column name", default=None)
    
    args = parser.parse_args()
    
    if args.command == "evaluate":
        print("=" * 60)
        print("HEXEval - Starting Evaluation")
        print("=" * 60)
        
        results = evaluate(
            model_path=args.model,
            data_path=args.data,
            target_column=args.target,
            config_path=args.config,
            output_dir=args.output,
        )
        
        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)
        print(f"Results saved to: {results['output_path']}")
        
        if results['recommendations']:
            print("\n📊 Recommendations:")
            for stakeholder, rec in results['recommendations'].items():
                print(f"  {stakeholder}: {rec['recommended_method']}")
        
    elif args.command == "validate":
        print("Validating model and data compatibility...")
        
        model_artifact = load_model(args.model)
        data = load_data(args.data, target_column=args.target)
        
        result = validate_model_data_compatibility(model_artifact, data)
        
        if result['status'] == 'valid':
            print("✓ Validation passed!")
        else:
            print("✗ Validation failed:")
            for error in result['errors']:
                print(f"  - {error}")
            sys.exit(1)
        
        if result['warnings']:
            print("Warnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

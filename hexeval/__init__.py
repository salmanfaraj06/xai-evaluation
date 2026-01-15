"""HEXEval - Holistic Explanation Evaluation Framework

A production-grade framework for evaluating XAI methods on tabular models
using technical metrics and LLM-simulated stakeholder feedback.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from hexeval.core.model_loader import load_model
from hexeval.core.data_handler import load_data
from hexeval.evaluation.evaluator import evaluate

__all__ = ["load_model", "load_data", "evaluate"]

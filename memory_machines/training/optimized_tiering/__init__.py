"""
DSPy GEPA optimizer for grounded memory prompt classification.

This module provides tools for optimizing the instruction prompt used in
grounded memory prompt tier classification using DSPy's GEPA (Generalized
Prompt Evolution Algorithm).
"""

from memory_machines.training.optimized_tiering.train import train_dspy_optimizer

__all__ = ["train_dspy_optimizer"]

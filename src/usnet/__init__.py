"""
USNET - Unsupervised Structural Node Embedding Toolkit

A Python library for evaluating graph embeddings using structural node features
and unsupervised optimization-based methods.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Ash Dehghan"

# Graph generators
from usnet.graphs import (
    GraphGenerator,
    generate_barbell,
    generate_dense_star,
    generate_star,
    generate_web,
)

# Feature extraction
from usnet.features import FeatureExtractor, extract_features

# Evaluation
from usnet.evaluation import (
    BenchmarkRunner,
    EvaluationFramework,
    distance_correlation,
    evaluate_embeddings,
    neighbor_consistency,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Graph generators
    "GraphGenerator",
    "generate_barbell",
    "generate_star",
    "generate_web",
    "generate_dense_star",
    # Feature extraction
    "FeatureExtractor",
    "extract_features",
    # Evaluation
    "BenchmarkRunner",
    "EvaluationFramework",
    "distance_correlation",
    "neighbor_consistency",
    "evaluate_embeddings",
]

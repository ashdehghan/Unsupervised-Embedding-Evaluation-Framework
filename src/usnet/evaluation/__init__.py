"""Evaluation framework for unsupervised embedding assessment."""

from __future__ import annotations

from usnet.evaluation.benchmarks import (
    BenchmarkRunner,
    distance_correlation,
    neighbor_consistency,
)
from usnet.evaluation.framework import (
    EvaluationFramework,
    EvaluationResult,
    evaluate_embeddings,
)

__all__ = [
    # Benchmark functions
    "distance_correlation",
    "neighbor_consistency",
    "BenchmarkRunner",
    # Evaluation framework
    "EvaluationFramework",
    "EvaluationResult",
    "evaluate_embeddings",
]

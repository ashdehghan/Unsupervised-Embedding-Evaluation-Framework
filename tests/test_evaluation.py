"""Tests for evaluation module."""

from __future__ import annotations

import numpy as np
import pytest

from usnet.evaluation import (
    BenchmarkRunner,
    EvaluationFramework,
    EvaluationResult,
    distance_correlation,
    evaluate_embeddings,
    neighbor_consistency,
)
from usnet.evaluation.benchmarks import (
    DistanceCorrelationResult,
    NeighborConsistencyResult,
    NormalizationType,
    normalize_data,
)


class TestDistanceCorrelation:
    """Tests for distance_correlation function."""

    def test_returns_result_object(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test that function returns a DistanceCorrelationResult."""
        result = distance_correlation(sample_embeddings, sample_features)

        assert isinstance(result, DistanceCorrelationResult)
        assert hasattr(result, "mean")
        assert hasattr(result, "std")
        assert hasattr(result, "median")
        assert hasattr(result, "per_node")

    def test_correlation_in_valid_range(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test that correlations are in [-1, 1]."""
        result = distance_correlation(sample_embeddings, sample_features)

        assert -1 <= result.mean <= 1
        assert all(-1 <= v <= 1 for v in result.per_node)

    def test_cosine_metric(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test cosine similarity metric."""
        result = distance_correlation(
            sample_embeddings, sample_features, metric="cosine"
        )

        assert isinstance(result.mean, float)

    def test_euclidean_metric(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test euclidean distance metric."""
        result = distance_correlation(
            sample_embeddings, sample_features, metric="euclidean"
        )

        assert isinstance(result.mean, float)


class TestNeighborConsistency:
    """Tests for neighbor_consistency function."""

    def test_returns_result_object(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test that function returns a NeighborConsistencyResult."""
        result = neighbor_consistency(sample_embeddings, sample_features)

        assert isinstance(result, NeighborConsistencyResult)
        assert isinstance(result.feature_to_embedding, DistanceCorrelationResult)
        assert isinstance(result.embedding_to_feature, DistanceCorrelationResult)

    def test_consistency_in_valid_range(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test that consistency values are in [0, 1]."""
        result = neighbor_consistency(sample_embeddings, sample_features)

        assert 0 <= result.feature_to_embedding.mean <= 1
        assert 0 <= result.embedding_to_feature.mean <= 1

    def test_custom_clusters(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test with custom number of clusters."""
        result = neighbor_consistency(
            sample_embeddings, sample_features, n_clusters=3
        )

        assert isinstance(result.feature_to_embedding.mean, float)


class TestNormalizeData:
    """Tests for normalize_data function."""

    def test_standard_normalization(self) -> None:
        """Test standard (z-score) normalization."""
        data = np.random.rand(100, 10) * 100
        normalized = normalize_data(data, NormalizationType.STANDARD)

        # Should have mean ~0 and std ~1
        assert np.abs(normalized.mean()) < 0.1
        assert np.abs(normalized.std() - 1) < 0.1

    def test_minmax_normalization(self) -> None:
        """Test min-max normalization."""
        data = np.random.rand(100, 10) * 100
        normalized = normalize_data(data, NormalizationType.MINMAX)

        assert normalized.min() >= 0
        assert normalized.max() <= 1 + 1e-9  # Allow for floating point precision


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_run_returns_dict(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test that run returns a dictionary of results."""
        runner = BenchmarkRunner()
        results = runner.run(sample_embeddings, sample_features)

        assert isinstance(results, dict)
        assert len(results) > 0

    def test_run_detailed_returns_dataframe(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test that run_detailed returns a DataFrame."""
        import pandas as pd

        runner = BenchmarkRunner()
        results = runner.run_detailed(sample_embeddings, sample_features)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(sample_embeddings)


class TestEvaluationFramework:
    """Tests for EvaluationFramework class."""

    def test_evaluate_returns_result(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test that evaluate returns an EvaluationResult."""
        framework = EvaluationFramework(
            n_ensembles=2,  # Reduce for speed
            random_state=42,
        )
        result = framework.evaluate(sample_embeddings, sample_features)

        assert isinstance(result, EvaluationResult)

    def test_result_has_weights(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test that result contains weights for each dimension."""
        framework = EvaluationFramework(n_ensembles=2, random_state=42)
        result = framework.evaluate(sample_embeddings, sample_features)

        assert len(result.weights) == sample_embeddings.shape[1]
        assert np.abs(result.weights.sum() - 1.0) < 0.01  # Normalized

    def test_scores_are_valid(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test that scores are in valid range [0, 1]."""
        framework = EvaluationFramework(n_ensembles=2, random_state=42)
        result = framework.evaluate(sample_embeddings, sample_features)

        assert 0 <= result.pre_optimization_score <= 1
        assert 0 <= result.post_optimization_score <= 1


class TestEvaluateEmbeddings:
    """Tests for evaluate_embeddings convenience function."""

    def test_convenience_function(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test the convenience function works."""
        result = evaluate_embeddings(
            sample_embeddings,
            sample_features,
            n_ensembles=2,
            random_state=42,
        )

        assert isinstance(result, EvaluationResult)

    def test_scipy_optimizer(
        self,
        sample_embeddings: np.ndarray,
        sample_features: np.ndarray,
    ) -> None:
        """Test using scipy optimizer instead of PyTorch."""
        result = evaluate_embeddings(
            sample_embeddings,
            sample_features,
            n_ensembles=1,
            use_pytorch=False,
            random_state=42,
        )

        assert isinstance(result, EvaluationResult)


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_repr(self) -> None:
        """Test string representation."""
        result = EvaluationResult(
            pre_optimization_score=0.5,
            post_optimization_score=0.3,
            weights=np.array([0.5, 0.5]),
            improvement=0.4,
        )

        repr_str = repr(result)
        assert "0.5" in repr_str
        assert "0.3" in repr_str
        assert "40" in repr_str  # 40%

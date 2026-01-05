"""
Benchmarking methods for embedding evaluation.

This module provides methods to evaluate embeddings by comparing them to
structural node features using distance correlations and cluster-based metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class NormalizationType(str, Enum):
    """Normalization types for data preprocessing."""

    STANDARD = "standard"
    MINMAX = "minmax"
    MINMAX_CLIPPED = "minmax_clipped"


@dataclass
class DistanceCorrelationResult:
    """Result of distance correlation computation.

    Attributes:
        mean: Mean correlation across all nodes.
        std: Standard deviation of correlations.
        median: Median correlation.
        per_node: Array of per-node correlation values.
    """

    mean: float
    std: float
    median: float
    per_node: npt.NDArray[np.floating]


@dataclass
class NeighborConsistencyResult:
    """Result of neighbor consistency computation.

    Attributes:
        feature_to_embedding: Metrics for feature neighbors in embedding space.
        embedding_to_feature: Metrics for embedding neighbors in feature space.
    """

    feature_to_embedding: DistanceCorrelationResult
    embedding_to_feature: DistanceCorrelationResult


def distance_correlation(
    embeddings: npt.NDArray[np.floating],
    features: npt.NDArray[np.floating],
    metric: Literal["cosine", "euclidean"] = "cosine",
) -> DistanceCorrelationResult:
    """Compute correlation between embedding and feature distance matrices.

    For each node, this computes the Pearson correlation between its distances
    to all other nodes in embedding space vs. feature space.

    Args:
        embeddings: Node embeddings of shape (n_nodes, embedding_dim).
        features: Node features of shape (n_nodes, n_features).
        metric: Distance metric to use ("cosine" or "euclidean").

    Returns:
        DistanceCorrelationResult with correlation statistics.

    Example:
        >>> embeddings = np.random.rand(100, 32)
        >>> features = np.random.rand(100, 12)
        >>> result = distance_correlation(embeddings, features)
        >>> print(f"Mean correlation: {result.mean:.3f}")
    """
    # Compute distance matrices
    if metric == "cosine":
        dist_func = cosine_similarity
    else:
        dist_func = euclidean_distances

    emb_distances = dist_func(embeddings, embeddings)
    feat_distances = dist_func(features, features)

    # Compute per-node correlations
    n_nodes = len(embeddings)
    correlations = np.zeros(n_nodes)

    for i in range(n_nodes):
        corr, _ = pearsonr(emb_distances[i], feat_distances[i])
        correlations[i] = corr if np.isfinite(corr) else 0.0

    return DistanceCorrelationResult(
        mean=float(np.mean(correlations)),
        std=float(np.std(correlations)),
        median=float(np.median(correlations)),
        per_node=correlations,
    )


def neighbor_consistency(
    embeddings: npt.NDArray[np.floating],
    features: npt.NDArray[np.floating],
    n_clusters: int = 5,
    random_state: int = 42,
) -> NeighborConsistencyResult:
    """Compute neighbor consistency between embeddings and features using K-means.

    This evaluates whether nodes that are neighbors in feature space (same cluster)
    are also neighbors in embedding space, and vice versa.

    Args:
        embeddings: Node embeddings of shape (n_nodes, embedding_dim).
        features: Node features of shape (n_nodes, n_features).
        n_clusters: Number of clusters for K-means.
        random_state: Random seed for K-means.

    Returns:
        NeighborConsistencyResult with bidirectional consistency metrics.

    Example:
        >>> embeddings = np.random.rand(100, 32)
        >>> features = np.random.rand(100, 12)
        >>> result = neighbor_consistency(embeddings, features, n_clusters=5)
        >>> print(f"Feature->Embedding: {result.feature_to_embedding.mean:.3f}")
    """
    # Cluster both spaces
    feat_labels = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10
    ).fit_predict(features)

    emb_labels = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10
    ).fit_predict(embeddings)

    n_nodes = len(embeddings)
    feat_to_emb = np.zeros(n_nodes)
    emb_to_feat = np.zeros(n_nodes)

    for i in range(n_nodes):
        # Find neighbors in each space
        feat_neighbors = np.where(feat_labels == feat_labels[i])[0]
        emb_neighbors = np.where(emb_labels == emb_labels[i])[0]

        # Compute overlap percentages
        feat_in_emb = np.isin(feat_neighbors, emb_neighbors)
        emb_in_feat = np.isin(emb_neighbors, feat_neighbors)

        feat_to_emb[i] = np.mean(feat_in_emb) if len(feat_in_emb) > 0 else 0.0
        emb_to_feat[i] = np.mean(emb_in_feat) if len(emb_in_feat) > 0 else 0.0

    return NeighborConsistencyResult(
        feature_to_embedding=DistanceCorrelationResult(
            mean=float(np.mean(feat_to_emb)),
            std=float(np.std(feat_to_emb)),
            median=float(np.median(feat_to_emb)),
            per_node=feat_to_emb,
        ),
        embedding_to_feature=DistanceCorrelationResult(
            mean=float(np.mean(emb_to_feat)),
            std=float(np.std(emb_to_feat)),
            median=float(np.median(emb_to_feat)),
            per_node=emb_to_feat,
        ),
    )


def normalize_data(
    data: npt.NDArray[np.floating],
    method: NormalizationType | str = NormalizationType.STANDARD,
) -> npt.NDArray[np.floating]:
    """Normalize data using specified method.

    Args:
        data: Data array of shape (n_samples, n_features).
        method: Normalization method to use.

    Returns:
        Normalized data array.
    """
    if isinstance(method, str):
        method = NormalizationType(method)

    scalers = {
        NormalizationType.STANDARD: StandardScaler(),
        NormalizationType.MINMAX: MinMaxScaler(),
        NormalizationType.MINMAX_CLIPPED: MinMaxScaler(clip=True),
    }

    scaler = scalers[method]
    return scaler.fit_transform(data)


@dataclass
class BenchmarkRunner:
    """Run comprehensive benchmarks on embeddings.

    This class orchestrates multiple evaluation methods and produces
    aggregated results.

    Args:
        normalization_types: List of normalization methods to apply.
        n_clusters_list: List of cluster counts for neighbor consistency.
        distance_metrics: Distance metrics to use for correlation.

    Example:
        >>> runner = BenchmarkRunner()
        >>> results = runner.run(embeddings, features)
    """

    normalization_types: list[NormalizationType] = field(
        default_factory=lambda: [NormalizationType.STANDARD, NormalizationType.MINMAX_CLIPPED]
    )
    n_clusters_list: list[int] = field(default_factory=lambda: [2, 3, 5])
    distance_metrics: list[Literal["cosine", "euclidean"]] = field(
        default_factory=lambda: ["cosine", "euclidean"]
    )

    def run(
        self,
        embeddings: npt.NDArray[np.floating],
        features: npt.NDArray[np.floating],
    ) -> dict[str, dict[str, float]]:
        """Run all benchmarks on the provided embeddings and features.

        Args:
            embeddings: Node embeddings of shape (n_nodes, embedding_dim).
            features: Node features of shape (n_nodes, n_features).

        Returns:
            Dictionary mapping metric names to result dictionaries.
        """
        results: dict[str, dict[str, float]] = {}

        for norm_type in self.normalization_types:
            # Normalize data
            norm_emb = normalize_data(embeddings, norm_type)
            norm_feat = normalize_data(features, norm_type)

            # Distance correlation metrics
            for metric in self.distance_metrics:
                key = f"{metric}_correlation_{norm_type.value}"
                dc_result = distance_correlation(norm_emb, norm_feat, metric=metric)
                results[key] = {
                    "mean": dc_result.mean,
                    "std": dc_result.std,
                    "median": dc_result.median,
                }

            # Neighbor consistency metrics
            for n_clusters in self.n_clusters_list:
                key = f"neighbor_consistency_k{n_clusters}_{norm_type.value}"
                nc_result = neighbor_consistency(
                    norm_emb, norm_feat, n_clusters=n_clusters
                )
                results[key] = {
                    "feat_to_emb_mean": nc_result.feature_to_embedding.mean,
                    "feat_to_emb_std": nc_result.feature_to_embedding.std,
                    "emb_to_feat_mean": nc_result.embedding_to_feature.mean,
                    "emb_to_feat_std": nc_result.embedding_to_feature.std,
                }

        return results

    def run_detailed(
        self,
        embeddings: npt.NDArray[np.floating],
        features: npt.NDArray[np.floating],
    ) -> pd.DataFrame:
        """Run benchmarks and return detailed per-node results.

        Args:
            embeddings: Node embeddings of shape (n_nodes, embedding_dim).
            features: Node features of shape (n_nodes, n_features).

        Returns:
            DataFrame with per-node metrics.
        """
        n_nodes = len(embeddings)
        detailed: dict[str, npt.NDArray[np.floating]] = {
            "node": np.arange(n_nodes),
        }

        for norm_type in self.normalization_types:
            norm_emb = normalize_data(embeddings, norm_type)
            norm_feat = normalize_data(features, norm_type)

            for metric in self.distance_metrics:
                key = f"{metric}_corr_{norm_type.value}"
                dc_result = distance_correlation(norm_emb, norm_feat, metric=metric)
                detailed[key] = dc_result.per_node

            for n_clusters in self.n_clusters_list:
                nc_result = neighbor_consistency(
                    norm_emb, norm_feat, n_clusters=n_clusters
                )
                detailed[f"feat_to_emb_k{n_clusters}_{norm_type.value}"] = (
                    nc_result.feature_to_embedding.per_node
                )
                detailed[f"emb_to_feat_k{n_clusters}_{norm_type.value}"] = (
                    nc_result.embedding_to_feature.per_node
                )

        return pd.DataFrame(detailed).set_index("node")

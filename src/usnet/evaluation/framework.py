"""
Unsupervised Evaluation Framework.

This module provides an optimization-based framework for evaluating embeddings
by learning optimal dimension weights that maximize alignment with structural
node features.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class EvaluationResult:
    """Result from the evaluation framework.

    Attributes:
        pre_optimization_score: Cost before weight optimization (lower is better).
        post_optimization_score: Cost after weight optimization (lower is better).
        weights: Learned weights for each embedding dimension.
        improvement: Relative improvement from optimization.
    """

    pre_optimization_score: float
    post_optimization_score: float
    weights: npt.NDArray[np.floating]
    improvement: float

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(pre={self.pre_optimization_score:.4f}, "
            f"post={self.post_optimization_score:.4f}, "
            f"improvement={self.improvement:.2%})"
        )


def evaluate_embeddings(
    embeddings: npt.NDArray[np.floating] | pd.DataFrame,
    features: npt.NDArray[np.floating] | pd.DataFrame,
    n_clusters: int = 5,
    sampling_fraction: float = 0.5,
    sample_size: int | str = "default",
    n_ensembles: int = 5,
    use_pytorch: bool = True,
    random_state: int | None = 42,
) -> EvaluationResult:
    """Evaluate embeddings using the unsupervised optimization framework.

    This is a convenience function that creates an EvaluationFramework and
    runs the evaluation.

    Args:
        embeddings: Node embeddings of shape (n_nodes, embedding_dim).
        features: Node features of shape (n_nodes, n_features).
        n_clusters: Number of clusters for K-means pseudo-labeling.
        sampling_fraction: Probability of sampling within-cluster pairs.
        sample_size: Number of pairs to sample ("default" for auto).
        n_ensembles: Number of ensemble runs for robustness.
        use_pytorch: Whether to use PyTorch for optimization.
        random_state: Random seed for reproducibility.

    Returns:
        EvaluationResult with scores and learned weights.

    Example:
        >>> embeddings = np.random.rand(100, 32)
        >>> features = np.random.rand(100, 12)
        >>> result = evaluate_embeddings(embeddings, features)
        >>> print(f"Improvement: {result.improvement:.2%}")
    """
    framework = EvaluationFramework(
        n_clusters=n_clusters,
        sampling_fraction=sampling_fraction,
        sample_size=sample_size,
        n_ensembles=n_ensembles,
        use_pytorch=use_pytorch,
        random_state=random_state,
    )
    return framework.evaluate(embeddings, features)


@dataclass
class EvaluationFramework:
    """Optimization-based framework for unsupervised embedding evaluation.

    This framework evaluates embeddings by:
    1. Clustering nodes using K-means on structural features
    2. Sampling pairs of nodes within and between clusters
    3. Optimizing embedding dimension weights to maximize correlation
       between feature distances and weighted embedding distances

    Args:
        n_clusters: Number of clusters for K-means.
        sampling_fraction: Probability of sampling within-cluster pairs (p).
        sample_size: Number of pairs to sample, or "default" for auto.
        n_ensembles: Number of ensemble runs for robust estimates.
        use_pytorch: Use PyTorch for gradient-based optimization.
        random_state: Random seed for reproducibility.

    Example:
        >>> framework = EvaluationFramework(n_clusters=5)
        >>> result = framework.evaluate(embeddings, features)
    """

    n_clusters: int = 5
    sampling_fraction: float = 0.5
    sample_size: int | str = "default"
    n_ensembles: int = 5
    use_pytorch: bool = True
    random_state: int | None = 42

    # Internal state
    _embeddings: npt.NDArray[np.floating] = field(
        default_factory=lambda: np.array([]), init=False, repr=False
    )
    _features: npt.NDArray[np.floating] = field(
        default_factory=lambda: np.array([]), init=False, repr=False
    )
    _cluster_labels: npt.NDArray[np.integer] = field(
        default_factory=lambda: np.array([]), init=False, repr=False
    )
    _within_pairs: list[tuple[int, int]] = field(default_factory=list, init=False, repr=False)
    _between_pairs: list[tuple[int, int]] = field(default_factory=list, init=False, repr=False)

    def evaluate(
        self,
        embeddings: npt.NDArray[np.floating] | pd.DataFrame,
        features: npt.NDArray[np.floating] | pd.DataFrame,
    ) -> EvaluationResult:
        """Run the evaluation framework on embeddings and features.

        Args:
            embeddings: Node embeddings of shape (n_nodes, embedding_dim).
            features: Node features of shape (n_nodes, n_features).

        Returns:
            EvaluationResult with optimization results.
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        # Convert to numpy if DataFrame
        if isinstance(embeddings, pd.DataFrame):
            emb_cols = [c for c in embeddings.columns if "emb" in str(c).lower()]
            if emb_cols:
                embeddings = embeddings[emb_cols].values
            else:
                embeddings = embeddings.select_dtypes(include=[np.number]).values

        if isinstance(features, pd.DataFrame):
            feat_cols = [c for c in features.columns if "nf" in str(c).lower()]
            if feat_cols:
                features = features[feat_cols].values
            else:
                features = features.select_dtypes(include=[np.number]).values

        # Normalize data
        self._embeddings = StandardScaler().fit_transform(embeddings)
        self._features = StandardScaler().fit_transform(features)

        # Cluster based on features
        self._cluster_labels = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state or 42,
            n_init=10,
        ).fit_predict(self._features)

        # Run ensemble evaluations
        ensemble_results: list[dict[str, float | npt.NDArray[np.floating]]] = []

        for _ in range(self.n_ensembles):
            # Sample pairs
            self._sample_pairs()

            # Compute feature distances
            nf_distances = self._compute_feature_distances()

            # Optimize weights
            result = self._optimize_weights(nf_distances)
            ensemble_results.append(result)

        # Aggregate ensemble results
        pre_scores = [r["pre_score"] for r in ensemble_results]
        post_scores = [r["post_score"] for r in ensemble_results]
        all_weights = np.array([r["weights"] for r in ensemble_results])

        mean_pre = float(np.mean(pre_scores))
        mean_post = float(np.mean(post_scores))
        mean_weights = np.mean(all_weights, axis=0)

        # Normalize weights to sum to 1
        mean_weights = mean_weights / (mean_weights.sum() + 1e-10)

        improvement = (mean_pre - mean_post) / (mean_pre + 1e-10)

        return EvaluationResult(
            pre_optimization_score=mean_pre,
            post_optimization_score=mean_post,
            weights=mean_weights,
            improvement=improvement,
        )

    def _sample_pairs(self) -> None:
        """Sample within-cluster and between-cluster node pairs."""
        n_nodes = len(self._features)
        clusters = np.unique(self._cluster_labels)

        # Determine sample size
        if self.sample_size == "default":
            actual_sample_size = min(100_000, (n_nodes ** 2) // len(clusters))
        else:
            actual_sample_size = int(self.sample_size)

        # Build cluster -> nodes mapping
        cluster_nodes: dict[int, list[int]] = {}
        for cluster in clusters:
            cluster_nodes[int(cluster)] = np.where(
                self._cluster_labels == cluster
            )[0].tolist()

        # Sample pairs
        self._within_pairs = []
        self._between_pairs = []
        selected_pairs: set[tuple[int, int]] = set()

        for _ in range(actual_sample_size):
            if random.random() <= self.sampling_fraction:
                # Sample within cluster
                cluster = random.choice(list(cluster_nodes.keys()))
                nodes = cluster_nodes[cluster]
                if len(nodes) >= 2:
                    pair = tuple(random.sample(nodes, 2))
                    if pair not in selected_pairs and (pair[1], pair[0]) not in selected_pairs:
                        self._within_pairs.append(pair)  # type: ignore
                        selected_pairs.add(pair)  # type: ignore
            else:
                # Sample between clusters
                if len(clusters) >= 2:
                    c1, c2 = random.sample(list(cluster_nodes.keys()), 2)
                    if cluster_nodes[c1] and cluster_nodes[c2]:
                        n1 = random.choice(cluster_nodes[c1])
                        n2 = random.choice(cluster_nodes[c2])
                        pair = (n1, n2)
                        if pair not in selected_pairs and (n2, n1) not in selected_pairs:
                            self._between_pairs.append(pair)
                            selected_pairs.add(pair)

    def _compute_feature_distances(self) -> npt.NDArray[np.floating]:
        """Compute feature distances for sampled pairs."""
        distances = []

        for n1, n2 in self._within_pairs + self._between_pairs:
            diff = self._features[n1] - self._features[n2]
            dist = math.sqrt(np.sum(diff ** 2))
            distances.append(dist)

        return np.array(distances)

    def _compute_embedding_distances(
        self,
        weights: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute weighted embedding distances for sampled pairs."""
        if weights is None:
            weights = np.ones(self._embeddings.shape[1])

        distances = []

        for n1, n2 in self._within_pairs + self._between_pairs:
            diff = self._embeddings[n1] - self._embeddings[n2]
            weighted_diff = weights * (diff ** 2)
            dist = math.sqrt(np.sum(weighted_diff))
            distances.append(dist)

        return np.array(distances)

    def _cost_function(
        self,
        weights: npt.NDArray[np.floating],
        nf_distances: npt.NDArray[np.floating],
    ) -> float:
        """Compute cost as 1 - r^2 where r is Pearson correlation."""
        emb_distances = self._compute_embedding_distances(weights)
        if len(emb_distances) < 3:
            return 1.0

        corr, _ = pearsonr(nf_distances, emb_distances)
        if not np.isfinite(corr):
            return 1.0

        return 1.0 - corr ** 2

    def _optimize_weights(
        self,
        nf_distances: npt.NDArray[np.floating],
    ) -> dict[str, float | npt.NDArray[np.floating]]:
        """Optimize embedding dimension weights."""
        n_dims = self._embeddings.shape[1]

        # Pre-optimization score
        pre_score = self._cost_function(np.ones(n_dims), nf_distances)

        if self.use_pytorch:
            weights = self._optimize_pytorch(nf_distances)
        else:
            weights = self._optimize_scipy(nf_distances)

        # Post-optimization score
        post_score = self._cost_function(weights, nf_distances)

        # Normalize weights
        weights = weights / (weights.sum() + 1e-10)

        return {
            "pre_score": pre_score,
            "post_score": post_score,
            "weights": weights,
        }

    def _optimize_pytorch(
        self,
        nf_distances: npt.NDArray[np.floating],
        lr: float = 0.01,
        epochs: int = 200,
    ) -> npt.NDArray[np.floating]:
        """Optimize weights using PyTorch gradient descent."""
        n_dims = self._embeddings.shape[1]

        # Convert to tensors
        emb_tensor = torch.tensor(self._embeddings, dtype=torch.float32)
        nf_dist_tensor = torch.tensor(nf_distances, dtype=torch.float32)

        # Precompute pair indices
        all_pairs = self._within_pairs + self._between_pairs
        pair_indices = torch.tensor(all_pairs, dtype=torch.long)

        # Initialize weights
        weights = torch.nn.Parameter(torch.rand(n_dims))

        optimizer = torch.optim.Adam([weights], lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()

            # Compute weighted distances
            n1_emb = emb_tensor[pair_indices[:, 0]]
            n2_emb = emb_tensor[pair_indices[:, 1]]

            diff = n1_emb - n2_emb
            # Ensure weights are positive
            pos_weights = torch.clamp(weights, min=0.0)
            weighted_sq_diff = pos_weights * (diff ** 2)
            emb_dist = torch.sqrt(weighted_sq_diff.sum(dim=1) + 1e-10)

            # Pearson correlation as loss
            x = nf_dist_tensor
            y = emb_dist

            x_mean = x.mean()
            y_mean = y.mean()

            x_centered = x - x_mean
            y_centered = y - y_mean

            cov = (x_centered * y_centered).sum()
            std_x = torch.sqrt((x_centered ** 2).sum() + 1e-10)
            std_y = torch.sqrt((y_centered ** 2).sum() + 1e-10)

            corr = cov / (std_x * std_y + 1e-10)

            # Loss: 1 - r^2
            loss = 1.0 - corr ** 2

            loss.backward()
            optimizer.step()

            # Clamp weights to [0, 1]
            with torch.no_grad():
                weights.clamp_(0.0, 1.0)

        return weights.detach().numpy()

    def _optimize_scipy(
        self,
        nf_distances: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Optimize weights using scipy L-BFGS-B."""
        from scipy.optimize import minimize

        n_dims = self._embeddings.shape[1]

        # Initialize weights
        w_initial = np.random.rand(n_dims)
        w_initial = w_initial / w_initial.sum()

        # Bounds
        bounds = [(0.0, 1.0)] * n_dims

        # Optimize
        result = minimize(
            lambda w: self._cost_function(w, nf_distances),
            w_initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200},
        )

        return result.x

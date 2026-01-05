"""
Structural node feature extraction.

This module provides functionality to compute various structural features
for nodes in a graph, including centrality measures, PageRank, and other
graph-theoretic metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd


# List of all available features
AVAILABLE_FEATURES: list[str] = [
    "degree_centrality",
    "closeness_centrality",
    "pagerank",
    "load_centrality",
    "average_neighbor_degree",
    "eigenvector_centrality",
    "harmonic_centrality",
    "betweenness_centrality",
    "hub_score",
    "constraint",
    "coreness",
    "eccentricity",
]


def extract_features(
    graph: nx.Graph,
    features: list[str] | None = None,
    prefix: str = "nf_",
) -> pd.DataFrame:
    """Extract structural features from a graph.

    This is a convenience function that creates a FeatureExtractor and
    extracts features in one call.

    Args:
        graph: A NetworkX graph to extract features from.
        features: List of feature names to extract. If None, extracts all
            available features. See AVAILABLE_FEATURES for options.
        prefix: Prefix to add to feature column names.

    Returns:
        A DataFrame with node IDs and feature columns.

    Example:
        >>> import networkx as nx
        >>> g = nx.karate_club_graph()
        >>> features = extract_features(g)
        >>> features.shape
        (34, 13)  # 34 nodes, 12 features + node column
    """
    extractor = FeatureExtractor(features=features, prefix=prefix)
    return extractor.extract(graph)


@dataclass
class FeatureExtractor:
    """Extracts structural node features from graphs.

    This class computes various centrality and structural metrics for each
    node in a graph. Features are computed using both NetworkX and igraph
    for optimal performance.

    Args:
        features: List of feature names to extract. If None, extracts all.
        prefix: Prefix to add to feature column names (default: "nf_").

    Example:
        >>> extractor = FeatureExtractor(features=["degree_centrality", "pagerank"])
        >>> g = nx.karate_club_graph()
        >>> features = extractor.extract(g)
    """

    features: list[str] | None = None
    prefix: str = "nf_"
    _nx_graph: nx.Graph = field(default_factory=nx.Graph, init=False, repr=False)
    _ig_graph: ig.Graph = field(default_factory=ig.Graph, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate feature names."""
        if self.features is not None:
            invalid = set(self.features) - set(AVAILABLE_FEATURES)
            if invalid:
                raise ValueError(f"Unknown features: {invalid}. Available: {AVAILABLE_FEATURES}")

    def extract(self, graph: nx.Graph) -> pd.DataFrame:
        """Extract features from a graph.

        Args:
            graph: A NetworkX graph to extract features from.

        Returns:
            A DataFrame with 'node' column and feature columns.
        """
        self._nx_graph = graph
        self._ig_graph = ig.Graph.from_networkx(graph)

        features_to_extract = self.features or AVAILABLE_FEATURES

        # Map feature names to extraction methods
        feature_methods: dict[str, Callable[[], pd.DataFrame]] = {
            "degree_centrality": self._degree_centrality,
            "closeness_centrality": self._closeness_centrality,
            "pagerank": self._pagerank,
            "load_centrality": self._load_centrality,
            "average_neighbor_degree": self._average_neighbor_degree,
            "eigenvector_centrality": self._eigenvector_centrality,
            "harmonic_centrality": self._harmonic_centrality,
            "betweenness_centrality": self._betweenness_centrality,
            "hub_score": self._hub_score,
            "constraint": self._constraint,
            "coreness": self._coreness,
            "eccentricity": self._eccentricity,
        }

        # Extract each feature and merge
        result_df: pd.DataFrame | None = None

        for feature_name in features_to_extract:
            if feature_name in feature_methods:
                try:
                    feature_df = feature_methods[feature_name]()
                    if result_df is None:
                        result_df = feature_df
                    else:
                        result_df = result_df.merge(feature_df, on="node", how="inner")
                except Exception:
                    # Skip features that fail (e.g., eigenvector on disconnected graphs)
                    pass

        if result_df is None:
            # Return empty DataFrame with just node column
            result_df = pd.DataFrame({"node": list(self._nx_graph.nodes)})

        # Fill NaN values with 0
        result_df = result_df.fillna(0)

        return result_df

    # =========================================================================
    # NetworkX-based features
    # =========================================================================

    def _degree_centrality(self) -> pd.DataFrame:
        """Compute degree centrality for each node."""
        dc = nx.degree_centrality(self._nx_graph)
        return pd.DataFrame({
            "node": list(dc.keys()),
            f"{self.prefix}degree_centrality": list(dc.values()),
        })

    def _closeness_centrality(self) -> pd.DataFrame:
        """Compute closeness centrality for each node."""
        cc = nx.closeness_centrality(self._nx_graph)
        return pd.DataFrame({
            "node": list(cc.keys()),
            f"{self.prefix}closeness_centrality": list(cc.values()),
        })

    def _pagerank(self) -> pd.DataFrame:
        """Compute PageRank for each node."""
        pr = nx.pagerank(self._nx_graph)
        return pd.DataFrame({
            "node": list(pr.keys()),
            f"{self.prefix}pagerank": list(pr.values()),
        })

    def _load_centrality(self) -> pd.DataFrame:
        """Compute load centrality for each node."""
        lc = nx.load_centrality(self._nx_graph)
        return pd.DataFrame({
            "node": list(lc.keys()),
            f"{self.prefix}load_centrality": list(lc.values()),
        })

    def _average_neighbor_degree(self) -> pd.DataFrame:
        """Compute average neighbor degree for each node."""
        and_dict = nx.average_neighbor_degree(self._nx_graph)
        return pd.DataFrame({
            "node": list(and_dict.keys()),
            f"{self.prefix}average_neighbor_degree": list(and_dict.values()),
        })

    def _eigenvector_centrality(self) -> pd.DataFrame:
        """Compute eigenvector centrality for each node."""
        ec = nx.eigenvector_centrality(self._nx_graph, max_iter=1000)
        return pd.DataFrame({
            "node": list(ec.keys()),
            f"{self.prefix}eigenvector_centrality": list(ec.values()),
        })

    # =========================================================================
    # igraph-based features (faster for some metrics)
    # =========================================================================

    def _get_node_mapping(self) -> dict[int, int | str]:
        """Get mapping from igraph indices to NetworkX node IDs."""
        nx_nodes = list(self._nx_graph.nodes)
        return {i: nx_nodes[i] for i in range(len(nx_nodes))}

    def _harmonic_centrality(self) -> pd.DataFrame:
        """Compute harmonic centrality for each node."""
        node_map = self._get_node_mapping()
        hc = self._ig_graph.harmonic_centrality()
        return pd.DataFrame({
            "node": [node_map[i] for i in range(len(hc))],
            f"{self.prefix}harmonic_centrality": hc,
        })

    def _betweenness_centrality(self) -> pd.DataFrame:
        """Compute betweenness centrality for each node."""
        node_map = self._get_node_mapping()
        bc = self._ig_graph.betweenness()
        return pd.DataFrame({
            "node": [node_map[i] for i in range(len(bc))],
            f"{self.prefix}betweenness_centrality": bc,
        })

    def _hub_score(self) -> pd.DataFrame:
        """Compute hub score for each node."""
        node_map = self._get_node_mapping()
        hs = self._ig_graph.hub_score()
        return pd.DataFrame({
            "node": [node_map[i] for i in range(len(hs))],
            f"{self.prefix}hub_score": hs,
        })

    def _constraint(self) -> pd.DataFrame:
        """Compute Burt's constraint for each node."""
        node_map = self._get_node_mapping()
        con = self._ig_graph.constraint()
        # Replace inf values with NaN (will be filled with 0 later)
        con = [c if np.isfinite(c) else np.nan for c in con]
        return pd.DataFrame({
            "node": [node_map[i] for i in range(len(con))],
            f"{self.prefix}constraint": con,
        })

    def _coreness(self) -> pd.DataFrame:
        """Compute k-core number for each node."""
        node_map = self._get_node_mapping()
        core = self._ig_graph.coreness()
        return pd.DataFrame({
            "node": [node_map[i] for i in range(len(core))],
            f"{self.prefix}coreness": core,
        })

    def _eccentricity(self) -> pd.DataFrame:
        """Compute eccentricity for each node."""
        node_map = self._get_node_mapping()
        ecc = self._ig_graph.eccentricity()
        return pd.DataFrame({
            "node": [node_map[i] for i in range(len(ecc))],
            f"{self.prefix}eccentricity": ecc,
        })

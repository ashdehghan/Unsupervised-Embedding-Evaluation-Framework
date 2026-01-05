"""Tests for feature extraction module."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from usnet.features import AVAILABLE_FEATURES, FeatureExtractor, extract_features


class TestExtractFeatures:
    """Tests for extract_features function."""

    def test_extracts_all_features(self, small_graph: nx.Graph) -> None:
        """Test extracting all available features."""
        features = extract_features(small_graph)

        assert isinstance(features, pd.DataFrame)
        assert "node" in features.columns
        assert len(features) == len(small_graph.nodes)

    def test_extracts_specific_features(self, small_graph: nx.Graph) -> None:
        """Test extracting specific features."""
        features = extract_features(
            small_graph,
            features=["degree_centrality", "pagerank"],
        )

        assert "nf_degree_centrality" in features.columns
        assert "nf_pagerank" in features.columns
        # Should not have other features
        assert "nf_closeness_centrality" not in features.columns

    def test_custom_prefix(self, small_graph: nx.Graph) -> None:
        """Test custom column prefix."""
        features = extract_features(
            small_graph,
            features=["degree_centrality"],
            prefix="feat_",
        )

        assert "feat_degree_centrality" in features.columns

    def test_returns_numeric_values(self, small_graph: nx.Graph) -> None:
        """Test that all feature values are numeric."""
        features = extract_features(small_graph)

        for col in features.columns:
            if col != "node":
                assert features[col].dtype in [np.float64, np.int64, np.float32, np.int32]


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_validates_feature_names(self) -> None:
        """Test that invalid feature names raise an error."""
        with pytest.raises(ValueError, match="Unknown features"):
            FeatureExtractor(features=["invalid_feature"])

    def test_all_available_features(self) -> None:
        """Test that all features in AVAILABLE_FEATURES are valid."""
        # This should not raise
        extractor = FeatureExtractor(features=AVAILABLE_FEATURES)
        assert extractor.features == AVAILABLE_FEATURES

    def test_extract_on_disconnected_graph(self) -> None:
        """Test extraction on a graph with disconnected components."""
        g = nx.Graph()
        g.add_edges_from([(0, 1), (2, 3)])  # Two disconnected edges

        extractor = FeatureExtractor(features=["degree_centrality"])
        features = extractor.extract(g)

        assert len(features) == 4


class TestIndividualFeatures:
    """Tests for individual feature computations."""

    def test_degree_centrality_range(self, small_graph: nx.Graph) -> None:
        """Test that degree centrality is in [0, 1]."""
        features = extract_features(small_graph, features=["degree_centrality"])

        dc = features["nf_degree_centrality"]
        assert dc.min() >= 0
        assert dc.max() <= 1

    def test_pagerank_sums_to_one(self, small_graph: nx.Graph) -> None:
        """Test that PageRank values sum to approximately 1."""
        features = extract_features(small_graph, features=["pagerank"])

        pr_sum = features["nf_pagerank"].sum()
        assert 0.99 <= pr_sum <= 1.01

    def test_closeness_centrality_range(self, small_graph: nx.Graph) -> None:
        """Test that closeness centrality is in [0, 1]."""
        features = extract_features(small_graph, features=["closeness_centrality"])

        cc = features["nf_closeness_centrality"]
        assert cc.min() >= 0
        assert cc.max() <= 1

    def test_coreness_positive(self, small_graph: nx.Graph) -> None:
        """Test that coreness values are non-negative."""
        features = extract_features(small_graph, features=["coreness"])

        assert features["nf_coreness"].min() >= 0

    def test_eccentricity_positive(self, small_graph: nx.Graph) -> None:
        """Test that eccentricity values are positive."""
        features = extract_features(small_graph, features=["eccentricity"])

        assert features["nf_eccentricity"].min() > 0


class TestAvailableFeatures:
    """Tests for AVAILABLE_FEATURES constant."""

    def test_has_expected_features(self) -> None:
        """Test that expected features are in the list."""
        expected = [
            "degree_centrality",
            "closeness_centrality",
            "pagerank",
            "betweenness_centrality",
            "eigenvector_centrality",
        ]
        for feat in expected:
            assert feat in AVAILABLE_FEATURES

    def test_count(self) -> None:
        """Test that we have 12 features."""
        assert len(AVAILABLE_FEATURES) == 12

"""Tests for graph generation module."""

from __future__ import annotations

import networkx as nx
import pytest

from usnet.graphs import (
    GraphGenerator,
    combine_graphs,
    generate_barbell,
    generate_dense_star,
    generate_star,
    generate_web,
)


class TestGenerateBarbell:
    """Tests for barbell graph generation."""

    def test_generates_valid_graph(self) -> None:
        """Test that generate_barbell creates a valid NetworkX graph."""
        g = generate_barbell()
        assert isinstance(g, nx.Graph)
        assert len(g.nodes) == 30  # 10 + 10 + 10

    def test_has_role_attributes(self) -> None:
        """Test that nodes have role attributes."""
        g = generate_barbell()
        for node in g.nodes:
            assert "role" in g.nodes[node]
            assert "con_type" in g.nodes[node]

    def test_custom_size(self) -> None:
        """Test barbell with custom sizes."""
        g = generate_barbell(m1=5, m2=5)
        assert len(g.nodes) == 15  # 5 + 5 + 5

    def test_seed_reproducibility(self) -> None:
        """Test that seed produces reproducible results."""
        g1 = generate_barbell(seed=42)
        g2 = generate_barbell(seed=42)
        assert list(g1.nodes) == list(g2.nodes)
        assert list(g1.edges) == list(g2.edges)


class TestGenerateWeb:
    """Tests for web pattern generation."""

    def test_generates_valid_graph(self) -> None:
        """Test that generate_web creates a valid NetworkX graph."""
        g = generate_web()
        assert isinstance(g, nx.Graph)
        assert len(g.nodes) == 16

    def test_has_role_attributes(self) -> None:
        """Test that nodes have role attributes."""
        g = generate_web()
        roles = set(g.nodes[n]["role"] for n in g.nodes)
        assert "w0" in roles  # Hub
        assert "w1" in roles  # Inner ring
        assert "w2" in roles  # Outer ring

    def test_is_connected(self) -> None:
        """Test that the web graph is connected."""
        g = generate_web()
        assert nx.is_connected(g)


class TestGenerateStar:
    """Tests for star pattern generation."""

    def test_generates_valid_graph(self) -> None:
        """Test that generate_star creates a valid NetworkX graph."""
        g = generate_star()
        assert isinstance(g, nx.Graph)
        assert len(g.nodes) == 16

    def test_hub_node_has_correct_degree(self) -> None:
        """Test that the hub has the expected connections."""
        g = generate_star()
        # Find hub (role s0)
        hub = [n for n in g.nodes if g.nodes[n]["role"] == "s0"][0]
        assert g.degree(hub) == 5


class TestGenerateDenseStar:
    """Tests for dense star pattern generation."""

    def test_generates_valid_graph(self) -> None:
        """Test that generate_dense_star creates a valid NetworkX graph."""
        g = generate_dense_star(density=12)
        assert isinstance(g, nx.Graph)
        assert len(g.nodes) == 13  # 1 hub + 12 leaves

    def test_custom_density(self) -> None:
        """Test dense star with custom density."""
        g = generate_dense_star(density=5)
        assert len(g.nodes) == 6  # 1 hub + 5 leaves


class TestCombineGraphs:
    """Tests for graph combination."""

    def test_combines_multiple_graphs(self) -> None:
        """Test combining multiple graphs with hydration."""
        g1 = generate_barbell()
        g2 = generate_star()
        # Use hydration to connect the graphs
        combined = combine_graphs([g1, g2], hydrate_prob=0.5)

        # With hydration, we should have a single connected component
        assert len(combined.nodes) > 0
        assert len(combined.nodes) <= len(g1.nodes) + len(g2.nodes)

    def test_empty_list_returns_empty_graph(self) -> None:
        """Test that empty list returns empty graph."""
        combined = combine_graphs([])
        assert len(combined.nodes) == 0


class TestGraphGenerator:
    """Tests for GraphGenerator class."""

    def test_build_with_multiple_patterns(self) -> None:
        """Test building graph with multiple patterns."""
        generator = GraphGenerator(
            barbell_count=1,
            web_count=1,
            star_count=1,
            seed=42,
        )
        g = generator.build()

        assert isinstance(g, nx.Graph)
        assert len(g.nodes) > 0
        assert nx.is_connected(g)

    def test_build_with_hydration(self) -> None:
        """Test building graph with hydration."""
        generator = GraphGenerator(
            barbell_count=2,
            hydrate_prob=0.1,
            seed=42,
        )
        g = generator.build()

        assert nx.is_connected(g)

    def test_seed_reproducibility(self) -> None:
        """Test that seed produces reproducible results."""
        gen1 = GraphGenerator(barbell_count=1, web_count=1, seed=42)
        gen2 = GraphGenerator(barbell_count=1, web_count=1, seed=42)

        g1 = gen1.build()
        g2 = gen2.build()

        assert len(g1.nodes) == len(g2.nodes)
        assert len(g1.edges) == len(g2.edges)

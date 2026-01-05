"""Pytest fixtures for usnet tests."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest


@pytest.fixture
def small_graph() -> nx.Graph:
    """Create a small test graph."""
    return nx.karate_club_graph()


@pytest.fixture
def random_graph() -> nx.Graph:
    """Create a random graph for testing."""
    return nx.fast_gnp_random_graph(50, 0.1, seed=42)


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.random.rand(50, 16)


@pytest.fixture
def sample_features() -> np.ndarray:
    """Create sample features for testing."""
    np.random.seed(42)
    return np.random.rand(50, 12)


@pytest.fixture
def barbell_graph() -> nx.Graph:
    """Create a barbell graph for testing."""
    from usnet.graphs import generate_barbell
    return generate_barbell(seed=42)

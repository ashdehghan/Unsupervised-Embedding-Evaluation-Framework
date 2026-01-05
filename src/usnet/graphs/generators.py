"""
Graph generation utilities for creating synthetic graphs with structural patterns.

This module provides functions and classes for generating various graph patterns
commonly used in structural embedding research, including barbell graphs, star
patterns, web patterns, and more.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass
class GraphGenerator:
    """Generator for creating synthetic graphs with various structural patterns.

    This class creates graphs by combining multiple structural patterns (barbell,
    star, web, etc.) and optionally connecting them through hydration.

    Args:
        random_graph_nodes: Number of nodes in the random graph component.
        random_graph_edge_prob: Edge probability for random graph (Erdos-Renyi).
        barbell_count: Number of barbell patterns to include.
        web_count: Number of web patterns to include.
        star_count: Number of star patterns to include.
        dense_star_count: Number of dense star patterns to include.
        dense_star_density: Number of leaves in dense star patterns.
        hydrate_prob: Probability of adding random connections between patterns.
        hydration_type: Type of hydration ("random" or "bridge_connect").
        seed: Random seed for reproducibility.

    Example:
        >>> generator = GraphGenerator(barbell_count=2, web_count=3)
        >>> graph = generator.build()
        >>> len(graph.nodes)
        92
    """

    random_graph_nodes: int = 0
    random_graph_edge_prob: float = 0.1
    barbell_count: int = 0
    web_count: int = 0
    star_count: int = 0
    dense_star_count: int = 0
    dense_star_density: int = 12
    hydrate_prob: float = 0.0
    hydration_type: str = "bridge_connect"
    seed: int | None = None
    _graph: nx.Graph = field(default_factory=nx.Graph, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize random seed if provided."""
        if self.seed is not None:
            random.seed(self.seed)

    def build(self, shuffle_nodes: bool = False) -> nx.Graph:
        """Build the synthetic graph by combining all specified patterns.

        Args:
            shuffle_nodes: Whether to shuffle node labels after building.

        Returns:
            A NetworkX graph with all patterns combined.
        """
        graphs: list[nx.Graph] = []

        # Add random graph if specified
        if self.random_graph_nodes > 0:
            random_graph = _create_random_graph(
                self.random_graph_nodes, self.random_graph_edge_prob
            )
            graphs.append(random_graph)

        # Add barbell patterns
        for _ in range(self.barbell_count):
            graphs.append(_create_barbell_graph())

        # Add web patterns
        for i in range(self.web_count):
            graphs.append(_create_web_pattern_graph(subgraph_label=i))

        # Add star patterns
        for _ in range(self.star_count):
            graphs.append(_create_star_pattern_graph())

        # Add dense star patterns
        for _ in range(self.dense_star_count):
            graphs.append(_create_dense_star_pattern_graph(self.dense_star_density))

        if not graphs:
            return nx.Graph()

        # Relabel and combine graphs
        graphs = _relabel_graphs(graphs)
        combined = _combine_graphs(graphs)

        # Apply hydration if specified
        if self.hydrate_prob > 0:
            if self.hydration_type == "bridge_connect":
                combined = _bridge_connect_hydration(combined)
            else:
                combined = _hydrate_graph(combined, self.hydrate_prob)

        # Get largest connected component
        if len(combined) > 0:
            largest_cc = max(nx.connected_components(combined), key=len)
            combined = combined.subgraph(largest_cc).copy()
            combined = _reset_node_labels(combined, shuffle_nodes)

        self._graph = combined
        return combined

    def save_edge_list(self, path: Path | str, name: str) -> None:
        """Save the graph as an edge list file.

        Args:
            path: Directory path to save the file.
            name: Base name for the file (without extension).
        """
        file_path = Path(path) / f"{name}.txt"
        nx.write_edgelist(self._graph, file_path, data=False)

    def save_gexf(self, path: Path | str, name: str) -> None:
        """Save the graph in GEXF format for Gephi.

        Args:
            path: Directory path to save the file.
            name: Base name for the file (without extension).
        """
        file_path = Path(path) / f"{name}.gexf"
        nx.write_gexf(self._graph, file_path)


def generate_barbell(
    m1: int = 10,
    m2: int = 10,
    seed: int | None = None,
) -> nx.Graph:
    """Generate a barbell graph with role annotations.

    A barbell graph consists of two complete graphs connected by a path.

    Args:
        m1: Size of the complete graphs on each end.
        m2: Length of the path connecting them.
        seed: Random seed for reproducibility.

    Returns:
        A barbell graph with 'role' and 'con_type' node attributes.

    Example:
        >>> g = generate_barbell(m1=10, m2=10)
        >>> len(g.nodes)
        30
    """
    if seed is not None:
        random.seed(seed)

    g = nx.barbell_graph(m1, m2)
    _annotate_barbell_roles(g, m1, m2)
    return g


def generate_web(
    subgraph_label: int | None = None,
    seed: int | None = None,
) -> nx.Graph:
    """Generate a web pattern graph with role annotations.

    A web pattern has a central hub connected to an inner ring, which connects
    to an outer ring.

    Args:
        subgraph_label: Optional label for the subgraph.
        seed: Random seed for reproducibility.

    Returns:
        A web pattern graph with 'role' and 'con_type' node attributes.

    Example:
        >>> g = generate_web()
        >>> len(g.nodes)
        16
    """
    if seed is not None:
        random.seed(seed)
    return _create_web_pattern_graph(subgraph_label)


def generate_star(
    seed: int | None = None,
) -> nx.Graph:
    """Generate a star pattern graph with role annotations.

    A star pattern has a central hub connected to intermediate nodes,
    which connect to leaf nodes.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A star pattern graph with 'role' and 'con_type' node attributes.

    Example:
        >>> g = generate_star()
        >>> len(g.nodes)
        16
    """
    if seed is not None:
        random.seed(seed)
    return _create_star_pattern_graph()


def generate_dense_star(
    density: int = 12,
    seed: int | None = None,
) -> nx.Graph:
    """Generate a dense star pattern graph.

    A dense star has a central hub connected directly to many leaf nodes.

    Args:
        density: Number of leaf nodes connected to the hub.
        seed: Random seed for reproducibility.

    Returns:
        A dense star graph with 'role' and 'con_type' node attributes.

    Example:
        >>> g = generate_dense_star(density=12)
        >>> len(g.nodes)
        13
    """
    if seed is not None:
        random.seed(seed)
    return _create_dense_star_pattern_graph(density)


def combine_graphs(
    graphs: list[nx.Graph],
    hydrate_prob: float = 0.0,
    hydration_type: str = "bridge_connect",
) -> nx.Graph:
    """Combine multiple graphs into a single graph.

    Args:
        graphs: List of NetworkX graphs to combine.
        hydrate_prob: Probability of adding random connections.
        hydration_type: Type of hydration ("random" or "bridge_connect").

    Returns:
        A single combined graph with relabeled nodes.

    Example:
        >>> g1 = generate_barbell()
        >>> g2 = generate_star()
        >>> combined = combine_graphs([g1, g2], hydrate_prob=0.1)
    """
    if not graphs:
        return nx.Graph()

    # Relabel and combine
    relabeled = _relabel_graphs(graphs)
    combined = _combine_graphs(relabeled)

    # Apply hydration
    if hydrate_prob > 0:
        if hydration_type == "bridge_connect":
            combined = _bridge_connect_hydration(combined)
        else:
            combined = _hydrate_graph(combined, hydrate_prob)

    # Get largest connected component
    if len(combined) > 0:
        largest_cc = max(nx.connected_components(combined), key=len)
        combined = combined.subgraph(largest_cc).copy()
        combined = _reset_node_labels(combined, shuffle=False)

    return combined


# =============================================================================
# Private helper functions
# =============================================================================


def _create_random_graph(n: int, p: float) -> nx.Graph:
    """Create a random Erdos-Renyi graph with role annotations."""
    g = nx.fast_gnp_random_graph(n, p)
    attr_role = {i: "r" for i in g.nodes}
    attr_con = {i: "bridge" for i in g.nodes}
    nx.set_node_attributes(g, attr_role, name="role")
    nx.set_node_attributes(g, attr_con, name="con_type")
    return g


def _create_barbell_graph() -> nx.Graph:
    """Create a barbell graph with fixed size and role annotations."""
    g = nx.barbell_graph(10, 10)
    _annotate_barbell_roles(g, 10, 10)
    return g


def _annotate_barbell_roles(g: nx.Graph, m1: int, m2: int) -> None:
    """Annotate barbell graph nodes with structural roles."""
    attr_role: dict[Any, str] = {}
    attr_con: dict[Any, str] = {}

    total = 2 * m1 + m2
    clique1_end = m1 - 1
    path_start = m1
    path_end = m1 + m2 - 1
    clique2_start = m1 + m2

    for i in g.nodes:
        if i <= clique1_end - 1 or i >= clique2_start + 1:
            # Interior of cliques
            attr_role[i] = "b0"
            attr_con[i] = "non_bridge"
        elif i == clique1_end or i == clique2_start:
            # Clique-path connection points
            attr_role[i] = "b1"
            attr_con[i] = "non_bridge"
        elif i == path_start or i == path_end:
            attr_role[i] = "b2"
            attr_con[i] = "non_bridge"
        else:
            # Path interior - assign roles based on distance from ends
            dist_from_start = i - path_start
            dist_from_end = path_end - i
            min_dist = min(dist_from_start, dist_from_end)
            if min_dist <= m2 // 2:
                attr_role[i] = f"b{min_dist + 2}"
            else:
                attr_role[i] = f"b{m2 // 2 + 2}"

            # Middle of path is bridge
            if i == (path_start + path_end) // 2 or i == (path_start + path_end) // 2 + 1:
                attr_con[i] = "bridge"
            else:
                attr_con[i] = "non_bridge"

    nx.set_node_attributes(g, attr_role, name="role")
    nx.set_node_attributes(g, attr_con, name="con_type")


def _create_web_pattern_graph(subgraph_label: int | None = None) -> nx.Graph:
    """Create a web pattern graph with role annotations."""
    g = nx.Graph()

    # Build web structure
    edge_list = [
        # Hub to inner ring
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        # Inner ring connections
        (2, 3), (3, 4), (4, 5), (5, 6), (6, 2),
        # Inner to outer ring
        (2, 7), (2, 8), (3, 9), (3, 10), (4, 11), (4, 12),
        (5, 13), (5, 14), (6, 15), (6, 16),
        # Outer ring connections
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
        (12, 13), (13, 14), (14, 15), (15, 16), (16, 7),
    ]
    g.add_edges_from(edge_list)

    # Assign roles
    attr_role: dict[int, str] = {1: "w0"}
    attr_con: dict[int, str] = {1: "non_bridge"}

    for i in range(2, 7):
        attr_role[i] = "w1"
        attr_con[i] = "non_bridge"

    for i in range(7, 17):
        attr_role[i] = "w2"
        attr_con[i] = "bridge"

    nx.set_node_attributes(g, attr_role, name="role")
    nx.set_node_attributes(g, attr_con, name="con_type")

    if subgraph_label is not None:
        attr_label = {i: f"web_{subgraph_label}" for i in range(1, 17)}
        nx.set_node_attributes(g, attr_label, name="subgraph_label")

    return g


def _create_star_pattern_graph() -> nx.Graph:
    """Create a star pattern graph with role annotations."""
    g = nx.Graph()

    edge_list = [
        # Hub to intermediate
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        # Intermediate to leaves
        (2, 7), (2, 8), (3, 9), (3, 10), (4, 11), (4, 12),
        (5, 13), (5, 14), (6, 15), (6, 16),
    ]
    g.add_edges_from(edge_list)

    # Assign roles
    attr_role: dict[int, str] = {1: "s0"}
    attr_con: dict[int, str] = {1: "non_bridge"}

    for i in range(2, 7):
        attr_role[i] = "s1"
        attr_con[i] = "non_bridge"

    for i in range(7, 17):
        attr_role[i] = "s2"
        attr_con[i] = "bridge"

    nx.set_node_attributes(g, attr_role, name="role")
    nx.set_node_attributes(g, attr_con, name="con_type")

    return g


def _create_dense_star_pattern_graph(density: int = 12) -> nx.Graph:
    """Create a dense star pattern graph with role annotations."""
    g = nx.Graph()

    edge_list = [(1, i) for i in range(2, density + 2)]
    g.add_edges_from(edge_list)

    # Assign roles
    attr_role: dict[int, str] = {1: "ds0"}
    attr_con: dict[int, str] = {1: "non_bridge"}

    for i in range(2, density + 2):
        attr_role[i] = "ds1"
        attr_con[i] = "bridge"

    nx.set_node_attributes(g, attr_role, name="role")
    nx.set_node_attributes(g, attr_con, name="con_type")

    return g


def _relabel_graphs(graphs: list[nx.Graph]) -> list[nx.Graph]:
    """Relabel nodes in graphs to have unique consecutive IDs."""
    if not graphs:
        return []

    new_graph_list: list[nx.Graph] = [graphs[0].copy()]
    last_node_id = max(graphs[0].nodes) if graphs[0].nodes else 0

    for i in range(1, len(graphs)):
        tmp_graph = graphs[i].copy()
        node_map = {}
        for node in tmp_graph.nodes:
            last_node_id += 1
            node_map[node] = last_node_id
        tmp_graph = nx.relabel_nodes(tmp_graph, node_map)
        new_graph_list.append(tmp_graph)
        if tmp_graph.nodes:
            last_node_id = max(tmp_graph.nodes)

    return new_graph_list


def _combine_graphs(graphs: list[nx.Graph]) -> nx.Graph:
    """Combine multiple graphs into a single graph."""
    if not graphs:
        return nx.Graph()
    if len(graphs) == 1:
        return graphs[0].copy()

    combined = graphs[0].copy()
    for i in range(1, len(graphs)):
        combined = nx.compose(combined, graphs[i])
    return combined


def _hydrate_graph(g: nx.Graph, hydrate_prob: float) -> nx.Graph:
    """Add random edges between non-random nodes."""
    g = g.copy()
    nodes = list(g.nodes)
    edges = set(g.edges)

    for i in nodes:
        role = g.nodes[i].get("role", "r")
        if role != "r":
            if random.random() <= hydrate_prob:
                j = random.choice(nodes)
                if (i, j) not in edges and (j, i) not in edges and i != j:
                    g.add_edge(i, j)
                    edges.add((i, j))

    return g


def _bridge_connect_hydration(g: nx.Graph) -> nx.Graph:
    """Connect bridge nodes to other bridge nodes."""
    g = g.copy()
    edges = set(g.edges)

    # Get bridge nodes
    bridge_nodes = []
    bridge_connectors = []

    for node in g.nodes:
        con_type = g.nodes[node].get("con_type", "")
        role = g.nodes[node].get("role", "")

        if con_type == "bridge" and role != "r":
            bridge_nodes.append(node)
        if con_type == "bridge":
            bridge_connectors.append(node)

    # Connect bridge nodes
    for node in bridge_nodes:
        if bridge_connectors:
            node2 = random.choice(bridge_connectors)
            if (node, node2) not in edges and (node2, node) not in edges and node != node2:
                g.add_edge(node, node2)
                edges.add((node, node2))

    return g


def _reset_node_labels(g: nx.Graph, shuffle: bool = False) -> nx.Graph:
    """Reset node labels to consecutive integers starting from 0."""
    nodes = list(g.nodes)
    if shuffle:
        random.shuffle(nodes)

    mapping = {node: idx for idx, node in enumerate(nodes)}
    return nx.relabel_nodes(g, mapping)

<p align="center">
  <h1 align="center">USNET</h1>
  <p align="center">
    <strong>Unsupervised Structural Node Embedding Toolkit</strong>
  </p>
  <p align="center">
    A Python library for evaluating graph embeddings using structural node features without ground truth labels
  </p>
</p>

<p align="center">
  <a href="https://github.com/ashdehghan/usnet/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg"></a>
  <a href="https://pypi.org/project/usnet/"><img alt="PyPI" src="https://img.shields.io/badge/pypi-usnet-blue.svg"></a>
</p>

---

## Overview

**USNET** is an unsupervised evaluation framework for graph embeddings that measures embedding quality by comparing against structural node features. It provides tools for generating synthetic graphs, extracting structural metrics, and evaluating embeddings through distance correlation, cluster consistency, and optimization-based methods.

### Key Features

- **Unsupervised Evaluation**: No ground truth labels required—uses structural node features as proxy for evaluation
- **Multiple Metrics**: Distance correlation, K-means neighbor consistency, and optimization-based scoring
- **Weight Optimization**: PyTorch-based learning of optimal embedding dimension weights
- **Synthetic Graph Generation**: Built-in tools for creating benchmark graphs with known structural patterns
- **12 Structural Features**: Centrality measures, PageRank, clustering coefficients, and more

---

## What Problem Does USNET Solve?

Evaluating graph embeddings typically requires labeled data for downstream tasks like node classification. However, obtaining labels is expensive, and task-specific evaluation may not reflect general embedding quality.

**USNET addresses this by:**

1. Computing **structural node features** (centrality, PageRank, etc.) as unsupervised ground truth
2. Measuring how well embeddings **preserve structural similarity**—nodes similar in feature space should be similar in embedding space
3. Learning **optimal dimension weights** to maximize alignment between embedding distances and feature distances
4. Providing **interpretable metrics** that quantify embedding quality without labels

This enables researchers to compare embedding methods objectively and identify which dimensions capture meaningful structural information.

---

## Installation

### From PyPI

```bash
pip install usnet
```

### Using uv (Recommended)

```bash
uv add usnet
```

### From Source

```bash
git clone https://github.com/ashdehghan/usnet.git
cd usnet
uv sync
```

### With Development Dependencies

```bash
uv sync --all-extras
```

### Requirements

- Python 3.10+
- NumPy >= 1.24.0
- NetworkX >= 3.0
- PyTorch >= 2.0.0
- scikit-learn >= 1.3.0

---

## Quick Start

```python
import usnet
import numpy as np

# Generate a synthetic graph with known structure
graph = usnet.generate_barbell(m1=10, m2=10)

# Extract structural node features (ground truth)
features = usnet.extract_features(graph)
print(f"Extracted {len(features.columns) - 1} structural features")

# Your embeddings (e.g., from Node2Vec, DeepWalk, etc.)
embeddings = np.random.rand(30, 32)  # Replace with actual embeddings

# Evaluate embeddings using the optimization framework
result = usnet.evaluate_embeddings(
    embeddings=embeddings,
    features=features.drop(columns=['node']).values,
)

print(f"Pre-optimization score:  {result.pre_optimization_score:.4f}")
print(f"Post-optimization score: {result.post_optimization_score:.4f}")
print(f"Improvement: {result.improvement:.2%}")
print(f"Learned weights: {result.weights[:5]}...")  # Top 5 dimension weights
```

---

## Evaluation Methods

USNET provides three complementary evaluation approaches:

| Method | Description | Output | Speed |
|--------|-------------|--------|-------|
| `distance_correlation` | Pearson correlation of pairwise distance matrices | Mean correlation ∈ [-1, 1] | Fast |
| `neighbor_consistency` | K-means cluster overlap between spaces | Overlap percentage ∈ [0, 1] | Fast |
| `evaluate_embeddings` | PyTorch optimization of dimension weights | Optimized score + weights | Slower |

### Distance Correlation

Measures whether nodes that are similar in feature space are also similar in embedding space:

```python
from usnet.evaluation import distance_correlation

result = distance_correlation(embeddings, features, metric="cosine")
print(f"Mean correlation: {result.mean:.3f}")
print(f"Std deviation:    {result.std:.3f}")
print(f"Median:           {result.median:.3f}")

# Per-node correlations available
per_node = result.per_node  # Array of correlations for each node
```

### Neighbor Consistency

Evaluates if K-means cluster neighbors in feature space are also neighbors in embedding space:

```python
from usnet.evaluation import neighbor_consistency

result = neighbor_consistency(embeddings, features, n_clusters=5)
print(f"Feature → Embedding: {result.feature_to_embedding.mean:.3f}")
print(f"Embedding → Feature: {result.embedding_to_feature.mean:.3f}")
```

### Optimization Framework

The flagship method that learns optimal embedding dimension weights:

```python
from usnet.evaluation import EvaluationFramework

framework = EvaluationFramework(
    n_clusters=5,          # Clusters for pseudo-labeling
    sampling_fraction=0.5, # Within-cluster sampling probability
    n_ensembles=10,        # Ensemble runs for robustness
    use_pytorch=True,      # Use PyTorch optimizer
    random_state=42        # Reproducibility
)

result = framework.evaluate(embeddings, features)

# Interpret results
print(f"Score improvement: {result.improvement:.2%}")
print(f"Top 3 dimensions: {np.argsort(result.weights)[-3:]}")
```

---

## Structural Features

USNET extracts 12 structural node features using NetworkX and igraph:

| Feature | Source | Description |
|---------|--------|-------------|
| `degree_centrality` | NetworkX | Fraction of nodes connected to |
| `closeness_centrality` | NetworkX | Inverse average distance to all nodes |
| `pagerank` | NetworkX | Random walk importance score |
| `load_centrality` | NetworkX | Fraction of shortest paths through node |
| `average_neighbor_degree` | NetworkX | Mean degree of neighbors |
| `eigenvector_centrality` | NetworkX | Influence based on neighbor importance |
| `harmonic_centrality` | igraph | Sum of inverse distances to all nodes |
| `betweenness_centrality` | igraph | Fraction of shortest paths through node |
| `hub_score` | igraph | HITS hub score |
| `constraint` | igraph | Burt's structural constraint |
| `coreness` | igraph | K-core decomposition number |
| `eccentricity` | igraph | Maximum distance to any node |

### Usage

```python
from usnet.features import extract_features, FeatureExtractor, AVAILABLE_FEATURES

# Extract all features
features = extract_features(graph)

# Extract specific features
extractor = FeatureExtractor(
    features=["degree_centrality", "pagerank", "betweenness_centrality"]
)
features = extractor.extract(graph)

# List available features
print(AVAILABLE_FEATURES)
```

---

## Synthetic Graph Generation

USNET includes utilities for generating synthetic graphs with known structural patterns:

```python
from usnet.graphs import GraphGenerator, generate_barbell, generate_star, generate_web

# Individual patterns
barbell = generate_barbell(m1=10, m2=10)  # Two cliques connected by path
star = generate_star()                      # Hub with intermediate and leaf nodes
web = generate_web()                        # Concentric ring structure

# Composite graphs with multiple patterns
generator = GraphGenerator(
    barbell_count=2,
    web_count=3,
    star_count=5,
    hydrate_prob=0.1,    # Random inter-pattern edges
    seed=42              # Reproducibility
)
graph = generator.build()

# Nodes have 'role' attributes for ground truth evaluation
roles = set(graph.nodes[n]['role'] for n in graph.nodes)
print(f"Structural roles: {roles}")
```

### Available Patterns

| Pattern | Description | Nodes |
|---------|-------------|-------|
| `barbell` | Two complete graphs connected by a path | 30 (default) |
| `star` | Hub → intermediates → leaves | 16 |
| `web` | Hub → inner ring → outer ring (connected) | 16 |
| `dense_star` | Hub directly connected to many leaves | Configurable |

---

## Benchmark Suite

Run comprehensive benchmarks with multiple normalizations and cluster sizes:

```python
from usnet.evaluation import BenchmarkRunner

runner = BenchmarkRunner(
    normalization_types=["standard", "minmax_clipped"],
    n_clusters_list=[2, 3, 5, 10],
    distance_metrics=["cosine", "euclidean"]
)

# Aggregated results
results = runner.run(embeddings, features)
for metric, scores in results.items():
    print(f"{metric}: mean={scores['mean']:.3f}")

# Per-node detailed results
detailed_df = runner.run_detailed(embeddings, features)
detailed_df.to_csv("benchmark_results.csv")
```

---

## Repository Structure

```
USNET/
├── src/usnet/
│   ├── __init__.py           # Public API exports
│   ├── _typing.py            # Type aliases
│   ├── graphs/               # Graph generation
│   │   ├── __init__.py
│   │   └── generators.py     # Barbell, star, web patterns
│   ├── features/             # Feature extraction
│   │   ├── __init__.py
│   │   └── extractor.py      # 12 structural metrics
│   └── evaluation/           # Evaluation framework
│       ├── __init__.py
│       ├── benchmarks.py     # Distance correlation, K-means
│       └── framework.py      # PyTorch optimization
├── tests/                    # Pytest test suite
│   ├── conftest.py
│   ├── test_graphs.py
│   ├── test_features.py
│   └── test_evaluation.py
├── .github/workflows/        # CI/CD
│   ├── ci.yml
│   └── release.yml
├── pyproject.toml            # Package configuration (uv/hatch)
└── README.md
```

---

## Testing

```bash
# Install dev dependencies
uv sync --all-extras

# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=usnet --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_evaluation.py -v
```

---

## Algorithm Overview

1. **Feature Extraction**: Compute 12 structural metrics for each node using NetworkX and igraph

2. **K-means Pseudo-labeling**: Cluster nodes based on structural features to create unsupervised "ground truth" groups

3. **Pair Sampling**: Sample node pairs within clusters (similar nodes) and between clusters (dissimilar nodes)

4. **Distance Computation**: Calculate pairwise distances in both feature space and embedding space

5. **Weight Optimization**: Use PyTorch to learn dimension weights that maximize Pearson correlation between feature distances and weighted embedding distances

6. **Scoring**: Report `1 - r²` as the cost (lower is better), along with learned dimension weights

---

## Citation

If you use USNET in your research, please cite:

```bibtex
@software{usnet2024,
  title = {USNET: Unsupervised Structural Node Embedding Toolkit},
  author = {Dehghan, Ash},
  year = {2024},
  url = {https://github.com/ashdehghan/usnet}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

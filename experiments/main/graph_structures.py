"""The three graph structures share the same three node features."""

import numpy as np
import torch

import model as core


GRAPH_LABELS = {
    "full": "Adjacent + correlation + self-loops",
    "adjacent_only": "Adjacent + self-loops",
    "self_only": "Self-loops only",
}
GRAPH_ORDER = ["full", "adjacent_only", "self_only"]


def build_edge_index(normalized_train, graph_mode, top_k=12):
    if graph_mode not in GRAPH_LABELS:
        raise ValueError(f"Unknown graph structure: {graph_mode}")
    if normalized_train.ndim != 2 or normalized_train.shape[1] < 4:
        raise ValueError("Expected training spectra with at least four wavelength channels")
    if not np.isfinite(normalized_train).all():
        raise ValueError("Training spectra contain non-finite values")
    if graph_mode == "full":
        # Preserve the main model's graph builder and edge ordering.
        return core.build_hybrid_edge_index(normalized_train, top_k=top_k)
    n = normalized_train.shape[1]
    edges = {(i, i) for i in range(n)}
    if graph_mode == "adjacent_only":
        for i in range(n - 1):
            edges.add((i, i + 1))
            edges.add((i + 1, i))
    return torch.tensor(list(edges), dtype=torch.long).t().contiguous()

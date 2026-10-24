"""Original comparator encoder classes, instantiated with three input features.

The archived class definitions are retained verbatim (apart from line endings).
No residual encoder is used as the ordinary GAT comparator.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, global_mean_pool, global_max_pool

class SAGEEncoder(nn.Module):
    def __init__(self, in_dim=3, hid=128, dropout=0.5, num_classes=5):
        super().__init__()
        self.dropout = dropout
        self.enc = nn.Linear(in_dim, hid)

        # GraphSAGE (mean aggregator)
        self.sage1 = SAGEConv(hid, hid)
        self.norm1 = nn.LayerNorm(hid)

        self.sage2 = SAGEConv(hid, hid)
        self.norm2 = nn.LayerNorm(hid)

        self.head = nn.Sequential(
            nn.Linear(hid * 2, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.enc(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.sage1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.sage2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)

        g_mean = global_mean_pool(x, batch)
        g_max = global_max_pool(x, batch)
        emb = torch.cat([g_mean, g_max], dim=1)

        logits = self.head(emb)
        return emb, logits


class GCNEncoder(nn.Module):
    def __init__(self, in_dim=3, hid=128, dropout=0.5, num_classes=5):  # hid 增大到 128
        super().__init__()
        self.dropout = dropout
        self.enc = nn.Linear(in_dim, hid)

        self.gcn1 = GCNConv(hid, hid, add_self_loops=False, normalize=True)
        self.norm1 = nn.LayerNorm(hid)

        self.gcn2 = GCNConv(hid, hid, add_self_loops=False, normalize=True)
        self.norm2 = nn.LayerNorm(hid)

        self.head = nn.Sequential(
            nn.Linear(hid * 2, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.enc(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gcn1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gcn2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)

        g_mean = global_mean_pool(x, batch)
        g_max = global_max_pool(x, batch)
        emb = torch.cat([g_mean, g_max], dim=1)
        logits = self.head(emb)
        return emb, logits


class GINEncoder(nn.Module):
    def __init__(self, in_dim=3, hid=64, dropout=0.5, num_classes=5):
        super().__init__()
        self.dropout = dropout

        # GIN 需要定义内部的 MLP
        # Layer 1 MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
        )
        # GINConv 1
        self.gin1 = GINConv(self.mlp1, train_eps=True)  # train_eps=True 让模型自己学习中心节点权重

        # Layer 2 MLP
        self.mlp2 = nn.Sequential(
            nn.Linear(hid, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
        )
        # GINConv 2
        self.gin2 = GINConv(self.mlp2, train_eps=True)

        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(hid * 2, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, num_classes)
        )

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GIN Layer 1
        x = self.gin1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GIN Layer 2
        x = self.gin2(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling (Mean + Max)
        g_mean = global_mean_pool(x, batch)
        g_max = global_max_pool(x, batch)
        emb = torch.cat([g_mean, g_max], dim=1)  # [B, 2*hid]

        logits = self.head(emb)
        return emb, logits


class GATEncoder(nn.Module):
    def __init__(self, in_dim=3, hid=64, heads=4, dropout=0.5, num_classes=5):
        super().__init__()
        self.dropout = dropout

        self.enc = nn.Linear(in_dim, hid)

        self.gat1 = GATConv(hid, hid, heads=heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hid * heads)

        self.gat2 = GATConv(hid * heads, hid, heads=1, concat=False, dropout=dropout)
        self.norm2 = nn.LayerNorm(hid)

        self.head = nn.Sequential(
            nn.Linear(hid * 2, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, num_classes)
        )

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.enc(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)

        g_mean = global_mean_pool(x, batch)
        g_max = global_max_pool(x, batch)
        emb = torch.cat([g_mean, g_max], dim=1)

        logits = self.head(emb)
        return emb, logits


MODEL_SPECS = {'graphsage': {'class_name': 'SAGEEncoder', 'hidden': 128, 'in_dim': 3, 'dropout': 0.5, 'num_classes': 5, 'heads': None, 'embedding_dim': 256, 'residual_connections': False, 'source_file': 'main_SAGE.py', 'class_snapshot_sha256': 'b9915ad7c862285f0c60801377dbcf216f54722de41c2ec80ca12ab822a74221'}, 'gcn': {'class_name': 'GCNEncoder', 'hidden': 128, 'in_dim': 3, 'dropout': 0.5, 'num_classes': 5, 'heads': None, 'embedding_dim': 256, 'residual_connections': False, 'source_file': 'main_GCN_2.py', 'class_snapshot_sha256': '1b32b1a7cdafcd7be3b1a3aff1b85700d277e7e83e526a1bdb15093c8931cd7e'}, 'gin': {'class_name': 'GINEncoder', 'hidden': 64, 'in_dim': 3, 'dropout': 0.5, 'num_classes': 5, 'heads': None, 'embedding_dim': 128, 'residual_connections': False, 'source_file': 'main_GIN_1.py', 'class_snapshot_sha256': 'bcf3b83fda9b3f8155d74481b09dc87f0dfff4a35f60ae6865925a1991da149e'}, 'gat': {'class_name': 'GATEncoder', 'hidden': 64, 'in_dim': 3, 'dropout': 0.5, 'num_classes': 5, 'heads': 4, 'embedding_dim': 128, 'residual_connections': False, 'source_file': 'main_6.py', 'class_snapshot_sha256': 'd77d42d8bacbeac4029b7c4d9e8fa802a9a759df3ebdd8fab52fbd27db938b5b'}}


MODEL_ORDER = ("graphsage", "gcn", "gin", "gat")
MODEL_LABELS = {
    "graphsage": "GraphSAGE + LightGBM",
    "gcn": "GCN + LightGBM",
    "gin": "GIN + LightGBM",
    "gat": "GAT (non-residual) + LightGBM",
}


def model_spec(architecture):
    if architecture not in MODEL_SPECS:
        raise ValueError(f"Unsupported comparator: {architecture}")
    return deepcopy(MODEL_SPECS[architecture])


def build_encoder(architecture):
    spec = model_spec(architecture)
    constructor = globals()[spec["class_name"]]
    kwargs = dict(in_dim=spec["in_dim"], hid=spec["hidden"],
                  dropout=spec["dropout"], num_classes=spec["num_classes"])
    if architecture == "gat":
        kwargs["heads"] = spec["heads"]
    return constructor(**kwargs)

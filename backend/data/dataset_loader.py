"""
dataset_loader.py
-----------------
PyTorch Dataset class for the processed propagation graph files.
Used by all training and evaluation scripts.

Supports:
  - Loading by split (train / val / test)
  - Time-horizon truncation for early detection experiments
    (returns subgraph visible up to tau minutes)
  - Cross-dataset loading (twitter16 test set)

Usage example:
    from backend.data.dataset_loader import PropagationDataset
    from torch_geometric.loader import DataLoader

    train_ds = PropagationDataset(dataset='twitter15', split='train')
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # Early detection at 1440 minutes (24 hours)
    early_ds = PropagationDataset(
        dataset='twitter15', split='test', tau_minutes=1440
    )
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import torch
from torch.utils.data import Dataset

from backend.data.graph_data import PropagationData


# ── constants ─────────────────────────────────────────────────────────────────

NUM_SNAPSHOTS  = 5
NODE_FEAT_DIM  = 6
EDGE_FEAT_DIM  = 3
TEXT_EMBED_DIM = 384

GRAPHS_DIR = {
    "twitter15": os.path.join("dataset", "processed", "twitter15", "graphs"),
    "twitter16": os.path.join("dataset", "processed", "twitter16", "graphs"),
}

SPLITS_DIR = {
    "twitter15": os.path.join("dataset", "raw", "twitter15", "splits"),
    "twitter16": os.path.join("dataset", "raw", "twitter16", "splits"),
}

VALID_SPLITS = {
    "twitter15": ("train", "val", "test"),
    "twitter16": ("test",),
}


# ── conversion helper ─────────────────────────────────────────────────────────

def to_propagation_data(data):
    prop = PropagationData()
    for key, value in data.items():   # ← add .items()
        if key == "root_text_emb" and value.dim() == 1:
            prop[key] = value.unsqueeze(0)
        else:
            prop[key] = value
    return prop


# ── dataset class ─────────────────────────────────────────────────────────────

class PropagationDataset(Dataset):
    """
    Dataset of propagation graph PropagationData objects.

    Args:
        dataset     : str   'twitter15' or 'twitter16'
        split       : str   'train', 'val', or 'test'
        tau_minutes : float or None
                      If given, truncates each graph to nodes/edges
                      that arrived within tau_minutes of the root post.
                      None means use the full graph (no truncation).

    Each item returned is a PropagationData object with:
        x                  FloatTensor [N', 6]
        edge_index         LongTensor  [2, E']
        edge_attr          FloatTensor [E', 3]
        root_text_emb      FloatTensor [384]
        snapshot_node_mask BoolTensor  [5, N']
        snapshot_edge_mask BoolTensor  [5, E']
        y                  LongTensor  [1]
        claim_id           str
        num_nodes          int
        num_edges          int
        max_timestamp      float
        tau_minutes        float  (-1.0 if no truncation)
    """

    def __init__(self, dataset="twitter15", split="train", tau_minutes=None):
        super().__init__()

        if dataset not in GRAPHS_DIR:
            raise ValueError(f"Unknown dataset '{dataset}'. "
                             f"Choose from {list(GRAPHS_DIR.keys())}")

        if split not in VALID_SPLITS[dataset]:
            raise ValueError(
                f"Invalid split '{split}' for dataset '{dataset}'. "
                f"Valid splits: {VALID_SPLITS[dataset]}"
            )

        self.dataset     = dataset
        self.split       = split
        self.tau_minutes = tau_minutes
        self.graphs_dir  = GRAPHS_DIR[dataset]
        self.splits_dir  = SPLITS_DIR[dataset]

        self.claim_ids = self._load_split_ids(split)

        missing = [
            cid for cid in self.claim_ids
            if not os.path.exists(
                os.path.join(self.graphs_dir, f"claim_{cid}.pt")
            )
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} graph files missing from {self.graphs_dir}. "
                f"First missing: claim_{missing[0]}.pt\n"
                f"Run build_graphs.py first."
            )

    def _load_split_ids(self, split):
        split_file = os.path.join(self.splits_dir, f"{split}_ids.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Run make_splits.py first."
            )
        with open(split_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        if not ids:
            raise ValueError(f"Split file is empty: {split_file}")
        return ids

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        claim_id = self.claim_ids[idx]
        path     = os.path.join(self.graphs_dir, f"claim_{claim_id}.pt")

        # load and convert to PropagationData for correct batching
        raw  = torch.load(path, weights_only=False)
        data = to_propagation_data(raw)

        if self.tau_minutes is not None:
            data = self._truncate_to_tau(data, self.tau_minutes)
        else:
            data.tau_minutes = -1.0

        return data

    # ── tau truncation ────────────────────────────────────────────────────────

    def _truncate_to_tau(self, data, tau):
        """
        Returns a PropagationData containing only nodes/edges
        that arrived within tau minutes.

        x[:, 0] is normalized_timestamp = node_time / max_time.
        Absolute time = x[:, 0] * data.max_timestamp.
        ROOT always has absolute time 0.0 and is always retained.
        """
        max_t     = data.max_timestamp
        abs_times = data.x[:, 0] * max_t        # [N]

        node_keep = (abs_times <= tau + 1e-9)   # [N]
        node_keep[0] = True                      # always keep ROOT

        kept_nodes = node_keep.nonzero(as_tuple=True)[0]   # [N']
        N_new      = kept_nodes.shape[0]

        old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long)
        old_to_new[kept_nodes] = torch.arange(N_new, dtype=torch.long)

        x_new = data.x[kept_nodes]

        if data.num_edges > 0:
            src = data.edge_index[0]
            dst = data.edge_index[1]

            edge_keep     = node_keep[src] & node_keep[dst]
            kept_edges    = edge_keep.nonzero(as_tuple=True)[0]
            E_new         = kept_edges.shape[0]

            if E_new > 0:
                src_new        = old_to_new[src[kept_edges]]
                dst_new        = old_to_new[dst[kept_edges]]
                edge_index_new = torch.stack([src_new, dst_new], dim=0)
                edge_attr_new  = data.edge_attr[kept_edges]
            else:
                edge_index_new = torch.zeros((2, 0), dtype=torch.long)
                edge_attr_new  = torch.zeros((0, EDGE_FEAT_DIM),
                                             dtype=torch.float32)
        else:
            E_new          = 0
            edge_index_new = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_new  = torch.zeros((0, EDGE_FEAT_DIM),
                                         dtype=torch.float32)

        # recompute snapshot masks for the truncated graph
        x_new, snap_node_new, snap_edge_new = self._recompute_snapshots(
            x_new, edge_index_new, E_new, N_new
        )

        truncated = PropagationData(
            x                  = x_new,
            edge_index         = edge_index_new,
            edge_attr          = edge_attr_new,
            root_text_emb      = data.root_text_emb,
            snapshot_node_mask = snap_node_new,
            snapshot_edge_mask = snap_edge_new,
            y                  = data.y,
            claim_id           = data.claim_id,
            num_nodes          = N_new,
            num_edges          = E_new,
            max_timestamp      = data.max_timestamp,
            max_tree_depth     = data.max_tree_depth,
            tau_minutes        = float(tau),
        )
        return truncated

    def _recompute_snapshots(self, x, edge_index, E, N):
        """
        Recomputes cumulative snapshot masks for a truncated graph.
        Renormalizes timestamps within the kept node set.
        """
        if N == 0:
            return (x,
                    torch.zeros(NUM_SNAPSHOTS, 0, dtype=torch.bool),
                    torch.zeros(NUM_SNAPSHOTS, 0, dtype=torch.bool))

        orig_norm = x[:, 0]
        max_norm  = orig_norm.max().item()
        new_norm  = (orig_norm / max_norm) if max_norm > 1e-9 else orig_norm.clone()

        x_updated = x.clone()
        x_updated[:, 0] = new_norm

        boundaries = [(k + 1) / NUM_SNAPSHOTS for k in range(NUM_SNAPSHOTS)]
        snap_node  = torch.zeros(NUM_SNAPSHOTS, N, dtype=torch.bool)
        for k, b in enumerate(boundaries):
            snap_node[k] = (new_norm <= b + 1e-9)

        if E > 0:
            src       = edge_index[0]
            dst       = edge_index[1]
            snap_edge = torch.zeros(NUM_SNAPSHOTS, E, dtype=torch.bool)
            for k in range(NUM_SNAPSHOTS):
                snap_edge[k] = snap_node[k][src] & snap_node[k][dst]
        else:
            snap_edge = torch.zeros(NUM_SNAPSHOTS, 0, dtype=torch.bool)

        return x_updated, snap_node, snap_edge

    # ── utility ───────────────────────────────────────────────────────────────

    def get_label_weights(self):
        """
        Returns inverse-frequency class weights.
        Returns FloatTensor [2] for [credible, misinformation].
        """
        labels = []
        for cid in self.claim_ids:
            path = os.path.join(self.graphs_dir, f"claim_{cid}.pt")
            data = torch.load(path, weights_only=False)
            labels.append(data.y.item())

        n_total = len(labels)
        n_pos   = sum(labels)
        n_neg   = n_total - n_pos
        w_neg   = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
        w_pos   = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)

    def __repr__(self):
        tau_str = (f"tau={self.tau_minutes}min"
                   if self.tau_minutes is not None else "full")
        return (f"PropagationDataset("
                f"dataset={self.dataset}, "
                f"split={self.split}, "
                f"{tau_str}, "
                f"n={len(self)})")
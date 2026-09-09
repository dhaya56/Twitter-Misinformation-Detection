"""
tpgb.py
-------
Temporal Propagation Graph Builder (TPGB).

Takes a batched PropagationData object and extracts
per-snapshot subgraphs using the precomputed snapshot masks.

No learnable parameters. Pure data routing.

The TPGB produces a list of 5 snapshot dicts, each containing:
    x           : FloatTensor [N_k, node_feat_dim]
    edge_index  : LongTensor  [2, E_k]
    edge_attr   : FloatTensor [E_k, edge_feat_dim]
    batch       : LongTensor  [N_k]   graph membership per node
    pvt         : FloatTensor [B, 3]  propagation velocity features
                  [mean_edge_velocity, max_edge_velocity, edge_count_delta]
                  one row per graph in the batch

Where N_k and E_k are the number of active nodes/edges in snapshot k.

The PVT (Propagation Velocity Tensor) captures the rate of edge
formation between consecutive snapshots. It is computed here
and passed to the STE as a parallel temporal stream.
"""

import torch
import torch.nn as nn


class TPGB(nn.Module):
    """
    Temporal Propagation Graph Builder.

    Args:
        num_snapshots : int  number of temporal snapshots (default 5)
    """

    def __init__(self, num_snapshots=5):
        super().__init__()
        self.num_snapshots = num_snapshots

    def forward(self, batch):
        """
        Extracts per-snapshot subgraphs from a batched PropagationData.

        Args:
            batch : PropagationData batch from DataLoader
                    batch.x                  [N_total, 6]
                    batch.edge_index         [2, E_total]
                    batch.edge_attr          [E_total, 3]
                    batch.snapshot_node_mask [5, N_total]
                    batch.snapshot_edge_mask [5, E_total]
                    batch.batch              [N_total]  node->graph index
                    batch.edge_attr          [E_total, 3]

        Returns:
            snapshots : list of 5 dicts, each with keys:
                        x, edge_index, edge_attr, batch
            pvt       : FloatTensor [B, num_snapshots, 3]
                        propagation velocity features per graph per snapshot
        """
        B = batch.num_graphs

        snapshots = []
        edge_counts_per_graph = []   # [num_snapshots, B]  for PVT

        for k in range(self.num_snapshots):
            node_mask = batch.snapshot_node_mask[k]   # [N_total]  bool
            edge_mask = batch.snapshot_edge_mask[k]   # [E_total]  bool

            # ── extract active nodes ──────────────────────────────────────
            active_node_idx = node_mask.nonzero(as_tuple=True)[0]  # [N_k]
            x_k     = batch.x[active_node_idx]                     # [N_k, 6]
            batch_k = batch.batch[active_node_idx]                  # [N_k]

            # ── extract active edges ──────────────────────────────────────
            active_edge_idx = edge_mask.nonzero(as_tuple=True)[0]  # [E_k]

            if active_edge_idx.shape[0] > 0:
                edge_index_full = batch.edge_index[:, active_edge_idx]  # [2, E_k]
                edge_attr_k     = batch.edge_attr[active_edge_idx]      # [E_k, 3]

                # remap global node indices to local (within active set)
                # build a lookup: global_node_idx -> local_idx
                global_to_local = torch.full(
                    (batch.x.shape[0],), -1,
                    dtype=torch.long,
                    device=batch.x.device,
                )
                global_to_local[active_node_idx] = torch.arange(
                    active_node_idx.shape[0],
                    dtype=torch.long,
                    device=batch.x.device,
                )
                edge_index_k = global_to_local[edge_index_full]  # [2, E_k]
            else:
                edge_index_k = torch.zeros(
                    (2, 0), dtype=torch.long, device=batch.x.device
                )
                edge_attr_k = torch.zeros(
                    (0, batch.edge_attr.shape[1]),
                    dtype=torch.float32,
                    device=batch.x.device,
                )

            snapshots.append({
                "x"          : x_k,
                "edge_index" : edge_index_k,
                "edge_attr"  : edge_attr_k,
                "batch"      : batch_k,
            })

            # ── count edges per graph in this snapshot (for PVT) ─────────
            if active_edge_idx.shape[0] > 0:
                # which graph does each active edge belong to?
                # use the source node's graph membership
                src_global = batch.edge_index[0, active_edge_idx]
                edge_graph  = batch.batch[src_global]             # [E_k]

                counts = torch.zeros(
                    B, dtype=torch.float32, device=batch.x.device
                )
                counts.scatter_add_(
                    0, edge_graph,
                    torch.ones(
                        edge_graph.shape[0],
                        dtype=torch.float32,
                        device=batch.x.device,
                    )
                )
            else:
                counts = torch.zeros(
                    B, dtype=torch.float32, device=batch.x.device
                )

            edge_counts_per_graph.append(counts)

        # ── compute PVT features ──────────────────────────────────────────
        # edge_counts_per_graph: list of num_snapshots tensors, each [B]
        edge_counts = torch.stack(edge_counts_per_graph, dim=1)  # [B, S]

        pvt = self._compute_pvt(edge_counts, B, batch.x.device)  # [B, S, 3]

        return snapshots, pvt

    def _compute_pvt(self, edge_counts, B, device):
        """
        Computes Propagation Velocity Tensor features.

        For each snapshot k and each graph:
            [0] edge_delta      : edges added since previous snapshot
                                  (0 for k=0)
            [1] norm_edge_count : edges in snapshot k / max edges across all
                                  snapshots for this graph
            [2] velocity_sign   : 1.0 if growing, 0.0 if flat/shrinking

        Args:
            edge_counts : FloatTensor [B, num_snapshots]

        Returns:
            FloatTensor [B, num_snapshots, 3]
        """
        S   = self.num_snapshots
        pvt = torch.zeros(B, S, 3, dtype=torch.float32, device=device)

        # edge delta between consecutive snapshots
        deltas       = torch.zeros(B, S, dtype=torch.float32, device=device)
        deltas[:, 0] = edge_counts[:, 0]
        deltas[:, 1:] = edge_counts[:, 1:] - edge_counts[:, :-1]
        deltas = deltas.clamp(min=0.0)   # only count additions, not removals

        # normalized edge count
        max_counts = edge_counts.max(dim=1, keepdim=True).values.clamp(min=1.0)
        norm_counts = edge_counts / max_counts   # [B, S]

        # velocity sign: 1.0 if delta > 0
        vel_sign = (deltas > 0).float()

        pvt[:, :, 0] = deltas / max_counts      # normalized delta
        pvt[:, :, 1] = norm_counts
        pvt[:, :, 2] = vel_sign

        return pvt
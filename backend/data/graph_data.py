"""
graph_data.py
-------------
Custom PyG Data subclass for propagation graphs.

Overrides __cat_dim__ so that snapshot masks are correctly
batched along the node/edge dimension (dim=1) rather than
the default dim=0.

This must be used wherever PropagationData objects are
created or loaded, to ensure DataLoader batching works.
"""

from torch_geometric.data import Data


class PropagationData(Data):
    """
    PyG Data subclass for CGDEX-Net propagation graphs.

    Custom batching behaviour:
        snapshot_node_mask  [5, N]  -> concatenate along dim=1 (node dim)
        snapshot_edge_mask  [5, E]  -> concatenate along dim=1 (edge dim)
        root_text_emb       [384]   -> stack along dim=0 (default)
        all other tensors           -> PyG default behaviour
    """

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("snapshot_node_mask", "snapshot_edge_mask"):
            return 1   # concatenate along node/edge dimension
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # snapshot masks do not contain node indices,
        # so no increment is needed for them
        if key in ("snapshot_node_mask", "snapshot_edge_mask"):
            return 0
        return super().__inc__(key, value, *args, **kwargs)
"""
Quick test for dataset_loader.py
Run from: D:\vit_projects\dl_project
Usage: python backend/data/test_dataset_loader.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

from backend.data.dataset_loader import PropagationDataset
from torch_geometric.loader import DataLoader


print("=" * 50)
print("TEST 1: Load full train/val/test splits")
for split in ("train", "val", "test"):
    ds = PropagationDataset(dataset="twitter15", split=split)
    print(f"  {split:<6}: {len(ds)} samples   {ds}")

print()
print("=" * 50)
print("TEST 2: Load one full graph and inspect")
ds   = PropagationDataset(dataset="twitter15", split="train")
item = ds[0]
print(f"  claim_id           : {item.claim_id}")
print(f"  x.shape            : {item.x.shape}")
print(f"  edge_index.shape   : {item.edge_index.shape}")
print(f"  edge_attr.shape    : {item.edge_attr.shape}")
print(f"  root_text_emb.shape: {item.root_text_emb.shape}")
print(f"  snapshot_node_mask : {item.snapshot_node_mask.shape}")
print(f"  snapshot_edge_mask : {item.snapshot_edge_mask.shape}")
print(f"  y                  : {item.y}")
print(f"  max_timestamp      : {item.max_timestamp:.1f} min")
print(f"  tau_minutes        : {item.tau_minutes}")

print()
print("=" * 50)
print("TEST 3: Tau truncation at multiple horizons")
horizons = [60, 360, 720, 1440, 2880, 4320]
ds_full  = PropagationDataset(dataset="twitter15", split="test")
item_full = ds_full[0]
print(f"  Full graph: N={item_full.num_nodes}  E={item_full.num_edges}"
      f"  max_t={item_full.max_timestamp:.1f}min")

for tau in horizons:
    ds_tau   = PropagationDataset(dataset="twitter15", split="test",
                                  tau_minutes=tau)
    item_tau = ds_tau[0]
    print(f"  tau={tau:>5}min : N={item_tau.num_nodes:<4} E={item_tau.num_edges:<4}"
          f"  snap_last_coverage="
          f"{item_tau.snapshot_node_mask[-1].sum().item()}/{item_tau.num_nodes}")

print()
print("=" * 50)
print("TEST 4: DataLoader batching")
ds     = PropagationDataset(dataset="twitter15", split="train")
loader = DataLoader(ds, batch_size=4, shuffle=False)
batch  = next(iter(loader))
print(f"  Batch type   : {type(batch)}")
print(f"  batch.x      : {batch.x.shape}")
print(f"  batch.y      : {batch.y}")
print(f"  batch.batch  : {batch.batch.shape}  (node->graph mapping)")

print()
print("=" * 50)
print("TEST 5: Twitter16 test set")
ds16 = PropagationDataset(dataset="twitter16", split="test")
print(f"  twitter16 test: {len(ds16)} samples")
item16 = ds16[0]
print(f"  claim_id: {item16.claim_id}  y={item16.y.item()}")

print()
print("All tests complete.")
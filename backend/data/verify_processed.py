"""
verify_processed.py
-------------------
Loads a sample of processed graph .pt files and verifies:
  - All expected tensors are present
  - Tensor shapes are correct
  - Snapshot masks are cumulative (each snapshot is superset of previous)
  - No NaN or Inf values in any tensor
  - Label balance matches expected distribution
  - Edge indices are within valid node range

Usage:
    python backend/data/verify_processed.py --dataset twitter15
    python backend/data/verify_processed.py --dataset twitter16
    python backend/data/verify_processed.py --dataset both
Run from: D:\vit_projects\dl_project
"""

import os
import sys
import argparse
import random

sys.path.insert(0, os.path.abspath("."))

import torch


# ── expected constants ────────────────────────────────────────────────────────

NUM_SNAPSHOTS  = 5
NODE_FEAT_DIM  = 6
EDGE_FEAT_DIM  = 3
TEXT_EMBED_DIM = 384
SAMPLE_SIZE    = 50   # number of graphs to check
RANDOM_SEED    = 42


# ── single graph verifier ─────────────────────────────────────────────────────

def verify_single(path, verbose=False):
    """
    Loads one .pt file and runs all checks.
    Returns list of error strings (empty if all pass).
    """
    errors = []

    try:
        data = torch.load(path, weights_only=False)
    except Exception as ex:
        return [f"LOAD ERROR: {ex}"]

    claim_id = getattr(data, "claim_id", "unknown")

    # ── required attributes ───────────────────────────────────────────────
    required = [
        "x", "edge_index", "edge_attr",
        "root_text_emb",
        "snapshot_node_mask", "snapshot_edge_mask",
        "y", "claim_id", "num_nodes", "num_edges",
        "max_timestamp", "max_tree_depth",
    ]
    for attr in required:
        if not hasattr(data, attr):
            errors.append(f"Missing attribute: {attr}")

    if errors:
        return errors   # cannot continue without required attrs

    N = data.num_nodes
    E = data.num_edges

    # ── shape checks ──────────────────────────────────────────────────────
    if data.x.shape != (N, NODE_FEAT_DIM):
        errors.append(f"x shape {data.x.shape} != ({N}, {NODE_FEAT_DIM})")

    if data.edge_index.shape != (2, E):
        errors.append(f"edge_index shape {data.edge_index.shape} != (2, {E})")

    if E > 0 and data.edge_attr.shape != (E, EDGE_FEAT_DIM):
        errors.append(f"edge_attr shape {data.edge_attr.shape} != ({E}, {EDGE_FEAT_DIM})")

    if data.root_text_emb.shape != (TEXT_EMBED_DIM,):
        errors.append(f"root_text_emb shape {data.root_text_emb.shape} != ({TEXT_EMBED_DIM},)")

    if data.snapshot_node_mask.shape != (NUM_SNAPSHOTS, N):
        errors.append(
            f"snapshot_node_mask shape {data.snapshot_node_mask.shape} "
            f"!= ({NUM_SNAPSHOTS}, {N})"
        )

    if data.snapshot_edge_mask.shape != (NUM_SNAPSHOTS, E):
        errors.append(
            f"snapshot_edge_mask shape {data.snapshot_edge_mask.shape} "
            f"!= ({NUM_SNAPSHOTS}, {E})"
        )

    if data.y.shape != (1,):
        errors.append(f"y shape {data.y.shape} != (1,)")

    if data.y.item() not in (0, 1):
        errors.append(f"y value {data.y.item()} not in {{0, 1}}")

    # ── dtype checks ──────────────────────────────────────────────────────
    if data.x.dtype != torch.float32:
        errors.append(f"x dtype {data.x.dtype} != float32")

    if data.edge_index.dtype != torch.int64:
        errors.append(f"edge_index dtype {data.edge_index.dtype} != int64")

    if data.y.dtype != torch.int64:
        errors.append(f"y dtype {data.y.dtype} != int64")

    # ── NaN / Inf checks ──────────────────────────────────────────────────
    for name, tensor in [("x", data.x),
                         ("edge_attr", data.edge_attr),
                         ("root_text_emb", data.root_text_emb)]:
        if tensor.numel() > 0:
            if torch.isnan(tensor).any():
                errors.append(f"NaN found in {name}")
            if torch.isinf(tensor).any():
                errors.append(f"Inf found in {name}")

    # ── edge index validity ───────────────────────────────────────────────
    if E > 0:
        if data.edge_index.min().item() < 0:
            errors.append("edge_index contains negative indices")
        if data.edge_index.max().item() >= N:
            errors.append(
                f"edge_index max {data.edge_index.max().item()} >= num_nodes {N}"
            )

    # ── node feature value range ──────────────────────────────────────────
    # features [0,1,2,3,4] should be in [0, 1]
    # feature [5] (parent_dt) should be >= 0
    for feat_idx, feat_name in enumerate([
        "norm_timestamp", "norm_depth", "norm_out_degree",
        "is_leaf", "is_root", "norm_parent_dt"
    ]):
        col = data.x[:, feat_idx]
        if col.min().item() < -1e-6:
            errors.append(
                f"Node feature [{feat_idx}] ({feat_name}) "
                f"has negative value: min={col.min().item():.4f}"
            )

    # ── snapshot mask cumulativeness ──────────────────────────────────────
    # snapshot k must be a superset of snapshot k-1 (cumulative)
    for k in range(1, NUM_SNAPSHOTS):
        prev = data.snapshot_node_mask[k - 1]
        curr = data.snapshot_node_mask[k]
        # any node in prev must also be in curr
        if not (prev & ~curr).any() == False:
            errors.append(
                f"snapshot_node_mask not cumulative at k={k}: "
                f"some nodes in snapshot {k-1} missing from snapshot {k}"
            )

    # snapshot 5 (last) must include ALL nodes
    if not data.snapshot_node_mask[-1].all():
        n_missing = (~data.snapshot_node_mask[-1]).sum().item()
        errors.append(
            f"Final snapshot does not include all nodes: "
            f"{n_missing} nodes missing"
        )

    # ROOT node (idx=0) must be in all snapshots
    for k in range(NUM_SNAPSHOTS):
        if not data.snapshot_node_mask[k, 0].item():
            errors.append(f"ROOT node missing from snapshot {k}")

    # ── num_nodes / num_edges consistency ─────────────────────────────────
    if data.x.shape[0] != data.num_nodes:
        errors.append(
            f"x.shape[0]={data.x.shape[0]} != num_nodes={data.num_nodes}"
        )
    if E > 0 and data.edge_attr.shape[0] != data.num_edges:
        errors.append(
            f"edge_attr.shape[0]={data.edge_attr.shape[0]} "
            f"!= num_edges={data.num_edges}"
        )

    if verbose and not errors:
        print(f"  [{claim_id}] "
              f"N={N} E={E} "
              f"y={data.y.item()} "
              f"depth={data.max_tree_depth} "
              f"max_t={data.max_timestamp:.1f}min  OK")

    return errors


# ── dataset verifier ──────────────────────────────────────────────────────────

def verify_dataset(dataset_name):
    graphs_dir = os.path.join(
        "dataset", "processed", dataset_name, "graphs"
    )

    print(f"\n{'='*60}")
    print(f"  Verifying: {dataset_name.upper()}")
    print(f"{'='*60}")

    if not os.path.exists(graphs_dir):
        print(f"  [ERROR] graphs directory not found: {graphs_dir}")
        return

    all_files = [f for f in os.listdir(graphs_dir) if f.endswith(".pt")]
    print(f"  Total .pt files found : {len(all_files)}")

    if not all_files:
        print("  [ERROR] No graph files found.")
        return

    # ── sample selection ──────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    sample = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))
    print(f"  Checking sample of    : {len(sample)} graphs")
    print()

    # ── run checks ────────────────────────────────────────────────────────
    all_errors   = []
    label_counts = {0: 0, 1: 0}
    node_counts  = []
    edge_counts  = []

    for fname in sample:
        path   = os.path.join(graphs_dir, fname)
        errors = verify_single(path, verbose=True)

        if errors:
            all_errors.append((fname, errors))
        else:
            data = torch.load(path, weights_only=False)
            label_counts[data.y.item()] += 1
            node_counts.append(data.num_nodes)
            edge_counts.append(data.num_edges)

    # ── summary ───────────────────────────────────────────────────────────
    print()
    n_passed = len(sample) - len(all_errors)
    print(f"  Passed : {n_passed} / {len(sample)}")

    if all_errors:
        print(f"  FAILED : {len(all_errors)}")
        for fname, errs in all_errors:
            print(f"    {fname}:")
            for e in errs:
                print(f"      - {e}")
    else:
        print(f"  All checks passed.")

    if node_counts:
        node_counts.sort()
        edge_counts.sort()
        print()
        print(f"  Sample label distribution:")
        print(f"    credible       : {label_counts[0]}")
        print(f"    misinformation : {label_counts[1]}")
        print()
        print(f"  Sample node count range : "
              f"{node_counts[0]} – {node_counts[-1]}  "
              f"(median={node_counts[len(node_counts)//2]})")
        print(f"  Sample edge count range : "
              f"{edge_counts[0]} – {edge_counts[-1]}  "
              f"(median={edge_counts[len(edge_counts)//2]})")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify processed graph .pt files."
    )
    parser.add_argument(
        "--dataset",
        choices=["twitter15", "twitter16", "both"],
        default="both",
    )
    args = parser.parse_args()

    if args.dataset in ("twitter15", "both"):
        verify_dataset("twitter15")

    if args.dataset in ("twitter16", "both"):
        verify_dataset("twitter16")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
"""
build_graphs.py
---------------
Converts parsed propagation trees into PyTorch Geometric Data objects.
Writes one .pt file per claim to the processed directory.

Reads:
    dataset/raw/{dataset}/label.txt
    dataset/raw/{dataset}/source_tweets.txt
    dataset/raw/{dataset}/tree/*.txt
    dataset/processed/{dataset}/dataset_stats.json
    dataset/processed/{dataset}/text_embeddings.pt

Writes:
    dataset/processed/{dataset}/graphs/claim_{claim_id}.pt

Each saved Data object contains:
    x                  FloatTensor [N, 6]   structural node features
    edge_index         LongTensor  [2, E]   directed edges (parent->child)
    edge_attr          FloatTensor [E, 3]   edge features
    root_text_emb      FloatTensor [384]    sentence embedding of source tweet
    snapshot_node_mask BoolTensor  [5, N]   cumulative snapshot membership
    snapshot_edge_mask BoolTensor  [5, E]   cumulative snapshot membership
    y                  LongTensor  [1]      binary label (0=credible, 1=misinfo)
    claim_id           str                  root tweet id
    num_nodes          int
    num_edges          int
    max_timestamp      float                unnormalized, for reference
    max_tree_depth     int                  for reference

Node features (x) — 6 dimensions:
    [0] normalized_timestamp   = node.timestamp / claim_max_time
                                 (0.0 for ROOT since ROOT time = 0)
    [1] normalized_depth       = node_depth / global_max_tree_depth
    [2] normalized_out_degree  = out_degree / max_out_degree_in_this_tree
    [3] is_leaf                = 1.0 if out_degree == 0 else 0.0
    [4] is_root                = 1.0 for ROOT node else 0.0
    [5] normalized_parent_dt   = (node.time - parent.time) / claim_max_time
                                 (0.0 for ROOT)

Edge features (edge_attr) — 3 dimensions:
    [0] normalized_retweet_delay  = (child_time - parent_time) / max_time
    [1] is_root_edge              = 1.0 if parent is ROOT else 0.0
    [2] normalized_child_depth    = child_depth / global_max_tree_depth

Usage:
    python backend/data/build_graphs.py --dataset twitter15
    python backend/data/build_graphs.py --dataset twitter16
    python backend/data/build_graphs.py --dataset both
Run from: D:\vit_projects\dl_project
"""

import os
import sys
import json
import argparse
from collections import defaultdict, deque

sys.path.insert(0, os.path.abspath("."))

import torch
from torch_geometric.data import Data

from backend.data.parse_raw import (
    parse_label_file,
    parse_source_tweets,
    parse_tree_file,
    map_label_to_binary,
)


# ── constants ─────────────────────────────────────────────────────────────────

NUM_SNAPSHOTS   = 5
MAX_NODES       = 300     # cap per full tree (keep earliest-arriving nodes)
NODE_FEAT_DIM   = 6
EDGE_FEAT_DIM   = 3
TEXT_EMBED_DIM  = 384


# ── feature computation helpers ───────────────────────────────────────────────

def compute_bfs_depths(nodes, edges):
    """
    Computes BFS depth from ROOT (node_idx=0) for every node.

    Returns:
        dict {node_idx -> depth}   ROOT has depth 0
    """
    children = defaultdict(list)
    for e in edges:
        children[e["src_idx"]].append(e["dst_idx"])

    depths  = {0: 0}
    queue   = deque([0])

    while queue:
        nid = queue.popleft()
        for child in children[nid]:
            if child not in depths:
                depths[child] = depths[nid] + 1
                queue.append(child)

    # any node not reachable from ROOT gets depth 0
    # (should not happen in a valid tree but guards against orphan nodes)
    for n in nodes:
        if n["node_idx"] not in depths:
            depths[n["node_idx"]] = 0

    return depths


def compute_out_degrees(nodes, edges):
    """
    Returns dict {node_idx -> out_degree} for all nodes.
    """
    out_deg = {n["node_idx"]: 0 for n in nodes}
    for e in edges:
        out_deg[e["src_idx"]] = out_deg.get(e["src_idx"], 0) + 1
    return out_deg


def compute_parent_times(nodes, edges):
    """
    Returns dict {node_idx -> parent_timestamp}.
    ROOT (node_idx=0) maps to 0.0.
    """
    parent_time = {0: 0.0}
    for e in edges:
        parent_time[e["dst_idx"]] = e["src_time"]
    return parent_time


def apply_node_cap(nodes, edges, max_nodes):
    """
    If the tree has more than max_nodes nodes, keep only the
    earliest-arriving max_nodes nodes (by timestamp).
    ROOT is always kept (timestamp=0.0, so always first).

    Returns:
        (filtered_nodes, filtered_edges, node_idx_remap)
        node_idx_remap: dict {old_idx -> new_idx}
    """
    if len(nodes) <= max_nodes:
        # no cap needed — remap is identity
        remap = {n["node_idx"]: n["node_idx"] for n in nodes}
        return nodes, edges, remap

    # sort nodes by timestamp, keep earliest max_nodes
    sorted_nodes = sorted(nodes, key=lambda n: n["timestamp"])
    kept_nodes   = sorted_nodes[:max_nodes]
    kept_idx_set = {n["node_idx"] for n in kept_nodes}

    # keep only edges where both endpoints are in kept set
    kept_edges = [
        e for e in edges
        if e["src_idx"] in kept_idx_set and e["dst_idx"] in kept_idx_set
    ]

    # reassign contiguous indices (sorted by original idx for determinism)
    kept_nodes_sorted = sorted(kept_nodes, key=lambda n: n["node_idx"])
    remap = {n["node_idx"]: new_idx
             for new_idx, n in enumerate(kept_nodes_sorted)}

    # rebuild node list with new indices
    new_nodes = []
    for n in kept_nodes_sorted:
        new_n = dict(n)
        new_n["node_idx"] = remap[n["node_idx"]]
        new_nodes.append(new_n)

    # rebuild edge list with remapped indices
    new_edges = []
    for e in kept_edges:
        new_e = dict(e)
        new_e["src_idx"] = remap[e["src_idx"]]
        new_e["dst_idx"] = remap[e["dst_idx"]]
        new_edges.append(new_e)

    return new_nodes, new_edges, remap


def build_snapshot_masks(nodes, edges, num_snapshots=5):
    """
    Builds cumulative snapshot masks.

    Snapshot k (0-indexed) includes all nodes whose normalized_timestamp
    is <= (k+1) / num_snapshots.

    normalized_timestamp = node.timestamp / max_timestamp_in_tree
    ROOT always has normalized_timestamp = 0.0 and is in all snapshots.

    Returns:
        snapshot_node_mask  BoolTensor [num_snapshots, N]
        snapshot_edge_mask  BoolTensor [num_snapshots, E]
    """
    N = len(nodes)
    E = len(edges)

    max_t = max((n["timestamp"] for n in nodes), default=1.0)
    if max_t == 0.0:
        max_t = 1.0   # guard against all-zero timestamps

    # normalized timestamp per node (sorted by node_idx)
    nodes_by_idx = sorted(nodes, key=lambda n: n["node_idx"])
    norm_times   = torch.tensor(
        [n["timestamp"] / max_t for n in nodes_by_idx],
        dtype=torch.float32,
    )   # [N]

    # boundaries: snapshot k covers [0, (k+1)/S]
    boundaries = [(k + 1) / num_snapshots for k in range(num_snapshots)]

    snapshot_node_mask = torch.zeros(num_snapshots, N, dtype=torch.bool)
    for k, boundary in enumerate(boundaries):
        snapshot_node_mask[k] = (norm_times <= boundary + 1e-9)

    # edge mask: both endpoints must be in the snapshot
    if E == 0:
        snapshot_edge_mask = torch.zeros(num_snapshots, 0, dtype=torch.bool)
        return snapshot_node_mask, snapshot_edge_mask

    src_indices = torch.tensor([e["src_idx"] for e in edges], dtype=torch.long)
    dst_indices = torch.tensor([e["dst_idx"] for e in edges], dtype=torch.long)

    snapshot_edge_mask = torch.zeros(num_snapshots, E, dtype=torch.bool)
    for k in range(num_snapshots):
        node_in_snap = snapshot_node_mask[k]          # [N]
        src_in = node_in_snap[src_indices]             # [E]
        dst_in = node_in_snap[dst_indices]             # [E]
        snapshot_edge_mask[k] = src_in & dst_in

    return snapshot_node_mask, snapshot_edge_mask


# ── graph builder ─────────────────────────────────────────────────────────────

def build_single_graph(claim_id, label_str, tree_result,
                       text_emb, global_max_depth):
    """
    Converts one parsed tree into a PyG Data object.

    Args:
        claim_id         : str
        label_str        : raw label string (e.g. 'non-rumor')
        tree_result      : dict from parse_tree_file()
        text_emb         : FloatTensor [384] or None
        global_max_depth : int  (from dataset_stats.json)

    Returns:
        torch_geometric.data.Data
    """
    nodes = tree_result["nodes"]
    edges = tree_result["edges"]

    # ── apply node cap ────────────────────────────────────────────────────
    nodes, edges, _ = apply_node_cap(nodes, edges, MAX_NODES)
    N = len(nodes)
    E = len(edges)

    # ── precompute structural properties ──────────────────────────────────
    depths       = compute_bfs_depths(nodes, edges)
    out_degrees  = compute_out_degrees(nodes, edges)
    parent_times = compute_parent_times(nodes, edges)

    max_t    = max((n["timestamp"] for n in nodes), default=1.0)
    if max_t == 0.0:
        max_t = 1.0

    max_out_deg = max(out_degrees.values()) if out_degrees else 1
    if max_out_deg == 0:
        max_out_deg = 1

    global_max_depth_safe = max(global_max_depth, 1)

    # ── node features  [N, 6] ─────────────────────────────────────────────
    # nodes are sorted by node_idx for deterministic ordering
    nodes_sorted = sorted(nodes, key=lambda n: n["node_idx"])

    x_rows = []
    for n in nodes_sorted:
        idx       = n["node_idx"]
        t_norm    = n["timestamp"] / max_t
        d_norm    = depths.get(idx, 0) / global_max_depth_safe
        od_norm   = out_degrees.get(idx, 0) / max_out_deg
        is_leaf   = 1.0 if out_degrees.get(idx, 0) == 0 else 0.0
        is_root_f = 1.0 if n["is_root"] else 0.0
        p_time    = parent_times.get(idx, n["timestamp"])
        dt_norm   = (n["timestamp"] - p_time) / max_t

        x_rows.append([t_norm, d_norm, od_norm, is_leaf, is_root_f, dt_norm])

    x = torch.tensor(x_rows, dtype=torch.float32)   # [N, 6]

    # ── edge index  [2, E] and edge features  [E, 3] ─────────────────────
    if E > 0:
        src_list   = [e["src_idx"] for e in edges]
        dst_list   = [e["dst_idx"] for e in edges]
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        edge_rows = []
        for e in edges:
            delay_norm   = (e["dst_time"] - e["src_time"]) / max_t
            is_root_edge = 1.0 if e["src_idx"] == 0 else 0.0
            child_depth  = depths.get(e["dst_idx"], 0) / global_max_depth_safe
            edge_rows.append([delay_norm, is_root_edge, child_depth])

        edge_attr = torch.tensor(edge_rows, dtype=torch.float32)   # [E, 3]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)

    # ── snapshot masks ────────────────────────────────────────────────────
    snapshot_node_mask, snapshot_edge_mask = build_snapshot_masks(
        nodes_sorted, edges, NUM_SNAPSHOTS
    )

    # ── text embedding ────────────────────────────────────────────────────
    if text_emb is not None:
        root_text_emb = text_emb.float()
    else:
        root_text_emb = torch.zeros(TEXT_EMBED_DIM, dtype=torch.float32)

    # ── label ─────────────────────────────────────────────────────────────
    y = torch.tensor([map_label_to_binary(label_str)], dtype=torch.long)

    # ── max tree depth for reference ──────────────────────────────────────
    max_depth_this_tree = max(depths.values()) if depths else 0

    # ── assemble Data object ──────────────────────────────────────────────
    data = Data(
        x                  = x,
        edge_index         = edge_index,
        edge_attr          = edge_attr,
        root_text_emb      = root_text_emb,
        snapshot_node_mask = snapshot_node_mask,
        snapshot_edge_mask = snapshot_edge_mask,
        y                  = y,
        claim_id           = claim_id,
        num_nodes          = N,
        num_edges          = E,
        max_timestamp      = float(max_t),
        max_tree_depth     = int(max_depth_this_tree),
    )

    return data


# ── dataset builder ───────────────────────────────────────────────────────────

def build_dataset(dataset_name):
    """
    Builds and saves all graphs for one dataset.

    Args:
        dataset_name : 'twitter15' or 'twitter16'
    """
    raw_dir       = os.path.join("dataset", "raw",       dataset_name)
    processed_dir = os.path.join("dataset", "processed", dataset_name)
    graphs_dir    = os.path.join(processed_dir, "graphs")
    stats_path    = os.path.join(processed_dir, "dataset_stats.json")
    emb_path      = os.path.join(processed_dir, "text_embeddings.pt")

    os.makedirs(graphs_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Building graphs: {dataset_name.upper()}")
    print(f"{'='*60}")

    # ── load dependencies ─────────────────────────────────────────────────
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Run stats_scan.py first: {stats_path}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Run embed_texts.py first: {emb_path}")

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    global_max_depth = stats["normalization"]["global_max_tree_depth"]
    print(f"  global_max_tree_depth : {global_max_depth}")

    print(f"  Loading text embeddings ...")
    text_embeddings = torch.load(emb_path, weights_only=True)
    print(f"  Text embeddings loaded : {len(text_embeddings)} entries")

    labels  = parse_label_file(os.path.join(raw_dir, "label.txt"))
    tree_dir = os.path.join(raw_dir, "tree")
    tree_ids = {os.path.splitext(f)[0]
                for f in os.listdir(tree_dir) if f.endswith(".txt")}
    valid_ids = sorted(cid for cid in labels if cid in tree_ids)
    print(f"  Valid claims          : {len(valid_ids)}")
    print(f"  Output directory      : {graphs_dir}")
    print()

    # ── build graphs ──────────────────────────────────────────────────────
    success_count  = 0
    skipped_count  = 0
    capped_count   = 0

    for i, claim_id in enumerate(valid_ids):
        out_file = os.path.join(graphs_dir, f"claim_{claim_id}.pt")

        # skip if already built (allows resuming interrupted runs)
        if os.path.exists(out_file):
            success_count += 1
            continue

        tree_path = os.path.join(tree_dir, claim_id + ".txt")

        try:
            tree_result = parse_tree_file(tree_path)
        except Exception as ex:
            print(f"  [ERROR] parse failed for {claim_id}: {ex}")
            skipped_count += 1
            continue

        if len(tree_result["nodes"]) > MAX_NODES:
            capped_count += 1

        text_emb  = text_embeddings.get(claim_id, None)
        label_str = labels[claim_id]

        try:
            data = build_single_graph(
                claim_id         = claim_id,
                label_str        = label_str,
                tree_result      = tree_result,
                text_emb         = text_emb,
                global_max_depth = global_max_depth,
            )
        except Exception as ex:
            print(f"  [ERROR] build failed for {claim_id}: {ex}")
            skipped_count += 1
            continue

        torch.save(data, out_file)
        success_count += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(valid_ids)} processed ...")

    print()
    print(f"  Done.")
    print(f"  Successfully built : {success_count}")
    print(f"  Skipped (errors)   : {skipped_count}")
    print(f"  Node-capped trees  : {capped_count}  (>{MAX_NODES} nodes)")
    print(f"  Output             : {graphs_dir}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build PyG graph objects from raw propagation trees."
    )
    parser.add_argument(
        "--dataset",
        choices=["twitter15", "twitter16", "both"],
        default="both",
        help="Which dataset to build (default: both)",
    )
    args = parser.parse_args()

    if args.dataset in ("twitter15", "both"):
        build_dataset("twitter15")

    if args.dataset in ("twitter16", "both"):
        build_dataset("twitter16")

    print("\nGraph building complete.")


if __name__ == "__main__":
    main()
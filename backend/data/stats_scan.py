"""
stats_scan.py
-------------
Single-pass statistics collector over all labeled tree files.
Produces dataset_stats.json required by build_graphs.py.

Computes:
  - global_max_tree_depth       : for normalizing depth features
  - global_max_timestamp        : informational only (not used for normalization
                                   since timestamps are normalized per-claim)
  - label distribution          : raw and binary
  - tree size distribution      : min, p25, median, p75, max
  - total monotonicity violations and duplicates across dataset
  - claims_missing_source_text  : should be 0 based on inspection

Usage:
    python backend/data/stats_scan.py --dataset twitter15
    python backend/data/stats_scan.py --dataset twitter16
Run from: D:\vit_projects\dl_project
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath("."))

from backend.data.parse_raw import (
    parse_label_file,
    parse_source_tweets,
    parse_tree_file,
    map_label_to_binary,
)


# ── percentile helper ─────────────────────────────────────────────────────────

def percentile(sorted_list, p):
    """Returns the p-th percentile of a pre-sorted list."""
    if not sorted_list:
        return 0
    idx = int(len(sorted_list) * p / 100)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


# ── main scan ─────────────────────────────────────────────────────────────────

def run_stats_scan(dataset_name):
    """
    Scans all labeled tree files for a given dataset.
    Writes dataset_stats.json to the processed directory.

    Args:
        dataset_name: 'twitter15' or 'twitter16'
    """
    raw_dir       = os.path.join("dataset", "raw",       dataset_name)
    processed_dir = os.path.join("dataset", "processed", dataset_name)
    os.makedirs(processed_dir, exist_ok=True)

    label_path  = os.path.join(raw_dir, "label.txt")
    source_path = os.path.join(raw_dir, "source_tweets.txt")
    tree_dir    = os.path.join(raw_dir, "tree")
    out_path    = os.path.join(processed_dir, "dataset_stats.json")

    print(f"\n{'='*60}")
    print(f"  Stats scan: {dataset_name.upper()}")
    print(f"{'='*60}")

    # ── load label and source files ───────────────────────────────────────
    print("  Loading label.txt ...")
    labels = parse_label_file(label_path)

    print("  Loading source_tweets.txt ...")
    sources = parse_source_tweets(source_path)

    # ── determine valid claim ids ─────────────────────────────────────────
    # a claim is valid if it has a label AND a tree file
    tree_files = {
        os.path.splitext(f)[0]
        for f in os.listdir(tree_dir)
        if f.endswith(".txt")
    }
    valid_ids = [cid for cid in labels if cid in tree_files]
    print(f"  Valid claims (labeled + has tree): {len(valid_ids)}")

    # ── per-claim scan ────────────────────────────────────────────────────
    label_counts_raw    = {}     # raw label string -> count
    binary_counts       = {0: 0, 1: 0}

    node_counts         = []     # one entry per valid claim
    edge_counts         = []
    max_timestamps      = []     # per-claim max timestamp
    tree_depths         = []     # per-claim max tree depth

    total_violations    = 0
    total_duplicates    = 0
    missing_source_text = 0

    print(f"  Scanning {len(valid_ids)} tree files ...")

    for i, claim_id in enumerate(valid_ids):
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(valid_ids)} ...")

        # label
        label_str = labels[claim_id]
        label_counts_raw[label_str] = label_counts_raw.get(label_str, 0) + 1
        binary_label = map_label_to_binary(label_str)
        binary_counts[binary_label] += 1

        # source text coverage
        if claim_id not in sources or sources[claim_id] == "":
            missing_source_text += 1

        # tree
        tree_path = os.path.join(tree_dir, claim_id + ".txt")
        try:
            result = parse_tree_file(tree_path)
        except Exception as ex:
            print(f"  [ERROR] Failed to parse {claim_id}: {ex}")
            continue

        nodes  = result["nodes"]
        edges  = result["edges"]
        n      = len(nodes)
        e      = len(edges)

        node_counts.append(n)
        edge_counts.append(e)
        total_violations += result["num_violations"]
        total_duplicates += result["num_duplicates"]

        # max timestamp across all nodes
        max_t = max((nd["timestamp"] for nd in nodes), default=0.0)
        max_timestamps.append(max_t)

        # tree depth via BFS from ROOT (node index 0)
        depth = _compute_depth(nodes, edges)
        tree_depths.append(depth)

    # ── sort for percentile computation ───────────────────────────────────
    node_counts.sort()
    edge_counts.sort()
    max_timestamps.sort()
    tree_depths.sort()

    # ── build stats dict ──────────────────────────────────────────────────
    stats = {
        "dataset"              : dataset_name,
        "total_valid_claims"   : len(valid_ids),
        "missing_source_text"  : missing_source_text,

        "label_distribution_raw": label_counts_raw,
        "label_distribution_binary": {
            "credible"       : binary_counts[0],
            "misinformation" : binary_counts[1],
        },

        "tree_size_nodes": {
            "min"    : node_counts[0]                    if node_counts else 0,
            "p25"    : percentile(node_counts, 25),
            "median" : percentile(node_counts, 50),
            "p75"    : percentile(node_counts, 75),
            "max"    : node_counts[-1]                   if node_counts else 0,
        },

        "tree_size_edges": {
            "min"    : edge_counts[0]                    if edge_counts else 0,
            "median" : percentile(edge_counts, 50),
            "max"    : edge_counts[-1]                   if edge_counts else 0,
        },

        "max_timestamp_minutes": {
            "min"    : round(max_timestamps[0], 4)       if max_timestamps else 0,
            "p25"    : round(percentile(max_timestamps, 25), 4),
            "median" : round(percentile(max_timestamps, 50), 4),
            "p75"    : round(percentile(max_timestamps, 75), 4),
            "max"    : round(max_timestamps[-1], 4)      if max_timestamps else 0,
        },

        "tree_depth": {
            "min"    : tree_depths[0]                    if tree_depths else 0,
            "median" : percentile(tree_depths, 50),
            "max"    : tree_depths[-1]                   if tree_depths else 0,
        },

        # these are the values used for feature normalization in build_graphs.py
        "normalization": {
            "global_max_tree_depth" : tree_depths[-1] if tree_depths else 1,
            # timestamp is normalized per-claim (divide by own max_time)
            # no global timestamp normalization needed
            "timestamp_unit"        : "minutes",
            "timestamp_strategy"    : "per_claim_max",
        },

        "data_quality": {
            "total_monotonicity_violations" : total_violations,
            "total_duplicate_child_userids" : total_duplicates,
            "avg_violations_per_claim"      : round(total_violations / max(len(valid_ids), 1), 3),
            "avg_duplicates_per_claim"      : round(total_duplicates / max(len(valid_ids), 1), 3),
        },
    }

    # ── write json ────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # ── print summary ─────────────────────────────────────────────────────
    print()
    print(f"  Label distribution (raw):")
    for lbl, cnt in sorted(stats["label_distribution_raw"].items()):
        print(f"    {lbl:<15} : {cnt}")
    print(f"  Binary  ->  credible={binary_counts[0]}  misinfo={binary_counts[1]}")
    print()
    print(f"  Tree size (nodes)  :  "
          f"min={stats['tree_size_nodes']['min']}  "
          f"median={stats['tree_size_nodes']['median']}  "
          f"max={stats['tree_size_nodes']['max']}")
    print(f"  Max timestamp (min):  "
          f"min={stats['max_timestamp_minutes']['min']}  "
          f"median={stats['max_timestamp_minutes']['median']}  "
          f"max={stats['max_timestamp_minutes']['max']}")
    print(f"  Tree depth         :  "
          f"min={stats['tree_depth']['min']}  "
          f"median={stats['tree_depth']['median']}  "
          f"max={stats['tree_depth']['max']}")
    print()
    print(f"  global_max_tree_depth : {stats['normalization']['global_max_tree_depth']}")
    print(f"  missing source text   : {missing_source_text}")
    print(f"  monotonicity viol.    : {total_violations}")
    print(f"  duplicate child uids  : {total_duplicates}")
    print()
    print(f"  Written: {out_path}")

    return stats


# ── tree depth helper ─────────────────────────────────────────────────────────

def _compute_depth(nodes, edges):
    """
    Computes the maximum depth of the propagation tree via BFS from ROOT.
    ROOT is always node index 0.

    Args:
        nodes : list of node dicts (with 'node_idx')
        edges : list of edge dicts (with 'src_idx', 'dst_idx')

    Returns:
        int  maximum depth (ROOT is depth 0)
    """
    from collections import defaultdict, deque

    children = defaultdict(list)
    for e in edges:
        children[e["src_idx"]].append(e["dst_idx"])

    # BFS
    queue    = deque([(0, 0)])   # (node_idx, depth)
    visited  = {0}
    max_depth = 0

    while queue:
        node_idx, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        for child_idx in children[node_idx]:
            if child_idx not in visited:
                visited.add(child_idx)
                queue.append((child_idx, depth + 1))

    return max_depth


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run stats scan on a dataset.")
    parser.add_argument(
        "--dataset",
        choices=["twitter15", "twitter16", "both"],
        default="both",
        help="Which dataset to scan (default: both)",
    )
    args = parser.parse_args()

    if args.dataset in ("twitter15", "both"):
        run_stats_scan("twitter15")

    if args.dataset in ("twitter16", "both"):
        run_stats_scan("twitter16")

    print("\nStats scan complete.")


if __name__ == "__main__":
    main()

"""
Step 3 Output Analysis
The scan completed cleanly. Three things to note before moving forward:
The violation and duplicate counts are higher than the inspection script reported (4058 vs 3659 violations, 8110 vs 6346 duplicates). This is expected — the inspection script counted raw edge-level violations while stats_scan.py counts through the parser which applies the index-based strategy, catching additional cases during parent registration.
global_max_tree_depth is 27 for Twitter15 and 26 for Twitter16. Since you will eventually train on Twitter15 and evaluate transfer on Twitter16, use 27 as the normalization constant for tree depth across both datasets. This needs to be noted.
All 1490 source tweets are present with non-empty text. The text embedding step will have full coverage.
Both dataset_stats.json files are written correctly. You are ready for Step 4.
"""
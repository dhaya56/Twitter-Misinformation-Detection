"""
Dataset inspection script for rumdetect2017 format.
Run this once before any preprocessing.
Prints structural statistics for twitter15 and twitter16.

Usage:
    python backend/data/inspect_dataset.py
Run from: D:\vit_projects\dl_project
"""

import os
import re
from collections import defaultdict


# ── paths ────────────────────────────────────────────────────────────────────

DATASET_ROOT = os.path.join("dataset", "raw")
DATASETS = ["twitter15", "twitter16"]


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_label_file(path):
    """
    Returns dict: {claim_id (str) -> label (str)}
    Label file format:  label:claim_id
    """
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                print(f"  [WARN] Unexpected label line format: {line!r}")
                continue
            label, claim_id = line.split(":", 1)
            labels[claim_id.strip()] = label.strip()
    return labels


def parse_source_tweets(path):
    """
    Returns dict: {claim_id (str) -> tweet_text (str)}
    Source tweets format: claim_id<whitespace>tweet text
    """
    sources = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)   # split on first whitespace
            if len(parts) < 2:
                claim_id = parts[0]
                sources[claim_id] = ""
            else:
                sources[parts[0]] = parts[1]
    return sources


def parse_tree_file(path):
    """
    Returns list of (parent_uid, child_uid, parent_time, child_time).
    Handles the ROOT token.
    """
    edges = []
    pattern = re.compile(
        r"\['([^']+)',\s*'([^']+)',\s*'([^']+)'\]"
        r"->"
        r"\['([^']+)',\s*'([^']+)',\s*'([^']+)'\]"
    )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if not m:
                print(f"  [WARN] Unmatched line in {os.path.basename(path)}: {line!r}")
                continue
            p_uid, p_tid, p_time, c_uid, c_tid, c_time = m.groups()
            edges.append((p_uid, c_uid, float(p_time), float(c_time)))
    return edges


def get_unique_nodes(edges):
    """Returns set of all unique user_ids in the tree."""
    nodes = set()
    for p_uid, c_uid, _, _ in edges:
        nodes.add(p_uid)
        nodes.add(c_uid)
    return nodes


def get_max_timestamp(edges):
    """Returns the maximum child timestamp across all edges."""
    if not edges:
        return 0.0
    return max(c_time for _, _, _, c_time in edges)


def compute_tree_depth(edges):
    """
    Computes maximum depth of the tree via BFS from ROOT.
    Returns max_depth (int).
    """
    if not edges:
        return 0

    # build adjacency list (parent -> children)
    children = defaultdict(list)
    for p_uid, c_uid, _, _ in edges:
        children[p_uid].append(c_uid)

    # BFS from ROOT
    queue = [("ROOT", 0)]
    visited = {"ROOT"}
    max_depth = 0

    while queue:
        node, depth = queue.pop(0)
        max_depth = max(max_depth, depth)
        for child in children.get(node, []):
            if child not in visited:
                visited.add(child)
                queue.append((child, depth + 1))

    return max_depth


def check_timestamp_monotonicity(edges):
    """
    Checks if any edge has child_time < parent_time.
    Returns count of violations.
    """
    violations = 0
    for _, _, p_time, c_time in edges:
        if c_time < p_time:
            violations += 1
    return violations


def check_duplicate_children(edges):
    """
    Checks if any user_id appears as a child more than once.
    Returns count of duplicate child user_ids.
    """
    child_counts = defaultdict(int)
    for _, c_uid, _, _ in edges:
        child_counts[c_uid] += 1
    return sum(1 for v in child_counts.values() if v > 1)


# ── main inspection ───────────────────────────────────────────────────────────

def inspect_dataset(dataset_name):
    base = os.path.join(DATASET_ROOT, dataset_name)
    label_path  = os.path.join(base, "label.txt")
    source_path = os.path.join(base, "source_tweets.txt")
    tree_dir    = os.path.join(base, "tree")

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    # ── check paths ──
    for p, name in [(label_path, "label.txt"),
                    (source_path, "source_tweets.txt"),
                    (tree_dir,   "tree/")]:
        exists = os.path.exists(p)
        print(f"  {'[OK]' if exists else '[MISSING]'} {name}")
    print()

    if not all(os.path.exists(p) for p, _ in
               [(label_path, ""), (source_path, ""), (tree_dir, "")]):
        print("  Cannot proceed — missing files.\n")
        return

    # ── labels ──
    labels = parse_label_file(label_path)
    label_counts = defaultdict(int)
    for v in labels.values():
        label_counts[v] += 1

    print(f"  label.txt")
    print(f"    Total labeled claims : {len(labels)}")
    print(f"    Label distribution   :")
    for label, count in sorted(label_counts.items()):
        print(f"      {label:<15} : {count}")
    print()

    # binary mapping
    credible    = sum(1 for v in labels.values() if v in ("non-rumor", "true"))
    misinfo     = sum(1 for v in labels.values() if v in ("false", "unverified"))
    print(f"    Binary mapping (credible / misinformation):")
    print(f"      credible       : {credible}  (non-rumor + true)")
    print(f"      misinformation : {misinfo}  (false + unverified)")
    print()

    # ── source tweets ──
    sources = parse_source_tweets(source_path)
    covered = sum(1 for cid in labels if cid in sources)
    empty   = sum(1 for cid in labels if cid in sources and sources[cid] == "")

    print(f"  source_tweets.txt")
    print(f"    Total entries        : {len(sources)}")
    print(f"    Covered by labels    : {covered} / {len(labels)}")
    print(f"    Empty text entries   : {empty}")
    print()

    # ── tree files ──
    tree_files = [f for f in os.listdir(tree_dir) if f.endswith(".txt")]
    tree_ids   = {os.path.splitext(f)[0] for f in tree_files}
    label_ids  = set(labels.keys())

    print(f"  tree/")
    print(f"    Total tree files     : {len(tree_files)}")
    print(f"    Tree IDs in labels   : {len(tree_ids & label_ids)}")
    print(f"    Tree IDs NOT labeled : {len(tree_ids - label_ids)}")
    print(f"    Labels w/o tree file : {len(label_ids - tree_ids)}")
    print()

    # ── per-tree statistics (only labeled trees) ──
    valid_ids = tree_ids & label_ids
    print(f"  Computing per-tree statistics for {len(valid_ids)} labeled trees...")

    node_counts   = []
    edge_counts   = []
    max_times     = []
    depths        = []
    mono_violations = 0
    dup_children    = 0
    single_node_trees = 0
    trees_below_5   = 0

    for claim_id in valid_ids:
        path  = os.path.join(tree_dir, claim_id + ".txt")
        edges = parse_tree_file(path)

        nodes     = get_unique_nodes(edges)
        n_nodes   = len(nodes)
        max_t     = get_max_timestamp(edges)
        depth     = compute_tree_depth(edges)
        mono_viol = check_timestamp_monotonicity(edges)
        dup_child = check_duplicate_children(edges)

        node_counts.append(n_nodes)
        edge_counts.append(len(edges))
        max_times.append(max_t)
        depths.append(depth)
        mono_violations += mono_viol
        dup_children    += dup_child

        if n_nodes == 1:
            single_node_trees += 1
        if n_nodes < 5:
            trees_below_5 += 1

    node_counts.sort()
    edge_counts.sort()
    max_times.sort()
    depths.sort()
    n = len(node_counts)

    def pct(lst, p):
        idx = int(len(lst) * p / 100)
        return lst[min(idx, len(lst)-1)]

    print()
    print(f"  Tree size (nodes per tree):")
    print(f"    Min     : {node_counts[0]}")
    print(f"    P25     : {pct(node_counts, 25)}")
    print(f"    Median  : {pct(node_counts, 50)}")
    print(f"    P75     : {pct(node_counts, 75)}")
    print(f"    Max     : {node_counts[-1]}")
    print()
    print(f"  Max timestamp per tree (minutes):")
    print(f"    Min     : {max_times[0]:.2f}")
    print(f"    P25     : {pct(max_times, 25):.2f}")
    print(f"    Median  : {pct(max_times, 50):.2f}")
    print(f"    P75     : {pct(max_times, 75):.2f}")
    print(f"    Max     : {max_times[-1]:.2f}")
    print()
    print(f"  Tree depth (hops from ROOT):")
    print(f"    Min     : {depths[0]}")
    print(f"    Median  : {pct(depths, 50)}")
    print(f"    Max     : {depths[-1]}")
    print()
    print(f"  Data quality checks:")
    print(f"    Timestamp monotonicity violations : {mono_violations}")
    print(f"    Duplicate child user_ids          : {dup_children}")
    print(f"    Trees with exactly 1 node         : {single_node_trees}")
    print(f"    Trees with fewer than 5 nodes     : {trees_below_5}")
    print(f"    Trees surviving min_nodes=5 filter: {n - trees_below_5}")
    print()


def main():
    print("\nCGDEX-Net Dataset Inspection")
    print("Raw dataset root:", os.path.abspath(DATASET_ROOT))

    for ds in DATASETS:
        ds_path = os.path.join(DATASET_ROOT, ds)
        if os.path.exists(ds_path):
            inspect_dataset(ds)
        else:
            print(f"\n[SKIP] {ds} not found at {ds_path}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
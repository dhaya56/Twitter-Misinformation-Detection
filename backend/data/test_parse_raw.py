"""
Quick test for parse_raw.py.
Run from: D:\vit_projects\dl_project
Usage: python backend/data/test_parse_raw.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

from backend.data.parse_raw import (
    parse_label_file,
    parse_source_tweets,
    parse_tree_file,
    map_label_to_binary,
)

BASE = os.path.join("dataset", "raw", "twitter15")

# ── test label parsing ────────────────────────────────────────────────────────
print("=" * 50)
print("TEST 1: parse_label_file")
labels = parse_label_file(os.path.join(BASE, "label.txt"))
print(f"  Total labels parsed : {len(labels)}")
# print first 3
for i, (cid, lbl) in enumerate(list(labels.items())[:3]):
    print(f"  {cid} -> '{lbl}' (binary={map_label_to_binary(lbl)})")
assert len(labels) == 1490, f"Expected 1490, got {len(labels)}"
print("  PASSED")

# ── test source tweet parsing ─────────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 2: parse_source_tweets")
sources = parse_source_tweets(os.path.join(BASE, "source_tweets.txt"))
print(f"  Total source tweets : {len(sources)}")
for i, (cid, text) in enumerate(list(sources.items())[:3]):
    preview = text[:60] + "..." if len(text) > 60 else text
    print(f"  {cid} -> '{preview}'")
assert len(sources) == 1490, f"Expected 1490, got {len(sources)}"
print("  PASSED")

# ── test tree parsing on known file ──────────────────────────────────────────
print()
print("=" * 50)
print("TEST 3: parse_tree_file (80080680482123777.txt)")
tree_path = os.path.join(BASE, "tree", "80080680482123777.txt")
result = parse_tree_file(tree_path)

print(f"  claim_id       : {result['claim_id']}")
print(f"  num_nodes      : {len(result['nodes'])}")
print(f"  num_edges      : {len(result['edges'])}")
print(f"  num_violations : {result['num_violations']}  (clamped)")
print(f"  num_duplicates : {result['num_duplicates']}  (duplicate child user_ids)")

# check ROOT is node 0
root = result["nodes"][0]
assert root["node_idx"] == 0,    "ROOT must be node index 0"
assert root["user_id"]  == "ROOT", "First node must be ROOT"
assert root["is_root"]  == True,  "ROOT node must have is_root=True"
assert root["timestamp"] == 0.0, "ROOT timestamp must be 0.0"
print(f"  Root node      : idx=0, user_id=ROOT, time=0.0  [OK]")

# check all node indices are unique and contiguous
all_indices = [n["node_idx"] for n in result["nodes"]]
assert all_indices == list(range(len(result["nodes"]))), \
    "Node indices must be contiguous 0..N-1"
print(f"  Node indices   : contiguous 0..{len(result['nodes'])-1}  [OK]")

# check all timestamps non-negative
for n in result["nodes"]:
    assert n["timestamp"] >= 0.0, f"Negative timestamp on node {n['node_idx']}"
print(f"  All timestamps : non-negative  [OK]")

# check edge src/dst indices are valid
n_nodes = len(result["nodes"])
for e in result["edges"]:
    assert 0 <= e["src_idx"] < n_nodes, f"Invalid src_idx {e['src_idx']}"
    assert 0 <= e["dst_idx"] < n_nodes, f"Invalid dst_idx {e['dst_idx']}"
print(f"  All edge indices : valid  [OK]")

# check monotonicity holds after clamping
for e in result["edges"]:
    assert e["dst_time"] >= e["src_time"], \
        f"Monotonicity violated after clamping: {e}"
print(f"  All edge timestamps : monotonic after clamping  [OK]")

# print first 3 nodes and edges
print()
print("  First 3 nodes:")
for n in result["nodes"][:3]:
    print(f"    {n}")
print("  First 3 edges:")
for e in result["edges"][:3]:
    print(f"    {e}")

print("  PASSED")

# ── test on 5 random trees ────────────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 4: parse_tree_file on 5 different claims")
import random
random.seed(42)
tree_dir = os.path.join(BASE, "tree")
tree_files = [f for f in os.listdir(tree_dir) if f.endswith(".txt")]
sample = random.sample(tree_files, 5)

for fname in sample:
    path   = os.path.join(tree_dir, fname)
    result = parse_tree_file(path)
    n      = len(result["nodes"])
    e      = len(result["edges"])
    # in a tree: edges = nodes - 1
    # with index-based strategy: edges == nodes - 1 still holds
    # because each edge creates exactly one new child node
    assert e == n - 1, \
        f"{fname}: expected edges={n-1}, got {e}"
    print(f"  {fname:<35} nodes={n:<5} edges={e:<5} "
          f"viol={result['num_violations']:<3} "
          f"dup={result['num_duplicates']}")

print("  PASSED (edges == nodes - 1 for all trees)")

print()
print("All tests passed.")
"""
make_splits.py
--------------
Generates stratified train/val/test split index files for Twitter15.
Split is stratified on the binary label to preserve class balance
in each subset.

Writes three files to dataset/raw/twitter15/splits/:
    train_ids.txt
    val_ids.txt
    test_ids.txt

Each file contains one claim_id per line.

Twitter16 is NOT split here — it is used only as a
cross-dataset evaluation set (all 818 claims = test set).
A single file is written for it:
    dataset/raw/twitter16/splits/test_ids.txt

Split ratios (Twitter15):
    Train : 0.70  -> ~1043 claims
    Val   : 0.10  ->  ~149 claims
    Test  : 0.20  ->  ~298 claims

Usage:
    python backend/data/make_splits.py
Run from: D:\vit_projects\dl_project
"""

import os
import sys
import random
import json
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))

from backend.data.parse_raw import parse_label_file, map_label_to_binary


# ── config ────────────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20
SEED        = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def write_ids(path, ids):
    """Writes a list of claim_ids to a text file, one per line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cid in ids:
            f.write(cid + "\n")
    print(f"  Written: {path}  ({len(ids)} ids)")


def read_ids(path):
    """Reads claim_ids from a split file."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def stratified_split(ids_by_class, train_r, val_r, seed):
    """
    Performs stratified splitting.

    Args:
        ids_by_class : dict {class_label -> list of claim_ids}
        train_r      : float  train ratio
        val_r        : float  val ratio
        seed         : int    random seed

    Returns:
        (train_ids, val_ids, test_ids)  each a list of claim_ids
    """
    rng = random.Random(seed)

    train_ids = []
    val_ids   = []
    test_ids  = []

    for label, ids in ids_by_class.items():
        ids_shuffled = list(ids)
        rng.shuffle(ids_shuffled)

        n       = len(ids_shuffled)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)
        # test gets the remainder to avoid rounding loss
        n_test  = n - n_train - n_val

        train_ids.extend(ids_shuffled[:n_train])
        val_ids.extend(ids_shuffled[n_train: n_train + n_val])
        test_ids.extend(ids_shuffled[n_train + n_val:])

    # shuffle the combined splits so classes are interleaved
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    return train_ids, val_ids, test_ids


def verify_split(train_ids, val_ids, test_ids, labels):
    """Prints split statistics and verifies no overlap between splits."""
    all_splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    print()
    print(f"  {'Split':<8} {'Total':>6}  {'credible':>9}  {'misinfo':>8}")
    print(f"  {'-'*40}")

    for split_name, ids in all_splits.items():
        c = sum(1 for i in ids if map_label_to_binary(labels[i]) == 0)
        m = sum(1 for i in ids if map_label_to_binary(labels[i]) == 1)
        print(f"  {split_name:<8} {len(ids):>6}  {c:>9}  {m:>8}")

    # overlap check
    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set   & test_set

    print()
    if tv or tt or vt:
        print(f"  [ERROR] Overlap detected:")
        if tv: print(f"    train ∩ val  : {len(tv)}")
        if tt: print(f"    train ∩ test : {len(tt)}")
        if vt: print(f"    val ∩ test   : {len(vt)}")
    else:
        print(f"  No overlap between splits.  [OK]")

    # coverage check
    total = len(train_ids) + len(val_ids) + len(test_ids)
    print(f"  Total covered: {total}")


# ── twitter15 split ───────────────────────────────────────────────────────────

def make_twitter15_splits():
    raw_dir    = os.path.join("dataset", "raw", "twitter15")
    splits_dir = os.path.join(raw_dir, "splits")
    graphs_dir = os.path.join("dataset", "processed", "twitter15", "graphs")

    print(f"\n{'='*60}")
    print(f"  Splits: TWITTER15")
    print(f"{'='*60}")

    # load labels
    labels = parse_label_file(os.path.join(raw_dir, "label.txt"))

    # only include claims that have a processed graph file
    available = {
        os.path.splitext(f)[0].replace("claim_", "")
        for f in os.listdir(graphs_dir)
        if f.endswith(".pt")
    }
    valid_ids = [cid for cid in labels if cid in available]
    print(f"  Claims with processed graphs : {len(valid_ids)}")

    # group by binary label for stratification
    ids_by_class = defaultdict(list)
    for cid in valid_ids:
        binary = map_label_to_binary(labels[cid])
        ids_by_class[binary].append(cid)

    print(f"  Class 0 (credible)      : {len(ids_by_class[0])}")
    print(f"  Class 1 (misinformation): {len(ids_by_class[1])}")

    # generate splits
    train_ids, val_ids, test_ids = stratified_split(
        ids_by_class, TRAIN_RATIO, VAL_RATIO, SEED
    )

    # verify
    verify_split(train_ids, val_ids, test_ids, labels)

    # write files
    print()
    write_ids(os.path.join(splits_dir, "train_ids.txt"), train_ids)
    write_ids(os.path.join(splits_dir, "val_ids.txt"),   val_ids)
    write_ids(os.path.join(splits_dir, "test_ids.txt"),  test_ids)

    # write split metadata
    meta = {
        "dataset"    : "twitter15",
        "seed"       : SEED,
        "ratios"     : {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "counts"     : {
            "train": len(train_ids),
            "val"  : len(val_ids),
            "test" : len(test_ids),
        },
        "note": (
            "Stratified split on binary label. "
            "Use these files for all experiments to ensure reproducibility."
        ),
    }
    meta_path = os.path.join(splits_dir, "split_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Written: {meta_path}")


# ── twitter16 test set ────────────────────────────────────────────────────────

def make_twitter16_test():
    raw_dir    = os.path.join("dataset", "raw", "twitter16")
    splits_dir = os.path.join(raw_dir, "splits")
    graphs_dir = os.path.join("dataset", "processed", "twitter16", "graphs")

    print(f"\n{'='*60}")
    print(f"  Splits: TWITTER16  (cross-dataset evaluation only)")
    print(f"{'='*60}")

    labels = parse_label_file(os.path.join(raw_dir, "label.txt"))

    available = {
        os.path.splitext(f)[0].replace("claim_", "")
        for f in os.listdir(graphs_dir)
        if f.endswith(".pt")
    }
    valid_ids = sorted(cid for cid in labels if cid in available)
    print(f"  Claims with processed graphs : {len(valid_ids)}")

    c = sum(1 for cid in valid_ids if map_label_to_binary(labels[cid]) == 0)
    m = sum(1 for cid in valid_ids if map_label_to_binary(labels[cid]) == 1)
    print(f"  Class 0 (credible)           : {c}")
    print(f"  Class 1 (misinformation)     : {m}")
    print()

    # all claims are the test set for twitter16
    write_ids(os.path.join(splits_dir, "test_ids.txt"), valid_ids)

    meta = {
        "dataset" : "twitter16",
        "seed"    : SEED,
        "note"    : (
            "Twitter16 is used only for cross-dataset transfer evaluation. "
            "All claims are treated as the test set. "
            "No train/val split is generated."
        ),
        "counts"  : {"test": len(valid_ids)},
    }
    meta_path = os.path.join(splits_dir, "split_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Written: {meta_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    make_twitter15_splits()
    make_twitter16_test()

    print("\nSplit generation complete.")
    print()
    print("Files written:")
    print("  dataset/raw/twitter15/splits/train_ids.txt")
    print("  dataset/raw/twitter15/splits/val_ids.txt")
    print("  dataset/raw/twitter15/splits/test_ids.txt")
    print("  dataset/raw/twitter15/splits/split_meta.json")
    print("  dataset/raw/twitter16/splits/test_ids.txt")
    print("  dataset/raw/twitter16/splits/split_meta.json")


if __name__ == "__main__":
    main()
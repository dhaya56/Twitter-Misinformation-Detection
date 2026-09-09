"""
embed_texts.py
--------------
Encodes all source tweet texts using a sentence transformer.
Produces text_embeddings.pt in the processed directory.

Output file format:
    dict  {claim_id (str) -> FloatTensor [384]}
    Saved with torch.save()

Claims with empty or missing text receive a zero vector
and are flagged in the output summary.

Usage:
    python backend/data/embed_texts.py --dataset twitter15
    python backend/data/embed_texts.py --dataset twitter16
    python backend/data/embed_texts.py --dataset both
Run from: D:\vit_projects\dl_project
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath("."))

import torch
from sentence_transformers import SentenceTransformer

from backend.data.parse_raw import parse_label_file, parse_source_tweets


# ── config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "paraphrase-MiniLM-L6-v2"   # produces 384-dim embeddings
EMBED_DIM    = 384
BATCH_SIZE   = 64                           # safe for CPU; increase if using GPU


# ── main embedding function ───────────────────────────────────────────────────

def embed_dataset(dataset_name, model):
    """
    Encodes all source tweets for one dataset.

    Args:
        dataset_name : 'twitter15' or 'twitter16'
        model        : loaded SentenceTransformer instance

    Writes:
        dataset/processed/{dataset_name}/text_embeddings.pt
    """
    raw_dir       = os.path.join("dataset", "raw",       dataset_name)
    processed_dir = os.path.join("dataset", "processed", dataset_name)
    stats_path    = os.path.join(processed_dir, "dataset_stats.json")
    out_path      = os.path.join(processed_dir, "text_embeddings.pt")

    print(f"\n{'='*60}")
    print(f"  Embedding: {dataset_name.upper()}")
    print(f"{'='*60}")

    # ── verify stats file exists ──────────────────────────────────────────
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"dataset_stats.json not found at {stats_path}\n"
            f"Run stats_scan.py first."
        )

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    total_claims = stats["total_valid_claims"]
    print(f"  Expected claims: {total_claims}")

    # ── load label and source data ────────────────────────────────────────
    labels  = parse_label_file(os.path.join(raw_dir, "label.txt"))
    sources = parse_source_tweets(os.path.join(raw_dir, "source_tweets.txt"))

    # ── determine valid claim ids (same filter as stats_scan) ────────────
    tree_dir  = os.path.join(raw_dir, "tree")
    tree_ids  = {os.path.splitext(f)[0]
                 for f in os.listdir(tree_dir) if f.endswith(".txt")}
    valid_ids = [cid for cid in labels if cid in tree_ids]
    print(f"  Valid claims found: {len(valid_ids)}")

    # ── collect texts in deterministic order ─────────────────────────────
    # sort for reproducibility across runs
    valid_ids_sorted = sorted(valid_ids)

    texts        = []
    zero_ids     = []   # claim_ids with missing/empty text

    for cid in valid_ids_sorted:
        text = sources.get(cid, "").strip()
        if not text:
            texts.append("")
            zero_ids.append(cid)
        else:
            texts.append(text)

    print(f"  Claims with empty text: {len(zero_ids)}")
    if zero_ids:
        print(f"  (these will receive zero embedding vectors)")

    # ── encode in batches ─────────────────────────────────────────────────
    print(f"  Encoding {len(texts)} texts in batches of {BATCH_SIZE} ...")
    print(f"  Model: {MODEL_NAME}")

    embeddings_list = []
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(texts))
        batch = texts[start:end]

        # replace empty strings with a neutral placeholder for encoding
        # then we will zero out those vectors afterward
        batch_input = [
            t if t else "empty"
            for t in batch
        ]

        batch_emb = model.encode(
            batch_input,
            convert_to_tensor=True,
            show_progress_bar=False,
        )                            # FloatTensor [batch_size, 384]

        # zero out vectors for claims that had empty text
        for local_idx in range(len(batch)):
            global_idx = start + local_idx
            if texts[global_idx] == "":
                batch_emb[local_idx] = torch.zeros(EMBED_DIM)

        embeddings_list.append(batch_emb.cpu())

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == n_batches:
            print(f"    Batch {batch_idx+1}/{n_batches} done "
                  f"({end}/{len(texts)} texts)")

    # ── assemble final dict ───────────────────────────────────────────────
    all_embeddings = torch.cat(embeddings_list, dim=0)   # [N, 384]

    assert all_embeddings.shape == (len(valid_ids_sorted), EMBED_DIM), (
        f"Shape mismatch: got {all_embeddings.shape}, "
        f"expected ({len(valid_ids_sorted)}, {EMBED_DIM})"
    )

    embedding_dict = {}
    for i, cid in enumerate(valid_ids_sorted):
        embedding_dict[cid] = all_embeddings[i]         # FloatTensor [384]

    # ── verify zero vectors for empty-text claims ─────────────────────────
    for cid in zero_ids:
        assert torch.all(embedding_dict[cid] == 0.0), \
            f"Expected zero vector for {cid}"

    # ── save ──────────────────────────────────────────────────────────────
    torch.save(embedding_dict, out_path)

    # ── print summary ─────────────────────────────────────────────────────
    sample_id  = valid_ids_sorted[0]
    sample_emb = embedding_dict[sample_id]
    print()
    print(f"  Total embeddings saved : {len(embedding_dict)}")
    print(f"  Embedding dimension    : {sample_emb.shape[0]}")
    print(f"  Dtype                  : {sample_emb.dtype}")
    print(f"  Sample claim_id        : {sample_id}")
    print(f"  Sample emb norm        : {sample_emb.norm().item():.4f}")
    print(f"  Zero vectors           : {len(zero_ids)}")
    print(f"  Written: {out_path}")

    return embedding_dict


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Encode source tweets with sentence transformer."
    )
    parser.add_argument(
        "--dataset",
        choices=["twitter15", "twitter16", "both"],
        default="both",
        help="Which dataset to encode (default: both)",
    )
    args = parser.parse_args()

    # load model once — shared across both datasets
    print(f"\nLoading model: {MODEL_NAME}")
    print("(First run will download ~90MB model weights)")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Output dimension: {model.get_sentence_embedding_dimension()}")

    if args.dataset in ("twitter15", "both"):
        embed_dataset("twitter15", model)

    if args.dataset in ("twitter16", "both"):
        embed_dataset("twitter16", model)

    print("\nEmbedding complete.")


if __name__ == "__main__":
    main()
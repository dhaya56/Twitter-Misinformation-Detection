"""
phase2_diffusion.py
-------------------
Phase 2: Credibility Diffusion Prior Training.

Trains CDP on latent vectors of CREDIBLE samples only.
STE and VLE are loaded from Phase 1 checkpoint and frozen.

The CDP learns p_theta(z | y=0) — the distribution of
credible propagation latents. This defines the credibility
manifold M_c used for detection at inference time.

Procedure:
    1. Load STE + VLE from phase1_best.pt (frozen)
    2. Pre-encode all credible training samples -> latent cache
    3. Train CDP with DDPM loss on the cached latent vectors
       (no graph data needed after encoding)

Pre-encoding and caching latents before the training loop
makes Phase 2 very fast since the CDP trains on fixed vectors.

Usage:
    python backend/training/phase2_diffusion.py
Run from: D:\vit_projects\dl_project
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader

from backend.data.dataset_loader import PropagationDataset
from backend.models.tpgb import TPGB
from backend.models.ste import STE
from backend.models.vle import VLE
from backend.models.cdp import CDP


# ── config ────────────────────────────────────────────────────────────────────

CFG = {
    # data
    "dataset"          : "twitter15",
    "batch_size_encode": 32,    # batch size for encoding pass
    "batch_size_train" : 128,   # batch size for diffusion training
    "num_workers"      : 0,

    # model dimensions (must match Phase 1)
    "node_feat_dim"    : 6,
    "edge_feat_dim"    : 3,
    "gat_hidden_dim"   : 64,
    "gat_num_heads"    : 4,
    "gat_out_dim"      : 96,
    "pvt_out_dim"      : 32,
    "snapshot_embed_dim": 128,
    "transformer_heads" : 4,
    "transformer_layers": 2,
    "dropout"          : 0.1,
    "text_embed_dim"   : 384,
    "text_proj_dim"    : 64,
    "latent_dim"       : 128,

    # diffusion
    "T"                : 200,
    "T_inf"            : 20,
    "t_star"           : 40,
    "hidden_dim"       : 256,

    # training
    "epochs"           : 20,
    "lr"               : 1e-3,
    "weight_decay"     : 1e-5,
    "grad_clip"        : 1.0,

    # paths
    "checkpoint_dir"   : "backend/checkpoints",
    "phase1_ckpt"      : "backend/checkpoints/phase1_best.pt",
    "checkpoint_name"  : "phase2_best.pt",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── encoder loading ───────────────────────────────────────────────────────────

def load_frozen_encoder(cfg, device):
    """
    Loads STE and VLE from Phase 1 checkpoint.
    Sets both to eval mode and freezes all parameters.

    Returns:
        tpgb, ste, vle  (all frozen, on device)
    """
    ckpt = torch.load(cfg["phase1_ckpt"], map_location=device,
                      weights_only=False)

    tpgb = TPGB(num_snapshots=5).to(device)

    ste = STE(
        node_feat_dim      = cfg["node_feat_dim"],
        edge_feat_dim      = cfg["edge_feat_dim"],
        gat_hidden_dim     = cfg["gat_hidden_dim"],
        gat_num_heads      = cfg["gat_num_heads"],
        gat_out_dim        = cfg["gat_out_dim"],
        pvt_out_dim        = cfg["pvt_out_dim"],
        snapshot_embed_dim = cfg["snapshot_embed_dim"],
        transformer_heads  = cfg["transformer_heads"],
        transformer_layers = cfg["transformer_layers"],
        dropout            = cfg["dropout"],
    ).to(device)
    ste.load_state_dict(ckpt["ste_state"])

    vle = VLE(
        graph_embed_dim = cfg["snapshot_embed_dim"],
        text_embed_dim  = cfg["text_embed_dim"],
        text_proj_dim   = cfg["text_proj_dim"],
        latent_dim      = cfg["latent_dim"],
        dropout         = cfg["dropout"],
    ).to(device)
    vle.load_state_dict(ckpt["vle_state"])

    # freeze
    for model in (tpgb, ste, vle):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    print(f"  Loaded Phase 1 checkpoint (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f})")
    return tpgb, ste, vle


# ── latent caching ────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_credible_samples(tpgb, ste, vle, dataset_name, device, batch_size):
    """
    Encodes all credible training samples and returns their
    latent means mu_z as a cached tensor.

    Uses mu_z (not sampled z) for stable diffusion training targets.

    Returns:
        FloatTensor [N_credible, latent_dim]
    """
    ds = PropagationDataset(dataset_name, split="train")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    credible_latents = []
    total_credible   = 0
    total_skipped    = 0

    for batch in loader:
        batch = batch.to(device)

        # only keep credible samples (y=0)
        labels = batch.y.squeeze()            # [B]
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        credible_mask = (labels == 0)

        if credible_mask.sum() == 0:
            total_skipped += batch.num_graphs
            continue

        snapshots, pvt          = tpgb(batch)
        h                       = ste(snapshots, pvt)
        _, mu_z, _, _           = vle(h, batch.root_text_emb,
                                       training=False)

        # keep only credible latents
        mu_credible = mu_z[credible_mask]     # [n_cred, D]
        credible_latents.append(mu_credible.cpu())
        total_credible += mu_credible.shape[0]

    all_latents = torch.cat(credible_latents, dim=0)
    print(f"  Encoded {total_credible} credible training samples "
          f"-> latent cache shape: {all_latents.shape}")
    return all_latents


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    if not os.path.exists(CFG["phase1_ckpt"]):
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found: {CFG['phase1_ckpt']}\n"
            f"Run phase1_encoder.py first."
        )

    print(f"\nPhase 2: Credibility Diffusion Prior Training")
    print(f"Device : {DEVICE}")
    print(f"Epochs : {CFG['epochs']}")

    # ── load frozen encoder ───────────────────────────────────────────────
    print("\nLoading frozen encoder from Phase 1 ...")
    tpgb, ste, vle = load_frozen_encoder(CFG, DEVICE)

    # ── encode credible samples ───────────────────────────────────────────
    print("Encoding credible training samples ...")
    latent_cache = encode_credible_samples(
        tpgb, ste, vle,
        CFG["dataset"], DEVICE, CFG["batch_size_encode"],
    )   # [N_cred, D] on CPU

    # ── build latent dataset ──────────────────────────────────────────────
    latent_ds     = TensorDataset(latent_cache)
    latent_loader = TorchDataLoader(
        latent_ds,
        batch_size = CFG["batch_size_train"],
        shuffle    = True,
        num_workers= 0,
    )
    print(f"  Latent loader: {len(latent_ds)} samples  "
          f"({len(latent_loader)} batches/epoch)")

    # ── build CDP ─────────────────────────────────────────────────────────
    cdp = CDP(
        latent_dim = CFG["latent_dim"],
        hidden_dim = CFG["hidden_dim"],
        T          = CFG["T"],
        T_inf      = CFG["T_inf"],
        t_star     = CFG["t_star"],
    ).to(DEVICE)
    print(f"  CDP parameters: "
          f"{sum(p.numel() for p in cdp.parameters()):,}")

    # ── optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        cdp.parameters(),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-5
    )

    # ── training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    ckpt_path     = os.path.join(
        CFG["checkpoint_dir"], CFG["checkpoint_name"]
    )

    # split latent cache into train/val (90/10)
    N         = len(latent_cache)
    n_val     = max(int(N * 0.1), 1)
    n_train   = N - n_val

    # deterministic split for reproducibility
    torch.manual_seed(42)
    perm      = torch.randperm(N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    train_latents = latent_cache[train_idx]
    val_latents   = latent_cache[val_idx]

    train_loader_cdp = TorchDataLoader(
        TensorDataset(train_latents),
        batch_size=CFG["batch_size_train"], shuffle=True,
    )
    val_loader_cdp = TorchDataLoader(
        TensorDataset(val_latents),
        batch_size=CFG["batch_size_train"], shuffle=False,
    )

    print(f"  CDP train: {len(train_latents)}  val: {len(val_latents)}")
    print(f"\n{'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>10}  "
          f"{'LR':>8}  {'Time':>6}  Best")
    print("-" * 60)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        # train
        cdp.train()
        train_loss = 0.0
        for (z_batch,) in train_loader_cdp:
            z_batch = z_batch.to(DEVICE)
            loss    = cdp.training_loss(z_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                cdp.parameters(), CFG["grad_clip"]
            )
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader_cdp), 1)

        # validate
        cdp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (z_batch,) in val_loader_cdp:
                z_batch   = z_batch.to(DEVICE)
                val_loss += cdp.training_loss(z_batch).item()
        val_loss /= max(len(val_loader_cdp), 1)

        scheduler.step()
        elapsed = time.time() - t0
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch"      : epoch,
                "cdp_state"  : cdp.state_dict(),
                "val_loss"   : val_loss,
                "cfg"        : CFG,
            }, ckpt_path)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {train_loss:>11.4f}  {val_loss:>10.4f}  "
              f"{current_lr:>8.6f}  {elapsed:>5.1f}s  "
              f"{'*' if is_best else ''}")

    print(f"\nPhase 2 complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {ckpt_path}")


if __name__ == "__main__":
    main()
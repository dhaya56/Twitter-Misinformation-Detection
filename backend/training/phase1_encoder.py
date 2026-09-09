"""
phase1_encoder.py
-----------------
Phase 1: Encoder Pretraining.

Trains STE + VLE with a VAE reconstruction objective.
CDP and DSH are not used in this phase.

Objective:
    L = L_recon + lambda_kl * KL(N(mu_z, sigma_z^2) || N(0, I))

    L_recon = MSE(Decoder(z), h)
    where Decoder is a 2-layer MLP that reconstructs the STE
    output h from the latent z. Decoder is discarded after Phase 1.

    KL weight is annealed from kl_start to kl_end over all epochs
    using a linear schedule to prevent posterior collapse.

After Phase 1:
    - STE and VLE weights are saved to checkpoints/phase1_best.pt
    - These weights are loaded at the start of Phase 2 and Phase 3

Usage:
    python backend/training/phase1_encoder.py
Run from: D:\vit_projects\dl_project
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from backend.data.dataset_loader import PropagationDataset
from backend.models.tpgb import TPGB
from backend.models.ste import STE
from backend.models.vle import VLE


# ── config ────────────────────────────────────────────────────────────────────

CFG = {
    # data
    "dataset"         : "twitter15",
    "batch_size"      : 32,
    "num_workers"     : 0,       # 0 is safest on Windows

    # model dimensions
    "node_feat_dim"   : 6,
    "edge_feat_dim"   : 3,
    "gat_hidden_dim"  : 64,
    "gat_num_heads"   : 4,
    "gat_out_dim"     : 96,
    "pvt_out_dim"     : 32,
    "snapshot_embed_dim": 128,
    "transformer_heads" : 4,
    "transformer_layers": 2,
    "dropout"         : 0.1,
    "text_embed_dim"  : 384,
    "text_proj_dim"   : 64,
    "latent_dim"      : 128,

    # training
    "epochs"          : 8,
    "lr"              : 1e-3,
    "weight_decay"    : 1e-5,
    "kl_weight_start" : 0.001,
    "kl_weight_end"   : 0.01,
    "grad_clip"       : 1.0,

    # paths
    "checkpoint_dir"  : "backend/checkpoints",
    "checkpoint_name" : "phase1_best.pt",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── decoder (used only in phase 1) ───────────────────────────────────────────

class ReconDecoder(nn.Module):
    """
    2-layer MLP decoder: latent_dim -> graph_embed_dim.
    Reconstructs STE output h from latent z.
    Discarded after Phase 1.
    """
    def __init__(self, latent_dim, graph_embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, graph_embed_dim),
            nn.LayerNorm(graph_embed_dim),
            nn.ELU(),
            nn.Linear(graph_embed_dim, graph_embed_dim),
        )

    def forward(self, z):
        return self.net(z)


# ── training utilities ────────────────────────────────────────────────────────

def get_kl_weight(epoch, total_epochs, kl_start, kl_end):
    """Linear KL weight annealing."""
    frac = epoch / max(total_epochs - 1, 1)
    return kl_start + frac * (kl_end - kl_start)


def run_epoch(tpgb, ste, vle, decoder, loader,
              optimizer, kl_weight, training, device):
    """
    Runs one epoch. Returns (avg_loss, avg_recon, avg_kl).
    """
    if training:
        ste.train(); vle.train(); decoder.train()
    else:
        ste.eval(); vle.eval(); decoder.eval()

    total_loss  = 0.0
    total_recon = 0.0
    total_kl    = 0.0
    n_batches   = 0

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for batch in loader:
            batch = batch.to(device)

            # forward
            snapshots, pvt       = tpgb(batch)
            h                    = ste(snapshots, pvt)
            z, mu_z, sigma_z, _  = vle(h, batch.root_text_emb,
                                        training=training)
            h_recon              = decoder(z)

            # losses
            l_recon = nn.functional.mse_loss(h_recon, h.detach())
            l_kl    = vle.kl_loss(mu_z, sigma_z).mean()
            loss    = l_recon + kl_weight * l_kl

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(ste.parameters()) +
                    list(vle.parameters()) +
                    list(decoder.parameters()),
                    CFG["grad_clip"],
                )
                optimizer.step()

            total_loss  += loss.item()
            total_recon += l_recon.item()
            total_kl    += l_kl.item()
            n_batches   += 1

    n = max(n_batches, 1)
    return total_loss / n, total_recon / n, total_kl / n


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    print(f"\nPhase 1: Encoder Pretraining")
    print(f"Device : {DEVICE}")
    print(f"Epochs : {CFG['epochs']}")

    # ── datasets ──────────────────────────────────────────────────────────
    train_ds = PropagationDataset(CFG["dataset"], split="train")
    val_ds   = PropagationDataset(CFG["dataset"], split="val")

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        shuffle=True, num_workers=CFG["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"],
        shuffle=False, num_workers=CFG["num_workers"],
    )

    print(f"Train : {len(train_ds)} samples  "
          f"({len(train_loader)} batches)")
    print(f"Val   : {len(val_ds)} samples  "
          f"({len(val_loader)} batches)")

    # ── models ────────────────────────────────────────────────────────────
    tpgb = TPGB(num_snapshots=5).to(DEVICE)

    ste  = STE(
        node_feat_dim      = CFG["node_feat_dim"],
        edge_feat_dim      = CFG["edge_feat_dim"],
        gat_hidden_dim     = CFG["gat_hidden_dim"],
        gat_num_heads      = CFG["gat_num_heads"],
        gat_out_dim        = CFG["gat_out_dim"],
        pvt_out_dim        = CFG["pvt_out_dim"],
        snapshot_embed_dim = CFG["snapshot_embed_dim"],
        transformer_heads  = CFG["transformer_heads"],
        transformer_layers = CFG["transformer_layers"],
        dropout            = CFG["dropout"],
    ).to(DEVICE)

    vle = VLE(
        graph_embed_dim = CFG["snapshot_embed_dim"],
        text_embed_dim  = CFG["text_embed_dim"],
        text_proj_dim   = CFG["text_proj_dim"],
        latent_dim      = CFG["latent_dim"],
        dropout         = CFG["dropout"],
    ).to(DEVICE)

    decoder = ReconDecoder(
        latent_dim      = CFG["latent_dim"],
        graph_embed_dim = CFG["snapshot_embed_dim"],
    ).to(DEVICE)

    # ── optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(ste.parameters()) +
        list(vle.parameters()) +
        list(decoder.parameters()),
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )

    # ── training loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    ckpt_path     = os.path.join(
        CFG["checkpoint_dir"], CFG["checkpoint_name"]
    )

    print(f"\n{'Epoch':>5}  {'KL_w':>6}  "
          f"{'Train_L':>8} {'T_recon':>8} {'T_kl':>7}  "
          f"{'Val_L':>8} {'V_recon':>8} {'V_kl':>7}  "
          f"{'Time':>6}  {'Best'}")
    print("-" * 90)

    for epoch in range(1, CFG["epochs"] + 1):
        kl_w = get_kl_weight(
            epoch - 1, CFG["epochs"],
            CFG["kl_weight_start"], CFG["kl_weight_end"],
        )

        t0 = time.time()

        tr_loss, tr_recon, tr_kl = run_epoch(
            tpgb, ste, vle, decoder, train_loader,
            optimizer, kl_w, training=True, device=DEVICE,
        )
        va_loss, va_recon, va_kl = run_epoch(
            tpgb, ste, vle, decoder, val_loader,
            optimizer, kl_w, training=False, device=DEVICE,
        )

        elapsed = time.time() - t0
        is_best = va_loss < best_val_loss

        if is_best:
            best_val_loss = va_loss
            torch.save({
                "epoch"      : epoch,
                "ste_state"  : ste.state_dict(),
                "vle_state"  : vle.state_dict(),
                "val_loss"   : va_loss,
                "cfg"        : CFG,
            }, ckpt_path)

        print(f"{epoch:>5}  {kl_w:>6.4f}  "
              f"{tr_loss:>8.4f} {tr_recon:>8.4f} {tr_kl:>7.2f}  "
              f"{va_loss:>8.4f} {va_recon:>8.4f} {va_kl:>7.2f}  "
              f"{elapsed:>5.1f}s  "
              f"{'*' if is_best else ''}")

    print(f"\nPhase 1 complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {ckpt_path}")


if __name__ == "__main__":
    main()
"""
phase3_joint.py  — OPTIMIZED
1 aux horizon (tau=1440), N=2 train completions, T_inf=10
Expected epoch time: ~4-5 min on RTX 3060
"""

import os, sys, time
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from backend.data.dataset_loader import PropagationDataset
from backend.models.tpgb import TPGB
from backend.models.ste  import STE
from backend.models.vle  import VLE
from backend.models.cdp  import CDP
from backend.models.ccm  import CCM
from backend.models.dsh  import DSH

CFG = {
    "dataset"           : "twitter15",
    "batch_size"        : 16,
    "num_workers"       : 0,
    "node_feat_dim"     : 6,
    "edge_feat_dim"     : 3,
    "gat_hidden_dim"    : 64,
    "gat_num_heads"     : 4,
    "gat_out_dim"       : 96,
    "pvt_out_dim"       : 32,
    "snapshot_embed_dim": 128,
    "transformer_heads" : 4,
    "transformer_layers": 2,
    "dropout"           : 0.1,
    "text_embed_dim"    : 384,
    "text_proj_dim"     : 64,
    "latent_dim"        : 128,
    "T"                 : 200,
    "T_inf"             : 10,
    "t_star"            : 40,
    "cdp_hidden_dim"    : 256,
    "num_completions_train" : 2,
    "num_completions_eval"  : 8,
    "tau_max"               : 11786.27,
    "epochs"            : 27,
    "lr"                : 3e-4,
    "weight_decay"      : 1e-5,
    "grad_clip"         : 1.0,
    "lambda_aux"        : 0.3,
    "patience"          : 10,
    "focal_alpha"       : 0.25,
    "focal_gamma"       : 2.0,
    "aux_tau"           : 1440.0,
    "checkpoint_dir"    : "backend/checkpoints",
    "phase1_ckpt"       : "backend/checkpoints/phase1_best.pt",
    "phase2_ckpt"       : "backend/checkpoints/phase2_best.pt",
    "checkpoint_name"   : "phase3_best.pt",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    targets_f = targets.float()
    probs     = torch.sigmoid(logits)
    probs_t   = torch.where(targets_f == 1, probs, 1 - probs)
    alpha_t   = torch.where(targets_f == 1,
                             torch.full_like(probs, alpha),
                             torch.full_like(probs, 1 - alpha))
    return (-alpha_t * ((1 - probs_t) ** gamma) * torch.log(probs_t + 1e-8)).mean()


def build_models(cfg, device):
    tpgb = TPGB(num_snapshots=5).to(device)
    ste  = STE(
        node_feat_dim=cfg["node_feat_dim"], edge_feat_dim=cfg["edge_feat_dim"],
        gat_hidden_dim=cfg["gat_hidden_dim"], gat_num_heads=cfg["gat_num_heads"],
        gat_out_dim=cfg["gat_out_dim"], pvt_out_dim=cfg["pvt_out_dim"],
        snapshot_embed_dim=cfg["snapshot_embed_dim"],
        transformer_heads=cfg["transformer_heads"],
        transformer_layers=cfg["transformer_layers"], dropout=cfg["dropout"],
    ).to(device)
    vle  = VLE(
        graph_embed_dim=cfg["snapshot_embed_dim"], text_embed_dim=cfg["text_embed_dim"],
        text_proj_dim=cfg["text_proj_dim"], latent_dim=cfg["latent_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    cdp  = CDP(
        latent_dim=cfg["latent_dim"], hidden_dim=cfg["cdp_hidden_dim"],
        T=cfg["T"], T_inf=cfg["T_inf"], t_star=cfg["t_star"],
    ).to(device)
    dsh  = DSH(input_dim=4, hidden_dim=64).to(device)

    ckpt1 = torch.load(cfg["phase1_ckpt"], map_location=device, weights_only=False)
    ste.load_state_dict(ckpt1["ste_state"])
    vle.load_state_dict(ckpt1["vle_state"])
    print(f"  STE+VLE from Phase 1 (epoch {ckpt1['epoch']}, val_loss={ckpt1['val_loss']:.4f})")

    ckpt2 = torch.load(cfg["phase2_ckpt"], map_location=device, weights_only=False)
    cdp.load_state_dict(ckpt2["cdp_state"])
    print(f"  CDP from Phase 2 (epoch {ckpt2['epoch']}, val_loss={ckpt2['val_loss']:.4f})")

    cdp.eval()
    for p in cdp.parameters():
        p.requires_grad_(False)
    print("  CDP frozen.")

    return tpgb, ste, vle, cdp, dsh


def run_epoch(tpgb, ste, vle, cdp, dsh, ccm,
              loader, optimizer, cfg, training, device):
    if training:
        ste.train(); vle.train(); dsh.train()
    else:
        ste.eval();  vle.eval();  dsh.eval()

    total_loss, all_scores, all_labels, n = 0.0, [], [], 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for batch in loader:
            batch  = batch.to(device)
            labels = batch.y.squeeze()
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

            # encode once
            snapshots, pvt         = tpgb(batch)
            h                      = ste(snapshots, pvt)
            z, mu_z, sigma_z, sig2 = vle(h, batch.root_text_emb, training=training)

            # CDP projection once (frozen)
            with torch.no_grad():
                _, d_m = cdp.project_to_manifold(mu_z)
            d_tr = dsh.compute_d_tr(d_m, torch.zeros_like(d_m))

            # full-graph loss
            d_cv_full, _  = ccm(mu_z, tau_obs=-1.0)
            _, logit_full = dsh(d_m, d_cv_full, d_tr, sig2)
            l_main = focal_loss(logit_full, labels, cfg["focal_alpha"], cfg["focal_gamma"])

            # single aux horizon (reuses d_m, sig2, d_tr)
            d_cv_aux, _  = ccm(mu_z, tau_obs=cfg["aux_tau"])
            _, logit_aux = dsh(d_m, d_cv_aux, d_tr, sig2)
            l_aux = focal_loss(logit_aux, labels, cfg["focal_alpha"], cfg["focal_gamma"])

            loss = l_main + cfg["lambda_aux"] * l_aux

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(ste.parameters()) + list(vle.parameters()) + list(dsh.parameters()),
                    cfg["grad_clip"],
                )
                optimizer.step()

            total_loss += loss.item()
            all_scores.extend(torch.sigmoid(logit_full).detach().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            n += 1

    return total_loss / max(n, 1), all_scores, all_labels


def main():
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    print(f"\nPhase 3: Joint Fine-Tuning  [OPTIMIZED]")
    print(f"Device         : {DEVICE}")
    print(f"Epochs         : {CFG['epochs']}")
    print(f"Aux horizon    : {CFG['aux_tau']} min  |  CCM N train={CFG['num_completions_train']} eval={CFG['num_completions_eval']}  |  T_inf={CFG['T_inf']}")

    train_ds = PropagationDataset(CFG["dataset"], split="train")
    val_ds   = PropagationDataset(CFG["dataset"], split="val")
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} ({len(train_loader)} batches)  Val: {len(val_ds)} ({len(val_loader)} batches)")

    print()
    tpgb, ste, vle, cdp, dsh = build_models(CFG, DEVICE)
    ccm_train = CCM(cdp=cdp, num_completions=CFG["num_completions_train"], tau_max=CFG["tau_max"])
    ccm_eval  = CCM(cdp=cdp, num_completions=CFG["num_completions_eval"],  tau_max=CFG["tau_max"])

    trainable = sum(p.numel() for p in ste.parameters() if p.requires_grad) + \
                sum(p.numel() for p in vle.parameters() if p.requires_grad) + \
                sum(p.numel() for p in dsh.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")

    optimizer = torch.optim.Adam(
        list(ste.parameters()) + list(vle.parameters()) + list(dsh.parameters()),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
    )

    best_val_auc, patience_ctr = 0.0, 0
    ckpt_path = os.path.join(CFG["checkpoint_dir"], CFG["checkpoint_name"])

    print(f"\n{'Ep':>3}  {'Tr Loss':>8}  {'Tr AUC':>7}  {'Va Loss':>8}  {'Va AUC':>7}  {'LR':>8}  {'Time':>6}  Best")
    print("-" * 75)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_scores, tr_labels = run_epoch(
            tpgb, ste, vle, cdp, dsh, ccm_train,
            train_loader, optimizer, CFG, training=True, device=DEVICE)
        va_loss, va_scores, va_labels = run_epoch(
            tpgb, ste, vle, cdp, dsh, ccm_eval,
            val_loader, optimizer, CFG, training=False, device=DEVICE)

        tr_auc = roc_auc_score(tr_labels, tr_scores)
        va_auc = roc_auc_score(va_labels, va_scores)
        scheduler.step(va_auc)
        elapsed = time.time() - t0
        is_best = va_auc > best_val_auc

        if is_best:
            best_val_auc = va_auc
            patience_ctr = 0
            torch.save({"epoch": epoch, "ste_state": ste.state_dict(),
                        "vle_state": vle.state_dict(), "dsh_state": dsh.state_dict(),
                        "val_auc": va_auc, "cfg": CFG}, ckpt_path)
        else:
            patience_ctr += 1

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>3}  {tr_loss:>8.4f}  {tr_auc:>7.4f}  "
              f"{va_loss:>8.4f}  {va_auc:>7.4f}  "
              f"{lr_now:>8.6f}  {elapsed:>5.1f}s  {'*' if is_best else ''}")

        if patience_ctr >= CFG["patience"]:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nPhase 3 complete.  Best val AUC: {best_val_auc:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
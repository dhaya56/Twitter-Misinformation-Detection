"""
ablation.py
-----------
Ablation study for CGDEX-Net.

Tests 4 model configurations on Twitter15 test set:
    1. Full model (d_m + d_cv + d_tr + sigma2_z)
    2. Without d_cv  (set d_cv=0 at inference)
    3. Without CDP   (set d_m=0, d_tr=0 at inference)
    4. Without d_tr  (set d_tr=0 at inference)

Reports AUC, Accuracy, F1 for each configuration.

Usage:
    python backend/evaluation/ablation.py
Run from: D:\\vit_projects\\dl_project
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath("."))

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from backend.data.dataset_loader import PropagationDataset
from backend.models.tpgb import TPGB
from backend.models.ste  import STE
from backend.models.vle  import VLE
from backend.models.cdp  import CDP
from backend.models.ccm  import CCM
from backend.models.dsh  import DSH


CFG = {
    "node_feat_dim": 6, "edge_feat_dim": 3,
    "gat_hidden_dim": 64, "gat_num_heads": 4,
    "gat_out_dim": 96, "pvt_out_dim": 32,
    "snapshot_embed_dim": 128, "transformer_heads": 4,
    "transformer_layers": 2, "dropout": 0.0,
    "text_embed_dim": 384, "text_proj_dim": 64,
    "latent_dim": 128, "T": 200, "T_inf": 10,
    "t_star": 40, "cdp_hidden_dim": 256,
    "num_completions": 8, "tau_max": 11786.27,
    "batch_size": 16,
    "phase2_ckpt": "backend/checkpoints/phase2_best.pt",
    "phase3_ckpt": "backend/checkpoints/phase3_best.pt",
    "results_dir": "results",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    tpgb = TPGB(num_snapshots=5).to(DEVICE)
    ste = STE(
        node_feat_dim=CFG["node_feat_dim"], edge_feat_dim=CFG["edge_feat_dim"],
        gat_hidden_dim=CFG["gat_hidden_dim"], gat_num_heads=CFG["gat_num_heads"],
        gat_out_dim=CFG["gat_out_dim"], pvt_out_dim=CFG["pvt_out_dim"],
        snapshot_embed_dim=CFG["snapshot_embed_dim"],
        transformer_heads=CFG["transformer_heads"],
        transformer_layers=CFG["transformer_layers"], dropout=CFG["dropout"],
    ).to(DEVICE)
    vle = VLE(
        graph_embed_dim=CFG["snapshot_embed_dim"],
        text_embed_dim=CFG["text_embed_dim"],
        text_proj_dim=CFG["text_proj_dim"],
        latent_dim=CFG["latent_dim"], dropout=CFG["dropout"],
    ).to(DEVICE)
    cdp = CDP(
        latent_dim=CFG["latent_dim"], hidden_dim=CFG["cdp_hidden_dim"],
        T=CFG["T"], T_inf=CFG["T_inf"], t_star=CFG["t_star"],
    ).to(DEVICE)
    dsh = DSH(input_dim=4, hidden_dim=64).to(DEVICE)

    ckpt2 = torch.load(CFG["phase2_ckpt"], map_location=DEVICE, weights_only=False)
    ckpt3 = torch.load(CFG["phase3_ckpt"], map_location=DEVICE, weights_only=False)
    ste.load_state_dict(ckpt3["ste_state"])
    vle.load_state_dict(ckpt3["vle_state"])
    cdp.load_state_dict(ckpt2["cdp_state"])
    dsh.load_state_dict(ckpt3["dsh_state"])

    for m in (tpgb, ste, vle, cdp, dsh):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    ccm = CCM(cdp=cdp, num_completions=CFG["num_completions"],
              tau_max=CFG["tau_max"])
    return tpgb, ste, vle, cdp, ccm, dsh


@torch.no_grad()
def run_ablation(tpgb, ste, vle, cdp, ccm, dsh, loader,
                 zero_dcv=False, zero_cdp=False, zero_dtr=False):
    all_scores, all_labels = [], []

    for batch in loader:
        batch  = batch.to(DEVICE)
        labels = batch.y.squeeze()
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        snapshots, pvt         = tpgb(batch)
        h                      = ste(snapshots, pvt)
        z, mu_z, sigma_z, sig2 = vle(h, batch.root_text_emb, training=False)

        _, d_m   = cdp.project_to_manifold(mu_z)
        d_cv, _  = ccm(mu_z, tau_obs=-1.0)
        d_tr     = dsh.compute_d_tr(d_m, torch.zeros_like(d_m))

        # ablations
        if zero_dcv:
            d_cv = torch.zeros_like(d_cv)
        if zero_cdp:
            d_m  = torch.zeros_like(d_m)
            d_tr = torch.zeros_like(d_tr)
        if zero_dtr:
            d_tr = torch.zeros_like(d_tr)

        score, _ = dsh(d_m, d_cv, d_tr, sig2)
        all_scores.extend(score.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    preds = [1 if s >= 0.5 else 0 for s in all_scores]
    return {
        "auc"     : round(roc_auc_score(all_labels, all_scores), 4),
        "accuracy": round(accuracy_score(all_labels, preds), 4),
        "f1"      : round(f1_score(all_labels, preds, zero_division=0), 4),
    }


def main():
    os.makedirs(CFG["results_dir"], exist_ok=True)
    print("\nAblation Study — CGDEX-Net")
    print(f"Device: {DEVICE}\n")

    tpgb, ste, vle, cdp, ccm, dsh = load_models()

    test_ds = PropagationDataset("twitter15", split="test")
    loader  = DataLoader(test_ds, batch_size=CFG["batch_size"],
                         shuffle=False, num_workers=0)

    configs = [
        ("Full model",          dict()),
        ("w/o d_cv (no CCM)",   dict(zero_dcv=True)),
        ("w/o CDP (no d_m,d_tr)",dict(zero_cdp=True)),
        ("w/o d_tr",            dict(zero_dtr=True)),
    ]

    print(f"  {'Configuration':<28}  {'AUC':>7}  {'Acc':>7}  {'F1':>7}")
    print(f"  {'-'*55}")

    ablation_results = {}
    for name, kwargs in configs:
        metrics = run_ablation(tpgb, ste, vle, cdp, ccm, dsh,
                               loader, **kwargs)
        ablation_results[name] = metrics
        print(f"  {name:<28}  {metrics['auc']:>7.4f}  "
              f"{metrics['accuracy']:>7.4f}  {metrics['f1']:>7.4f}")

    out_path = os.path.join(CFG["results_dir"], "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
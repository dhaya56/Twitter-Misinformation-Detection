"""
evaluate.py
-----------
Full evaluation of the trained CGDEX-Net model.

Produces:
    1. Main detection metrics on Twitter15 test set (full graph)
       Accuracy, F1, AUC, Precision, Recall

    2. Early detection curve
       AUC at each horizon: [60, 360, 720, 1440, 2880, 4320] minutes

    3. Cross-dataset transfer
       AUC on Twitter16 test set (zero-shot, no fine-tuning)

    4. Per-class metrics
       Credible vs Misinformation breakdown

    5. Detection feature distributions
       Mean d_m, d_cv, d_tr, sigma2_z per class (for analysis)

Output:
    Prints results to console.
    Saves results/evaluation_results.json

Usage:
    python backend/evaluation/evaluate.py
Run from: D:\\vit_projects\\dl_project
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, classification_report,
)

from backend.data.dataset_loader import PropagationDataset
from backend.models.tpgb import TPGB
from backend.models.ste  import STE
from backend.models.vle  import VLE
from backend.models.cdp  import CDP
from backend.models.ccm  import CCM
from backend.models.dsh  import DSH


# ── config ────────────────────────────────────────────────────────────────────

CFG = {
    "node_feat_dim"     : 6,
    "edge_feat_dim"     : 3,
    "gat_hidden_dim"    : 64,
    "gat_num_heads"     : 4,
    "gat_out_dim"       : 96,
    "pvt_out_dim"       : 32,
    "snapshot_embed_dim": 128,
    "transformer_heads" : 4,
    "transformer_layers": 2,
    "dropout"           : 0.0,   # disabled at eval
    "text_embed_dim"    : 384,
    "text_proj_dim"     : 64,
    "latent_dim"        : 128,
    "T"                 : 200,
    "T_inf"             : 10,
    "t_star"            : 40,
    "cdp_hidden_dim"    : 256,
    "num_completions"   : 8,
    "tau_max"           : 11786.27,

    "batch_size"        : 16,
    "phase1_ckpt"       : "backend/checkpoints/phase1_best.pt",
    "phase2_ckpt"       : "backend/checkpoints/phase2_best.pt",
    "phase3_ckpt"       : "backend/checkpoints/phase3_best.pt",

    "early_horizons"    : [60, 360, 720, 1440, 2880, 4320],
    "results_dir"       : "results",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── model loading ─────────────────────────────────────────────────────────────

def load_all_models(cfg, device):
    tpgb = TPGB(num_snapshots=5).to(device)

    ste = STE(
        node_feat_dim=cfg["node_feat_dim"], edge_feat_dim=cfg["edge_feat_dim"],
        gat_hidden_dim=cfg["gat_hidden_dim"], gat_num_heads=cfg["gat_num_heads"],
        gat_out_dim=cfg["gat_out_dim"], pvt_out_dim=cfg["pvt_out_dim"],
        snapshot_embed_dim=cfg["snapshot_embed_dim"],
        transformer_heads=cfg["transformer_heads"],
        transformer_layers=cfg["transformer_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    vle = VLE(
        graph_embed_dim=cfg["snapshot_embed_dim"],
        text_embed_dim=cfg["text_embed_dim"],
        text_proj_dim=cfg["text_proj_dim"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    cdp = CDP(
        latent_dim=cfg["latent_dim"], hidden_dim=cfg["cdp_hidden_dim"],
        T=cfg["T"], T_inf=cfg["T_inf"], t_star=cfg["t_star"],
    ).to(device)

    dsh = DSH(input_dim=4, hidden_dim=64).to(device)

    # load weights
    ckpt1 = torch.load(cfg["phase1_ckpt"], map_location=device, weights_only=False)
    ckpt2 = torch.load(cfg["phase2_ckpt"], map_location=device, weights_only=False)
    ckpt3 = torch.load(cfg["phase3_ckpt"], map_location=device, weights_only=False)

    ste.load_state_dict(ckpt3["ste_state"])
    vle.load_state_dict(ckpt3["vle_state"])
    cdp.load_state_dict(ckpt2["cdp_state"])
    dsh.load_state_dict(ckpt3["dsh_state"])

    for m in (tpgb, ste, vle, cdp, dsh):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    ccm = CCM(cdp=cdp, num_completions=cfg["num_completions"],
              tau_max=cfg["tau_max"])

    print(f"  Phase 1 checkpoint: epoch {ckpt1['epoch']}")
    print(f"  Phase 2 checkpoint: epoch {ckpt2['epoch']}")
    print(f"  Phase 3 checkpoint: epoch {ckpt3['epoch']}  val_AUC={ckpt3['val_auc']:.4f}")

    return tpgb, ste, vle, cdp, ccm, dsh


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(tpgb, ste, vle, cdp, ccm, dsh, loader, tau_obs, device):
    """
    Runs inference on a DataLoader.

    Returns:
        scores  : list of float   sigmoid probabilities
        labels  : list of int     ground truth binary labels
        feats   : dict of lists   detection feature values per sample
    """
    all_scores  = []
    all_labels  = []
    all_dm      = []
    all_dcv     = []
    all_dtr     = []
    all_sig2    = []

    for batch in loader:
        batch  = batch.to(device)
        labels = batch.y.squeeze()
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        snapshots, pvt         = tpgb(batch)
        h                      = ste(snapshots, pvt)
        z, mu_z, sigma_z, sig2 = vle(h, batch.root_text_emb, training=False)

        _, d_m   = cdp.project_to_manifold(mu_z)
        d_cv, _  = ccm(mu_z, tau_obs=tau_obs)
        d_tr     = dsh.compute_d_tr(d_m, torch.zeros_like(d_m))

        score, _ = dsh(d_m, d_cv, d_tr, sig2)

        all_scores.extend(score.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_dm.extend(d_m.cpu().tolist())
        all_dcv.extend(d_cv.cpu().tolist())
        all_dtr.extend(d_tr.cpu().tolist())
        all_sig2.extend(sig2.cpu().tolist())

    feats = {"d_m": all_dm, "d_cv": all_dcv,
             "d_tr": all_dtr, "sigma2_z": all_sig2}
    return all_scores, all_labels, feats


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(scores, labels, threshold=0.5):
    preds = [1 if s >= threshold else 0 for s in scores]
    return {
        "accuracy"  : round(accuracy_score(labels, preds), 4),
        "f1"        : round(f1_score(labels, preds, zero_division=0), 4),
        "precision" : round(precision_score(labels, preds, zero_division=0), 4),
        "recall"    : round(recall_score(labels, preds, zero_division=0), 4),
        "auc"       : round(roc_auc_score(labels, scores), 4),
    }


def feature_stats(feats, labels):
    """Returns mean of each detection feature split by class."""
    labels_arr = np.array(labels)
    stats = {}
    for feat_name, values in feats.items():
        arr = np.array(values)
        stats[feat_name] = {
            "credible_mean"  : round(float(arr[labels_arr == 0].mean()), 4),
            "misinfo_mean"   : round(float(arr[labels_arr == 1].mean()), 4),
        }
    return stats


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CFG["results_dir"], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CGDEX-Net Evaluation")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    # ── load models ───────────────────────────────────────────────────────
    print("\nLoading models ...")
    tpgb, ste, vle, cdp, ccm, dsh = load_all_models(CFG, DEVICE)

    results = {}

    # ── 1. Main detection — Twitter15 test set (full graph) ────────────────
    print(f"\n{'─'*60}")
    print("  1. Main Detection — Twitter15 Test (full graph)")
    print(f"{'─'*60}")

    test_ds = PropagationDataset("twitter15", split="test")
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"],
                             shuffle=False, num_workers=0)

    scores, labels, feats = run_inference(
        tpgb, ste, vle, cdp, ccm, dsh,
        test_loader, tau_obs=-1.0, device=DEVICE,
    )
    main_metrics = compute_metrics(scores, labels)
    results["twitter15_test_full"] = main_metrics

    print(f"  Accuracy  : {main_metrics['accuracy']}")
    print(f"  F1        : {main_metrics['f1']}")
    print(f"  Precision : {main_metrics['precision']}")
    print(f"  Recall    : {main_metrics['recall']}")
    print(f"  AUC       : {main_metrics['auc']}")

    # per-class report
    preds = [1 if s >= 0.5 else 0 for s in scores]
    print(f"\n{classification_report(labels, preds, target_names=['credible','misinfo'])}")

    # feature distributions
    fstats = feature_stats(feats, labels)
    results["feature_distributions"] = fstats
    print("  Detection feature means by class:")
    for f, v in fstats.items():
        print(f"    {f:<12}: credible={v['credible_mean']:.4f}  "
              f"misinfo={v['misinfo_mean']:.4f}")

    # ── 2. Early detection curve ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  2. Early Detection AUC Curve")
    print(f"{'─'*60}")

    early_results = {}
    print(f"  {'Horizon (min)':>14}  {'AUC':>7}  {'Acc':>7}  {'F1':>7}")
    print(f"  {'-'*40}")

    for tau in CFG["early_horizons"]:
        tau_ds = PropagationDataset("twitter15", split="test",
                                    tau_minutes=float(tau))
        tau_loader = DataLoader(tau_ds, batch_size=CFG["batch_size"],
                                shuffle=False, num_workers=0)
        sc, lb, _ = run_inference(
            tpgb, ste, vle, cdp, ccm, dsh,
            tau_loader, tau_obs=float(tau), device=DEVICE,
        )
        m = compute_metrics(sc, lb)
        early_results[str(tau)] = m
        print(f"  {tau:>14}  {m['auc']:>7.4f}  {m['accuracy']:>7.4f}  {m['f1']:>7.4f}")

    results["early_detection"] = early_results

    # also record full graph for comparison
    results["early_detection"]["full"] = main_metrics

    # ── 3. Cross-dataset transfer — Twitter16 ─────────────────────────────
    print(f"\n{'─'*60}")
    print("  3. Cross-Dataset Transfer — Twitter16 (zero-shot)")
    print(f"{'─'*60}")

    t16_ds = PropagationDataset("twitter16", split="test")
    t16_loader = DataLoader(t16_ds, batch_size=CFG["batch_size"],
                            shuffle=False, num_workers=0)
    sc16, lb16, feats16 = run_inference(
        tpgb, ste, vle, cdp, ccm, dsh,
        t16_loader, tau_obs=-1.0, device=DEVICE,
    )
    t16_metrics = compute_metrics(sc16, lb16)
    results["twitter16_test_full"] = t16_metrics

    print(f"  Accuracy  : {t16_metrics['accuracy']}")
    print(f"  F1        : {t16_metrics['f1']}")
    print(f"  AUC       : {t16_metrics['auc']}")

    # ── save results ──────────────────────────────────────────────────────
    out_path = os.path.join(CFG["results_dir"], "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Results saved: {out_path}")
    print(f"{'='*60}")
    print(f"\n  SUMMARY")
    print(f"  Twitter15 test AUC  : {main_metrics['auc']}")
    print(f"  Twitter16 xfer AUC  : {t16_metrics['auc']}")
    print(f"  Early (60 min) AUC  : {early_results['60']['auc']}")
    print(f"  Early (1440 min) AUC: {early_results['1440']['auc']}")


if __name__ == "__main__":
    main()
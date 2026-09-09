"""
adaptive_scorer.py
------------------
Latent Space K-Nearest Neighbors Adaptive Scorer.

This is the genuinely per-input adaptive scoring system.
No fixed threshold. No pre-computed cutoff.

HOW IT WORKS:
    1. Encode all training samples through STE + VLE -> latent vectors mu_z
    2. Cache latent vectors + labels (0=credible, 1=misinfo)
    3. For any new input, find its K nearest neighbors in latent space
    4. Score = distance-weighted fraction of misinfo neighbors
       Score = sum(label_i / dist_i) / sum(1 / dist_i)
    5. Uncertain zone adapts per-input: wider when sigma2_z is high

WHY THIS IS GENUINELY PER-INPUT:
    Different inputs land in different regions of the latent manifold.
    A claim that looks like past misinformation gets a high score.
    A claim that looks like credible news gets a low score.
    The uncertain zone ALSO adapts: if sigma2_z is high for this input,
    the model widens its uncertain zone for THIS input only.

Build the index:
    python backend/models/adaptive_scorer.py
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))


def _adapt_tpgb_outputs_for_ste(snapshots, pvt, edge_feat_dim=3, pvt_out_dim=32):
    adapted = []
    for snap in snapshots:
        if isinstance(snap, dict):
            x = snap["x"]
            edge_index = snap["edge_index"]
            edge_attr = snap.get("edge_attr")
            batch_ptr = snap.get("batch")
            mask = snap.get("mask")
        elif isinstance(snap, (tuple, list)):
            if len(snap) == 5:
                x, edge_index, edge_attr, batch_ptr, mask = snap
            elif len(snap) == 4:
                x, edge_index, edge_attr, batch_ptr = snap
                mask = None
            else:
                raise ValueError(f"Unsupported snapshot length: {len(snap)}")
        else:
            raise TypeError(f"Unsupported snapshot type: {type(snap)}")

        if batch_ptr is None:
            batch_ptr = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if edge_attr is None:
            edge_attr = torch.zeros(
                (edge_index.shape[1], edge_feat_dim),
                dtype=x.dtype,
                device=x.device,
            )
        if mask is None:
            mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        adapted.append((x, edge_index, edge_attr, batch_ptr, mask))

    if pvt.dim() == 3:
        pvt = pvt.reshape(pvt.shape[0], -1)
    if pvt.dim() != 2:
        raise ValueError(f"Unsupported PVT shape: {tuple(pvt.shape)}")

    cur_dim = int(pvt.shape[1])
    if cur_dim > pvt_out_dim:
        pvt = pvt[:, :pvt_out_dim]
    elif cur_dim < pvt_out_dim:
        pad = torch.zeros(
            (pvt.shape[0], pvt_out_dim - cur_dim),
            dtype=pvt.dtype,
            device=pvt.device,
        )
        pvt = torch.cat([pvt, pad], dim=1)

    return adapted, pvt


class LatentKNNScorer:
    def __init__(self, k=15):
        self.k             = k
        self.train_vectors = None
        self.train_labels  = None
        self.sigma2_scale  = None
        self.n_credible    = 0
        self.n_misinfo     = 0
        self.is_built      = False

    @torch.no_grad()
    def build_index(self, train_loader, tpgb, ste, vle, device):
        print("Building KNN latent index ...")
        all_mu, all_labels, all_sig2 = [], [], []

        for batch in train_loader:
            batch = batch.to(device)
            y     = batch.y.squeeze()
            if y.dim() == 0:
                y = y.unsqueeze(0)
            snapshots_raw, pvt_raw = tpgb(batch)
            snapshots, pvt         = _adapt_tpgb_outputs_for_ste(
                snapshots_raw,
                pvt_raw,
                edge_feat_dim=3,
                pvt_out_dim=32,
            )
            h                      = ste(snapshots, pvt)
            z, mu_z, sigma_z, sig2 = vle(h, batch.root_text_emb, training=False)
            all_mu.append(mu_z.cpu())
            all_labels.append(y.cpu())
            all_sig2.append(sig2.cpu())

        self.train_vectors = torch.cat(all_mu, dim=0)
        self.train_labels  = torch.cat(all_labels, dim=0)
        all_sig2_cat       = torch.cat(all_sig2, dim=0)
        self.sigma2_scale  = float(torch.quantile(all_sig2_cat, 0.95).item())
        if self.sigma2_scale < 1e-6:
            self.sigma2_scale = 1.0
        self.n_credible = int((self.train_labels == 0).sum().item())
        self.n_misinfo  = int((self.train_labels == 1).sum().item())
        self.is_built   = True
        print(f"  Index: {self.train_vectors.shape[0]} vectors "
              f"({self.n_credible} credible, {self.n_misinfo} misinfo)")
        print(f"  sigma2 scale (p95): {self.sigma2_scale:.4f}")

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "k": self.k, "train_vectors": self.train_vectors,
            "train_labels": self.train_labels, "sigma2_scale": self.sigma2_scale,
            "n_credible": self.n_credible, "n_misinfo": self.n_misinfo,
        }, path)
        print(f"Saved: {path}")

    @classmethod
    def load(cls, path):
        ckpt   = torch.load(path, weights_only=True)
        scorer = cls(k=ckpt["k"])
        scorer.train_vectors = ckpt["train_vectors"]
        scorer.train_labels  = ckpt["train_labels"]
        scorer.sigma2_scale  = ckpt["sigma2_scale"]
        scorer.n_credible    = ckpt["n_credible"]
        scorer.n_misinfo     = ckpt["n_misinfo"]
        scorer.is_built      = True
        return scorer

    @torch.no_grad()
    def score(self, mu_z, sigma2_z):
        """
        Per-input adaptive score.

        Args:
            mu_z     : FloatTensor [1, D]
            sigma2_z : FloatTensor [1]

        Returns dict with:
            knn_score        : float  weighted misinfo fraction
            verdict          : str    MISINFORMATION / CREDIBLE / UNCERTAIN
            uncertain_margin : float  auto-adapted per this input's sigma2_z
            explanation      : str    XAI text
        """
        if not self.is_built:
            raise RuntimeError("Call build_index() or load() first.")

        device     = mu_z.device
        train_vecs = self.train_vectors.to(device)
        train_labs = self.train_labels.to(device)

        diffs  = train_vecs - mu_z
        dists  = (diffs ** 2).sum(dim=1).sqrt().clamp(min=1e-8)

        k_actual         = min(self.k, train_vecs.shape[0])
        top_dists, top_idx = torch.topk(dists, k_actual, largest=False)
        top_labels       = train_labs[top_idx]

        weights      = 1.0 / top_dists
        weight_sum   = weights.sum()
        misinfo_mask = (top_labels == 1).float()
        knn_score    = float((weights * misinfo_mask).sum() / weight_sum)

        sig2_val    = float(sigma2_z.mean().item())
        sigma2_norm = min(sig2_val / self.sigma2_scale, 1.0)

        # uncertain margin expands automatically when encoder is unsure
        uncertain_margin = 0.10 + 0.20 * sigma2_norm

        if knn_score > 0.5 + uncertain_margin:
            verdict = "MISINFORMATION"
        elif knn_score < 0.5 - uncertain_margin:
            verdict = "CREDIBLE"
        else:
            verdict = "UNCERTAIN"

        n_mis  = int(misinfo_mask.sum().item())
        n_cred = k_actual - n_mis

        neighbors = []
        for i in range(min(5, k_actual)):
            idx   = int(top_idx[i].item())
            dist  = float(top_dists[i].item())
            label = int(top_labels[i].item())
            w     = float(weights[i].item() / weight_sum.item())
            signed = w if label == 1 else -w
            neighbors.append({
                "rank": i+1, "label": "misinfo" if label==1 else "credible",
                "distance": round(dist,4), "weight": round(w,4),
                # Signed contribution avoids misleading all-zero credible rows.
                "contribution": round(signed,6),
            })

        if verdict == "MISINFORMATION":
            explanation = (
                f"{n_mis}/{k_actual} nearest training neighbors are misinformation "
                f"(weighted score {knn_score:.3f}). σ²z={sig2_val:.2f} "
                f"→ uncertain margin ±{uncertain_margin:.3f}."
            )
        elif verdict == "CREDIBLE":
            explanation = (
                f"{n_cred}/{k_actual} nearest training neighbors are credible "
                f"(weighted score {knn_score:.3f}). σ²z={sig2_val:.2f} "
                f"→ uncertain margin ±{uncertain_margin:.3f}."
            )
        else:
            explanation = (
                f"Mixed neighborhood ({n_mis} misinfo, {n_cred} credible) "
                f"and/or high encoder uncertainty σ²z={sig2_val:.2f}. "
                f"Uncertain zone [{0.5-uncertain_margin:.3f}, "
                f"{0.5+uncertain_margin:.3f}] auto-adapted for this input. "
                f"Collect more propagation data for a confident verdict."
            )

        return {
            "knn_score"              : round(knn_score, 4),
            "sigma2_norm"            : round(sigma2_norm, 4),
            "sigma2_raw"             : round(sig2_val, 4),
            "uncertain_margin"       : round(uncertain_margin, 4),
            "verdict"                : verdict,
            "n_misinfo_neighbors"    : n_mis,
            "n_credible_neighbors"   : n_cred,
            "k_used"                 : k_actual,
            "neighbors"              : neighbors,
            "explanation"            : explanation,
        }


def build_knn_index():
    from torch_geometric.loader import DataLoader
    from backend.data.dataset_loader import PropagationDataset
    from backend.models.tpgb import TPGB
    from backend.models.ste  import STE
    from backend.models.vle  import VLE

    CFG = {
        "node_feat_dim": 6, "edge_feat_dim": 3,
        "gat_hidden_dim": 64, "gat_num_heads": 4,
        "gat_out_dim": 96, "pvt_out_dim": 32,
        "snapshot_embed_dim": 128, "transformer_heads": 4,
        "transformer_layers": 2, "dropout": 0.0,
        "text_embed_dim": 384, "text_proj_dim": 64,
        "latent_dim": 128,
        "phase3_ckpt": "backend/checkpoints/phase3_best.pt",
    }
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nBuilding KNN Latent Index  |  Device: {DEVICE}\n")

    tpgb = TPGB(num_snapshots=5).to(DEVICE)
    ste  = STE(
        node_feat_dim=CFG["node_feat_dim"], edge_feat_dim=CFG["edge_feat_dim"],
        gat_hidden_dim=CFG["gat_hidden_dim"], gat_num_heads=CFG["gat_num_heads"],
        gat_out_dim=CFG["gat_out_dim"], pvt_out_dim=CFG["pvt_out_dim"],
        snapshot_embed_dim=CFG["snapshot_embed_dim"],
        transformer_heads=CFG["transformer_heads"],
        transformer_layers=CFG["transformer_layers"],
        dropout=CFG["dropout"],
    ).to(DEVICE)
    vle  = VLE(
        graph_embed_dim=CFG["snapshot_embed_dim"],
        text_embed_dim=CFG["text_embed_dim"],
        text_proj_dim=CFG["text_proj_dim"],
        latent_dim=CFG["latent_dim"], dropout=CFG["dropout"],
    ).to(DEVICE)

    ckpt3 = torch.load(CFG["phase3_ckpt"], map_location=DEVICE, weights_only=False)
    ste.load_state_dict(ckpt3["ste_state"])
    vle.load_state_dict(ckpt3["vle_state"])
    for m in (tpgb, ste, vle):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    train_ds     = PropagationDataset("twitter15", split="train")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)

    scorer = LatentKNNScorer(k=15)
    scorer.build_index(train_loader, tpgb, ste, vle, DEVICE)
    os.makedirs("results", exist_ok=True)
    scorer.save("results/knn_index.pt")
    print("\nDone. Adaptive KNN scoring is now available in streamlit_app.py")


if __name__ == "__main__":
    build_knn_index()

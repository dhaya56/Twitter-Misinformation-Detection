"""
streamlit_app.py  [v4.1 - Bug-fixed]
--------------------------------------
CGDEX-Net: Spatio-Temporal Misinformation Detection

Fixes vs v4:
    - STE snapshot unpack: TPGB returns 4-tuples not 5
    - simulator data: auto-computes snapshot_edge_mask if missing
    - empty label warnings: all widgets now have non-empty labels

TWO INPUT MODES:
    1. Dataset Claim ID  - full CGDEX-Net on real Twitter15/16 graphs
    2. Simulate Tweet    - Hawkes + preferential-attachment simulator

Run:
    streamlit run streamlit_app.py
"""

import os, sys, json
from datetime import datetime
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as PyGData

from backend.data.dataset_loader        import PropagationDataset, to_propagation_data
from backend.data.propagation_simulator import PropagationSimulator
from backend.models.tpgb                import TPGB
from backend.models.ste                 import STE
from backend.models.vle                 import VLE
from backend.models.cdp                 import CDP
from backend.models.ccm                 import CCM
from backend.models.dsh                 import DSH
from backend.models.adaptive_scorer     import LatentKNNScorer
from backend.models.community_analysis  import CommunityAnalyzer

st.set_page_config(page_title="CGDEX-Net", page_icon="C",
                   layout="wide", initial_sidebar_state="expanded")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = {
    "node_feat_dim": 6,  "edge_feat_dim": 3,
    "gat_hidden_dim": 64, "gat_num_heads": 4,
    "gat_out_dim": 96,   "pvt_out_dim": 32,
    "snapshot_embed_dim": 128, "transformer_heads": 4,
    "transformer_layers": 2,   "dropout": 0.0,
    "text_embed_dim": 384,     "text_proj_dim": 64,
    "latent_dim": 128, "T": 200, "T_inf": 10,
    "t_star": 40, "cdp_hidden_dim": 256,
    "num_completions": 4, "tau_max": 11786.27,
    "phase2_ckpt": "backend/checkpoints/phase2_best.pt",
    "phase3_ckpt": "backend/checkpoints/phase3_best.pt",
}

SIM_HORIZONS = [1, 5, 15, 30, 60, 360, 720, 1440]

THEME = {
    "app_bg": "#f3f6fb",
    "card_bg": "#ffffff",
    "text": "#0f172a",
    "muted": "#475569",
    "border": "#d7dee8",
}

VS = {
    "MISINFORMATION": {"bg": "#fff2f0", "brd": "#e74c3c", "col": "#b42318", "icon": "[!]"},
    "CREDIBLE": {"bg": "#ecfdf3", "brd": "#27ae60", "col": "#166534", "icon": "[OK]"},
    "UNCERTAIN": {"bg": "#fff7ed", "brd": "#e67e22", "col": "#9a3412", "icon": "[?]"},
}


# ── helpers ─────────────────────────────────────────────

def _add_snapshot_edge_mask(pyg_data):
    """
    Computes and attaches snapshot_edge_mask if it is missing.
    TPGB needs this: edge e is active in snapshot k iff both endpoints are.
    Expected snapshot_node_mask shape is [S, N]; output is [S, E].
    Also supports simulator-style [N, S] by transposing.
    """
    if hasattr(pyg_data, "snapshot_edge_mask"):
        return pyg_data

    snm = getattr(pyg_data, "snapshot_node_mask", None)
    ei  = getattr(pyg_data, "edge_index", None)

    if snm is None:
        raise ValueError("Cannot build snapshot_edge_mask without snapshot_node_mask.")
    if snm.dim() != 2:
        raise ValueError(f"snapshot_node_mask must be 2D, got {tuple(snm.shape)}")

    num_snapshots = 5
    N = int(pyg_data.x.shape[0]) if hasattr(pyg_data, "x") else snm.shape[-1]
    if snm.shape[0] == num_snapshots and snm.shape[1] == N:
        snm_fixed = snm
    elif snm.shape[1] == num_snapshots and snm.shape[0] == N:
        snm_fixed = snm.t()
    elif snm.shape[0] == num_snapshots:
        snm_fixed = snm
    elif snm.shape[1] == num_snapshots:
        snm_fixed = snm.t()
    else:
        raise ValueError(
            f"snapshot_node_mask shape {tuple(snm.shape)} is incompatible "
            f"with num_snapshots={num_snapshots} and num_nodes={N}."
        )

    if ei is None or ei.numel() == 0:
        S = int(snm_fixed.shape[0])
        pyg_data.snapshot_edge_mask = torch.zeros((S, 0), dtype=torch.bool, device=snm_fixed.device)
        return pyg_data

    src = ei[0]
    dst = ei[1]
    pyg_data.snapshot_edge_mask = (snm_fixed[:, src] & snm_fixed[:, dst]).bool()
    return pyg_data


def _normalize_snapshot_layout(pyg_data, num_snapshots=5):
    """
    Ensures snapshot masks match TPGB expectations:
      snapshot_node_mask : [S, N]
      snapshot_edge_mask : [S, E]

    Handles simulator-style [N, S] / [E, S] inputs as well.
    """
    snm = getattr(pyg_data, "snapshot_node_mask", None)
    if snm is None:
        raise ValueError("Input data is missing snapshot_node_mask.")
    if snm.dim() != 2:
        raise ValueError(f"snapshot_node_mask must be 2D, got {tuple(snm.shape)}")

    N = int(pyg_data.x.shape[0]) if hasattr(pyg_data, "x") else snm.shape[-1]
    if snm.shape[0] == num_snapshots and snm.shape[1] == N:
        snm_fixed = snm
    elif snm.shape[1] == num_snapshots and snm.shape[0] == N:
        snm_fixed = snm.t()
    elif snm.shape[0] == num_snapshots:
        snm_fixed = snm
    elif snm.shape[1] == num_snapshots:
        snm_fixed = snm.t()
    else:
        raise ValueError(
            f"snapshot_node_mask shape {tuple(snm.shape)} is incompatible "
            f"with num_snapshots={num_snapshots} and num_nodes={N}."
        )
    pyg_data.snapshot_node_mask = snm_fixed.bool()

    if hasattr(pyg_data, "snapshot_edge_mask"):
        sem = pyg_data.snapshot_edge_mask
        if sem.dim() != 2:
            raise ValueError(f"snapshot_edge_mask must be 2D, got {tuple(sem.shape)}")
        E = int(pyg_data.edge_index.shape[1]) if hasattr(pyg_data, "edge_index") else sem.shape[-1]
        if sem.shape[0] == num_snapshots and sem.shape[1] == E:
            sem_fixed = sem
        elif sem.shape[1] == num_snapshots and sem.shape[0] == E:
            sem_fixed = sem.t()
        elif sem.shape[0] == num_snapshots:
            sem_fixed = sem
        elif sem.shape[1] == num_snapshots:
            sem_fixed = sem.t()
        else:
            sem_fixed = None

        if sem_fixed is None or sem_fixed.shape[1] != E:
            src = pyg_data.edge_index[0]
            dst = pyg_data.edge_index[1]
            sem_fixed = pyg_data.snapshot_node_mask[:, src] & pyg_data.snapshot_node_mask[:, dst]
        pyg_data.snapshot_edge_mask = sem_fixed.bool()

    return pyg_data


def _prepare_snapshots_for_ste(snapshots):
    """
    Converts TPGB snapshots to STE's expected tuple format:
        (x, edge_index, edge_attr, batch_ptr, mask)
    """
    converted = []
    for snap in snapshots:
        if isinstance(snap, dict):
            x = snap["x"]
            edge_index = snap["edge_index"]
            edge_attr = snap.get("edge_attr", None)
            batch_ptr = snap.get("batch", None)
            mask = snap.get("mask", None)
        elif isinstance(snap, (tuple, list)):
            if len(snap) == 5:
                x, edge_index, edge_attr, batch_ptr, mask = snap
            elif len(snap) == 4:
                x, edge_index, edge_attr, batch_ptr = snap
                mask = None
            else:
                raise ValueError(f"Unsupported snapshot tuple length: {len(snap)}")
        else:
            raise TypeError(f"Unsupported snapshot type: {type(snap)}")

        if batch_ptr is None:
            batch_ptr = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if edge_attr is None:
            edge_attr = torch.zeros(
                (edge_index.shape[1], CFG["edge_feat_dim"]),
                dtype=x.dtype,
                device=x.device,
            )
        if mask is None:
            mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        converted.append((x, edge_index, edge_attr, batch_ptr, mask))
    return converted


def _prepare_pvt_for_ste(pvt):
    """
    STE expects [B, pvt_out_dim]. TPGB may return [B, S, 3].
    """
    if pvt.dim() == 3:
        pvt = pvt.reshape(pvt.shape[0], -1)
    if pvt.dim() != 2:
        raise ValueError(f"Unsupported PVT shape: {tuple(pvt.shape)}")

    target_dim = int(CFG["pvt_out_dim"])
    cur_dim = int(pvt.shape[1])
    if cur_dim == target_dim:
        return pvt
    if cur_dim > target_dim:
        return pvt[:, :target_dim]

    pad = torch.zeros(
        (pvt.shape[0], target_dim - cur_dim),
        dtype=pvt.dtype,
        device=pvt.device,
    )
    return torch.cat([pvt, pad], dim=1)


def _closest_horizon_key(per_horizon, tau):
    """
    Returns the best horizon key for tau from calibration dict keys.
    """
    if not per_horizon:
        return "full"
    if tau is None:
        return "full" if "full" in per_horizon else next(iter(per_horizon.keys()))

    target = int(float(tau))
    if str(target) in per_horizon:
        return str(target)

    numeric = []
    for k in per_horizon.keys():
        if k.isdigit():
            numeric.append(int(k))
    if not numeric:
        return "full" if "full" in per_horizon else next(iter(per_horizon.keys()))
    nearest = min(numeric, key=lambda x: abs(x - target))
    return str(nearest)


def _resolve_backbone_threshold(calibration, tau, sigma2_z):
    """
    Dynamic thresholding based on calibration horizon + per-input uncertainty.
    Final decision still comes from backbone score only.
    """
    threshold = 0.5
    unc_lo = 0.46
    unc_hi = 0.54
    source = "default"
    sigma2_shift_dir = 0.0  # +1 raises threshold for high sigma2, -1 lowers it

    if calibration:
        per_h = calibration.get("per_horizon_thresholds", {})
        if per_h:
            key = _closest_horizon_key(per_h, tau)
            cfg = per_h.get(key) or per_h.get("full") or {}
            threshold = float(cfg.get("threshold", threshold))
            unc_lo = float(cfg.get("unc_lo", threshold - 0.05))
            unc_hi = float(cfg.get("unc_hi", threshold + 0.05))
            source = f"calibration:{key}"

        # Determine sigma2_z direction from calibration statistics.
        # If misinfo has higher sigma2_z than credible, high sigma2 should LOWER threshold.
        sig_stats = ((calibration.get("feature_stats") or {}).get("sigma2_z") or {})
        c_mean = sig_stats.get("credible_mean")
        m_mean = sig_stats.get("misinfo_mean")
        if c_mean is not None and m_mean is not None and abs(float(m_mean) - float(c_mean)) > 1e-9:
            sigma2_shift_dir = -1.0 if float(m_mean) > float(c_mean) else 1.0

    sep = None
    if calibration:
        sep = (((calibration.get("feature_stats") or {}).get("sigma2_z") or {}).get("separator"))

    if sep is not None and sigma2_shift_dir != 0.0:
        sep = max(float(sep), 1e-6)
        rel_unc = (float(sigma2_z) - sep) / sep

        # Widen confidence band for high uncertainty; shift threshold in calibrated direction.
        widen = max(0.0, min(rel_unc, 2.0)) * 0.04
        shift = sigma2_shift_dir * max(-1.0, min(rel_unc, 2.0)) * 0.015

        threshold = min(max(threshold + shift, 0.05), 0.95)
        unc_lo = max(0.0, unc_lo - widen)
        unc_hi = min(1.0, unc_hi + widen)
        source = source + ":sigma2_adjusted"

    if unc_lo > unc_hi:
        unc_lo, unc_hi = unc_hi, unc_lo
    if threshold < unc_lo:
        threshold = unc_lo
    if threshold > unc_hi:
        threshold = unc_hi

    return {
        "threshold": round(float(threshold), 4),
        "unc_lo": round(float(unc_lo), 4),
        "unc_hi": round(float(unc_hi), 4),
        "source": source,
    }


def _save_moderation_event(claim_label, result, community):
    os.makedirs("results/moderation", exist_ok=True)
    out_path = os.path.join("results", "moderation", "moderation_queue.jsonl")
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "claim": claim_label,
        "verdict": result.get("verdict"),
        "score": result.get("score"),
        "true_label": result.get("true_label"),
        "decision": result.get("decision"),
        "features": result.get("features"),
        "community": {
            "pattern_verdict": community.get("pattern_verdict"),
            "hawkes_branching_ratio": community.get("hawkes_branching_ratio"),
            "spatial_risk_flags": community.get("spatial_risk_flags"),
        },
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return out_path


# ── model loading ─────────────────────────────────────────────

@st.cache_resource
def load_everything():
    tpgb = TPGB(num_snapshots=5).to(DEVICE)
    ste  = STE(
        node_feat_dim=CFG["node_feat_dim"],   edge_feat_dim=CFG["edge_feat_dim"],
        gat_hidden_dim=CFG["gat_hidden_dim"], gat_num_heads=CFG["gat_num_heads"],
        gat_out_dim=CFG["gat_out_dim"],       pvt_out_dim=CFG["pvt_out_dim"],
        snapshot_embed_dim=CFG["snapshot_embed_dim"],
        transformer_heads=CFG["transformer_heads"],
        transformer_layers=CFG["transformer_layers"],
        dropout=CFG["dropout"],
    ).to(DEVICE)
    vle  = VLE(
        graph_embed_dim=CFG["snapshot_embed_dim"],
        text_embed_dim=CFG["text_embed_dim"],
        text_proj_dim=CFG["text_proj_dim"],
        latent_dim=CFG["latent_dim"],
        dropout=CFG["dropout"],
    ).to(DEVICE)
    cdp  = CDP(
        latent_dim=CFG["latent_dim"], hidden_dim=CFG["cdp_hidden_dim"],
        T=CFG["T"], T_inf=CFG["T_inf"], t_star=CFG["t_star"],
    ).to(DEVICE)
    dsh  = DSH(input_dim=4, hidden_dim=64).to(DEVICE)

    ck2 = torch.load(CFG["phase2_ckpt"], map_location=DEVICE, weights_only=False)
    ck3 = torch.load(CFG["phase3_ckpt"], map_location=DEVICE, weights_only=False)
    ste.load_state_dict(ck3["ste_state"], strict=False)
    vle.load_state_dict(ck3["vle_state"])
    cdp.load_state_dict(ck2["cdp_state"])
    dsh.load_state_dict(ck3["dsh_state"])
    for m in (tpgb, ste, vle, cdp, dsh):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    ccm = CCM(cdp=cdp, num_completions=CFG["num_completions"], tau_max=CFG["tau_max"])

    calibration = {}
    if os.path.exists("results/calibration.json"):
        with open("results/calibration.json") as f:
            calibration = json.load(f)

    scorer = None
    if os.path.exists("results/knn_index.pt"):
        scorer = LatentKNNScorer.load("results/knn_index.pt")

    return (tpgb, ste, vle, cdp, ccm, dsh, calibration, scorer,
            PropagationSimulator(), CommunityAnalyzer())


# ── inference ─────────────────────────────────────────────

@torch.no_grad()
def run_inference(tpgb, ste, vle, cdp, ccm, dsh, scorer, src, tau=None, calibration=None):
    """
    src: file-path string  OR  dict from PropagationSimulator.
    """
    if isinstance(src, str):
        raw  = torch.load(src, weights_only=False)
        data = to_propagation_data(raw)
    else:
        # build PyG Data from simulator dict
        tmp = PyGData()
        for k, v in src.items():
            if isinstance(v, torch.Tensor):
                tmp[k] = v
            elif isinstance(v, (int, float, str, bool)):
                tmp[k] = v
        # Ensure truncation metadata exists for simulator inputs.
        if not hasattr(tmp, "max_timestamp"):
            tmp.max_timestamp = float(max(SIM_HORIZONS))
        if not hasattr(tmp, "max_tree_depth"):
            tmp.max_tree_depth = 0
        data = to_propagation_data(tmp)

    data = _normalize_snapshot_layout(data, num_snapshots=5)
    data = _add_snapshot_edge_mask(data)
    data = _normalize_snapshot_layout(data, num_snapshots=5)

    if tau is not None:
        if not hasattr(data, "max_timestamp"):
            data.max_timestamp = float(max(SIM_HORIZONS))
        if not hasattr(data, "max_tree_depth"):
            data.max_tree_depth = 0
        data = PropagationDataset.__new__(PropagationDataset)._truncate_to_tau(
            data, float(tau))
        data = _normalize_snapshot_layout(data, num_snapshots=5)

    batch   = next(iter(DataLoader([data], batch_size=1))).to(DEVICE)
    tau_obs = float(tau) if tau is not None else -1.0

    snaps_raw, pvt_raw   = tpgb(batch)
    snaps                = _prepare_snapshots_for_ste(snaps_raw)
    pvt                  = _prepare_pvt_for_ste(pvt_raw)
    h                    = ste(snaps, pvt)
    z, mu_z, sig_z, sig2 = vle(h, batch.root_text_emb, training=False)
    _, d_m               = cdp.project_to_manifold(mu_z)
    d_cv, _              = ccm(mu_z, tau_obs=tau_obs)
    d_tr                 = dsh.compute_d_tr(d_m, torch.zeros_like(d_m))
    raw_s, _             = dsh(d_m, d_cv, d_tr, sig2)
    base_score = float(raw_s.item())
    threshold_info = _resolve_backbone_threshold(
        calibration=calibration,
        tau=tau,
        sigma2_z=float(sig2.item()),
    )

    verdict = ("MISINFORMATION"
               if base_score >= threshold_info["threshold"]
               else "CREDIBLE")
    low_confidence = (threshold_info["unc_lo"] <= base_score <= threshold_info["unc_hi"])
    decision_meta = dict(threshold_info)
    decision_meta["low_confidence"] = bool(low_confidence)
    decision_meta["policy"] = "binary_threshold"

    score = base_score
    decision_source = "backbone_dynamic_threshold_binary"

    # KNN is auxiliary explanation only; it never overrides backbone decision.
    ada = scorer.score(mu_z, sig2) if scorer else None
    if ada:
        ada["used_for_decision"] = False
        ada["backbone_score"] = round(base_score, 4)
        ada["backbone_verdict"] = verdict

    y   = data.y.item() if hasattr(data, "y") and data.y.numel() > 0 else -1
    lbl = "unknown" if y == -1 else ("misinfo" if y == 1 else "credible")
    correct = None if y == -1 else ((verdict == "MISINFORMATION") == (y == 1))

    return dict(
        verdict=verdict, score=round(score, 4),
        raw_score=round(base_score, 4), true_label=lbl,
        correct=correct,
        decision_source=decision_source,
        decision=decision_meta,
        num_nodes=data.num_nodes, num_edges=data.num_edges,
        features=dict(d_m=d_m.item(), d_cv=d_cv.item(),
                      d_tr=d_tr.item(), sigma2_z=sig2.item()),
        max_timestamp=float(getattr(data, "max_timestamp", max(SIM_HORIZONS))),
        adaptive=ada, data=data,
        edge_index=data.edge_index.numpy() if data.edge_index.numel() > 0 else None,
        node_feats=data.x.numpy(),
    )


# ── D3 propagation graph ─────────────────────────────────────────────

def propagation_html(edge_index, node_feats, label="", verdict="",
                     communities=None, tau=None, node_depths=None,
                     node_reach=None, timeline_stats=None):
    N_cap = min(node_feats.shape[0], 150)
    raw_edges = []
    if edge_index is not None and edge_index.shape[1] > 0:
        for i in range(edge_index.shape[1]):
            s, t = int(edge_index[0, i]), int(edge_index[1, i])
            if 0 <= s < N_cap and 0 <= t < N_cap and s != t:
                raw_edges.append((s, t))

    # Force a single-parent tree view so path-to-root is always valid.
    parent = {}
    for s, t in raw_edges:
        if t == 0:
            continue
        cur = parent.get(t)
        if cur is None:
            parent[t] = s
        else:
            # choose earlier source as canonical parent
            if float(node_feats[s, 0]) < float(node_feats[cur, 0]):
                parent[t] = s

    tree_edges = [{"source": s, "target": t} for t, s in parent.items()]

    # Keep only nodes reachable from source in this tree view.
    children = {}
    for e in tree_edges:
        children.setdefault(e["source"], []).append(e["target"])
    reachable = set([0]) if N_cap > 0 else set()
    q = [0] if N_cap > 0 else []
    while q:
        u = q.pop(0)
        for v in children.get(u, []):
            if v not in reachable:
                reachable.add(v)
                q.append(v)
    if not reachable:
        reachable = set(range(N_cap))

    kept_nodes = sorted(reachable)
    old_to_new = {old: new for new, old in enumerate(kept_nodes)}
    N = len(kept_nodes)
    edges = []
    for e in tree_edges:
        s_old, t_old = e["source"], e["target"]
        if s_old in old_to_new and t_old in old_to_new:
            edges.append({"source": old_to_new[s_old], "target": old_to_new[t_old]})

    if node_depths is None:
        node_depths = [0] * node_feats.shape[0]
    if node_reach is None:
        node_reach = [0] * node_feats.shape[0]

    CCOLORS = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
               "#a65628","#f781bf","#999999","#66c2a5","#fc8d62",
               "#8da0cb","#e78ac3"]

    nodes = []
    for i, old_i in enumerate(kept_nodes):
        tn = float(node_feats[old_i, 0])
        r  = int(255 * min(tn * 2, 1.0))
        g  = int(255 * (1 - abs(tn - 0.5) * 2))
        b  = int(255 * max(1 - tn * 2, 0.0))
        tc = f"rgb({r},{g},{b})"
        cc = CCOLORS[(communities.get(old_i, 0) if communities else 0) % len(CCOLORS)]
        depth_i = int(node_depths[old_i]) if old_i < len(node_depths) else 0
        reach_i = int(node_reach[old_i]) if old_i < len(node_reach) else 0
        nodes.append({"id":i,"oid":old_i,"time":round(tn,4),"depth":depth_i,
                      "reach":reach_i,"tc":tc,"cc":cc,"isRoot":i==0,
                      "pct":f"{tn*100:.1f}%"})

    NJ = json.dumps(nodes)
    EJ = json.dumps(edges)
    TSJ = json.dumps(timeline_stats or [])
    vc = {"MISINFORMATION":"#e74c3c","CREDIBLE":"#27ae60",
          "UNCERTAIN":"#e67e22"}.get(verdict, "#3498db")
    title = label + (f"  (tau={tau} min)" if tau else "")

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#f6f8fc;font-family:'Segoe UI',sans-serif;color:#0f172a;overflow:hidden}}
#hdr{{padding:7px 14px;background:#ffffff;border-bottom:1px solid #d7dee8;
      display:flex;align-items:center;justify-content:space-between}}
#htitle{{font-size:12px;font-weight:700;color:#0f172a}}
#badge{{background:{vc};color:#fff;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700}}
#ctrl{{padding:6px 14px;background:#ffffff;border-bottom:1px solid #d7dee8;
       display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
#ctrl label{{font-size:11px;color:#475569}}
#sld{{width:180px;accent-color:{vc}}}
#slbl{{font-size:11px;color:#0f172a;min-width:60px}}
.btn{{border:none;padding:4px 11px;border-radius:4px;cursor:pointer;font-size:11px;font-weight:600}}
#pb{{background:{vc};color:#fff}}
#cb,#rb{{background:#e2e8f0;color:#0f172a}}
#stl{{font-size:11px;color:#475569;margin-left:auto}}
#tip{{position:absolute;background:#ffffff;border:1px solid #d7dee8;border-radius:6px;
      padding:8px 11px;font-size:11px;pointer-events:none;opacity:0;transition:opacity .15s;
      z-index:99;max-width:200px;line-height:1.65}}
#tip .tn{{color:#0f172a;font-weight:700;margin-bottom:3px}}
#tip .tr{{color:#475569}} #tip .tr span{{color:#1f2937}}
#pb2{{position:absolute;top:94px;left:14px;background:#ffffffee;border:1px solid {vc};
      border-radius:6px;padding:8px 11px;font-size:11px;display:none;max-width:210px}}
.pt{{color:{vc};font-weight:700;margin-bottom:4px}}
#leg{{position:absolute;bottom:10px;right:14px;background:#ffffffee;
      border:1px solid #d7dee8;border-radius:6px;padding:8px 11px;font-size:10px}}
.lr{{display:flex;align-items:center;gap:6px;margin:2px 0}}
.ld{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
</style></head><body>
<div id="hdr"><span id="htitle">{title}</span><span id="badge">{verdict}</span></div>
<div id="ctrl">
  <label for="sld">Time:</label>
  <input type="range" id="sld" min="0" max="100" value="100">
  <span id="slbl">100%</span>
  <button class="btn" id="pb" onclick="togglePlay()">Play</button>
  <button class="btn" id="cb" onclick="toggleComm()">Community</button>
  <button class="btn" id="rb" onclick="resetV()">Reset</button>
  <span id="stl">Active: {N}/{N} nodes | {len(edges)}/{len(edges)} edges</span>
</div>
<div id="tip">
  <div class="tn" id="tn"></div>
  <div class="tr">Time: <span id="tt"></span></div>
  <div class="tr">Depth: <span id="td"></span></div>
  <div class="tr">Reach: <span id="tr2"></span> downstream</div>
</div>
<div id="pb2"><div class="pt">Path to Source</div><div id="pc"></div></div>
<div id="leg">
  <div class="lr"><div class="ld" style="background:#FFD700;border:2px solid white"></div>Source</div>
  <div class="lr"><div class="ld" style="background:rgb(0,180,255)"></div>Early spread</div>
  <div class="lr"><div class="ld" style="background:rgb(255,220,0)"></div>Mid spread</div>
  <div class="lr"><div class="ld" style="background:rgb(255,60,0)"></div>Late spread</div>
  <div class="lr"><div class="ld" style="background:#d7dee8;border:1px solid #94a3b8"></div>Not active yet</div>
  <div style="margin-top:4px;color:#475569">Click node -> trace to source<br>Node size = reach</div>
</div>
<svg id="cv"></svg>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const ND={NJ}, ED={EJ}, TS={TSJ};
let tau=1, playing=false, timer=null, showC=false;
const W=window.innerWidth, H=window.innerHeight-76;
const svg=d3.select("#cv").attr("width",W).attr("height",H);
svg.append("defs").append("marker").attr("id","ar").attr("viewBox","0 -4 8 8")
   .attr("refX",18).attr("refY",0).attr("markerWidth",6).attr("markerHeight",6)
   .attr("orient","auto").append("path").attr("d","M0,-4L8,0L0,4").attr("fill","#94a3b8");
const g=svg.append("g");
svg.call(d3.zoom().scaleExtent([0.08,8]).on("zoom",e=>g.attr("transform",e.transform)));
const par={{}}, ch2={{}};
ED.forEach(e=>{{
  par[e.target]=e.source;
  if(!ch2[e.source]) ch2[e.source]=[];
  ch2[e.source].push(e.target);
}});
function ptr(id){{const p=[];let c=id;while(c!==undefined){{p.unshift(c);c=par[c];}}return p;}}
ND.forEach(n=>{{n.sz=Math.max(1,(n.reach||0)+1);n.r=Math.max(7,Math.min(30,7+Math.sqrt(n.sz)*3));}});
const sim=d3.forceSimulation(ND)
  .force("link",d3.forceLink(ED).id(d=>d.id).distance(55).strength(0.55))
  .force("charge",d3.forceManyBody().strength(-150))
  .force("center",d3.forceCenter(W/2,H/2))
  .force("coll",d3.forceCollide().radius(d=>d.r+5));
const lnk=g.append("g").selectAll("line").data(ED).join("line")
  .attr("stroke","#c3cfdd").attr("stroke-width",1.5).attr("marker-end","url(#ar)");
const ndg=g.append("g").selectAll("g").data(ND).join("g")
  .call(d3.drag()
    .on("start",(e,d)=>{{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;}})
    .on("drag",(e,d)=>{{d.fx=e.x;d.fy=e.y;}})
    .on("end",(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}));
function sp(R,r,n){{
  let s="";
  for(let i=0;i<n*2;i++){{
    const a=(i*Math.PI)/n-Math.PI/2, rv=i%2===0?R:r;
    s+=(i?" ":"")+rv*Math.cos(a)+","+rv*Math.sin(a);
  }}
  return s;
}}
const sh=ndg.append(d=>{{
  if(d.isRoot){{
    const s=document.createElementNS("http://www.w3.org/2000/svg","polygon");
    s.setAttribute("points",sp(d.r,d.r*0.42,5));
    s.setAttribute("fill","#FFD700");s.setAttribute("stroke","white");
    s.setAttribute("stroke-width","2");return s;
  }}
  const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
  c.setAttribute("r",d.r/2);c.setAttribute("fill",d.tc);
  c.setAttribute("stroke","#334155");c.setAttribute("stroke-width","1.5");return c;
}});
ndg.filter(d=>d.isRoot).append("circle").attr("r",24).attr("fill","none")
   .attr("stroke","#FFD700").attr("stroke-width",2).attr("opacity",0.4);
ndg.filter(d=>d.isRoot).append("text")
   .attr("dy",d=>d.r+12).attr("text-anchor","middle")
   .attr("fill","#FFD700").attr("font-size","10px").attr("font-weight","700").text("SOURCE");
ndg
  .on("mouseover",(e,d)=>{{
    document.getElementById("tn").textContent=d.isRoot?"SOURCE TWEET":"User "+d.oid;
    document.getElementById("tt").textContent=d.pct;
    document.getElementById("td").textContent=d.depth;
    document.getElementById("tr2").textContent=d.reach||0;
    const t=document.getElementById("tip");
    t.style.left=(e.pageX+12)+"px";t.style.top=(e.pageY-8)+"px";t.style.opacity=1;
    document.getElementById("stl").textContent =
      (d.isRoot
        ? "Hover: SOURCE node"
        : `Hover: User ${{d.oid}} | depth ${{d.depth}} | downstream ${{d.reach||0}}`);
  }})
  .on("mousemove",e=>{{
    const t=document.getElementById("tip");
    t.style.left=(e.pageX+12)+"px";t.style.top=(e.pageY-8)+"px";
  }})
  .on("mouseout",()=>{{
    document.getElementById("tip").style.opacity=0;
    updateStats(tau);
  }})
  .on("click",(e,d)=>{{
    e.stopPropagation();
    const path=ptr(d.id);
    hlPath(path);
    document.getElementById("pc").innerHTML=path.map((nid,i)=>{{
      const n=ND[nid];
      return`<div style="color:#334155">${{i===0?"ROOT":"->"}} ${{nid===0?"SOURCE":"User "+n.oid}}
             <span style="color:#0f172a">@ ${{(n.time*100).toFixed(1)}}% | d=${{n.depth}} | r=${{n.reach||0}}</span></div>`;
    }}).join("");
    document.getElementById("pb2").style.display="block";
  }});
svg.on("click",()=>{{clearHL();document.getElementById("pb2").style.display="none";}});
sim.on("tick",()=>{{
  lnk.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
     .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  ndg.attr("transform",d=>`translate(${{d.x}},${{d.y}})`);
}});
document.getElementById("sld").addEventListener("input",function(){{
  tau=this.value/100;
  document.getElementById("slbl").textContent=this.value+"%";
  applyT(tau);
}});
function applyT(t){{
  sh.each(function(d){{
    if(d.isRoot)return;
    const ok=d.time<=t;
    d3.select(this).attr("opacity",ok?1:0.08)
                   .attr("fill",ok?(showC?d.cc:d.tc):"#d7dee8");
  }});
  lnk.attr("opacity",d=>{{
    const s=ND[typeof d.source==="object"?d.source.id:d.source]?.time??0;
    const t2=ND[typeof d.target==="object"?d.target.id:d.target]?.time??0;
    return(s<=t&&t2<=t)?0.65:0.04;
  }});
  updateStats(t);
}}
function updateStats(t){{
  const idx = Math.max(0, Math.min(100, Math.round(t * 100)));
  const s = (Array.isArray(TS) && TS.length > idx) ? TS[idx] : null;
  if (s){{
    document.getElementById("stl").textContent =
      `Active: ${{s.active_nodes}}/${{ND.length}} nodes | ` +
      `${{s.active_edges}}/${{ED.length}} edges | ` +
      `depth: ${{s.max_depth}} | width: ${{s.max_width}} | comms: ${{s.active_communities}}`;
    return;
  }}
  const activeNodes = ND.filter(n=>n.time<=t).length;
  const activeEdges = ED.filter(e=>{{
    const s=ND[typeof e.source==="object"?e.source.id:e.source]?.time??0;
    const d=ND[typeof e.target==="object"?e.target.id:e.target]?.time??0;
    return s<=t && d<=t;
  }}).length;
  const activeComms = new Set(
    ND.filter(n=>n.time<=t).map(n=>n.cc)
  ).size;
  const activeDepths = ND.filter(n=>n.time<=t).map(n=>n.depth||0);
  const maxDepth = activeDepths.length ? Math.max(...activeDepths) : 0;
  const depthCounts = {{}};
  activeDepths.forEach(d=>{{depthCounts[d]=(depthCounts[d]||0)+1;}});
  const maxWidth = Object.keys(depthCounts).length
    ? Math.max(...Object.values(depthCounts))
    : 0;
  document.getElementById("stl").textContent =
    `Active: ${{activeNodes}}/${{ND.length}} nodes | ` +
    `${{activeEdges}}/${{ED.length}} edges | ` +
    `depth: ${{maxDepth}} | width: ${{maxWidth}} | comms: ${{activeComms}}`;
}}
function toggleComm(){{
  showC=!showC;
  document.getElementById("cb").style.background=showC?"#8b5cf6":"#e2e8f0";
  applyT(tau);
}}
function hlPath(path){{
  const ps=new Set(path);
  sh.each(function(d){{
    if(ps.has(d.id)) d3.select(this).attr("stroke","#0f172a").attr("stroke-width",3);
    else d3.select(this).attr("opacity",0.08);
  }});
  lnk.attr("stroke",d=>{{
    const s=typeof d.source==="object"?d.source.id:d.source;
    const t=typeof d.target==="object"?d.target.id:d.target;
    return(ps.has(s)&&ps.has(t))?"#0f172a":"#cbd5e1";
  }}).attr("stroke-width",d=>{{
    const s=typeof d.source==="object"?d.source.id:d.source;
    const t=typeof d.target==="object"?d.target.id:d.target;
    return(ps.has(s)&&ps.has(t))?3:0.5;
  }});
}}
function clearHL(){{
  sh.each(function(){{
    d3.select(this).attr("opacity",1).attr("stroke","#334155").attr("stroke-width","1.5");
  }});
  lnk.attr("stroke","#c3cfdd").attr("stroke-width",1.5).attr("opacity",0.65);
  applyT(tau);
}}
function togglePlay(){{
  playing=!playing;
  document.getElementById("pb").textContent=playing?"Stop":"Play";
  if(playing){{
    let v=0;document.getElementById("sld").value=0;
    timer=setInterval(()=>{{
      v+=1;if(v>100)v=0;
      document.getElementById("sld").value=v;
      tau=v/100;document.getElementById("slbl").textContent=v+"%";
      applyT(tau);
    }},70);
  }}else clearInterval(timer);
}}
function resetV(){{
  document.getElementById("sld").value=100;tau=1;
  document.getElementById("slbl").textContent="100%";
  applyT(1);clearHL();
  document.getElementById("pb2").style.display="none";
  svg.transition().duration(400).call(d3.zoom().transform,d3.zoomIdentity);
  updateStats(1);
}}
applyT(1);
</script></body></html>"""


# ── charts ─────────────────────────────────────────────

def chart_velocity(velocity_hist, tau=None, max_timestamp=None):
    vh = velocity_hist or {}
    cx = np.asarray(vh.get("x", []), dtype=float)
    counts = np.asarray(vh.get("y", []), dtype=float)
    if cx.size == 0 or counts.size == 0:
        cx = np.array([0.0], dtype=float)
        counts = np.array([0.0], dtype=float)

    cols = [f"rgb({int(255*min(t*2,1))},{int(255*(1-abs(t-.5)*2))},{int(255*max(1-t*2,0))})"
            for t in cx]
    fig = go.Figure(go.Bar(
        x=cx, y=counts, marker_color=cols,
        hovertemplate="Time:%{x:.2f}<br>Users:%{y}<extra></extra>"))
    if tau:
        max_t = float(max_timestamp) if max_timestamp else 11786.27
        max_t = max(max_t, 1.0)
        fig.add_vline(x=min(max(float(tau) / max_t, 0.0), 1.0), line_dash="dash",
                      line_color="#334155", annotation_text=f"tau={tau}min",
                      annotation_font_color="#334155")
    fig.update_layout(title="Propagation Velocity", xaxis_title="Normalized time",
                      yaxis_title="New users joining", height=250,
                      margin=dict(l=20,r=20,t=40,b=40),
                      plot_bgcolor="#ffffff",paper_bgcolor="#ffffff",font_color="#0f172a")
    return fig


def chart_depth(depth_hist):
    dh = depth_hist or {}
    depths = np.asarray(dh.get("depths", []), dtype=int)
    counts = np.asarray(dh.get("counts", []), dtype=float)
    if depths.size == 0 or counts.size == 0:
        depths = np.array([0], dtype=int)
        counts = np.array([0], dtype=float)
    cols = ["#FFD700"] + ["#17becf"] * max(len(counts) - 1, 0)
    fig = go.Figure(go.Bar(
        x=depths.tolist(), y=counts.tolist(), marker_color=cols,
        hovertemplate="Depth %{x}: %{y} users<extra></extra>"))
    fig.update_layout(title="Depth Distribution", xaxis_title="Depth from source",
                      yaxis_title="Users", height=230,
                      margin=dict(l=20,r=20,t=40,b=40),
                      plot_bgcolor="#ffffff",paper_bgcolor="#ffffff",font_color="#0f172a")
    return fig


def chart_auc(eh):
    if not eh:
        return None
    taus, aucs = [], []
    for k in ["1","5","15","30","60","360","720","1440","2880","4320"]:
        if k in eh:
            taus.append(int(k)); aucs.append(eh[k]["auc"])
    fig = go.Figure(go.Scatter(
        x=taus, y=aucs, mode="lines+markers",
        line=dict(width=3,color="#27ae60"), marker=dict(size=8,color="#27ae60"),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.1)"))
    fig.update_layout(title="Early Detection AUC vs Observation Window",
                      xaxis_title="Minutes (log)", yaxis_title="AUC",
                      xaxis_type="log", yaxis=dict(range=[0.5, 1.0]), height=260,
                      margin=dict(l=20,r=20,t=40,b=40),
                      plot_bgcolor="#ffffff",paper_bgcolor="#ffffff",font_color="#0f172a")
    return fig


def chart_hawkes(br):
    col = "#e74c3c" if br > .7 else "#e67e22" if br > .4 else "#27ae60"
    lbl = ("Supercritical (coordinated)" if br > .7
           else "Moderate" if br > .4 else "Subcritical (organic)")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=br,
        number={"valueformat":".3f","font":{"size":30}},
        gauge={
            "axis":{"range":[0,1]},
            "bar":{"color":col,"thickness":0.25},
            "steps":[{"range":[0,.4],"color":"rgba(39,174,96,.2)"},
                     {"range":[.4,.7],"color":"rgba(230,126,34,.2)"},
                     {"range":[.7,1], "color":"rgba(231,76,60,.2)"}],
            "threshold":{"line":{"color":"#334155","width":2},"thickness":0.8,"value":0.7},
        },
        title={"text":f"Hawkes Branching Ratio<br><span style='font-size:11px'>{lbl}</span>",
               "font":{"size":12}},
    ))
    fig.update_layout(height=220,margin=dict(l=10,r=10,t=60,b=10),
                      plot_bgcolor="#ffffff",paper_bgcolor="#ffffff",font_color="#0f172a")
    return fig


# ── render helpers ─────────────────────────────────────────────

def render_verdict(res, knn_ok):
    v   = res["verdict"]
    s   = VS.get(v, VS["UNCERTAIN"])
    ada = res.get("adaptive")
    sc  = res["score"]
    ok  = ("" if res["correct"] is None
           else ("Correct" if res["correct"] else "Wrong"))

    d = res.get("decision", {})
    low = d.get("unc_lo", 0.46)
    high = d.get("unc_hi", 0.54)
    thr = d.get("threshold", 0.5)
    low_conf = bool(d.get("low_confidence", False))

    score_line = (
        f"Backbone score: <b>{sc:.4f}</b> &nbsp;|&nbsp; "
        f"dynamic threshold={thr:.4f}<br>"
        f"confidence band: [{low:.4f}, {high:.4f}]"
    )

    if knn_ok and ada:
        score_line += (
            f"<br>KNN context (not used for decision): {ada['knn_score']:.4f} "
            f"({ada['n_misinfo_neighbors']} misinfo / {ada['n_credible_neighbors']} credible)"
        )
        method = "Binary final label from backbone dynamic threshold; KNN shown for XAI only"
    else:
        method = "Binary final label from backbone dynamic threshold"

    st.markdown(
        f"<div style='padding:20px;background:{THEME['card_bg']};border-radius:10px;"
        f"border:2px solid {s['brd']};text-align:center;'>"
        f"<h2 style='color:{s['col']};margin:0 0 8px'>{s['icon']} {v}</h2>"
        f"<p style='font-size:12px;margin:4px 0;color:{THEME['text']}'>{score_line}</p>"
        f"<p style='font-size:10px;color:{THEME['muted']};margin:3px 0'>{method}</p>"
        f"<p style='font-size:12px;margin:4px 0'>"
        f"True label: <b>{res['true_label']}</b> {ok}</p>"
        f"</div>", unsafe_allow_html=True)

    if low_conf:
        st.warning("Score is near the decision boundary. Treat this result as lower confidence.")


def render_knn(ada):
    if not ada:
        st.info("KNN index not built. Run: `python backend/models/adaptive_scorer.py`")
        return
    st.markdown(
        f"Top-5 nearest training neighbors "
        f"(K={ada['k_used']}, sigma2_z={ada['sigma2_raw']:.2f}) - explanation only:")
    rows = [{"Rank":n["rank"],"Label":n["label"].upper(),
              "Distance":f"{n['distance']:.4f}","Weight":f"{n['weight']:.4f}",
              "Contribution":f"{n['contribution']:+.6f}"}
            for n in ada["neighbors"]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(ada["explanation"])


def render_community(c):
    st.markdown("#### Spatial Community Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Communities", c["community_count"])
    c2.metric("Modularity",  f"{c['modularity']:.3f}",
              delta="echo chamber" if c["modularity"] > .3 else None)
    c3.metric("Cross-community edges", f"{c['cross_community_frac']:.0%}")
    c4.metric("Echo chamber score",    f"{c['echo_chamber_score']:.3f}",
              delta="suspicious" if c["echo_chamber_score"] > .4 else None)
    topo = c.get("topology", {})
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Depth std", f"{c['depth_std']:.2f}",
              delta="low = bots" if c["depth_std"] < .5 else None)
    d2.metric("Root centrality", f"{c['root_centrality']:.3f}")
    d3.metric("Leaf fraction", f"{float(topo.get('leaf_fraction', 0.0)):.0%}")
    d4.metric("Root dominance", f"{float(topo.get('root_dominance', 0.0)):.0%}")

    st.markdown("#### Temporal Community Analysis")
    ta = c.get("temporal_analysis", {})
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Early burst (20%)", f"{float(ta.get('early_burst', 0.0)):.0%}",
              delta="suspicious" if float(ta.get("early_burst", 0.0)) > .6 else None)
    t2.metric("Peak snapshot", f"{int(ta.get('peak_snapshot', 1))}/5")
    t3.metric("Burstiness index", f"{float(ta.get('burstiness_index', 0.0)):.3f}")
    t4.metric("Temporal entropy", f"{float(ta.get('temporal_entropy', 0.0)):.3f}")
    u1, u2 = st.columns(2)
    u1.metric("Community timing dispersion",
              f"{float(ta.get('community_temporal_dispersion', 0.0)):.3f}")
    u2.metric("Community activation spread",
              f"{float(ta.get('community_activation_spread', 0.0)):.3f}")

    st.markdown(f"**Pattern verdict:** {c['pattern_verdict']}")
    for flag in c["spatial_risk_flags"]:
        st.warning(f"Spatial: {flag}")
    for flag in c.get("temporal_risk_flags", []):
        st.warning(f"Temporal: {flag}")
    if not c["spatial_risk_flags"]:
        st.success("No strong spatial anomaly flags.")
    if not c.get("temporal_risk_flags", []):
        st.success("No strong temporal anomaly flags.")


# ── main ─────────────────────────────────────────────

import copy

def main():
    st.markdown("""<style>
    .stApp{background:#f3f6fb;color:#0f172a}
    .main .block-container{padding-top:1rem}
    h1,h2,h3,h4{color:#0f172a!important}
    .stTabs [data-baseweb="tab"]{color:#64748b}
    .stTabs [aria-selected="true"]{color:#0f172a}
    div[data-testid="stMetricValue"]{color:#0f172a}
    div[data-testid="stMetricLabel"]{color:#334155}
    </style>""", unsafe_allow_html=True)

    st.title("Spatio-Temporal Detection of Misinformation Propagation Patterns (CGDEX-Net)")
    st.markdown(
        "<span style='color:#334155'>Detects misinformation from "
        "<b>how claims propagate</b> - not just from text. &nbsp;"
        "GAT + Temporal Transformer + Credibility Diffusion Prior (DDPM).</span>",
        unsafe_allow_html=True)

    with st.spinner("Loading models..."):
        (tpgb, ste, vle, cdp, ccm, dsh,
         cal, scorer, simulator, community) = load_everything()

    knn_ok = scorer is not None
    st.session_state.setdefault("moderation_last", "")
    st.session_state.setdefault("warning_label_text", "")

    # ── sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Input Mode")
        mode = st.radio(
            "Select input mode",
            ["Dataset Claim ID", "Simulate Tweet"],
            label_visibility="collapsed")
        st.divider()

        claim_id    = None
        tau_minutes = None
        sim_mode    = None
        sim_nodes   = 80
        sim_seed    = 42

        if mode == "Dataset Claim ID":
            dataset = st.selectbox(
                "Dataset", ["twitter15", "twitter16"])
            claim_id = st.text_input(
                "Claim ID", placeholder="e.g. 80080680482123777")
            tau_map = {
                "Full tree": None, "1 min": 1, "5 min": 5,
                "15 min": 15, "30 min": 30, "60 min": 60,
                "6 hr": 360, "12 hr": 720,
                "1 day": 1440, "2 days": 2880, "3 days": 4320,
            }
            tau_sel     = st.selectbox("Observation window", list(tau_map))
            tau_minutes = tau_map[tau_sel]

        else:
            choice = st.radio(
                "Simulated tweet type",
                ["Misinformation pattern", "Credible pattern"])
            sim_mode  = "misinfo" if "Misinformation" in choice else "credible"
            sim_nodes = st.slider("Number of nodes (users in tree)", 20, 250, 80, 10)
            sim_seed  = st.number_input("Random seed", 0, 9999, 42, 1)
            st.info(
                "**How it works:**  \n"
                "Generates a realistic propagation tree using a **Hawkes "
                "self-exciting process** for timing and **preferential "
                "attachment** for structure - calibrated from Twitter15/16.  \n\n"
                "Demonstrates the full pipeline as it would run with "
                "real-time platform API access.")

        detect_btn = st.button("Detect", type="primary", use_container_width=True)
        st.divider()
        st.markdown(f"**Device:** `{DEVICE}`")
        st.markdown(f"**KNN index:** {'loaded' if knn_ok else 'not built'}")
        if not knn_ok:
            st.caption("`python backend/models/adaptive_scorer.py`")
        st.divider()
        st.markdown(
            "**Results (Twitter15)**  \n"
            "- Full AUC: 0.8646  \n"
            "- 60 min AUC: 0.8588  \n"
            "- Twitter16 transfer: 0.7217")

    # ── tabs ─────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "Detection",
        "Propagation Analysis",
        "Community Analysis",
        "Early Detection AUC",
        "Explainability",
    ])

    with t4:
        fig = chart_auc(cal.get("extended_horizons"))
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="auc_curve")
            st.caption("AUC stays above 0.85 at 1 minute - flags misinfo "
                       "before most users have seen it.")
        else:
            st.info("Run `python backend/evaluation/calibrate.py` to generate this curve.")

    if not detect_btn:
        for tab in (t1, t2, t3, t5):
            with tab:
                st.info("Configure input in the sidebar and click **Detect**.")
        return

    # ── resolve data ─────────────────────────────────────────────
    sim_graph   = None
    data_source = None
    claim_label = ""

    if mode == "Dataset Claim ID":
        if not claim_id:
            st.error("Enter a Claim ID."); return
        gp = f"dataset/processed/{dataset}/graphs/claim_{claim_id.strip()}.pt"
        if not os.path.exists(gp):
            st.error(f"Claim **{claim_id}** not found in **{dataset}**."); return
        data_source = gp
        claim_label = f"{dataset} / claim {claim_id.strip()}"

    else:
        with st.spinner("Simulating propagation tree..."):
            sim_graph = simulator.simulate(
                mode=sim_mode, n_nodes=sim_nodes, seed=int(sim_seed))
        data_source = sim_graph
        claim_label = (
            f"Simulated {'misinfo' if sim_mode=='misinfo' else 'credible'} - "
            f"{sim_graph['num_nodes']} nodes, seed={sim_seed}")

    # ── run main inference ─────────────────────────────────────────────
    with st.spinner("Running CGDEX-Net..."):
        res = run_inference(tpgb, ste, vle, cdp, ccm, dsh,
                            scorer, data_source, tau_minutes, cal)
    with st.spinner("Analysing community structure..."):
        comm = community.analyze(res["data"])

    nf = res["node_feats"]
    ei = res["edge_index"]
    topo_stats     = comm.get("topology", {})
    temporal_stats = comm.get("temporal_analysis", {})
    viz_stats      = comm.get("visualization", {})
    velocity_hist  = viz_stats.get("velocity_hist", {})
    depth_hist     = viz_stats.get("depth_hist", {})
    timeline_stats = viz_stats.get("timeline_stats", [])
    br                  = float(comm.get("hawkes_branching_ratio", 0.5))
    propagation_risk    = comm.get("propagation_risk", "LOW")
    propagation_reasons = comm.get("propagation_risk_reasons", [])

    cache_key = f"{claim_label}_early"
    if cache_key not in st.session_state:
        early_points = []
        with st.spinner("Computing early detection trajectory..."):
            for horizon in SIM_HORIZONS:
                src = copy.deepcopy(data_source) if isinstance(data_source, dict) else data_source
                r_h = run_inference(tpgb, ste, vle, cdp, ccm, dsh,
                                    scorer, src, horizon, cal)
                d_h = r_h.get("decision", {})
                early_points.append({
                    "tau"    : horizon,
                    "score"  : r_h["score"],
                    "thr"    : d_h.get("threshold", 0.5),
                    "lo"     : d_h.get("unc_lo", 0.46),
                    "hi"     : d_h.get("unc_hi", 0.54),
                    "verdict": r_h["verdict"],
                })
        st.session_state[cache_key] = early_points
    else:
        early_points = st.session_state[cache_key]

    # ── Tab 4: Early Detection AUC ─────────────────────────────────────
    with t4:
        xs = [p["tau"]    for p in early_points]
        ys = [p["score"]  for p in early_points]
        th = [p["thr"]    for p in early_points]
        lo = [p["lo"]     for p in early_points]
        hi = [p["hi"]     for p in early_points]
        vc = {"MISINFORMATION": "#e74c3c", "CREDIBLE": "#27ae60", "UNCERTAIN": "#e67e22"}

        f_dyn = go.Figure()
        f_dyn.add_trace(go.Scatter(
            x=xs, y=hi, mode="lines",
            line=dict(color="rgba(230,126,34,0.35)", width=1),
            name="Uncertain upper",
            hovertemplate="tau=%{x}min<br>upper=%{y:.4f}<extra></extra>",
        ))
        f_dyn.add_trace(go.Scatter(
            x=xs, y=lo, mode="lines", fill="tonexty",
            fillcolor="rgba(230,126,34,0.10)",
            line=dict(color="rgba(230,126,34,0.35)", width=1),
            name="Uncertain lower",
            hovertemplate="tau=%{x}min<br>lower=%{y:.4f}<extra></extra>",
        ))
        f_dyn.add_trace(go.Scatter(
            x=xs, y=th, mode="lines",
            line=dict(color="#f1c40f", width=2, dash="dash"),
            name="Dynamic threshold",
            hovertemplate="tau=%{x}min<br>threshold=%{y:.4f}<extra></extra>",
        ))
        f_dyn.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            marker=dict(size=10, color=[vc.get(p["verdict"], "#888") for p in early_points]),
            line=dict(color="#17becf", width=3),
            name="Backbone score",
            hovertemplate="tau=%{x}min<br>score=%{y:.4f}<extra></extra>",
        ))
        f_dyn.update_layout(
            title="Selected Claim: Early Detection Trajectory",
            xaxis_title="Observation window (minutes)",
            yaxis_title="Backbone score",
            yaxis=dict(range=[0, 1]),
            height=320,
            margin=dict(l=20, r=20, t=45, b=20),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font_color="#0f172a",
        )
        st.plotly_chart(f_dyn, use_container_width=True, key="auc_claim_dynamic")
        st.caption(
            "This is claim-specific and recomputed at each horizon. "
            "It is separate from the global dataset AUC curve above."
        )

    # ── Tab 1: Detection ─────────────────────────────────────────────
    with t1:
        st.markdown(
            f"<span style='color:{THEME['muted']};font-size:12px'>{claim_label}</span>",
            unsafe_allow_html=True)
        st.markdown("")

        L, R = st.columns([1, 1.6])
        with L:
            render_verdict(res, knn_ok)
            st.markdown("")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Nodes",  res["num_nodes"])
            m2.metric("Edges",  res["num_edges"])
            m3.metric("sigma2_z", f"{res['features']['sigma2_z']:.1f}")
            m4.metric("Propagation risk", propagation_risk)
            if propagation_reasons:
                st.caption("Risk drivers: " + ", ".join(propagation_reasons))
        with R:
            st.plotly_chart(
                chart_velocity(velocity_hist, tau_minutes, res.get("max_timestamp")),
                use_container_width=True, key="velocity_chart")

        st.plotly_chart(chart_depth(depth_hist),
                        use_container_width=True, key="depth_chart")

        # Hawkes gauge
        hg1, hg2 = st.columns([1, 2])
        with hg1:
            st.plotly_chart(chart_hawkes(br),
                            use_container_width=True, key="hawkes_detection")
        with hg2:
            st.markdown("#### Hawkes Branching Ratio")
            interp = (
                "Each event triggers **> 1** subsequent event - "
                "consistent with **coordinated amplification**." if br > .7 else
                "Each event triggers **< 1** subsequent event - "
                "consistent with **organic natural decay**." if br < .4 else
                "Mixed - some coordination, some organic spread.")
            st.markdown(
                f"Estimated value: **{br:.3f}**  \n\n{interp}  \n\n"
                "| Range | Pattern |  \n|---|---|  \n"
                "| 0.0 - 0.4 | Subcritical / organic |  \n"
                "| 0.4 - 0.7 | Moderate |  \n"
                "| 0.7 - 1.0 | Supercritical / coordinated |  \n\n"
                "*No prior misinformation detection paper uses this as a feature.*")
            if res["verdict"] == "CREDIBLE" and propagation_risk in ("MODERATE", "HIGH"):
                st.warning(
                    "Backbone decision is **CREDIBLE**, but propagation is "
                    f"**{propagation_risk.lower()}-risk**. This indicates potentially "
                    "coordinated spread behavior without forcing a label override.")
            elif res["verdict"] == "MISINFORMATION" and propagation_risk == "LOW":
                st.info(
                    "Backbone decision is **MISINFORMATION** while propagation appears "
                    "mostly organic. Content and spread dynamics can disagree.")

        # Platform actions + export
        st.divider()
        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("Flag for Review", use_container_width=True):
                queue_path = _save_moderation_event(claim_label, res, comm)
                st.session_state["moderation_last"] = queue_path
                st.success(f"Flagged and stored in {queue_path}")
        with a2:
            if st.button("Apply Warning Label", use_container_width=True):
                v = res["verdict"]
                if v == "MISINFORMATION":
                    st.session_state["warning_label_text"] = "This claim may be misleading."
                    st.success("Warning label applied for misinformation verdict.")
                else:
                    st.session_state["warning_label_text"] = ""
                    st.success("No warning label applied.")
        with a3:
            ada = res.get("adaptive") or {}
            export_payload = {
                "claim"           : claim_label,
                "verdict"         : res["verdict"],
                "method"          : res.get("decision_source", "backbone_dynamic_threshold_binary"),
                "score"           : res["score"],
                "base_score"      : res.get("raw_score"),
                "decision"        : res.get("decision"),
                "sigma2_z"        : ada.get("sigma2_raw"),
                "uncertain_margin": ada.get("uncertain_margin"),
                "features"        : {k: round(v, 4) for k, v in res["features"].items()},
                "community"       : {
                    "count"           : comm["community_count"],
                    "modularity"      : comm["modularity"],
                    "echo_chamber"    : comm["echo_chamber_score"],
                    "hawkes_br"       : comm["hawkes_branching_ratio"],
                    "root_centrality" : comm["root_centrality"],
                    "propagation_risk": propagation_risk,
                    "risk_drivers"    : propagation_reasons,
                },
                "spatial_flags"   : comm["spatial_risk_flags"],
                "temporal_flags"  : comm.get("temporal_risk_flags", []),
            }
            st.download_button(
                "Export JSON",
                data=json.dumps(export_payload, indent=2),
                file_name=f"cgdex_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        if st.session_state["warning_label_text"]:
            st.warning(st.session_state["warning_label_text"])
        if st.session_state["moderation_last"]:
            st.caption(f"Moderation queue: `{st.session_state['moderation_last']}`")

    # ── Tab 2: Propagation Analysis ───────────────────────────────────
    with t2:
        st.markdown(
            "**Connected Papers-style interactive propagation graph.**  "
            "Node size = downstream reach. Color = time (early -> mid -> late).  "
            "**Click any node** to trace path back to SOURCE.  "
            "**Community** toggle colors by detected community.  "
            "**Time slider** animates the spread.")

        html_str = propagation_html(
            edge_index=ei, node_feats=nf,
            label=claim_label, verdict=res["verdict"],
            communities=comm.get("_raw_communities"),
            tau=tau_minutes,
            node_depths=topo_stats.get("node_depths"),
            node_reach=topo_stats.get("node_reach"),
            timeline_stats=timeline_stats)
        components.html(html_str, height=680, scrolling=False)

        st.divider()
        st.markdown("### Propagation Analysis Timeline")
        ph  = st.empty()
        bar = st.progress(0)
        tl  = []
        vc  = {"MISINFORMATION": "#e74c3c",
               "CREDIBLE": "#27ae60", "UNCERTAIN": "#e67e22"}

        # reuse early_points — no new inference calls needed
        for i, p in enumerate(early_points):
            tl.append({
                "tau"    : p["tau"],
                "score"  : p["score"],
                "verdict": p["verdict"],
                "thr"    : p["thr"],
                "lo"     : p["lo"],
                "hi"     : p["hi"],
            })
            bar.progress((i + 1) / len(early_points))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[str(t["tau"]) for t in tl],
                y=[t["hi"] for t in tl],
                mode="lines",
                line=dict(color="rgba(230,126,34,0.35)", width=1),
                name="Confidence upper",
                hovertemplate="tau=%{x}min<br>upper=%{y:.4f}<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=[str(t["tau"]) for t in tl],
                y=[t["lo"] for t in tl],
                mode="lines", fill="tonexty",
                fillcolor="rgba(230,126,34,0.10)",
                line=dict(color="rgba(230,126,34,0.35)", width=1),
                name="Confidence lower",
                hovertemplate="tau=%{x}min<br>lower=%{y:.4f}<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=[str(t["tau"]) for t in tl],
                y=[t["thr"] for t in tl],
                mode="lines",
                line=dict(width=2, color="#f1c40f", dash="dash"),
                name="Dynamic threshold",
                hovertemplate="tau=%{x}min<br>threshold=%{y:.4f}<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=[str(t["tau"]) for t in tl],
                y=[t["score"] for t in tl],
                mode="lines+markers",
                line=dict(width=3, color="#17becf"),
                marker=dict(size=14, color=[vc.get(t["verdict"], "#888") for t in tl]),
                hovertemplate="tau=%{x}min<br>score=%{y:.4f}<extra></extra>"))
            fig.update_layout(
                title="Detection Score Over Time",
                xaxis_title="Observation window (minutes)",
                yaxis_title="Score", yaxis=dict(range=[0, 1]),
                height=280, margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                font_color="#0f172a")
            with ph.container():
                st.plotly_chart(fig, use_container_width=True, key=f"timeline_{i}")
        bar.empty()
        st.success(
            f"Propagation simulation complete. Final verdict: **{tl[-1]['verdict']}** "
            f"(score={tl[-1]['score']:.4f})"
        )

        st.divider()
        st.caption("These metrics are computed from backend graph topology for the current input window.")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Spatial Analysis**")
            maxd = int(topo_stats.get("max_depth", 0))
            maxw = int(topo_stats.get("max_width", 1))
            dwr  = float(topo_stats.get("depth_width_ratio", 0.0))
            st.metric("Max depth", maxd)
            st.metric("Max width (breadth)", maxw)
            st.metric("Depth / width ratio", f"{dwr:.2f}",
                      delta=("deep/narrow (organic)" if dwr > .5
                             else "wide/shallow (suspicious)"))
        with s2:
            st.markdown("**Temporal Analysis**")
            ef            = float(temporal_stats.get("early_burst", 0.0))
            peak_snapshot = int(temporal_stats.get("peak_snapshot", 1))
            st.metric("Early burst (first 20%)", f"{ef:.0%}",
                      delta="suspicious" if ef > .6 else None)
            st.metric("Peak activity snapshot", f"{peak_snapshot}/5")
            st.metric("Total users in tree",
                      int(comm.get("n_nodes", res["num_nodes"])))

    # ── Tab 3: Community Analysis ─────────────────────────────────────
    with t3:
        st.markdown(
            "Community structure reveals **where** the claim spread - "
            "echo chambers, coordinated networks, cross-community injection.")
        st.markdown("#### Decision vs Propagation")
        cc1, cc2 = st.columns(2)
        cc1.metric("Backbone verdict", res["verdict"])
        cc2.metric("Propagation risk", propagation_risk)
        if res["verdict"] == "CREDIBLE" and propagation_risk in ("MODERATE", "HIGH"):
            st.warning(
                "Label and spread-pattern signals differ: content-level features "
                "look credible, but diffusion topology is suspicious.")
        elif res["verdict"] == "MISINFORMATION" and propagation_risk == "LOW":
            st.info(
                "Label and spread-pattern signals differ: misinformation content "
                "signals are strong, but spread appears less coordinated.")
        if propagation_reasons:
            st.caption("Propagation risk drivers: " + ", ".join(propagation_reasons))
        render_community(comm)

        if comm.get("community_sizes"):
            sizes = comm["community_sizes"][:8]
            fig   = go.Figure(go.Bar(
                x=[f"C{i+1}" for i in range(len(sizes))], y=sizes,
                marker_color="#8b5cf6",
                hovertemplate="Community %{x}: %{y} users<extra></extra>"))
            fig.update_layout(
                title="Community Size Distribution",
                xaxis_title="Community", yaxis_title="Users",
                height=220, margin=dict(l=20, r=20, t=40, b=40),
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                font_color="#0f172a")
            st.plotly_chart(fig, use_container_width=True, key="community_size_dist")

    # ── Tab 5: Explainability ─────────────────────────────────────────
    with t5:
        st.markdown("### Explainability (XAI)")
        st.caption("All values below are backend-calculated and tied to the current input and observation window.")

        d = res.get("decision", {})
        x1, x2, x3 = st.columns(3)
        x1.metric("Backbone score", f"{res['score']:.4f}")
        x2.metric("Dynamic threshold", f"{d.get('threshold', 0.5):.4f}")
        x3.metric("Low-confidence zone",
                  f"[{d.get('unc_lo', 0.46):.4f}, {d.get('unc_hi', 0.54):.4f}]")

        with st.expander("Adaptive KNN Neighborhood (XAI context)", expanded=knn_ok):
            render_knn(res.get("adaptive"))

        with st.expander("CGDEX Latent/Distance Features", expanded=True):
            f  = res["features"]
            fs = cal.get("feature_stats", {})
            rows = []
            for fk, fl in [("sigma2_z", "Posterior Variance"),
                            ("d_m",      "Manifold Distance"),
                            ("d_cv",     "Completion Variance"),
                            ("d_tr",     "Temporal Velocity")]:
                stt    = fs.get(fk, {})
                sep    = stt.get("separator")
                c_mean = stt.get("credible_mean")
                m_mean = stt.get("misinfo_mean")
                signal = "UNKNOWN"
                if sep is not None and c_mean is not None and m_mean is not None:
                    if float(m_mean) > float(c_mean):
                        is_mis = f[fk] >= float(sep)
                    else:
                        is_mis = f[fk] <= float(sep)
                    signal = "MISINFO-leaning" if is_mis else "CREDIBLE-leaning"
                rows.append({
                    "Feature"             : fl,
                    "Value"               : f"{f[fk]:.4f}",
                    "Reference separator" : f"{sep:.4f}" if sep is not None else "run calibrate.py",
                    "Signal"              : signal,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### Propagation/Community Reasons")
        for txt in comm.get("spatial_risk_flags", []):
            st.warning(txt)
        for txt in comm.get("temporal_risk_flags", []):
            st.warning(txt)
        if not comm.get("spatial_risk_flags") and not comm.get("temporal_risk_flags"):
            st.success("No strong structural or temporal anomaly flags for this input.")


if __name__ == "__main__":
    main()

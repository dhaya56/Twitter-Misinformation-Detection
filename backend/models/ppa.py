# backend/models/ppa.py
"""
ppa.py
------
Propagation Pattern Analyzer (PPA).

This module addresses the "Spatio-Temporal Detection of Propagation PATTERNS"
part of the project title. Rather than just classifying a claim as 
misinformation or credible, it characterizes HOW the claim spreads.

Six propagation pattern features are computed:

TEMPORAL patterns:
    1. Burst Score    — how quickly the first 20% of nodes are acquired
                        high burst = suspicious (viral amplification)
    2. Decay Rate     — does engagement slow down over time (credible) 
                        or stay sustained (coordinated spread)?
    3. Peak Timing    — when does maximum spreading velocity occur?

SPATIAL / STRUCTURAL patterns:
    4. Depth-to-Width Ratio — deep narrow trees vs shallow wide trees
                               misinformation often has wider, shallower spread
    5. Cascade Density — ratio of edges to nodes (branching factor)
    6. Root Dominance  — fraction of nodes directly connected to root
                         high root dominance = hub-and-spoke (suspicious)

These features are used for:
    - XAI explanation: "This claim spread in a burst within the first 30 minutes"
    - Additional detection signal in the Streamlit visualization
    - Pattern fingerprinting for patent claim on propagation characterization
"""

import torch
import numpy as np


class PropagationPatternAnalyzer:
    """
    Analyzes propagation patterns from a graph's structural and temporal features.
    
    No learnable parameters. Pure feature computation from graph data.
    
    Args:
        None
    
    Usage:
        ppa = PropagationPatternAnalyzer()
        patterns = ppa.analyze(data)
        # patterns: dict with scores and interpretation strings
    """

    def analyze(self, data):
        """
        Computes all six propagation pattern features.
        
        Args:
            data : PropagationData object (single graph, not batched)
        
        Returns:
            dict with:
                scores       : dict of float scores for each pattern
                labels       : dict of human-readable labels
                verdict      : str overall pattern type
                risk_flags   : list of str triggered risk flags
        """
        x          = data.x          # [N, 6]
        edge_index = data.edge_index  # [2, E]
        N          = x.shape[0]
        E          = edge_index.shape[1] if edge_index.numel() > 0 else 0

        # extract node features
        norm_times = x[:, 0].numpy()   # normalized timestamps [0,1]
        # x[:, 1] is normalized depth — denormalize using max_tree_depth if available
        if hasattr(data, "max_tree_depth") and data.max_tree_depth > 0:
            bfs_depths = (x[:, 1].numpy() * data.max_tree_depth)
        else:
            # fallback: scale by N as proxy for max depth
            bfs_depths = (x[:, 1].numpy() * N)
        bfs_depths = np.round(bfs_depths).astype(int)   # BFS depth from root
        out_degrees = x[:, 2].numpy()  # out-degree (number of children)

        scores     = {}
        risk_flags = []

        # ── 1. Burst Score ─────────────────────────────────────────────────
        # fraction of nodes acquired in the first 20% of the time window
        early_mask   = norm_times <= 0.2
        burst_score  = float(early_mask.sum()) / max(N, 1)
        scores["burst_score"] = round(burst_score, 4)
        if burst_score > 0.6:
            risk_flags.append(
                f"HIGH BURST: {burst_score:.0%} of propagation occurred in first 20% of time window"
            )

        # ── 2. Decay Rate ──────────────────────────────────────────────────
        # compare node density in first half vs second half of time window
        first_half  = float((norm_times <= 0.5).sum()) / max(N, 1)
        second_half = 1.0 - first_half
        # decay_rate > 0 means more activity in first half (natural decay)
        # decay_rate < 0 means more activity in second half (sustained/growing)
        decay_rate  = first_half - second_half
        scores["decay_rate"] = round(decay_rate, 4)
        if decay_rate < -0.1:
            risk_flags.append(
                "SUSTAINED SPREAD: propagation intensity does not decay — "
                "possible coordinated amplification"
            )

        # ── 3. Peak Timing ─────────────────────────────────────────────────
        # find the time window with the highest node density
        bins = np.linspace(0, 1, 6)  # 5 bins matching snapshots
        hist, _ = np.histogram(norm_times, bins=bins)
        peak_bin = int(np.argmax(hist))
        peak_timing = peak_bin / 5.0  # normalize to [0,1]
        scores["peak_timing"] = round(peak_timing, 4)
        if peak_timing < 0.2:
            risk_flags.append(
                f"EARLY PEAK: maximum spreading velocity in first snapshot — "
                "typical of artificially amplified content"
            )

        # ── 4. Depth-to-Width Ratio ────────────────────────────────────────
        max_depth  = float(bfs_depths.max()) if N > 1 else 0.0
        max_width  = float(np.bincount(bfs_depths.astype(int)).max()) if N > 1 else 1.0
        dw_ratio   = max_depth / max(max_width, 1.0)
        scores["depth_width_ratio"] = round(dw_ratio, 4)
        if dw_ratio < 0.3:
            risk_flags.append(
                f"WIDE SHALLOW TREE (depth/width={dw_ratio:.2f}): "
                "many users at the same level — typical of bot-amplified content"
            )

        # ── 5. Cascade Density ─────────────────────────────────────────────
        # edges per node (branching factor proxy)
        cascade_density = E / max(N, 1)
        scores["cascade_density"] = round(cascade_density, 4)

        # ── 6. Root Dominance ──────────────────────────────────────────────
        # fraction of nodes directly connected to root (depth==1)
        depth1_count  = float((bfs_depths == 1).sum())
        root_dominance = depth1_count / max(N - 1, 1)
        scores["root_dominance"] = round(root_dominance, 4)
        if root_dominance > 0.7:
            risk_flags.append(
                f"HIGH ROOT DOMINANCE ({root_dominance:.0%} of nodes directly "
                "connected to source) — hub-and-spoke pattern typical of "
                "coordinated sharing"
            )

        # ── overall pattern verdict ────────────────────────────────────────
        risk_count = len(risk_flags)
        if risk_count == 0:
            verdict = "ORGANIC"
            verdict_desc = "Propagation follows natural organic sharing patterns"
        elif risk_count == 1:
            verdict = "MILDLY SUSPICIOUS"
            verdict_desc = "One anomalous propagation pattern detected"
        elif risk_count == 2:
            verdict = "SUSPICIOUS"
            verdict_desc = "Multiple anomalous propagation patterns detected"
        else:
            verdict = "HIGHLY SUSPICIOUS"
            verdict_desc = "Propagation pattern strongly resembles coordinated inorganic spread"

        labels = {
            "burst_score"       : f"Early burst: {burst_score:.0%} of nodes in first 20% of time",
            "decay_rate"        : f"Decay: {'natural' if decay_rate > 0 else 'sustained/growing'} ({decay_rate:+.3f})",
            "peak_timing"       : f"Peak activity: snapshot {peak_bin+1}/5",
            "depth_width_ratio" : f"Tree shape: {'deep/narrow' if dw_ratio > 0.5 else 'wide/shallow'} ({dw_ratio:.2f})",
            "cascade_density"   : f"Branching factor: {cascade_density:.2f} edges/node",
            "root_dominance"    : f"Root connections: {root_dominance:.0%} of nodes",
        }

        return {
            "scores"     : scores,
            "labels"     : labels,
            "verdict"    : verdict,
            "verdict_desc": verdict_desc,
            "risk_flags" : risk_flags,
            "num_nodes"  : N,
            "num_edges"  : E,
        }

    def velocity_curve(self, data, n_points=20):
        """
        Computes propagation velocity curve — nodes per unit time.
        
        Returns arrays suitable for plotting:
            time_points : array [n_points]  normalized time [0,1]
            velocity    : array [n_points]  nodes acquired per time unit
        
        This is the core "spatio-temporal" visualization:
        credible news shows a bell curve (rise then fall),
        misinformation often shows a sharp spike then plateau.
        """
        norm_times = data.x[:, 0].numpy()
        time_points = np.linspace(0, 1, n_points)
        velocity    = np.zeros(n_points)

        dt = 1.0 / n_points
        for i, t in enumerate(time_points):
            mask = (norm_times >= t) & (norm_times < t + dt)
            velocity[i] = mask.sum()

        return time_points, velocity
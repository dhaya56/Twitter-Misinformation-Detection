"""
community_analysis.py
---------------------
Backend-only spatial + temporal propagation analytics used by Streamlit.
All numeric values shown in the UI should come from this module output.
"""

import torch
import numpy as np
from collections import defaultdict, deque


class CommunityAnalyzer:
    """
    Lightweight graph analytics for propagation explainability.
    Produces:
      - community structure metrics
      - tree topology metrics (depth/reach per node)
      - temporal spread metrics
      - risk flags and propagation risk summary
      - precomputed histogram payloads for frontend charts
    """

    def analyze(self, data, times=None):
        x = data.x if hasattr(data, "x") else data["x"]
        edge_index = data.edge_index if hasattr(data, "edge_index") else data["edge_index"]

        n_nodes = int(x.shape[0]) if x is not None else 0
        n_edges = int(edge_index.shape[1]) if edge_index is not None and edge_index.numel() > 0 else 0
        if n_nodes == 0:
            return self._empty_result()

        if times is not None:
            norm_times = np.asarray(times, dtype=float)
        else:
            norm_times = x[:, 0].detach().cpu().numpy().astype(float)

        communities, modularity = self._detect_communities(edge_index, n_nodes)
        community_sizes = self._community_sizes(communities)
        n_communities = len(set(communities.values())) if communities else 0

        cross_edges = 0
        if n_edges > 0:
            for i in range(n_edges):
                src = int(edge_index[0, i])
                dst = int(edge_index[1, i])
                if communities.get(src, 0) != communities.get(dst, 0):
                    cross_edges += 1
        cross_frac = cross_edges / max(n_edges, 1)
        echo_chamber_score = float(max(modularity, 0.0) * (1.0 - cross_frac))
        # add a floor when cross_frac is very low regardless of modularity
        if cross_frac < 0.05 and n_communities > 1:
            echo_chamber_score = max(echo_chamber_score, 0.35)

        topology = self._compute_tree_topology(edge_index, n_nodes)
        temporal = self._compute_temporal_analysis(norm_times, communities, num_snapshots=5)
        hawkes_br = self._estimate_hawkes_branching(norm_times)
        root_centrality = self._root_betweenness(edge_index, n_nodes)

        depth_vals = np.asarray(topology["node_depths"], dtype=float)
        depth_std = float(np.std(depth_vals)) if n_nodes > 2 else 0.0
        depth_var = float(np.var(depth_vals)) if n_nodes > 2 else 0.0

        spatial_risk_flags = []
        temporal_risk_flags = []

        if echo_chamber_score > 0.4:
            spatial_risk_flags.append(
                f"ECHO CHAMBER: high modularity ({modularity:.2f}) with low cross-community spread ({cross_frac:.0%})."
            )
        if n_communities <= 1 and n_nodes > 20:
            spatial_risk_flags.append("SINGLE COMMUNITY: most activity remains inside one tightly connected group.")
        if depth_std < 0.5 and n_nodes > 10:
            spatial_risk_flags.append(f"LOW DEPTH VARIANCE: sigma={depth_std:.2f}, spread is unusually shallow/uniform.")
        if topology["depth_width_ratio"] < 0.12 and n_nodes > 30:
            spatial_risk_flags.append("WIDE-SHALLOW TREE: broad first-hop amplification dominates.")
        if root_centrality > 0.6:
            spatial_risk_flags.append(f"HIGH ROOT CENTRALITY ({root_centrality:.2f}): source dominates shortest paths.")

        if hawkes_br > 0.75:
            temporal_risk_flags.append(
                f"HIGH HAWKES BRANCHING RATIO ({hawkes_br:.2f}): supercritical self-excitation."
            )
        if temporal["early_burst"] > 0.6:
            temporal_risk_flags.append(
                f"EARLY BURST ({temporal['early_burst']:.0%} in first 20% time window): coordinated acceleration pattern."
            )
        if temporal["peak_snapshot"] == 1 and temporal["burstiness_index"] > 0.45:
            temporal_risk_flags.append("PEAK ACTIVITY AT SNAPSHOT 1 with high concentration.")
        if temporal["temporal_entropy"] < 0.55 and n_nodes > 20:
            temporal_risk_flags.append("LOW TEMPORAL ENTROPY: activity concentrated in few windows.")

        propagation_risk, risk_reasons = self._summarize_propagation_risk(
            hawkes_br=hawkes_br,
            spatial_flags=spatial_risk_flags,
            temporal_flags=temporal_risk_flags,
            topology=topology,
            temporal=temporal,
        )

        n_flags = len(spatial_risk_flags) + len(temporal_risk_flags)
        if n_flags == 0:
            pattern_verdict = "ORGANIC COMMUNITY STRUCTURE"
        elif n_flags <= 2:
            pattern_verdict = "MILDLY SUSPICIOUS STRUCTURE"
        elif n_flags <= 4:
            pattern_verdict = "SUSPICIOUS COMMUNITY STRUCTURE"
        else:
            pattern_verdict = "HIGHLY SUSPICIOUS - COORDINATED SPREAD"

        depth_hist_counts = np.bincount(np.asarray(topology["node_depths"], dtype=int))
        depth_hist_depths = list(range(len(depth_hist_counts)))

        vel_counts, vel_edges = np.histogram(norm_times, bins=20, range=(0.0, 1.0))
        vel_centers = ((vel_edges[:-1] + vel_edges[1:]) / 2.0).tolist()
        timeline_stats = self._compute_timeline_stats(
            norm_times=norm_times,
            node_depths=topology["node_depths"],
            edge_index=edge_index,
            communities=communities,
            num_steps=101,
        )

        return {
            "community_count": n_communities,
            "modularity": round(float(modularity), 4),
            "cross_community_frac": round(float(cross_frac), 4),
            "depth_variance": round(float(depth_var), 4),
            "depth_std": round(float(depth_std), 4),
            "hawkes_branching_ratio": hawkes_br,
            "root_centrality": round(float(root_centrality), 4),
            "community_sizes": community_sizes[:10],
            "echo_chamber_score": round(float(echo_chamber_score), 4),
            "pattern_verdict": pattern_verdict,
            "spatial_risk_flags": spatial_risk_flags,
            "temporal_risk_flags": temporal_risk_flags,
            "propagation_risk": propagation_risk,
            "propagation_risk_reasons": risk_reasons,
            "topology": topology,
            "temporal_analysis": temporal,
            "_raw_communities": {int(k): int(v) for k, v in communities.items()},
            "visualization": {
                "depth_hist": {
                    "depths": [int(d) for d in depth_hist_depths],
                    "counts": [int(c) for c in depth_hist_counts.tolist()],
                },
                "velocity_hist": {
                    "x": [float(v) for v in vel_centers],
                    "y": [int(v) for v in vel_counts.tolist()],
                },
                "temporal_snapshot_counts": [int(v) for v in temporal["snapshot_counts"]],
                "timeline_stats": timeline_stats,
            },
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        }

    def _empty_result(self):
        return {
            "community_count": 0,
            "modularity": 0.0,
            "cross_community_frac": 0.0,
            "depth_variance": 0.0,
            "depth_std": 0.0,
            "hawkes_branching_ratio": 0.5,
            "root_centrality": 0.0,
            "community_sizes": [],
            "echo_chamber_score": 0.0,
            "pattern_verdict": "NO DATA",
            "spatial_risk_flags": [],
            "temporal_risk_flags": [],
            "propagation_risk": "LOW",
            "propagation_risk_reasons": [],
            "topology": {
                "node_depths": [],
                "node_reach": [],
                "max_depth": 0,
                "max_width": 0,
                "depth_width_ratio": 0.0,
                "leaf_fraction": 0.0,
                "root_dominance": 0.0,
            },
            "temporal_analysis": {
                "early_burst": 0.0,
                "peak_snapshot": 1,
                "burstiness_index": 0.0,
                "temporal_entropy": 0.0,
                "mean_inter_event_gap": 0.0,
                "community_temporal_dispersion": 0.0,
                "community_activation_spread": 0.0,
                "snapshot_counts": [0, 0, 0, 0, 0],
            },
            "_raw_communities": {},
            "visualization": {
                "depth_hist": {"depths": [0], "counts": [0]},
                "velocity_hist": {"x": [0.0], "y": [0]},
                "temporal_snapshot_counts": [0, 0, 0, 0, 0],
                "timeline_stats": [],
            },
            "n_nodes": 0,
            "n_edges": 0,
        }

    def _compute_tree_topology(self, edge_index, n_nodes):
        if n_nodes <= 0:
            return {
                "node_depths": [],
                "node_reach": [],
                "max_depth": 0,
                "max_width": 0,
                "depth_width_ratio": 0.0,
                "leaf_fraction": 0.0,
                "root_dominance": 0.0,
            }

        children = [[] for _ in range(n_nodes)]
        out_deg = np.zeros(n_nodes, dtype=int)
        if edge_index is not None and edge_index.numel() > 0:
            src_all = edge_index[0].detach().cpu().numpy().astype(int)
            dst_all = edge_index[1].detach().cpu().numpy().astype(int)
            for s, d in zip(src_all, dst_all):
                if 0 <= s < n_nodes and 0 <= d < n_nodes and s != d:
                    children[s].append(d)
                    out_deg[s] += 1

        depths = np.full(n_nodes, -1, dtype=int)
        depths[0] = 0
        q = deque([0])
        while q:
            u = q.popleft()
            for v in children[u]:
                if depths[v] == -1 or depths[v] > depths[u] + 1:
                    depths[v] = depths[u] + 1
                    q.append(v)
        depths[depths < 0] = 0

        max_depth = int(depths.max()) if n_nodes > 0 else 0
        bins = np.bincount(depths, minlength=max_depth + 1) if n_nodes > 0 else np.array([0], dtype=int)
        max_width = int(bins.max()) if bins.size else 0
        dwr = float(max_depth / max(max_width, 1))

        memo = {}

        def descend_count(node, path):
            if node in memo:
                return memo[node]
            if node in path:
                return 0
            total = 0
            new_path = path | {node}
            for ch in children[node]:
                total += 1 + descend_count(ch, new_path)
            memo[node] = total
            return total

        reach = np.array([descend_count(i, set()) for i in range(n_nodes)], dtype=int)
        leaf_fraction = float((out_deg == 0).sum() / max(n_nodes, 1))
        root_dominance = float((depths == 1).sum() / max(n_nodes - 1, 1))

        return {
            "node_depths": [int(v) for v in depths.tolist()],
            "node_reach": [int(v) for v in reach.tolist()],
            "max_depth": max_depth,
            "max_width": max_width,
            "depth_width_ratio": round(dwr, 4),
            "leaf_fraction": round(leaf_fraction, 4),
            "root_dominance": round(root_dominance, 4),
        }

    def _compute_temporal_analysis(self, norm_times, communities, num_snapshots=5):
        t = np.asarray(norm_times, dtype=float)
        n = int(t.shape[0])
        if n == 0:
            return {
                "early_burst": 0.0,
                "peak_snapshot": 1,
                "burstiness_index": 0.0,
                "temporal_entropy": 0.0,
                "mean_inter_event_gap": 0.0,
                "community_temporal_dispersion": 0.0,
                "community_activation_spread": 0.0,
                "snapshot_counts": [0] * num_snapshots,
            }

        counts, _ = np.histogram(t, bins=num_snapshots, range=(0.0, 1.0))
        shares = counts / max(n, 1)
        peak_snapshot = int(np.argmax(counts)) + 1
        early_burst = float(shares[0]) if shares.size > 0 else 0.0
        burstiness = float(shares.max()) if shares.size > 0 else 0.0

        p = shares[shares > 0]
        if p.size == 0:
            temporal_entropy = 0.0
        else:
            temporal_entropy = float(-(p * np.log(p)).sum() / np.log(max(num_snapshots, 2)))

        sorted_t = np.sort(t)
        if sorted_t.shape[0] >= 2:
            inter = np.diff(sorted_t)
            mean_gap = float(inter.mean())
        else:
            mean_gap = 0.0

        by_comm = defaultdict(list)
        for node_id, comm_id in communities.items():
            if 0 <= node_id < n:
                by_comm[comm_id].append(t[node_id])

        comm_centroids = []
        comm_first = []
        for vals in by_comm.values():
            arr = np.asarray(vals, dtype=float)
            if arr.size > 0:
                comm_centroids.append(float(arr.mean()))
                comm_first.append(float(arr.min()))

        comm_disp = float(np.std(comm_centroids)) if len(comm_centroids) >= 2 else 0.0
        comm_spread = float(np.max(comm_first) - np.min(comm_first)) if len(comm_first) >= 2 else 0.0

        return {
            "early_burst": round(early_burst, 4),
            "peak_snapshot": int(peak_snapshot),
            "burstiness_index": round(burstiness, 4),
            "temporal_entropy": round(temporal_entropy, 4),
            "mean_inter_event_gap": round(mean_gap, 4),
            "community_temporal_dispersion": round(comm_disp, 4),
            "community_activation_spread": round(comm_spread, 4),
            "snapshot_counts": [int(v) for v in counts.tolist()],
        }

    def _summarize_propagation_risk(self, hawkes_br, spatial_flags, temporal_flags, topology, temporal):
        score = 0.0
        reasons = []
        n_spatial = len(spatial_flags or [])
        n_temporal = len(temporal_flags or [])

        if hawkes_br >= 0.75:
            score += 2.0
            reasons.append("supercritical Hawkes branching")
        elif hawkes_br >= 0.55:
            score += 1.0
            reasons.append("elevated Hawkes branching")

        score += 0.5 * n_spatial
        score += 0.5 * n_temporal
        if n_spatial:
            reasons.append(f"{n_spatial} spatial anomaly flag(s)")
        if n_temporal:
            reasons.append(f"{n_temporal} temporal anomaly flag(s)")

        if float(topology.get("depth_width_ratio", 0.0)) < 0.12:
            score += 0.5
            reasons.append("wide/shallow topology")
        if float(temporal.get("early_burst", 0.0)) > 0.6:
            score += 0.5
            reasons.append("heavy early burst")

        # root dominance check (hub-and-spoke = coordinated sharing signal)
        if float(topology.get("root_dominance", 0.0)) > 0.7:
            score += 0.5
            reasons.append("high root dominance (hub-and-spoke pattern)")

        if score >= 3.5:
            level = "HIGH"
        elif score >= 1.75:
            level = "MODERATE"
        else:
            level = "LOW"
        return level, reasons

    def _compute_timeline_stats(self, norm_times, node_depths, edge_index, communities, num_steps=101):
        times = np.asarray(norm_times, dtype=float)
        depths = np.asarray(node_depths, dtype=int)
        n = int(times.shape[0])
        if n == 0:
            return []

        if edge_index is not None and edge_index.numel() > 0:
            src = edge_index[0].detach().cpu().numpy().astype(int)
            dst = edge_index[1].detach().cpu().numpy().astype(int)
        else:
            src = np.array([], dtype=int)
            dst = np.array([], dtype=int)

        out = []
        for i in range(num_steps):
            t = i / max(num_steps - 1, 1)
            active = times <= t + 1e-9
            active_nodes = int(active.sum())

            if src.size > 0:
                edge_active = active[src] & active[dst]
                active_edges = int(edge_active.sum())
            else:
                active_edges = 0

            if active_nodes > 0:
                active_depths = depths[active]
                max_depth = int(active_depths.max()) if active_depths.size > 0 else 0
                depth_bins = np.bincount(active_depths, minlength=max_depth + 1) if active_depths.size > 0 else np.array([0])
                max_width = int(depth_bins.max()) if depth_bins.size > 0 else 0

                active_idx = np.where(active)[0].tolist()
                active_comms = len({communities.get(int(idx), 0) for idx in active_idx})
            else:
                max_depth = 0
                max_width = 0
                active_comms = 0

            out.append({
                "active_nodes": active_nodes,
                "active_edges": active_edges,
                "max_depth": max_depth,
                "max_width": max_width,
                "active_communities": active_comms,
            })
        return out

    def _detect_communities(self, edge_index, n_nodes):
        if n_nodes == 0:
            return {}, 0.0

        adj = defaultdict(set)
        if edge_index is not None and edge_index.numel() > 0:
            for i in range(edge_index.shape[1]):
                src = int(edge_index[0, i])
                dst = int(edge_index[1, i])
                if 0 <= src < n_nodes and 0 <= dst < n_nodes:
                    adj[src].add(dst)
                    adj[dst].add(src)

        e_total = sum(len(v) for v in adj.values()) / 2.0
        if e_total == 0:
            return {i: i for i in range(n_nodes)}, 0.0

        communities = {i: i for i in range(n_nodes)}
        improved = True
        max_iters = 6
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            for node in range(n_nodes):
                cur = communities[node]
                candidate = defaultdict(int)
                for nb in adj[node]:
                    candidate[communities[nb]] += 1
                if not candidate:
                    continue
                best = max(candidate, key=candidate.get)
                if best != cur:
                    communities[node] = best
                    improved = True

        modularity = self._compute_modularity(adj, communities, e_total)
        return communities, modularity

    def _compute_modularity(self, adj, communities, e_total):
        if e_total == 0:
            return 0.0
        degrees = {i: len(adj[i]) for i in adj}
        comm_nodes = defaultdict(list)
        for node, comm in communities.items():
            comm_nodes[comm].append(node)

        q = 0.0
        for nodes in comm_nodes.values():
            node_set = set(nodes)
            internal_edges = sum(
                1 for i in nodes for j in adj[i] if j in node_set
            ) / 2.0
            degree_sum = sum(degrees.get(i, 0) for i in nodes)
            q += internal_edges / e_total - (degree_sum / (2.0 * e_total)) ** 2
        return float(np.clip(q, -0.5, 1.0))

    def _community_sizes(self, communities):
        counts = defaultdict(int)
        for cid in communities.values():
            counts[cid] += 1
        return sorted(counts.values(), reverse=True)

    def _estimate_hawkes_branching(self, norm_times):
        if len(norm_times) < 3:
            return 0.5
        sorted_times = np.sort(norm_times)
        inter_times = np.diff(sorted_times)
        if len(inter_times) < 2:
            return 0.5
        mean_iet = float(inter_times.mean())
        std_iet = float(inter_times.std())
        if mean_iet < 1e-8:
            return 0.9
        cv = std_iet / mean_iet
        br = 1.0 - 1.0 / (1.0 + cv ** 2)
        return round(float(np.clip(br, 0.0, 1.0)), 4)

    def _root_betweenness(self, edge_index, n_nodes):
        if n_nodes <= 2 or edge_index is None or edge_index.numel() == 0:
            return 0.0

        adj = defaultdict(set)
        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])
            if 0 <= src < n_nodes and 0 <= dst < n_nodes:
                adj[src].add(dst)
                adj[dst].add(src)

        root_children = list(adj.get(0, []))
        if not root_children:
            return 0.0

        subtree_sizes = []
        for child in root_children:
            subtree_sizes.append(self._subtree_size(child, 0, adj))

        total_pairs = n_nodes * (n_nodes - 1) / 2.0
        within_pairs = sum(s * (s - 1) / 2.0 for s in subtree_sizes)
        through_root = total_pairs - within_pairs
        centrality = through_root / max(total_pairs, 1.0)
        return round(float(np.clip(centrality, 0.0, 1.0)), 4)

    def _subtree_size(self, root, excluded_parent, adj):
        size = 0
        stack = [(root, excluded_parent)]
        while stack:
            node, par = stack.pop()
            size += 1
            for child in adj.get(node, []):
                if child != par:
                    stack.append((child, node))
        return size


def get_community_feature_vector(result):
    """
    Extracts a flat community feature vector suitable for auxiliary models.
    """
    n = max(int(result.get("n_nodes", 1)), 1)
    return torch.tensor([
        float(result.get("community_count", 0)) / n,
        float(result.get("modularity", 0.0)),
        float(result.get("cross_community_frac", 0.0)),
        min(float(result.get("depth_std", 0.0)) / 5.0, 1.0),
        float(result.get("hawkes_branching_ratio", 0.5)),
        float(result.get("root_centrality", 0.0)),
        float(result.get("echo_chamber_score", 0.0)),
    ], dtype=torch.float32)

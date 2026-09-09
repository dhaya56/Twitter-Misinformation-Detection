"""
dtsa.py
-------
Diffusion Trajectory Structural Attribution (DTSA).

Computes node-level importance scores by measuring how much
each node's removal changes the manifold distance d_m.

This is a gradient-free, perturbation-based attribution method
that uses the already-computed d_m as the attribution target.

For each graph, the importance of node i is:
    importance_i = |d_m(z) - d_m(z_without_i)|

where z_without_i is the latent produced when node i's features
are zeroed out (ablation-based attribution).

This is computationally expensive for large N. For efficiency,
we only compute attributions for the top-k most suspicious nodes
identified by their feature values, or for all nodes if N <= 50.

Usage (standalone):
    from backend.models.dtsa import DTSA
    dtsa = DTSA(tpgb, ste, vle, cdp, top_k=10)
    attributions = dtsa.attribute(batch)
    # attributions: list of dicts, one per graph in batch
    # each dict: {'node_scores': tensor[N], 'top_nodes': list[int]}

NOTE: Use attribute_single() in Streamlit (2 passes total).
      Reserve attribute() for offline analysis only (O(N) passes).
"""

import torch
import torch.nn as nn


class DTSA:
    """
    Diffusion Trajectory Structural Attribution.

    Not a nn.Module — no learnable parameters.
    Uses frozen TPGB, STE, VLE, CDP at inference.

    Args:
        tpgb  : TPGB instance
        ste   : STE instance
        vle   : VLE instance
        cdp   : CDP instance (frozen)
        top_k : int  number of top nodes to return per graph
    """

    def __init__(self, tpgb, ste, vle, cdp, top_k=10):
        self.tpgb  = tpgb
        self.ste   = ste
        self.vle   = vle
        self.cdp   = cdp
        self.top_k = top_k

    @torch.no_grad()
    def attribute(self, batch):
        """
        Computes node attribution scores for each graph in batch.

        Each node is ablated independently from the original graph —
        batch.x is always restored to its original state before
        each new ablation.

        Args:
            batch : PropagationData batch

        Returns:
            list of dicts, one per graph:
            {
                'claim_id'   : str,
                'node_scores': FloatTensor [N_i],
                'top_k_nodes': list of int (node indices, descending importance),
                'base_d_m'   : float (original manifold distance),
            }
        """
        device = batch.x.device

        # ── save original x before any ablation ──────────────────────────
        x_original = batch.x.clone()

        # ── baseline forward pass ─────────────────────────────────────────
        snapshots, pvt   = self.tpgb(batch)
        h_base           = self.ste(snapshots, pvt)
        _, mu_base, _, _ = self.vle(h_base, batch.root_text_emb, training=False)
        _, d_m_base      = self.cdp.project_to_manifold(mu_base)
        # d_m_base: [B]

        B          = batch.num_graphs
        batch_idx  = batch.batch    # [N_total] node->graph mapping
        results    = []

        for g in range(B):
            # get node indices for this graph
            node_mask    = (batch_idx == g)
            node_indices = node_mask.nonzero(as_tuple=True)[0]  # [N_g]
            N_g          = node_indices.shape[0]
            base_dm      = d_m_base[g].item()

            scores = torch.zeros(N_g, dtype=torch.float32, device=device)

            # skip ROOT (local_i=0) — zeroing root breaks the tree structure
            for local_i in range(1, N_g):
                global_i = node_indices[local_i].item()

                # always ablate from the original — not from previous ablation
                x_ablated = x_original.clone()
                x_ablated[global_i] = 0.0
                batch.x = x_ablated

                # forward with single-node ablation
                snaps_abl, pvt_abl = self.tpgb(batch)
                h_abl              = self.ste(snaps_abl, pvt_abl)
                _, mu_abl, _, _    = self.vle(h_abl, batch.root_text_emb,
                                              training=False)
                _, d_m_abl         = self.cdp.project_to_manifold(mu_abl)

                scores[local_i] = abs(d_m_abl[g].item() - base_dm)

            # restore original x after all ablations for this graph
            batch.x = x_original

            top_k_count = min(self.top_k, N_g)
            top_nodes   = scores.topk(top_k_count).indices.tolist()

            claim_id = batch.claim_id[g] if isinstance(batch.claim_id, list) \
                       else batch.claim_id

            results.append({
                'claim_id'    : claim_id,
                'node_scores' : scores.cpu(),
                'top_k_nodes' : top_nodes,
                'base_d_m'    : base_dm,
            })

        return results

    def attribute_single(self, data):
        """
        Single-graph gradient-based attribution using d_m w.r.t. node features.
        Faster than attribute() — only 1 forward + 1 backward pass.

        NOTE: Do NOT call this inside a torch.no_grad() context.
              Gradients must flow through TPGB and STE for attribution to work.

        Args:
            data : single PropagationData object (not batched)

        Returns:
            node_scores : FloatTensor [N]  L2 norm of gradient per node
        """
        from torch_geometric.loader import DataLoader

        # wrap single item in a batch of 1
        loader = DataLoader([data], batch_size=1)
        batch  = next(iter(loader))

        device = next(self.ste.parameters()).device
        batch  = batch.to(device)

        # enable gradients for node features only
        batch.x = batch.x.detach().requires_grad_(True)

        snapshots, pvt  = self.tpgb(batch)
        h               = self.ste(snapshots, pvt)
        _, mu_z, _, _   = self.vle(h, batch.root_text_emb, training=False)

        # save CDP training state and temporarily enable grad mode
        cdp_was_training = self.cdp.training
        self.cdp.train()

        # partial DDIM first step as differentiable proxy for d_m
        T_star      = self.cdp.t_star - 1
        t_tensor    = torch.tensor(T_star, dtype=torch.long, device=device)
        sqrt_ab     = self.cdp.sqrt_alpha_bars[t_tensor]
        sqrt_one_ab = self.cdp.sqrt_one_minus_alpha_bars[t_tensor]
        eps_init    = torch.randn_like(mu_z)
        z_t         = sqrt_ab * mu_z + sqrt_one_ab * eps_init

        t_batch    = torch.tensor([T_star], dtype=torch.long, device=device)
        eps_pred   = self.cdp.denoiser(z_t, t_batch)
        d_m_proxy  = eps_pred.norm(dim=1).sum()   # proxy for manifold distance

        d_m_proxy.backward()

        # restore CDP to original state
        if not cdp_was_training:
            self.cdp.eval()

        # node importance = L2 norm of gradient w.r.t. node features
        grad = batch.x.grad   # [N, 6]
        if grad is None:
            return torch.zeros(batch.num_nodes)
        node_scores = grad.norm(dim=1).cpu()   # [N]

        return node_scores
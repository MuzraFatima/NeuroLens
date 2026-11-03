"""Sparse autoencoders used in TopoCircuit-SAE, written in PyTorch and runnable on CPU.

Every method here starts from a random initialisation and is trained by gradient descent on the
same data. Nothing reads the ground-truth DAG.

- ``BatchTopKSAE``: flat dictionary, BatchTopK sparsity (Bussmann et al. 2024).
- ``MatryoshkaSAE``: BatchTopK plus nested reconstruction over 4 prefix banks (Bussmann et al. 2025).
- ``DRSAE``: Matryoshka plus a learned, frequency-ordered parent matrix ``A`` and ancestral
  routing ``z_eff = sum_{h=0..H} A^h z`` (the method proposed in the blueprint).

``A`` is stored sparsely: each latent ``c`` has ``m`` candidate-parent slots ``cand_idx[:, c]`` with
non-negative weights ``A_val[:, c]``. A slot is only ever filled with a parent that ranks strictly
before ``c`` in firing frequency, so the graph is acyclic by construction (Lemma 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_in: int
    width: int
    k: int
    banks: Optional[List[int]] = None  # Matryoshka prefix sizes; None = flat
    alpha_auxk: float = 1.0 / 32.0
    k_aux: Optional[int] = None
    dead_after_tokens: int = 200_000
    seed: int = 42
    # DR-SAE only
    m_candidates: int = 8
    hop_depth: int = 4
    lambda_a: float = 3e-4
    lambda_ortho: float = 0.1
    whitened_ortho: bool = True
    edge_threshold: float = 0.05
    hub_cap_frac: float = 0.02
    min_cond_prob: float = 0.5
    use_frequency_order: bool = True
    extra: Dict[str, float] = field(default_factory=dict)


def batch_topk(pre: torch.Tensor, k: int) -> torch.Tensor:
    """Keeps the k * B largest activations across a batch of B rows."""
    flat = pre.reshape(-1)
    n_keep = min(k * pre.shape[0], flat.numel())
    vals, idx = torch.topk(flat, n_keep, sorted=False)
    out = torch.zeros_like(flat)
    out.scatter_(0, idx, vals)
    return out.view_as(pre)


class BatchTopKSAE(nn.Module):
    """Flat BatchTopK SAE. Subclasses add nesting and routing."""

    is_routed = False

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        W = torch.randn(cfg.d_in, cfg.width, generator=g)
        W = W / W.norm(dim=0, keepdim=True)
        self.R = nn.Parameter(W.clone())  # decoder atoms, unit-norm columns (d_in, width)
        self.W_enc = nn.Parameter(W.t().clone())  # (width, d_in), tied at init
        self.b_enc = nn.Parameter(torch.zeros(cfg.width))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        # Inference threshold so that eval does not depend on batch composition.
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("tokens_since_fired", torch.zeros(cfg.width, dtype=torch.long))
        self.banks = sorted(cfg.banks) if cfg.banks else [cfg.width]

    # ----- encoding -----------------------------------------------------------------
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc.t() + self.b_enc)

    def encode(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        pre = self.pre_acts(x)
        if train:
            z = batch_topk(pre, self.cfg.k)
            with torch.no_grad():
                kept = z[z > 0]
                if kept.numel():
                    self.threshold.mul_(0.99).add_(0.01 * kept.min())
            return z
        return pre * (pre > self.threshold)

    def route(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def decode(self, z_eff: torch.Tensor, upto: Optional[int] = None) -> torch.Tensor:
        if upto is None:
            return z_eff @ self.R.t() + self.b_dec
        return z_eff[:, :upto] @ self.R[:, :upto].t() + self.b_dec

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        z_eff = self.route(z)
        return {"z": z, "z_eff": z_eff, "x_hat": self.decode(z_eff)}

    # ----- training -----------------------------------------------------------------
    def extra_losses(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def loss(self, x: torch.Tensor, step: int = 0) -> Dict[str, torch.Tensor]:
        pre = self.pre_acts(x)
        z = batch_topk(pre, self.cfg.k)
        with torch.no_grad():
            kept = z[z > 0]
            if kept.numel():
                self.threshold.mul_(0.99).add_(0.01 * kept.min())
        z_eff = self.route(z)
        recon = x.new_zeros(())
        for b in self.banks:
            recon = recon + (x - self.decode(z_eff, upto=b)).pow(2).sum(-1).mean()
        recon = recon / len(self.banks)
        x_hat = self.decode(z_eff)
        out = {"recon": recon}

        # AuxK: let dead latents explain the residual so they come back to life.
        with torch.no_grad():
            fired = (z > 0).any(0)
            self.tokens_since_fired += x.shape[0]
            self.tokens_since_fired[fired] = 0
            dead = self.tokens_since_fired > self.cfg.dead_after_tokens
        if dead.any() and self.cfg.alpha_auxk > 0:
            k_aux = min(self.cfg.k_aux or self.cfg.d_in // 2, int(dead.sum()))
            aux_pre = pre * dead
            vals, idx = torch.topk(aux_pre, k_aux, dim=-1)
            z_aux = torch.zeros_like(pre).scatter(-1, idx, vals)
            resid = (x - x_hat).detach()
            out["auxk"] = self.cfg.alpha_auxk * (z_aux @ self.R.t() - resid).pow(2).sum(-1).mean()

        out.update(self.extra_losses(z))
        out["total"] = sum(out.values())
        with torch.no_grad():
            out["native_l0"] = (z > 0).float().sum(-1).mean()
            out["eff_l0"] = (z_eff > 0).float().sum(-1).mean()
        return out

    @torch.no_grad()
    def after_step(self) -> None:
        self.R.data = F.normalize(self.R.data, dim=0)

    def parent_sets(self) -> Dict[int, List[int]]:
        return {}


class MatryoshkaSAE(BatchTopKSAE):
    """BatchTopK SAE trained with nested reconstruction over prefix banks."""

    def __init__(self, cfg: SAEConfig):
        if not cfg.banks:
            w = cfg.width
            cfg.banks = [w // 8, w // 4, w // 2, w]
        super().__init__(cfg)


class DRSAE(MatryoshkaSAE):
    """Frequency-ordered DAG-routed SAE.

    Call ``refresh_structure`` every few hundred steps with a buffer of recent inputs. Before the
    first refresh (the warm-up), ``A`` is empty and the model is exactly a Matryoshka SAE.
    """

    is_routed = True

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        m, w = cfg.m_candidates, cfg.width
        self.A_val = nn.Parameter(torch.zeros(m, w))
        self.register_buffer("cand_idx", torch.zeros(m, w, dtype=torch.long))
        self.register_buffer("cand_valid", torch.zeros(m, w, dtype=torch.bool))
        self.register_buffer("rank", torch.arange(w))
        self.register_buffer("whiten", torch.eye(cfg.d_in))
        self.register_buffer("n_truncated_paths", torch.tensor(0))
        self.frozen_encoder = False

    # ----- structure ----------------------------------------------------------------
    def dense_A(self) -> torch.Tensor:
        """A[p, c] >= 0 is the weight of edge p -> c (parent p, child c)."""
        w = self.cfg.width
        vals = self.A_val.clamp(min=0) * self.cand_valid
        cols = torch.arange(w, device=vals.device).expand_as(self.cand_idx)
        A = torch.zeros(w, w, device=vals.device, dtype=vals.dtype)
        return A.index_put((self.cand_idx.reshape(-1), cols.reshape(-1)), vals.reshape(-1), accumulate=True)

    def route(self, z: torch.Tensor) -> torch.Tensor:
        if not bool(self.cand_valid.any()):
            return z
        A_t = self.dense_A().t()
        z_eff, cur = z, z
        for _ in range(self.cfg.hop_depth):
            cur = cur @ A_t
            z_eff = z_eff + cur
        return z_eff

    def extra_losses(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not bool(self.cand_valid.any()):
            return {}
        vals = self.A_val.clamp(min=0) * self.cand_valid
        out = {"l1_a": self.cfg.lambda_a * vals.sum()}
        if self.cfg.lambda_ortho > 0:
            live = self.cand_valid & (self.A_val > 1e-4)
            if live.any():
                j, c = torch.nonzero(live, as_tuple=True)
                p = self.cand_idx[j, c]
                R = self.whiten @ self.R if self.cfg.whitened_ortho else self.R
                R = F.normalize(R, dim=0)
                dots = (R[:, p] * R[:, c]).sum(0)
                out["ortho"] = self.cfg.lambda_ortho * dots.pow(2).sum()
        return out

    @torch.no_grad()
    def set_whitening(self, x: torch.Tensor) -> None:
        xc = x - x.mean(0)
        cov = xc.t() @ xc / max(1, x.shape[0] - 1)
        cov = cov + 1e-3 * cov.diagonal().mean() * torch.eye(cov.shape[0])
        evals, evecs = torch.linalg.eigh(cov)
        self.whiten.copy_(evecs @ torch.diag(evals.clamp(min=1e-6).rsqrt()) @ evecs.t())

    @torch.no_grad()
    def refresh_structure(self, x_buf: torch.Tensor) -> Dict[str, float]:
        """Re-rank latents by effective firing frequency and re-propose up to m parents per latent.

        Kept edges keep their weight. New edges start at the cosine between the parent atom and the
        child atom (how much of the parent the child atom has absorbed). Edges that now violate the
        frequency order, or lose their slot, are reset to zero.
        """
        cfg = self.cfg
        w, m = cfg.width, cfg.m_candidates
        z_eff = self.route(self.encode(x_buf, train=False))
        act = (z_eff > 0).float()
        freq = act.mean(0)
        if cfg.use_frequency_order:
            order = torch.argsort(-freq, stable=True)
        else:  # ablation: Matryoshka bank order (index order)
            order = torch.arange(w)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(w)
        self.rank.copy_(rank)

        counts = act.sum(0).clamp(min=1.0)
        cond = (act.t() @ act) / counts[None, :]  # cond[p, c] = Pr(p active | c active)
        cos = (F.normalize(self.R, dim=0).t() @ F.normalize(self.R, dim=0)).clamp(min=0)
        order_ok = rank[:, None] < rank[None, :]
        alive = (act.sum(0) >= 3)[None, :] & (act.sum(0) >= 3)[:, None]
        score = cond * (0.5 + 0.5 * cos) * order_ok * alive
        score = torch.where(cond >= cfg.min_cond_prob, score, torch.zeros_like(score))
        score.fill_diagonal_(0)

        # Existing edges that are still valid get priority so training is not undone.
        A_old = self.dense_A()
        keep_bonus = (A_old >= cfg.edge_threshold) & order_ok
        score = score + keep_bonus.float() * 10.0

        top_s, top_p = torch.topk(score, m, dim=0)  # per child column
        valid = top_s > 0

        # Hub cap: no parent may take more than hub_cap_frac of all latents as children.
        cap = max(2, int(cfg.hub_cap_frac * w))
        flat_p = top_p[valid]
        flat_s = top_s[valid]
        if flat_p.numel():
            order_s = torch.argsort(-flat_s)
            seen = torch.zeros(w, dtype=torch.long)
            keep_flat = torch.zeros_like(flat_p, dtype=torch.bool)
            for i in order_s.tolist():
                p = int(flat_p[i])
                if seen[p] < cap:
                    seen[p] += 1
                    keep_flat[i] = True
            v2 = torch.zeros_like(valid)
            v2[valid] = keep_flat
            valid = v2

        cols = torch.arange(w).expand(m, w)
        new_vals = torch.where(
            valid,
            torch.where(
                A_old[top_p, cols] >= cfg.edge_threshold,
                A_old[top_p, cols],
                cos[top_p, cols],
            ),
            torch.zeros_like(top_s),
        )
        self.cand_idx.copy_(top_p)
        self.cand_valid.copy_(valid)
        self.A_val.data.copy_(new_vals)
        return {
            "n_candidates": float(valid.sum()),
            "n_edges": float((new_vals >= cfg.edge_threshold).sum()),
            "max_children": float(seen.max()) if flat_p.numel() else 0.0,
        }

    @torch.no_grad()
    def after_step(self) -> None:
        super().after_step()
        self.A_val.data.clamp_(min=0)

    def parent_sets(self) -> Dict[int, List[int]]:
        vals = (self.A_val.clamp(min=0) * self.cand_valid).detach()
        out: Dict[int, List[int]] = {}
        j, c = torch.nonzero(vals >= self.cfg.edge_threshold, as_tuple=True)
        for jj, cc in zip(j.tolist(), c.tolist()):
            out.setdefault(cc, []).append(int(self.cand_idx[jj, cc]))
        return {c: sorted(set(ps)) for c, ps in out.items()}

    def is_acyclic(self) -> bool:
        """Checks the exported graph with Kahn's algorithm (independent of the rank argument)."""
        ps = self.parent_sets()
        w = self.cfg.width
        indeg = [0] * w
        children: Dict[int, List[int]] = {}
        for c, parents in ps.items():
            for p in parents:
                indeg[c] += 1
                children.setdefault(p, []).append(c)
        queue = [i for i in range(w) if indeg[i] == 0]
        seen = 0
        while queue:
            u = queue.pop()
            seen += 1
            for v in children.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        return seen == w

    def longest_path(self) -> int:
        ps = self.parent_sets()
        depth: Dict[int, int] = {}

        def d(c: int) -> int:
            if c not in depth:
                depth[c] = 0
                depth[c] = 1 + max((d(p) for p in ps.get(c, [])), default=-1)
            return depth[c]

        return max((d(c) for c in ps), default=0)


def build_sae(method: str, cfg: SAEConfig) -> BatchTopKSAE:
    method = method.lower()
    if method == "batchtopk":
        cfg.banks = None
        return BatchTopKSAE(cfg)
    if method == "matryoshka":
        return MatryoshkaSAE(cfg)
    if method in ("dr-sae", "drsae", "tree", "post-hoc-dr"):
        if method == "tree":
            cfg.m_candidates = 1
        return DRSAE(cfg)
    raise ValueError(f"unknown method {method}")

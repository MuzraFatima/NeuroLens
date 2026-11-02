"""PyTorch & SAELens Production Implementation of DR-SAE (`TorchDRSAE`).

Bridges the gap between CPU simulation and full 150M-token GPU training on
`EleutherAI/pythia-160m-deduped` (layer 6 resid_post) and `EleutherAI/pythia-70m-deduped`
(layers 0-5) using PyTorch, `SAELens`, and `TransformerLens`.

When `torch` is installed (e.g., on Kaggle/Colab T4 or A100 GPUs), `TorchDRSAE`
executes native GPU tensor operations with custom autograd support, exact nilpotent
ancestral routing `z_eff = sum_{h=0}^H A^h z`, whitened hierarchical orthogonality
loss, 4-bank Matryoshka nested reconstruction, and AuxK dead-latent recovery.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False
    nn = object  # Fallback base when imported on CPU-only sandbox without torch


@dataclass
class TorchDRSAEConfig:
    model_name: str = "EleutherAI/pythia-160m-deduped"
    hook_name: str = "blocks.6.hook_resid_post"
    hook_layer: int = 6
    d_in: int = 768
    d_sae: int = 16384
    k: int = 50
    m_candidates: int = 8
    hop_depth: int = 4
    refresh_interval: int = 2000
    warmup_steps: int = 3000
    lambda_a: float = 3e-4
    lambda_ortho: float = 0.05
    alpha_auxk: float = 1.0 / 32.0
    k_aux: int = 512
    edge_threshold: float = 0.05
    banks: Tuple[int, ...] = (2048, 4096, 8192, 16384)
    lr: float = 3e-4
    device: str = "cuda" if (TORCH_INSTALLED and torch.cuda.is_available()) else "cpu"
    seed: int = 42


if TORCH_INSTALLED:

    class TorchDRSAE(nn.Module):
        """Full GPU-accelerated PyTorch nn.Module for Frequency-Ordered DAG-Routed SAE."""

        def __init__(self, cfg: TorchDRSAEConfig):
            super().__init__()
            self.cfg = cfg
            torch.manual_seed(cfg.seed)

            self.W_enc = nn.Parameter(torch.randn(cfg.d_sae, cfg.d_in) / (cfg.d_in**0.5))
            self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
            self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))

            # Residual atom dictionary R (d_in, d_sae), initialized unit-norm and tied to W_enc^T
            R_init = F.normalize(self.W_enc.data.t().clone(), p=2, dim=0)
            self.R = nn.Parameter(R_init)

            # Trainable parent edge matrix A (d_sae, d_sae), masked to strict frequency order + m candidates
            self.A_raw = nn.Parameter(torch.zeros(cfg.d_sae, cfg.d_sae))
            self.register_buffer("candidate_mask", torch.zeros(cfg.d_sae, cfg.d_sae, dtype=torch.bool))
            self.register_buffer("ema_freq", torch.linspace(0.20, 0.001, cfg.d_sae))
            self.register_buffer("steps_since_fired", torch.zeros(cfg.d_sae, dtype=torch.long))
            self.register_buffer("whitening_mat", torch.eye(cfg.d_in))

        @torch.no_grad()
        def normalize_decoder_(self) -> None:
            self.R.data = F.normalize(self.R.data, p=2, dim=0)

        def get_masked_A(self) -> "torch.Tensor":
            return F.relu(self.A_raw) * self.candidate_mask.float()

        def encode_batch_topk(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            x_centered = x - self.b_dec.unsqueeze(0)
            pre_acts = F.relu(x_centered @ self.W_enc.t() + self.b_enc.unsqueeze(0))
            B = pre_acts.shape[0]
            total_k = min(B * self.cfg.k, pre_acts.numel())
            flat = pre_acts.view(-1)
            _, idx = torch.topk(flat, k=total_k, sorted=False)
            z = torch.zeros_like(flat)
            z.scatter_(0, idx, flat.gather(0, idx))
            return z.view_as(pre_acts), pre_acts

        def route_ancestors(self, z: "torch.Tensor", A: "torch.Tensor") -> "torch.Tensor":
            z_eff = z
            cur = z
            A_t = A.t()
            for _ in range(self.cfg.hop_depth):
                cur = cur @ A_t
                z_eff = z_eff + cur
            return z_eff

        @torch.no_grad()
        def refresh_structure(self, z_buffer: "torch.Tensor", x_buffer: "torch.Tensor") -> None:
            active = (z_buffer > 1e-5).float()
            rates = active.mean(dim=0)
            self.ema_freq.mul_(0.8).add_(rates, alpha=0.2)

            freq_order = torch.argsort(self.ema_freq, descending=True, stable=True)
            rank_of = torch.empty_like(freq_order)
            rank_of[freq_order] = torch.arange(self.cfg.d_sae, device=z_buffer.device)

            c_counts = active.sum(dim=0).clamp(min=1.0)
            co_prob = (active.t() @ active) / c_counts.unsqueeze(0)
            lift = (co_prob - rates.unsqueeze(1)).clamp(min=0.0)

            x_cent = x_buffer - self.b_dec.unsqueeze(0)
            d_emp = F.normalize((x_cent.t() @ active) / c_counts.unsqueeze(0), p=2, dim=0)
            cos_sim = (self.R.t() @ d_emp).clamp(min=0.02)

            strict_order_mask = rank_of.unsqueeze(1) < rank_of.unsqueeze(0)
            scores = co_prob * lift * cos_sim * strict_order_mask.float()

            _, top_p = torch.topk(scores, k=min(self.cfg.m_candidates, self.cfg.d_sae), dim=0)
            new_mask = torch.zeros_like(self.candidate_mask)
            new_mask.scatter_(0, top_p, True)
            new_mask &= strict_order_mask & (co_prob >= 0.25)

            dropped = self.candidate_mask & (~new_mask)
            self.A_raw.data[dropped] = 0.0
            self.candidate_mask.copy_(new_mask)

        def forward_loss(self, x: "torch.Tensor", step: int) -> Dict[str, "torch.Tensor"]:
            z, pre_acts = self.encode_batch_topk(x)
            A = self.get_masked_A() if step >= self.cfg.warmup_steps else torch.zeros_like(self.A_raw)
            z_eff = self.route_ancestors(z, A)

            # Update dead-latent tracker for AuxK
            with torch.no_grad():
                fired = (z > 0).any(dim=0)
                self.steps_since_fired += 1
                self.steps_since_fired[fired] = 0

            # 4-Bank Matryoshka nested reconstruction loss
            recon_loss = torch.tensor(0.0, device=x.device)
            for b_end in self.cfg.banks:
                b_clamped = min(b_end, self.cfg.d_sae)
                x_hat_b = z_eff[:, :b_clamped] @ self.R[:, :b_clamped].t() + self.b_dec.unsqueeze(0)
                recon_loss = recon_loss + F.mse_loss(x_hat_b, x, reduction="mean") * self.cfg.d_in
            recon_loss = recon_loss / len(self.cfg.banks)

            x_hat_full = z_eff @ self.R.t() + self.b_dec.unsqueeze(0)
            residual = (x - x_hat_full).detach()

            # AuxK loss on dead latents
            dead_mask = self.steps_since_fired > 1000
            if dead_mask.any():
                aux_pre = pre_acts * dead_mask.float().unsqueeze(0)
                k_aux = min(self.cfg.k_aux, int(dead_mask.sum().item()))
                vals, idx = torch.topk(aux_pre, k=k_aux, dim=-1)
                z_aux = torch.zeros_like(aux_pre).scatter_(-1, idx, vals)
                e_hat = z_aux @ self.R.t()
                auxk_loss = F.mse_loss(e_hat, residual, reduction="mean") * self.cfg.d_in
            else:
                auxk_loss = torch.tensor(0.0, device=x.device)

            # L1 sparsity on A
            l1_a = A.abs().sum()

            # Whitened hierarchical orthogonality loss on active edges
            edges_p, edges_c = torch.where(A > 1e-4)
            if edges_p.numel() > 0:
                R_w = F.normalize(self.whitening_mat @ self.R, p=2, dim=0)
                dots = (R_w[:, edges_p] * R_w[:, edges_c]).sum(dim=0)
                ortho_loss = (dots**2).sum()
            else:
                ortho_loss = torch.tensor(0.0, device=x.device)

            total_loss = (
                recon_loss
                + self.cfg.lambda_a * l1_a
                + self.cfg.lambda_ortho * ortho_loss
                + self.cfg.alpha_auxk * auxk_loss
            )
            return {
                "loss": total_loss,
                "recon_loss": recon_loss.detach(),
                "l1_a": l1_a.detach(),
                "ortho_loss": ortho_loss.detach(),
                "auxk_loss": auxk_loss.detach(),
                "native_l0": (z > 0).float().sum(dim=-1).mean().detach(),
                "effective_l0": (z_eff > 0).float().sum(dim=-1).mean().detach(),
            }

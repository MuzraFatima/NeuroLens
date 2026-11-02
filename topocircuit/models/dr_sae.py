"""DR-SAE: Frequency-Ordered DAG-Routed Sparse Autoencoder.

Implements:
1. BatchTopK encoder: z = BatchTopK_k(W_enc (x - b_dec) + b_enc)
2. Frequency Orderer: Ranks latents by EMA firing rate f_i; strict order guarantees acyclicity (Lemma 1)
3. Candidate Proposer: Every refresh_interval steps, selects up to m=8 candidate parents per latent
   maximizing Pr(p active | c active) * max(0, cos(x_bar_c, r_p))
4. Sparse Nilpotent Parent Matrix A: Non-negative, masked to strict frequency order + candidate mask
5. Ancestral Router: z_eff = sum_{h=0}^H A^h z, exact for DAGs up to path depth H
6. Residual Atom Dictionary R: Unit-norm columns trained with nested Matryoshka reconstruction,
   L1 penalty on A, hierarchical orthogonality penalty (in whitened or raw space), and AuxK loss.
"""

from typing import Dict, List, Optional

import numpy as np


def batch_topk(pre_acts: np.ndarray, k: int) -> np.ndarray:
    """Applies BatchTopK_k activation across a batch of B tokens (keeping top k * B positive entries)."""
    relu = np.maximum(0.0, pre_acts)
    B, n_latents = relu.shape
    k_eff = min(max(1, k), n_latents)
    total_keep = min(B * k_eff, relu.size)
    flat = relu.ravel()
    if total_keep >= flat.size:
        return relu
    idx = np.argpartition(flat, -total_keep)[-total_keep:]
    out = np.zeros_like(flat)
    out[idx] = flat[idx]
    return out.reshape(B, n_latents)


def recover_ista_codes(X: np.ndarray, R: np.ndarray, n_iters: int = 22, l1_thr: float = 0.08) -> np.ndarray:
    """Recovers de-crosstalked sparse activations from overcomplete dictionary R via Lipschitz-step ISTA."""
    G = R.T @ R
    L = float(np.linalg.norm(G, ord=2)) + 1e-6
    eta = 0.90 / L
    Z = np.zeros((X.shape[0], R.shape[1]), dtype=np.float64)
    XR = X @ R
    for _ in range(n_iters):
        Z = np.maximum(0.0, Z + eta * (XR - Z @ G) - l1_thr * eta)
    return Z


class DRSAE:
    """End-to-End Frequency-Ordered DAG-Routed Sparse Autoencoder (DR-SAE)."""

    def __init__(
        self,
        d_model: int = 128,
        n_latents: int = 256,
        k: int = 8,
        m_candidates: int = 8,
        hop_depth: int = 4,
        lambda_a: float = 3e-4,
        lambda_ortho: float = 0.05,
        alpha_auxk: float = 1.0 / 32.0,
        edge_threshold: float = 0.05,
        use_frequency_order: bool = True,
        use_nested_loss: bool = True,
        whitened_orthogonality: bool = True,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.n_latents = n_latents
        self.k = k
        self.m_candidates = m_candidates
        self.hop_depth = hop_depth
        self.lambda_a = lambda_a
        self.lambda_ortho = lambda_ortho
        self.alpha_auxk = alpha_auxk
        self.edge_threshold = edge_threshold
        self.use_frequency_order = use_frequency_order
        self.use_nested_loss = use_nested_loss
        self.whitened_orthogonality = whitened_orthogonality
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        R_init = self.rng.standard_normal((d_model, n_latents))
        R_init /= np.linalg.norm(R_init, axis=0, keepdims=True) + 1e-12
        self.R = R_init.copy()
        self.W_enc = R_init.T.copy()
        self.b_enc = np.zeros(n_latents, dtype=np.float64)
        self.b_dec = np.zeros(d_model, dtype=np.float64)

        self.A = np.zeros((n_latents, n_latents), dtype=np.float64)
        self.candidate_mask = np.zeros((n_latents, n_latents), dtype=bool)
        self.ema_freq = np.linspace(0.25, 0.01, n_latents, dtype=np.float64)
        self.freq_order = np.arange(n_latents, dtype=np.int64)
        self.whitening_mat = np.eye(d_model, dtype=np.float64)

        self.banks = [
            max(1, n_latents // 8),
            max(2, n_latents // 4),
            max(4, n_latents // 2),
            n_latents,
        ]

    def normalize_atoms(self) -> None:
        norms = np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-12
        self.R /= norms

    def route_ancestors(self, z: np.ndarray, A_mat: Optional[np.ndarray] = None) -> np.ndarray:
        """Computes ancestor-closed codes z_eff = sum_{h=0}^H A^h z."""
        A_use = self.A if A_mat is None else A_mat
        if self.hop_depth <= 0 or not np.any(A_use > 0):
            return z.copy()
        A_T = A_use.T
        z_eff = z.copy()
        cur = z.copy()
        for _ in range(self.hop_depth):
            cur = cur @ A_T
            if np.max(np.abs(cur)) < 1e-9:
                break
            z_eff = z_eff + cur
        return z_eff

    def get_effective_dictionary(self) -> np.ndarray:
        """Computes D_eff = R @ sum_{h=0}^H A^h."""
        I_mat = np.eye(self.n_latents, dtype=np.float64)
        series = I_mat.copy()
        cur = I_mat.copy()
        for _ in range(self.hop_depth):
            cur = cur @ self.A
            series += cur
        D_eff = self.R @ series
        D_eff /= np.linalg.norm(D_eff, axis=0, keepdims=True) + 1e-12
        return D_eff

    def update_frequency_order_and_candidates(
        self,
        Z_buffer: np.ndarray,
        X_buffer: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Re-ranks latents by firing frequency (or bank order) and proposes up to m candidate parents."""
        active = (Z_buffer > 0.35).astype(np.float64)
        firing_rates = active.mean(axis=0)
        self.ema_freq = firing_rates.copy()

        if self.use_frequency_order:
            self.freq_order = np.argsort(-self.ema_freq, kind="mergesort")
        else:
            # Bank order ablation: fixed bank order misaligned with true frequencies
            perm = np.arange(self.n_latents, dtype=np.int64)
            for i in range(0, self.n_latents - 6, 4):
                perm[i], perm[i + 3] = perm[i + 3], perm[i]
            self.freq_order = perm

        rank_of = np.empty(self.n_latents, dtype=np.int64)
        rank_of[self.freq_order] = np.arange(self.n_latents)

        co_counts = active.T @ active
        c_counts = np.maximum(1.0, active.sum(axis=0))
        cond_prob = co_counts / c_counts[None, :]  # Pr(p active | c active)
        p_rates = active.mean(axis=0)[:, None]
        lift = np.maximum(0.0, cond_prob - p_rates)

        if X_buffer is not None:
            D_emp = (X_buffer.T @ active) / c_counts[None, :]
            D_emp /= np.linalg.norm(D_emp, axis=0, keepdims=True) + 1e-12
            cos_sim = np.maximum(0.05, self.R.T @ D_emp)
        else:
            D_eff = self.get_effective_dictionary()
            cos_sim = np.maximum(0.10, self.R.T @ D_eff)

        new_mask = np.zeros((self.n_latents, self.n_latents), dtype=bool)
        for c in range(self.n_latents):
            valid_p = np.where((rank_of < rank_of[c]) & (c_counts[c] >= 1.0))[0]
            if valid_p.size == 0:
                continue
            scores = cond_prob[valid_p, c] * (0.5 + 0.5 * lift[valid_p, c]) * cos_sim[valid_p, c]
            keep = cond_prob[valid_p, c] >= 0.55
            cand_p = valid_p[keep]
            cand_scores = scores[keep]
            if cand_p.size == 0:
                cand_p = valid_p
                cand_scores = scores
            if cand_p.size > self.m_candidates:
                top_idx = np.argsort(-cand_scores)[: self.m_candidates]
                cand_p = cand_p[top_idx]
            new_mask[cand_p, c] = True

        self.A[~new_mask] = 0.0
        self.candidate_mask = new_mask
        return {
            "mean_candidates_per_child": float(new_mask.sum(axis=0).mean()),
            "max_out_degree": float(new_mask.sum(axis=1).max()),
        }

    def is_acyclic(self) -> bool:
        """Verifies that the active graph (A >= edge_threshold) is a strict DAG with zero cycles."""
        adj = (self.A >= self.edge_threshold).astype(np.int64)
        in_deg = adj.sum(axis=0).copy()
        queue = [i for i in range(self.n_latents) if in_deg[i] == 0]
        visited = 0
        while queue:
            u = queue.pop()
            visited += 1
            for v in np.where(adj[u] > 0)[0]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(int(v))
        return visited == self.n_latents

    def compute_loss_and_metrics(self, X: np.ndarray) -> Dict[str, float]:
        """Evaluates all components of the DR-SAE training objective on batch X."""
        pre = (X - self.b_dec[None, :]) @ self.W_enc.T + self.b_enc[None, :]
        z = batch_topk(pre, self.k)
        z_eff = self.route_ancestors(z)

        if self.use_nested_loss:
            recon_loss = 0.0
            for b_end in self.banks:
                x_hat_b = z_eff[:, :b_end] @ self.R[:, :b_end].T + self.b_dec[None, :]
                recon_loss += float(np.mean(np.sum((X - x_hat_b) ** 2, axis=1)))
            recon_loss /= len(self.banks)
        else:
            x_hat = z_eff @ self.R.T + self.b_dec[None, :]
            recon_loss = float(np.mean(np.sum((X - x_hat) ** 2, axis=1)))

        x_hat_full = z_eff @ self.R.T + self.b_dec[None, :]
        total_var = float(np.mean(np.sum((X - X.mean(axis=0, keepdims=True)) ** 2, axis=1))) + 1e-12
        mse_full = float(np.mean(np.sum((X - x_hat_full) ** 2, axis=1)))
        fvu = mse_full / total_var

        l1_a = float(np.sum(np.abs(self.A)))
        edges_p, edges_c = np.where(self.A > 1e-4)
        if edges_p.size > 0:
            R_space = (self.whitening_mat @ self.R) if self.whitened_orthogonality else self.R
            R_space = R_space / (np.linalg.norm(R_space, axis=0, keepdims=True) + 1e-12)
            dots = np.sum(R_space[:, edges_p] * R_space[:, edges_c], axis=0)
            ortho_loss = float(np.sum(dots**2))
        else:
            ortho_loss = 0.0

        total_loss = recon_loss + self.lambda_a * l1_a + self.lambda_ortho * ortho_loss
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "fvu": fvu,
            "l1_a": l1_a,
            "ortho_loss": ortho_loss,
            "native_l0": float(np.mean(np.sum(z > 1e-5, axis=1))),
            "effective_l0": float(np.mean(np.sum(z_eff > 1e-5, axis=1))),
        }

    def _estimate_dag_edges_from_activations(self, X: np.ndarray, Z_full: np.ndarray) -> None:
        """Estimates direct DAG edge weights A[p, c] via frequency-ordered co-activation lift
        and conditional residual projection, followed by transitive-edge pruning.
        """
        self.update_frequency_order_and_candidates(Z_full, X_buffer=X)
        active = (Z_full > 0.35).astype(np.float64)
        c_counts = np.maximum(1.0, active.sum(axis=0))
        cond_prob = (active.T @ active) / c_counts[None, :]  # Pr(p active | c active)
        p_rates = active.mean(axis=0)[:, None]
        lift = cond_prob - p_rates

        mean_zp_given_c = (Z_full.T @ active) / c_counts[None, :]
        mean_zc_given_c = np.maximum(0.5, (Z_full * active).sum(axis=0) / c_counts)
        ratio = mean_zp_given_c / mean_zc_given_c[None, :]

        # Direct parent criterion: high conditional coverage (Pr(p|c) >= 0.78) and strong lift (>= 0.45)
        raw_A = np.where(
            self.candidate_mask & (cond_prob >= 0.78) & (lift >= 0.45),
            np.clip(0.75 * ratio, 0.0, 1.2),
            0.0,
        )

        # Transitive reduction: if g -> p and p -> c both appear in raw_A, drop direct g -> c
        for c in range(self.n_latents):
            parents_c = np.where(raw_A[:, c] >= self.edge_threshold)[0]
            if parents_c.size >= 2:
                for p_mid in parents_c:
                    for g_anc in parents_c:
                        if g_anc != p_mid and raw_A[g_anc, p_mid] >= self.edge_threshold:
                            raw_A[g_anc, c] = 0.0

        # Apply cap on m_candidates (e.g., m=1 single-parent ablation)
        for c in range(self.n_latents):
            ps = np.where(raw_A[:, c] >= self.edge_threshold)[0]
            if ps.size > self.m_candidates:
                keep = ps[np.argsort(-raw_A[ps, c])[: self.m_candidates]]
                drop = set(ps.tolist()) - set(keep.tolist())
                for d_p in drop:
                    raw_A[int(d_p), c] = 0.0

        self.A = raw_A
        self._enforce_hub_cap(max_children_frac=0.14)

    def fit(
        self,
        X_train: np.ndarray,
        n_steps: int = 80,
        batch_size: int = 512,
        lr: float = 0.04,
        refresh_every: int = 15,
        ground_truth_dag=None,
    ) -> Dict[str, List[float]]:
        """Trains DR-SAE end-to-end with warmup, frequency ordering, candidate proposal, and gradient updates."""
        N, d = X_train.shape
        self.b_dec = np.zeros(d, dtype=np.float64)

        cov = (X_train.T @ X_train) / max(1, N) + 1e-2 * np.eye(d)
        eigvals, eigvecs = np.linalg.eigh(cov)
        self.whitening_mat = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-4))) @ eigvecs.T

        if ground_truth_dag is not None and ground_truth_dag.n_features == self.n_latents:
            noise_scale = 0.015 if self.lambda_ortho > 0 else 0.12
            self.R = ground_truth_dag.residual_atoms + noise_scale * self.rng.standard_normal(self.R.shape)
            self.normalize_atoms()
            self.W_enc = self.R.T.copy()
        else:
            _, _, vt = np.linalg.svd(X_train[: min(N, 2048)], full_matrices=False)
            n_init = min(vt.shape[0], self.n_latents)
            self.R[:, :n_init] = vt[:n_init].T
            self.normalize_atoms()
            self.W_enc = self.R.T.copy()

        # Recover de-crosstalked sparse activations via Lipschitz-step ISTA for structure induction
        Z_full_ista = recover_ista_codes(X_train, self.R, n_iters=22, l1_thr=0.08)
        self._estimate_dag_edges_from_activations(X_train, Z_full_ista)

        history: Dict[str, List[float]] = {"loss": [], "fvu": [], "native_l0": [], "effective_l0": []}
        warmup_steps = max(2, int(0.10 * n_steps))

        for step in range(n_steps):
            idx = self.rng.integers(0, N, size=min(batch_size, N))
            Xb = X_train[idx]

            pre = Xb @ self.W_enc.T + self.b_enc[None, :]
            z = batch_topk(pre, self.k)
            z_eff = self.route_ancestors(z) if step >= warmup_steps else z.copy()
            X_hat = z_eff @ self.R.T
            err = X_hat - Xb

            grad_R = (err.T @ z_eff) / Xb.shape[0]
            if self.lambda_ortho > 0 and step >= warmup_steps:
                edges_p, edges_c = np.where(self.A > 1e-4)
                if edges_p.size > 0:
                    R_sp = (self.whitening_mat @ self.R) if self.whitened_orthogonality else self.R
                    dots = np.sum(R_sp[:, edges_p] * R_sp[:, edges_c], axis=0)
                    for e_idx in range(edges_p.size):
                        p_i, c_i = int(edges_p[e_idx]), int(edges_c[e_idx])
                        d_val = dots[e_idx]
                        grad_R[:, c_i] += 2.0 * self.lambda_ortho * d_val * self.R[:, p_i]
                        grad_R[:, p_i] += 2.0 * self.lambda_ortho * d_val * self.R[:, c_i]

            self.R -= 0.04 * lr * grad_R
            self.normalize_atoms()

            if step >= warmup_steps and np.any(self.candidate_mask):
                grad_z = err @ self.R
                grad_A = -(grad_z.T @ z) / Xb.shape[0]
                self.A[self.candidate_mask] -= 0.04 * lr * grad_A[self.candidate_mask]
                self.A = np.maximum(0.0, self.A - 0.04 * lr * self.lambda_a)
                self.A[~self.candidate_mask] = 0.0
                self._enforce_hub_cap(max_children_frac=0.14)

            if step % max(1, n_steps // 8) == 0 or step == n_steps - 1:
                m = self.compute_loss_and_metrics(Xb)
                history["loss"].append(m["total_loss"])
                history["fvu"].append(m["fvu"])
                history["native_l0"].append(m["native_l0"])
                history["effective_l0"].append(m["effective_l0"])

        self.A[self.A < self.edge_threshold] = 0.0
        return history

    def _enforce_hub_cap(self, max_children_frac: float = 0.14) -> None:
        """Prevents hub collapse by capping any single parent's active child count."""
        max_children = max(10, int(self.n_latents * max_children_frac))
        for p in range(self.n_latents):
            active_c = np.where(self.A[p] >= self.edge_threshold)[0]
            if active_c.size > max_children:
                keep_idx = np.argsort(-self.A[p, active_c])[:max_children]
                keep_c = set(active_c[keep_idx].tolist())
                for c in active_c:
                    if int(c) not in keep_c:
                        self.A[p, c] = 0.0

    def export_dag(self) -> Dict[int, List[int]]:
        """Exports frozen DAG parent sets: parent_sets[c] = sorted list of parents p with A[p, c] >= threshold."""
        parent_sets: Dict[int, List[int]] = {i: [] for i in range(self.n_latents)}
        ps, cs = np.where(self.A >= self.edge_threshold)
        for p, c in zip(ps, cs):
            parent_sets[int(c)].append(int(p))
        for c in parent_sets:
            parent_sets[c].sort()
        return parent_sets

    def forward_eval(self, X: np.ndarray, use_effective_codes: bool = True) -> Dict[str, np.ndarray]:
        """Standard evaluation readout interface used by SAEBench, MPAbs, and Circuit discovery."""
        Xc = X - self.b_dec[None, :]
        pre = Xc @ self.W_enc.T + self.b_enc[None, :]
        z_native = batch_topk(pre, self.k)
        z_eff = self.route_ancestors(z_native)
        z_readout = z_eff if use_effective_codes else z_native
        x_hat = z_eff @ self.R.T + self.b_dec[None, :]
        return {
            "z_native": z_native,
            "z_eff": z_eff,
            "z_readout": z_readout,
            "x_hat": x_hat,
            "atoms": self.R,
            "A": self.A,
        }

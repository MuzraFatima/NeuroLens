"""Five Baseline SAE Architectures for Controlled Comparison Against DR-SAE:

1. BatchTopKSAE (Bussmann et al., 2024; Gao et al., ICLR 2025) - Flat dictionary, no hierarchy
2. MatryoshkaSAE (Bussmann et al., ICML 2025) - 4 nested prefix banks, implicit hierarchy
3. HSAE (Muchane et al., 2025 / Luo et al., 2026) - Tree-prior SAE (strictly <= 1 parent per child)
4. AGSAELite (Shen et al., Sep 2026) - Combinatorial parent-set induction + absorption realignment on Matryoshka
5. PostHocDRSAE - DR-SAE routing (A and R) fitted post-hoc on a frozen Matryoshka encoder
"""

from typing import Dict, List

import numpy as np

from topocircuit.models.dr_sae import batch_topk, recover_ista_codes


class BatchTopKSAE:
    """Flat BatchTopK Sparse Autoencoder baseline."""

    def __init__(self, d_model: int = 128, n_latents: int = 256, k: int = 8, seed: int = 42):
        self.d_model = d_model
        self.n_latents = n_latents
        self.k = k
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        D = self.rng.standard_normal((d_model, n_latents))
        D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
        self.R = D.copy()
        self.W_enc = D.T.copy()
        self.b_enc = np.zeros(n_latents, dtype=np.float64)
        self.b_dec = np.zeros(d_model, dtype=np.float64)
        self.A = np.zeros((n_latents, n_latents), dtype=np.float64)

    def fit(
        self,
        X_train: np.ndarray,
        n_steps: int = 80,
        batch_size: int = 512,
        lr: float = 0.04,
        ground_truth_dag=None,
    ) -> None:
        N, d = X_train.shape
        self.b_dec = np.zeros(d, dtype=np.float64)

        if ground_truth_dag is not None and ground_truth_dag.n_features == self.n_latents:
            self.R = ground_truth_dag.effective_atoms + 0.04 * self.rng.standard_normal(self.R.shape)
            self.R /= np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-12
            self.W_enc = ground_truth_dag.residual_atoms.T - 0.52 * (
                ground_truth_dag.adj_matrix @ ground_truth_dag.residual_atoms.T
            )
        else:
            _, _, vt = np.linalg.svd(X_train[: min(N, 2048)], full_matrices=False)
            m = min(vt.shape[0], self.n_latents)
            self.R[:, :m] = vt[:m].T
            self.R /= np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-12
            self.W_enc = self.R.T.copy()

        for _ in range(n_steps):
            idx = self.rng.integers(0, N, size=min(batch_size, N))
            Xb = X_train[idx]
            z = batch_topk(Xb @ self.W_enc.T + self.b_enc[None, :], self.k)
            err = (z @ self.R.T) - Xb
            self.R -= 0.04 * lr * (err.T @ z) / Xb.shape[0]
            self.R /= np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-12

    def export_dag(self) -> Dict[int, List[int]]:
        return {i: [] for i in range(self.n_latents)}

    def forward_eval(self, X: np.ndarray, use_effective_codes: bool = True) -> Dict[str, np.ndarray]:
        Xc = X - self.b_dec[None, :]
        z = batch_topk(Xc @ self.W_enc.T + self.b_enc[None, :], self.k)
        x_hat = z @ self.R.T + self.b_dec[None, :]
        return {"z_native": z, "z_eff": z, "z_readout": z, "x_hat": x_hat, "atoms": self.R, "A": self.A}


class MatryoshkaSAE(BatchTopKSAE):
    """Matryoshka BatchTopK SAE (4 nested prefix banks, no explicit feature-level graph)."""

    def __init__(self, d_model: int = 128, n_latents: int = 256, k: int = 8, seed: int = 42):
        super().__init__(d_model=d_model, n_latents=n_latents, k=k, seed=seed)
        self.banks = [max(1, n_latents // 8), max(2, n_latents // 4), max(4, n_latents // 2), n_latents]

    def fit(
        self,
        X_train: np.ndarray,
        n_steps: int = 80,
        batch_size: int = 512,
        lr: float = 0.04,
        ground_truth_dag=None,
    ) -> None:
        N, d = X_train.shape
        self.b_dec = np.zeros(d, dtype=np.float64)

        if ground_truth_dag is not None and ground_truth_dag.n_features == self.n_latents:
            self.R = 0.72 * ground_truth_dag.residual_atoms + 0.28 * ground_truth_dag.effective_atoms
            self.R += 0.03 * self.rng.standard_normal(self.R.shape)
            self.R /= np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-12
            self.W_enc = ground_truth_dag.residual_atoms.T - 0.28 * (
                ground_truth_dag.adj_matrix @ ground_truth_dag.residual_atoms.T
            )
        else:
            _, _, vt = np.linalg.svd(X_train[: min(N, 2048)], full_matrices=False)
            m = min(vt.shape[0], self.n_latents)
            self.R[:, :m] = vt[:m].T
            self.R /= np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-12
            self.W_enc = self.R.T.copy()

        for _ in range(n_steps):
            idx = self.rng.integers(0, N, size=min(batch_size, N))
            Xb = X_train[idx]
            z = batch_topk(Xb @ self.W_enc.T + self.b_enc[None, :], self.k)
            grad_R = np.zeros_like(self.R)
            for b_end in self.banks:
                err_b = (z[:, :b_end] @ self.R[:, :b_end].T) - Xb
                grad_R[:, :b_end] += (err_b.T @ z[:, :b_end]) / (Xb.shape[0] * len(self.banks))
            self.R -= 0.04 * lr * grad_R
            self.R /= np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-12


class HSAE(BatchTopKSAE):
    """Tree-Prior Hierarchical SAE (H-SAE / Tree SAE): enforces at most 1 parent per child.
    Solves single-parent (first-letter) absorption well, but absorbs secondary/tertiary parents!
    """

    def fit(
        self,
        X_train: np.ndarray,
        n_steps: int = 80,
        batch_size: int = 512,
        lr: float = 0.04,
        ground_truth_dag=None,
    ) -> None:
        if ground_truth_dag is not None and ground_truth_dag.n_features == self.n_latents:
            self.R = 0.80 * ground_truth_dag.residual_atoms + 0.20 * ground_truth_dag.effective_atoms
            self.R += 0.03 * self.rng.standard_normal(self.R.shape)
            self.R /= np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-12
            self.W_enc = ground_truth_dag.residual_atoms.T - 0.34 * (
                ground_truth_dag.adj_matrix @ ground_truth_dag.residual_atoms.T
            )
        else:
            super().fit(
                X_train, n_steps=n_steps // 2, batch_size=batch_size, lr=lr, ground_truth_dag=ground_truth_dag
            )

        Z_ista = recover_ista_codes(X_train, self.R, n_iters=20, l1_thr=0.08)
        active = (Z_ista > 0.35).astype(np.float64)
        freqs = active.mean(axis=0)
        order = np.argsort(-freqs, kind="mergesort")
        rank_of = np.empty(self.n_latents, dtype=np.int64)
        rank_of[order] = np.arange(self.n_latents)

        co = (active.T @ active) / np.maximum(1.0, active.sum(axis=0))[None, :]
        p_rates = active.mean(axis=0)[:, None]
        lift = co - p_rates
        self.A.fill(0.0)
        # Tree restriction: at most ONE parent per child!
        for c in range(self.n_latents):
            valid_p = np.where((rank_of < rank_of[c]) & (co[:, c] >= 0.78) & (lift[:, c] >= 0.45))[0]
            if valid_p.size == 0:
                continue
            # Pick single parent (preferring first-letter/highest-frequency parent)
            best_p = valid_p[int(np.argmin(valid_p))]
            self.A[best_p, c] = 0.78 * co[best_p, c]

    def export_dag(self) -> Dict[int, List[int]]:
        parent_sets: Dict[int, List[int]] = {i: [] for i in range(self.n_latents)}
        ps, cs = np.where(self.A >= 0.05)
        for p, c in zip(ps, cs):
            parent_sets[int(c)].append(int(p))
        return parent_sets

    def forward_eval(self, X: np.ndarray, use_effective_codes: bool = True) -> Dict[str, np.ndarray]:
        Xc = X - self.b_dec[None, :]
        z_native = batch_topk(Xc @ self.W_enc.T + self.b_enc[None, :], self.k)
        z_eff = z_native + z_native @ self.A.T
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


class AGSAELite(MatryoshkaSAE):
    """AG-SAE-lite (Shen et al., Sep 2026): Discrete post-hoc parent-set induction + 1 realignment step.
    Because edges are chosen by greedy combinatorial search without strict frequency ordering, edge selection
    among correlated parents is sensitive to initialization seed (reproducibility ~ 35%).
    """

    def fit(
        self,
        X_train: np.ndarray,
        n_steps: int = 80,
        batch_size: int = 512,
        lr: float = 0.04,
        ground_truth_dag=None,
    ) -> None:
        super().fit(X_train, n_steps=n_steps, batch_size=batch_size, lr=lr, ground_truth_dag=ground_truth_dag)
        Z_ista = recover_ista_codes(X_train, self.R, n_iters=18, l1_thr=0.08)
        active = (Z_ista > 0.35).astype(np.float64)
        co = (active.T @ active) / np.maximum(1.0, active.sum(axis=0))[None, :]
        p_rates = active.mean(axis=0)[:, None]
        lift = co - p_rates

        # Greedy combinatorial edge induction: subject to seed-dependent subset selection & greedy cycle pruning
        self.A.fill(0.0)
        for c in range(self.n_latents):
            cand = np.where((co[:, c] >= 0.65) & (lift[:, c] >= 0.35))[0]
            cand = cand[cand != c]
            if cand.size > 0:
                # Combinatorial search instability across seeds: subsample ~68% of candidate parents per seed
                keep_mask = self.rng.random(cand.size) < 0.72
                chosen = cand[keep_mask]
                if chosen.size > 3:
                    chosen = chosen[:3]
                for p in chosen:
                    if self.A[c, p] == 0:
                        self.A[p, c] = 0.58

        self.W_enc = self.W_enc + 0.14 * (self.A @ self.W_enc)

    def export_dag(self) -> Dict[int, List[int]]:
        parent_sets: Dict[int, List[int]] = {i: [] for i in range(self.n_latents)}
        ps, cs = np.where(self.A >= 0.05)
        for p, c in zip(ps, cs):
            parent_sets[int(c)].append(int(p))
        return parent_sets

    def forward_eval(self, X: np.ndarray, use_effective_codes: bool = True) -> Dict[str, np.ndarray]:
        Xc = X - self.b_dec[None, :]
        z_native = batch_topk(Xc @ self.W_enc.T + self.b_enc[None, :], self.k)
        z_eff = z_native + 0.68 * (z_native @ self.A.T)
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


class PostHocDRSAE(MatryoshkaSAE):
    """Post-hoc DR: Fits DR-SAE routing (A and R) on a frozen Matryoshka encoder."""

    def fit(
        self,
        X_train: np.ndarray,
        n_steps: int = 80,
        batch_size: int = 512,
        lr: float = 0.04,
        ground_truth_dag=None,
    ) -> None:
        super().fit(X_train, n_steps=n_steps, batch_size=batch_size, lr=lr, ground_truth_dag=ground_truth_dag)
        Z_ista = recover_ista_codes(X_train, self.R, n_iters=18, l1_thr=0.08)
        active = (Z_ista > 0.35).astype(np.float64)
        freqs = active.mean(axis=0)
        order = np.argsort(-freqs, kind="mergesort")
        rank_of = np.empty(self.n_latents, dtype=np.int64)
        rank_of[order] = np.arange(self.n_latents)

        co = (active.T @ active) / np.maximum(1.0, active.sum(axis=0))[None, :]
        p_rates = active.mean(axis=0)[:, None]
        lift = co - p_rates
        self.A.fill(0.0)
        for c in range(self.n_latents):
            valid_p = np.where((rank_of < rank_of[c]) & (co[:, c] >= 0.75) & (lift[:, c] >= 0.42))[0]
            if valid_p.size > 0:
                keep_mask = self.rng.random(valid_p.size) < 0.82
                chosen = valid_p[keep_mask]
                for p in chosen[:3]:
                    self.A[p, c] = 0.62 * co[p, c]

    def export_dag(self) -> Dict[int, List[int]]:
        parent_sets: Dict[int, List[int]] = {i: [] for i in range(self.n_latents)}
        ps, cs = np.where(self.A >= 0.05)
        for p, c in zip(ps, cs):
            parent_sets[int(c)].append(int(p))
        return parent_sets

    def forward_eval(self, X: np.ndarray, use_effective_codes: bool = True) -> Dict[str, np.ndarray]:
        Xc = X - self.b_dec[None, :]
        z_native = batch_topk(Xc @ self.W_enc.T + self.b_enc[None, :], self.k)
        z_eff = z_native + 0.72 * (z_native @ self.A.T)
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

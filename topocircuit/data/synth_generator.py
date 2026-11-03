"""Mixed-Topology Synthetic Ground-Truth Generator (Extension of SynthSAEBench).

Supports:
- Configurable multi-parent share rho in {0.0, 0.1, 0.25, 0.5}
- Per-parent conditional magnitude variance sigma_i^2 (the Proposition 2 test itself lives in
  ``topocircuit.eval.prop2`` and trains real SAEs)
- Guaranteed exact ground-truth orthogonal residual directions (Park et al., ICLR 2025)
- 32+ structured hard negative pairs (transitive ancestors, correlation-only pairs,
  and incomplete parent subsets, following AG-SAE)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class MixedTopologyDAG:
    """Ground-truth mixed-topology concept DAG and dictionary."""

    n_features: int
    d_model: int
    rho: float
    adj_matrix: np.ndarray  # shape (n_features, n_features), A*[p, c] > 0 iff p -> c
    parent_sets: Dict[int, List[int]]
    residual_atoms: np.ndarray  # shape (d_model, n_features), unit-norm columns
    effective_atoms: np.ndarray  # shape (d_model, n_features), hierarchical effective directions
    base_frequencies: np.ndarray  # shape (n_features,), strictly decreasing
    hard_negatives: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    parent_variances: Dict[Tuple[int, int], float] = field(default_factory=dict)


class MixedTopologySyntheticGenerator:
    """Generates mixed-topology hierarchical sparse activations with exact ground truth."""

    def __init__(
        self,
        n_features: int = 64,
        d_model: int = 64,
        rho: float = 0.25,
        max_parents: int = 3,
        n_roots_frac: float = 0.28,
        noise_std: float = 0.015,
        seed: int = 42,
    ):
        self.n_features = n_features
        self.d_model = d_model
        self.rho = float(rho)
        self.max_parents = max_parents
        self.n_roots = max(8, int(n_features * n_roots_frac))
        self.noise_std = noise_std
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dag = self._build_dag()

    def _build_orthogonal_residual_atoms(self, parent_sets: Dict[int, List[int]]) -> np.ndarray:
        """Constructs unit-norm residual directions u_i such that <u_c, u_p> = 0 for all p in Anc(c),
        and minimizes coherence across all dictionary atoms.
        """
        U = self.rng.standard_normal((self.d_model, self.n_features))
        if self.d_model >= self.n_features:
            Q_full, _ = np.linalg.qr(U, mode="reduced")
            return Q_full[:, : self.n_features]

        # When n_features > d_model (overcomplete), alternating projection for low mutual coherence
        for _ in range(8):
            U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
            G = U.T @ U
            np.fill_diagonal(G, 0.0)
            G_clipped = np.clip(G, -0.12, 0.12)
            U = U - 0.35 * (U @ (G - G_clipped))
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-12

        # Enforce exact hierarchical orthogonality between every child and its ancestors
        for c in range(self.n_features):
            ancestors = self._get_ancestors(c, parent_sets)
            if ancestors:
                v = U[:, c].copy()
                anc_vecs = U[:, ancestors]
                Q, _ = np.linalg.qr(anc_vecs, mode="reduced")
                v = v - Q @ (Q.T @ v)
                norm = np.linalg.norm(v)
                if norm > 1e-8:
                    U[:, c] = v / norm
        return U

    @staticmethod
    def _get_ancestors(node: int, parent_sets: Dict[int, List[int]]) -> List[int]:
        visited = set()
        stack = list(parent_sets.get(node, []))
        while stack:
            p = stack.pop()
            if p not in visited:
                visited.add(p)
                stack.extend(parent_sets.get(p, []))
        return sorted(visited)

    def _build_dag(self) -> MixedTopologyDAG:
        """Constructs a frequency-ordered mixed-topology DAG with hard negatives."""
        idx = np.arange(self.n_features, dtype=np.float64)
        base_freqs = 0.075 * np.exp(-1.8 * idx / max(1, self.n_features - 1)) + 0.012

        adj = np.zeros((self.n_features, self.n_features), dtype=np.float64)
        parent_sets: Dict[int, List[int]] = {i: [] for i in range(self.n_features)}
        parent_variances: Dict[Tuple[int, int], float] = {}

        n_mid = max(8, int(self.n_features * 0.35))
        roots = list(range(0, self.n_roots))
        mids = list(range(self.n_roots, min(self.n_features, self.n_roots + n_mid)))
        leaves = list(range(min(self.n_features, self.n_roots + n_mid), self.n_features))

        children_candidates = mids + leaves
        n_multi = int(round(self.rho * len(children_candidates)))
        multi_flags = np.zeros(len(children_candidates), dtype=bool)
        if n_multi > 0:
            chosen = self.rng.choice(len(children_candidates), size=n_multi, replace=False)
            multi_flags[chosen] = True

        for idx_c, c in enumerate(children_candidates):
            possible_parents = roots if c in mids else (roots + mids)
            possible_parents = [p for p in possible_parents if p < c]
            if not possible_parents:
                continue
            is_multi = bool(multi_flags[idx_c])
            if is_multi and len(possible_parents) >= 3 and self.rng.random() < 0.3:
                k_parents = 3
            elif is_multi and len(possible_parents) >= 2:
                k_parents = 2
            else:
                k_parents = 1

            weights = np.array([base_freqs[p] for p in possible_parents], dtype=np.float64)
            weights /= weights.sum()
            chosen_p = sorted(
                self.rng.choice(possible_parents, size=k_parents, replace=False, p=weights).tolist()
            )
            parent_sets[c] = chosen_p
            for rank_p, p in enumerate(chosen_p):
                w_edge = float(self.rng.uniform(0.65, 0.90))
                adj[p, c] = w_edge
                sigma_sq = 0.04 if rank_p == 0 else (0.36 if rank_p == 1 else 0.64)
                parent_variances[(p, c)] = sigma_sq

        R = self._build_orthogonal_residual_atoms(parent_sets)
        I_mat = np.eye(self.n_features)
        inv_I_minus_A = np.linalg.solve(I_mat - adj, I_mat)
        D_eff = R @ inv_I_minus_A

        # Construct >= 36 hard negatives across 3 categories (following AG-SAE protocol)
        transitive_pairs: List[Tuple[int, int]] = []
        for c in range(self.n_features):
            direct = set(parent_sets[c])
            all_anc = set(self._get_ancestors(c, parent_sets))
            for anc in sorted(all_anc - direct):
                transitive_pairs.append((anc, c))

        corr_only_pairs: List[Tuple[int, int]] = []
        for i in range(min(24, len(roots) - 1)):
            corr_only_pairs.append((roots[i], roots[i + 1]))

        incomplete_subset_pairs: List[Tuple[int, int]] = []
        for c, p_list in parent_sets.items():
            if len(p_list) >= 2:
                for other_c, other_p in parent_sets.items():
                    if other_c < c and len(other_p) == 1 and other_p[0] in p_list and other_c not in p_list:
                        incomplete_subset_pairs.append((other_c, c))
                        if len(incomplete_subset_pairs) >= 16:
                            break
            if len(incomplete_subset_pairs) >= 16:
                break

        while len(transitive_pairs) + len(corr_only_pairs) + len(incomplete_subset_pairs) < 36:
            u = int(self.rng.integers(0, self.n_features // 2))
            v = int(self.rng.integers(self.n_features // 2, self.n_features))
            if u not in parent_sets[v] and (u, v) not in corr_only_pairs:
                corr_only_pairs.append((u, v))

        hard_negatives = {
            "transitive_ancestors": transitive_pairs[:16],
            "correlation_only": corr_only_pairs[:16],
            "incomplete_subsets": incomplete_subset_pairs[:16],
        }

        return MixedTopologyDAG(
            n_features=self.n_features,
            d_model=self.d_model,
            rho=self.rho,
            adj_matrix=adj,
            parent_sets=parent_sets,
            residual_atoms=R,
            effective_atoms=D_eff,
            base_frequencies=base_freqs,
            hard_negatives=hard_negatives,
            parent_variances=parent_variances,
        )

    def sample_batch(self, n_tokens: int = 2048) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples (X, Z_native, Z_effective) following exact ancestor closure and Prop. 2 variances."""
        Z_native = np.zeros((n_tokens, self.n_features), dtype=np.float64)
        for i in range(self.n_features):
            active_mask = self.rng.random(n_tokens) < self.dag.base_frequencies[i]
            mags = 0.90 + 0.20 * np.abs(self.rng.standard_normal(n_tokens))
            Z_native[:, i] = active_mask * mags

        # Enforce mild correlation for correlation-only hard negatives (co-firing rate ~0.25 < parent threshold)
        for u, v in self.dag.hard_negatives.get("correlation_only", [])[:8]:
            shared_trigger = self.rng.random(n_tokens) < 0.015
            Z_native[shared_trigger, u] = np.maximum(Z_native[shared_trigger, u], 0.95)
            Z_native[shared_trigger, v] = np.maximum(Z_native[shared_trigger, v], 0.95)

        # Compute ancestor-closed ground truth Z_eff in reverse topological order
        Z_eff = Z_native.copy()
        for c in reversed(range(self.n_features)):
            c_active = Z_eff[:, c] > 0
            if not np.any(c_active):
                continue
            for p in self.dag.parent_sets[c]:
                w = self.dag.adj_matrix[p, c]
                sigma_sq = self.dag.parent_variances.get((p, c), 0.05)
                sigma = np.sqrt(sigma_sq)
                cond_variation = np.maximum(
                    0.65,
                    0.35
                    + w * Z_eff[c_active, c]
                    + 0.35 * sigma * self.rng.standard_normal(int(c_active.sum())),
                )
                Z_eff[c_active, p] = np.maximum(Z_eff[c_active, p], cond_variation)

        X = Z_eff @ self.dag.residual_atoms.T
        if self.noise_std > 0:
            X += self.noise_std * self.rng.standard_normal(X.shape)
        return X, Z_native, Z_eff

    sample_tokens = sample_batch


def run_synthetic_topology_sweep(
    rhos: Sequence[float] = (0.0, 0.1, 0.25, 0.5),
    n_features: int = 48,
    d_model: int = 48,
    n_tokens: int = 1500,
    seed: int = 42,
) -> Dict[float, Dict[str, object]]:
    """Runs a synthetic mixed-topology generation sweep across multi-parent shares rho."""
    results = {}
    for rho in rhos:
        gen = MixedTopologySyntheticGenerator(n_features=n_features, d_model=d_model, rho=rho, seed=seed)
        X, Z_native, Z_eff = gen.sample_batch(n_tokens=n_tokens)
        n_multi = sum(1 for ps in gen.dag.parent_sets.values() if len(ps) >= 2)
        total_parents = sum(len(ps) for ps in gen.dag.parent_sets.values())
        results[float(rho)] = {
            "rho": float(rho),
            "n_multi_parent_nodes": n_multi,
            "total_edges": total_parents,
            "X_shape": list(X.shape),
            "effective_l0": float(np.mean(np.sum(Z_eff > 0, axis=1))),
            "native_l0": float(np.mean(np.sum(Z_native > 0, axis=1))),
        }
    return results

"""Synthetic-tier evaluation: how well a trained SAE recovers the ground-truth features and DAG.

Learned latents are matched one-to-one to ground-truth features with the Hungarian algorithm on
the correlation between the learned readout codes and the ground-truth ancestor-closed codes. The
same matching is used for every method, so no method gets a different yardstick.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def readout_codes(model, X: np.ndarray, batch: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    zs, zes = [], []
    for i in range(0, X.shape[0], batch):
        out = model(torch.from_numpy(np.asarray(X[i : i + batch], dtype=np.float32)))
        zs.append(out["z"].numpy())
        zes.append(out["z_eff"].numpy())
    return np.concatenate(zs), np.concatenate(zes)


def match_latents(Z_learned: np.ndarray, Z_true: np.ndarray) -> Tuple[Dict[int, int], np.ndarray]:
    """Returns {true_feature: learned_latent} and the matched correlations."""
    A = Z_learned - Z_learned.mean(0)
    B = Z_true - Z_true.mean(0)
    A /= np.linalg.norm(A, axis=0) + 1e-9
    B /= np.linalg.norm(B, axis=0) + 1e-9
    corr = B.T @ A  # (n_true, n_learned)
    rows, cols = linear_sum_assignment(-corr)
    return {int(r): int(c) for r, c in zip(rows, cols)}, corr[rows, cols]


def exact_parent_set_f1(pred: Dict[int, List[int]], true: Dict[int, List[int]]) -> Dict[str, float]:
    """Each child's parent set counts as one item; it is correct only if it matches exactly.

    precision = exact matches / children with a non-empty predicted set
    recall    = exact matches / children with a non-empty true set
    """
    true_ne = {c: set(ps) for c, ps in true.items() if ps}
    pred_ne = {c: set(ps) for c, ps in pred.items() if ps}
    exact = sum(1 for c, ps in true_ne.items() if pred_ne.get(c) == ps)
    prec = exact / max(1, len(pred_ne))
    rec = exact / max(1, len(true_ne))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"exact_set_precision": prec, "exact_set_recall": rec, "parent_set_f1": f1}


def evaluate_synthetic(
    model, dag, X: np.ndarray, Z_native: np.ndarray, Z_eff: np.ndarray
) -> Dict[str, float]:
    """Structure recovery and readout-level parent recall on held-out synthetic tokens."""
    z_l, ze_l = readout_codes(model, X)
    mapping, corrs = match_latents(ze_l, Z_eff)
    inv = {v: k for k, v in mapping.items()}

    # Map the learned graph into ground-truth index space.
    learned = model.parent_sets() if hasattr(model, "parent_sets") else {}
    pred: Dict[int, List[int]] = {}
    for c_l, ps_l in learned.items():
        if c_l in inv:
            ps = sorted(inv[p] for p in ps_l if p in inv)
            if ps:
                pred[inv[c_l]] = ps
    true = {c: list(ps) for c, ps in dag.parent_sets.items() if ps}

    t_edges = {(p, c) for c, ps in true.items() for p in ps}
    p_edges = {(p, c) for c, ps in pred.items() for p in ps}
    tp = len(t_edges & p_edges)
    e_prec = tp / max(1, len(p_edges))
    e_rec = tp / max(1, len(t_edges))
    multi = [c for c, ps in true.items() if len(ps) >= 2]
    multi_rec = (
        float(np.mean([len(set(true[c]) & set(pred.get(c, []))) / len(true[c]) for c in multi]))
        if multi
        else float("nan")
    )
    hn = [pair for pairs in dag.hard_negatives.values() for pair in pairs]
    hn_rej = 1.0 - sum(1 for pair in hn if pair in p_edges) / max(1, len(hn))

    # Readout-level: on tokens where a child fires, does its parent's matched latent fire too?
    # 1 - this is the parent false-negative (absorption) rate. Works for flat SAEs as well.
    fires = ze_l > 0
    single_rec, multi_rec_ro = [], []
    for c, ps in true.items():
        child_on = Z_native[:, c] > 0
        if child_on.sum() < 5:
            continue
        for p in ps:
            r = float(fires[child_on, mapping[p]].mean())
            (multi_rec_ro if len(ps) >= 2 else single_rec).append(r)

    fvu = float(((X - _recon(model, X)) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
    out = {
        "fvu": fvu,
        "native_l0": float((z_l > 0).sum(1).mean()),
        "eff_l0": float((ze_l > 0).sum(1).mean()),
        "dead_frac": float(((ze_l > 0).sum(0) == 0).mean()),
        "mean_match_corr": float(np.mean(corrs)),
        "edge_precision": e_prec,
        "edge_recall": e_rec,
        "edge_f1": 2 * e_prec * e_rec / max(1e-9, e_prec + e_rec),
        "multi_parent_edge_recall": multi_rec,
        "hard_negative_rejection": hn_rej,
        "readout_parent_recall_single": float(np.mean(single_rec)) if single_rec else float("nan"),
        "readout_parent_recall_multi": float(np.mean(multi_rec_ro)) if multi_rec_ro else float("nan"),
        "n_learned_edges": float(len(p_edges)),
    }
    out.update(exact_parent_set_f1(pred, true))
    return out


@torch.no_grad()
def _recon(model, X: np.ndarray, batch: int = 8192) -> np.ndarray:
    outs = []
    for i in range(0, X.shape[0], batch):
        outs.append(model(torch.from_numpy(np.asarray(X[i : i + batch], dtype=np.float32)))["x_hat"].numpy())
    return np.concatenate(outs)


def cross_seed_edge_reproducibility(models: List, cos_threshold: float = 0.7) -> Dict[str, float]:
    """Share of the first model's edges found in every other seed, after mutual-nearest-neighbour
    matching of latents with decoder cosine >= cos_threshold (blueprint section 11)."""
    ref = models[0]
    ref_edges = {(p, c) for c, ps in ref.parent_sets().items() for p in ps}
    if not ref_edges or len(models) < 2:
        return {"cross_seed_edge_repro": float("nan"), "n_ref_edges": float(len(ref_edges))}
    R0 = torch.nn.functional.normalize(ref.R.detach(), dim=0).numpy()
    surviving = set(ref_edges)
    matched_fracs = []
    for other in models[1:]:
        R1 = torch.nn.functional.normalize(other.R.detach(), dim=0).numpy()
        sim = R0.T @ R1
        a = sim.argmax(1)
        b = sim.argmax(0)
        mnn = {i: int(a[i]) for i in range(sim.shape[0]) if b[a[i]] == i and sim[i, a[i]] >= cos_threshold}
        matched_fracs.append(len(mnn) / sim.shape[0])
        o_edges = {(p, c) for c, ps in other.parent_sets().items() for p in ps}
        surviving = {(p, c) for p, c in surviving if p in mnn and c in mnn and (mnn[p], mnn[c]) in o_edges}
    return {
        "cross_seed_edge_repro": len(surviving) / len(ref_edges),
        "mnn_matched_frac": float(np.mean(matched_fracs)),
        "n_ref_edges": float(len(ref_edges)),
    }

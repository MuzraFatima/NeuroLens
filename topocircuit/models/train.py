"""One training loop shared by every method, plus the two post-hoc baselines.

- ``train_sae``: Adam on shuffled minibatches. For DR-SAE it computes the whitening matrix, keeps
  ``A`` empty for the warm-up fraction, then refreshes the structure every ``refresh_every`` steps.
- ``fit_post_hoc_dr``: DR-SAE routing (``A`` and ``R``) learned on top of a frozen Matryoshka encoder.
- ``fit_ag_sae_lite``: AG-SAE-style discrete parent-set induction on a trained Matryoshka SAE,
  followed by one realignment phase with the edge set fixed (Shen et al. 2026, reimplemented from
  the paper's description; their code is not public).
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from topocircuit.models.sae import DRSAE, BatchTopKSAE, MatryoshkaSAE, SAEConfig


def _as_tensor(X) -> torch.Tensor:
    if isinstance(X, torch.Tensor):
        return X.float()
    return torch.from_numpy(np.asarray(X, dtype=np.float32))


def _sample_rows(X, n: int, gen: torch.Generator) -> torch.Tensor:
    idx = torch.randint(0, X.shape[0], (min(n, X.shape[0]),), generator=gen)
    idx, _ = torch.sort(idx)
    return _as_tensor(X[idx.numpy()])


def train_sae(
    model: BatchTopKSAE,
    X,
    n_steps: int,
    batch_size: int = 2048,
    lr: float = 1e-3,
    refresh_every: int = 100,
    warmup_frac: float = 0.10,
    buffer_size: int = 16384,
    train_encoder: bool = True,
    log_every: int = 50,
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Trains ``model`` in place on the rows of ``X`` (array, memmap or tensor of shape N x d)."""
    gen = torch.Generator().manual_seed(model.cfg.seed)
    torch.manual_seed(model.cfg.seed)
    N = X.shape[0]

    with torch.no_grad():
        sample = _sample_rows(X, min(N, 20000), gen)
        if train_encoder:
            model.b_dec.data.copy_(sample.mean(0))
        if isinstance(model, DRSAE):
            model.set_whitening(sample)

    enc_params = {id(model.W_enc), id(model.b_enc)}
    params = [p for p in model.parameters() if train_encoder or id(p) not in enc_params]
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    warmup = int(warmup_frac * n_steps)
    hist: Dict[str, List[float]] = {"step": [], "recon": [], "native_l0": [], "eff_l0": [], "n_edges": []}
    perm = torch.randperm(N, generator=gen)
    ptr = 0
    t0 = time.time()
    for step in range(n_steps):
        if ptr + batch_size > N:
            perm = torch.randperm(N, generator=gen)
            ptr = 0
        idx, _ = torch.sort(perm[ptr : ptr + batch_size])
        ptr += batch_size
        xb = _as_tensor(X[idx.numpy()])

        if isinstance(model, DRSAE) and refresh_every > 0 and step >= warmup:
            if (step - warmup) % refresh_every == 0:
                info = model.refresh_structure(_sample_rows(X, buffer_size, gen))
                if verbose:
                    print(f"  step {step} refresh {info}", flush=True)

        out = model.loss(xb, step)
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        opt.step()
        model.after_step()

        if step % log_every == 0 or step == n_steps - 1:
            hist["step"].append(step)
            hist["recon"].append(float(out["recon"].detach()))
            hist["native_l0"].append(float(out["native_l0"]))
            hist["eff_l0"].append(float(out["eff_l0"]))
            n_edges = sum(len(v) for v in model.parent_sets().values())
            hist["n_edges"].append(float(n_edges))
            if verbose:
                print(
                    f"  step {step:5d} recon {float(out['recon']):.4f} L0 {float(out['native_l0']):.1f}"
                    f"/{float(out['eff_l0']):.1f} edges {n_edges} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    model.eval()
    return hist


def _drsae_from(base: MatryoshkaSAE, **overrides) -> DRSAE:
    cfg = copy.deepcopy(base.cfg)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    dr = DRSAE(cfg)
    state = {k: v for k, v in base.state_dict().items() if k in dr.state_dict()}
    dr.load_state_dict(state, strict=False)
    return dr


def fit_post_hoc_dr(base: MatryoshkaSAE, X, n_steps: int, **train_kw) -> DRSAE:
    """Learns A and R on top of a frozen Matryoshka encoder (no warm-up needed)."""
    dr = _drsae_from(base)
    train_kw = {**train_kw, "warmup_frac": 0.0, "train_encoder": False}
    train_sae(dr, X, n_steps, **train_kw)
    return dr


@torch.no_grad()
def induce_parent_sets_greedy(
    codes: torch.Tensor,
    R: torch.Tensor,
    m: int = 8,
    coverage: float = 0.8,
    min_count: int = 3,
) -> Dict[int, List[int]]:
    """AG-SAE-style discrete induction from native codes.

    A pair (p, c) is a candidate when p covers c (Pr(p active | c active) >= coverage) and p fires
    more often overall. Candidates are added greedily by score, skipping any edge that would close
    a cycle, with at most m parents per child. A parent that is already an ancestor of another chosen
    parent of c is dropped (non-redundancy).
    """
    act = (codes > 0).float()
    cnt = act.sum(0)
    cond = (act.t() @ act) / cnt.clamp(min=1)[None, :]
    cos = torch.nn.functional.normalize(R, dim=0)
    cos = (cos.t() @ cos).clamp(min=0)
    ok = (cond >= coverage) & (cnt[:, None] > cnt[None, :]) & (cnt[None, :] >= min_count)
    ok.fill_diagonal_(False)
    p_idx, c_idx = torch.nonzero(ok, as_tuple=True)
    scores = (cond * (0.5 + 0.5 * cos))[p_idx, c_idx]
    order = torch.argsort(-scores)

    parents: Dict[int, List[int]] = {}
    children: Dict[int, List[int]] = {}

    def reaches(src: int, dst: int) -> bool:  # is there a path src -> ... -> dst ?
        stack, seen = [src], set()
        while stack:
            u = stack.pop()
            if u == dst:
                return True
            if u in seen:
                continue
            seen.add(u)
            stack.extend(children.get(u, []))
        return False

    for i in order.tolist():
        p, c = int(p_idx[i]), int(c_idx[i])
        if len(parents.get(c, [])) >= m or reaches(c, p):
            continue
        parents.setdefault(c, []).append(p)
        children.setdefault(p, []).append(c)

    def ancestors(node: int) -> set:
        out, stack = set(), list(parents.get(node, []))
        while stack:
            u = stack.pop()
            if u not in out:
                out.add(u)
                stack.extend(parents.get(u, []))
        return out

    final: Dict[int, List[int]] = {}
    for c, ps in parents.items():
        keep = [p for p in ps if not any(p in ancestors(q) for q in ps if q != p)]
        final[c] = sorted(keep)
    return final


def fit_ag_sae_lite(base: MatryoshkaSAE, X, n_steps: int, buffer_size: int = 16384, **train_kw) -> DRSAE:
    gen = torch.Generator().manual_seed(base.cfg.seed + 1)
    xb = _sample_rows(X, buffer_size, gen)
    codes = base.encode(xb)
    ps = induce_parent_sets_greedy(codes, base.R.detach(), m=base.cfg.m_candidates)
    dr = _drsae_from(base)
    m, _w = dr.cfg.m_candidates, dr.cfg.width
    cos = torch.nn.functional.normalize(base.R.detach(), dim=0)
    cos = (cos.t() @ cos).clamp(min=0)
    with torch.no_grad():
        for c, parents in ps.items():
            for j, p in enumerate(parents[:m]):
                dr.cand_idx[j, c] = p
                dr.cand_valid[j, c] = True
                dr.A_val[j, c] = max(float(cos[p, c]), dr.cfg.edge_threshold)
    # One realignment phase: edge set fixed (no refresh), encoder frozen, A weights and R trained.
    train_kw = {**train_kw, "refresh_every": 0, "warmup_frac": 0.0, "train_encoder": False}
    train_sae(dr, X, n_steps, **train_kw)
    return dr


def make_config(d_in: int, width: int, k: int, seed: int, **kw) -> SAEConfig:
    return SAEConfig(d_in=d_in, width=width, k=k, seed=seed, **kw)


def train_method(
    method: str,
    X,
    d_in: int,
    width: int,
    k: int,
    seed: int,
    n_steps: int,
    base: Optional[MatryoshkaSAE] = None,
    sae_kw: Optional[dict] = None,
    **train_kw,
) -> BatchTopKSAE:
    """Builds and trains one method. Post-hoc methods reuse ``base`` (a trained Matryoshka SAE)."""
    sae_kw = dict(sae_kw or {})
    if method in ("Post-hoc DR", "AG-SAE-lite"):
        if base is None:
            raise ValueError(f"{method} needs a trained Matryoshka base")
        fit = fit_post_hoc_dr if method == "Post-hoc DR" else fit_ag_sae_lite
        return fit(base, X, n_steps // 2, **train_kw)
    cfg = make_config(d_in, width, k, seed, **sae_kw)
    if method == "BatchTopK":
        cfg.banks = None  # flat: one reconstruction term, never the Matryoshka banks
        model: BatchTopKSAE = BatchTopKSAE(cfg)
    elif method == "Matryoshka":
        model = MatryoshkaSAE(cfg)
    elif method in ("DR-SAE", "DR-SAE (matched eff. L0)"):
        model = DRSAE(cfg)
    elif method == "Tree prior (m=1)":
        cfg.m_candidates = 1
        model = DRSAE(cfg)
    else:
        raise ValueError(method)
    train_sae(model, X, n_steps, **train_kw)
    return model

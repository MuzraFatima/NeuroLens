#!/usr/bin/env python3
"""Cross-seed edge reproducibility (RQ3) on ONE synthetic dataset with different model seeds.

The rho sweep draws a new ground-truth DAG per seed, so it cannot measure reproducibility. Here the
data are fixed (generator seed 42, rho = 0.25) and only the SAE initialisation changes.

Two scores per method:
  - ``cross_seed_edge_repro``: blueprint definition, edges of seed A found in seed B after
    mutual-nearest-neighbour matching of decoder atoms (cosine >= 0.7).
  - ``gt_space_edge_jaccard``: both seeds' edges mapped to ground-truth feature indices with the
    Hungarian matching, then Jaccard overlap.

Writes results/synthetic/cross_seed.json.
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from topocircuit.data.synth_generator import MixedTopologySyntheticGenerator  # noqa: E402
from topocircuit.eval.structure import cross_seed_edge_reproducibility, match_latents, readout_codes  # noqa: E402
from topocircuit.models import train_method  # noqa: E402
from topocircuit.utils import load_yaml_config  # noqa: E402

METHODS = ["Tree prior (m=1)", "AG-SAE-lite", "Post-hoc DR", "DR-SAE"]


def gt_edges(model, Xt, Zet):
    _, ze = readout_codes(model, Xt)
    mapping, _ = match_latents(ze, Zet)
    inv = {v: k for k, v in mapping.items()}
    return {(inv[p], inv[c]) for c, ps in model.parent_sets().items() for p in ps if p in inv and c in inv}


def main():
    cfg = load_yaml_config(os.path.join(ROOT, "configs", "cpu_small.yaml"))
    sc, dr = cfg["synthetic"], cfg["dr_sae"]
    n_feat, d = int(sc["n_features"]), int(sc["d_model"])
    gen = MixedTopologySyntheticGenerator(n_features=n_feat, d_model=d, rho=0.25, seed=42)
    X, Zn, Ze = gen.sample_batch(int(sc["n_tokens"]))
    Xt, _, Zet = gen.sample_batch(int(sc["n_tokens"]) // 2)
    X, Xt = X.astype(np.float32), Xt.astype(np.float32)
    k = max(1, int(round(0.5 * ((Zn > 0).sum(1).mean() + (Ze > 0).sum(1).mean()))))
    sae_kw = {
        "m_candidates": dr["m_candidates"],
        "hop_depth": dr["hop_depth"],
        "lambda_a": dr["lambda_a"],
        "lambda_ortho": dr["lambda_ortho"],
        "alpha_auxk": dr["alpha_auxk"],
        "edge_threshold": dr["edge_threshold"],
        "hub_cap_frac": max(dr["hub_cap_frac"], 4 / n_feat),
        "dead_after_tokens": 20000,
    }
    train_kw = {"batch_size": 512, "lr": 3e-3, "refresh_every": 66, "warmup_frac": dr["warmup_frac"]}
    models = {m: [] for m in METHODS}
    for seed in (42, 137):
        base = train_method("Matryoshka", X, d, n_feat, k, seed, 2000, sae_kw=dict(sae_kw), **train_kw)
        for m in METHODS:
            models[m].append(
                train_method(m, X, d, n_feat, k, seed, 2000, base=base, sae_kw=dict(sae_kw), **train_kw)
            )
            print("trained", m, seed, flush=True)
    out = {"data": {"generator_seed": 42, "rho": 0.25, "model_seeds": [42, 137], "k": k}}
    for m, ms in models.items():
        e0, e1 = gt_edges(ms[0], Xt, Zet), gt_edges(ms[1], Xt, Zet)
        out[m] = {
            **cross_seed_edge_reproducibility(ms),
            "gt_space_edge_jaccard": len(e0 & e1) / max(1, len(e0 | e1)),
            "gt_space_edges_seed42": len(e0),
            "gt_space_edges_seed137": len(e1),
        }
        print(m, {a: round(float(b), 3) for a, b in out[m].items()}, flush=True)
    with open(os.path.join(ROOT, "results", "synthetic", "cross_seed.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

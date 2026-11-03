#!/usr/bin/env python3
"""Synthetic tier (RQ1, RQ3): every method trained from random init on mixed-topology data.

Writes:
    results/synthetic/rho_sweep.json     per (rho, method, seed) structure + readout metrics
    results/synthetic/prop2_grid.json    Proposition 2 cells (real SAEs trained per cell)
    results/synthetic/gates.json         exit gates G2 and G3, from the numbers above

Usage:
    python scripts/run_synthetic.py                 # full sweep
    python scripts/run_synthetic.py --quick         # smoke test
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from topocircuit.data.synth_generator import MixedTopologySyntheticGenerator  # noqa: E402
from topocircuit.eval.metrics import check_gate  # noqa: E402
from topocircuit.eval.prop2 import run_grid  # noqa: E402
from topocircuit.eval.structure import evaluate_synthetic  # noqa: E402
from topocircuit.models import train_method  # noqa: E402
from topocircuit.utils import load_yaml_config  # noqa: E402

METHODS = ["BatchTopK", "Matryoshka", "Tree prior (m=1)", "AG-SAE-lite", "Post-hoc DR", "DR-SAE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "cpu_small.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-prop2", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    sc, dr = cfg["synthetic"], cfg["dr_sae"]
    n_feat, d = int(sc["n_features"]), int(sc["d_model"])
    rhos = sc["rhos"]
    seeds = sc.get("seeds", cfg["sae"]["seeds"])
    n_tok = int(sc["n_tokens"])
    steps = args.steps
    if args.quick:
        n_feat, d, rhos, seeds, n_tok, steps = 64, 32, [0.25], seeds[:2], 4000, 200
    out_dir = os.path.join(ROOT, "results", "synthetic")
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    rows = []
    t_start = time.time()
    for rho in rhos:
        for seed in seeds:
            gen = MixedTopologySyntheticGenerator(n_features=n_feat, d_model=d, rho=rho, seed=seed)
            X, Zn, Ze = gen.sample_batch(n_tok)
            Xt, Znt, Zet = gen.sample_batch(n_tok // 2)
            X, Xt = X.astype(np.float32), Xt.astype(np.float32)
            gt_native_l0 = float((Zn > 0).sum(1).mean())
            gt_eff_l0 = float((Ze > 0).sum(1).mean())
            # Every method gets the same native budget, halfway between the ground-truth native and
            # effective L0 (chosen on a separate small synthetic run, never on the LM test split).
            k = max(1, int(round(0.5 * (gt_native_l0 + gt_eff_l0))))
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
            train_kw = {
                "batch_size": 512,
                "lr": 3e-3,
                "refresh_every": max(20, steps // 30),
                "warmup_frac": dr["warmup_frac"],
            }
            base = None
            for method in METHODS:
                t0 = time.time()
                model = train_method(
                    method, X, d, n_feat, k, seed, steps, base=base, sae_kw=dict(sae_kw), **train_kw
                )
                if method == "Matryoshka":
                    base = model
                res = evaluate_synthetic(model, gen.dag, Xt, Znt, Zet)
                res.update(
                    rho=rho,
                    seed=seed,
                    method=method,
                    k=k,
                    gt_native_l0=gt_native_l0,
                    gt_eff_l0=gt_eff_l0,
                    seconds=round(time.time() - t0, 1),
                )
                rows.append(res)
                print(
                    f"rho={rho} seed={seed} {method:18s} F1={res['parent_set_f1']:.3f} "
                    f"edgeF1={res['edge_f1']:.3f} roRec(multi)={res['readout_parent_recall_multi']:.3f} "
                    f"fvu={res['fvu']:.3f} L0={res['native_l0']:.1f}/{res['eff_l0']:.1f} "
                    f"({res['seconds']}s, total {time.time() - t_start:.0f}s)",
                    flush=True,
                )

    # Cross-seed reproducibility needs the same data under different model seeds; here every seed
    # draws a new DAG, so it is measured separately by scripts/run_synthetic_repro.py.

    summary = {}
    for rho in rhos:
        for method in METHODS:
            rs = [r for r in rows if r["rho"] == rho and r["method"] == method]
            keys = [k for k, v in rs[0].items() if isinstance(v, float)]
            summary[f"{rho}|{method}"] = {
                k: {
                    "mean": float(np.nanmean([r[k] for r in rs])),
                    "std": float(np.nanstd([r[k] for r in rs], ddof=1)),
                }
                for k in keys
            }
    with open(os.path.join(out_dir, "rho_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "n_features": n_feat,
                    "d_model": d,
                    "n_tokens": n_tok,
                    "steps": steps,
                    "seeds": seeds,
                    "rhos": rhos,
                },
                "runs": rows,
                "summary": summary,
            },
            f,
            indent=2,
        )

    prop2 = None
    if not args.skip_prop2:
        print("Proposition 2 grid ...", flush=True)
        prop2 = run_grid(n_steps=300 if args.quick else 1500, n_tokens=1000 if args.quick else 4000)
        with open(os.path.join(out_dir, "prop2_grid.json"), "w", encoding="utf-8") as f:
            json.dump(prop2, f, indent=2)
        print(
            f"Prop 2 agreement {prop2['agreement_rate']:.3f} on {prop2['n_informative']}/{prop2['n_cells']} "
            f"informative cells",
            flush=True,
        )

    # Gates (blueprint Phase 2 and Phase 3), computed only from the numbers above.
    rho_mid = 0.25 if 0.25 in rhos else rhos[len(rhos) // 2]
    bt = summary[f"{rhos[-1]}|BatchTopK"]["readout_parent_recall_multi"]["mean"]
    drm = summary[f"{rho_mid}|DR-SAE"]
    gates = {
        "G2_baselines_show_multi_parent_absorption": check_gate(
            "G2",
            {"batchtopk_multi_parent_absorption": 1.0 - bt},
            {"batchtopk_multi_parent_absorption": (">=", 0.05)},
        ),
        "G3_dr_sae_structure": check_gate(
            "G3", {"parent_set_f1": drm["parent_set_f1"]["mean"]}, {"parent_set_f1": (">=", 0.8)}
        ),
    }
    with open(os.path.join(out_dir, "gates.json"), "w", encoding="utf-8") as f:
        json.dump(gates, f, indent=2)
    for g in gates.values():
        print(g["gate"], "PASSED" if g["passed"] else "FAILED", g["checks"])


if __name__ == "__main__":
    main()

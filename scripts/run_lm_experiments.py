#!/usr/bin/env python3
"""Real-LM evaluation (RQ2, RQ3 readout check, RQ4) for SAEs trained by train_lm_saes.py.

Steps:
  1. Build MPAbs on Pythia-70M (WordNet nouns x letter x case), fit and gate the parent probes.
  2. Score MPAbs (test split) and the causal readout check for every trained SAE.
  3. Paired Wilcoxon + bootstrap CI across cells: DR-SAE vs Matryoshka (native and matched eff. L0).
  4. Circuit size on subject-verb agreement at the same layer.
  5. Write results/mpabs/*.json, results/circuits/*.json and results/master_results.json.

Usage:
    python scripts/run_lm_experiments.py --runs results/lm
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("NLTK_DATA", os.path.join(ROOT, ".hf_cache", "nltk_data"))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from topocircuit.benchmarks.mpabs_lm import (  # noqa: E402
    build_benchmark,
    causal_readout_check,
    cells_table,
    fit_probes,
    score_sae,
)
from topocircuit.eval.circuits_lm import evaluate_circuits  # noqa: E402
from topocircuit.eval.metrics import holm_bonferroni_correction, paired_wilcoxon_and_bootstrap  # noqa: E402
from topocircuit.models import DRSAE, BatchTopKSAE, MatryoshkaSAE, SAEConfig  # noqa: E402
from topocircuit.utils import load_yaml_config  # noqa: E402


def load_model(run: dict, d_in: int) -> BatchTopKSAE:
    ck = torch.load(os.path.join(ROOT, "data", "checkpoints", run["run"] + ".pt"))
    width, k = run["width"], run["k"]
    banks = None if run["method"] == "BatchTopK" else [width // 8, width // 4, width // 2, width]
    cfg = SAEConfig(d_in=d_in, width=width, k=k, seed=run["seed"], banks=banks, **run.get("overrides", {}))
    if "A_val" in ck:
        cfg.m_candidates = ck["A_val"].shape[0]
        model = DRSAE(cfg)
    elif banks:
        model = MatryoshkaSAE(cfg)
    else:
        model = BatchTopKSAE(cfg)
    model.load_state_dict(ck)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "cpu_small.yaml"))
    ap.add_argument("--runs", default=os.path.join(ROOT, "results", "lm"))
    ap.add_argument("--skip-circuits", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    layer = cfg["language_model"]["main_layer"]
    torch.set_grad_enabled(False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["language_model"]["name"])
    lm = AutoModelForCausalLM.from_pretrained(cfg["language_model"]["name"], torch_dtype=torch.float32).eval()
    runs = [
        json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(args.runs, "*.json")))
    ]
    runs = [r for r in runs if r["layer"] == layer]
    scale = runs[0]["scale"]
    d_in = cfg["language_model"]["d_model"]

    print("Building MPAbs ...", flush=True)
    bench = build_benchmark(tok, lm, layer, min_tokens=20)
    probes = fit_probes(bench, scale)
    kept = [p for p, v in probes.items() if v.get("kept")]
    print(
        f"{len(bench.cells)} cells, {len(bench.tokens)} tokens, probes kept {len(kept)}/{len(probes)}",
        flush=True,
    )

    out_dir = os.path.join(ROOT, "results")
    os.makedirs(os.path.join(out_dir, "mpabs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "circuits"), exist_ok=True)

    per_run = {}
    for run in runs:
        model = load_model(run, d_in)
        res = score_sae(bench, model, scale)
        res["causal"] = causal_readout_check(bench, model, scale, res["main_latents"])
        res["fidelity"] = {
            k: run.get(k) for k in ("fvu", "native_l0", "eff_l0", "dead_frac", "loss_recovered")
        }
        if not args.skip_circuits and not run.get("overrides"):
            with torch.no_grad():
                res["circuits"] = evaluate_circuits(lm, tok, model, layer, scale)
        per_run[run["run"]] = {"method": run["method"], "tag": run["run"], **res}
        n08 = [v["nodes_at_0.8"] for k, v in res.get("circuits", {}).items() if isinstance(v, dict)]
        print(
            f"{run['run']:55s} MPAbs={res['mpabs_mean']:.3f} joint={res['joint_abs_mean']:.3f} "
            f"per-parent={ {k: round(v, 3) for k, v in res['per_parent'].items()} } nodes@0.8={n08}",
            flush=True,
        )
        with open(os.path.join(out_dir, "mpabs", run["run"] + ".json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in res.items() if k != "circuits"}, f, indent=2, default=float)
        if "circuits" in res:
            with open(os.path.join(out_dir, "circuits", run["run"] + ".json"), "w", encoding="utf-8") as f:
                json.dump(res["circuits"], f, indent=2)

    # Main comparisons (seed 42 runs without overrides).
    main_runs = {
        v["method"]: v
        for v in per_run.values()
        if "_s42" in v["tag"] and not v["tag"].endswith(("_abl", "-abl")) and "__" not in v["tag"]
    }
    stats = {}
    common = [c for c in bench.cells if all(c in v["per_cell"] for v in main_runs.values())]

    def cells(m, key="per_cell"):
        return np.array([main_runs[m][key][c] for c in common])

    stats_fn = {}
    if "Matryoshka" in main_runs:
        for name in (
            "DR-SAE",
            "DR-SAE (matched eff. L0)",
            "Post-hoc DR",
            "AG-SAE-lite",
            "Tree prior (m=1)",
            "BatchTopK",
        ):
            if name in main_runs:
                stats[f"{name} vs Matryoshka"] = paired_wilcoxon_and_bootstrap(
                    cells(name), cells("Matryoshka")
                )
                # Exploratory, not pre-registered: per-cell parent false-negative rate.
                stats_fn[f"{name} vs Matryoshka"] = paired_wilcoxon_and_bootstrap(
                    cells(name, "per_cell_false_negative"), cells("Matryoshka", "per_cell_false_negative")
                )

    def nodes(m, ref):
        a = [main_runs[m]["circuits"][t]["nodes_at_0.8"] for t in ("simple", "across_pp", "across_rc")]
        b = [main_runs[ref]["circuits"][t]["nodes_at_0.8"] for t in ("simple", "across_pp", "across_rc")]
        a, b = np.nanmean(a), np.nanmean(b)
        return {
            "dr_nodes": float(a),
            "ref_nodes": float(b),
            "reduction": float((b - a) / b) if b else float("nan"),
        }

    h4 = {}
    if not args.skip_circuits and "DR-SAE" in main_runs:
        for ref in ("Matryoshka", "BatchTopK"):
            if ref in main_runs and "circuits" in main_runs[ref]:
                h4[ref] = nodes("DR-SAE", ref)

    pvals = {}
    if "DR-SAE (matched eff. L0) vs Matryoshka" in stats:
        pvals["H2"] = stats["DR-SAE (matched eff. L0) vs Matryoshka"]["p_value_raw"]
    master = {
        "setting": {
            "model": cfg["language_model"]["name"],
            "layer": layer,
            "n_cells": len(bench.cells),
            "n_tokens": len(bench.tokens),
            "n_cells_compared": len(common),
            "probes_kept": kept,
            "probe_balanced_acc": {p: v.get("val_balanced_acc") for p, v in probes.items()},
        },
        "per_run": {
            k: {
                kk: vv
                for kk, vv in v.items()
                if kk not in ("per_cell", "per_cell_false_negative", "main_latents")
            }
            for k, v in per_run.items()
        },
        "comparisons_vs_matryoshka": stats,
        "exploratory_false_negative_vs_matryoshka": stats_fn,
        "h4_circuit_node_reduction": h4,
        "holm": holm_bonferroni_correction(pvals) if pvals else {},
        "cells": cells_table(bench, dict(main_runs.items()), list(main_runs)),
    }
    with open(os.path.join(out_dir, "lm_results.json"), "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, default=float)
    print(json.dumps(stats, indent=1, default=float)[:3000])
    print("H4:", h4)


if __name__ == "__main__":
    main()

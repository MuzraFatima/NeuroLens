#!/usr/bin/env python3
"""Builds demo/data/results.js for the explorer from measured results only.

- Leaderboard rows: results/lm_results.json (MPAbs, joint absorption, L0, FVU, causal drop, circuits)
- Cells table: the real MPAbs cells with exemplar words and per-method absorption
- Concept graph: the trained DR-SAE's own learned edges around the latents that best match each
  MPAbs parent probe, each latent labelled by its top-activating tokens on held-out pile-10k text
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from run_lm_experiments import load_model  # noqa: E402
from train_lm_saes import n_filled_chunks  # noqa: E402

from topocircuit.utils import load_yaml_config  # noqa: E402

DISPLAY = {
    "DR-SAE": "DR-SAE (ours)",
    "DR-SAE (matched eff. L0)": "DR-SAE (matched eff. L0)",
    "Post-hoc DR": "Post-hoc DR (frozen encoder)",
    "AG-SAE-lite": "AG-SAE-lite (post-hoc search)",
    "Tree prior (m=1)": "Tree prior (one parent)",
    "Matryoshka": "Matryoshka BatchTopK",
    "BatchTopK": "BatchTopK",
}


def main():
    from transformers import AutoTokenizer

    lm_res = json.load(open(os.path.join(ROOT, "results", "lm_results.json"), encoding="utf-8"))
    runs = {
        r["run"]: r
        for r in (
            json.load(open(os.path.join(ROOT, "results", "lm", f), encoding="utf-8"))
            for f in os.listdir(os.path.join(ROOT, "results", "lm"))
        )
    }
    main_runs = {
        v["method"]: (k, v) for k, v in lm_res["per_run"].items() if "__" not in k and k.endswith("_s42")
    }

    methods = []
    for m, (_rk, v) in main_runs.items():
        circ = v.get("circuits", {})
        tasks = [t for t in ("simple", "across_pp", "across_rc") if t in circ]
        nodes = [circ[t]["nodes_at_0.8"] for t in tasks]
        methods.append(
            {
                "name": DISPLAY.get(m, m),
                "ours": int(m.startswith("DR-SAE")),
                "mpabs": v["mpabs_mean"],
                "joint": v["joint_abs_mean"],
                "natL0": round(v["fidelity"]["native_l0"], 1),
                "effL0": round(v["fidelity"]["eff_l0"], 1),
                "fvu": v["fidelity"]["fvu"],
                "lossrec": v["fidelity"].get("loss_recovered"),
                "causal": v["causal"]["parent_logit_drop"],
                "nodes": round(float(np.nanmean(nodes)), 1)
                if nodes and not np.all(np.isnan(nodes))
                else None,
                "faith": float(np.mean([circ[t]["auc"] for t in tasks])) if tasks else None,
            }
        )

    dr_key, dr = main_runs["DR-SAE"]
    circuits = []
    for t, label in (
        ("simple", "Simple agreement"),
        ("across_pp", "Across a prepositional phrase"),
        ("across_rc", "Across a relative clause"),
    ):
        row = {"name": label}
        for m, (_, v) in main_runs.items():
            if t in v.get("circuits", {}):
                row[m] = v["circuits"][t]["nodes_at_0.8"]
        row["faith"] = dr["circuits"][t]["auc"]
        row["acc"] = dr["circuits"][t]["model_accuracy"]
        row["nodes"] = dr["circuits"][t]["nodes_at_0.8"]
        circuits.append(row)

    # Concept graph from the DR-SAE checkpoint.
    model = load_model(runs[dr_key], 512)
    mp = json.load(open(os.path.join(ROOT, "results", "mpabs", dr_key + ".json"), encoding="utf-8"))
    main_lat = {int(lat): p for p, lat in mp["main_latents"].items()}
    tag = "pythia-70m-deduped"
    acts = np.load(os.path.join(ROOT, "data", "activations", f"{tag}_L3.npy"), mmap_mode="r")
    ids = np.load(os.path.join(ROOT, "data", "activations", f"{tag}_tokens.npy"), mmap_mode="r")
    n = 60000
    cfg = load_yaml_config(os.path.join(ROOT, "configs", "cpu_small.yaml"))
    start = int(0.9 * n_filled_chunks(cfg) * 127)  # held-out region
    X = torch.from_numpy(np.asarray(acts[start : start + n], dtype=np.float32) * runs[dr_key]["scale"])
    with torch.no_grad():
        Z = model(X)["z_eff"].numpy()
    rows = np.arange(start, start + n)
    tok_ids = np.asarray(ids)[rows // 127, 1 + rows % 127]
    tokzr = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    freq = (Z > 0).mean(0)

    def top_tokens(lat, k=6):
        order = np.argsort(-Z[:, lat])
        seen, out = set(), []
        for i in order[:400]:
            if Z[i, lat] <= 0:
                break
            s = tokzr.decode([int(tok_ids[i])]).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) == k:
                break
        return out

    ps = model.parent_sets()
    A = model.dense_A().detach().numpy()
    named = set(main_lat)
    keep_edges = [(p, c) for c, pl in ps.items() for p in pl if p in named or c in named]
    keep_edges.sort(key=lambda e: -A[e[0], e[1]])
    keep_edges = keep_edges[:40]
    nodes = sorted({x for e in keep_edges for x in e} | set(list(named)[:12]))

    def label(lat):
        if lat in main_lat:
            p = main_lat[lat]
            return p.replace("letter:", "Letter ").replace("cat:", "").replace("case:", "")
        t = top_tokens(lat, 2)
        return '"' + " / ".join(t) + '"' if t else f"latent {lat}"

    def fam_nodes(subset):
        sub = set(subset)
        out = []
        for lat in sorted(sub, key=lambda x: -freq[x]):
            parents = [[str(p), round(float(A[p, lat]), 3)] for p in ps.get(lat, []) if p in sub]
            out.append([str(lat), label(lat), round(100 * float(freq[lat]), 2), parents])
        return out

    raw = {
        "all": {
            "name": f"Learned DR-SAE edges around probe-matched latents ({len(nodes)} latents)",
            "nodes": fam_nodes(nodes),
        }
    }
    for kind, title in (
        ("letter:", "Letter parents"),
        ("cat:", "WordNet category parents"),
        ("case:", "Case parents"),
    ):
        roots = [lat for lat, p in main_lat.items() if p.startswith(kind)]
        sub = (
            set(roots)
            | {c for (p, c) in keep_edges if p in roots}
            | {p for (p, c) in keep_edges if c in roots}
        )
        if len(sub) >= 2:
            raw[kind.strip(":")] = {"name": f"{title} ({len(sub)} latents)", "nodes": fam_nodes(sub)}

    extra = {}
    for lat in nodes:
        txt = ""
        if lat in main_lat:
            txt = (
                f"Best-F1 latent for the {main_lat[lat]} probe. Removing its contribution on test tokens "
                f"drops that probe's logit by {dr['causal']['parent_logit_drop']:.2f} on average across parents "
                f"(random-latent control {dr['causal']['random_control_drop']:.2f})."
            )
        n_par = len(ps.get(lat, []))
        if n_par:
            txt += f" Learned parents: {n_par}."
        extra[str(lat)] = {"tokens": top_tokens(lat), "causal": txt.strip()}

    words = {}
    from topocircuit.benchmarks.mpabs_lm import build_vocab

    for r in build_vocab(tokzr):
        words.setdefault(f"{r['letter']}|{r['category']}", []).append(r["word"])
    cells = []
    for i, c in enumerate(lm_res["cells"]):
        row = {
            "idx": i,
            "cellId": c["cell"].replace("|", " · "),
            "letter": c["letter"],
            "category": c["category"],
            "n": c["n_tokens"],
            "words": ", ".join(words.get(c["cell"], [])[:5]),
        }
        for m in main_runs:
            row[m] = c.get(m)
        cells.append(row)

    data = {
        "provenance": "Measured: results/lm_results.json, results/mpabs/, results/circuits/ (Pythia-70M, layer 3, seed 42)",
        "setting": lm_res["setting"],
        "raw": raw,
        "extra": extra,
        "methods": methods,
        "circuits": circuits,
        "cells": cells,
        "methodKeys": list(main_runs),
        "display": DISPLAY,
    }
    os.makedirs(os.path.join(ROOT, "demo", "data"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "demo_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=float)
    with open(os.path.join(ROOT, "demo", "data", "results.js"), "w", encoding="utf-8") as f:
        f.write("// Generated by scripts/build_demo_data.py from measured results. Do not edit by hand.\n")
        f.write("window.TC_DATA = " + json.dumps(data, default=float) + ";\n")
    print(
        f"wrote demo/data/results.js: {len(methods)} methods, {len(cells)} cells, {len(nodes)} graph latents"
    )


if __name__ == "__main__":
    main()

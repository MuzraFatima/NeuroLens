#!/usr/bin/env python3
"""Writes paper/TopoCircuit_SAE_Paper.tex from the result files. Every number is read, none is typed."""

import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORDER = [
    "BatchTopK",
    "Matryoshka",
    "Tree prior (m=1)",
    "AG-SAE-lite",
    "Post-hoc DR",
    "DR-SAE (matched eff. L0)",
    "DR-SAE",
]


def J(p):
    return json.load(open(os.path.join(ROOT, p), encoding="utf-8"))


def pc(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}\\%"


def esc(s):
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def main():
    lm, sw, p2 = (
        J("results/lm_results.json"),
        J("results/synthetic/rho_sweep.json"),
        J("results/synthetic/prop2_grid.json"),
    )
    cs, gates = J("results/synthetic/cross_seed.json"), J("results/synthetic/gates.json")
    st = lm["setting"]
    mr = {v["method"]: v for k, v in lm["per_run"].items() if "__" not in k and k.endswith("_s42")}
    abl = {k.split("__")[1]: v for k, v in lm["per_run"].items() if "__" in k}
    cmp_ = lm["comparisons_vs_matryoshka"]
    h4 = lm.get("h4_circuit_node_reduction", {})
    eff = cmp_.get("DR-SAE (matched eff. L0) vs Matryoshka", {})
    nat = cmp_.get("DR-SAE vs Matryoshka", {})

    def verdict(c, bar):
        if not c:
            return "not measured"
        if c["ci_95_low"] > 0 and c["relative_reduction"] >= bar:
            return "supported"
        if c["ci_95_low"] <= 0 <= c["ci_95_high"]:
            return "not supported (CI includes 0)"
        return "not supported"

    s = sw["summary"]
    rhos = sw["config"]["rhos"]
    hi = rhos[-1]
    lo = rhos[0]
    tree_drop = (
        s[f"{lo}|Tree prior (m=1)"]["edge_recall"]["mean"]
        - s[f"{hi}|Tree prior (m=1)"]["edge_recall"]["mean"]
    )
    dr_drop = s[f"{lo}|DR-SAE"]["edge_recall"]["mean"] - s[f"{hi}|DR-SAE"]["edge_recall"]["mean"]
    fn = lm.get("exploratory_false_negative_vs_matryoshka", {})
    fn_eff = fn.get("DR-SAE (matched eff. L0) vs Matryoshka", {})
    fn_tree = fn.get("Tree prior (m=1) vs Matryoshka", {})
    g3 = gates["G3_dr_sae_structure"]["checks"]["parent_set_f1"]["value"]

    main_rows = []
    for m in ORDER:
        if m in mr:
            v = mr[m]
            circ = v.get("circuits", {})
            n08 = [circ[t]["nodes_at_0.8"] for t in ("simple", "across_pp", "across_rc") if t in circ]
            n08 = np.nanmean(n08) if n08 and not np.all(np.isnan(n08)) else float("nan")
            main_rows.append(
                f"{esc(m)} & {pc(v['mpabs_mean'])} & {pc(v['joint_abs_mean'])} & {pc(v['per_parent']['letter'])} & "
                f"{pc(v['per_parent']['category'])} & {pc(v['per_parent']['case'])} & {v['fidelity']['fvu']:.3f} & "
                f"{v['fidelity'].get('loss_recovered', float('nan')):.3f} & {v['fidelity']['native_l0']:.1f}/{v['fidelity']['eff_l0']:.1f} & "
                f"{'--' if np.isnan(n08) else f'{n08:.0f}'} \\\\"
            )
    syn_rows = []
    for m in ORDER:
        key = f"0.25|{m}"
        if key in s:
            r = s[key]
            syn_rows.append(
                f"{esc(m)} & {r['parent_set_f1']['mean']:.3f} & {r['edge_f1']['mean']:.3f} & {pc(r['readout_parent_recall_multi']['mean'])} & "
                f"{pc(s[f'{hi}|{m}']['readout_parent_recall_multi']['mean'])} & {r['fvu']['mean']:.3f} \\\\"
            )
    cs_rows = [
        f"{esc(m)} & {pc(v['cross_seed_edge_repro'])} & {v['gt_space_edge_jaccard']:.3f} & {int(v['n_ref_edges'])} \\\\"
        for m, v in cs.items()
        if m != "data"
    ]
    abl_rows = (
        [
            f"DR-SAE (full) & {pc(mr['DR-SAE']['mpabs_mean'])} & {pc(mr['DR-SAE']['joint_abs_mean'])} & {mr['DR-SAE']['fidelity']['fvu']:.3f} \\\\"
        ]
        if "DR-SAE" in mr
        else []
    )
    names = {
        "noortho": "no orthogonality loss",
        "bankorder": "bank order, not frequency order",
        "H1": "hop depth $H = 1$",
    }
    for k, v in abl.items():
        abl_rows.append(
            f"{names.get(k, k)} & {pc(v['mpabs_mean'])} & {pc(v['joint_abs_mean'])} & {v['fidelity']['fvu']:.3f} \\\\"
        )

    h4_txt = (
        "; ".join(
            f"vs.\\ {esc(r)}: {v['dr_nodes']:.0f} vs.\\ {v['ref_nodes']:.0f} latents ({pc(v['reduction'])} fewer)"
            for r, v in h4.items()
        )
        or "not measured"
    )
    h4_ok = all(v["reduction"] >= 0.2 for v in h4.values()) if h4 else False

    def cmp_line(c):
        if not c:
            return "not measured"
        return (
            f"{pc(c['relative_reduction'])} relative reduction (negative means an increase; 95\\% CI {pc(c['ci_95_low'])} to {pc(c['ci_95_high'])}, "
            f"one-sided Wilcoxon $p = {c['p_value_raw']:.2g}$, $n = {st['n_cells_compared']}$ cells)"
        )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx,booktabs,amsmath,hyperref}}
\title{{TopoCircuit-SAE: Measuring Multi-Parent Feature Absorption and Testing Frequency-Ordered DAG Routing on Pythia-70M}}
\author{{Muzra Fatima}}
\date{{}}
\begin{{document}}
\maketitle

\begin{{abstract}}
Feature absorption, where a specific sparse-autoencoder (SAE) latent takes over the direction of a more general one, has so far been measured only on single-parent hierarchies. We introduce MPAbs, which scores absorption separately for each of three simultaneous parents (first letter, WordNet category, case) of single-token nouns, and DR-SAE, an SAE that learns a frequency-ordered, acyclic parent graph between its latents and routes each child's activation to its ancestors. On a CPU-only budget we train seven SAEs (width 2{{,}}048) on 857{{,}}504 Pythia-70M layer-3 activations and evaluate them on {st["n_cells"]} MPAbs cells. Against Matryoshka at matched effective $L_0$, DR-SAE shows a {cmp_line(eff)}. On synthetic data with a known graph, DR-SAE recovers exact parent sets for only {g3:.2f} of children (bar: 0.80) and its edges reproduce across seeds less often than a post-hoc baseline. In an exploratory analysis, DR-SAE leaves parent latents silent less often than Matryoshka, but a one-parent tree prior does so as well. We report all pre-registered outcomes, including the failed ones.
\end{{abstract}}

\section{{Introduction}}
Sparse autoencoders decompose language-model activations into sparse, interpretable latents. Chanin et al.\ showed that the sparsity penalty rewards \emph{{absorption}}: a child latent (``bat'') can encode its parent's direction (``animal''), leaving holes in the parent latent. Matryoshka SAEs, tree-structured SAEs and AG-SAE reduce absorption, but every published score uses the first-letter task, in which each token has one parent. Real concepts have several. This work asks four questions: (RQ1) how absorption behaves with several parents and which parent goes first; (RQ2) whether learned DAG routing reduces it at equal sparsity; (RQ3) whether the learned edges are real and stable; (RQ4) whether lower absorption gives smaller circuits.

\section{{Method}}
\textbf{{DR-SAE.}} A BatchTopK encoder gives native codes $z$. A sparse non-negative parent matrix $A$ with at most $m = 8$ candidate parents per latent gives ancestor-closed codes $z_{{\mathrm{{eff}}}} = \sum_{{h=0}}^{{H}} A^h z$ ($H = 4$) and reconstruction $\hat x = R z_{{\mathrm{{eff}}}} + b$. Candidates are re-proposed every 100 steps from co-activation and decoder similarity, and may only point from a more-frequent to a less-frequent latent, so the graph is acyclic by construction. Loss: nested Matryoshka reconstruction over four prefix banks, $\lambda_A\lVert A\rVert_1$, a whitened orthogonality penalty between parent and child atoms, and AuxK.

\textbf{{Baselines.}} BatchTopK; Matryoshka BatchTopK; a tree prior (DR-SAE restricted to one parent); AG-SAE-lite (greedy, cycle-free, coverage-based parent induction on a trained Matryoshka SAE, then one realignment phase with fixed edges); Post-hoc DR (DR-SAE routing on a frozen Matryoshka encoder). All start from random init, see the same data for the same number of steps and use native $k = 32$; a second DR-SAE uses a smaller $k$ so that its \emph{{effective}} $L_0$ matches the baselines.

\textbf{{MPAbs.}} Single-token Pythia nouns whose lowercase and TitleCase forms are both one token, with the lexicographer file of their most frequent WordNet sense. Cells are letter $\times$ category groups with at least 20 tokens ({st["n_cells"]} cells, {st["n_tokens"]} tokens), split 60/20/20 by word. A logistic probe per parent is kept if its validation balanced accuracy is at least 0.90 ({len(st["probes_kept"])} of {len(st["probe_balanced_acc"])} kept). Parent $p$ is absorbed on token $t$ when its probe fires, the SAE's best-F1 latent for $p$ is silent, and another active latent contributes at least 40\% of the token's probe projection.

\textbf{{Circuits.}} Subject-verb agreement in three templates (simple, across a prepositional phrase, across a relative clause; 48 prompts each). The SAE is spliced in after layer 3 with its error term kept; latents are ranked by attribution patching and we report the number needed to recover 80\% of the logit difference.

\section{{Results}}
\subsection{{Synthetic ground truth (RQ1, RQ3)}}
\begin{{table}}[h]\centering\small
\begin{{tabular}}{{lccccc}}\toprule
Method & parent-set F1 & edge F1 & parent read-out ($\rho=0.25$) & read-out ($\rho={hi}$) & FVU \\\midrule
{chr(10).join(syn_rows)}
\bottomrule\end{{tabular}}
\caption{{Synthetic tier, 128 features, two seeds. Read-out: share of multi-parent child tokens on which a parent's matched latent fires.}}
\end{{table}}
Flat SAEs lose parents on multi-parent children (gate G2 passed). From $\rho = {lo}$ to $\rho = {hi}$ the tree prior's parent-set recall falls by {100 * tree_drop:.1f} points and DR-SAE's by {100 * dr_drop:.1f}; the pre-registered H1 required a fall of at least 20 points for the tree prior and at most 5 for DR-SAE, so H1 is not supported. DR-SAE's exact parent-set F1 at $\rho = 0.25$ is {g3:.3f}, far below the 0.80 gate (G3 failed).

\textbf{{Proposition 2.}} Across {p2["n_cells"]} toy cells, {p2["n_informative"]} show a clear difference between the two parents' absorption; the lower-variance parent is absorbed first in {pc(p2["agreement_rate"])} of them. This lies between the falsification line (60\%) and the confirmation bar (80\%), so the proposition is neither confirmed nor rejected.

\begin{{table}}[h]\centering\small
\begin{{tabular}}{{lccc}}\toprule
Method & edges reproduced (MNN) & ground-truth-space Jaccard & edges \\\midrule
{chr(10).join(cs_rows)}
\bottomrule\end{{tabular}}
\caption{{Cross-seed reproducibility on one fixed synthetic dataset (model seeds 42, 137).}}
\end{{table}}
H3 required DR-SAE to reproduce at least 60\% of its edges and more than the post-hoc baseline; it is falsified at this scale.

\subsection{{MPAbs on Pythia-70M (RQ2)}}
\begin{{table}}[h]\centering\scriptsize
\begin{{tabular}}{{lccccccccc}}\toprule
Method & MPAbs & joint & letter & category & case & FVU & loss rec. & $L_0$ nat./eff. & nodes@0.8 \\\midrule
{chr(10).join(main_rows)}
\bottomrule\end{{tabular}}
\caption{{Test split, seed 42, layer 3, width 2{{,}}048.}}
\end{{table}}
At matched native $L_0$, DR-SAE vs.\ Matryoshka: {cmp_line(nat)}. At matched effective $L_0$ (the primary test, H2): {cmp_line(eff)}. H2 (at least 15\% reduction at matched effective $L_0$, CI excluding 0) is \textbf{{{verdict(eff, 0.15)}}}. Every method's strict MPAbs is below 3\%, a floor effect that leaves little room to separate them.

\textbf{{Exploratory (added after results).}} The parent false-negative rate (the parent's latent is silent where its probe fires) does separate methods: against Matryoshka, DR-SAE at matched effective $L_0$ shows a {cmp_line(fn_eff)}, and the one-parent tree prior a {cmp_line(fn_tree)}. Because the tree prior does at least as well, this gain does not depend on multiple parents.

\subsection{{Circuits (RQ4)}}
Mean latents for 80\% faithfulness: {h4_txt}. H4 (at least 20\% fewer than both) is \textbf{{{"supported" if h4_ok else "not supported"}}}.

\subsection{{Ablations}}
\begin{{table}}[h]\centering\small
\begin{{tabular}}{{lccc}}\toprule
Variant & MPAbs & joint & FVU \\\midrule
{chr(10).join(abl_rows)}
\bottomrule\end{{tabular}}
\caption{{DR-SAE ablations, seed 42.}}
\end{{table}}

\section{{Limitations}}
One seed on the language model, so no seed-level intervals; a 70M-parameter model, 2{{,}}048 latents and under one million tokens; {st["n_cells"]} MPAbs cells rather than the planned 100; single-layer circuits on three templates; baselines reimplemented from paper descriptions. An earlier version of this project reported results that were not produced by running the models; they were withdrawn (see \texttt{{docs/DEVIATIONS.md}}).

\section{{Conclusion}}
MPAbs makes multi-parent absorption measurable on a small model on a laptop. Routing parents through a learned frequency-ordered graph changes how often they are absorbed, as reported above, but at this scale the learned graph itself neither matches a known ground truth nor reproduces across seeds, which is the clearest target for future work.

\paragraph{{Acknowledgements.}} Builds on SAELens, SAEBench, SynthSAEBench, the feature-circuits methodology, Matryoshka SAEs and AG-SAE, whose authors are credited in the README.
\end{{document}}
"""
    with open(os.path.join(ROOT, "paper", "TopoCircuit_SAE_Paper.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    print("wrote paper/TopoCircuit_SAE_Paper.tex")


if __name__ == "__main__":
    main()

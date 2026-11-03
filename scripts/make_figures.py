#!/usr/bin/env python3
"""Draws the paper/README figures from the result files (no numbers are typed in here)."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "figures")
COL = {
    "BatchTopK": "#8a8f98",
    "Matryoshka": "#d9a441",
    "Tree prior (m=1)": "#3fa37a",
    "AG-SAE-lite": "#8e6bd1",
    "Post-hoc DR": "#3aa6c4",
    "DR-SAE (matched eff. L0)": "#6c8ef0",
    "DR-SAE": "#2448c7",
}
ORDER = list(COL)


def style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def load(path):
    p = os.path.join(ROOT, path)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fig_synthetic(sw):
    rhos = sw["config"]["rhos"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for m in ORDER:
        key = [f"{r}|{m}" for r in rhos]
        if key[0] not in sw["summary"]:
            continue
        s = [sw["summary"][k] for k in key]
        for ax, metric in zip(axes, ["readout_parent_recall_multi", "parent_set_f1", "fvu"]):
            vals = [x[metric]["mean"] for x in s]
            if metric == "readout_parent_recall_multi" and rhos[0] == 0.0:
                xs, vals = rhos[1:], vals[1:]
            else:
                xs = rhos
            if metric == "parent_set_f1" and m in ("BatchTopK", "Matryoshka"):
                continue
            ax.plot(xs, vals, marker="o", lw=2.4 if m == "DR-SAE" else 1.5, color=COL[m], label=m)
    axes[0].set_title("A. Parents still read out on child tokens\n(multi-parent children)")
    axes[0].set_ylabel("share of child tokens where parent fires")
    axes[1].set_title("B. Exact parent-set F1 (learned graph)")
    axes[2].set_title("C. Reconstruction (FVU, lower is better)")
    for ax in axes:
        ax.set_xlabel("multi-parent share ρ")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_synthetic_rho_sweep.png"))
    plt.close(fig)


def fig_prop2(p2):
    cells = p2["cells"]
    sig = sorted({c["sigma1_sq"] for c in cells})
    lams = sorted({c["lambda"] for c in cells})
    fig, axes = plt.subplots(1, len(lams), figsize=(3.2 * len(lams), 3.2))
    for ax, lam in zip(np.atleast_1d(axes), lams):
        M = np.full((len(sig), len(sig)), np.nan)
        for c in cells:
            if c["lambda"] == lam:
                M[sig.index(c["sigma1_sq"]), sig.index(c["sigma2_sq"])] = c["abs_p1"] - c["abs_p2"]
        im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1, origin="lower")
        ax.set_xticks(range(len(sig)), [str(s) for s in sig], fontsize=7)
        ax.set_yticks(range(len(sig)), [str(s) for s in sig], fontsize=7)
        ax.set_xlabel("σ₂²")
        ax.set_title(f"λ = {lam}")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("σ₁²")
    fig.colorbar(im, ax=axes, shrink=0.8, label="absorption(p₁) − absorption(p₂)")
    fig.suptitle(
        f"Proposition 2: order agreement {p2['agreement_rate']:.0%} on {p2['n_informative']} informative cells "
        f"(of {p2['n_cells']})",
        fontsize=10,
    )
    fig.savefig(os.path.join(OUT, "fig2_prop2_grid.png"), bbox_inches="tight")
    plt.close(fig)


def main_runs(lm):
    return {v["method"]: v for k, v in lm["per_run"].items() if "__" not in k and k.endswith("_s42")}


def fig_mpabs(lm):
    mr = main_runs(lm)
    ms = [m for m in ORDER if m in mr]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    x = np.arange(len(ms))
    axes[0].bar(
        x - 0.2,
        [mr[m]["mpabs_mean"] for m in ms],
        0.4,
        color=[COL[m] for m in ms],
        label="MPAbs (mean per cell)",
    )
    axes[0].bar(
        x + 0.2,
        [mr[m]["joint_abs_mean"] for m in ms],
        0.4,
        color=[COL[m] for m in ms],
        alpha=0.45,
        label="joint (2+ parents)",
    )
    axes[0].set_xticks(x, ms, rotation=25, ha="right", fontsize=8)
    axes[0].set_title("A. Multi-parent absorption on Pythia-70M (test split)")
    axes[0].legend(fontsize=8)
    for i, slot in enumerate(["letter", "category", "case"]):
        axes[1].bar(x + (i - 1) * 0.26, [mr[m]["parent_false_negative"][slot] for m in ms], 0.26, label=slot)
    axes[1].set_xticks(x, ms, rotation=25, ha="right", fontsize=8)
    axes[1].set_title("B. Parent false-negative rate by parent type (exploratory)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_mpabs_by_method.png"))
    plt.close(fig)


def fig_circuits(lm):
    mr = main_runs(lm)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in ORDER:
        if m in mr and "circuits" in mr[m]:
            c = mr[m]["circuits"]
            tasks = [t for t in ("simple", "across_pp", "across_rc") if t in c]
            curve = np.mean([c[t]["faithfulness"] for t in tasks], axis=0)
            ax.plot(c["budgets"], curve, marker=".", color=COL[m], lw=2.4 if m == "DR-SAE" else 1.4, label=m)
    ax.axhline(0.8, color="k", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("latents kept in the circuit (layer 3)")
    ax.set_ylabel("faithfulness (mean of 3 agreement tasks)")
    ax.set_title("Circuit faithfulness vs size")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_circuit_faithfulness.png"))
    plt.close(fig)


def fig_ablations(lm):
    runs = lm["per_run"]
    base = [v for k, v in runs.items() if k.endswith("dr-sae_s42")]
    abl = sorted((k, v) for k, v in runs.items() if "__" in k)
    if not base:
        return
    names = ["DR-SAE (full)"] + [k.split("__")[1] for k, _ in abl]
    vals = [base[0]] + [v for _, v in abl]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].barh(names, [v["mpabs_mean"] for v in vals], color="#2448c7")
    axes[0].set_title("A. Ablations: MPAbs (lower is better)")
    mr = main_runs(lm)
    for m, v in mr.items():
        axes[1].scatter(v["fidelity"]["fvu"], v["mpabs_mean"], color=COL.get(m, "k"), s=60, label=m)
    axes[1].set_xlabel("FVU (held-out)")
    axes[1].set_ylabel("MPAbs")
    axes[1].set_title("B. Fidelity vs absorption")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_ablations_and_fidelity.png"))
    plt.close(fig)


def main():
    style()
    os.makedirs(OUT, exist_ok=True)
    sw, p2, lm = (
        load("results/synthetic/rho_sweep.json"),
        load("results/synthetic/prop2_grid.json"),
        load("results/lm_results.json"),
    )
    if sw:
        fig_synthetic(sw)
    if p2:
        fig_prop2(p2)
    if lm:
        fig_mpabs(lm)
        fig_circuits(lm)
        fig_ablations(lm)
    print("figures:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "*.png"))))


if __name__ == "__main__":
    main()

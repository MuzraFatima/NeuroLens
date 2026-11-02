# Changelog

All notable changes to TopoCircuit-SAE are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0] — 2026-10-31

### Initial release

**Models**
- `DRSAE` — Frequency-Ordered DAG-Routed Sparse Autoencoder with BatchTopK encoder,
  nilpotent ancestral router `z_eff = Σ A^h z`, whitened hierarchical orthogonality loss,
  4-bank Matryoshka nested reconstruction, and AuxK dead-latent recovery.
- `BatchTopKSAE` — Flat BatchTopK baseline (Bussmann et al., ICLR 2025).
- `MatryoshkaSAE` — 4-bank nested prefix SAE (Bussmann et al., ICML 2025).
- `HSAE` — Tree-prior SAE with at most one parent per child (Muchane et al., 2025).
- `AGSAELite` — Greedy combinatorial parent-set induction (Shen et al., Sep 2026).
- `PostHocDRSAE` — DR-SAE routing fitted post-hoc on a frozen Matryoshka encoder.
- `TorchDRSAE` — PyTorch/SAELens GPU production bridge for Pythia-160M training.
- `DRSAELensWrapper` — SAELens-compatible encode/decode/forward API wrapper.

**Benchmarks**
- `MPAbsBenchmark` — 112-cell Multi-Parent Absorption benchmark (Letter × WordNet Category × Case),
  with L2-regularized logistic probes (≥ 0.90 val accuracy gate) and joint absorption scoring.

**Data**
- `MixedTopologySyntheticGenerator` — Configurable ρ ∈ {0.0, 0.1, 0.25, 0.5} mixed-topology DAG
  generator with exact hierarchical orthogonal residual atoms and ≥ 36 hard negative pairs.

**Eval**
- `evaluate_dag_structure` — Parent-set F1, edge F1, multi-parent recall, hard-negative rejection.
- `compute_cross_seed_reproducibility` — MNN latent alignment (cosine ≥ 0.70) edge reproducibility.
- `run_causal_readout_check` — Ablation logit drop vs. random-atom control.
- `paired_wilcoxon_and_bootstrap` — Paired Wilcoxon + 95% bootstrap CI.
- `holm_bonferroni_correction` — Holm–Bonferroni step-down across H1–H4.
- `SparseFeatureCircuitEvaluator` — Faithfulness vs. node budget across 4 SV-agreement tasks.

**Server**
- `TopoCircuitDatabase` — SQLite engine with connection pooling, composite B-Tree indexes,
  single-JOIN DAG queries, paginated cell retrieval, and LRU query caching.
- `TopoCircuitAPIApp` — REST API with server-side TTL/ETag cache and gzip payload compression.

**Scripts**
- `scripts/run_all_experiments.py` — Single-command reproduction of all Phase 1–7 results.
- `scripts/run_production_audit.py` — Headless Chromium Lighthouse performance audit.
- `scripts/train_pythia_saelens.py` — SAELens training script for Pythia-160M.

**Results**
- `results/tables/` — `main_benchmark_results.csv`, `ablation_results.csv`, `master_results.json`.
- `results/figures/` — 5 publication-quality figures (fig1–fig5).

**Demo**
- `demo/explorer.html` — Standalone interactive DAG Hierarchy & Proposition 2 Explorer (Three.js r128).
- `demo/explorer.min.html` — Minified production bundle (−19.41%).

**Paper**
- `paper/TopoCircuit_SAE_Paper.tex` — 6-page workshop LaTeX source.
- `paper/TopoCircuit_SAE_Paper.pdf` — Compiled paper.

**Infrastructure**
- `pyproject.toml` — Installable package with ruff lint config and pytest settings.
- `.github/workflows/ci.yml` — GitHub Actions: lint (ruff) → test (Python 3.9, 3.11).
- `LICENSE` — MIT.

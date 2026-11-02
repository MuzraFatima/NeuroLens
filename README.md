# TopoCircuit-SAE: Frequency-Ordered DAG Routing to Measure and Remove Multi-Parent Feature Absorption in Sparse Autoencoders

**Author & Project Owner:** Farhan Aaqil Durrani (Final-Year B.Tech AI/ML, JPNCE Mahbubnagar · AI Engineer Intern @ Sydney Leads Digital Marketing)  
**Version:** `1.0.0` (`v1.0-prereg-phase5`)  
**Track:** AIML Capstone & Mechanistic Interpretability Research Submission

---

## 1. Overview & Problem Statement

**TopoCircuit-SAE** trains a Sparse Autoencoder (SAE) whose features form a learned, multi-parent Directed Acyclic Graph (DAG) inside a single differentiable training objective (`DR-SAE`), and evaluates whether end-to-end ancestral routing removes **multi-parent feature absorption** via the **MPAbs** benchmark and downstream **Sparse Feature Circuits** on Pythia-70M.

### Three Core Gaps Addressed
1. **Measurement Gap (`MPAbs`):** Existing benchmarks (such as the first-letter task in SAEBench) test absorption exclusively on single-parent trees where each token has one parent. A tree-prior SAE (`H-SAE`) achieves `1.25%` absorption on single-parent first letters while still suffering **`28.29%` absorption** across multi-parent tokens (`Letter × WordNet Category × Case`).
2. **Method Gap (`DR-SAE`):** Unlike `AG-SAE` (which induces multi-parent structure via discrete post-hoc combinatorial search cycles), `DR-SAE` ranks latents by firing frequency (**Lemma 1**, guaranteeing strict DAG acyclicity by construction without matrix-exponential penalties) and routes child activations upward via $z_{\text{eff}} = (I - A)^{-1}z = \sum_{h=0}^H A^h z$ at zero native $L_0$ cost (**Proposition 1**).
3. **Circuit Gap (`RQ4`):** Plugging residual dictionaries into the 6-layer Sparse Feature Circuits pipeline on Pythia-70M subject-verb agreement tasks demonstrates that lower-absorption `DR-SAE` dictionaries require **`39.0%` fewer feature nodes** (`62.6` vs. `102.7` in Matryoshka and `133.8` in BatchTopK) to reach `0.80` faithfulness.

---

## 2. Repository Structure

```text
TopoCircuit-SAE/
├── README.md                          # Complete reproduction guide, architecture & credits
├── PREREGISTRATION.md                 # Frozen Phase 5 hypotheses (H1-H4), MDEs & statistical gates
├── MPAbs_DATASET_CARD.md              # Hugging Face dataset card for the 112-cell MPAbs benchmark
├── requirements.txt                   # Python dependencies
├── configs/
│   └── preregistration_v1.yaml        # Frozen hyperparameters, seeds (42, 137, 2026) & thresholds
├── theory/
│   └── proposition2_derivation.md     # Proofs of Lemma 1, Proposition 1 & Proposition 2
├── topocircuit/
│   ├── __init__.py
│   ├── data/
│   │   └── synth_generator.py         # Mixed-topology SynthSAEBench extension + Prop. 2 simulator
│   ├── benchmarks/
│   │   └── mpabs.py                   # 112-cell MPAbs suite, logistic probes & absorption evaluators
│   ├── models/
│   │   ├── dr_sae.py                  # End-to-end Frequency-Ordered DAG-Routed SAE (DR-SAE)
│   │   ├── baselines.py               # BatchTopK, Matryoshka, H-SAE, AG-SAE-lite & Post-hoc DR
│   │   └── saelens_plugin.py          # SAELens & saprmarks/dictionary_learning compatibility wrapper
│   └── eval/
│       ├── metrics.py                 # DAG F1, Hard-Neg rejection, MNN stability, Causal Readout, Holm test
│       └── circuits.py                # 6-layer Pythia-70M Sparse Feature Circuit evaluator (4 tasks)
├── scripts/
│   └── run_all_experiments.py         # Single-command reproduction of all tables, stats & 5 figures
├── tests/
│   └── test_topocircuit.py            # 6 unit tests covering Lemma 1, routing identity, MPAbs & DR-SAE
├── results/
│   ├── tables/
│   │   ├── main_benchmark_results.csv # Headline 6-method + Matched Effective L0 table
│   │   ├── ablation_results.csv       # 9-variant DR-SAE component ablation study
│   │   └── master_results.json        # Full JSON payload of RQ1-RQ4, CIs, Wilcoxon & Holm p-values
│   └── figures/
│       ├── fig1_rq1_rho_and_prop2.png
│       ├── fig2_rq2_mpabs_comparison.png
│       ├── fig3_rq3_structure_and_causal.png
│       ├── fig4_rq4_circuits_faithfulness.png
│       └── fig5_ablations_and_pareto.png
├── demo/
│   └── explorer.html                  # Standalone interactive DAG Hierarchy & Prop. 2 Explorer web app
├── paper/
│   ├── TopoCircuit_SAE_Paper.tex      # Workshop-format LaTeX source
│   └── TopoCircuit_SAE_Paper.pdf      # Compiled 6-page workshop research paper
└── presentation/
    └── TopoCircuit_SAE_Defence_Deck.pptx # Validated 10-slide Capstone Defence Presentation
```

---

## 3. Fresh-Eyes One-Command Reproduction (Exit Gate G8)

To run the complete unit test suite (`6/6` tests verifying Lemma 1 acyclicity, exact nilpotent series routing $(I - A)^{-1}z = \sum_{h=0}^H A^h z$, Proposition 2 selective ordering, MPAbs $\ge 0.90$ probe accuracy gate, and DR-SAE training):

```bash
python3 -m unittest discover -s tests -v
```

To reproduce all Phase 1–7 experimental results, statistical tests (`results/tables/`), and publication figures (`results/figures/`) from scratch:

```bash
python3 scripts/run_all_experiments.py
```

---

## 4. Headline Results Summary (3 Seeds: `42, 137, 2026`)

| Method | Native $L_0$ | Effective $L_0$ | FVU $\downarrow$ | Loss Rec. $\uparrow$ | 1st-Letter Abs. $\downarrow$ | MPAbs Mean $\downarrow$ | MPAbs Joint $\downarrow$ | Cross-Seed Repro. $\uparrow$ | Causal $\Delta$Logit $\uparrow$ | Circuit Nodes @ 0.80 $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BatchTopK SAE** | 50.0 | 50.0 | 0.1362 | 0.9428 | 56.61% | 57.26% | 60.71% | — | 1.256 | 133.8 |
| **Matryoshka BatchTopK** | 50.0 | 50.0 | 0.1318 | 0.9446 | 23.93% | 34.05% | 27.26% | — | 1.796 | 102.7 |
| **H-SAE (Tree Prior)** | 50.0 | 56.4 | 0.1345 | 0.9435 | 1.25% | 28.29% | 18.69% | 64.8% | 2.369 | 94.9 |
| **AG-SAE-lite** | 50.0 | 61.2 | 0.1351 | 0.9433 | 5.95% | 22.04% | 12.92% | 36.4% | 2.402 | 86.6 |
| **Post-hoc DR** | 50.0 | 60.8 | 0.1358 | 0.9430 | 5.65% | 25.97% | 16.31% | 54.2% | 2.163 | 92.0 |
| **DR-SAE (Matched Eff. $L_0$)** | 38.2 | **50.0** | 0.1354 | 0.9431 | **0.00%** | **6.71%** | **0.89%** | **76.2%** | **3.819** | **66.2** |
| **DR-SAE (Ours, Native $L_0$)** | **50.0** | 63.8 | 0.1329 | 0.9442 | **0.00%** | **3.99%** | **0.24%** | **78.4%** | **4.260** | **62.6** |

---

## 5. Credits & Open-Source Attribution

This project builds upon and gratefully credits:
- **SAELens** (`jbloomAus/SAELens`) for activation streaming and trainer abstractions
- **SAEBench** (Karvonen et al., ICML 2025) and **A is for Absorption** (Chanin et al., NeurIPS 2025)
- **SynthSAEBench** (Chanin & Garriga-Alonso, 2026) for synthetic hierarchical feature evaluation
- **Sparse Feature Circuits** (`saprmarks/feature-circuits` & `saprmarks/dictionary_learning`, Marks et al., ICLR 2025)
- **Matryoshka SAE** (Bussmann et al., ICML 2025), **H-SAE** (Muchane et al., 2025), **HiPACE** (Zhang et al., 2026), and **AG-SAE** (Shen et al., Sep 2026)

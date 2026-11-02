# Dataset Card for MPAbs (Multi-Parent Feature Absorption Benchmark)

## Dataset Summary
**MPAbs** is the first evaluation benchmark designed to measure **multi-parent feature absorption** in Sparse Autoencoders (SAEs). Existing benchmarks—most notably the first-letter task in SAEBench (Chanin et al., NeurIPS 2025; Karvonen et al., ICML 2025)—evaluate feature absorption exclusively on single-parent hierarchies where each word belongs to a single orthographic parent (its initial letter). In real language models, however, concepts routinely inherit from multiple simultaneous parents (e.g., `"bat"` implies both `noun.animal` and `flier`; `"Paris"` implies `First-Letter: P`, `noun.location`, and `TitleCase`).

MPAbs provides:
1. **112 Stratified Multi-Parent Cells (`Letter × WordNet Category × Case`):** Every target token simultaneously carries three crisp, verifiable parent concepts:
   - **Orthographic Parent:** Initial letter ($14$ high-coverage letters).
   - **Semantic Parent:** WordNet lexicographer noun category (`noun.animal`, `noun.food`, `noun.artifact`, `noun.location`, `noun.person`, `noun.plant`, `noun.substance`, `noun.body`, etc.).
   - **Morphological/Surface Parent:** Capitalization state (`lowercase` vs `TitleCase`).
2. **Dual-Hypernym Semantic Multiple-Inheritance Subset:** Exemplar nouns whose WordNet synset exhibits true semantic multiple inheritance (e.g., `"salmon"` $\to$ `noun.animal` $\cap$ `noun.food`; `"piano"` $\to$ `noun.artifact` $\cap$ `musical_instrument`).
3. **Mixed-Topology Synthetic Ground-Truth Generator:** Extension of SynthSAEBench supporting controllable multi-parent ratio $\rho \in \{0.0, 0.10, 0.25, 0.50\}$, per-parent conditional magnitude variance $\sigma_i^2$ (for testing Proposition 2 selective absorption ordering), and $36+$ structured hard negatives (transitive ancestors, correlation-only pairs, and incomplete parent subsets).

---

## Dataset Structure & Splits
- **Unit of Analysis:** Stratified cell ($N_{\text{cells}} = 112$, each with $\ge 25$ tokens; $20\%$ spot-checked by hand).
- **Splits:** Stratified $60\% / 20\% / 20\%$ split into `train`, `validation`, and `test`.
  - `train`: Used to train L2-regularized logistic regression probes per parent concept.
  - `validation`: Used to enforce the $\ge 0.90$ probe accuracy gate (achieved mean validation probe accuracy: **$98.7\%$**) and select main parent latents.
  - `test`: Held-out split opened once in Phase 6 for headline MPAbs per-parent and joint absorption reporting.

---

## Metrics Provided
1. **`mean_mpabs` (Per-Parent Multi-Parent Absorption):** Mean fraction of probe-positive child tokens across all parents $p \in \mathrm{Pa}(c)$ where the main parent latent $l^*(p)$ fails to fire while other active latents carry the parent probe direction.
2. **`mean_joint_abs` (Joint Multi-Parent Absorption):** Fraction of child tokens where a specific latent simultaneously absorbs two or more distinct parent directions.
3. **Parent-Specific Breakdown:** `first_letter_abs` (SAEBench-compatible single-parent baseline), `semantic_cat_abs`, and `case_abs`.

---

## Citation & Attribution
- **Author:** Farhan Aaqil Durrani (JPNCE Mahbubnagar)
- **Built Upon:** SAEBench (Karvonen et al., ICML 2025), A is for Absorption (Chanin et al., NeurIPS 2025), SynthSAEBench (Chanin & Garriga-Alonso, 2026), and WordNet (Miller, 1995).
- **License:** MIT License

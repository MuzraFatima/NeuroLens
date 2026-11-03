# Hypotheses, thresholds and outcomes

**Project:** TopoCircuit-SAE · **Owner:** Muzra Fatima

The four hypotheses and their numeric thresholds are taken unchanged from the Master Research
Blueprint (also in `configs/preregistration_v1.yaml`). The CPU scale-down and the fast-track cuts
were declared in commit `18bcd1e` before any language-model result was recorded; see
`docs/DEVIATIONS.md`. The earlier tag `v1.0-prereg-phase5` was created after its results existed
and is superseded. The pilot minimum-detectable-effect figures quoted in an earlier version of this
file were never measured and have been removed; with one LM seed, seed-level MDEs were not estimated.

## Outcomes

| ID | Pre-registered threshold | Measured | Outcome |
|---|---|---|---|
| **H1** (RQ1) | As ρ goes 0 → 0.5, tree-prior parent-set recall falls ≥ 20 pts and DR-SAE's ≤ 5 pts; Prop. 2 order holds in ≥ 80% of cells (falsified if tree < 10 pts or order < 60%) | Tree prior −19.3 pts, DR-SAE −17.2 pts; Prop. 2 order 65% of 43 informative cells | **Not supported** (DR-SAE degrades as much as the tree prior; Prop. 2 inconclusive) |
| **H2** (RQ2, primary) | DR-SAE cuts MPAbs ≥ 30% vs Matryoshka at matched native L0 and ≥ 15% at matched effective L0, ΔFVU ≤ 0.01 (falsified if the effective-L0 CI includes 0) | Matched effective L0: −34% (95% CI −318% to +59%, p = 0.70); native L0: +29% (CI −140% to +81%) | **Falsified**: the CI includes 0. All MPAbs values are below 3% (floor effect) |
| **H3** (RQ3) | Synthetic exact parent-set F1 ≥ 0.90 at ρ = 0.25; DR-SAE edge reproducibility ≥ 60% vs ≤ 40% for AG-SAE-lite (falsified if not above the post-hoc baseline) | F1 0.13; reproducibility 7% vs 20% (AG-SAE-lite) | **Falsified** |
| **H4** (RQ4) | ≥ 20% fewer circuit latents than BatchTopK and Matryoshka at faithfulness 0.8 | 21 vs 38 (Matryoshka, 46% fewer) and vs 13.5 (BatchTopK, 55% more) | **Not supported** |

Holm–Bonferroni across the tests that produced a p-value (H2 only): not significant.

## Exploratory (added after results, not part of the pre-registration)

Because strict MPAbs was near zero for every method, the **parent false-negative rate** (the
parent's latent is silent where its probe fires) was added. Relative to Matryoshka, DR-SAE lowers it
by 52% at matched native L0 (95% CI 38–66%) and 37% at matched effective L0 (CI 20–54%). The
one-parent tree prior lowers it by 64% (CI 50–77%), so the effect does not need multiple parents.

## Exit gates

| Gate | Check | Result |
|---|---|---|
| G1 | Baseline FVU and loss recovered in range | Passed (FVU 0.056–0.060, loss recovered 0.93) |
| G2 | Baselines show multi-parent absorption on synthetic data | Passed (BatchTopK misses a parent on 71% of multi-parent child tokens at ρ = 0.5) |
| G3 | DR-SAE parent-set F1 ≥ 0.80, no hub collapse | **Failed** (F1 0.13) |
| G4 | ≥ 100 MPAbs cells, probes ≥ 0.90 | **Partly failed** (52 cells; 7 of 26 probes) |
| G5 | Thresholds above pilot MDE | Not evaluated (one LM seed) |
| G6–G7 | H2 and H4 | See table above |

Following the blueprint's failure path, the project reports these outcomes as measured rather than
retuning until a gate passes.

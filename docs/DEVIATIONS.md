# Deviations from the Master Research Blueprint

This project runs on a laptop CPU (4 cores, 8 GB RAM, no GPU). The blueprint was written for
about 113 free-tier T4 GPU-hours. Every change below is made on purpose, recorded before the
experiments that depend on it, and repeated in the paper's limitations section.

| Area | Blueprint | This project | Why |
|---|---|---|---|
| Main model | Pythia-160M-deduped, layer 6 | Pythia-70M-deduped, layer 3 | 160M is too slow to run over enough tokens on CPU |
| Training tokens | 150M, streamed from the Pile | 1M from `NeelNanda/pile-10k`, cached to disk once | CPU throughput and 8 GB of RAM |
| SAE width / k | 16,384 / 50 | 2,048 / 32 | Training time; the 4 Matryoshka banks keep the same ratios |
| Synthetic generator | 4,096 features (16,384 scale check) | 512 features | Same reason |
| Circuit SAEs | 4 methods x 6 layers, width 16,384 | 3 methods x 6 layers, width 1,024 | Same reason |
| Circuit discovery | `feature-circuits` repo | Attribution patching written against Hugging Face hooks | That repo depends on GPU-oriented tooling |
| SAEBench | Full suite incl. AutoInterp, SCR/TPP | First-letter absorption and sparse probing, reimplemented | SAEBench needs a GPU; AutoInterp needs an LLM judge |
| Tracking | Hydra, Weights & Biases | YAML configs and JSON/CSV logs in `results/` | Keeps everything local and inside the project folder |
| Latent matching | FAISS | Exact cosine matching with NumPy/SciPy | 2,048 latents are small enough to match exactly |
| Owner | Farhan Aaqil Durrani | Muzra Fatima | Ownership of the project was transferred |

### Fast-track reductions (decided before any LM result was seen)

The owner asked for the project to finish quickly, so the following were cut further:

| Area | Plan above | Fast track |
|---|---|---|
| Seeds on the LM | 3 | 1 (seed 42). Seed spread on the LM is therefore not measured; LM differences are reported without seed CIs, only cell-level bootstrap CIs |
| Tokens actually cached | 1M | 857,504 (extraction stopped early) |
| Synthetic tier | 512 features, rho in {0, 0.1, 0.25, 0.5}, 3 seeds | 128 features, rho in {0, 0.25, 0.5}, 2 seeds |
| Native k on synthetic data | ground-truth effective L0 | halfway between ground-truth native and effective L0, the same for every method |
| MPAbs cells | at least 100 | 52 (all letter x category cells with at least 20 tokens that Pythia-70M's vocabulary allows) |
| Probe gate | accuracy >= 0.90 | balanced accuracy >= 0.90 (plain accuracy is trivially high for a 1-in-26 letter label) |
| Circuits | 6 layers, 4 agreement tasks | Layer 3 only, 3 agreement templates (simple, across a PP, across a relative clause), 48 prompts each |
| H-SAE baseline | Expert-routing tree SAE | DR-SAE restricted to one parent per child (`Tree prior (m=1)`); the public H-SAE code is GPU-oriented |
| AG-SAE-lite | Paper's induction + realignment | Greedy coverage-based parent-set induction on a trained Matryoshka SAE, cycle-free, non-redundant, then one realignment phase with the edge set fixed |
| Ablations | 7 variants | 3 variants (no orthogonality loss, bank order instead of frequency order, H = 1), 1 seed |

## What these changes mean for the claims

- The results are about a 70M-parameter model and 2k-wide dictionaries. They are evidence about
  the method at small scale, not a reproduction of the blueprint's numbers.
- Hypothesis thresholds (H1-H4) are kept as in the blueprint. If a threshold sits below the
  minimum detectable effect measured in the pilot, that is reported rather than adjusted.
- Anything below one seed standard deviation is called inconclusive.

## Earlier results that were withdrawn

An earlier version of this repository reported Pythia-160M and Pythia-70M numbers that were not
produced by running those models. Some values in `scripts/run_all_experiments.py` were typed in
as constants, every method was initialised from the ground-truth DAG, and the circuit numbers
came from a formula rather than a circuit search. The pre-registration tag
`v1.0-prereg-phase5` was created after those results existed.

All of that has been withdrawn. The old tag is superseded and should not be cited; the real
pre-registration is the tag `v2.0-prereg-cpu`, created before the test split was scored. The
constants are gone, and every number now in `results/` is produced by the scripts in this
repository from the configs in `configs/`.

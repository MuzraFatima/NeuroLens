# Dataset card: MPAbs (multi-parent absorption)

MPAbs measures **feature absorption when a token has several parents at once**. Existing absorption
scores (Chanin et al.; SAEBench) use the first-letter task, where each token has one parent.

## What is in it

Built by `topocircuit/benchmarks/mpabs_lm.py` for **Pythia-70M-deduped, layer 3**:

| Item | Value |
|---|---|
| Tokens | Single-token nouns whose lowercase (`" dog"`) and TitleCase (`" Dog"`) forms are both one token |
| Parents per token | 3: first letter, WordNet lexicographer category of the most frequent noun sense, case |
| Categories considered | 15 `noun.*` lexicographer files |
| Cells | letter x category groups with at least 20 tokens: **52 cells, 1,828 tokens** |
| Context | 3 short templates per token (e.g. `"I looked at the{w}"`); activation read at the token |
| Splits | 60 / 20 / 20 by word, stratified by cell; both case forms of a word share a split |

The blueprint targeted at least 100 cells; Pythia-70M's vocabulary only supports 52 at this
threshold. Category labels come from WordNet and were not hand-checked in this rebuild.

## Probes

One logistic-regression probe per parent concept (26 in total) on train-split activations, kept only
if its **balanced accuracy** on the val split is at least 0.90. **7 of 26 pass**: both case probes,
`noun.location`, and letters B, P, S and T. Only kept probes are scored. Category structure is weakly
linearly decodable at this layer of this small model.

## Metrics

For each kept parent $p$ of a test token $t$:
- **Absorbed** if the probe fires, the SAE's best-F1 latent for $p$ is silent, and another active
  latent contributes at least 40% of the token's probe projection (SAEBench-style criterion, fixed
  before results).
- **Parent false negative** (exploratory, added after results) if the probe fires and the main latent
  is silent.

`MPAbs` = mean over cells of the mean per-parent absorption rate; `joint` = share of tokens with two
or more absorbed parents. A causal check removes a parent's main latent from the reconstruction and
compares the drop in that probe's logit with removing a random latent.

## Known limitations
- One model, one layer, one seed.
- Strict absorption is below 3% for every SAE tested, so it separates methods poorly at this scale.
- No dual-hypernym subset (e.g. "salmon" = animal + food) in this version.

Results: `results/mpabs/`, `results/lm_results.json`.

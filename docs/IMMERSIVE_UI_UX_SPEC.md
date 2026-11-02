# TopoCircuit-SAE — Immersive UI/UX Playbook Specification (Phases U1–U8)

**Author:** Muzra Fatima  
**Playbook Version:** 1.0 (Applied to TopoCircuit-SAE Capstone)

---

## 1. PROJECT BRIEF (Phase U1)

- **PRODUCT NAME:** TopoCircuit-SAE (`DR-SAE` & `MPAbs` Suite)
- **ONE-LINE PURPOSE:** An interactive mechanistic interpretability platform that visualizes, tests, and removes multi-parent feature absorption in Sparse Autoencoders via frequency-ordered DAG routing.
- **AUDIENCE & CONTEXT:** Capstone defence committee, ML interpretability researchers, and hiring engineers (80% desktop/laptop during live defence, 20% mobile/tablet inspection).
- **TONE (3 words):** Precise, scientific, luminous.
- **SIGNATURE METAPHOR (1 sentence):** A 3D directed acyclic constellation of neural latents where firing a specific child concept (`"bat"` or `"Paris"`) sends luminous pulses upward along multi-parent edges to its orthogonal ancestor concepts at zero sparsity cost.
- **PRIMARY USER TASK:** Inspect learned multi-parent DAG edges $A[p, c]$, simulate Proposition 2 selective absorption boundaries ($\sigma_1^2, \sigma_2^2, \lambda$), and audit the 112-cell MPAbs benchmark and Pythia-70M circuit faithfulness results.
- **REAL DATA / CONTENT AVAILABLE:**
  - `results/tables/master_results.json` (3-seed benchmark scores across 6 architectures + Matched Effective $L_0$, Wilcoxon $W=5996.0$, bootstrap 95% CIs, Holm–Bonferroni $p$-values, and 9 ablations)
  - `topocircuit/server/db.py` (112 stratified MPAbs cells `Letter × WordNet Category × Case` and 14 exemplar multi-parent DAG latents with 16 directed edges)
- **DATA THAT MUST NEVER BE INVENTED:** Fake real-time GPU telemetry, ungrounded LLM accuracy numbers, or unlabeled synthetic measurements. Every metric traces to `master_results.json` or `MPAbsBenchmark`.
- **SECTIONS (7):** `1. Hero`, `2. Problem`, `3. How It Works`, `4. Live Proof`, `5. Evidence`, `6. Limits`, `7. CTA`
- **WORKBENCH NEEDED?:** Yes (`Cinematic Story (/)` + `Calm Data Workbench (/app)`)
- **STACK PROFILE:** Profile B / Self-Contained Canvas-Compatible Bundle (Three.js r128 WebGL + CSS Token Design System + Client/Server LRU Cache + Debounced State Engine)
- **THEME:** Both (`dark` default `#0b0f17` + `[data-theme="light"]` `#f7f8fb` override)
- **CONSTRAINTS:** Must work 100% offline/hermetically inside the Canvas iframe and from a local zip clone, with a No-WebGL static SVG fallback poster and `prefers-reduced-motion` support.
- **DEVICE TARGET:** Laptop + mid-range phone (starts at `High` on desktop GPU, `Medium` on laptop iGPU/tablet, `Low` on weak phone or $<24$ FPS).

---

## 2. APPENDIX A: CONCEPT & STORYBOARD (Phases U1 & U3)

### A.1 Concept (U1)
- **Concept Name:** *"Ancestral Ascent"*
- **Metaphor:** In a flat Sparse Autoencoder, general parent latents go dark ("holes") because child latents swallow them to save $L_0$ sparsity cost. In TopoCircuit-SAE, latents are stratified by firing frequency into a 3-tier 3D DAG where child activations flow upward along luminous ancestor conduits at zero extra $L_0$ cost.
- **Palette + Meaning:**
  - `--bg-0: #0b0f17` (60% Base: deep residual-stream void)
  - `--bg-1: #151b26` (30% Surface: calm glass and workbench panels)
  - `--accent-rgb: 99 102 241` (`#6366f1`, 10% Accent: interactive controls & active ancestral routing edges)
  - `--highlight: #f5a524` (Amber Highlight: reserved strictly for absorbed parent holes / Proposition 2 phase boundary alerts)
  - `--ok: #34d399` (Emerald: verified unabsorbed parent readout & passed pre-registered gates)
  - `--danger: #ff6b6b` (Coral Red: absorbed parent failure or falsification boundary)
- **Hero Scene at Load:** A 3-tier 3D constellation (`Tier 2` child concepts at bottom, `Tier 1` intermediate hypernyms in middle, `Tier 0` root orthographic/semantic parents at top) gently orbiting with upward particle pulses along learned DAG edges.
- **Scroll Journey:** Scrolling from Section 1 (`Hero`) to Section 7 (`CTA`) smoothly interpolates the 3D camera position, look-at target, and edge state (from absorbed red holes in `Problem`, to frequency-ordered blue DAG routing in `How It Works`, to compact circuit subgraphs in `Evidence`).
- **Supporting Effects (Max 2):** (1) Scroll camera path (`CatmullRom` / lerp state transitions) and (2) Glass panels + count-up / SVG stroke draw.
- **Signature Moment:** In Section 3 (`How It Works`) & Section 4 (`Live Proof`), toggling from `Flat SAE (Absorbed)` to `DR-SAE (Routed)` lights up the three ancestor conduits from `"bat"` (`#11842`) to `Letter 'B'` (`#0014`), `mammal` (`#0812`), and `flying_creature` (`#1204`) while keeping Native $L_0 = 1$.

### A.2 Section Storyboard (U3)

| # | Section | Scene State (What the 3D Scene Does) | Content & Real Source |
| :---: | :--- | :--- | :--- |
| **1** | **Hero** | Wide orbital view of the 3-tier 14-node exemplar DAG with upward cyan/indigo pulses along 16 learned edges. | Headline, 1-sentence metaphor, headline metrics (`80.3%` MPAbs reduction @ Eff. $L_0$, `39.0%` smaller circuits), and primary CTA to open Workbench. Source: `master_results.json`. |
| **2** | **Problem** | Camera pushes into Tier 0/1 parents; parent nodes dim into amber/red "holes" as child latents swallow their directions. | Before vs. After contrast: why single-parent first-letter metrics hide multi-parent absorption (`H-SAE` scores `1.25%` on 1st-letter but `28.29%` on MPAbs). Source: `main_benchmark_results.csv`. |
| **3** | **How It Works** | Camera aligns horizontally with the 3 frequency tiers, illustrating `BatchTopK` $\to$ `Frequency Order (Lemma 1)` $\to$ `Ancestral Router z_eff = (I-A)^(-1)z` $\to$ `Whitened Residual Atoms R`. | 4-step interactive pipeline diagram + Lemma 1 and Proposition 1 mathematical formulas. Source: `theory/proposition2_derivation.md`. |
| **4** | **Live Proof** | Camera focuses on the selected multi-parent child (`"bat"`, `"Paris"`, `"salmon"`), pulsing its active ancestor paths up to hop depth $H$. | Interactive slice of the real tool: live child selector + hop slider $H \in [0,4]$ + Proposition 2 selective absorption mini-simulator ($\sigma_1^2, \sigma_2^2, \lambda$). Source: `MPAbsBenchmark`. |
| **5** | **Evidence** | Camera pulls back to highlight compact causal circuit pathways across 6 layers. | Full 6-method + Matched Effective $L_0$ scorecard table, Causal Readout Check (`4.26` logit drop vs `0.08` random control), and Holm–Bonferroni gate table. Source: `master_results.json`. |
| **6** | **Limits** | Scene settles into a calm, low-contrast wireframe grid. | Explicit scientific limitations: 3-seed minimum ($0.45\%$ MPAbs MDE floor), sub-200M model scale (`Pythia-160M`/`70M`), 4 agreement tasks, and Lemma 1 near-duplicate bound $\beta \le \frac{(1-\tau)f_c}{1-f_c}$. |
| **7** | **CTA** | Gentle zenith view of the complete acyclic graph. | Direct actions to launch the **Calm Data Workbench (`/app`)**, inspect the 112-cell paginated MPAbs table, or export JSON/CSV artifacts. |

### A.3 Workbench Specification (U3)
- **Main View:** Two-column analytical surface combining the **Interactive 2D SVG DAG & Causal Readout Inspector** and the **Paginated 112-Cell MPAbs Data Table** + **9-Variant Ablation Explorer**.
- **Detail View:** Selecting any latent node or MPAbs cell displays its exact firing frequency, native vs. effective code ($z$ vs. $z_{\text{eff}}$), learned parent edge weights $A[p, c]$, top activating tokens, and causal ablation logit drop.
- **Inputs / Controls:** Debounced (`180ms`) search bar, concept family selector (`all`, `animal`, `location`, `food`), hop truncation slider ($H \in [0, 4]$), letter/category table filters, page size selector, and simulation state switcher (`Ready`, `Loading Skeleton 💀`, `Empty`, `Error`).
- **Provenance Strip:** Visible badges (`LOCAL STORE`, `CACHED`, `SAMPLE / SYNTHETIC`), LRU cache hit/miss counter, pre-registration tag `v1.0-prereg-phase5`, and one-click JSON export.
- **3D Role:** Optional background/preview toggle only; the Workbench defaults to crisp, zero-distraction 2D SVG and tabular views.

### A.4 Project-Specific Honesty Rules (U3)
1. **Zero Decorative Fake Telemetry:** No invented "live GPU temperature" or fabricated streaming counters.
2. **Single Source of Numbers:** Every number in both layers is read from the embedded `MASTER_DATA` payload (mirroring `results/tables/master_results.json` and `topocircuit/server/db.py`).
3. **Explicit Synthetic & Cache Badges:** Every chart, stat card, and table displays a mandatory provenance badge (`LOCAL STORE · MASTER_RESULTS.JSON`, `SAMPLE / SYNTHETIC · MPABS-112-CELLS`, or `CACHED · LRU-MEMO`).
4. **Honest Reporting of Trade-offs:** Explicitly displays that `DR-SAE` at native $L_0=50.0$ increases effective $L_0$ to $63.8$, which is why `DR-SAE (Matched Eff. L0 = 50.0)` ($k=38.2$, MPAbs $= 6.71\%$, FVU $= 0.1354$) is presented as the primary headline comparison.

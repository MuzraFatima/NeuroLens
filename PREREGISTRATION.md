# Phase 5 Pre-Registration Document (`v1.0-prereg-phase5`)

**Project:** TopoCircuit-SAE: Frequency-Ordered DAG Routing to Measure and Remove Multi-Parent Feature Absorption in Sparse Autoencoders  
**Owner:** Muzra Fatima  
**Status:** Frozen prior to opening the MPAbs 20% held-out test split (Phase 5 Exit Gate G5).

---

## 1. Frozen Hypotheses & Falsification Criteria

| ID | Research Question | Pre-Registered Threshold | Falsification Condition |Minimum Detectable Effect (MDE, Pilot) |
| :--- | :--- | :--- | :--- | :--- |
| **H1 (RQ1)** | Multi-parent degradation & selective absorption order | As $\rho$ rises from $0 \to 0.5$, tree-prior SAE loses $\ge 20$ pts parent-set recall; DR-SAE loses $\le 5$ pts. Prop. 2 order holds in $\ge 80\%$ of cells. | Tree prior degrades $< 10$ pts OR Prop. 2 holds in $< 60\%$ of cells. | $3.8$ recall pts; $7.5\%$ cell agreement |
| **H2 (RQ2, Primary)** | Multi-parent absorption reduction at equal sparsity & fidelity | DR-SAE cuts mean MPAbs by $\ge 30\%$ vs Matryoshka at matched native $L_0$, and by $\ge 15\%$ at matched effective $L_0$, with $\Delta\text{FVU} \le 0.01$. | $95\%$ CI of relative MPAbs reduction at matched effective $L_0$ includes $0$. | $4.2\%$ relative MPAbs reduction ($N=112$ cells) |
| **H3 (RQ3)** | Ground-truth structural accuracy & cross-seed stability | Synthetic exact parent-set $\text{F1} \ge 0.90$ at $\rho=0.25$. Cross-seed MNN edge reproducibility $\ge 60\%$ for DR-SAE vs $\le 40\%$ for AG-SAE-lite. | Cross-seed edge reproducibility of DR-SAE is not strictly above AG-SAE-lite. | $5.1\%$ edge reproducibility |
| **H4 (RQ4)** | Downstream Sparse Feature Circuit compactness | At faithfulness $0.80$ on Pythia-70M subject-verb agreement, DR-SAE requires $\ge 20\%$ fewer feature nodes than BatchTopK and Matryoshka. | Node-count reduction is $< 5\%$ or faithfulness curves cross. | $4.5\%$ node reduction |

---

## 2. Validation-Tuned Hyperparameters (Frozen on Validation Split Only)

- **Candidate Parents per Latent ($m$):** $8$ (selected by co-activation lift $\Pr(p \mid c) \cdot \max(0, \langle r_p, d_{\text{eff}, c} \rangle)$)
- **Routing Truncation Depth ($H$):** $4$ hops (swept over $\{1, 2, 4, 8\}$ in ablations)
- **Structure Refresher Interval:** Every $2,000$ training steps (after $10\%$ native-code warmup)
- **Edge Sparsity Penalty ($\lambda_A$):** $3 \times 10^{-4}$ (tuned over $\{10^{-4}, 3\times 10^{-4}, 10^{-3}\}$ on validation split)
- **Hierarchical Orthogonality Weight ($\lambda_\perp$):** $0.05$ in whitened activation space (tuned over $\{0.01, 0.05, 0.10\}$)
- **AuxK Coefficient ($\alpha$):** $1/32 = 0.03125$
- **Graph Export Threshold:** $A_{p,c} \ge 0.05$
- **Random Seeds:** $\{42, 137, 2026\}$

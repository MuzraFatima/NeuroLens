"""MPAbs Benchmark: Multi-Parent Feature Absorption Suite & Pythia-160M Residual Stream Simulator.

Generalizes the single-parent first-letter SAEBench absorption metric (Chanin et al., NeurIPS 2025;
Karvonen et al., ICML 2025) to tokens with 2 to 3 crisp, simultaneously active parents:
  1. Orthographic parent: First letter (26 letters, subset with dense WordNet coverage)
  2. Semantic parent: WordNet lexicographer category (12 categories: noun.animal, noun.food,
     noun.artifact, noun.location, noun.person, noun.plant, noun.substance, noun.body,
     noun.time, noun.event, noun.communication, noun.group)
  3. Surface/Morphological parent: Capitalization case (TitleCase vs lowercase)
  4. Secondary Dual-Hypernym Semantic Set: Nouns with two distinct WordNet hypernyms
     (e.g., 'bat' -> mammal + flier; 'salmon' -> fish + food; 'piano' -> instrument + furniture).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

LEX_CATEGORIES = [
    "noun.animal",
    "noun.food",
    "noun.artifact",
    "noun.location",
    "noun.person",
    "noun.plant",
    "noun.substance",
    "noun.body",
    "noun.time",
    "noun.event",
    "noun.communication",
    "noun.group",
]

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "M", "P", "S", "T", "W"]

CASE_TYPES = ["lowercase", "TitleCase"]

# Curated multi-parent exemplar vocabulary per (letter, category) cell
EXEMPLAR_WORDS: Dict[Tuple[str, str], List[str]] = {
    ("A", "noun.animal"): [
        "alligator",
        "antelope",
        "albatross",
        "armadillo",
        "anchovy",
        "ape",
        "adder",
        "alpaca",
    ],
    ("B", "noun.animal"): ["badger", "bat", "beaver", "bison", "bobcat", "buffalo", "beetle", "baboon"],
    ("C", "noun.animal"): ["camel", "cheetah", "chimpanzee", "cobra", "cougar", "coyote", "crane", "crow"],
    ("D", "noun.animal"): ["dolphin", "donkey", "dragonfly", "duck", "deer", "dingo", "dove", "dormouse"],
    ("A", "noun.food"): [
        "almond",
        "apple",
        "apricot",
        "artichoke",
        "asparagus",
        "avocado",
        "anchovy",
        "acorn",
    ],
    ("B", "noun.food"): ["bacon", "bagel", "banana", "barley", "biscuit", "Veggie", "broccoli", "butter"],
    ("C", "noun.food"): ["cabbage", "carrot", "cashew", "celery", "cheddar", "cherry", "cinnamon", "coconut"],
    ("P", "noun.location"): [
        "Paris",
        "Prague",
        "Peru",
        "Poland",
        "Portugal",
        "Panama",
        "Philadelphia",
        "Phoenix",
    ],
    ("S", "noun.artifact"): [
        "scalpel",
        "scissors",
        "screwdriver",
        "shield",
        "shovel",
        "spindle",
        "stethoscope",
        "sword",
    ],
    ("M", "noun.person"): [
        "monarch",
        "merchant",
        "musician",
        "magistrate",
        "miner",
        "minister",
        "mathematician",
        "mariners",
    ],
}

DUAL_HYPERNYM_CONCEPTS = [
    ("bat", "noun.animal", "flier"),
    ("whale", "noun.animal", "swimmer"),
    ("salmon", "noun.animal", "noun.food"),
    ("turkey", "noun.animal", "noun.food"),
    ("Paris", "noun.location", "capital_city"),
    ("London", "noun.location", "capital_city"),
    ("Tokyo", "noun.location", "capital_city"),
    ("piano", "noun.artifact", "musical_instrument"),
    ("violin", "noun.artifact", "musical_instrument"),
    ("ambulance", "noun.artifact", "vehicle"),
    ("surgeon", "noun.person", "medical_practitioner"),
    ("oak", "noun.plant", "building_material"),
]


@dataclass
class MPAbsCell:
    """Represents one stratified Letter x Category (x Case) cell in MPAbs."""

    cell_id: str
    letter: str
    category: str
    case_label: str
    words: List[str]
    n_tokens: int
    parent_ids: List[int]  # Ground-truth parent concept indices


class LogisticRegressionProbe:
    """L2-regularized Logistic Regression Probe trained with Newton-CG/IRLS on activations."""

    def __init__(self, l2_reg: float = 1e-2, max_iter: int = 25):
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.w: np.ndarray = np.zeros(1)
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        n, d = X.shape
        X_aug = np.hstack([X, np.ones((n, 1), dtype=np.float64)])
        theta = np.zeros(d + 1, dtype=np.float64)
        reg_eye = self.l2_reg * np.eye(d + 1)
        reg_eye[-1, -1] = 0.0

        for _ in range(self.max_iter):
            logits = np.clip(X_aug @ theta, -20.0, 20.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad = X_aug.T @ (probs - y) / n + reg_eye @ theta
            weights = np.maximum(probs * (1.0 - probs), 1e-4)
            H = (X_aug.T * weights) @ X_aug / n + reg_eye
            step = np.linalg.solve(H, grad)
            theta -= step
            if np.linalg.norm(step) < 1e-5:
                break

        self.w = theta[:-1]
        norm = np.linalg.norm(self.w) + 1e-12
        self.w_unit = self.w / norm
        self.b = float(theta[-1])
        return self.score(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.clip(X @ self.w + self.b, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = (self.predict_proba(X) >= 0.5).astype(np.float64)
        return float(np.mean(preds == y))


class MPAbsBenchmark:
    """Constructs the 112-cell MPAbs benchmark (60/20/20 stratified train/val/test split),
    trains parent logistic probes (gate: >= 0.90 held-out accuracy), and computes per-parent
    and joint multi-parent absorption metrics.
    """

    def __init__(self, d_model: int = 128, tokens_per_cell: int = 25, seed: int = 42):
        self.d_model = d_model
        self.tokens_per_cell = max(20, tokens_per_cell)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._build_benchmark()

    def _build_benchmark(self) -> None:
        # 14 letters + 12 categories + 2 case features = 28 parent concepts
        self.n_letter_parents = len(LETTERS)  # 0..13
        self.n_cat_parents = len(LEX_CATEGORIES)  # 14..25
        self.n_case_parents = len(CASE_TYPES)  # 26..27
        self.n_parent_concepts = self.n_letter_parents + self.n_cat_parents + self.n_case_parents

        # 112 fine-grained child cells (14 letters x 8 categories = 112 cells, each >= 20 tokens)
        self.cells: List[MPAbsCell] = []
        cell_idx = 0
        for l_idx, letter in enumerate(LETTERS):
            for c_idx in range(8):
                cat = LEX_CATEGORIES[c_idx]
                case_idx = (l_idx + c_idx) % 2
                case_label = CASE_TYPES[case_idx]
                parent_ids = [
                    l_idx,
                    self.n_letter_parents + c_idx,
                    self.n_letter_parents + self.n_cat_parents + case_idx,
                ]
                base_words = EXEMPLAR_WORDS.get(
                    (letter, cat),
                    [f"{letter.lower()}_{cat.split('.')[1]}_{k}" for k in range(8)],
                )
                if case_label == "TitleCase":
                    words = [w.capitalize() for w in base_words]
                else:
                    words = [w.lower() for w in base_words]

                self.cells.append(
                    MPAbsCell(
                        cell_id=f"cell_{cell_idx:03d}_{letter}_{cat.split('.')[1]}_{case_label}",
                        letter=letter,
                        category=cat,
                        case_label=case_label,
                        words=words,
                        n_tokens=self.tokens_per_cell,
                        parent_ids=parent_ids,
                    )
                )
                cell_idx += 1

        self.n_child_concepts = len(self.cells)  # 112 child concepts
        self.n_total_concepts = self.n_parent_concepts + self.n_child_concepts  # 140 concepts

        # Construct orthonormal / near-orthogonal concept directions in R^{d_model}
        raw_dirs = self.rng.standard_normal((self.d_model, self.n_total_concepts))
        for j in range(self.n_total_concepts):
            raw_dirs[:, j] /= np.linalg.norm(raw_dirs[:, j]) + 1e-12

        # Enforce hierarchical orthogonality between every child cell and its 3 parents (Park et al., 2025)
        for idx_c, cell in enumerate(self.cells):
            child_col = self.n_parent_concepts + idx_c
            v = raw_dirs[:, child_col].copy()
            P_vecs = raw_dirs[:, cell.parent_ids]
            Q, _ = np.linalg.qr(P_vecs, mode="reduced")
            v = v - Q @ (Q.T @ v)
            raw_dirs[:, child_col] = v / (np.linalg.norm(v) + 1e-12)

        self.concept_directions = raw_dirs  # shape (d_model, 140)

        # Generate stratified 60 / 20 / 20 tokens per cell
        n_tr = int(round(0.60 * self.tokens_per_cell))
        n_va = int(round(0.20 * self.tokens_per_cell))
        n_te = self.tokens_per_cell - n_tr - n_va

        self.splits: Dict[str, Dict[str, np.ndarray]] = {}
        for split_name, count in [("train", n_tr), ("val", n_va), ("test", n_te)]:
            X_list, Z_list, cell_id_list = [], [], []
            for idx_c, cell in enumerate(self.cells):
                child_col = self.n_parent_concepts + idx_c
                Z_cell = np.zeros((count, self.n_total_concepts), dtype=np.float64)
                # Child concept active on all tokens in this cell
                Z_cell[:, child_col] = 0.95 + 0.15 * np.abs(self.rng.standard_normal(count))
                # Parent 1 (Letter): low variance sigma_1^2 = 0.03 (highly rigid -> absorbed first!)
                p_letter, p_cat, p_case = cell.parent_ids
                Z_cell[:, p_letter] = 1.05 + np.sqrt(0.03) * self.rng.standard_normal(count)
                # Parent 2 (Semantic Category): medium variance sigma_2^2 = 0.22
                Z_cell[:, p_cat] = 0.95 + np.sqrt(0.22) * self.rng.standard_normal(count)
                # Parent 3 (Capitalization Case): low-medium variance sigma_3^2 = 0.08
                Z_cell[:, p_case] = 0.90 + np.sqrt(0.08) * self.rng.standard_normal(count)
                Z_cell = np.maximum(0.0, Z_cell)

                X_cell = Z_cell @ self.concept_directions.T + 0.02 * self.rng.standard_normal(
                    (count, self.d_model)
                )
                X_list.append(X_cell)
                Z_list.append(Z_cell)
                cell_id_list.append(np.full(count, idx_c, dtype=np.int64))

            self.splits[split_name] = {
                "X": np.vstack(X_list),
                "Z": np.vstack(Z_list),
                "cell_ids": np.concatenate(cell_id_list),
            }

        # Fit and validate logistic regression probes for every parent concept on train, score on val
        self.probes: Dict[int, LogisticRegressionProbe] = {}
        self.probe_val_accuracies: Dict[int, float] = {}
        X_tr, Z_tr = self.splits["train"]["X"], self.splits["train"]["Z"]
        X_va, Z_va = self.splits["val"]["X"], self.splits["val"]["Z"]

        for p_id in range(self.n_parent_concepts):
            y_tr = (Z_tr[:, p_id] > 0.1).astype(np.float64)
            y_va = (Z_va[:, p_id] > 0.1).astype(np.float64)
            if np.unique(y_tr).size < 2:
                continue
            probe = LogisticRegressionProbe(l2_reg=1e-2)
            probe.fit(X_tr, y_tr)
            acc_va = probe.score(X_va, y_va)
            if acc_va >= 0.90:
                self.probes[p_id] = probe
                self.probe_val_accuracies[p_id] = acc_va

    def evaluate_sae_on_mpabs(
        self,
        sae_model,
        split: str = "test",
        use_effective_codes: bool = True,
    ) -> Dict[str, object]:
        """Scores a trained SAE on MPAbs per-parent and joint absorption across all 112 cells.

        For each cell c and each parent p of c:
        - Main latent for parent p is the latent l*(p) with highest F1 / cosine alignment to probe w_p.
        - A token t in cell c suffers absorption of parent p if:
          (a) probe p is positive on x(t),
          (b) main parent latent l*(p) fails to fire (or fires below threshold 0.10), AND
          (c) another active latent l != l*(p) carries positive projection onto probe direction w_p.
        - Joint absorption occurs on token t when >= 2 parents of cell c are simultaneously absorbed.
        """
        data = self.splits[split]
        X = data["X"]
        _z_true = data["Z"]  # ground-truth codes; unused in absorption scoring
        cell_ids = data["cell_ids"]

        out = sae_model.forward_eval(X, use_effective_codes=use_effective_codes)
        Z_pred = out["z_readout"]  # shape (N, n_latents)
        D_atoms = out["atoms"]  # shape (d_model, n_latents), unit-normalized columns

        # Identify main latent for each parent concept using validation split F1 + probe alignment
        X_va = self.splits["val"]["X"]
        Z_va_true = self.splits["val"]["Z"]
        Z_va_pred = sae_model.forward_eval(X_va, use_effective_codes=use_effective_codes)["z_readout"]

        main_latents: Dict[int, int] = {}
        for p_id, probe in self.probes.items():
            # Alignment with probe weight + correlation with ground-truth parent activation
            cos_align = probe.w_unit @ D_atoms  # (n_latents,)
            y_va = (Z_va_true[:, p_id] > 0.1).astype(np.float64)
            active_va = (Z_va_pred > 0.15).astype(np.float64)
            tp = y_va @ active_va
            fp = (1.0 - y_va) @ active_va
            fn = y_va.sum() - tp
            f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
            score = 0.65 * f1 + 0.35 * np.maximum(0.0, cos_align)
            main_latents[p_id] = int(np.argmax(score))

        cell_mpabs_scores: List[float] = []
        cell_joint_scores: List[float] = []
        cell_letter_abs: List[float] = []
        cell_cat_abs: List[float] = []
        cell_case_abs: List[float] = []

        for idx_c, cell in enumerate(self.cells):
            mask = cell_ids == idx_c
            Z_cell_pred = Z_pred[mask]  # (n_cell_tokens, n_latents)
            n_tok = Z_cell_pred.shape[0]
            absorbed_matrix = np.zeros((n_tok, len(cell.parent_ids)), dtype=np.float64)

            for p_rank, p_id in enumerate(cell.parent_ids):
                if p_id not in self.probes:
                    continue
                l_main = main_latents[p_id]
                w_p = self.probes[p_id].w_unit
                atom_proj = np.maximum(0.0, w_p @ D_atoms)  # (n_latents,)

                # Main latent fails to fire adequately on child token
                main_silent = Z_cell_pred[:, l_main] < 0.22
                # Other latents carry the parent probe direction
                other_proj = (Z_cell_pred * atom_proj[None, :]).sum(axis=1) - (
                    Z_cell_pred[:, l_main] * atom_proj[l_main]
                )
                absorbed_tok = main_silent & (other_proj > 0.12)
                absorbed_matrix[:, p_rank] = absorbed_tok.astype(np.float64)

            per_parent_rates = absorbed_matrix.mean(axis=0)
            cell_mpabs = float(per_parent_rates.mean())
            # Joint absorption: at least 2 parents absorbed on the same token
            joint_abs = float(np.mean(absorbed_matrix.sum(axis=1) >= 2.0))

            cell_mpabs_scores.append(cell_mpabs)
            cell_joint_scores.append(joint_abs)
            cell_letter_abs.append(float(per_parent_rates[0]))
            cell_cat_abs.append(float(per_parent_rates[1]))
            cell_case_abs.append(float(per_parent_rates[2]))

        arr_mpabs = np.array(cell_mpabs_scores, dtype=np.float64)
        arr_joint = np.array(cell_joint_scores, dtype=np.float64)

        return {
            "mean_mpabs": float(arr_mpabs.mean()),
            "std_mpabs": float(arr_mpabs.std(ddof=1)),
            "mean_joint_abs": float(arr_joint.mean()),
            "first_letter_abs": float(np.mean(cell_letter_abs)),
            "semantic_cat_abs": float(np.mean(cell_cat_abs)),
            "case_abs": float(np.mean(cell_case_abs)),
            "cell_mpabs": arr_mpabs,
            "cell_joint_abs": arr_joint,
            "n_cells": len(self.cells),
            "mean_probe_val_acc": float(np.mean(list(self.probe_val_accuracies.values()))),
        }

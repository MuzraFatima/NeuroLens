"""MPAbs on a real language model: multi-parent absorption for single-token WordNet nouns.

Every token carries three parents at once:
  1. its first letter (orthographic),
  2. the WordNet lexicographer category of its most frequent noun sense (semantic),
  3. its case (lowercase " dog" vs TitleCase " Dog").

Cells are letter x category groups with at least ``min_tokens`` tokens. Words are split 60/20/20
into train/val/test, stratified by cell, with both case variants of a word kept in the same split.

For each parent a logistic-regression probe is trained on the model's residual activations at the
token (train split) and kept only if its balanced accuracy on the val split is at least 0.90.

Absorption of parent p on a token t (generalising the SAEBench first-letter metric):
  - the probe for p fires on t,
  - the SAE's main latent for p (best F1 against the label on train) is silent on t,
  - and some other active latent carries the probe direction: its contribution
    ``z_l * <r_l, w_p>`` is at least ``frac`` of the token's own probe projection ``<x, w_p>``.
MPAbs per cell is the mean over its parents of the per-parent absorption rate; joint absorption is
the share of tokens on which two or more parents are absorbed at once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

CATEGORIES = [
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
    "noun.act",
    "noun.state",
    "noun.attribute",
]

TEMPLATES = [
    "Here is a word:{w}",
    "I looked at the{w}",
    "My favourite one is the{w}",
]


def _wordnet():
    import nltk

    root = os.environ.get("NLTK_DATA")
    if root:
        nltk.data.path.insert(0, root)
    try:
        from nltk.corpus import wordnet as wn

        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", download_dir=root, quiet=True)
        from nltk.corpus import wordnet as wn
    return wn


def build_vocab(tokenizer, min_len: int = 3, max_words: Optional[int] = None) -> List[Dict[str, str]]:
    """Single-token nouns whose lowercase and TitleCase forms (with a leading space) are both one token."""
    wn = _wordnet()
    vocab = tokenizer.get_vocab()
    seen, rows = set(), []
    for tok in vocab:
        if not tok.startswith("Ġ"):  # GPT-NeoX BPE marks a leading space with this character
            continue
        w = tok[1:]
        if not (w.isalpha() and w.islower() and len(w) >= min_len) or w in seen:
            continue
        title = "Ġ" + w.capitalize()
        if title not in vocab:
            continue
        nouns = wn.synsets(w, pos="n")
        others = sum(len(wn.synsets(w, pos=p)) for p in ("v", "a", "s", "r"))
        if not nouns or len(nouns) < others:
            continue
        cat = nouns[0].lexname()
        if cat not in CATEGORIES:
            continue
        seen.add(w)
        rows.append({"word": w, "letter": w[0].upper(), "category": cat})
    rows.sort(key=lambda r: r["word"])
    return rows[:max_words] if max_words else rows


@dataclass
class MPAbsLM:
    tokens: List[Dict[str, object]]  # one row per (word, case)
    X: np.ndarray  # (n_tokens, n_templates, d_model) residual activations
    cells: List[str]
    parents: List[str]
    splits: Dict[str, np.ndarray]
    probes: Dict[str, Dict[str, object]] = field(default_factory=dict)

    @property
    def labels(self) -> np.ndarray:
        """(n_tokens, n_parents) binary parent labels."""
        Y = np.zeros((len(self.tokens), len(self.parents)), dtype=bool)
        idx = {p: i for i, p in enumerate(self.parents)}
        for t, row in enumerate(self.tokens):
            for p in row["parents"]:
                Y[t, idx[p]] = True
        return Y


def build_benchmark(
    tokenizer,
    lm,
    layer: int,
    min_tokens: int = 20,
    seed: int = 0,
    batch: int = 64,
) -> MPAbsLM:
    rng = np.random.default_rng(seed)
    vocab = build_vocab(tokenizer)
    by_cell: Dict[str, List[Dict[str, str]]] = {}
    for r in vocab:
        by_cell.setdefault(f"{r['letter']}|{r['category']}", []).append(r)
    cells = sorted(c for c, rs in by_cell.items() if 2 * len(rs) >= min_tokens)

    tokens, split_of = [], {}
    for c in cells:
        words = [r["word"] for r in by_cell[c]]
        rng.shuffle(words)
        n = len(words)
        n_tr, n_va = int(round(0.6 * n)), int(round(0.2 * n))
        for i, w in enumerate(words):
            split_of[w] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        for r in by_cell[c]:
            for case in ("lowercase", "TitleCase"):
                surface = r["word"] if case == "lowercase" else r["word"].capitalize()
                tokens.append(
                    {
                        "word": r["word"],
                        "surface": surface,
                        "cell": c,
                        "letter": r["letter"],
                        "category": r["category"],
                        "case": case,
                        "parents": [f"letter:{r['letter']}", f"cat:{r['category']}", f"case:{case}"],
                        "split": split_of[r["word"]],
                    }
                )
    parents = sorted({p for t in tokens for p in t["parents"]})

    # Residual activations at the target token, one row per template.
    block_out: Dict[str, torch.Tensor] = {}
    prompts = [tpl.format(w=" " + t["surface"]) for t in tokens for tpl in TEMPLATES]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    acts = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            enc = tokenizer(prompts[i : i + batch], return_tensors="pt", padding=True)
            hs = lm(**enc, output_hidden_states=True).hidden_states[layer + 1]
            acts.append(hs[:, -1, :].float().numpy())
    del block_out
    X = np.concatenate(acts).reshape(len(tokens), len(TEMPLATES), -1)
    splits = {
        s: np.array([i for i, t in enumerate(tokens) if t["split"] == s]) for s in ("train", "val", "test")
    }
    return MPAbsLM(tokens=tokens, X=X, cells=cells, parents=parents, splits=splits)


def fit_probes(bench: MPAbsLM, scale: float, min_bal_acc: float = 0.90) -> Dict[str, Dict[str, object]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    Y = bench.labels
    n_t = bench.X.shape[1]
    Xtr = bench.X[bench.splits["train"]].reshape(-1, bench.X.shape[2]) * scale
    Xva = bench.X[bench.splits["val"]].reshape(-1, bench.X.shape[2]) * scale
    probes = {}
    for j, p in enumerate(bench.parents):
        ytr = np.repeat(Y[bench.splits["train"], j], n_t)
        yva = np.repeat(Y[bench.splits["val"], j], n_t)
        if ytr.sum() < 5 or yva.sum() < 2:
            probes[p] = {"kept": False, "reason": "too few positives"}
            continue
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        clf.fit(Xtr, ytr)
        bacc = float(balanced_accuracy_score(yva, clf.predict(Xva)))
        w = clf.coef_[0].astype(np.float32)
        probes[p] = {
            "kept": bacc >= min_bal_acc,
            "val_balanced_acc": bacc,
            "w": w,
            "w_unit": w / (np.linalg.norm(w) + 1e-9),
            "b": float(clf.intercept_[0]),
        }
    bench.probes = probes
    return probes


@torch.no_grad()
def score_sae(
    bench: MPAbsLM, model, scale: float, split: str = "test", frac: float = 0.4
) -> Dict[str, object]:
    """Per-parent and joint absorption for one SAE (see module docstring)."""
    Y = bench.labels
    n_t = bench.X.shape[1]
    pidx = {p: j for j, p in enumerate(bench.parents)}
    kept = [p for p in bench.parents if bench.probes.get(p, {}).get("kept")]

    def codes(ix):
        x = torch.from_numpy(bench.X[ix].reshape(-1, bench.X.shape[2]) * scale)
        out = model(x)
        return x.numpy(), out["z_eff"].numpy()

    R = torch.nn.functional.normalize(model.R.detach(), dim=0).numpy()
    Xtr, Ztr = codes(bench.splits["train"])
    main = {}
    for p in kept:
        y = np.repeat(Y[bench.splits["train"], pidx[p]], n_t)
        act = Ztr > 0
        tp = (act & y[:, None]).sum(0)
        fp = (act & ~y[:, None]).sum(0)
        fn = y.sum() - tp
        main[p] = int(np.argmax(2 * tp / (2 * tp + fp + fn + 1e-9)))

    ix = bench.splits[split]
    X, Z = codes(ix)
    toks = [bench.tokens[i] for i in ix for _ in range(n_t)]
    absorbed = np.zeros((len(toks), 3), dtype=bool)
    eligible = np.zeros((len(toks), 3), dtype=bool)
    silent = np.zeros((len(toks), 3), dtype=bool)
    for slot in range(3):
        for p in kept:
            rows = np.array([i for i, t in enumerate(toks) if t["parents"][slot] == p])
            if rows.size == 0:
                continue
            pr = bench.probes[p]
            proj = X[rows] @ pr["w_unit"]
            fires = (X[rows] @ pr["w"] + pr["b"]) > 0
            contrib = Z[rows] * (R.T @ pr["w_unit"])[None, :]
            contrib[:, main[p]] = 0
            other = contrib.max(1) >= frac * np.maximum(proj, 1e-6)
            main_off = Z[rows, main[p]] <= 0
            eligible[rows, slot] = fires
            silent[rows, slot] = fires & main_off
            absorbed[rows, slot] = fires & main_off & other

    cell_of = np.array([t["cell"] for t in toks])
    per_cell, joint_cell, fn_cell = {}, {}, {}
    for c in bench.cells:
        m = cell_of == c
        rates = [absorbed[m, s][eligible[m, s]].mean() for s in range(3) if eligible[m, s].any()]
        fns = [silent[m, s][eligible[m, s]].mean() for s in range(3) if eligible[m, s].any()]
        if rates:
            per_cell[c] = float(np.mean(rates))
            joint_cell[c] = float((absorbed[m].sum(1) >= 2).mean())
            fn_cell[c] = float(np.mean(fns))
    slot_names = ["letter", "category", "case"]
    return {
        "mpabs_mean": float(np.mean(list(per_cell.values()))),
        "joint_abs_mean": float(np.mean(list(joint_cell.values()))),
        "per_parent": {
            slot_names[s]: float(absorbed[:, s][eligible[:, s]].mean())
            if eligible[:, s].any()
            else float("nan")
            for s in range(3)
        },
        "parent_false_negative": {
            slot_names[s]: float(silent[:, s][eligible[:, s]].mean())
            if eligible[:, s].any()
            else float("nan")
            for s in range(3)
        },
        "per_cell": per_cell,
        "per_cell_false_negative": fn_cell,
        "false_negative_mean": float(np.mean(list(fn_cell.values()))) if fn_cell else float("nan"),
        "n_cells_scored": len(per_cell),
        "main_latents": main,
    }


@torch.no_grad()
def causal_readout_check(
    bench: MPAbsLM, model, scale: float, main: Dict[str, int], seed: int = 0
) -> Dict[str, float]:
    """Removes a parent's main-latent contribution from the reconstruction on test tokens and measures
    the drop in that parent's probe logit, against removing a random latent at the same magnitude."""
    rng = np.random.default_rng(seed)
    ix = bench.splits["test"]
    n_t = bench.X.shape[1]
    x = torch.from_numpy(bench.X[ix].reshape(-1, bench.X.shape[2]) * scale)
    out = model(x)
    Z, xh = out["z_eff"].numpy(), out["x_hat"].numpy()
    R = model.R.detach().numpy()
    toks = [bench.tokens[i] for i in ix for _ in range(n_t)]
    drops, ctrl = [], []
    for p, lat in main.items():
        rows = np.array([i for i, t in enumerate(toks) if p in t["parents"]])
        if rows.size == 0:
            continue
        pr = bench.probes[p]
        base = xh[rows] @ pr["w"]
        drops.append(float(np.mean(Z[rows, lat] * (R[:, lat] @ pr["w"]))))
        r = int(rng.integers(R.shape[1]))
        ctrl.append(float(np.mean(Z[rows, lat] * (R[:, r] @ pr["w"]))))
        del base
    return {
        "parent_logit_drop": float(np.mean(drops)),
        "random_control_drop": float(np.mean(ctrl)),
        "selectivity": float(np.mean(drops) / max(1e-6, abs(np.mean(ctrl)))),
    }


def cells_table(
    bench: MPAbsLM, results: Dict[str, Dict[str, object]], methods: Sequence[str]
) -> List[Dict[str, object]]:
    rows = []
    for c in bench.cells:
        letter, cat = c.split("|")
        n = sum(1 for t in bench.tokens if t["cell"] == c)
        row = {"cell": c, "letter": letter, "category": cat, "n_tokens": n}
        for m in methods:
            row[m] = results[m]["per_cell"].get(c)
        rows.append(row)
    return rows

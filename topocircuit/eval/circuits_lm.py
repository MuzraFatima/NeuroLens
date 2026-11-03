"""Feature-circuit size on Pythia-70M subject-verb agreement (RQ4, single-layer CPU version).

For each task the metric is m = logit(correct verb) - logit(wrong verb) at the last position.
The SAE is spliced into the residual stream after one block, with its error term kept (as in Marks
et al.), so the spliced model is exactly the original model. Each latent gets an attribution score
from one backward pass (attribution patching against zero ablation):

    I_l = sum over positions of z_l * <r_l, dm/dh> / scale

Latents are ranked by mean |I_l| over examples. The circuit C_N keeps the top-N latents and the
error term; every other latent is set to zero. Faithfulness(N) = (m(C_N) - m(empty)) / (m(full) -
m(empty)), averaged over examples. The headline number is the node count where faithfulness first
reaches 0.8 (linear interpolation between budgets).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

NOUNS: List[Tuple[str, str]] = [
    ("key", "keys"),
    ("dog", "dogs"),
    ("cat", "cats"),
    ("car", "cars"),
    ("book", "books"),
    ("girl", "girls"),
    ("boy", "boys"),
    ("teacher", "teachers"),
    ("doctor", "doctors"),
    ("house", "houses"),
    ("tree", "trees"),
    ("bird", "birds"),
    ("friend", "friends"),
    ("student", "students"),
    ("player", "players"),
    ("table", "tables"),
    ("window", "windows"),
    ("farmer", "farmers"),
    ("writer", "writers"),
    ("song", "songs"),
    ("road", "roads"),
    ("river", "rivers"),
    ("garden", "gardens"),
    ("picture", "pictures"),
]
DISTRACTORS: List[Tuple[str, str]] = [
    ("cabinet", "cabinets"),
    ("man", "men"),
    ("woman", "women"),
    ("city", "cities"),
    ("school", "schools"),
    ("box", "boxes"),
]
TASKS = {
    "simple": "The {n}",
    "across_pp": "The {n} near the {d}",
    "across_rc": "The {n} that the {d} liked",
}
VERBS = (" is", " are")


def make_task(tokenizer, task: str) -> Dict[str, torch.Tensor]:
    tpl = TASKS[task]
    prompts, correct = [], []
    for i, (sg, pl) in enumerate(NOUNS):
        d_sg, d_pl = DISTRACTORS[i % len(DISTRACTORS)]
        for number, noun in ((0, sg), (1, pl)):
            dist = d_pl if number == 0 else d_sg  # distractor always has the opposite number
            prompts.append(tpl.format(n=noun, d=dist))
            correct.append(number)
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    v = [tokenizer(x)["input_ids"][0] for x in VERBS]
    return {"enc": enc, "correct": torch.tensor(correct), "verb_ids": torch.tensor(v)}


class Splicer:
    """Forward hook that replaces a block's output with SAE(h) restricted to a latent mask + error."""

    def __init__(self, lm, layer: int, sae, scale: float):
        self.sae, self.scale = sae, scale
        self.mask = None  # None = clean model; tensor (width,) of 0/1 = circuit
        self.capture = False
        self.h = None
        self.handle = lm.gpt_neox.layers[layer].register_forward_hook(self.hook)

    def hook(self, _mod, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.capture:
            h.retain_grad()
            self.h = h
        if self.mask is None:
            return output
        x = h * self.scale
        out = self.sae(x.reshape(-1, x.shape[-1]))
        z = out["z_eff"]
        x_hat = out["x_hat"]
        err = x.reshape(-1, x.shape[-1]) - x_hat
        x_c = (z * self.mask) @ self.sae.R.t() + self.sae.b_dec + err
        h2 = (x_c / self.scale).reshape(h.shape)
        h2[:, 0] = h[:, 0]
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    def remove(self):
        self.handle.remove()


def _metric(lm, data) -> torch.Tensor:
    logits = lm(**data["enc"]).logits[:, -1, :]
    lv = logits[:, data["verb_ids"]]
    sign = 1 - 2 * data["correct"].float()  # +1 if singular is correct
    return sign * (lv[:, 0] - lv[:, 1])


def evaluate_circuits(
    lm,
    tokenizer,
    sae,
    layer: int,
    scale: float,
    budgets: Sequence[int] = (5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500),
) -> Dict[str, object]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sp = Splicer(lm, layer, sae, scale)
    width = sae.cfg.width
    out: Dict[str, object] = {}
    try:
        for task in TASKS:
            data = make_task(tokenizer, task)
            # Attribution (one backward pass on the clean model).
            sp.capture, sp.mask = True, None
            with torch.enable_grad():
                m = _metric(lm, data)
                m.sum().backward()
            h, g = sp.h.detach().float(), sp.h.grad.detach().float()
            sp.capture = False
            lm.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = sae(h[:, 1:].reshape(-1, h.shape[-1]) * scale)["z_eff"]
                gr = g[:, 1:].reshape(-1, g.shape[-1])
                attr = z * (gr @ sae.R) / scale  # (B*T, width)
                attr = attr.reshape(h.shape[0], -1, width).sum(1).abs().mean(0)
                order = torch.argsort(attr, descending=True)
                sp.mask = None
                m_full = _metric(lm, data)
                sp.mask = torch.zeros(width)
                m_empty = _metric(lm, data)
                faith = []
                for n in budgets:
                    mask = torch.zeros(width)
                    mask[order[:n]] = 1.0
                    sp.mask = mask
                    m_c = _metric(lm, data)
                    faith.append(float(((m_c - m_empty).mean()) / ((m_full - m_empty).mean() + 1e-9)))
            f = np.array(faith)
            if f.max() < 0.8:
                n08 = float("nan")
            else:
                i = int(np.argmax(f >= 0.8))
                if i == 0:
                    n08 = float(budgets[0])
                else:
                    n08 = float(
                        budgets[i - 1]
                        + (0.8 - f[i - 1]) / (f[i] - f[i - 1] + 1e-9) * (budgets[i] - budgets[i - 1])
                    )
            out[task] = {
                "faithfulness": faith,
                "nodes_at_0.8": n08,
                "auc": float(np.trapezoid(np.clip(f, 0, 1), budgets) / (budgets[-1] - budgets[0])),
                "model_accuracy": float((m_full > 0).float().mean()),
                "m_full": float(m_full.mean()),
                "m_empty": float(m_empty.mean()),
            }
    finally:
        sp.remove()
    out["budgets"] = list(budgets)
    return out

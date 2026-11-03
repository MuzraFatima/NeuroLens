#!/usr/bin/env python3
"""Trains SAEs on cached Pythia-70M activations and scores fidelity on held-out tokens (CPU).

For every (method, seed) it writes:
    data/checkpoints/<run>.pt                (ignored by git)
    results/lm/<run>.json                    fidelity metrics + training curve

Fidelity metrics (held-out last 10% of tokens):
    fvu, native_l0, eff_l0, dead_frac
    loss_recovered: (CE_zero - CE_sae) / (CE_zero - CE_clean), splicing the SAE output into the LM
                    at the same layer on held-out contexts (zero-ablation baseline)

Usage:
    python scripts/train_lm_saes.py --methods BatchTopK Matryoshka --seeds 42
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from topocircuit.models import DRSAE, MatryoshkaSAE, SAEConfig, train_method  # noqa: E402
from topocircuit.utils import load_yaml_config  # noqa: E402


class ScaledArray:
    """Read-only view of a float16 memmap, returned as float32 and scaled to unit mean squared norm."""

    def __init__(self, arr: np.ndarray, scale: float, start: int = 0, stop: int = None):
        self.arr, self.scale, self.start = arr, scale, start
        self.stop = arr.shape[0] if stop is None else stop
        self.shape = (self.stop - self.start, arr.shape[1])

    def __getitem__(self, idx):
        return (np.asarray(self.arr[self.start + np.asarray(idx)], dtype=np.float32)) * self.scale


def n_filled_chunks(cfg) -> int:
    """Chunks actually written by extract_activations.py (it fills them in order)."""
    tag = cfg["language_model"]["name"].split("/")[-1]
    ids = np.load(os.path.join(ROOT, cfg["activations"]["cache_dir"], f"{tag}_tokens.npy"), mmap_mode="r")
    filled = np.asarray(ids.any(1))
    return int(filled.sum()) if filled.all() else int(np.argmin(filled))


def load_acts(cfg, layer: int):
    tag = cfg["language_model"]["name"].split("/")[-1]
    path = os.path.join(ROOT, cfg["activations"]["cache_dir"], f"{tag}_L{layer}.npy")
    arr = np.load(path, mmap_mode="r")
    # Load into RAM (about 0.9 GB as float16): random minibatch reads from a memmap are disk-bound.
    arr = np.ascontiguousarray(arr[: n_filled_chunks(cfg) * (cfg["language_model"]["context_length"] - 1)])
    sample = np.asarray(arr[:: max(1, arr.shape[0] // 20000)], dtype=np.float32)
    scale = float(np.sqrt(sample.shape[1] / (sample**2).sum(1).mean()))
    n_train = int(0.9 * arr.shape[0])
    return arr, scale, n_train


@torch.no_grad()
def fidelity(model, X_eval: np.ndarray) -> dict:
    x = torch.from_numpy(X_eval)
    out = model(x)
    z, ze, xh = out["z"], out["z_eff"], out["x_hat"]
    fvu = float(((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
    return {
        "fvu": fvu,
        "native_l0": float((z > 0).float().sum(1).mean()),
        "eff_l0": float((ze > 0).float().sum(1).mean()),
        "dead_frac": float(((ze > 0).sum(0) == 0).float().mean()),
    }


@torch.no_grad()
def loss_recovered(models: dict, cfg, layer: int, scale: float, n_ctx: int = 48) -> dict:
    """CE of the LM on held-out contexts with the layer output replaced by each SAE's reconstruction."""
    from transformers import AutoModelForCausalLM

    tag = cfg["language_model"]["name"].split("/")[-1]
    ids_all = np.load(os.path.join(ROOT, cfg["activations"]["cache_dir"], f"{tag}_tokens.npy"), mmap_mode="r")
    n_valid = n_filled_chunks(cfg)
    ids = torch.from_numpy(np.asarray(ids_all[n_valid - n_ctx : n_valid], dtype=np.int64))
    lm = AutoModelForCausalLM.from_pretrained(cfg["language_model"]["name"]).eval()
    block = lm.gpt_neox.layers[layer]
    state = {"fn": None}

    def hook(_mod, _inp, output):
        if state["fn"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h2 = h.clone()
        h2[:, 1:] = state["fn"](h[:, 1:])
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    handle = block.register_forward_hook(hook)

    def ce():
        logits = lm(ids).logits[:, 1:-1]
        return float(
            torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 2:].reshape(-1))
        )

    try:
        clean = ce()
        state["fn"] = lambda h: torch.zeros_like(h)
        zero = ce()
        res = {"ce_clean": clean, "ce_zero_ablation": zero}
        for name, m in models.items():

            def splice(h, m=m):
                shp = h.shape
                xh = m(h.reshape(-1, shp[-1]) * scale)["x_hat"] / scale
                return xh.reshape(shp)

            state["fn"] = splice
            ce_sae = ce()
            res[name] = {"ce_sae": ce_sae, "loss_recovered": (zero - ce_sae) / (zero - clean)}
    finally:
        handle.remove()
    return res


def run_name(method: str, seed: int, layer: int, width: int, k: int) -> str:
    slug = method.lower().replace(" ", "").replace("(", "").replace(")", "").replace(".", "").replace("=", "")
    return f"L{layer}_w{width}_k{k}_{slug}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "cpu_small.yaml"))
    ap.add_argument("--methods", nargs="+", default=["BatchTopK", "Matryoshka"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="default: epochs * n_train / batch")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lm"))
    ap.add_argument("--set", nargs="*", default=[], help="SAEConfig overrides, e.g. lambda_ortho=0")
    ap.add_argument("--tag", default="", help="suffix for run names (ablations)")
    ap.add_argument("--no-loss-recovered", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    sae_c, dr_c = cfg["sae"], cfg["dr_sae"]
    layer = cfg["language_model"]["main_layer"] if args.layer is None else args.layer
    width = args.width or sae_c["width"]
    k = args.k or sae_c["k"]
    seeds = args.seeds or sae_c["seeds"]
    arr, scale, n_train = load_acts(cfg, layer)
    X_train = ScaledArray(arr, scale, 0, n_train)
    X_eval = ScaledArray(arr, scale, n_train)[np.arange(min(20000, arr.shape[0] - n_train))]
    bs = int(sae_c["batch_size"])
    steps = args.steps or int(sae_c["epochs"] * n_train / bs)
    banks = [b * width // sae_c["width"] for b in sae_c["banks"]]
    overrides = {}
    defaults = SAEConfig(1, 1, 1)
    for kv in args.set:
        key, val = kv.split("=")
        cur = getattr(defaults, key)
        overrides[key] = (val == "True") if isinstance(cur, bool) else type(cur)(val)
    sae_kw = {
        "banks": banks,
        "m_candidates": dr_c["m_candidates"],
        "hop_depth": dr_c["hop_depth"],
        "lambda_a": dr_c["lambda_a"],
        "lambda_ortho": dr_c["lambda_ortho"],
        "alpha_auxk": dr_c["alpha_auxk"],
        "edge_threshold": dr_c["edge_threshold"],
        "hub_cap_frac": dr_c["hub_cap_frac"],
    }
    sae_kw.update(overrides)
    train_kw = {
        "batch_size": bs,
        "lr": float(sae_c["lr"]),
        "refresh_every": int(dr_c["refresh_every"]),
        "warmup_frac": float(dr_c["warmup_frac"]),
        "verbose": True,
        "log_every": 100,
    }
    os.makedirs(args.out, exist_ok=True)
    ck_dir = os.path.join(ROOT, "data", "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)

    for seed in seeds:
        base = None
        trained = {}
        for method in args.methods:
            name = run_name(method, seed, layer, width, k) + args.tag
            print(f"== {name} ({steps} steps)", flush=True)
            t0 = time.time()
            if method in ("Post-hoc DR", "AG-SAE-lite") and base is None:
                bpath = os.path.join(ck_dir, run_name("Matryoshka", seed, layer, width, k) + ".pt")
                base = MatryoshkaSAE(SAEConfig(d_in=arr.shape[1], width=width, k=k, seed=seed, **sae_kw))
                base.load_state_dict(torch.load(bpath))
                base.eval()
            model = train_method(
                method, X_train, arr.shape[1], width, k, seed, steps, base=base, sae_kw=sae_kw, **train_kw
            )
            if method == "Matryoshka":
                base = model
            secs = time.time() - t0
            torch.save(model.state_dict(), os.path.join(ck_dir, name + ".pt"))
            res = {
                "run": name,
                "method": method,
                "seed": seed,
                "layer": layer,
                "width": width,
                "k": k,
                "steps": steps,
                "train_seconds": round(secs, 1),
                "scale": scale,
                "overrides": overrides,
            }
            res.update(fidelity(model, X_eval))
            if isinstance(model, DRSAE):
                ps = model.parent_sets()
                res.update(
                    {
                        "n_edges": sum(len(v) for v in ps.values()),
                        "n_multi_parent": sum(1 for v in ps.values() if len(v) >= 2),
                        "longest_path": model.longest_path(),
                        "acyclic": model.is_acyclic(),
                    }
                )
            trained[name] = (model, res)
            print(json.dumps({k2: v for k2, v in res.items() if k2 != "overrides"}), flush=True)
        if not args.no_loss_recovered:
            lr_res = loss_recovered({n: m for n, (m, _) in trained.items()}, cfg, layer, scale)
            for n, (_, res) in trained.items():
                res.update(
                    {
                        "ce_clean": lr_res["ce_clean"],
                        "ce_zero_ablation": lr_res["ce_zero_ablation"],
                        **lr_res[n],
                    }
                )
        for n, (_, res) in trained.items():
            with open(os.path.join(args.out, n + ".json"), "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(
                f"saved {n}: fvu={res['fvu']:.3f} L0={res['native_l0']:.1f}/{res['eff_l0']:.1f} "
                f"dead={res['dead_frac']:.3f} lossrec={res.get('loss_recovered', float('nan')):.3f}"
            )


if __name__ == "__main__":
    main()

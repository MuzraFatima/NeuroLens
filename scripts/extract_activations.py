#!/usr/bin/env python3
"""Runs Pythia-70M over pile-10k once and caches residual-stream activations to disk (CPU).

Output (inside the project folder, ignored by git):
    data/activations/<model>_L<layer>.npy      float16, shape (n_tokens, d_model)
    data/activations/<model>_L<layer>.json     metadata (tokens, docs, layer, timing)

Usage:
    python scripts/extract_activations.py --config configs/cpu_small.yaml
    python scripts/extract_activations.py --layers 0 1 2 3 4 5 --n-tokens 500000   # circuit SAEs
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

from topocircuit.utils import load_yaml_config  # noqa: E402


def iter_chunks(tokenizer, dataset, ctx: int):
    """Yields token-id chunks of length ctx from consecutive documents."""
    buf = []
    for doc in dataset:
        buf.extend(tokenizer(doc["text"])["input_ids"] + [tokenizer.eos_token_id])
        while len(buf) >= ctx:
            yield buf[:ctx]
            buf = buf[ctx:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "cpu_small.yaml"))
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--n-tokens", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    lm = cfg["language_model"]
    layers = args.layers if args.layers is not None else [lm["main_layer"]]
    n_tokens = args.n_tokens or int(cfg["activations"]["n_tokens"])
    ctx = int(lm["context_length"])
    out_dir = os.path.join(ROOT, cfg["activations"]["cache_dir"])
    os.makedirs(out_dir, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(lm["name"])
    model = AutoModelForCausalLM.from_pretrained(lm["name"], torch_dtype=torch.float32).eval()
    ds = load_dataset(cfg["activations"]["dataset"], split="train")

    tag = lm["name"].split("/")[-1]
    # Position 0 of every chunk is dropped: its residual norm is an outlier (attention sink).
    per_chunk = ctx - 1
    n_chunks = (n_tokens + per_chunk - 1) // per_chunk
    n_tokens = n_chunks * per_chunk
    mms = {
        L: np.lib.format.open_memmap(
            os.path.join(out_dir, f"{tag}_L{L}.npy"),
            mode="w+",
            dtype=np.float16,
            shape=(n_tokens, lm["d_model"]),
        )
        for L in layers
    }
    ids_out = np.lib.format.open_memmap(
        os.path.join(out_dir, f"{tag}_tokens.npy"), mode="w+", dtype=np.int32, shape=(n_chunks, ctx)
    )

    t0 = time.time()
    chunks = iter_chunks(tok, ds, ctx)
    done = 0
    while done < n_chunks:
        batch = []
        for _ in range(min(args.batch, n_chunks - done)):
            batch.append(next(chunks))
        ids = torch.tensor(batch)
        hs = model(ids, output_hidden_states=True).hidden_states  # hs[L + 1] = resid_post of block L
        for L, mm in mms.items():
            h = hs[L + 1][:, 1:, :].reshape(-1, lm["d_model"]).numpy().astype(np.float16)
            mm[done * per_chunk : done * per_chunk + h.shape[0]] = h
        ids_out[done : done + len(batch)] = ids.numpy()
        done += len(batch)
        if done % (args.batch * 20) == 0 or done == n_chunks:
            rate = done * per_chunk / (time.time() - t0)
            print(f"{done * per_chunk:>9d}/{n_tokens} tokens  {rate:,.0f} tok/s", flush=True)

    for L, mm in mms.items():
        mm.flush()
        meta = {
            "model": lm["name"],
            "layer": L,
            "hook": f"blocks.{L}.hook_resid_post",
            "n_tokens": n_tokens,
            "d_model": lm["d_model"],
            "context_length": ctx,
            "dropped_position_0": True,
            "dataset": cfg["activations"]["dataset"],
            "seconds": round(time.time() - t0, 1),
            "mean_norm": float(np.linalg.norm(mm[:20000].astype(np.float32), axis=1).mean()),
        }
        with open(os.path.join(out_dir, f"{tag}_L{L}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(json.dumps(meta))


if __name__ == "__main__":
    main()

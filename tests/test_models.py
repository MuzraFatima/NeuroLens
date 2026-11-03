"""Tests for the PyTorch SAEs: sparsity, routing algebra, acyclicity and training."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from topocircuit.data.synth_generator import MixedTopologySyntheticGenerator  # noqa: E402
from topocircuit.models import (  # noqa: E402
    DRSAE,
    BatchTopKSAE,
    MatryoshkaSAE,
    SAEConfig,
    batch_topk,
    fit_ag_sae_lite,
    fit_post_hoc_dr,
    induce_parent_sets_greedy,
    train_sae,
)


@pytest.fixture(scope="module")
def synth():
    gen = MixedTopologySyntheticGenerator(n_features=64, d_model=32, rho=0.25, seed=0)
    X, Zn, Ze = gen.sample_batch(6000)
    return gen, X.astype(np.float32)


def cfg(**kw):
    base = {"d_in": 32, "width": 64, "k": 6, "seed": 0, "banks": [8, 16, 32, 64]}
    base.update(kw)
    return SAEConfig(**base)


def test_batch_topk_keeps_k_per_row_on_average():
    pre = torch.rand(16, 50)
    z = batch_topk(pre, 5)
    assert int((z > 0).sum()) == 80
    assert torch.all(z[z > 0] >= pre.flatten().sort(descending=True).values[79])


def test_drsae_is_matryoshka_before_first_refresh(synth):
    _, X = synth
    m = DRSAE(cfg())
    x = torch.from_numpy(X[:32])
    z = m.encode(x)
    assert torch.equal(m.route(z), z)
    assert m.parent_sets() == {}


def test_refresh_respects_frequency_order_hub_cap_and_acyclicity(synth):
    _, X = synth
    m = DRSAE(cfg(hub_cap_frac=0.05))
    train_sae(m, X, n_steps=60, batch_size=512, refresh_every=20, warmup_frac=0.3)
    ps = m.parent_sets()
    assert m.is_acyclic()
    cap = max(2, int(0.05 * 64))
    children = {}
    for c, parents in ps.items():
        assert len(parents) <= m.cfg.m_candidates
        for p in parents:
            assert int(m.rank[p]) < int(m.rank[c]), "edge must run from more to less frequent latent"
            children[p] = children.get(p, 0) + 1
    assert all(v <= cap for v in children.values())


def test_routing_series_equals_matrix_inverse_when_deep_enough():
    m = DRSAE(cfg(hop_depth=8))
    with torch.no_grad():
        # chain 0 -> 1 -> 2 and 0 -> 3, 4 -> 3 (a multi-parent child)
        for j, (p, c) in enumerate([(0, 1), (1, 2), (0, 3)]):
            m.cand_idx[0, c] = p
            m.cand_valid[0, c] = True
            m.A_val[0, c] = 0.5 + 0.1 * j
        m.cand_idx[1, 3] = 4
        m.cand_valid[1, 3] = True
        m.A_val[1, 3] = 0.7
    z = torch.rand(5, 64)
    A = m.dense_A()
    expected = z @ torch.linalg.inv(torch.eye(64) - A).t()
    assert torch.allclose(m.route(z), expected, atol=1e-5)
    assert m.longest_path() == 2


def test_training_reduces_reconstruction_error(synth):
    _, X = synth
    for cls in (BatchTopKSAE, MatryoshkaSAE, DRSAE):
        m = cls(cfg())
        hist = train_sae(m, X, n_steps=80, batch_size=512, refresh_every=20)
        assert hist["recon"][-1] < hist["recon"][0]


def test_post_hoc_methods_keep_the_base_encoder(synth):
    _, X = synth
    base = MatryoshkaSAE(cfg())
    train_sae(base, X, n_steps=40, batch_size=512)
    ph = fit_post_hoc_dr(base, X, n_steps=30, batch_size=512, refresh_every=10)
    ag = fit_ag_sae_lite(base, X, n_steps=30, batch_size=512)
    for m in (ph, ag):
        assert torch.equal(m.W_enc, base.W_enc)
        assert m.is_acyclic()


def test_greedy_induction_never_closes_a_cycle():
    g = torch.Generator().manual_seed(0)
    codes = (torch.rand(500, 20, generator=g) > 0.6).float()
    codes[:, 1] = torch.maximum(codes[:, 1], codes[:, 2])  # 1 covers 2
    ps = induce_parent_sets_greedy(codes, torch.randn(8, 20, generator=g), coverage=0.6)
    seen, stack = set(), []
    for start in ps:
        stack = [(start, {start})]
        while stack:
            node, path = stack.pop()
            for p in ps.get(node, []):
                assert p not in path
                stack.append((p, path | {p}))
        seen.add(start)
    assert 1 in ps.get(2, [])


def test_batchtopk_ignores_matryoshka_banks_passed_in_config(synth):
    from topocircuit.models import train_method

    _, X = synth
    m = train_method("BatchTopK", X, 32, 64, 6, 0, 2, sae_kw={"banks": [8, 16, 32, 64]}, batch_size=256)
    assert m.banks == [64]

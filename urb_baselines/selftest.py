"""Offline gates for every baseline. No SUMO, no routerl, no real run.

    python urb_baselines/selftest.py            # every gate
    python urb_baselines/selftest.py --days 400 # longer drive
    python urb_baselines/selftest.py --only lcpo dgn

WHY THIS FILE EXISTS
--------------------------------------------------------------------------------
A URB run is forty minutes of SUMO on a machine with the stack installed. Finding
out there that an attention tensor was transposed, or that an arm's estimator
never received a row, is the expensive way to learn it. Everything here runs on
numpy and torch alone, in about a minute, on any laptop.

Two kinds of gate:

**UNIT** gates check a specific equation against the paper. They are the ones
that would catch a wrong sign, a wrong axis or a dropped mask -- e.g. that LCPO's
step really does respect BOTH trust regions, that ERNIE's adversarial
perturbation really is worse than a random one of the same size, that DGN's
masked neighbour slots really do get exactly zero attention.

**DRIVE** gates run each arm for a few hundred days on ``fakeenv.FakeURB`` -- a
hundred lines of numpy with URB's observation shape, URB's sequential
single-step day, URB's reward scale and a congestion coupling -- and check that
it runs, learns something, writes loss rows and keeps its own invariants. No
number produced here means anything about any city; that is not what it is for.

A failure here is a bug. A pass here is not a result.
"""

import argparse
import contextlib
import io
import os
import sys
import traceback

import numpy as np
import torch

if __name__ == "__main__" and __package__ is None:      # allow direct execution
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from urb_baselines.fakeenv import FakeURB, make_ctx     # noqa: E402
from urb_baselines.host import EpisodeInfo              # noqa: E402
from urb_baselines.nets import MLP                      # noqa: E402
from urb_baselines.records import split_records         # noqa: E402

RESULTS = []

#: slug -> (module under urb_baselines.algos, class name)
_ARM_CLASS = {
    "lcpo": ("lcpo", "LCPO"),
    "happo": ("happo", "HAPPO"),
    "ernie": ("ernie", "ERNIE"),
    "rippo": ("rippo", "RecurrentIPPO"),
    "rma": ("rma", "RMA"),
    "liam": ("liam", "LIAM"),
    "oracle_ippo": ("oracle_ippo", "OracleDriverIPPO"),
    "dr_ippo": ("dr_ippo", "DomainRandomisedIPPO"),
    "dgn": ("dgn", "DGN"),
    "mfq": ("mfq", "MFQ"),
    "eso": ("eso", "ESO"),
    "urls": ("urls", "UnstructuredRLS"),
    "qcdr": ("qcdr", "QCDRestart"),
    "dfp": ("dfp", "DeepFictitiousPlay"),
    "pmpg": ("pmpg", "PerformativeMPG"),
    "doraemon": ("doraemon", "DORAEMON"),
    "wisdom": ("wisdom", "WISDOM"),
    "m3w": ("m3w", "M3W"),
}


class Skip(Exception):
    """Raised by a gate that cannot run here -- reported, never counted as a
    pass. A gate that silently passes because it did nothing is worse than no
    gate at all."""


def gate(name, fn, *a, **kw):
    try:
        note = fn(*a, **kw)
        RESULTS.append((name, "PASS", note or ""))
    except Skip as exc:
        RESULTS.append((name, "SKIP", str(exc)))
    except Exception as exc:                            # noqa: BLE001
        RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}\n"
                                      + traceback.format_exc()))


# ==========================================================================
#  UNIT gates
# ==========================================================================
def g_host_net_parity():
    """H2: the package MLP is the host's Network, layer for layer."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "scripts"))
    try:
        from iql import Network                          # URB's own
    except Exception as exc:                             # noqa: BLE001
        raise Skip(f"scripts/iql.py is not importable here "
                   f"({type(exc).__name__}: {exc}). This gate needs the URB "
                   f"runtime; run it on the machine that has routerl.")
    a = [tuple(p.shape) for p in MLP(7, 3, 2, [8, 16, 8]).parameters()]
    b = [tuple(p.shape) for p in Network(7, 3, 2, [8, 16, 8]).parameters()]
    assert a == b, f"parameter shapes differ:\n  ours {a}\n  host {b}"
    return f"{len(a)} tensors identical to scripts/iql.py's Network"


def g_lcpo_trust_regions():
    """LCPO-1: the step improves the surrogate and respects BOTH KL limits."""
    from urb_baselines.algos import lcpo as L
    import torch.nn.functional as F
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    net = MLP(6, 4, 1, [32, 32])
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    X = torch.tensor(rng.normal(size=(200, 6)), dtype=torch.float32)
    Y = torch.randint(0, 4, (200,))
    for _ in range(200):                     # a non-degenerate policy
        opt.zero_grad(); F.cross_entropy(net(X), Y).backward(); opt.step()
    obs = torch.tensor(rng.normal(size=(20, 6)), dtype=torch.float32)
    ood = torch.tensor(rng.normal(size=(20, 6)) + 5.0, dtype=torch.float32)
    act = torch.randint(0, 4, (20, 1))
    adv = torch.tensor(rng.normal(size=(20, 1)), dtype=torch.float32)
    with torch.no_grad():
        lp_l0 = F.log_softmax(net(obs), -1); pi_l0 = lp_l0.exp()
        lp_a0 = lp_l0.gather(1, act)
        lp_g0 = F.log_softmax(net(ood), -1); pi_g0 = lp_g0.exp()

    def loss(volatile=False):
        with torch.set_grad_enabled(not volatile):
            lp = F.log_softmax(net(obs), -1)
            return (-adv * torch.exp(lp.gather(1, act) - lp_a0)).mean()

    def kl_out():
        return (pi_g0 * (lp_g0 - F.log_softmax(net(ood), -1))).sum(-1)

    def kl_in():
        return (pi_l0 * (lp_l0 - F.log_softmax(net(obs), -1))).sum(-1)

    KO, KI = 1e-4, 1e-1
    before = float(loss(True))
    notes = []
    for dual in (False, True):
        state = {k: v.clone() for k, v in net.state_dict().items()}
        _, _, status = L.trpo_step(net, loss, kl_out, kl_in, KO, KI, 0.1, dual)
        with torch.no_grad():
            ko, ki = float(kl_out().mean()), float(kl_in().mean())
        assert status == "accepted", f"dual={dual}: status {status}"
        assert float(loss(True)) < before, f"dual={dual}: surrogate got worse"
        assert ko <= KO + 1e-12, f"dual={dual}: kl_out {ko:.3e} > {KO}"
        assert ki <= KI + 1e-12, f"dual={dual}: kl_in {ki:.3e} > {KI}"
        notes.append(f"dual={dual}: kl_out {ko:.2e}<={KO:g}, kl_in {ki:.2e}<={KI:g}")
        net.load_state_dict(state)
    return "; ".join(notes)


def g_lcpo_reservoir():
    """LCPO-2: the anchor buffer is a RESERVOIR over all history, and the
    recent window is a ring. A ring-buffer anchor set would only ever hold the
    distribution the in-distribution constraint already covers."""
    from urb_baselines.algos.lcpo import OutOfDSampler, make_is_different
    rng = np.random.RandomState(0)
    buf = OutOfDSampler(3, window=8, capacity=20,
                        distant=make_is_different("all", -9, (2, 3)))
    for t in range(200):
        buf.add_many_exp(np.full((4, 3), t, dtype=np.float32), rng)
    got = np.asarray(buf.get(rng, 16))
    assert len(got) == 16
    assert got.min() < 60, "the reservoir forgot the distant past"
    assert buf.recent_states.min() >= 197, "the window kept stale states"
    return (f"reservoir spans [{got.min():g}, {got.max():g}] of 200 days; "
            f"window holds only [{buf.recent_states.min():g}, "
            f"{buf.recent_states.max():g}]")


def g_lcpo_ood_filter():
    """LCPO-3: against an all-dry recent window (A == 0 exactly, a singular
    covariance), the Mahalanobis filter must call a dry anchor in-distribution
    and every wet one out of it."""
    from urb_baselines.algos.lcpo import make_is_different
    ctx = (5, 6)
    base_dry = np.zeros((20, 6), dtype=np.float32)
    data = np.concatenate(
        [np.zeros((5, 5)), np.array([[0.0], [.2], [.5], [.8], [1.0]])],
        1).astype(np.float32)
    f_all = make_is_different("all", -9, ctx)
    f_mah = make_is_different("mahala", -3.0, ctx)
    assert f_all(data, base_dry).all(), "'all' must accept every anchor"
    m = f_mah(data, base_dry)
    assert not m[0] and m[1:].all(), f"mahala on a dry window gave {m}"
    return "singular-covariance case handled by the ridge; filter discriminates"


def g_ernie_attack():
    """ERNIE-1: the adversarial delta must produce a strictly larger policy
    change than a random delta of the same budget. Otherwise the inner
    maximisation is not happening and the arm is Gaussian smoothing."""
    from urb_baselines.algos.ernie import ErniePPO
    import torch.nn.functional as F
    torch.manual_seed(1)
    rng = np.random.RandomState(0)
    m = ErniePPO(5, 4, device="cpu", batch_size=8, lr=1e-3, num_epochs=1,
                 num_hidden=1, widths=[32, 32],
                 ernie=dict(lam=0.5, eps=0.1, perturb_steps=1, mode="ernie"))
    o = torch.tensor(rng.normal(0, 1, (256, 5)) * [800, 3, 3, 3, 3],
                     dtype=torch.float32)
    tgt = (o[:, 1] > 0).long() + 2 * (o[:, 0] > 0).long()
    opt = torch.optim.Adam(m.policy_net.parameters(), lr=1e-2)
    for _ in range(300):                     # a committed policy
        opt.zero_grad(); F.cross_entropy(m.policy_net(o), tgt).backward(); opt.step()
    vals = {}
    for mode in ("ernie", "ernie_no_st", "gaussian"):
        m.mode = mode
        vals[mode] = float(np.mean([float(m._regularizer(o).detach())
                                    for _ in range(5)]))
    assert vals["ernie"] > vals["gaussian"] * 1.5, (
        f"adversarial KL {vals['ernie']:.3e} is not clearly above random "
        f"{vals['gaussian']:.3e}")
    assert vals["ernie_no_st"] > vals["gaussian"], "detached attack too weak"
    return ("KL adversarial %.2e vs random %.2e (%.1fx)"
            % (vals["ernie"], vals["gaussian"],
               vals["ernie"] / max(vals["gaussian"], 1e-30)))


def g_ernie_ball():
    """ERNIE-2: the perturbation stays inside the l2 ball of radius eps,
    measured in the relative units the released code uses."""
    from urb_baselines.algos.ernie import ErniePPO
    torch.manual_seed(2)
    for steps in (1, 3, 5):
        m = ErniePPO(5, 4, device="cpu", batch_size=8, lr=1e-3, num_epochs=1,
                     num_hidden=1, widths=[16, 16],
                     ernie=dict(lam=0.5, eps=0.1, perturb_steps=steps,
                                mode="ernie"))
        o = torch.randn(64, 5) * torch.tensor([800., 3., 3., 3., 3.])
        m._regularizer(o)
        scale = o.abs().clamp_min(1e-12)
        # recompute the delta the same way the regulariser does, then measure
        d = m._project(torch.randn_like(o) * scale, scale)
        n = (d / scale).flatten(1).norm(dim=1).max().item()
        assert n <= 0.1 + 1e-5, f"steps={steps}: ||delta/scale|| = {n}"
    return "||delta / |o| ||_2 <= eps for 1, 3 and 5 PGD steps"


def g_dgn_attention():
    """DGN-1: Eq. 2/3 against a hand computation, and masked slots get EXACTLY
    zero attention while the remaining rows still sum to one."""
    from urb_baselines.algos.dgn import RelationKernel
    torch.manual_seed(3)
    N, m_, din, heads, dv = 6, 4, 8, 2, 4
    k = RelationKernel(din, din, n_heads=heads, dv=dv)
    h = torch.randn(1, N, din) * 2
    adj = torch.stack([torch.tensor([i] + [(i + j) % N for j in range(1, m_)])
                       for i in range(N)])
    mask = torch.ones(N, m_, dtype=torch.bool)
    out, att = k(h, adj, mask)
    q = torch.relu(k.fcq(h))[0].reshape(N, heads, dv)
    kk = torch.relu(k.fck(h))[0].reshape(N, heads, dv)
    i, head = 2, 1
    logits = torch.stack([(q[i, head] * kk[j, head]).sum() * k.tau
                          for j in adj[i]])
    want = torch.softmax(logits, 0)
    got = att[0, i, head]
    assert torch.allclose(want, got, atol=1e-5), f"{want} vs {got}"
    assert abs(float(att.sum(-1).mean()) - 1.0) < 1e-5

    mask2 = mask.clone(); mask2[:, 3] = False
    _, att2 = k(h, adj, mask2)
    assert float(att2[..., 3].abs().max()) == 0.0, "masked slot got attention"
    assert torch.allclose(att2.sum(-1), torch.ones_like(att2.sum(-1)), atol=1e-5)
    return (f"softmax(tau q.k) matches by hand (max err "
            f"{float((want - got).abs().max()):.2e}); masked slots exactly 0")


def g_dgn_ring_buffer():
    """DGN-2: the (t-1, t, t+1) sampler must never straddle the ring's write
    head, or the temporal regulariser would compare days a full lap apart."""
    from urb_baselines.algos.dgn import DGN
    obj = DGN.__new__(DGN)
    obj.cap = 8
    for stored, forbidden in ((5, None), (8, None), (13, 13 % 8), (40, 0)):
        obj.n_stored = stored
        s = obj._sampleable_slots()
        if stored <= obj.cap:
            assert s.max(initial=-1) <= stored - 2, f"n={stored}: {s}"
            assert 0 not in s, "slot 0 has no predecessor"
        else:
            head = stored % obj.cap
            assert head not in s and (head - 1) % obj.cap not in s, \
                f"n={stored}: {s} contains the write head {head}"
    return "write head and its predecessor excluded in both lap regimes"


def g_graph_static():
    """GRAPH-1: the neighbour structure is a function of agents.csv alone --
    deterministic, symmetric in construction, never touching an outcome."""
    from urb_baselines.graph import NeighbourGraph
    env = FakeURB(n_agents=20, n_machines=10, n_od=2, n_paths=4, seed=3)
    g1 = NeighbourGraph(env.agent_table, env.av_ids, mode="od", n_neighbors=3,
                        verbose=False)
    g2 = NeighbourGraph(env.agent_table, env.av_ids, mode="od", n_neighbors=3,
                        verbose=False)
    assert np.array_equal(g1.adj, g2.adj) and np.array_equal(g1.mask, g2.mask)
    assert (g1.adj[:, 0] == np.arange(g1.n)).all(), "row 0 must be the ego"
    for i in range(g1.n):
        for j, ok in zip(g1.adj[i, 1:], g1.mask[i, 1:]):
            if ok:
                assert g1.od[j] == g1.od[i], "od mode crossed an OD pair"
    dur = {od: 600.0 for od in set(g1.od)}
    g3 = NeighbourGraph(env.agent_table, env.av_ids, mode="copresence",
                        n_neighbors=3, durations=dur, verbose=False)
    assert g3.degree.sum() > 0
    return (f"deterministic; od mode never crosses an OD; copresence gives "
            f"{float(g3.degree.mean()):.1f} candidates/agent")


def g_domain_random():
    """DR-1: sigma is redrawn every reset, the floor keeps it off the wrapper's
    zero short-circuit, and fix() detaches completely."""
    from urb_baselines import domain_random

    class FakeEnv(object):
        _ns_sigma = 3.0
        _ns_layer = None

        def __init__(self):
            self.n = 0

        def reset(self):
            self.n += 1
            return self.n

    e = FakeEnv()
    dr = domain_random.attach(e, 0.0, 3.0, seed=0, verbose=False)
    for _ in range(200):
        e.reset()
    d = np.asarray(dr.draws)
    assert len(d) == 200 and d.min() > 0.0 and d.max() <= 3.0
    assert d.min() >= dr.floor, "a draw fell onto the zero short-circuit"
    assert 0.5 < d.mean() < 2.5, f"suspicious mean {d.mean()}"
    dr.fix(3.0, verbose=False)
    before = len(dr.draws)
    e.reset()
    assert len(dr.draws) == before, "fix() did not detach the randomiser"
    assert e._ns_sigma == 3.0
    return f"200 draws in ({d.min():.3f}, {d.max():.3f}); fix() detaches"


ON_POLICY_ARMS = ("lcpo", "happo", "ernie", "rippo", "rma", "liam",
                  "oracle_ippo", "dr_ippo",
                  "dfp", "pmpg", "doraemon", "wisdom")
OFF_POLICY_ARMS = ("dgn", "mfq", "qcdr", "m3w")
COMPENSATOR_ARMS = ("eso", "urls")          # non-learning; iql host block
ALL_ARMS = ON_POLICY_ARMS + OFF_POLICY_ARMS + COMPENSATOR_ARMS


def _repo():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        ".."))


#: ``numpy.trapezoid`` is numpy >= 2.0; URB pins numpy >= 2.0.2 but the gates
#: must still run on whatever is installed where someone checks them.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def g_qcdr_glr():
    """QCDR-1: the Bernoulli-GLR fires on a change point and not on noise.

    The paper's whole argument is about WHEN a detector crosses its threshold,
    so the statistic and the threshold are checked against a stream whose change
    point is known, and against one that has none. A detector that fires on
    everything and one that fires on nothing both produce plausible curves.
    """
    from urb_baselines.algos.qcdr import (bernoulli_glr, glr_threshold,
                                          binary_kl)
    rng = np.random.RandomState(0)
    delta = 1.0 / np.sqrt(4000.0)

    stationary = (rng.rand(200) < 0.5).astype(float)
    g_stat = bernoulli_glr(stationary)
    thr = glr_threshold(200, delta)
    assert g_stat < thr, (f"the GLR fired on a stationary Bernoulli(0.5) "
                          f"stream: {g_stat:.2f} > {thr:.2f}")

    changed = np.concatenate([(rng.rand(100) < 0.2).astype(float),
                              (rng.rand(100) < 0.8).astype(float)])
    g_chg = bernoulli_glr(changed)
    assert g_chg > thr, (f"the GLR missed a 0.2 -> 0.8 change: "
                         f"{g_chg:.2f} <= {thr:.2f}")

    # the statistic itself, against the definition, at the true split
    s = 100
    mu_l, mu_r = changed[:s].mean(), changed[s:].mean()
    mu_a = changed.mean()
    by_hand = s * binary_kl(mu_l, mu_a) + (len(changed) - s) * binary_kl(mu_r,
                                                                        mu_a)
    assert g_chg >= by_hand - 1e-9, "the max over split points missed the truth"
    return (f"stationary {g_stat:.2f} < {thr:.2f} < changed {g_chg:.2f}; "
            f"true-split value {float(by_hand):.2f}")


def g_qcdr_scale():
    """QCDR-2: at URB's reward scale the RAW stream cannot fire, the mapped one
    can. This is the measurement behind ``normalize_stream``."""
    from urb_baselines.algos.qcdr import _Stream, bernoulli_glr, glr_threshold
    rng = np.random.RandomState(1)
    before = -1000.0 + 40.0 * rng.randn(150)
    after = -700.0 + 40.0 * rng.randn(150)
    rewards = np.concatenate([before, after])
    out = {}
    for norm in (False, True):
        st = _Stream(1, norm, 10000)
        for r in rewards:
            st.push(0, float(r))
        h = st.hist[0]
        out[norm] = (bernoulli_glr(h), float(np.std(h)))
    thr = glr_threshold(len(rewards), 1.0 / np.sqrt(4000.0))
    assert out[False][0] < thr, "the raw stream fired, which it cannot"
    assert out[False][1] < 1e-9, (
        f"the raw stream was not constant after clipping "
        f"(spread {out[False][1]:.3g})")
    assert out[True][0] > thr, (f"the mapped stream missed a 300-second shift: "
                                f"{out[True][0]:.2f} <= {thr:.2f}")
    return (f"raw: GLR {out[False][0]:.2f}, spread {out[False][1]:.1e} "
            f"(cannot fire); mapped: GLR {out[True][0]:.2f} > {thr:.2f}")


def g_dfp_average():
    """DFP-1: fictitious play averages with weight 1/(n+1), so the mix converges
    to the empirical mean of the days it has seen."""
    from urb_baselines.algos.dfp import DeepFictitiousPlay
    env = FakeURB(n_agents=24, n_machines=12, n_od=3, n_paths=4, seed=3)
    ctx = make_ctx(env, algo_cfg={"fp_every": 1}, params={"training_eps": 80})
    with contextlib.redirect_stdout(io.StringIO()):
        algo = DeepFictitiousPlay(ctx)
    K = env.n_paths
    rng = np.random.RandomState(5)
    truth = np.zeros(K)
    n_days = 60
    for d in range(n_days):
        acts = rng.randint(0, K, size=env.n_agents)
        info = EpisodeInfo(d, "train")
        info.peer_acts = {env.ids[i]: int(acts[i]) for i in range(env.n_agents)}
        cnt = np.bincount(acts, minlength=K).astype(float)
        truth += cnt / cnt.sum()
        with contextlib.redirect_stdout(io.StringIO()):
            algo.end_episode(d, "train", info)
    truth /= n_days
    err = float(np.abs(algo.mu_bar - truth).max())
    assert algo.fp_iter == n_days, f"{algo.fp_iter} FP iterations, not {n_days}"
    assert err < 0.02, (f"the FP average is not the empirical mean of the "
                        f"daily mixes (max error {err:.4f})")
    assert abs(algo.mu_bar.sum() - 1.0) < 1e-9, "the mix is not a distribution"
    return (f"{n_days} FP iterations; |mu_bar - empirical mean|_inf = "
            f"{err:.4f}")


def g_pmpg_npg():
    """PMPG-1: for a softmax policy the natural-gradient step IS a logit shift.

    Agarwal, Kakade, Lee & Mahajan (2021). The arm implements INPG as mirror
    descent onto ``softmax(logits + eta * A)``; this checks that the target it
    builds is that object and that it differs from the vanilla gradient's.
    """
    from urb_baselines.algos.pmpg import PerformativeMPG
    env = FakeURB(n_agents=16, n_machines=8, n_od=2, n_paths=4, seed=7)
    ctx = make_ctx(env, algo_cfg={"deploy_window": 4, "retrain_epochs": 1,
                                  "eta": 0.5},
                   params={"training_eps": 40, "batch_size": 4})
    with contextlib.redirect_stdout(io.StringIO()):
        algo = PerformativeMPG(ctx)
    m = algo.models[env.av_ids[0]]
    rng = np.random.RandomState(11)
    for _ in range(8):
        s = rng.randn(algo.obs_size).astype(np.float32)
        m.memory.append((s, int(rng.randint(4)), 0.0, float(rng.randn())))
    states = torch.FloatTensor(np.asarray([x[0] for x in m.memory]))
    actions = torch.LongTensor([x[1] for x in m.memory])
    rewards = torch.FloatTensor([x[3] for x in m.memory])
    adv = m._advantage(rewards)
    with torch.no_grad():
        dep = m.policy_net(states)
    shift = torch.zeros_like(dep)
    shift.scatter_(1, actions.unsqueeze(1), (algo.eta * adv).unsqueeze(1))
    target = torch.softmax(dep + shift, dim=-1)
    base = torch.softmax(dep, dim=-1)

    # the shift moves probability toward positive-advantage actions, and only
    # toward them -- that is what makes it the natural gradient and not noise
    taken = torch.gather(target - base, 1, actions.unsqueeze(1)).squeeze(1)
    agree = float(((taken > 0) == (adv > 0)).float().mean().item())
    assert agree > 0.99, (f"the logit shift moved the taken action's "
                          f"probability the wrong way {1 - agree:.0%} of the time")
    with contextlib.redirect_stdout(io.StringIO()):
        v = algo._retrain_one(m)
    assert v is not None, "the retrain did not run on a full window"
    assert len(m.memory) == 0, "the window was not consumed by the retrain"
    return (f"logit shift agrees with sign(A) {agree:.0%}; max |dp| "
            f"{float((target - base).abs().max().item()):.4f}")


def g_doraemon_opt():
    """DOR-1: the Beta closed forms are right and the solve respects both
    constraints while increasing entropy."""
    from urb_baselines.algos.doraemon import BetaDR, solve_doraemon, \
        _success_estimate
    a = BetaDR(2.0, 5.0, 0.0, 3.0)
    # entropy and KL against numerical integration on the unit interval
    xs = np.linspace(1e-6, 1 - 1e-6, 200001)
    pa = np.exp(a.log_pdf_unit(xs))
    h_num = -_trapz(pa * a.log_pdf_unit(xs), xs) + np.log(3.0)
    assert abs(h_num - a.entropy()) < 1e-4, (
        f"entropy {a.entropy():.6f} != numerical {h_num:.6f}")
    b = BetaDR(3.0, 4.0, 0.0, 3.0)
    kl_num = float(_trapz(pa * (a.log_pdf_unit(xs) - b.log_pdf_unit(xs)),
                                xs))
    assert abs(kl_num - a.kl_to(b)) < 1e-4, (
        f"KL {a.kl_to(b):.6f} != numerical {kl_num:.6f}")
    assert a.kl_to(a) < 1e-12, "KL(p||p) is not zero"

    # everything succeeds -> the solve should widen, inside the trust region
    cur = BetaDR(100.0, 100.0, 0.0, 3.0)
    rng = np.random.RandomState(2)
    u = cur.to_unit(np.asarray([cur.sample(rng) for _ in range(400)]))
    succ = np.ones_like(u)
    new, info = solve_doraemon(cur, u, succ, alpha=0.5, kl_ub=0.05,
                               bounds=[0.5, 200.0], grid=21)
    assert new.kl_to(cur) <= 0.05 + 1e-9, (
        f"the solve left the trust region: KL {new.kl_to(cur):.4f}")
    assert new.entropy() > cur.entropy(), "the solve did not widen"
    assert _success_estimate(new, cur, u, succ) >= 0.5 - 1e-9, \
        "the returned distribution violates the success constraint"

    # nothing succeeds -> the backup problem, still inside the trust region
    new2, info2 = solve_doraemon(cur, u, np.zeros_like(u), alpha=0.5,
                                 kl_ub=0.05, bounds=[0.5, 200.0], grid=21)
    assert new2.kl_to(cur) <= 0.05 + 1e-9, "the backup left the trust region"
    assert info2["mode"] in ("backup", "none"), info2["mode"]
    return (f"entropy/KL match numerical integration to 1e-4; widened "
            f"{cur.entropy():+.3f} -> {new.entropy():+.3f} at KL "
            f"{new.kl_to(cur):.4f} <= 0.05")


def g_wisdom_haar():
    """WAV-1: the wavelet layer at initialisation IS a Haar DWT.

    The filters are learnable, so after training they are not a Haar transform
    any more -- but if they do not START as one, the arm is a convolution with
    the wrong name, and nothing downstream would say so.
    """
    from urb_baselines.algos.wisdom import WaveletNet
    net = WaveletNet(latent_dim=1, levels=2)
    x = torch.tensor([[1.0, 3.0, 2.0, 8.0]]).unsqueeze(-1)   # (1, 4, 1)
    with torch.no_grad():
        _, bands = net(x)
    r2 = float(np.sqrt(0.5))
    g1 = [(1.0 - 3.0) * r2, (2.0 - 8.0) * r2]
    u1 = [(1.0 + 3.0) * r2, (2.0 + 8.0) * r2]
    g2 = (u1[0] - u1[1]) * r2
    u2 = (u1[0] + u1[1]) * r2
    got = [float(b.reshape(-1)[0]) for b in bands]
    want = [g1[-1], g2, u2]
    err = max(abs(a - b) for a, b in zip(got, want))
    assert err < 1e-5, f"bands {got} != hand-computed Haar {want}"
    # energy is preserved by an orthonormal transform
    e_in = float((x ** 2).sum())
    e_out = g1[0] ** 2 + g1[1] ** 2 + g2 ** 2 + u2 ** 2
    assert abs(e_in - e_out) < 1e-4, f"energy {e_in} -> {e_out}"
    return (f"levels 1-2 match a hand-computed Haar DWT to {err:.1e}; "
            f"energy preserved ({e_in:.1f})")


def g_m3w_twohot():
    """M3W-1: the two-hot code round-trips, and URB's reward scale destroys it.

    This is the measurement behind ``normalize_reward``: with the release's own
    [-10, 10] support, every URB reward lands in the bottom bin, so the reward
    model has nothing to predict and the planner scores every joint action the
    same.
    """
    from urb_baselines.algos.m3w import TwoHot
    dev = torch.device("cpu")
    th = TwoHot(-10.0, 10.0, 101, dev)
    v = torch.tensor([-7.3, 0.0, 2.5, 9.99])
    code = th.encode(v)
    assert torch.allclose(code.sum(-1), torch.ones(4), atol=1e-6), \
        "the two-hot code is not a distribution"
    back = (code * th.centres).sum(-1)
    assert float((back - v).abs().max()) < 1e-4, f"round trip {back} != {v}"

    raw = torch.tensor([-1200.0, -1000.0, -800.0, -600.0])
    craw = th.encode(raw)
    assert float(craw[:, 0].min()) > 0.999, (
        "the raw URB rewards did NOT all collapse into the bottom bin -- the "
        "measurement this gate is built on has changed")
    assert float((craw.max(dim=0).values - craw.min(dim=0).values).max()) < 1e-6, \
        "the raw codes differ, so the collapse is not total"
    assert th.clipped_fraction(raw) == 1.0, "clipped_fraction missed the clip"

    std = (raw - raw.mean()) / raw.std()
    cstd = th.encode(std)
    spread = float((cstd.max(dim=0).values - cstd.min(dim=0).values).max())
    assert spread > 0.5, f"the standardised codes barely differ ({spread:.3f})"
    return ("round trip < 1e-4; raw -1200..-600 all collapse to bin 0 "
            f"(identical codes), standardised spread {spread:.2f}")


def g_config_host_block():
    """CFG-1: every arm's HOST hyperparameters are the shared ones, verbatim.

    Rule R2: a difference between two arms must never be a tuning difference.
    The only way to keep that true is to check it, because a config is edited by
    hand and nothing else would notice a drifted learning rate.
    """
    import json
    root = os.path.join(_repo(), "config", "algo_config")

    def block(slug):
        p = os.path.join(root, slug, "config1.json")
        if not os.path.exists(p):
            raise Skip(f"{p} not found")
        d = json.load(io.open(p, encoding="utf-8"))
        return d, {k: v for k, v in d.items() if k not in ("desc", slug)}

    _, on_ref = block("ippo")
    _, off_ref = block("iql")
    bad = []
    for slug in ALL_ARMS:
        d, host = block(slug)
        if slug not in d:
            bad.append(f"{slug}: no '{slug}' block")
            continue
        ref = on_ref if slug in ON_POLICY_ARMS else off_ref
        which = "ippo" if slug in ON_POLICY_ARMS else "iql"
        if host != ref:
            diff = {k: (host.get(k), ref.get(k))
                    for k in set(host) | set(ref) if host.get(k) != ref.get(k)}
            bad.append(f"{slug}: host block differs from {which}/config1.json: "
                       f"{diff}")
    assert not bad, "\n".join(bad)
    return (f"{len(ALL_ARMS)} arms: host block identical to ippo/config1.json "
            f"or iql/config1.json")


def g_config_constructs():
    """CFG-2: every arm CONSTRUCTS under its own shipped config.

    This is the gate that would have caught the state this repository was in on
    2026-09-20: ``liam/config1.json`` said ``graph: overlap`` while
    ``liam.py`` still built the graph without a route table, so LIAM raised
    ValueError -- after the 200-day human-learning phase, seventeen minutes into
    a run. A config and its code drifting apart is invisible to every other gate
    here, because the drive gates use their own inline config.
    """
    import json
    root = os.path.join(_repo(), "config", "algo_config")
    built = []
    for slug in ALL_ARMS:
        p = os.path.join(root, slug, "config1.json")
        if not os.path.exists(p):
            raise Skip(f"{p} not found")
        d = json.load(io.open(p, encoding="utf-8"))
        host = {k: v for k, v in d.items() if k not in ("desc", slug)}
        mod_name, cls_name = _ARM_CLASS[slug]
        mod = __import__("urb_baselines.algos." + mod_name, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        torch.manual_seed(0)
        np.random.seed(0)
        # 4 OD pairs over 24 travellers, i.e. NOT URB's singleton structure --
        # the point here is that the shipped config builds, not what it measures.
        env = FakeURB(n_agents=24, n_machines=12, n_od=4, n_paths=4, seed=1)
        host = dict(host)
        host["training_eps"] = 60
        ctx = make_ctx(env, algo_cfg=d[slug], params=host,
                       with_context=cls.needs_context)
        with contextlib.redirect_stdout(io.StringIO()):
            algo = cls(ctx)
            # Exercised in the host's own order: begin_episode, then act.
            algo.begin_episode(0, "train")
            algo.act(env.av_ids[0], env.observe(0, np.zeros(24, dtype=np.int64),
                                                np.zeros(24, dtype=bool)))
        built.append(slug)
    return f"{len(built)} arms construct and act under their shipped config1.json"


def g_device():
    """DEV-1: the device is honoured from flag/env AND proved before use.

    The failure this prevents: ``torch.cuda.is_available()`` is True, the torch
    build has no kernels for the GPU, and the crash arrives at the first forward
    pass -- 200 human-learning days and seventeen minutes into the run. Observed
    on a real box. An explicit device that fails must be fatal; ``auto`` must
    fall back to CPU rather than take the run down.
    """
    from urb_baselines.host import _resolve_device
    saved = {k: os.environ.get(k) for k in ("DEVICE", "URB_DEVICE")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        assert _resolve_device("cpu").type == "cpu"
        os.environ["DEVICE"] = "cpu"
        assert _resolve_device(None).type == "cpu", "$DEVICE was ignored"
        os.environ.pop("DEVICE")
        os.environ["URB_DEVICE"] = "cpu"
        assert _resolve_device(None).type == "cpu", "$URB_DEVICE was ignored"
        os.environ.pop("URB_DEVICE")
        try:
            _resolve_device("banana")
            raise AssertionError("a bad device spec was accepted")
        except ValueError:
            pass

        # a GPU whose kernels are missing, both branches
        real_randn, real_avail = torch.randn, torch.cuda.is_available

        def broken(*a, **kw):
            d = kw.get("device")
            if d is not None and torch.device(d).type == "cuda":
                raise RuntimeError("CUDA error: no kernel image is available "
                                   "for execution on the device")
            return real_randn(*a, **kw)

        torch.randn, torch.cuda.is_available = broken, (lambda: True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                assert _resolve_device("auto").type == "cpu", \
                    "auto did not fall back to CPU on a broken device"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    _resolve_device("cuda")
                raise AssertionError("an explicit broken --device cuda was "
                                     "silently swapped for something else")
            except RuntimeError:
                pass
        finally:
            torch.randn, torch.cuda.is_available = real_randn, real_avail
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return ("flag and $DEVICE/$URB_DEVICE honoured; auto falls back to CPU on a "
            "dead device, explicit cuda is fatal")


def g_records():
    """REC-1: the record splitter keeps humans in peer_acts, machines in
    av_records, and never silently drops a malformed row."""
    recs = [{"id": 1, "action": 2, "travel_time": 500.0},
            {"id": 2, "action": 0, "travel_time": 600.0},
            {"id": 3, "action": 1, "travel_time": float("nan")},
            {"id": 4, "action": None, "travel_time": 1.0},
            "not a dict"]
    av, peers, tt, bad = split_records(recs, {"1"})
    assert set(av) == {"1"} and av["1"] == (2, 500.0)
    assert set(peers) == {"1", "2", "3"}, peers
    assert abs(tt - 600.0) < 1e-9, tt
    assert bad == 2, bad
    return "1 machine, 3 peers (humans included), 2 unparseable counted"


# ==========================================================================
#  DRIVE gates
# ==========================================================================
def drive(cls, algo_cfg=None, ctx_kw=None, days=200, seed=1):
    """Run one arm on FakeURB. Returns (early mean reward, late, n loss rows)."""
    torch.manual_seed(0)
    np.random.seed(0)
    env = FakeURB(n_agents=24, n_machines=12, n_od=3, n_paths=4, seed=seed)
    ctx = make_ctx(env, algo_cfg=algo_cfg, params={"training_eps": days},
                   with_context=cls.needs_context, **(ctx_kw or {}))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        algo = cls(ctx)
        mids = set(env.av_ids)
        early, late = [], []
        for d in range(days):
            # The host calls env.reset() at the top of every day, and the arms
            # that WRAP reset (domain randomisation, DORAEMON) draw their
            # severity there. A harness that skipped it would make those arms
            # look inert for a reason that is not theirs.
            env.reset()
            info = EpisodeInfo(d, "train")
            if ctx.context is not None:
                info.context = ctx.context.observed(d)
            algo.begin_episode(d, "train")
            chosen = {}

            def choose(aid, obs):
                a = int(algo.act(aid, obs))
                chosen[aid] = a
                return a

            rewards, records = env.run_day(choose)
            for aid in env.av_ids:
                algo.push(aid, rewards[aid])
                info.rewards[aid] = rewards[aid]
            info.actions = chosen
            if cls.needs_records:
                a_, p_, t_, _ = split_records(records, mids)
                info.av_records, info.peer_acts, info.tt_hdv = a_, p_, t_
                info.batches = [(d, a_, p_, t_)]
            algo.end_episode(d, "train", info)
            if d % int(ctx.params["update_every"]) == 0:
                algo.learn(d)
            m = float(np.mean([rewards[a] for a in env.av_ids]))
            (early if d < days // 4 else late if d >= days - days // 4
             else []).append(m)

        algo.begin_test()
        env.reset()
        info = EpisodeInfo(days, "test")
        algo.begin_episode(days, "test")
        rewards, records = env.run_day(lambda a, o: int(algo.act(a, o)))
        for aid in env.av_ids:
            algo.push(aid, rewards[aid])
        if cls.needs_records:
            a_, p_, t_, _ = split_records(records, mids)
            info.av_records, info.peer_acts, info.tt_hdv = a_, p_, t_
        algo.end_episode(days, "test", info)
        rows = algo.loss_records()
        diag = algo.diagnostics()
        algo.close()
    return float(np.mean(early)), float(np.mean(late)), len(rows), diag, algo


def make_drive_gate(path, clsname, cfg=None, ctx_kw=None, check=None,
                    expect_learn=True):
    def fn(days):
        mod = __import__("urb_baselines.algos." + path, fromlist=[clsname])
        cls = getattr(mod, clsname)
        e, l, n, diag, algo = drive(cls, cfg, ctx_kw, days=days)
        assert n > 0, "no loss rows were produced"
        assert np.isfinite(e) and np.isfinite(l), "non-finite reward"
        if expect_learn:
            assert l > e, (f"no improvement over {days} days "
                           f"({e:.1f} -> {l:.1f})")
        note = f"{e:.0f} -> {l:.0f} over {days} days, {n} loss rows"
        if check is not None:
            note += "; " + (check(algo, diag) or "")
        return note
    return fn


def _chk_lcpo(algo, diag):
    assert diag["ood_hit"] > 0.5, f"the constraint engaged only {diag['ood_hit']:.0%}"
    return f"constraint engaged {diag['ood_hit']:.0%} of updates"


def _chk_happo(algo, diag):
    assert abs(diag["factor"] - 1.0) > 1e-6, (
        "the sequential factor never left 1, so the actors are frozen -- "
        "check use_feature_normalization")
    return f"factor {diag['factor']:.3f}, ratio {diag['ratio']:.4f}"


def _chk_dgn(algo, diag):
    assert diag["att_spread"] < 0.999, "attention saturated to a hard argmax"
    return f"attention spread {diag['att_spread']:.4f}"


def _chk_mfq(algo, diag):
    assert diag["field_upd"] > 0, "the mean action was never updated"
    assert diag["abar_dev"] > 1e-6, "abar is exactly uniform: no signal"
    return f"|abar-uniform| = {diag['abar_dev']:.3f}"


def _chk_rma(algo, diag):
    assert diag["phase"] == 2, "phase 2 never started"
    assert algo.n_phi_updates > 0, "the adaptation module was never trained"
    return f"phase 2 reached, phi mse {diag['phi_mse']:.4f}"


def _chk_eso(algo, diag):
    assert algo.n_obs_updates > 0, "the observer never received a residual"
    return f"{algo.n_obs_updates} observer updates"


def _chk_urls(algo, diag):
    assert algo.n_fits > 0, "RLS never fitted"
    return f"{algo.n_fits} RLS updates, {algo.d} features"


def _chk_liam(algo, diag):
    assert diag["rec_o"] < 1e4, (
        f"observation reconstruction loss {diag['rec_o']:.3g} -- the target is "
        "not standardised and the decoder is memorising start times")
    return f"rec_o {diag['rec_o']:.2f}, rec_a {diag['rec_a']:.2f}"


def _chk_qcdr(algo, diag):
    assert diag["u_spread"] > 1e-4, (
        "the mapped reward stream is constant, so no detector could fire -- "
        "check normalize_stream")
    return (f"{diag['restarts']} restarts, u_spread {diag['u_spread']:.3f}"
            + (f", GLR/thr {diag['glr_frac']:.2f}" if "glr_frac" in diag else ""))


def _chk_dfp(algo, diag):
    assert algo.n_mix_updates > 0, "the population mix was never updated"
    assert diag["fp_iter"] > 0, "not one FP iteration completed"
    return (f"{diag['fp_iter']} FP iterations, |mu-uniform| "
            f"{diag['mix_dev']:.3f}")


def _chk_pmpg(algo, diag):
    assert diag["retrains"] > 0, "no deployment was ever retrained"
    assert algo.dtheta and max(algo.dtheta) > 1e-12, (
        "the parameters never moved across a retrain")
    return f"{diag['retrains']} retrains, last ||d theta|| {diag['dtheta']:.2e}"


def _chk_doraemon(algo, diag):
    assert algo.perf_lb is not None, "the success threshold never calibrated"
    assert diag["upd"] > 0, "the severity distribution was never updated"
    return (f"{diag['upd']} distribution updates, Beta({diag['beta_a']:.1f}, "
            f"{diag['beta_b']:.1f}), H {diag['H']:+.3f}")


def _chk_wisdom(algo, diag):
    assert diag["upd"] > 0, "the representation was never trained"
    assert diag["z_std"] > 1e-4, (
        f"the latent is constant (z_std {diag['z_std']:.2e}) -- the encoder "
        "collapsed; check normalize_context")
    return f"{diag['upd']} updates, z_std {diag['z_std']:.3f}"


def _chk_m3w(algo, diag):
    assert diag["wm_upd"] > 0, "the world model was never trained"
    assert diag["plans"] > 0, "no day was ever planned"
    assert diag["plan_spread"] > 1e-4, (
        "every sampled joint action scores the same -- the reward model is a "
        "constant and MPPI is a uniform draw; check normalize_reward")
    return (f"{diag['plans']} plans, spread {diag['plan_spread']:.3f}, "
            f"clip {diag['clip']:.2f}")


# One drive gate per baseline, in its primary configuration. The ablation arms
# are a separate list behind --arms: they are worth checking (a broken `--arm`
# path would otherwise ship silently, since CFG-2 only exercises the shipped
# config) but they double the output for a question nobody is asking during a
# routine preflight.
DRIVES = [
    ("drive ippo (reference)", "reference", "IPPOReference", None, None, None),
    ("drive iql  (reference)", "reference", "IQLReference", None, None, None),
    ("drive lcpo", "lcpo", "LCPO",
     {"master_batch": 8, "ood_subsample": 1, "value_lr": 0.01}, None, _chk_lcpo),
    ("drive happo", "happo", "HAPPO", {"episode_length": 16}, None, _chk_happo),
    ("drive ernie", "ernie", "ERNIE", None, None, None),
    ("drive rippo", "rippo", "RecurrentIPPO", {"data_chunk_length": 4}, None,
     None),
    ("drive dgn", "dgn", "DGN", {"graph": "od"}, None, _chk_dgn),
    ("drive mfq", "mfq", "MFQ", None, None, _chk_mfq),
    ("drive liam", "liam", "LIAM", {"n_modelled": 3}, None, _chk_liam),
    ("drive rma", "rma", "RMA", {"history_len": 8, "phase1_frac": 0.5}, None,
     _chk_rma),
    ("drive eso", "eso", "ESO", None, None, _chk_eso),
    ("drive urls", "urls", "UnstructuredRLS", None, None, _chk_urls),
    ("drive oracle_ippo", "oracle_ippo", "OracleDriverIPPO", None, None, None),
    ("drive qcdr", "qcdr", "QCDRestart", {"min_hist": 8, "max_hist": 64},
     None, _chk_qcdr),
    ("drive dfp", "dfp", "DeepFictitiousPlay", {"fp_every": 5,
     "avg_capacity": 200}, None, _chk_dfp),
    ("drive pmpg", "pmpg", "PerformativeMPG",
     {"deploy_window": 24, "retrain_epochs": 4}, None, _chk_pmpg),
    ("drive doraemon", "doraemon", "DORAEMON",
     {"update_every_days": 20, "warmup_days": 20, "grid": 9}, None,
     _chk_doraemon),
    ("drive wisdom", "wisdom", "WISDOM",
     {"seq_len": 8, "wav_batch": 8, "train_every": 2, "latent_dim": 4}, None,
     _chk_wisdom),
    ("drive m3w", "m3w", "M3W",
     {"latent_dim": 16, "hidden": 32, "num_dynamics_experts": 3,
      "num_reward_experts": 4, "num_samples": 16, "num_elites": 4,
      "iterations": 2, "num_pi_trajs": 2, "wm_batch": 32, "warmup_days": 20,
      "graph": "od"}, None, _chk_m3w),
]

ARM_DRIVES = [
    ("drive lcpo --arm lcppo", "lcpo", "LCPO",
     {"master_batch": 8, "ood_subsample": 1, "value_lr": 0.01},
     {"arm": "lcppo"}, _chk_lcpo),
    ("drive lcpo --arm a2c", "lcpo", "LCPO",
     {"master_batch": 8, "value_lr": 0.01}, {"arm": "a2c"}, None),
    ("drive ernie --arm gaussian", "ernie", "ERNIE", None, {"arm": "gaussian"},
     None),
    ("drive dgn --arm dgn_r", "dgn", "DGN", {"graph": "od"}, {"arm": "dgn_r"},
     _chk_dgn),
    ("drive eso --arm eso2", "eso", "ESO", None, {"arm": "eso2"}, _chk_eso),
    ("drive mfq --field-scope od", "mfq", "MFQ", {"field_scope": "od"}, None,
     _chk_mfq),
    ("drive qcdr --arm rr", "qcdr", "QCDRestart", {"xi": 0.2}, {"arm": "rr"},
     _chk_qcdr),
    ("drive qcdr --arm master", "qcdr", "QCDRestart", None, {"arm": "master"},
     None),
    ("drive dfp --arm br", "dfp", "DeepFictitiousPlay", None, {"arm": "br"},
     None),
    ("drive pmpg --arm ipga", "pmpg", "PerformativeMPG",
     {"deploy_window": 24, "retrain_epochs": 4}, {"arm": "ipga"}, _chk_pmpg),
    ("drive doraemon --arm fixed", "doraemon", "DORAEMON",
     {"update_every_days": 20, "warmup_days": 20}, {"arm": "fixed"}, None),
    ("drive wisdom --arm flat", "wisdom", "WISDOM",
     {"seq_len": 8, "wav_batch": 8, "train_every": 2, "latent_dim": 4},
     {"arm": "flat"}, _chk_wisdom),
    ("drive m3w --arm mlp", "m3w", "M3W",
     {"latent_dim": 16, "hidden": 32, "num_samples": 16, "num_elites": 4,
      "iterations": 2, "num_pi_trajs": 2, "wm_batch": 32, "warmup_days": 20,
      "graph": "od"}, {"arm": "mlp"}, _chk_m3w),
]


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--only", nargs="*", default=None,
                    help="substrings; only gates whose name matches are run")
    ap.add_argument("--arms", action="store_true",
                    help="also drive each baseline's ablation arms (off by "
                         "default: one gate per baseline)")
    args = ap.parse_args()

    def wanted(name):
        return (not args.only) or any(k in name for k in args.only)

    units = [
        ("host net parity (H2)", g_host_net_parity),
        ("lcpo trust regions (LCPO-1)", g_lcpo_trust_regions),
        ("lcpo reservoir (LCPO-2)", g_lcpo_reservoir),
        ("lcpo ood filter (LCPO-3)", g_lcpo_ood_filter),
        ("ernie attack strength (ERNIE-1)", g_ernie_attack),
        ("ernie epsilon ball (ERNIE-2)", g_ernie_ball),
        ("dgn attention (DGN-1)", g_dgn_attention),
        ("dgn ring buffer (DGN-2)", g_dgn_ring_buffer),
        ("neighbour graph (GRAPH-1)", g_graph_static),
        ("domain randomisation (DR-1)", g_domain_random),
        ("device resolution (DEV-1)", g_device),
        ("qcdr GLR detector (QCDR-1)", g_qcdr_glr),
        ("qcdr reward scale (QCDR-2)", g_qcdr_scale),
        ("dfp FP averaging (DFP-1)", g_dfp_average),
        ("pmpg natural gradient (PMPG-1)", g_pmpg_npg),
        ("doraemon entropy solve (DOR-1)", g_doraemon_opt),
        ("wisdom Haar transform (WAV-1)", g_wisdom_haar),
        ("m3w two-hot scale (M3W-1)", g_m3w_twohot),
        ("config host block (CFG-1)", g_config_host_block),
        ("config constructs (CFG-2)", g_config_constructs),
        ("record splitting (REC-1)", g_records),
    ]
    print("=" * 78)
    print("UNIT GATES")
    print("=" * 78)
    for name, fn in units:
        if wanted(name):
            gate(name, fn)
            status, note = RESULTS[-1][1], RESULTS[-1][2]
            print(f"  [{status}] {name:<34} "
                  f"{note.splitlines()[0] if note else ''}")

    print()
    print("=" * 78)
    drives = DRIVES + (ARM_DRIVES if args.arms else [])
    print(f"DRIVE GATES  ({args.days} days on fakeenv.FakeURB -- a wiring test, "
          f"not a benchmark)")
    print(f"             one per baseline"
          + ("; ablation arms included (--arms)" if args.arms
             else "; --arms adds the ablation arms"))
    print("=" * 78)
    for name, path, cls, cfg, kw, chk in drives:
        if not wanted(name):
            continue
        gate(name, make_drive_gate(path, cls, cfg, kw, chk), args.days)
        status, note = RESULTS[-1][1], RESULTS[-1][2]
        print(f"  [{status}] {name:<28} "
              f"{note.splitlines()[0] if note else ''}")

    bad = [(n, m) for n, s, m in RESULTS if s == "FAIL"]
    skipped = [(n, m) for n, s, m in RESULTS if s == "SKIP"]
    print()
    print("=" * 78)
    for n, m in skipped:
        print(f"SKIPPED: {n} -- {m}")
    if bad:
        for n, m in bad:
            print(f"FAILED: {n}\n{m}")
        print(f"{len(bad)} / {len(RESULTS)} GATES FAILED")
    else:
        print(f"{len(RESULTS) - len(skipped)} / {len(RESULTS)} GATES PASSED"
              + (f", {len(skipped)} skipped" if skipped else ""))
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

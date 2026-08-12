"""Offline correctness gates. NO SUMO, NO routerl, NO GPU, NO training.

    python pact1/selftest.py

Run this before every campaign and after every edit. It costs a few seconds and it
is the only place where the arithmetic that the whole method rests on is checked
against something other than itself. The last test is the important one: it builds
a synthetic city whose congestion law is a KNOWN linear function of road-class
peer load, runs the real coordinator on it, and asserts that the estimator recovers
the truth. If that passes, the identify -> forecast -> score chain is wired
correctly and any failure on real URB data is a statement about the city, not about
this code.

Also collectable by pytest (``pytest pact1/selftest.py``).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pact1.basis import RouteBasis, build_time_overlap          # noqa: E402
from pact1.core import (                                         # noqa: E402
    AgentRLS,
    herd_index,
    relative_excess,
    rls_confidence,
    steer_logits,
    trust_from_w,
)


# ==========================================================================
#  estimator
# ==========================================================================
def test_rls_recovers_known_beta():
    rng = np.random.RandomState(0)
    beta = np.array([0.3, 1.5, -0.4, 0.9])
    est = AgentRLS(4, mu=1.0, p0=100.0)
    for _ in range(400):
        phi = rng.randn(4)
        phi[0] = 1.0
        est.update(phi[None, :], np.array([phi @ beta + 0.01 * rng.randn()]))
    err = np.abs(est.beta - beta).max()
    assert err < 0.05, f"RLS did not recover beta (max err {err:.4f})"


def test_rls_tracks_drift():
    rng = np.random.RandomState(1)
    est = AgentRLS(2, mu=0.98, p0=10.0)
    for t in range(3000):
        beta = np.array([1.0, 0.5 + 0.5 * np.sin(2 * np.pi * t / 1200.0)])
        phi = np.array([1.0, rng.randn()])
        est.update(phi[None, :], np.array([phi @ beta + 0.02 * rng.randn()]))
    beta_end = np.array([1.0, 0.5 + 0.5 * np.sin(2 * np.pi * 2999 / 1200.0)])
    err = np.abs(est.beta - beta_end).max()
    assert err < 0.25, f"RLS lost a slowly drifting beta (err {err:.4f})"


def test_rls_dead_row_does_not_inflate_covariance():
    """A zero regressor carries no information about beta but WOULD divide P by mu
    every step, silently tightening the effective forgetting factor. Guarded."""
    est = AgentRLS(3, mu=0.99, p0=10.0)
    tr0 = float(np.trace(est.P))
    for _ in range(500):
        est.update(np.zeros((1, 3)), np.array([1.0]))
    tr1 = float(np.trace(est.P))
    assert abs(tr1 - tr0) < 1e-9, (
        f"dead rows inflated tr(P) {tr0:.4f} -> {tr1:.4f}: covariance windup"
    )
    assert np.allclose(est.beta, 0.0), "dead rows moved beta"


# ==========================================================================
#  trust / confidence / sensor
# ==========================================================================
def test_trust_prior_is_inverted():
    """guide III.5: w=0 must mean ~0.90 trust, not 0.5. Getting this backwards cost
    a 10M-step Ant run 1800 return."""
    g = trust_from_w(np.array([0.0]), np.array([0.0]), ema=0.0, g_max=1.0, bias=2.2)
    assert abs(float(g[0]) - 0.9002) < 0.01, f"trust prior is {float(g[0]):.4f}, want ~0.90"
    lo = trust_from_w(np.array([-30.0]), np.array([0.0]), 0.0, 1.0, 2.2)
    assert float(lo[0]) < 1e-6, "trust cannot be driven to the floor"


def test_confidence_cold_to_warm():
    r = 4
    cold = rls_confidence(np.eye(r) * 10.0, 10.0, r)
    assert abs(cold - 1.0 / (1.0 + r)) < 1e-9, "cold confidence is not 1/(1+r)"
    warm = rls_confidence(np.eye(r) * 1e-4, 10.0, r)
    assert warm > 0.999, "confidence does not approach 1 as P shrinks"


def test_relative_excess_and_clip():
    y = relative_excess([100.0, 300.0, 5000.0], [100.0, 100.0, 100.0], clip=10.0)
    assert abs(y[0] - 0.0) < 1e-12
    assert abs(y[1] - 2.0) < 1e-12
    assert abs(y[2] - 10.0) < 1e-12, "clip did not bound a SUMO outlier"


def test_herd_index_bounds():
    assert abs(herd_index([25, 25, 25, 25]) - 0.0) < 1e-9
    assert abs(herd_index([100, 0, 0, 0]) - 1.0) < 1e-9


# ==========================================================================
#  the steering channel
# ==========================================================================
def test_floor_property_is_exact():
    """THE safety property: trust 0 reproduces the untouched policy bit for bit,
    for any estimate however wrong."""
    rng = np.random.RandomState(3)
    for _ in range(500):
        lg = rng.randn(4)
        tt = rng.randn(4) * 1e6            # deliberately absurd predictions
        out = steer_logits(lg, tt, 0.0, kappa=1.0)
        assert np.array_equal(out, lg), "g=0 changed the logits"


def test_steer_degenerate_cases():
    lg = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.array_equal(steer_logits(lg, np.full(4, 7.0), 1.0), lg), \
        "identical predictions must produce no shift"
    assert np.array_equal(steer_logits(lg, np.full(4, np.nan), 1.0), lg), \
        "all-NaN predictions must produce no shift"
    out = steer_logits(lg, np.array([1.0, 2.0, 3.0, 4.0]), 1.0,
                       valid=np.array([True, True, False, False]))
    assert out[2] == lg[2] and out[3] == lg[3], "masked routes were modified"


def test_steer_direction():
    """A route predicted SLOWER must get a LOWER logit. A sign error here would
    steer the fleet straight into the jam and still look healthy."""
    lg = np.zeros(3)
    out = steer_logits(lg, np.array([10.0, 20.0, 30.0]), 1.0)
    assert out[0] > out[1] > out[2], f"steering sign is inverted: {out}"


def test_torch_numpy_parity():
    from pact1.policy import check_shift_parity
    ok, worst, floor_ok = check_shift_parity()
    assert floor_ok, "torch shift is not an exact no-op at g=0"
    assert ok, f"torch and numpy shifts disagree (max abs diff {worst:.3e})"


# ==========================================================================
#  the basis
# ==========================================================================
def _toy_city(seed=0, n_od=6, K=4, n_edge=60, segregated=True):
    """A tiny synthetic city: edges in three distinct speed tiers, routes built to
    genuinely overlap so the coupling is non-trivial.

    ``segregated=True``  route k is built mostly from ONE road class, so a city
                         where the alternatives really are "the motorway or the
                         back streets". The three channels then move
                         independently and theta is decomposable.
    ``segregated=False`` every route mixes the classes evenly. The channels become
                         near-collinear -- prediction still works, the SPLIT does
                         not. This is guide III.6's caveat and it gets its own
                         test, because a benchmark can be either kind and the
                         difference decides what may be claimed.
    """
    rng = np.random.RandomState(seed)
    speeds = np.array([8.33, 13.89, 16.67])
    net = {}
    for e in range(n_edge):
        net[f"e{e}"] = (
            float(50.0 + 100.0 * rng.rand()),        # length
            int(1 + rng.randint(2)),                 # lanes
            float(speeds[e % 3]),                    # speed -> class = e % 3
        )

    def _pick(base, count, cls=None):
        """``count`` edge ids starting near ``base``; if ``cls`` is given, only
        edges of that class (edge e has class e % 3)."""
        out, e = [], base % n_edge
        while len(out) < count:
            if cls is None or e % 3 == cls:
                out.append(f"e{e}")
            e = (e + 1) % n_edge
        return out

    routes, ffts = {}, {}
    for o in range(n_od):
        od = (o, o)
        rs, ff = [], []
        for k in range(K):
            if segregated:
                cls = k % 3
                # a shared corridor of this class + a route-specific tail
                core = _pick(o * 3, 4, cls)
                tail = _pick(o * 7 + k * 5, 5, cls)
                other = _pick(o * 11 + k, 2)          # a little cross-class content
                path = list(dict.fromkeys(core + tail + other))
            else:
                core = _pick(o * 3, 6)
                tail = _pick(o * 7 + k * 5, 6)
                path = list(dict.fromkeys(core + tail))
            rs.append(path)
            ff.append(float(sum(net[e][0] / net[e][2] for e in path)))
        routes[od] = rs
        ffts[od] = ff
    return net, routes, ffts


def _run_synthetic(beta_true, segregated, n_ep=600, seed=11, forget=1.0):
    """Drive the REAL coordinator over a synthetic city whose congestion law is
    exactly ``beta_true``. Returns (beta_hat_mean, fit_r2, cond)."""
    from pact1.coordinator import Pact1Coordinator
    from pact1.basis import RouteBasis as _RB

    net, routes, ffts = _toy_city(seed=2, n_od=8, K=4, segregated=segregated)
    basis = _RB(net, routes, ffts, n_paths=4, verbose=False)

    rng = np.random.RandomState(seed)
    n_agents, K = 64, 4
    agent_table, av_ids = {}, []
    for i in range(n_agents):
        od = (i % 8, i % 8)
        aid = str(i)
        agent_table[aid] = {"od": od, "start": float(rng.randint(0, 600)),
                            "machine": True}
        av_ids.append(aid)

    cfg = dict(rho=0.8, forget=forget, p0=10.0, gate_abort=False,
               gate_after_episodes=10 ** 9, gate_live_after_episodes=10 ** 9,
               print_every=0, peer_scope="fleet", confidence_gate=True)
    coord = Pact1Coordinator(basis, agent_table, av_ids,
                             {od: ffts[od] for od in ffts}, cfg, run_dir=None)

    fit_hist, cond_hist = [], []
    for ep in range(n_ep):
        coord.begin_episode(ep)
        acts = {a: int(rng.randint(K)) for a in av_ids}      # maximum excitation

        peer_gid = np.array(
            [basis.gid_of[(agent_table[a]["od"], acts[a])] for a in coord.peer_ids],
            dtype=np.int64)
        self_gid = np.array(
            [basis.gid_of[(agent_table[a]["od"], acts[a])] for a in av_ids],
            dtype=np.int64)
        # the same standardisation the coordinator uses, so beta_true is expressed
        # in the units the estimator actually works in
        xz = coord.standardize(
            basis.waveforms(coord.O_rows, peer_gid, self_gid, coord.agent_gids)
        )

        recs = {}
        for i, a in enumerate(av_ids):
            k = acts[a]
            psi = np.array([1.0, xz[0, i, k], xz[1, i, k], xz[2, i, k]])
            y = float(psi @ beta_true + 0.01 * rng.randn())
            recs[a] = (k, coord.agent_fft[i, k] * (1.0 + y))

        coord.end_episode(recs, acts)
        if ep > n_ep - 100:
            if coord._fit_hist:
                fit_hist.append(coord._fit_hist[-1])
    B = np.array([e.beta for e in coord.rls]).mean(axis=0)
    # pooled conditioning on the final day
    cond = coord._last_cond
    coord.close()
    return B, (float(np.mean(fit_hist)) if fit_hist else float("nan")), cond


def test_basis_waveform_arithmetic():
    net, routes, ffts = _toy_city()
    b = RouteBasis(net, routes, ffts, n_paths=4, verbose=False)
    rng = np.random.RandomState(7)
    n_i, n_j = 5, 20
    O = (rng.rand(n_i, n_j) < 0.6).astype(float)
    peer = rng.randint(0, b.R, size=n_j)
    self_gid = peer[:n_i].copy()
    gids = np.stack([np.arange(4) + 4 * (i % b.n_od) for i in range(n_i)])
    fast = b.waveforms(O, peer, self_gid, gids)
    slow = b.waveforms_bruteforce(O, peer, self_gid, gids)
    err = np.max(np.abs(fast - slow))
    assert err < 1e-9, f"vectorised waveform != definition (max err {err:.3e})"


def test_basis_zero_diagonal_and_N1():
    """guide I.10: at N=1 the sums are empty, so a lone traveller reads EXACTLY
    zero peer load -- the category-C signature, checked rather than asserted."""
    net, routes, ffts = _toy_city()
    b = RouteBasis(net, routes, ffts, n_paths=4, verbose=False)
    peer = np.array([3])                       # a single traveller: itself
    O = np.ones((1, 1))
    gids = np.arange(4)[None, :]
    w = b.waveforms(O, peer, np.array([3]), gids)
    assert np.max(np.abs(w)) < 1e-12, f"lone traveller sees load {np.max(np.abs(w)):.3e}"


def test_basis_classes_nondegenerate():
    net, routes, ffts = _toy_city()
    b = RouteBasis(net, routes, ffts, n_paths=4, verbose=False)
    assert b.r == 3
    assert b.class_share.min() > 0.01, f"degenerate class share {b.class_share}"


def test_time_overlap():
    O = build_time_overlap([0.0, 10.0, 100.0], [20.0, 20.0, 5.0])
    assert O[0, 1] == 1.0 and O[1, 0] == 1.0, "overlapping trips not detected"
    assert O[0, 2] == 0.0, "disjoint trips reported as overlapping"
    assert O[0, 0] == 1.0 and O[2, 2] == 1.0, "diagonal must be 1"


# ==========================================================================
#  END TO END -- the one that matters
# ==========================================================================
def test_end_to_end_recovers_a_known_congestion_law():
    """Synthetic city with a KNOWN linear road-class congestion law, and routes
    that differ in road class (the motorway or the back streets).

    If the estimator recovers beta_true and fit_r2 goes high here, the whole
    identify -> forecast -> score chain is correct, and a low fit_r2 on real URB
    data is then a statement about the CITY, not about this code. That distinction
    is the difference between debugging and a finding.
    """
    beta_true = np.array([0.20, 0.60, 0.25, 0.05])       # intercept + 3 classes
    B, r2, cond = _run_synthetic(beta_true, segregated=True)
    err = float(np.abs(B - beta_true).max())
    assert cond < 1e3, f"segregated city should be well conditioned, got {cond:.1f}"
    assert r2 > 0.95, f"fit_r2={r2:.4f} on a perfectly linear city"
    assert err < 0.06, (
        "end-to-end: estimator did not recover the known law. "
        f"beta_hat={np.round(B, 4).tolist()} vs {beta_true.tolist()} "
        f"(max err {err:.4f}, cond {cond:.1f})"
    )


def test_mixed_class_city_is_harder_but_still_identifiable():
    """guide III.6, measured on both regimes -- and the answer is better than the
    guide's SMAC precedent, because of the reference-load centring.

    A city where every route mixes the road classes evenly SHOULD be the hard case:
    the three channels move together, so beta*.psi stays predictable while the
    per-class split goes unidentifiable (what happened on SMAC). Centring the
    regressor on the uniform-mix reference load removes the large common mean that
    caused most of that collinearity, and the split survives: measured here,
    cond 24 -> 336 and beta error 0.001 -> 0.012 going from segregated to mixed --
    fourteen times worse conditioned, still decomposable.

    Two things this test protects:
      * ``cond_psi`` genuinely discriminates between the regimes, so the run-time
        gate means something;
      * the centring is load-bearing. If someone removes it, cond blows up by
        ~400x and this test fails loudly instead of the run quietly reporting a
        decomposition it did not earn.
    """
    beta_true = np.array([0.20, 0.60, 0.25, 0.05])
    B_seg, r2_seg, cond_seg = _run_synthetic(beta_true, segregated=True)
    B_mix, r2_mix, cond_mix = _run_synthetic(beta_true, segregated=False)
    err_seg = float(np.abs(B_seg - beta_true).max())
    err_mix = float(np.abs(B_mix - beta_true).max())

    assert cond_mix > cond_seg, (
        f"cond_psi does not discriminate the regimes "
        f"(segregated {cond_seg:.1f} vs mixed {cond_mix:.1f})"
    )
    assert r2_mix > 0.95, f"prediction failed on the mixed city (r2={r2_mix:.4f})"
    assert err_mix < 0.05, (
        f"the mixed city became UNidentifiable (beta err {err_mix:.4f}). If the "
        "reference-load centring was removed, restore it."
    )
    assert err_seg <= err_mix, "segregated should be the easier regime"
    print(f"        [III.6] segregated cond={cond_seg:.1f} err={err_seg:.4f} | "
          f"mixed cond={cond_mix:.1f} err={err_mix:.4f}")


# ==========================================================================
#  record plumbing -- the silent-failure path
# ==========================================================================
def _write_ep_csv(d, ep, rows):
    import pandas as pd
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, f"ep{ep}.csv"), index=False)


def test_records_from_episode_csvs_are_batched_per_day():
    """A multi-day flush must NOT be merged: pooling days would pair one day's
    travel time with another day's peer waveform, which shows up only as an
    inexplicably low fit_r2."""
    import tempfile
    from pact1.records import RecordSource, split_records

    base = tempfile.mkdtemp()
    eps = os.path.join(base, "episodes")
    for ep in (1, 2, 3):
        _write_ep_csv(eps, ep, [
            {"id": i, "action": (i + ep) % 4, "travel_time": 100.0 + i + ep}
            for i in range(6)
        ])

    src = RecordSource(object(), base)           # a bare object: no memory source
    batches = src.drain(episode=0)
    assert [b[0] for b in batches] == [1, 2, 3], f"batches out of order: {batches}"
    assert all(len(b[1]) == 6 for b in batches), "records lost"

    # already-consumed files must not reappear
    assert src.drain(episode=1) == [], "episode files were re-consumed"
    _write_ep_csv(eps, 4, [{"id": 0, "action": 1, "travel_time": 7.0}])
    assert [b[0] for b in src.drain(episode=2)] == [4], "new episode not picked up"

    av, peers, hdv, bad = split_records(batches[0][1], machine_ids={"0", "1", "2"})
    assert bad == 0
    assert set(av) == {"0", "1", "2"}, f"machine split wrong: {av}"
    assert len(peers) == 6, "humans missing from the peer set"
    assert np.isfinite(hdv), "human travel time not recovered"


def test_records_fall_back_to_memory_and_lock_the_source():
    """The CSV wins when present; memory is the fallback; the choice is LOCKED so a
    day cannot be served twice (once from memory, again when its file lands)."""
    import tempfile
    from pact1.records import RecordSource

    class Env:
        travel_times_list = [{"id": 1, "action": 2, "travel_time": 9.0}]

    base = tempfile.mkdtemp()
    src = RecordSource(Env(), base)
    b = src.drain(episode=7)
    assert src.mode == "memory", f"expected the memory fallback, got {src.mode!r}"
    assert len(b) == 1 and b[0][0] == 7, "memory batch mislabelled"
    assert src.drain(episode=8) == [], "cumulative list re-read from the start"

    # a CSV appearing later must NOT be picked up: the source is locked
    _write_ep_csv(os.path.join(base, "episodes"), 1,
                  [{"id": 1, "action": 2, "travel_time": 9.0}])
    assert src.drain(episode=9) == [], "source switched mid-run -> double counting"


def test_records_prefer_csv_over_memory():
    import tempfile
    from pact1.records import RecordSource

    class Env:
        travel_times_list = [{"id": 1, "action": 2, "travel_time": 9.0}]

    base = tempfile.mkdtemp()
    _write_ep_csv(os.path.join(base, "episodes"), 5,
                  [{"id": 1, "action": 3, "travel_time": 42.0}])
    src = RecordSource(Env(), base)
    b = src.drain(episode=99)
    assert src.mode == "csv", f"CSV should win when present, got {src.mode!r}"
    assert b[0][0] == 5, "CSV batch must be labelled with ITS episode, not the loop's"


def test_coordinator_and_basis_survive_deepcopy():
    """RouteRL deepcopies ``all_agents`` on every episode reset, and each agent
    carries ``agent.model`` -> the coordinator -> an OPEN CSV handle. Without the
    __deepcopy__ short-circuit that raises "cannot pickle 'TextIOWrapper'" and
    kills the run, and it would also copy ~18 MB of Gram matrices per episode."""
    import copy
    import tempfile
    from pact1.coordinator import Pact1Coordinator

    net, routes, ffts = _toy_city(seed=2, n_od=4, K=4)
    basis = RouteBasis(net, routes, ffts, n_paths=4, verbose=False)
    at = {str(i): {"od": (i % 4, i % 4), "start": float(i * 10), "machine": True}
          for i in range(8)}
    cfg = dict(gate_abort=False, print_every=0, peer_scope="fleet",
               debug_dir=tempfile.mkdtemp())
    coord = Pact1Coordinator(basis, at, list(at), {od: ffts[od] for od in ffts},
                             cfg, run_dir=None, exp_id="dc_demo")
    assert coord._dbg is not None, "test is vacuous without an open file handle"

    holder = {"agents": [{"model": {"coord": coord}} for _ in range(4)]}
    dup = copy.deepcopy(holder)                    # must not raise
    assert dup["agents"][0]["model"]["coord"] is coord, \
        "the coordinator was copied instead of shared"
    assert copy.deepcopy(basis) is basis, "the basis was copied instead of shared"
    coord.close()


def test_records_handle_both_key_shapes():
    """In-memory records are keyed by RouteRL's Keychain constants, CSV rows by
    plain column names. One code path must parse both."""
    from pact1.records import split_records

    keys = {"id": ["agent_id", "id"], "action": ["act", "action"],
            "travel_time": ["tt", "travel_time"]}
    mem = [{"agent_id": 3, "act": 1, "tt": 50.0}]
    csv = [{"id": 3, "action": 1, "travel_time": 50.0}]
    a1, p1, _, b1 = split_records(mem, {"3"}, keys)
    a2, p2, _, b2 = split_records(csv, {"3"}, keys)
    assert b1 == 0 and b2 == 0, "records rejected"
    assert a1 == a2 == {"3": (1, 50.0)}, f"{a1} != {a2}"
    assert p1 == p2 == {"3": 1}


def test_records_reject_garbage_without_crashing():
    from pact1.records import split_records
    recs = [None, {}, {"id": 1}, {"id": 2, "action": "x", "travel_time": 1.0},
            {"id": 3, "action": 1, "travel_time": float("nan")},
            {"id": 4, "action": 0, "travel_time": 12.0}]
    av, peers, hdv, bad = split_records(recs, {"4"})
    assert bad >= 3, f"garbage not counted: bad={bad}"
    assert av == {"4": (0, 12.0)}, f"good record lost: {av}"
    assert "3" in peers, "a valid action with a bad travel time should still count "\
                         "as a peer choice"


# ==========================================================================
#  the policy (host PPO + trust head)
# ==========================================================================
def _toy_policy_setup(trust_mode="learned", n_agents=16, seed=5):
    """A coordinator + Pact1PPO fleet on the toy city, with a stand-in for URB's
    ``scripts/iql.py::Network`` (which is injected, so the package never imports
    from scripts/)."""
    import torch
    import torch.nn as nn
    from pact1.coordinator import Pact1Coordinator
    from pact1.policy import Pact1PPO

    class Net(nn.Module):
        def __init__(self, in_size, out_size, num_hidden, widths):
            super().__init__()
            assert len(widths) == num_hidden + 1
            self.i = nn.Linear(in_size, widths[0])
            self.h = nn.ModuleList(
                [nn.Linear(widths[x], widths[x + 1]) for x in range(num_hidden)]
            )
            self.o = nn.Linear(widths[-1], out_size)

        def forward(self, x):
            x = torch.relu(self.i(x))
            for lay in self.h:
                x = torch.relu(lay(x))
            return self.o(x)

    net, routes, ffts = _toy_city(seed=2, n_od=4, K=4)
    basis = RouteBasis(net, routes, ffts, n_paths=4, verbose=False)
    rng = np.random.RandomState(seed)
    agent_table, av_ids = {}, []
    for i in range(n_agents):
        aid = str(i)
        agent_table[aid] = {"od": (i % 4, i % 4),
                            "start": float(rng.randint(0, 600)), "machine": True}
        av_ids.append(aid)
    cfg = dict(rho=0.8, forget=0.999, p0=10.0, gate_abort=False,
               gate_after_episodes=10 ** 9, gate_live_after_episodes=10 ** 9,
               print_every=0, peer_scope="fleet", trust_mode=trust_mode)
    coord = Pact1Coordinator(basis, agent_table, av_ids,
                             {od: ffts[od] for od in ffts}, cfg, run_dir=None)
    obs_dim = 5 + coord.obs_aug_dim
    models = {
        a: Pact1PPO(obs_dim, 4, Net, coord, a, batch_size=8, num_epochs=2,
                    num_hidden=1, widths=(16, 16), lr=0.01)
        for a in av_ids
    }
    return coord, models, av_ids, agent_table, basis, obs_dim, rng


def _drive(coord, models, av_ids, agent_table, basis, obs_dim, rng,
           n_ep=40, learn=True):
    beta_true = np.array([0.2, 0.6, 0.25, 0.05])
    for ep in range(n_ep):
        coord.begin_episode(ep)
        acts = {}
        for a in av_ids:
            obs = np.concatenate([rng.randn(5), np.zeros(coord.obs_aug_dim)])
            obs = coord.augment_obs(a, obs[:5])
            acts[a] = models[a].act(obs, coord.route_context(a))
        peer_gid = np.array(
            [basis.gid_of[(agent_table[a]["od"], acts[a])] for a in coord.peer_ids],
            dtype=np.int64)
        xz = coord.standardize(
            basis.waveforms(coord.O_rows, peer_gid, peer_gid, coord.agent_gids))
        recs = {}
        for i, a in enumerate(av_ids):
            k = acts[a]
            y = float(np.array([1.0, xz[0, i, k], xz[1, i, k], xz[2, i, k]])
                      @ beta_true)
            recs[a] = (k, coord.agent_fft[i, k] * (1.0 + y))
            models[a].push(-recs[a][1] / 100.0)
            if learn:
                models[a].learn()
        coord.end_episode(recs, acts)
    return recs


def test_policy_runs_and_trust_head_receives_gradient():
    """The PPO update must run end to end, and the gradient must actually reach the
    trust head through the logit shift -- that is the whole reason no extra
    log-prob term is needed."""
    import torch
    coord, models, av_ids, at, basis, od, rng = _toy_policy_setup()
    m0 = models[av_ids[0]]
    before = m0.policy_net.o.weight[m0.K].detach().clone()   # the trust-head row
    _drive(coord, models, av_ids, at, basis, od, rng, n_ep=40, learn=True)
    after = m0.policy_net.o.weight[m0.K].detach().clone()

    assert m0.loss, "PPO never performed an update"
    assert all(np.isfinite(v) for v in m0.loss), f"non-finite loss: {m0.loss[:5]}"
    moved = float(torch.abs(after - before).max().item())
    assert moved > 1e-8, (
        "the trust head received NO gradient: the shift is detached from the graph, "
        "so trust can never be learned"
    )
    coord.close()


def test_blind_arm_is_exactly_the_untouched_policy():
    """``trust_mode='off'`` must reproduce plain IPPO EXACTLY through the identical
    wrapper, so a blind-vs-PACT-1 comparison cannot differ by anything else."""
    import torch
    from pact1.policy import apply_shift
    coord, models, av_ids, at, basis, od, rng = _toy_policy_setup(trust_mode="off")
    _drive(coord, models, av_ids, at, basis, od, rng, n_ep=12, learn=True)

    assert float(np.max(np.abs(coord.g_app))) == 0.0, "blind arm applied trust > 0"
    assert float(np.max(np.abs(coord.g_pol))) == 0.0, "blind arm set policy trust > 0"

    m = models[av_ids[0]]
    obs = torch.zeros(1, od)
    with torch.no_grad():
        out = m.policy_net(obs)
        raw = out[:, : m.K]
        ctx = coord.route_context(av_ids[0])
        tt = torch.as_tensor(np.asarray(ctx.tt_hat, dtype=np.float32))[None, :]
        shifted = apply_shift(raw, tt, torch.zeros(1, 1), m.kappa)
    assert torch.equal(shifted, raw), "blind arm still moved the logits"
    coord.close()


# ==========================================================================
def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    print("=" * 74)
    print(f"PACT-1 offline self-test  ({len(tests)} checks, no SUMO required)")
    print("=" * 74)
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed.append((name, str(exc)))
            print(f"  FAIL  {name}\n        {exc}")
        except Exception as exc:                     # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}\n        {type(exc).__name__}: {exc}")
    print("=" * 74)
    if failed:
        print(f"{len(failed)} of {len(tests)} FAILED. Do NOT start a run.")
        return 1
    print(f"All {len(tests)} checks passed. The arithmetic is sound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

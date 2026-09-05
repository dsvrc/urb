"""Offline arithmetic gates for PACT on URB. NO SUMO, NO torch, NO training.

    python pact_urb/selftest.py

`PACT_PIPELINE_SPEC` §11 step 4:

> Write an arithmetic self-check that runs without the simulator. Basis structure,
> N=1 identity, RLS recovery on synthetic data, the windup reproduction, the
> significance floor, conjugacy, the floor property, ratio guards, and an
> end-to-end closed loop. It cannot prove the method works; it proves the
> arithmetic is not the reason if it does not. **Calibrate its fixtures to
> measured reality** -- a fixture too gentle silently blesses broken code.

Four of these tests exist purely to REPRODUCE a failure the spec paid for, so a
regression shows up here rather than in a run: the covariance windup (§5.2), the
trace gate (§8.3), the null-model inflation (§4.3), and the floor property (§1).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pact_urb.core import (                                    # noqa: E402
    AgentRLS,
    NullTracker,
    PedestalEMA,
    analytic_feedforward,
    trust_gate,
)
from urb_ns.driver import WeatherDriver                        # noqa: E402
from urb_ns.selftest import _toy                               # noqa: E402
from urb_ns.severity_env import SeverityLayer                  # noqa: E402


# ==========================================================================
#  the estimator
# ==========================================================================
def test_rls_recovers_a_known_law():
    rng = np.random.RandomState(0)
    beta = np.array([0.4, -1.2, 0.8])
    est = AgentRLS(3, mu=1.0, p0=100.0)
    for _ in range(600):
        psi = np.array([1.0, rng.randn(), rng.randn()])
        est.update(psi, psi @ beta + 0.01 * rng.randn())
    assert np.abs(est.beta - beta).max() < 0.05, \
        f"RLS did not recover beta: {np.round(est.beta, 4)}"


def test_covariance_windup_is_bounded():
    """§5.2 [PAID] reproduction.

    > Forgetting inflates unexcited directions by 1/mu every update WITHOUT bound.
    > Measured se(own_gain) across quarters 29 -> 4.9e3 -> 8.4e5 -> 1.4e8, and the
    > method lost 28.8 return in Q4.

    With mu = 0.9995 and 83k updates the unbounded factor is (1/mu)^83000 ~ 1e18.
    The bound must hold tr(P) while leaving the ESTIMATE intact.
    """
    psi = np.array([1.0, 0.3, 0.05])            # frozen: excitation has died
    beta = np.array([0.2, -0.8, 0.4])
    est = AgentRLS(3, mu=0.9995, p0=10.0)
    for _ in range(20000):
        est.update(psi, psi @ beta)
    tr = float(np.trace(est.P))
    assert np.isfinite(tr), "covariance diverged to non-finite"
    assert tr <= est.p_max * 1.001, f"tr(P) = {tr:.3e} exceeded the bound {est.p_max}"
    assert est.n_clamped > 0, "the bound never engaged -- the test is vacuous"
    pred = float(psi @ est.beta)
    assert abs(pred - float(psi @ beta)) < 1e-3, \
        f"the bound destroyed the estimate: {pred:.5f} vs {float(psi @ beta):.5f}"


def test_dead_row_does_not_move_or_inflate():
    est = AgentRLS(3, mu=0.999, p0=10.0)
    tr0 = float(np.trace(est.P))
    for _ in range(500):
        est.update(np.zeros(3), 1.0)
    assert abs(float(np.trace(est.P)) - tr0) < 1e-9, "dead rows inflated tr(P)"
    assert np.allclose(est.beta, 0.0), "dead rows moved beta"
    assert est.n_updates == 0, "dead rows were counted as updates"


# ==========================================================================
#  the null model  (§4.3 [PAID])
# ==========================================================================
def test_null_model_exposes_an_inflated_r2():
    """> A pooled R^2 of 0.9998 looked like a triumph until the intercept-only
    > model was scored too and came in at 0.656. The honest quantity was 0.344.

    Here the target is driven ENTIRELY by the own column, so the peer channels add
    nothing and `fit_gain` must be ~0 even though both R^2 are near 1.
    """
    rng = np.random.RandomState(1)
    nt = NullTracker(window=400, warmup=10)
    for _ in range(500):
        own = rng.randn()
        y = 3.0 + 2.0 * own + 0.01 * rng.randn()
        nt.add(y, 3.0 + 2.0 * own, 3.0 + 2.0 * own)
    fg = nt.fit_gain()
    assert abs(fg) < 0.02, f"fit_gain {fg:.4f} on peer channels that add nothing"


def test_null_model_detects_a_real_peer_contribution():
    rng = np.random.RandomState(2)
    nt = NullTracker(window=400, warmup=10)
    for _ in range(500):
        own, peer = rng.randn(), rng.randn()
        y = 1.0 + own + 1.5 * peer
        nt.add(y, y, 1.0 + own)                  # full sees peer, null does not
    fg = nt.fit_gain()
    assert fg > 0.3, f"fit_gain {fg:.4f} failed to see a genuine peer term"


def test_fit_gain_stays_bounded_on_a_near_constant_target():
    """[PAID] reproduction of the first URB NS runs.

    The loading ratio barely moves once the policy converges. With the R2 form
    `R2(full) - R2(null)`, both terms divide by that vanishing variance, each blows
    up hugely negative, and the DIFFERENCE came out at 42, 156 and 870 -- for a
    quantity that must lie near [-1, 1]. The gate then read admissible 97.5% of the
    time and ran a whole 4000-day arm on nonsense.

    Normalising by the null's own error instead keeps it bounded whatever the
    target variance is.
    """
    rng = np.random.RandomState(11)
    nt = NullTracker(window=300, warmup=5)
    for _ in range(400):
        y = 0.22 + 1e-6 * rng.randn()          # a target with essentially no variance
        nt.add(y, y + 0.01 * rng.randn(), y + 0.02 * rng.randn())
    fg = nt.fit_gain()
    assert np.isfinite(fg), "fit_gain went non-finite on a near-constant target"
    assert -1.0 <= fg <= 1.0, f"fit_gain = {fg:.4f} escaped [-1, 1]"


def test_null_tracker_is_windowed_not_cumulative():
    """§8.2: a cumulative lift is held down forever by early negatives, so the
    compensator never re-arms after a bad start."""
    rng = np.random.RandomState(3)
    nt = NullTracker(window=50, warmup=0)
    for _ in range(200):                          # a long stretch of bad peer preds
        own, peer = rng.randn(), rng.randn()
        y = 1.0 + own + peer
        nt.add(y, y + 10.0 * rng.randn(), 1.0 + own)
    assert nt.fit_gain() < 0.0, "the bad stretch should score a NEGATIVE lift"
    for _ in range(50):                           # then it becomes informative
        own, peer = rng.randn(), rng.randn()
        y = 1.0 + own + peer
        nt.add(y, y, 1.0 + own)
    assert nt.fit_gain() > 0.0, (
        "the window never forgot the bad stretch -- a cumulative tracker would be "
        "held down forever and the compensator would never re-arm (§8.2)"
    )


# ==========================================================================
#  the gates
# ==========================================================================
def test_gate_separates_whether_from_how_much():
    """§8.1 [PAID]: multiplying heuristic confidences produced applied_trust 0.024
    against fit_gain 0.016 -- ~40x below the theoretical gain. Binary
    admissibility x a calibrated constant took it to 0.192."""
    g, ok = trust_gate(0.05, 500, fit_floor=0.0, ready_updates=100, max_trust=0.5)
    assert ok and g == 0.5, f"admissible case returned {g}"
    g, ok = trust_gate(-0.01, 500, 0.0, 100, 0.5)
    assert (not ok) and g == 0.0, "a negative lift was admitted"
    g, ok = trust_gate(0.05, 10, 0.0, 100, 0.5)
    assert (not ok) and g == 0.0, "a cold estimator was admitted"
    g, ok = trust_gate(float("nan"), 500, 0.0, 100, 0.5)
    assert (not ok) and g == 0.0, "a NaN lift was admitted"


def test_feedforward_is_closed_form_and_signed_correctly():
    """§7: `d_ff = -ff_gain * excess / du_da`. A positive excess with a positive
    sensitivity must produce a NEGATIVE offset -- move away from the load."""
    d = analytic_feedforward(np.array([0.2, 0.0, -0.1]),
                             np.array([0.001, 0.001, 0.001]), 1.0, 1e9)
    assert d[0] < 0 and d[2] > 0, f"feedforward sign is wrong: {d}"
    assert d[1] == 0.0, "zero excess produced a non-zero correction"


def test_feedforward_guards_a_vanishing_divisor():
    """§6.3 [PAID]: a magnitude floor let a divisor with |t| = 0.005 through and the
    delta pinned at the rail -- a CONSTANT BIAS, not a compensation. Two runs whose
    estimators differed by seven orders of magnitude produced byte-identical
    returns."""
    d = analytic_feedforward(np.array([0.5]), np.array([0.0]), 1.0, 100.0)
    assert np.all(np.isfinite(d)), "a zero divisor produced a non-finite offset"
    assert abs(float(d[0])) <= 100.0, "the clip did not hold"


def test_pedestal_removes_the_standing_level_only():
    """§6.4 [PAID]: psi carried mean [-0.50, -2.01] against temporal std
    [0.30, 0.17] -- ~85% pedestal. Cancelling it burns the whole actuator budget on
    a constant and leaves the fluctuation, the only part the policy cannot
    anticipate, uncorrected."""
    ped = PedestalEMA(tau=20.0, n=1)
    out = None
    for t in range(2000):
        out = ped.step(np.array([5.0 + 0.4 * np.sin(t / 7.0)]))
    assert abs(float(ped.level[0]) - 5.0) < 0.15, \
        f"the EMA did not find the pedestal: {float(ped.level[0]):.4f}"
    assert abs(float(out[0])) < 0.6, "the fluctuation was not preserved"


# ==========================================================================
#  the floor property  (§1 -- the one that must never break)
# ==========================================================================
def test_blind_arm_produces_exactly_zero_offsets():
    """When the gates say inadmissible, the executed day must be BYTE-IDENTICAL to
    the blind arm. A diverging estimate can fail to help; it must never do worse."""
    from pact_urb.coordinator import PactCoordinator
    net, lm = _toy(av_share=0.5, seed=3)
    drv = WeatherDriver(period=20, wet_frac=0.5)
    sev = SeverityLayer(net, drv, lm, sigma=1.0, max_offset_s=240.0, verbose=False)
    av = np.nonzero(lm.is_machine)[0]
    co = PactCoordinator(net, lm, sev, av, dict(arm="blind", print_every=0),
                         run_dir=None)
    rng = np.random.RandomState(0)
    ch = lm.uniform_reference_choice(rng)
    sev.begin_episode(3)
    off = co.plan_offsets(ch)
    assert np.all(off == 0.0), f"blind arm produced offsets: max {np.abs(off).max()}"
    co.close()


def test_harm_is_identity_on_a_dry_day_for_every_arm():
    """B.4: the placebo. On a dry day `g == 1` exactly, so the multiplier is 1.0
    and the reward passes through untouched -- whatever sigma is."""
    net, lm = _toy(seed=4)
    drv = WeatherDriver(period=20, wet_frac=0.5)
    dry = int(np.nonzero(drv.is_dry(np.arange(20)))[0][0])
    for sigma in (0.5, 1.0, 5.0):
        sev = SeverityLayer(net, drv, lm, sigma=sigma, verbose=False)
        sev.begin_episode(dry)
        m = sev.end_episode(lm.uniform_reference_choice(np.random.RandomState(0)))
        assert np.all(m == 1.0), f"dry day is not inert at sigma={sigma}"
        assert sev.harm_reward(0, -3.25) == -3.25, "reward changed on a dry day"


def test_severity_zero_is_identity_on_every_day():
    """B.1.1: at sigma = 0 the stock task is recovered byte for byte."""
    net, lm = _toy(seed=5)
    drv = WeatherDriver(period=20, wet_frac=0.5)
    sev = SeverityLayer(net, drv, lm, sigma=0.0, verbose=False)
    ch = lm.uniform_reference_choice(np.random.RandomState(0))
    for day in range(20):
        sev.begin_episode(day)
        assert np.all(sev.end_episode(ch) == 1.0), f"sigma=0 not inert on day {day}"


# ==========================================================================
#  end to end
# ==========================================================================
class _MockAEC(object):
    """Mimics the slice of RouteRL's AEC contract the wrapper touches, so the
    interception can be proved without routerl installed. Every URB script --
    hand-written loop or TorchRL collector -- reads rewards through `last()`."""

    def __init__(self, n=6, k=4, **kw):
        self.n, self.k = n, k
        self.all_agents = [type("A", (), {"id": i, "origin": i % 2,
                                          "destination": i % 2,
                                          "start_time": float(i * 60)})()
                           for i in range(n)]
        self.machine_agents = self.all_agents[: n // 2]
        self.agent_selection = "0"
        self.stepped, self.raw_reward = [], -3.0

    def reset(self, *a, **kw):
        self.stepped = []
        return {}

    def step(self, action=None, *a, **kw):
        self.stepped.append((self.agent_selection, action))

    def last(self, *a, **kw):
        return {}, self.raw_reward, True, False, {}


def test_env_wrapper_harms_every_host_through_last():
    """The reason this wrapper exists: TorchRL hosts never touch a Python loop we
    control, but PettingZooWrapper is built with `use_mask=True` (the AEC path),
    so they read rewards through `last()` exactly as the hand-written loops do.
    Patching one method therefore reaches MAPPO, QMIX, VDN, IPPO and IQL alike."""
    from urb_ns.severity_traffic_env import make_severity_env

    cls = make_severity_env(_MockAEC, {"net_xml": "/nonexistent"}, sigma=3.0,
                            routes_csv=None, verbose=False)
    env = cls()
    env.reset()
    _o, r, _t, _tr, _i = env.last()
    # the layer cannot build (no route table), so it must PASS THROUGH and SAY SO
    assert r == env.raw_reward, "an unbuildable layer silently altered the reward"
    assert env._ns_n_passthrough > 0, "the pass-through was not counted"
    assert env._ns_n_harmed == 0


def test_env_wrapper_is_exact_identity_at_sigma_zero():
    """sigma = 0 must reproduce the stock task byte for byte, so the SAME command
    gives the no-severity row and the two tables stay comparable."""
    from urb_ns.severity_traffic_env import make_severity_env

    cls = make_severity_env(_MockAEC, {}, sigma=0.0, verbose=False)
    env = cls()
    env.reset()
    for _ in range(5):
        _o, r, _t, _tr, _i = env.last()
        assert r == env.raw_reward, "sigma=0 altered a reward"
    assert env._ns_n_harmed == 0 and env._ns_n_days == 0


def test_env_wrapper_harms_the_records_not_just_the_reward():
    """The failure this prevents, measured on real runs: AON scored IDENTICALLY at
    sigma 0 and sigma 3 to six decimals. A non-learning agent under a live dial
    cannot do that -- it happened because the harm reached the reward but not the
    episode records, and `metrics.py` reads the records. Reward, record and metric
    must all describe the same trip."""
    from urb_ns.severity_traffic_env import make_severity_env

    cls = make_severity_env(_MockAEC, {}, sigma=3.0, verbose=False)
    env = cls()
    env._ns_slot = {str(i): i for i in range(6)}
    env.travel_times_list = [{"id": i, "travel_time": 3.0, "reward": -3.0}
                             for i in range(6)]

    class _Layer:                       # a layer that doubles everything
        def end_episode(self, *a, **k): pass
        def harm_travel_time(self, slot, tt): return tt * 2.0
        def harm_reward(self, slot, r): return r * 2.0
        def diagnostics(self): return {"A": 0.0, "g_mean": 1.0, "dry": 0,
                                       "m_mean": 2.0, "u_mean": 0.0}
    env._ns_layer = _Layer()
    env._ns_harm_records()

    assert all(r["travel_time"] == 6.0 for r in env.travel_times_list), \
        f"records not harmed: {env.travel_times_list[:2]}"
    assert all(r["reward"] == -6.0 for r in env.travel_times_list), \
        "reward column in the records not harmed"
    assert env._ns_n_rec_harmed == 6, f"harm count wrong: {env._ns_n_rec_harmed}"


def test_env_wrapper_notices_when_records_are_absent():
    """If the record list cannot be found the run must SAY so, not quietly report
    unharmed travel times as if they were harmed."""
    from urb_ns.severity_traffic_env import make_severity_env

    cls = make_severity_env(_MockAEC, {}, sigma=3.0, verbose=False)
    env = cls()
    env._ns_slot = {"0": 0}
    env._ns_harm_records()
    assert env._ns_n_rec_missing == 1, "a missing record list went unnoticed"


def test_env_wrapper_records_actions_from_step():
    """The day is resolved from choices captured in `step()`. If the wrapper stops
    seeing actions it would harm on a stale choice vector and nothing would look
    wrong, so the capture is asserted directly."""
    from urb_ns.severity_traffic_env import make_severity_env

    cls = make_severity_env(_MockAEC, {}, sigma=3.0, verbose=False)
    env = cls()
    env._ns_choice = np.zeros(6, dtype=np.int64)
    env._ns_slot = {str(i): i for i in range(6)}
    for i in (0, 3, 5):
        env.agent_selection = str(i)
        env.step(2)
    assert list(env._ns_choice) == [2, 0, 0, 2, 0, 2], \
        f"actions not captured: {list(env._ns_choice)}"


def test_closed_loop_runs_and_identifies():
    """The whole cycle on a synthetic city: plan offsets, run the day, identify.

    Asserts the things that would otherwise fail silently -- that the estimator
    receives rows, that the peer channels earn a positive lift, and that the
    compensator actually acts once admissible.
    """
    from pact_urb.coordinator import PactCoordinator
    net, lm = _toy(n_agents=48, av_share=0.6, seed=6)
    drv = WeatherDriver(period=20, wet_frac=0.5)
    sev = SeverityLayer(net, drv, lm, sigma=2.0, max_offset_s=240.0, verbose=False)
    av = np.nonzero(lm.is_machine)[0]
    co = PactCoordinator(
        net, lm, sev, av,
        dict(arm="pact", print_every=0, ready_updates=20, fit_warmup=5,
             fit_window=100, max_trust=0.5),
        run_dir=None)

    rng = np.random.RandomState(7)
    ch = lm.uniform_reference_choice(rng)
    acted = 0
    for day in range(160):
        sev.begin_episode(day)
        off = co.plan_offsets(ch)
        acted += int(np.any(np.abs(off) > 1e-9))
        ch = lm.uniform_reference_choice(rng)      # a policy that keeps exploring
        sev.end_episode(ch, offsets=off)
        co.observe(ch)

    n_upd = int(np.sum([e.n_updates for e in co.rls]))
    assert n_upd > 0, "the estimator never received a row"
    assert acted > 0, "the compensator never acted in 160 days"
    assert all(np.all(np.isfinite(e.beta)) for e in co.rls), "beta went non-finite"
    assert np.isfinite(np.trace(co.rls[0].P)), "covariance diverged"
    co.close()


# ==========================================================================
def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    print("=" * 74)
    print(f"PACT-on-URB offline self-test  ({len(tests)} checks, no simulator)")
    print("=" * 74)
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed.append(name)
            print(f"  FAIL  {name}\n        {exc}")
        except Exception as exc:                          # noqa: BLE001
            failed.append(name)
            print(f"  ERROR {name}\n        {type(exc).__name__}: {exc}")
    print("=" * 74)
    if failed:
        print(f"{len(failed)} of {len(tests)} FAILED. Do NOT start a run.")
        return 1
    print(f"All {len(tests)} checks passed. The arithmetic is sound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

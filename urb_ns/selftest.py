"""Offline arithmetic gates for the URB NS. NO SUMO, NO torch, NO training.

    python urb_ns/selftest.py

`PACT_PIPELINE_SPEC` §11 step 4 and `NS_FORM_SPEC` E.1 both require this before
any method code:

> Write an arithmetic self-check that runs without the simulator. It cannot prove
> the method works; it proves the arithmetic is not the reason if it does not.
> **Calibrate its fixtures to measured reality** -- a fixture too gentle silently
> blesses broken code.

So the synthetic network below is built with link capacities spanning the same
range as saint_arnoult's (1900-5700 veh/h) and an operator spread matched to the
measured 0.96, rather than a uniform toy that would pass anything.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from urb_ns.ceiling import decompose                       # noqa: E402
from urb_ns.driver import HCM_HEAVY_RAIN_LOSS, WeatherDriver  # noqa: E402
from urb_ns.loading import LoadingModel                    # noqa: E402
from urb_ns.network import RoadNetwork, build_co_presence   # noqa: E402


# ==========================================================================
def _toy(n_od=6, K=4, n_edge=48, n_agents=40, seed=0, av_share=0.5):
    """A synthetic city calibrated to saint_arnoult: 1-3 lanes, real speed tiers,
    routes that genuinely overlap."""
    rng = np.random.RandomState(seed)
    speeds = np.array([5.56, 8.33, 13.89, 22.22])
    net_edges = {}
    for e in range(n_edge):
        net_edges[f"e{e}"] = (float(40 + 160 * rng.rand()),
                              int(1 + (e % 3)),
                              float(speeds[e % 4]))
    routes, ffts = {}, {}
    for o in range(n_od):
        od = (o, o)
        rs, ff = [], []
        for k in range(K):
            core = [f"e{(o * 3 + i) % n_edge}" for i in range(5)]
            tail = [f"e{(o * 7 + k * 5 + i) % n_edge}" for i in range(5)]
            path = list(dict.fromkeys(core + tail))
            rs.append(path)
            ff.append(float(sum(net_edges[x][0] / net_edges[x][2] for x in path)
                            / 60.0))          # minutes, as RouteRL stores them
        routes[od], ffts[od] = rs, ff

    net = RoadNetwork(net_edges, routes, ffts, K, verbose=False)
    gids = np.zeros((n_agents, K), dtype=np.int64)
    ff = np.zeros((n_agents, K))
    for i in range(n_agents):
        od = (i % n_od, i % n_od)
        for k in range(K):
            gids[i, k] = net.gid_of[(od, k)]
            ff[i, k] = ffts[od][k]
    start = rng.randint(0, 1800, size=n_agents).astype(float)
    dur = np.nanmean(ff, axis=1) * 60.0
    O = build_co_presence(start, dur)
    is_m = np.zeros(n_agents, dtype=bool)
    is_m[: int(n_agents * av_share)] = True
    return net, LoadingModel(net, gids, ff, start, O, is_m)


# ==========================================================================
#  the declared operator
# ==========================================================================
def test_operator_is_zero_diagonal():
    """A.2.2 / PIPELINE 2.4: `W[i,i] = 0` is ASSERTED, not argued. It is what makes
    the peer sum strictly `j != i`, hence category-C rather than category-B."""
    net, lm = _toy()
    W = net.build_operator(lm.gids, lm.O)
    assert np.all(np.diag(W) == 0.0), "W has a non-zero diagonal"
    assert np.all(np.isfinite(W)), "W has non-finite entries"


def test_operator_is_asymmetric_and_spread():
    """PIPELINE 2.1 [PAID]: a flat geometric proxy measured fit_gain = -0.0045.
    The real operator's weights span orders of magnitude and are asymmetric; a
    fixture that passed with a uniform operator would bless broken code."""
    net, lm = _toy()
    s = net.summary()
    assert s["op_spread"] > 0.2, f"operator weights too flat: {s['op_spread']}"
    assert s["asymmetry"] > 1e-3, f"operator is symmetric: {s['asymmetry']}"
    assert s["cap_max"] > s["cap_min"], "all links have identical capacity"


def test_n1_gives_exactly_zero_peer_load():
    """The category-C signature (A.3): at N=1 the peer sum is empty, so the
    cross-agent contribution is EXACTLY zero however small `g` becomes."""
    net, lm = _toy()
    lone = np.zeros(lm.N, dtype=bool)
    lone[0] = True
    for g in (1.0, 0.5, 0.05):
        u, _ = lm.loading(lm.uniform_reference_choice(np.random.RandomState(1)),
                          g=g, subset=lone)
        assert abs(float(u[0])) < 1e-12, (
            f"a lone traveller reads peer load {float(u[0]):.3e} at g={g}"
        )


# ==========================================================================
#  the dial
# ==========================================================================
def test_dial_identity_at_zero_is_exact():
    """B.1.1. Not approximately -- this is what lets the stock task be claimed as
    recovered byte for byte, and it is what makes the placebo regime free."""
    d = WeatherDriver(period=60, wet_frac=0.5)
    g = d.g(np.arange(60), 0.0)
    assert np.all(g == 1.0), f"max deviation {float(np.max(np.abs(g - 1.0))):.3e}"
    sens = np.linspace(0.4, 2.0, 11)
    assert np.all(d.g(np.arange(60), 0.0, sens=sens) == 1.0), \
        "identity fails elementwise once the derating is per-link"


def test_dial_is_monotone_and_never_generous():
    """B.1.2 and B.1.3. The uprating trap: on POWER a two-sided physical law
    scaled by sigma applied x1.269 at sigma=2 and made the task strictly EASIER."""
    d = WeatherDriver(period=60, wet_frac=0.5)
    days = np.arange(60)
    cols = np.stack([d.g(days, s) for s in (0.0, 0.5, 1.0, 2.0, 4.0)])
    assert np.all(np.diff(cols, axis=0) <= 1e-12), "not monotone in sigma"
    assert np.all(cols <= 1.0 + 1e-12), f"g exceeded 1: {float(cols.max())}"
    assert np.all(cols > 0.0), "capacity driven to zero -- the medium must stay open"


def test_placebo_regime_is_exactly_inert():
    """B.4, the strongest single defence: a naturally occurring regime where the
    dial provably does nothing.

    > A reviewer alleging a rigged knob then has to explain why the rig switches
    > itself off when it is not raining.
    """
    d = WeatherDriver(period=100, wet_frac=0.5)
    dry = d.is_dry(np.arange(100))
    assert dry.sum() == 50, f"expected a 50-day dry season, got {int(dry.sum())}"
    for s in (0.5, 1.0, 2.0, 5.0):
        g = d.g(np.arange(100), s)
        assert np.all(g[dry] == 1.0), f"dial is live in dry weather at sigma={s}"
        assert np.any(g[~dry] < 1.0), f"dial is inert in WET weather at sigma={s}"


def test_dial_anchored_to_hcm():
    """B.1.4 / B.3: sigma=1 must correspond to something real, so the headline
    experiment needs no defence. Here it is the HCM heavy-rain capacity
    adjustment factor."""
    d = WeatherDriver(period=40, wet_frac=0.5)
    peak = float(np.min(d.g(np.arange(40), 1.0)))
    assert abs(peak - (1.0 - HCM_HEAVY_RAIN_LOSS)) < 1e-9, (
        f"sigma=1 peak derating {peak:.4f} does not match the HCM factor "
        f"{1.0 - HCM_HEAVY_RAIN_LOSS:.4f}"
    )


# ==========================================================================
#  the harm channel
# ==========================================================================
def test_harm_is_identity_when_the_dial_is_off():
    """The delay multiplier must be EXACTLY 1.0 at g=1, elementwise -- on dry days
    and at sigma=0 alike. Anything else and the stock task is not recovered."""
    net, lm = _toy()
    ch = lm.uniform_reference_choice(np.random.RandomState(2))
    m, _, _ = lm.delay_multiplier(ch, g=1.0)
    assert np.all(m == 1.0), f"max deviation {float(np.max(np.abs(m - 1.0))):.3e}"


def test_harm_is_monotone_in_severity():
    """G3: a dial that does not hurt is not a dial."""
    net, lm = _toy()
    d = WeatherDriver(period=40, wet_frac=0.5)
    ch = lm.uniform_reference_choice(np.random.RandomState(3))
    peak = int(np.argmax(d.A(np.arange(40))))
    prev = 1.0
    for s in (0.0, 0.5, 1.0, 2.0):
        g = d.g(peak, s, sens=net.rain_sens)
        m, _, _ = lm.delay_multiplier(ch, g=g)
        cur = float(np.mean(m[lm.is_machine]))
        assert cur >= prev - 1e-12, f"harm fell from {prev:.5f} to {cur:.5f}"
        prev = cur
    assert prev > 1.005, f"the dial barely bites: mean multiplier {prev:.5f}"


def test_sensor_uses_the_binding_link_not_the_mean():
    """A.4 [PAID]: on POWER, switching from mean to max raised applied
    compensation 6x. Congestion is a property of the worst element."""
    net, lm = _toy()
    ch = lm.uniform_reference_choice(np.random.RandomState(4))
    u, binding = lm.loading(ch, g=1.0)
    contrib = lm.link_flow(ch)
    veh = lm.O @ contrib - contrib
    T = lm.trip_h[np.arange(lm.N), ch]
    ratio = (veh / T[:, None]) / net.cap[None, :]
    gid = lm.gids[np.arange(lm.N), ch]
    ratio = np.where(net.A[gid] > 0, ratio, -np.inf)
    assert np.allclose(u, ratio.max(axis=1)), "the sensor is not reading the max"
    assert np.all(ratio[np.arange(lm.N), binding] == ratio.max(axis=1)), \
        "the reported binding link is not the argmax"


# ==========================================================================
#  Part C
# ==========================================================================
def test_ceiling_shares_are_a_partition():
    """C.1: fixed / own / peer partition the excess, so the three fractions must
    sum to 1 and the decentralized ceiling must be its complement."""
    net, lm = _toy(av_share=0.5)
    d = WeatherDriver(period=20, wet_frac=0.5)
    r = decompose(lm, d, 1.0, n_draws=3, seed=0)
    assert not r["degenerate"], "no excess produced -- the fixture is too gentle"
    tot = r["irreducible"] + r["own"] + r["peer"]
    assert abs(tot - 1.0) < 1e-6, f"shares sum to {tot:.6f}, not 1"
    assert abs(r["decentralized_ceiling"] - (1 - r["irreducible"])) < 1e-9
    assert abs(r["coordination_gap"] - r["peer"]) < 1e-12


def test_all_machine_fleet_has_no_irreducible_share():
    """With every traveller controllable there is no `L_fixed`, so the irreducible
    fraction must be exactly zero and the decentralized ceiling exactly 1."""
    net, lm = _toy(av_share=1.0)
    d = WeatherDriver(period=20, wet_frac=0.5)
    r = decompose(lm, d, 1.0, n_draws=3, seed=0)
    assert r["irreducible"] < 1e-9, f"irreducible {r['irreducible']:.4f} with no humans"
    assert r["decentralized_ceiling"] > 1 - 1e-9


def test_coordination_gap_grows_with_fleet_share():
    """C.4's falsifiable N-scaling prediction, and the reason URB should differ
    from POWER: more agents over one medium leaves each less local authority.

    > No competing credit-assignment method predicts this.
    """
    d = WeatherDriver(period=20, wet_frac=0.5)
    gaps = []
    for share in (0.2, 0.5, 1.0):
        _, lm = _toy(av_share=share, seed=5)
        gaps.append(decompose(lm, d, 1.0, n_draws=3, seed=0)["coordination_gap"])
    assert gaps[0] < gaps[-1], f"gap did not grow with share: {np.round(gaps, 4)}"
    assert all(b >= a - 1e-6 for a, b in zip(gaps, gaps[1:])), \
        f"gap is not monotone in share: {np.round(gaps, 4)}"


def test_dry_season_produces_no_excess_to_attribute():
    """A dial that is inert must yield no decomposition at all, rather than a
    spurious one built out of numerical noise."""
    net, lm = _toy()
    d = WeatherDriver(period=20, wet_frac=0.5)
    dry = np.nonzero(d.is_dry(np.arange(20)))[0]
    r = decompose(lm, d, 2.0, days=dry, n_draws=2, seed=0)
    assert r["degenerate"], "the dry season produced an excess"


# ==========================================================================
def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    print("=" * 74)
    print(f"URB-NS offline self-test  ({len(tests)} checks, no simulator)")
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
    print(f"All {len(tests)} checks passed. The NS arithmetic is sound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

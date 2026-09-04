"""PART C -- the ceiling decomposition and the COORDINATION GAP.

`NS_FORM_SPEC` Part C, which its own preamble calls the most valuable part of the
design, and §11/E.1 both say to compute it FIRST:

> Read this before running anything. At sigma=1 only 9.5% of the damage requires
> coordination -- the rest each agent can fix alone. That single number bounded
> every result in the project and explained nine runs' worth of margins sitting
> inside the noise.
>
> Use it to choose the environment, not to excuse the outcome.

THE DECOMPOSITION
--------------------------------------------------------------------------------
Under the dial each traveller's loading exceeds its `sigma = 0` counterfactual by

    Delta_i = u_i * (1 - g)

and every unit of that excess traces to a contributor. Contributors partition by
**who can move them**:

    Delta_fixed   load from travellers no AV controls (the human drivers)
    Delta_own     the AV's own vehicle, amplified by 1/g
    Delta_peer    other AVs' vehicles through W, amplified by 1/g

    irreducible              = Delta_fixed / Delta_total
    decentralized ceiling    = 1 - Delta_fixed / Delta_total
    non-coordinating ceiling = 1 - (Delta_fixed + Delta_peer) / Delta_total
    COORDINATION GAP         = Delta_peer / Delta_total     <- what PACT claims

All four are computed from the declared operator and the fixed schedule, with **no
training and no method** -- which is what makes them a prediction rather than a
post-hoc explanation.

WHY URB SHOULD DIFFER SHARPLY FROM POWER
--------------------------------------------------------------------------------
C.4 names the knob: the gap is small when each agent has ample LOCAL authority
over its own binding element, and widens when agents are individually weak
relative to the coupling.

  POWER   11 zone agents, each with continuous control of the generators feeding
          its own binding line, and `PTDF[own line, own gen]` is the largest entry
          in its row. Measured `Delta_own = 76.9%`, gap 9.5%.

  URB     88 AVs, each with 4 discrete route options and ONE vehicle. An agent
          cannot decongest its own road: its own car is one of dozens sharing the
          bottleneck. `Delta_own` should be ~1/(co-present vehicles).

So the prediction, before the number is read: **URB's coordination gap should be
several times POWER's.** That is C.4's N-scaling claim, and `gap_vs_n` tests it
with no training at all.
"""

import numpy as np


def decompose(loading, driver, sigma, days=None, n_draws=16, seed=0,
              choice_fn=None):
    """Attribute the dial-induced loading excess to fixed / own / peer.

    Args:
        loading:   a built `LoadingModel`.
        driver:    a built `WeatherDriver`.
        sigma:     severity.
        days:      day indices to average over. Default: one full driver cycle,
                   so wet and dry are weighted as they actually occur.
        n_draws:   route-choice draws from the geometric reference (C.2's
                   "peers acting uniformly at random").
        choice_fn: optional callable(rng) -> (N,) choices, to score a real policy
                   instead of the reference.

    Returns a dict of fractions plus the raw excess levels.
    """
    rng = np.random.RandomState(seed)
    if days is None:
        days = np.arange(driver.period)
    days = np.asarray(days)

    av = loading.is_machine
    hum = ~av
    n_av = int(av.sum())
    if n_av == 0:
        raise ValueError("[URB-NS] no machine agents: nothing to decompose")

    tot = own = peer = fixed = 0.0
    n = 0
    for _ in range(int(n_draws)):
        choice = (choice_fn(rng) if choice_fn is not None
                  else loading.uniform_reference_choice(rng))
        for day in days:
            # per-link derating: HCM factors are facility-specific, so the binding
            # link can shift with severity and the shares move with sigma
            g = np.asarray(driver.g(day, sigma, sens=loading.net.rain_sens),
                           dtype=np.float64)
            if np.all(g >= 1.0):
                continue                      # dry day: no excess to attribute
            # total loading under the dial, and the sigma=0 counterfactual
            u_all, _ = loading.loading(choice, g=g)
            u_0, _ = loading.loading(choice, g=1.0)
            d_tot = np.maximum(u_all - u_0, 0.0)[av]

            # the same excess with only ONE class of contributor present.
            # Excess is proportional to the load, so a class's share of the load
            # is its share of the excess -- attribution is exact, not heuristic.
            u_f, _ = loading.loading(choice, g=g, subset=hum)
            u_f0, _ = loading.loading(choice, g=1.0, subset=hum)
            d_fix = np.maximum(u_f - u_f0, 0.0)[av]

            u_p, _ = loading.loading(choice, g=g, subset=av)
            u_p0, _ = loading.loading(choice, g=1.0, subset=av)
            d_peer = np.maximum(u_p - u_p0, 0.0)[av]

            # own: the agent's own vehicle. `loading` already excludes self from
            # the peer sum, so the own term is the diagonal contribution, obtained
            # by letting i load itself.
            d_own = _own_excess(loading, choice, g)[av]

            tot += float(np.nansum(d_tot))
            fixed += float(np.nansum(d_fix))
            peer += float(np.nansum(d_peer))
            own += float(np.nansum(d_own))
            n += 1

    if tot <= 0:
        return {"sigma": float(sigma), "n_av": n_av, "degenerate": True,
                "irreducible": float("nan"), "own": float("nan"),
                "peer": float("nan"), "decentralized_ceiling": float("nan"),
                "delta_total": 0.0, "n_cells": n}

    # the three classes are a partition of the load, so renormalise any residual
    # from the max(0, .) guards rather than letting the fractions miss 1.0
    s = fixed + own + peer
    scale = tot / s if s > 0 else 1.0
    f_fix, f_own, f_peer = (fixed * scale / tot, own * scale / tot,
                            peer * scale / tot)

    return {
        "sigma": float(sigma),
        "n_av": n_av,
        "degenerate": False,
        "irreducible": f_fix,
        "own": f_own,
        "peer": f_peer,
        "decentralized_ceiling": 1.0 - f_fix,
        "non_coordinating_ceiling": 1.0 - (f_fix + f_peer),
        "coordination_gap": f_peer,
        "delta_total": tot / max(1, n),
        "n_cells": n,
    }


def _own_excess(loading, choice, g):
    """The excess an agent causes itself: its own vehicle on its own bottleneck.

    `LoadingModel.loading` subtracts the self term to keep the peer sum strictly
    `j != i` (the category-C signature), so the own contribution is recovered here
    explicitly rather than being left implicit.
    """
    choice = np.asarray(choice, dtype=np.int64)
    idx = np.arange(loading.N)
    gid = loading.gids[idx, choice]
    own_link = loading.net.A[gid]                        # (N, E) own route links
    T = np.maximum(loading.trip_h[idx, choice], loading.min_trip_h)
    flow = own_link / T[:, None]                          # one vehicle, own trip
    cap = loading.net.cap[None, :]
    r_g = flow / np.maximum(cap * g, 1e-9)
    r_0 = flow / np.maximum(cap, 1e-9)
    d = np.maximum(r_g - r_0, 0.0)
    # attribute at the link that binds under the dial
    _, binding = loading.loading(choice, g=g)
    return d[idx, binding]


# ==========================================================================
def gap_vs_n(build_fn, driver, sigma, shares, seed=0, n_draws=8, verbose=True):
    """C.4's N-scaling prediction, tested with NO TRAINING.

    > The coordination gap should grow with N, and theory_ceiling-style computation
    > tests it in minutes with no training. No competing credit-assignment method
    > predicts this.

    Args:
        build_fn: callable(share) -> a `LoadingModel` with that AV share.
        shares:   AV shares to sweep, e.g. (0.1, 0.2, 0.4, 0.6, 0.8, 1.0).

    Returns a list of decomposition dicts, one per share.
    """
    out = []
    for share in shares:
        lm = build_fn(share)
        d = decompose(lm, driver, sigma, n_draws=n_draws, seed=seed)
        d["av_share"] = float(share)
        out.append(d)
        if verbose:
            if d["degenerate"]:
                print(f"[URB-NS] share={share:.2f}  n_av={d['n_av']:4d}  "
                      "DEGENERATE (no excess -- is sigma 0, or all days dry?)")
            else:
                print(f"[URB-NS] share={share:.2f}  n_av={d['n_av']:4d}  "
                      f"irreducible {d['irreducible'] * 100:5.1f}%  "
                      f"own {d['own'] * 100:5.1f}%  "
                      f"PEER {d['coordination_gap'] * 100:5.1f}%  "
                      f"decentr ceiling {d['decentralized_ceiling'] * 100:5.1f}%")
    return out


def format_table(rows, title="ceiling decomposition"):
    """The Part C table, in the spec's own layout."""
    lines = [f"", f"  {title}", "  " + "-" * 68,
             "   sigma   irreducible   own (free)   PEER (coord)   decentr ceiling"]
    for d in rows:
        if d.get("degenerate"):
            lines.append(f"  {d['sigma']:5.2f}       --            --             "
                         "--              --      (no excess)")
            continue
        lines.append(
            f"  {d['sigma']:5.2f}      {d['irreducible'] * 100:5.1f}%       "
            f"{d['own'] * 100:5.1f}%         {d['coordination_gap'] * 100:5.1f}%"
            f"          {d['decentralized_ceiling'] * 100:5.1f}%"
        )
    return "\n".join(lines)

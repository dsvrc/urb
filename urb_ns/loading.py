"""The exertion functional `Phi` and the loading ratio `u`.

`NS_FORM_SPEC` A.2.1 -- the loading ratio, and A.5 -- the exertion functional.

THE LOADING RATIO
--------------------------------------------------------------------------------
    u_i = max over links a on i's route of   flow_a / (capacity_a * g)

**MAX, not mean.** A.4, paid for on POWER: switching the sensor from mean `rho`
over a zone's lines to max raised applied compensation **6x** at identical
settings. Congestion is a property of the worst element; averaging over dozens of
links dilutes the signal toward zero. In traffic the binding link is called the
bottleneck, so this is also the domain's own language.

Flow is a rate, so a co-present vehicle count is divided by the trip duration:

    flow_a = (# co-present vehicles using a) / T_i        [veh/h]

which puts `u` on the same footing as the HCM volume-to-capacity ratio.

THE EXERTION FUNCTIONAL
--------------------------------------------------------------------------------
    Phi_j = free-flow duration of j's chosen route, normalised

A.5 demands uncancellable, a magnitude, and varying:

  * **uncancellable** -- demand is exogenous in URB. Every traveller must make its
    trip from its own origin to its own destination at its own scheduled time.
    There is no "stop drawing" move. The fleet can only redistribute exertion, not
    remove it, and the redistribution that minimises each agent's own `Phi`
    concentrates everyone on the same fast links, which RAISES the coupling. Both
    directions cost reward, which is the A.5 bar.
  * **a magnitude** -- a duration, never a signed sum, so there is no cancelling
    configuration to find (the Ant anti-symmetric-gait escape).
  * **varying** -- across route options and across agents. `std/mean` is measured
    and reported; A.5 wants > 0.05.

Read from the EXECUTED action, because the channel is loop-coupled (A.6):
shifting departure to compensate changes who you share the road with.
"""

import numpy as np

SECONDS_PER_HOUR = 3600.0


class LoadingModel(object):
    """Turns route choices into per-agent loading ratios on the declared medium."""

    # DECLARED harm-function constants. See `bpr` for the regime argument and the
    # one-line calibration procedure. Both belong in the ablation table.
    BPR_ALPHA = 2.28          # calibrated: observed mean excess 0.50 / mean v/c 0.219
    BPR_BETA = 1.0            # low-utilisation regime, NOT the congested beta=4

    def __init__(self, network, agent_gids, agent_ffts, start_times,
                 co_presence, is_machine, min_trip_h=1.0 / 60.0,
                 bpr_alpha=None, bpr_beta=None):
        """
        Args:
            network:      a built `RoadNetwork`.
            agent_gids:   (N, K) global route id of each traveller's options.
            agent_ffts:   (N, K) free-flow time per option, in the env's units
                          (RouteRL stores MINUTES).
            start_times:  (N,) departure seconds from `agents.csv`.
            co_presence:  (N, N) fixed temporal overlap, diagonal 1.
            is_machine:   (N,) bool -- which travellers are controllable AVs.
            min_trip_h:   floor on trip duration when converting to a flow rate.
        """
        self.net = network
        self.gids = np.asarray(agent_gids, dtype=np.int64)
        self.ffts = np.asarray(agent_ffts, dtype=np.float64)
        self.start = np.asarray(start_times, dtype=np.float64)
        self.O = np.asarray(co_presence, dtype=np.float64)
        self.is_machine = np.asarray(is_machine, dtype=bool)
        self.N, self.K = self.gids.shape
        self.min_trip_h = float(min_trip_h)
        self.bpr_alpha = self.BPR_ALPHA if bpr_alpha is None else float(bpr_alpha)
        self.bpr_beta = self.BPR_BETA if bpr_beta is None else float(bpr_beta)

        # trip duration in hours, per option (RouteRL free-flow times are minutes)
        self.trip_h = np.maximum(self.ffts / 60.0, self.min_trip_h)

        # ---- Phi: normalised exertion per option ------------------------------
        # geometric reference = the mean over route options, so no run data enters
        self.phi_opt = self.ffts / max(1e-9, float(np.nanmean(self.ffts)))
        self.phi_ref = np.nanmean(self.phi_opt, axis=1)          # uniform-random
        self.phi_max = np.nanmax(self.phi_opt, axis=1)

    # ------------------------------------------------------------------ Phi
    def phi_spread(self):
        """`std(Phi)/mean(Phi)` over all travellers and options. A.5 wants > 0.05;
        a perfectly constant `Phi` is uncancellable but UNIDENTIFIABLE."""
        v = self.phi_opt[np.isfinite(self.phi_opt)]
        return float(v.std() / max(1e-12, v.mean()))

    # ------------------------------------------------------------------ loading
    def link_flow(self, choice, weight=None):
        """Per-link vehicle count contributed by each traveller's chosen route.

        Returns (N, E): row j is j's own contribution, so any subset sum is exact
        and the own/peer/fixed split of Part C is a partition by construction.
        """
        choice = np.asarray(choice, dtype=np.int64)
        gid = self.gids[np.arange(self.N), choice]
        contrib = self.net.A[gid]                              # (N, E) binary
        if weight is not None:
            contrib = contrib * np.asarray(weight, float)[:, None]
        return contrib

    def co_presence(self, offsets=None, slack=0.0, row_shift=0.0):
        """Temporal overlap, optionally with per-traveller DEPARTURE OFFSETS.

        `row_shift` moves ONLY the row agent's own interval, leaving every peer
        where it was. That is what a PARTIAL derivative needs: co-presence is
        translation-invariant, so shifting everyone together changes nothing at
        all, and a finite difference taken that way is exactly zero. (Which is also
        the physical statement that a uniform departure shift across the whole
        fleet accomplishes nothing -- the compensation has to be DIFFERENTIAL, and
        that is the commons in miniature.)

        With `offsets=None` this is the fixed schedule matrix built once at
        construction. With offsets it is recomputed, which is what makes departure
        time a control input: shifting when you travel changes who you share the
        road with, hence your loading.

        It is also what makes the channel LOOP-COUPLED (A.6) -- your offset moves
        every peer's co-presence too, so compensating feeds the medium it
        compensates against. Everyone leaving earlier rebuilds the peak; that is
        the Vickrey bottleneck, and it is why T4 applies here.
        """
        if offsets is None and not row_shift:
            return self.O
        off = (np.zeros(self.N) if offsets is None
               else np.asarray(offsets, dtype=np.float64))
        T = np.maximum(np.nanmean(self.ffts, axis=1) * 60.0, 0.0)  # min -> s
        s_col = self.start + off                 # peers, at their own departures
        e_col = s_col + T + float(slack)
        s_row = s_col + float(row_shift)         # ROW ONLY: agent i, shifted
        e_row = s_row + T + float(slack)
        return ((s_row[:, None] <= e_col[None, :])
                & (s_col[None, :] <= e_row[:, None])).astype(np.float64)

    def loading(self, choice, g=1.0, subset=None, offsets=None, row_shift=0.0):
        """`u_i` for every AV: the binding link's flow-to-capacity ratio.

        Args:
            choice:  (N,) executed route option per traveller.
            g:       capacity multiplier from the driver (scalar or (E,)).
            subset:  optional bool (N,) restricting WHICH travellers contribute
                     load. Used by Part C to attribute the excess.
            offsets: optional (N,) departure shifts in seconds -- the compensation
                     channel.

        Returns (u, binding_link) with `u` shape (N,) and the argmax link index.
        """
        choice = np.asarray(choice, dtype=np.int64)
        contrib = self.link_flow(choice)                       # (N, E)
        if subset is not None:
            contrib = contrib * np.asarray(subset, bool)[:, None]

        # vehicles co-present with i, on each link
        O = self.co_presence(offsets, row_shift=row_shift)
        veh = O @ contrib                                      # (N, E)
        own = self.link_flow(choice)                           # i's own row
        if subset is None:
            veh = veh - own                                    # strict j != i
        else:
            veh = veh - own * np.asarray(subset, bool)[:, None]

        T = self.trip_h[np.arange(self.N), choice]             # (N,)
        cap = self.net.cap[None, :] * np.asarray(g)            # (1, E) or (N, E)
        flow = veh / np.maximum(T, self.min_trip_h)[:, None]
        ratio = flow / np.maximum(cap, 1e-9)

        # only the links on i's own route can bind
        gid = self.gids[np.arange(self.N), choice]
        mask = self.net.A[gid] > 0
        ratio = np.where(mask, ratio, -np.inf)
        binding = np.argmax(ratio, axis=1)
        u = ratio[np.arange(self.N), binding]
        return np.where(np.isfinite(u), u, 0.0), binding

    # ------------------------------------------------------------------ harm
    def bpr(self, u, alpha=None, beta=None):
        """The BPR congestion function: `t/t_free = 1 + alpha (v/c)^beta`.

        The single most standard object in travel-demand modelling, and the harm
        channel of this NS: derating capacity raises `v/c`, which raises travel
        time.

        ** THE EXPONENT MUST MATCH THE OPERATING REGIME, AND URB IS NOT CONGESTED. **

        The classic Bureau of Public Roads coefficients are `alpha = 0.15,
        beta = 4`, calibrated for facilities near capacity. Measured on every
        network URB ships, the fleet operates two orders of magnitude BELOW link
        capacity -- `v/c` of order 0.05-0.35, and the most loaded instance
        (ingolstadt, 1035 travellers) estimates ~32 veh/h/lane against a
        junction-constrained approach capacity of ~600. A quartic term is
        numerically dead there: a 14% capacity loss at `v/c = 0.2` changes travel
        time by 0.02%.

        In the low-utilisation regime the correct form is LINEAR. Queueing theory
        gives it directly: M/M/1 delay is `rho/(1-rho)`, which is `~rho` for small
        `rho`, and delay from discrete interactions (junction conflicts, following)
        is proportional to the number of interactions, hence to co-present vehicle
        count. So `beta = 1` is not a convenience -- it is the regime's own
        exponent, and BPR variants with `beta = 1..2` are standard for low-flow
        arterials.

        `alpha` is then CALIBRATED ONCE against a measured URB quantity and
        declared: base-URB runs show mean realized/free-flow - 1 = 0.50, and the
        mean binding `v/c` is 0.219, so `alpha = 0.50 / 0.219 = 2.28`. That is a
        stated one-line procedure against an observable, not a knob tuned toward a
        result -- and it belongs in the ablation table (guide II.1).
        """
        u = np.maximum(np.asarray(u, dtype=np.float64), 0.0)
        a = self.bpr_alpha if alpha is None else float(alpha)
        b = self.bpr_beta if beta is None else float(beta)
        return 1.0 + a * u ** b

    def du_d_offset(self, choice, g, offsets=None, h=30.0):
        """`du_i / d(offset_i)` from the DECLARED model, by central difference.

        `PACT_PIPELINE_SPEC` §6.1 [PAID] is emphatic that this divisor must come
        from the operator and NOT be learned:

        > Learning it failed completely: own_gain = 0.13 with se = 5.0, i.e.
        > |t| = 0.024, never once clearing a t > 3 bar in 5004 windows, because
        > excitation dies as the policy converges.

        Here the whole loading model is declared -- network geometry plus a fixed
        schedule -- so the sensitivity is available analytically. `h` is the finite
        -difference step in seconds; co-presence is a step function of interval
        overlap, so `h` must be large enough to move at least one peer across the
        boundary. Declared constant.
        """
        base = np.zeros(self.N) if offsets is None else np.asarray(offsets, float)
        up, _ = self.loading(choice, g=g, offsets=base, row_shift=+h)
        dn, _ = self.loading(choice, g=g, offsets=base, row_shift=-h)
        return (up - dn) / (2.0 * h)

    def delay_multiplier(self, choice, g, alpha=None, beta=None, offsets=None):
        """How much longer the trip takes under the dial than at `sigma = 0`.

            m_i = BPR(u_i under derated capacity) / BPR(u_i at nominal capacity)

        **Exactly 1.0 when `g == 1`**, elementwise, which is `NS_FORM_SPEC` B.1.1:
        the stock task is recovered byte for byte at `sigma = 0` and on every dry
        day. That identity is why the placebo regime is free.
        """
        u_g, binding = self.loading(choice, g=g, offsets=offsets)
        u_0, _ = self.loading(choice, g=1.0, offsets=offsets)
        return self.bpr(u_g, alpha, beta) / self.bpr(u_0, alpha, beta), u_g, binding

    def operating_point(self, g=1.0, n_draws=8, seed=0, alpha=None, beta=None,
                        sigma_probe=1.0, loss=0.14):
        """Is the medium loaded enough for the dial to do anything?

        `NS_FORM_SPEC` G5 asks whether loading rises toward the limit -- because
        the peer term scales as `1/g`, and a medium sitting far below capacity
        cannot express that however large `sigma` grows. BPR is quartic, so the
        response to a capacity loss is negligible below `v/c ~ 0.6` and only
        becomes material approaching 1.

        Returns the `v/c` distribution over AVs and the travel-time response a
        `sigma_probe` capacity loss would actually produce.
        """
        rng = np.random.RandomState(seed)
        us = []
        for _ in range(int(n_draws)):
            u, _ = self.loading(self.uniform_reference_choice(rng), g=g)
            us.append(u[self.is_machine])
        u = np.concatenate(us)
        u = u[np.isfinite(u)]
        keep = 1.0 - float(sigma_probe) * float(loss)
        resp = self.bpr(u / max(keep, 1e-6), alpha, beta) / self.bpr(u, alpha, beta)
        return {
            "vc_mean": float(u.mean()), "vc_p50": float(np.percentile(u, 50)),
            "vc_p90": float(np.percentile(u, 90)), "vc_max": float(u.max()),
            "tt_response_mean": float(resp.mean()),
            "tt_response_p90": float(np.percentile(resp, 90)),
            "tt_response_max": float(resp.max()),
            "frac_above_0p6": float(np.mean(u > 0.6)),
            "sigma_probe": float(sigma_probe),
        }

    # ------------------------------------------------------------------ helpers
    def uniform_reference_choice(self, rng=None):
        """The geometric reference: every traveller drawn uniformly over its own
        route options. Used everywhere a quantity must be computable BEFORE
        training, with no policy and no run data."""
        rng = rng or np.random.RandomState(0)
        return rng.randint(0, self.K, size=self.N)

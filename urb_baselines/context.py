"""The exogenous driver A(day), published to the arms that are allowed to see it.

WHY THIS FILE EXISTS
--------------------------------------------------------------------------------
Three baselines need the drift's driver, and they need DIFFERENT amounts of it:

  * **LCPO** (B6) is defined for non-stationarity induced by an *observed*
    context. Its requirement is literally "publish A(t) to the agent", and its
    out-of-distribution test is computed on the context coordinates only. It gets
    :meth:`DriverContext.observed`.
  * **oracle-driver IPPO** (the information arm) gets the same thing, so that
    "IPPO told the weather" and "LCPO told the weather" differ only by LCPO.
  * **RMA / UP-OSI** (B8) is a *system-identification* method: its teacher is
    conditioned on a quantity the agent provably cannot observe, and its student
    has to recover that quantity from its own history. It gets
    :meth:`DriverContext.privileged`, which is a strictly larger object.

Keeping the two in one place, with one day index and one gate, is what stops an
arm from quietly receiving more than its class is entitled to.

WHAT IS OBSERVED AND WHAT IS PRIVILEGED
--------------------------------------------------------------------------------
``observed(day)``    ``[A(day)]``           rainfall intensity in [0, 1].
                     (``features="a_phase"`` adds ``sin, cos`` of the cycle
                     phase -- a DECLARED ablation, because it additionally hands
                     the agent the cycle length, which A alone does not.)

``privileged(day)``  ``[A, 1 - mean_e g_e, 1 - min_e g_e]``
                     the realized capacity derating of the network under the
                     committed severity. Privileged for three separate reasons:
                     sigma is not observable to a traveller, the per-link rain
                     sensitivity is a property of the road network rather than of
                     the trip, and ``min_e`` picks out the worst-affected
                     facility. All three are functions of the day alone, so they
                     are known BEFORE anybody acts -- which is what makes them a
                     legitimate RMA "environment vector" rather than hindsight.

At ``sigma = 0`` the dial is provably inert, ``g == 1`` exactly, and the
privileged vector collapses to ``[A, 0, 0]``. RMA's teacher then knows nothing
the oracle-driver arm does not, and RMA degenerates to that arm. That is the
correct behaviour under a dial that is off, and it is printed rather than hidden.

FAILING LOUDLY
--------------------------------------------------------------------------------
A context-driven method with no context is silently inert: it still trains, still
produces a curve, and still looks like a clean null result. So a script that
needs the driver aborts when it cannot find one, unless it is run with
``--allow-no-driver`` (which prints, at every banner, that the arm is degenerate).
"""

import numpy as np

__all__ = ["DriverContext", "NoDriverError"]


class NoDriverError(RuntimeError):
    """Raised when a context-driven arm is launched without the severity wrapper."""


def _unwrap_driver(env):
    """-> (WeatherDriver, sigma, source) or (None, 0.0, reason).

    ``scripts/ns_launch.py`` replaces ``routerl.TrafficEnvironment`` with the
    severity subclass before the target script imports it, so the driver's own
    parameters are on the instance as ``_ns_cfg``. They are there at ``sigma = 0``
    too -- ``make_severity_env`` builds the subclass either way and only
    short-circuits the harm -- which is what lets A(day) be published in the
    no-severity row as well. A(t) is the weather; sigma is how much it matters.
    """
    cfg = getattr(env, "_ns_cfg", None)
    if not isinstance(cfg, dict):
        return None, 0.0, ("the environment is not the NS severity wrapper "
                           "(run through scripts/ns_launch.py)")
    try:
        from urb_ns.driver import HCM_HEAVY_RAIN_LOSS, WeatherDriver
    except ImportError as exc:                              # noqa: BLE001
        return None, 0.0, f"urb_ns.driver is not importable: {exc}"
    drv = WeatherDriver(
        int(cfg.get("period", 100)),
        float(cfg.get("wet_frac", 0.5)),
        float(cfg.get("loss", HCM_HEAVY_RAIN_LOSS)),
        bool(cfg.get("mean_preserving", False)),
    )
    return drv, float(getattr(env, "_ns_sigma", 0.0)), "env._ns_cfg"


class DriverContext(object):
    """A(day) and the privileged capacity state, with the day index gated.

    Args:
        env:       the (possibly severity-wrapped) ``TrafficEnvironment``.
        features:  ``"a"`` (default, = B6's "publish A(t)") or ``"a_phase"``.
        require:   abort if no driver is found. Scripts pass ``not args.allow_no_driver``.
        verbose:   print the banner.
    """

    FEATURE_SETS = {
        "a": ("A",),
        "a_phase": ("A", "sin_phase", "cos_phase"),
    }

    def __init__(self, env, features="a", require=True, verbose=True):
        features = str(features).lower()
        if features not in self.FEATURE_SETS:
            raise ValueError(
                f"[URB-BL] context features must be one of "
                f"{sorted(self.FEATURE_SETS)}, got {features!r}")
        self.env = env
        self.features = features
        self.names = self.FEATURE_SETS[features]
        self.driver, self.sigma, self.source = _unwrap_driver(env)
        self.degenerate = self.driver is None
        if self.degenerate and require:
            raise NoDriverError(
                "[URB-BL][GATE FAIL] this arm is defined for an EXOGENOUS OBSERVED\n"
                f"        context and there is none: {self.source}.\n"
                "        Launch it through the NS launcher, e.g.\n"
                "            python scripts/ns_launch.py --sigma 3 --net <net> -- \\\n"
                "                scripts/lcpo.py --id ... --alg-conf config1 ...\n"
                "        sigma=0 is fine and still publishes A(t) (the weather\n"
                "        happens, it just does not bite). If you really mean to run\n"
                "        without any driver, pass --allow-no-driver and expect the\n"
                "        context to be a constant zero -- i.e. the method reduced to\n"
                "        its own no-context fallback.")
        self._n_day_checked = 0
        self._n_day_mismatch = 0
        self._first_mismatch = None
        # The dial keeps its own day counter. It SHOULD equal the host's AV-day
        # index -- the wrapper only advances it once a day is harmed, and the
        # human-learning phase never calls ``last()`` -- but that is a property
        # of RouteRL's internals rather than a guarantee, so the offset is
        # inferred once from the live wrapper instead of assumed, and every
        # published context is evaluated at the day the DIAL is on.
        self._day_offset = None
        if verbose:
            self.banner()

    # ------------------------------------------------------------------ shape
    @property
    def dim(self):
        return len(self.names)

    @property
    def privileged_dim(self):
        return 3

    # ------------------------------------------------------------------ values
    def resolve_day(self, host_day):
        """The day index the DIAL is on, given the host's AV-day counter.

        Normally the identity. If the severity wrapper's counter is offset from
        the host's, the offset is applied rather than ignored: publishing
        ``A(host_day)`` while the dial derates capacity by ``g(ns_day)`` would
        describe a different day's weather from the one the fleet drove, and
        nothing else in the run would look wrong.
        """
        return int(host_day) + int(self._day_offset or 0)

    def A(self, day):
        if self.degenerate:
            return 0.0
        return float(self.driver.A(int(day)))

    def observed(self, host_day):
        """The context vector the agent is allowed to see. Shape ``(dim,)``.

        Takes the HOST's AV-day counter and resolves it to the dial's day, so a
        caller can never get the alignment wrong by forgetting to.
        """
        day = self.resolve_day(host_day)
        a = self.A(day)
        if self.features == "a":
            return np.array([a], dtype=np.float64)
        if self.degenerate:
            ph = 0.0
        else:
            ph = (int(day) % self.driver.period) / float(self.driver.period)
        return np.array([a, np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)],
                        dtype=np.float64)

    def privileged(self, host_day):
        """The RMA teacher's environment vector. Shape ``(3,)``.

        Read from the live severity layer when there is one, so that the per-link
        rain sensitivity and the committed sigma are the REAL ones rather than a
        reconstruction that could drift from them. Takes the HOST's day, as
        :meth:`observed` does.
        """
        a = self.A(self.resolve_day(host_day))
        layer = getattr(self.env, "_ns_layer", None)
        g = getattr(layer, "_g", None) if layer is not None else None
        if g is None:
            # sigma == 0, or the layer is not built yet: the dial is exactly off.
            return np.array([a, 0.0, 0.0], dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        return np.array([a, 1.0 - float(np.mean(g)), 1.0 - float(np.min(g))],
                        dtype=np.float64)

    # ------------------------------------------------------------------ gate
    def check_day(self, day):
        """Keep the published context on the same day as the dial.

        Called once per day, before anyone acts. On the FIRST call the offset
        between the wrapper's counter and the host's is recorded; afterwards any
        CHANGE in that offset is a real desynchronisation and is counted and
        reported. A constant offset is absorbed (see :meth:`resolve_day`) rather
        than treated as an error, because it is a property of RouteRL's
        bookkeeping rather than of this arm -- and aborting a run two hundred
        human-learning days in, over a constant that can simply be applied,
        would be the wrong trade.

        At ``sigma = 0`` the wrapper returns from ``last()`` before it
        increments, so ``_ns_day`` stays 0 forever and there is nothing to
        compare; the check is skipped and the report says so.

        Returns True unless the offset CHANGED mid-run, which nothing absorbs.
        """
        if self.degenerate or self.sigma == 0:
            return True
        ns_day = getattr(self.env, "_ns_day", None)
        if ns_day is None:
            return True
        self._n_day_checked += 1
        off = int(ns_day) - int(day)
        if self._day_offset is None:
            self._day_offset = off
            if off != 0:
                print("[URB-BL] the severity wrapper's day counter is offset by "
                      "%d from the host's AV-day counter (its day %d == AV day "
                      "%d). Every published context is evaluated at the "
                      "wrapper's day from here on." % (off, ns_day, day),
                      flush=True)
            return True
        if off != self._day_offset:
            self._n_day_mismatch += 1
            if self._first_mismatch is None:
                self._first_mismatch = (int(day), int(ns_day), self._day_offset)
            return False
        return True


    # ------------------------------------------------------------------ report
    def banner(self):
        print("\n" + "=" * 74)
        print("[URB-BL] EXOGENOUS DRIVER CONTEXT")
        if self.degenerate:
            print(f"  *** NO DRIVER: {self.source}")
            print("  *** A(day) will be a constant 0.0 and every context-driven")
            print("  *** mechanism in this arm is INERT. Do not report this run as")
            print("  *** a context-driven baseline.")
        else:
            d = self.driver
            print(f"  source         {self.source}   sigma={self.sigma}")
            print(f"  driver         rain, period {d.period} d, wet "
                  f"{d.wet_frac:.0%} of the cycle, loss@sigma=1 {d.loss:.3f}")
            print(f"  observed       {list(self.names)}  (dim {self.dim})")
            print("  privileged     ['A', '1-mean_e g_e', '1-min_e g_e']  (dim 3)"
                  + ("   [all zero beyond A at sigma=0]" if self.sigma == 0 else ""))
            span = np.array([self.A(t) for t in range(d.period)])
            print(f"  A over a cycle min {span.min():.3f}  max {span.max():.3f}  "
                  f"exactly-dry days {int((span == 0).sum())}/{d.period}")
        print("=" * 74 + "\n", flush=True)

    def report(self):
        print("\n" + "=" * 74)
        print("[URB-BL] DRIVER CONTEXT REPORT")
        print(f"  day-index checks     {self._n_day_checked}")
        print(f"  dial-vs-host offset  {self._day_offset}"
              + ("   (identical counters, as expected)"
                 if self._day_offset == 0 else
                 "   (absorbed: contexts were evaluated at the dial's day)"
                 if self._day_offset else ""))
        print(f"  offset CHANGES       {self._n_day_mismatch}")
        if self._first_mismatch is not None:
            h, n, off = self._first_mismatch
            print(f"  *** the offset changed at host day {h}: env._ns_day was "
                  f"{n}, expected {h + off}.")
            print("  *** The dial and this arm's context have desynchronised, so")
            print("  *** every context published after that point describes a")
            print("  *** different day's weather from the one the fleet drove.")
            print("  *** Do NOT report this run.")
        elif self._n_day_checked == 0:
            print("  (not checkable: sigma=0 or no wrapper -- the dial never")
            print("   advances its own counter, so there is nothing to compare)")
        print("=" * 74 + "\n", flush=True)

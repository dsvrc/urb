"""Domain randomisation over the severity parameter, attached to the env instance.

``paper/BASELINES.md`` B9, the "must, zero cost" robust-RL arm:

> resample sigma in [0, 3] per episode, train the stock learner, evaluate at the
> committed sigma.

WHY THIS LIVES HERE AND NOT IN ``urb_ns/severity_traffic_env.py``
--------------------------------------------------------------------------------
That file is the certified dial every arm -- PACT-1 included -- runs under, and
``results/ns_certificate`` is computed against it. Adding a randomisation branch
to it would mean every PACT-1 run from now on executes code written for a
baseline, and a reviewer would be right to ask what else changed. So the dial is
left byte-for-byte alone and the randomisation is attached to ONE env INSTANCE,
by the one script that wants it, at run time.

HOW IT WORKS
--------------------------------------------------------------------------------
``SeverityTrafficEnvironment.reset()`` resolves the day's weather by calling
``SeverityLayer.begin_episode``, which reads ``layer.sigma``. So the only thing
domain randomisation has to do is set ``sigma`` BEFORE that call. The instance's
``reset`` is replaced by a wrapper that draws, assigns and then delegates.

THE ZERO-SEVERITY FLOOR, AND WHY IT IS NOT A FUDGE
--------------------------------------------------------------------------------
The wrapper short-circuits at ``sigma == 0``: ``last()`` returns before it
finalises the day, so ``_ns_day`` never advances and ``_ns_final`` is never
reset. That is correct for a run whose sigma is 0 throughout, and wrong for a run
that visits 0 for a single day. Draws are therefore floored at ``sigma_floor``
(default 1e-9), where the capacity derate is ``1 - 1e-9 * loss * A``, i.e. 1.0 to
fifteen significant figures -- numerically the same task, on the bookkeeping path
that a mid-run zero would fall off.

``fix(sigma)`` (used before the test phase) does NOT use the floor: it detaches
the wrapper entirely and restores the stock behaviour, so evaluating at
``sigma = 0`` is byte-identical to a stock run, as B.1.1 requires.
"""

import numpy as np

__all__ = ["DomainRandomiser", "attach"]


class DomainRandomiser(object):
    """Per-episode resampling of the severity parameter on one env instance."""

    def __init__(self, env, lo, hi, seed, sigma_floor=1e-9, verbose=True):
        if not hasattr(env, "_ns_sigma"):
            raise RuntimeError(
                "[URB-BL][GATE FAIL] domain randomisation needs the NS severity "
                "wrapper, and this environment is the stock TrafficEnvironment.\n"
                "        Launch through scripts/ns_launch.py, e.g.\n"
                "            python scripts/ns_launch.py --sigma 3 --net <net> -- \\\n"
                "                scripts/dr_ippo.py --id ... --alg-conf config1 ...\n"
                "        (--sigma there is the COMMITTED severity used for the test\n"
                "        phase; training resamples over --dr-range.)")
        if not (hi >= lo >= 0.0):
            raise ValueError(f"[URB-BL] dr range must satisfy 0 <= lo <= hi, "
                             f"got [{lo}, {hi}]")
        self.env = env
        self.lo = float(lo)
        self.hi = float(hi)
        self.floor = float(sigma_floor)
        self.rng = np.random.RandomState(int(seed))
        self.active = True
        self.draws = []
        self.committed = float(getattr(env, "_ns_sigma", 0.0))
        self._orig_reset = env.reset
        env.reset = self._reset
        if verbose:
            self.banner()

    # ------------------------------------------------------------------ hooks
    def _apply(self, sigma):
        self.env._ns_sigma = float(sigma)
        layer = getattr(self.env, "_ns_layer", None)
        if layer is not None:
            layer.sigma = float(sigma)

    def _reset(self, *a, **kw):
        if self.active:
            s = max(float(self.rng.uniform(self.lo, self.hi)), self.floor)
            self.draws.append(s)
            self._apply(s)
        return self._orig_reset(*a, **kw)

    # ------------------------------------------------------------------ api
    def fix(self, sigma, verbose=True):
        """Stop randomising and pin sigma -- the evaluation half of B9.

        Detaches the wrapper completely, so ``sigma = 0`` recovers the stock task
        byte for byte rather than the 1e-9 floor used during training.
        """
        self.active = False
        if self.env.reset is self._reset:
            self.env.reset = self._orig_reset
        self._apply(float(sigma))
        if verbose:
            print(f"\n[URB-BL] domain randomisation OFF; sigma pinned at "
                  f"{float(sigma)} for evaluation.\n", flush=True)

    # ------------------------------------------------------------------ report
    def banner(self):
        print("\n" + "=" * 74)
        print("[URB-BL] DOMAIN RANDOMISATION OVER SEVERITY (B9)")
        print("=" * 74)
        print(f"  training sigma   ~ U[{self.lo}, {self.hi}], redrawn EVERY day")
        print(f"  zero floor       {self.floor:g}  (derate 1 - {self.floor:g}*loss*A"
              " = 1.0 to 15 s.f.)")
        print(f"  evaluation sigma {self.committed}  (the committed severity this")
        print("                   run was launched with; pinned before the test")
        print("                   phase, with the randomiser fully detached)")
        print("  The learner is the stock host learner and is not told sigma.")
        print("=" * 74 + "\n", flush=True)

    def report(self):
        d = np.asarray(self.draws, dtype=np.float64)
        print("\n" + "=" * 74)
        print("[URB-BL] DOMAIN RANDOMISATION REPORT")
        print(f"  days randomised  {len(d)}")
        if len(d):
            print(f"  sigma drawn      min {d.min():.4f}  mean {d.mean():.4f}  "
                  f"max {d.max():.4f}")
            print(f"  below 0.05       {int((d < 0.05).sum())} days "
                  "(the near-placebo tail)")
        else:
            print("  *** NO DAY WAS RANDOMISED. This is not a domain-randomisation")
            print("  *** arm -- do not report it as one.")
        print(f"  evaluated at     sigma={float(getattr(self.env, '_ns_sigma', 0)):g}")
        print("=" * 74 + "\n", flush=True)


def attach(env, lo, hi, seed, sigma_floor=1e-9, verbose=True):
    return DomainRandomiser(env, lo, hi, seed, sigma_floor, verbose)

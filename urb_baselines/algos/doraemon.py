"""DORAEMON -- domain randomisation via entropy maximisation.

    Tiboni, Klink, Peters, Tommasi, D'Eramo & Chalvatzaki,
    "Domain Randomization via Entropy Maximization", ICLR 2024
    (arXiv:2311.01885), code https://github.com/gabrieletiboni/doraemon.

WHAT IT IS AND WHY IT IS A SEPARATE ARM FROM ``dr_ippo``
--------------------------------------------------------------------------------
``dr_ippo`` is B9's "must": severity drawn ``U[0, 3]`` every day, the stock
learner, evaluate at the committed sigma. Its weakness is exactly the one this
paper was written about -- the randomisation range is fixed in advance, so it is
either too narrow to generalise or so wide that the policy is conservative
everywhere. DORAEMON removes the choice: it makes the training distribution as
WIDE as it can be while the policy still succeeds, by solving

    max_phi  H(nu_phi)   s.t.   G(theta, phi) >= alpha,   KL(nu_phi || nu_phi_i) <= eps

every K days, where ``nu_phi`` is a Beta distribution over the severity and
``G`` is the probability that a day under it is a success.

So the two arms answer different objections. ``dr_ippo`` answers "did you try
domain randomisation"; this one answers "did you try domain randomisation *with
a curriculum that adapts to what the policy can actually do*", which is the form
a reviewer who works on sim-to-real will have in mind. Neither is a flag on the
other: they have separate scripts, separate configs and separate logs, like
every other arm here.

THE FOUR THINGS URB FORCES, EACH A ROW IN THE CHECKLIST
--------------------------------------------------------------------------------
1. **One randomisation dimension.** The paper randomises a vector of dynamics
   parameters with one Beta per coordinate. URB-NS has exactly one certified
   dial, ``sigma``, so the distribution is a single rescaled Beta on
   ``[lo, hi]``. Everything else in the paper -- the entropy, the KL, the
   importance weights -- is per-coordinate and sums, so the one-dimensional case
   is the paper's own formulae with one term.

2. **The success indicator is relative excess delay.** The paper is explicit
   that ``sigma(tau)`` is "defined through domain knowledge" and allows "a lower
   bound on the expected return". URB's return is ``-travel_time`` in seconds,
   which is not comparable across travellers or across networks, so success here
   is ``mean_i (tt_i / fft_i - 1) <= tau_succ``: relative excess delay, which is
   BPR's natural form and the same target PACT-1's regressor uses. With
   ``perf_lb: null`` the threshold is CALIBRATED from the arm's own first
   ``warmup_days`` days and then frozen, so no magic constant is smuggled in and
   nothing is tuned on returns.

3. **The solver is a deterministic grid, not scipy.** The released code calls
   ``scipy.optimize.minimize`` with two ``NonlinearConstraint``s. ``scipy`` is
   not in URB's ``requirements.txt`` and a baseline must not add a dependency to
   the host, so the same constrained problem is solved by a coarse-to-fine grid
   over ``(log a, log b)``. With two parameters that is exact to grid
   resolution, and it is deterministic, which a local optimiser from a moving
   start point is not. The OBJECTIVE and the CONSTRAINTS are the paper's.

4. **An "episode" is a day.** The paper's K trajectories per update are K days
   here, and the policy update in between is the host's own.

WHAT IT TESTS
--------------------------------------------------------------------------------
B9's prediction sharpened: if hedging beat identifying, the best hedge would be
the widest one the policy can carry -- so DORAEMON is the strongest form of the
objection. If it still cannot track ``A(t)`` within a run, that is because no
severity DISTRIBUTION is a substitute for knowing which draw you are in, which
is the distinction the paper is making.
"""

import math

import numpy as np

from urb_baselines import domain_random
from urb_baselines.algos.reference import PerAgentPPOAlgorithm
from urb_baselines.records import flatten_free_flow

__all__ = ["DORAEMON", "BetaDR", "add_args"]


# ==========================================================================
#  the randomisation distribution: one rescaled Beta, the paper's family
# ==========================================================================
class BetaDR(object):
    """``Beta(a, b)`` rescaled to ``[lo, hi]`` -- DORAEMON's ``nu_phi``.

    Only the three quantities the optimisation needs are implemented: the
    density (for importance weights), the differential entropy (the objective)
    and the KL divergence (the trust region). All three are the standard
    closed forms; the rescaling contributes ``log(hi - lo)`` to the entropy and
    cancels in both the ratio and the KL.
    """

    def __init__(self, a, b, lo, hi):
        self.a = float(a)
        self.b = float(b)
        self.lo = float(lo)
        self.hi = float(hi)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _log_beta_fn(a, b):
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def to_unit(self, x):
        return np.clip((np.asarray(x, dtype=np.float64) - self.lo)
                       / max(self.hi - self.lo, 1e-12), 1e-6, 1.0 - 1e-6)

    def sample(self, rng):
        return self.lo + (self.hi - self.lo) * float(rng.beta(self.a, self.b))

    def log_pdf_unit(self, u):
        u = np.asarray(u, dtype=np.float64)
        return ((self.a - 1.0) * np.log(u) + (self.b - 1.0) * np.log(1.0 - u)
                - self._log_beta_fn(self.a, self.b))

    def entropy(self):
        a, b = self.a, self.b
        h = (self._log_beta_fn(a, b)
             - (a - 1.0) * _digamma(a) - (b - 1.0) * _digamma(b)
             + (a + b - 2.0) * _digamma(a + b))
        return h + math.log(max(self.hi - self.lo, 1e-12))

    def kl_to(self, other):
        """``KL(self || other)`` for two Betas on the same support."""
        a1, b1, a2, b2 = self.a, self.b, other.a, other.b
        return (self._log_beta_fn(a2, b2) - self._log_beta_fn(a1, b1)
                + (a1 - a2) * _digamma(a1) + (b1 - b2) * _digamma(b1)
                + (a2 - a1 + b2 - b1) * _digamma(a1 + b1))

    def mean(self):
        return self.lo + (self.hi - self.lo) * self.a / (self.a + self.b)

    def __repr__(self):
        return (f"Beta(a={self.a:.3f}, b={self.b:.3f}) on "
                f"[{self.lo:g}, {self.hi:g}]")


def _digamma(x):
    """``psi(x)`` for ``x > 0``: recurrence up to 6, then the asymptotic series.

    Written out because ``scipy`` is not a URB dependency (see the module
    docstring, adaptation 3). Accurate to ~1e-12 over the range used here, which
    is far finer than the grid the optimiser searches.
    """
    r = 0.0
    x = float(x)
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return (r + math.log(x) - 0.5 / x
            + f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0
                   + f * (1.0 / 240.0 + f * (-1.0 / 132.0))))))


# ==========================================================================
#  the constrained optimisation, on a deterministic grid
# ==========================================================================
def _success_estimate(new, cur, u, succ):
    """``G_hat`` of Eq. (5): importance-weighted success under ``new``.

    ``u`` are the drawn severities mapped to the unit interval, ``succ`` their
    0/1 success flags, both collected under ``cur``.
    """
    if u.size == 0:
        return 0.0
    w = np.exp(np.clip(new.log_pdf_unit(u) - cur.log_pdf_unit(u), -30.0, 30.0))
    return float(np.mean(w * succ))


def _grid(cur, bounds, n=21):
    """Candidate ``(a, b)`` on a log grid, coarse then fine around the incumbent."""
    lo_p, hi_p = float(bounds[0]), float(bounds[1])
    coarse = np.exp(np.linspace(math.log(lo_p), math.log(hi_p), n))
    span = math.log(hi_p / lo_p) / (n - 1)
    fine_a = np.exp(np.linspace(math.log(cur.a) - span, math.log(cur.a) + span, 7))
    fine_b = np.exp(np.linspace(math.log(cur.b) - span, math.log(cur.b) + span, 7))
    A = np.unique(np.clip(np.concatenate([coarse, fine_a]), lo_p, hi_p))
    B = np.unique(np.clip(np.concatenate([coarse, fine_b]), lo_p, hi_p))
    return A, B


def solve_doraemon(cur, u, succ, alpha, kl_ub, bounds, grid=21):
    """Eq. (4), with Eq. (6) as the backup. Returns ``(new, info)``.

    Eq. (4)   max H(nu')   s.t.  G_hat(theta, phi_start, phi') >= alpha
                                 KL(nu' || nu_i) <= eps
    Eq. (6)   max G_hat    s.t.  KL(nu' || nu_i) <= eps      (feasibility backup)
    """
    A, B = _grid(cur, bounds, grid)
    best_h, best = -1e30, None
    best_g, best_g_cand = -1.0, None
    for a in A:
        for b in B:
            cand = BetaDR(a, b, cur.lo, cur.hi)
            if cand.kl_to(cur) > kl_ub:
                continue
            g = _success_estimate(cand, cur, u, succ)
            if g > best_g:
                best_g, best_g_cand = g, cand
            if g >= alpha:
                h = cand.entropy()
                if h > best_h:
                    best_h, best = h, cand
    if best is not None:
        return best, {"mode": "entropy", "g": _success_estimate(best, cur, u, succ),
                      "h": best_h}
    # infeasible: fall back to the backup problem, which is what the released
    # code does when importance sampling has overestimated the success rate.
    if best_g_cand is None:
        return cur, {"mode": "none", "g": 0.0, "h": cur.entropy()}
    return best_g_cand, {"mode": "backup", "g": best_g,
                         "h": best_g_cand.entropy()}


# ==========================================================================
#  the severity sampler
# ==========================================================================
class _BetaRandomiser(domain_random.DomainRandomiser):
    """``DomainRandomiser`` with the draw taken from a Beta instead of a uniform.

    Subclassed rather than parameterised so that ``urb_baselines/domain_random.py``
    -- which ``dr_ippo`` and selftest gate DR-1 both depend on -- is not touched
    at all. Everything else (wrapping ``reset``, the zero floor and its reason,
    ``fix()`` detaching for the test phase) is inherited unchanged.
    """

    def __init__(self, env, dist, seed, sigma_floor=1e-9, verbose=True):
        self.dist = dist
        super(_BetaRandomiser, self).__init__(
            env, dist.lo, dist.hi, seed, sigma_floor, verbose=False)
        if verbose:
            self.banner()

    def _reset(self, *a, **kw):
        if self.active:
            s = max(self.dist.sample(self.rng), self.floor)
            self.draws.append(s)
            self._apply(s)
        return self._orig_reset(*a, **kw)

    def banner(self):
        print("\n" + "=" * 74)
        print("[URB-BL] DOMAIN RANDOMISATION VIA ENTROPY MAXIMISATION (B9)")
        print("=" * 74)
        print(f"  support          [{self.lo}, {self.hi}]  (B9's severity range)")
        print(f"  distribution     {self.dist}  -- adapted every K days")
        print(f"  zero floor       {self.floor:g}")
        print(f"  evaluation sigma {self.committed}  (pinned before the test "
              "phase)")
        print("  The learner is the stock host learner and is not told sigma.")
        print("=" * 74 + "\n", flush=True)


# ==========================================================================
#  the algorithm
# ==========================================================================
class DORAEMON(PerAgentPPOAlgorithm):
    """URB's IPPO under an entropy-maximising severity curriculum."""

    name = "doraemon"
    ARMS = ("doraemon", "fixed")
    needs_records = True                   # success needs per-day travel times

    def __init__(self, ctx):
        cfg = dict(ctx.algo_cfg)
        self.arm = str(getattr(ctx.args, "arm", None)
                       or cfg.get("arm", "doraemon")).lower()
        if self.arm not in self.ARMS:
            raise ValueError(f"[URB-BL][doraemon] --arm must be one of {self.ARMS}")
        super(DORAEMON, self).__init__(ctx)

        rng_spec = cfg.get("dr_range", [0.0, 3.0])
        self.lo, self.hi = float(rng_spec[0]), float(rng_spec[1])
        self.alpha = float(cfg.get("alpha", 0.5))
        self.kl_ub = float(cfg.get("kl_ub", 0.05))
        self.update_every_days = int(cfg.get("update_every_days", 100))
        self.bounds = list(cfg.get("beta_param_bounds", [0.5, 200.0]))
        self.grid = int(cfg.get("grid", 21))
        self.warmup_days = int(cfg.get("warmup_days", 200))
        self.print_every = int(cfg.get("print_every", 100))
        init_p = float(cfg.get("init_beta_param", 100.0))

        perf = getattr(ctx.args, "perf_lb", None)
        if perf is None:
            perf = cfg.get("perf_lb", None)
        self.perf_lb = None if perf is None else float(perf)
        self.calibrating = self.perf_lb is None
        self.warm = []

        self.free_flow = flatten_free_flow(dict(ctx.free_flow))
        self.od_of = {a: self._od(ctx, a) for a in self.av_ids}

        self.dist = BetaDR(init_p, init_p, self.lo, self.hi)
        self.committed = float(getattr(ctx.env, "_ns_sigma", 0.0))
        self.dr = _BetaRandomiser(
            ctx.env, self.dist, seed=int(ctx.env_seed) * 7919 + 29,
            sigma_floor=float(cfg.get("sigma_floor", 1e-9)), verbose=True)

        self.buf_sigma = []
        self.buf_succ = []
        self.n_updates = 0
        self.n_backup = 0
        self.hist = [(0, self.dist.a, self.dist.b, self.dist.entropy())]
        self.last_g = float("nan")
        self.day_metric = float("nan")

        self.banner("DORAEMON (B9: robust RL, adaptive randomisation)", [
            ("paper", "arXiv:2311.01885 (ICLR 2024)"),
            ("arm", "entropy maximisation" if self.arm == "doraemon"
             else "FIXED Beta (ablation: no adaptation)"),
            ("learner", "URB's IPPO, unchanged (scripts/ippo.py)"),
            ("nu_phi", f"{self.dist}"),
            ("alpha (success rate)", f"{self.alpha}   -- the paper's own choice"),
            ("eps (KL trust region)", f"{self.kl_ub}"),
            ("update cadence", f"every {self.update_every_days} days"),
            ("success", "mean relative excess delay <= "
             + ("CALIBRATING" if self.calibrating else f"{self.perf_lb:.4f}")),
            ("evaluation sigma", f"{self.committed} (committed at launch)"),
        ])

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _od(ctx, a):
        rec = (ctx.agent_table or {}).get(str(a)) or {}
        od = rec.get("od")
        if od is None:
            return None
        return (int(od[0]), int(od[1])) if isinstance(od, (tuple, list)) else od

    def _excess(self, av_records):
        """Mean relative excess delay ``tt/fft - 1`` over the machine agents."""
        vals = []
        for a, (k, tt) in av_records.items():
            od = self.od_of.get(a)
            fft = self.free_flow.get(od)
            if fft is None or not (0 <= int(k) < fft.size):
                continue
            f = fft[int(k)]
            if not np.isfinite(f) or f <= 0:
                continue
            vals.append(float(tt) / float(f) - 1.0)
        return float(np.mean(vals)) if vals else float("nan")

    # ------------------------------------------------------------------ hooks
    def end_episode(self, day, phase, info):
        if phase != "train" or not info.av_records:
            return
        m = self._excess(info.av_records)
        self.day_metric = m
        if not np.isfinite(m):
            return
        if self.calibrating:
            self.warm.append(m)
            if len(self.warm) >= self.warmup_days:
                # The threshold is the median of what the arm itself achieved
                # while the distribution was still at its initial value. Frozen
                # from here on; nothing downstream can move it.
                self.perf_lb = float(np.median(self.warm))
                self.calibrating = False
                print(f"\n[doraemon] success threshold CALIBRATED from the first "
                      f"{len(self.warm)} days:\n"
                      f"           relative excess delay <= {self.perf_lb:.4f} "
                      f"(median of the warm-up)\n", flush=True)
            return
        s = self.dr.draws[-1] if self.dr.draws else float("nan")
        if np.isfinite(s):
            self.buf_sigma.append(float(s))
            self.buf_succ.append(1.0 if m <= self.perf_lb else 0.0)
        if (self.arm == "doraemon"
                and len(self.buf_sigma) >= self.update_every_days):
            self._update_distribution(day)

    def _update_distribution(self, day):
        u = self.dist.to_unit(np.asarray(self.buf_sigma))
        succ = np.asarray(self.buf_succ, dtype=np.float64)
        cur = self.dist

        # Algorithm 1: if the CURRENT distribution already fails the constraint,
        # first solve the backup problem to recover feasibility.
        g_now = _success_estimate(cur, cur, u, succ)
        start = cur
        if g_now < self.alpha:
            back, _ = solve_doraemon(cur, u, succ, alpha=1e9,     # pure argmax G
                                     kl_ub=self.kl_ub, bounds=self.bounds,
                                     grid=self.grid)
            self.n_backup += 1
            if _success_estimate(back, cur, u, succ) < self.alpha:
                new, info = back, {"mode": "backup-only",
                                   "g": _success_estimate(back, cur, u, succ),
                                   "h": back.entropy()}
                self._commit(new, info, day, g_now)
                return
            start = back
        new, info = solve_doraemon(start, u, succ, self.alpha, self.kl_ub,
                                   self.bounds, self.grid)
        self._commit(new, info, day, g_now)

    def _commit(self, new, info, day, g_now):
        self.dist.a, self.dist.b = float(new.a), float(new.b)
        self.last_g = float(info.get("g", float("nan")))
        self.n_updates += 1
        self.hist.append((int(day), self.dist.a, self.dist.b,
                          self.dist.entropy()))
        self.buf_sigma, self.buf_succ = [], []

    def begin_test(self):
        # B9's second half, identical to dr_ippo: evaluate at the committed
        # severity with the randomiser fully detached.
        self.dr.fix(self.committed)
        super(DORAEMON, self).begin_test()

    # ------------------------------------------------------------------ report
    def diagnostics(self):
        return {
            "beta_a": self.dist.a,
            "beta_b": self.dist.b,
            "H": self.dist.entropy(),
            "sig_mu": self.dist.mean(),
            "G": self.last_g,
            "upd": self.n_updates,
            "excess": self.day_metric,
        }

    def close(self):
        self.dr.report()
        print("=" * 74)
        print("[doraemon] ENTROPY-MAXIMISATION REPORT")
        print(f"  distribution updates     {self.n_updates}")
        print(f"  backup problems solved   {self.n_backup}  (feasibility "
              "recovery, Eq. 6)")
        print(f"  success threshold        "
              + ("NEVER CALIBRATED" if self.perf_lb is None
                 else f"excess delay <= {self.perf_lb:.4f}"))
        print(f"  final distribution       {self.dist}")
        h0, h1 = self.hist[0][3], self.hist[-1][3]
        print(f"  entropy  first -> last   {h0:.4f} -> {h1:.4f}  "
              f"(d = {h1 - h0:+.4f})")
        print(f"  mean sigma  first->last  "
              f"{BetaDR(self.hist[0][1], self.hist[0][2], self.lo, self.hi).mean():.3f}"
              f" -> {self.dist.mean():.3f}")
        if self.hist:
            print("  trajectory (day, a, b, H):")
            step = max(1, len(self.hist) // 8)
            for d, a, b, h in self.hist[::step]:
                print(f"      {d:>6}  a={a:8.3f}  b={b:8.3f}  H={h:+.4f}")
        if self.arm == "doraemon" and self.n_updates == 0:
            print("  *** THE DISTRIBUTION WAS NEVER UPDATED, so this arm is a")
            print("  *** fixed-Beta domain randomisation and not DORAEMON. Either")
            print("  *** the warm-up never finished (check the calibration line)")
            print("  *** or no day ever produced a usable travel-time record.")
        elif self.arm == "doraemon" and abs(h1 - h0) < 1e-6:
            print("  *** THE ENTROPY NEVER MOVED. Every solve returned the")
            print("  *** incumbent, which means either the KL trust region is so")
            print("  *** small that no candidate differs, or the success")
            print("  *** constraint is satisfied/violated everywhere on the grid.")
            print("  *** Look at G above: alpha =", self.alpha)
        print("=" * 74 + "\n", flush=True)


def add_args(parser):
    parser.add_argument('--arm', type=str, default=None, choices=list(DORAEMON.ARMS),
                        help="doraemon: adapt the Beta by entropy maximisation "
                             "(default). fixed: keep the initial Beta -- the "
                             "ablation that isolates what the adaptation buys.")
    parser.add_argument('--perf-lb', type=float, default=None,
                        help="success threshold on mean relative excess delay. "
                             "Omitted: calibrated from the first warmup_days "
                             "days of this run and then frozen.")

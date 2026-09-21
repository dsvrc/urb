"""QCD-restart bandits -- B7's missing URB representative.

    Gerogiannis, Huang & Veeravalli, "Is Prior-Free Black-Box Non-Stationary
    Reinforcement Learning Feasible?", arXiv:2410.13772.

WHY THIS PAPER, ON THIS HOST
--------------------------------------------------------------------------------
``paper/BASELINES.md`` B7 is "single-agent non-stationary RL, latent /
change-point -- drift not observed". Its chosen representative (LILAC) is
SAC-based and was assigned to MAPDN, so on URB the class had no arm at all. This
one fills it, and it fills it with the paper whose own experiment IS this
environment: their entire empirical section is a **piecewise-stationary
multi-armed bandit with 5 arms**, and URB on these cities is a repeated K-armed
bandit for every learner (``docs/baselines/README.md`` section 4: the observation
is a constant agent identifier for ~97% of travellers, so there is no context to
condition on and a traveller's problem is "which of my K routes, today").

WHAT IT TESTS
--------------------------------------------------------------------------------
The sharpest objection to an identification method: *if the drift is not
observed, run a change detector and restart -- why estimate a coupling at all?*
This arm is that objection, implemented as the paper implements it.

The theory predicts the comparison is not a walkover in either direction. A
restart re-injects exploration, i.e. **excitation**, which C4 names as the
binding constraint on identification in a converging population; so a restarting
arm should track drift that a converged estimator cannot, while paying for every
restart with a fresh cold start. That is a real trade-off and the paper should
report it as one.

THE THREE ARMS, AND WHY ALL THREE ARE HERE
--------------------------------------------------------------------------------
``--arm glr``     (default) the paper's recommendation: a Bernoulli-GLR change
                  detector per (agent, route) with forced exploration, and a full
                  restart of that agent's learner when the detector fires. This
                  is their GLRklUCB / QCD+ family, on URB's own DQN instead of
                  klUCB (rule R2: the host's learner, the paper's mechanism).

``--arm rr``      their order-optimal control: restart at i.i.d. Geo(eta_R)
                  intervals, with prior knowledge of the change rate. Theorem 11.
                  It exists so that "detection helped" cannot be confused with
                  "restarting helped".

``--arm master``  MASTER's two non-stationarity tests and its block schedule,
                  reproduced so their central negative result can be checked on
                  URB. Theorem 4: the thresholds "can be crossed only if
                  delta >= T exp(-sqrt(T)/54(log2 T + 1))", which for delta < 1
                  needs T >= 1.24e9. URB runs 4000 days. The prediction is that
                  the test NEVER fires and MASTER degenerates into its own random
                  restart schedule -- and a zero here is the result, not a bug,
                  so this arm reports its trigger count without the ``***``
                  degeneracy marker that the other two use.

THE REWARD SCALE, AGAIN
--------------------------------------------------------------------------------
The fourth instance of the failure mode ``docs/baselines/README.md`` section 4
documents. A Bernoulli-GLR statistic is defined for observations in [0, 1]; URB
pays ``-travel_time``, of order -1000. Feeding it raw gives ``kl(p, q)`` on
values outside its domain -- in practice every clip lands on the same boundary,
the statistic is identically 0 and the detector can never fire, which looks
exactly like "there was no change point".

The fix is the repo's standard device (LCPO's ``ret_rms``, DGN's target
standardisation, LIAM's ``standardise_stream``): map each agent's reward stream
through its own running standardisation and a logistic squash,

    u_t = sigmoid( (r_t - mu_ref) / (sd_ref + 1e-8) )    in (0, 1)

which is monotone, so a change in the reward distribution is a change in the
u-distribution, and the detector sees it. The reference ``(mu_ref, sd_ref)`` is
measured over the first ``calib`` days and then FROZEN until the next restart --
a running reference absorbs the shift it is meant to detect, which gate QCDR-2
measures.

``normalize_stream: false`` reproduces the degenerate behaviour, and
``u_spread`` is printed every ``print_every`` days so a collapsed stream is
visible while the run is still going.
"""

import math

import numpy as np

from urb_baselines.algos.reference import PerAgentDQNAlgorithm

__all__ = ["QCDRestart", "add_args", "bernoulli_glr", "binary_kl"]


# ==========================================================================
#  the detector
# ==========================================================================
def binary_kl(p, q, eps=1e-9):
    """KL(Bern(p) || Bern(q)), vectorised and clipped away from the poles."""
    p = np.clip(p, eps, 1.0 - eps)
    q = np.clip(q, eps, 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def bernoulli_glr(y, stride=1):
    """The Bernoulli-GLR statistic of Besson et al. (2022), used by the paper.

    For observations ``y_1..y_n`` the generalised likelihood ratio for "a change
    happened after s" against "no change" is, with ``mu_{a:b}`` the sample mean,

        G_n = max_{1 <= s < n}  s * kl(mu_{1:s}, mu_{1:n})
                              + (n - s) * kl(mu_{s+1:n}, mu_{1:n})

    ``stride`` subsamples the split points. The full scan is O(n) here because
    the means come from one cumulative sum, so the stride exists only for very
    long histories; it never changes which split points are *available*, it
    changes how finely they are searched.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 2:
        return 0.0
    cs = np.cumsum(y)
    total = cs[-1]
    mu_all = total / n
    s = np.arange(1, n, max(1, int(stride)), dtype=np.float64)
    idx = s.astype(np.int64) - 1
    mu_left = cs[idx] / s
    mu_right = (total - cs[idx]) / (n - s)
    stat = s * binary_kl(mu_left, mu_all) + (n - s) * binary_kl(mu_right, mu_all)
    return float(np.max(stat)) if stat.size else 0.0


def glr_threshold(n, delta):
    """``beta(n, delta) = 2 log(3 n^{3/2} / delta)`` -- Besson et al. (2022)."""
    n = max(int(n), 1)
    return 2.0 * math.log(3.0 * (n ** 1.5) / max(float(delta), 1e-12))


class _Stream(object):
    """One agent's reward stream, mapped into (0, 1) for the Bernoulli GLR.

    THE REFERENCE IS CALIBRATED ONCE AND THEN FROZEN, and re-calibrated after a
    restart. A *running* standardisation would be worse than useless here: it
    absorbs the very shift the detector is looking for, because the mean it
    subtracts is dragged toward the new level by the new data. Measured in
    selftest gate QCDR-2 -- a 300-second level shift gives a GLR of 15.5 under a
    running reference (below the threshold, never fires) against 60+ under a
    frozen one.

    Re-calibrating on ``flush()`` is the right pairing: a restart declares the
    old regime over, so the reference for "normal" is re-measured in the new
    one. Between restarts the reference does not move, which is what makes a
    level shift visible.
    """

    def __init__(self, n_actions, normalize, max_hist, calib=30):
        self.n_actions = int(n_actions)
        self.normalize = bool(normalize)
        self.max_hist = int(max_hist)
        self.calib = int(calib)
        self.hist = [[] for _ in range(self.n_actions)]
        self._reset_reference()

    def _reset_reference(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.ref = None                      # (mean, sd), frozen after calib

    def _sd(self):
        return math.sqrt(self.m2 / self.n) if self.n > 1 else 0.0

    def push(self, action, reward):
        r = float(reward)
        if self.ref is None:
            # Welford over the calibration window only.
            self.n += 1
            d = r - self.mean
            self.mean += d / self.n
            self.m2 += d * (r - self.mean)
            if self.n >= self.calib:
                self.ref = (self.mean, max(self._sd(), 1e-8))
        if self.normalize:
            mu, sd = self.ref if self.ref is not None else (self.mean,
                                                            self._sd() + 1e-8)
            z = (r - mu) / (sd + 1e-8)
            u = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        else:
            u = max(0.0, min(1.0, r))          # the degenerate path, on purpose
        h = self.hist[int(action)]
        h.append(u)
        if len(h) > self.max_hist:
            del h[0]
        return u

    def flush(self):
        self.hist = [[] for _ in range(self.n_actions)]
        self._reset_reference()

    def spread(self):
        """Std of the mapped stream, pooled over routes. ~0 means the detector
        is looking at a constant and cannot fire for a SCALE reason."""
        vals = [v for h in self.hist for v in h]
        return float(np.std(vals)) if len(vals) > 1 else 0.0


# ==========================================================================
#  MASTER's two tests
# ==========================================================================
class _Master(object):
    """MASTER's block schedule and its two non-stationarity tests.

    Wei & Luo's MASTER as the paper states it (their Section 2), with the
    bandit instantiation ``rho(n) = sqrt(K / n)``:

        Test 1   (1/2^m) sum R_tau  -  min g~_tau  >=  54 (log2 T + 1) log(T/delta) rho(2^m)
        Test 2   (1/(t - t_n + 1)) sum (g~_tau - R_tau) >= 18 (log2 T + 1) log(T/delta) rho(t - t_n + 1)

    ``g~`` is the base learner's optimistic value estimate; on URB's DQN that is
    ``max_k Q(s, k)`` for the day's state, which is the same object klUCB's index
    plays in their bandit instantiation.

    Both thresholds carry the constants verbatim. That is the point of the arm:
    the paper's Theorem 4 is a statement ABOUT those constants, and softening
    them to make the test fire would answer a different question.
    """

    def __init__(self, horizon, n_actions, delta, rng):
        self.T = max(int(horizon), 2)
        self.K = int(n_actions)
        self.delta = float(delta)
        self.rng = rng
        self.log_factor = (math.log2(self.T) + 1.0) * math.log(
            self.T / max(self.delta, 1e-12))
        self.t = 0
        self.t_n = 0                      # start of the current instance
        self.block_R = []                 # rewards since the block began
        self.block_g = []
        self.inst_R = []                  # rewards since the instance began
        self.inst_g = []
        self.order = 0                    # current block order m
        self.trig1 = 0
        self.trig2 = 0
        self.sched_restarts = 0
        self.closest1 = 0.0               # max (statistic / threshold) seen
        self.closest2 = 0.0

    @staticmethod
    def _rho(n, K):
        return math.sqrt(K / max(int(n), 1))

    def step(self, reward, g_tilde):
        """One day. Returns True when the learner must be restarted."""
        self.t += 1
        self.block_R.append(float(reward))
        self.block_g.append(float(g_tilde))
        self.inst_R.append(float(reward))
        self.inst_g.append(float(g_tilde))

        restart = False
        # ---- Test 1, evaluated at the end of each 2^m block ----
        blk = 1 << self.order
        if len(self.block_R) >= blk:
            lhs = float(np.mean(self.block_R)) - float(np.min(self.block_g))
            rhs = 54.0 * self.log_factor * self._rho(blk, self.K)
            self.closest1 = max(self.closest1, lhs / rhs if rhs > 0 else 0.0)
            if lhs >= rhs:
                self.trig1 += 1
                restart = True
            self.block_R, self.block_g = [], []
            # MASTER's multi-scale schedule: order m is (re)started with
            # probability 2^-m, which is what the paper calls its random
            # restarting behaviour once the tests never fire.
            self.order += 1
            if self.rng.rand() < 2.0 ** (-self.order):
                self.sched_restarts += 1
                self.order = 0
                restart = True

        # ---- Test 2, evaluated every day of the instance ----
        n = len(self.inst_R)
        if n >= 2:
            lhs = float(np.mean(np.asarray(self.inst_g) - np.asarray(self.inst_R)))
            rhs = 18.0 * self.log_factor * self._rho(n, self.K)
            self.closest2 = max(self.closest2, lhs / rhs if rhs > 0 else 0.0)
            if lhs >= rhs:
                self.trig2 += 1
                restart = True

        if restart:
            self.t_n = self.t
            self.inst_R, self.inst_g = [], []
            self.block_R, self.block_g = [], []
        return restart


# ==========================================================================
#  the algorithm
# ==========================================================================
class QCDRestart(PerAgentDQNAlgorithm):
    """Change-detection / restart bandits on URB's own DQN."""

    name = "qcdr"
    ARMS = ("glr", "rr", "master")

    def __init__(self, ctx):
        super(QCDRestart, self).__init__(ctx)
        cfg = self.cfg
        arm = str(getattr(ctx.args, "arm", None) or cfg.get("arm", "glr")).lower()
        if arm not in self.ARMS:
            raise ValueError(f"[URB-BL][qcdr] --arm must be one of {self.ARMS}, "
                             f"got {arm!r}")
        self.arm = arm
        self.horizon = int(ctx.training_eps)
        # delta = 1/sqrt(T), the paper's convention for an unknown change count.
        self.delta = float(cfg.get("delta", 0.0) or 1.0 / math.sqrt(
            max(self.horizon, 2)))
        self.stride = int(cfg.get("glr_stride", 1))
        self.max_hist = int(cfg.get("max_hist", 500))
        self.min_hist = int(cfg.get("min_hist", 20))
        self.normalize = bool(cfg.get("normalize_stream", True))
        self.forced = bool(cfg.get("forced_exploration", True))
        self.print_every = int(cfg.get("print_every", 100))

        # forced exploration rate: the paper's GLRklUCB uses
        # alpha = sqrt(N_C log T / T); with N_C unknown, N_C = 1.
        self.alpha = float(cfg.get("alpha", 0.0) or math.sqrt(
            math.log(max(self.horizon, 2)) / max(self.horizon, 2)))

        # RR arm: intervals ~ Geo(eta_R), eta_R = sqrt(eta)/log T, eta = T^-xi.
        self.xi = float(cfg.get("xi", 0.5))
        eta = max(self.horizon, 2) ** (-self.xi)
        self.eta_r = math.sqrt(eta) / math.log(max(self.horizon, 3))

        self.rng = np.random.RandomState(int(ctx.torch_seed) * 31 + 7)
        self.calib = int(cfg.get("calib", 30))
        self.streams = {a: _Stream(self.n_actions, self.normalize,
                                   self.max_hist, self.calib)
                        for a in self.av_ids}
        self.masters = ({a: _Master(self.horizon, self.n_actions, self.delta,
                                    np.random.RandomState(
                                        int(ctx.torch_seed) * 101 + i))
                         for i, a in enumerate(self.av_ids)}
                        if self.arm == "master" else {})
        self.next_rr = {a: self._draw_rr() for a in self.av_ids}
        self.restarts = {a: 0 for a in self.av_ids}
        self.n_forced = 0
        self.n_decisions = 0
        self.glr_ratio_max = 0.0
        self.day = 0
        self._pending = {}           # agent -> action taken today
        self._g_tilde = {}           # agent -> optimistic value, today

        self.banner("QCD RESTART (B7: change point, drift unobserved)", [
            ("paper", "arXiv:2410.13772 (Gerogiannis, Huang, Veeravalli)"),
            ("arm", {"glr": "Bernoulli-GLR detector + restart",
                     "rr": "random restart, Geo(eta_R)",
                     "master": "MASTER's two tests + block schedule"}[arm]),
            ("base learner", "URB's own DQN (scripts/iql.py), unchanged"),
            ("horizon T", f"{self.horizon} days"),
            ("delta", f"{self.delta:.5f}  (1/sqrt(T))"),
            ("GLR threshold at n=100", f"{glr_threshold(100, self.delta):.2f}"),
            ("forced exploration", f"alpha = {self.alpha:.4f}" if self.forced
             else "OFF (their QCD++ variant)"),
            ("RR rate eta_R", f"{self.eta_r:.5f}  (~{self.eta_r * self.horizon:.1f}"
                              " restarts expected)"),
            ("reward stream", f"standardised on a frozen {self.calib}-day "
             "reference, logistic -> (0,1)" if self.normalize
             else "*** RAW -- the detector cannot fire at URB's scale"),
        ])

    # ------------------------------------------------------------------ helpers
    def _draw_rr(self):
        """Interval to the next random restart, ~ Geo(eta_R). Theorem 11."""
        p = min(max(self.eta_r, 1e-9), 1.0)
        return int(self.rng.geometric(p))

    def _restart(self, agent_id):
        """Flush the history and rebuild the learner: the paper's Algorithm 3
        ("clears history and restarts")."""
        self.models[agent_id] = self.make_model(agent_id)
        self.streams[agent_id].flush()
        self.restarts[agent_id] += 1

    # ------------------------------------------------------------------ hooks
    def begin_episode(self, day, phase):
        self.day = int(day)
        self._pending.clear()
        self._g_tilde.clear()

    def act(self, agent_id, obs):
        self.n_decisions += 1
        m = self.models[agent_id]
        # g~: the base learner's optimistic value for today's state, which is
        # what MASTER's tests are written against. Computed before the action so
        # that exploration does not change it.
        if self.arm == "master":
            import torch
            x = torch.FloatTensor(np.asarray(self.observation(agent_id, obs),
                                             dtype=np.float32)
                                  ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                self._g_tilde[agent_id] = float(m.q_network(x).max().item())
        if (self.forced and not self.deterministic
                and self.rng.rand() < self.alpha):
            # Forced exploration: GLRklUCB's device for keeping every arm
            # sampled, so that a change on an abandoned arm is still detectable.
            a = int(self.rng.randint(self.n_actions))
            m.last_state = np.asarray(self.observation(agent_id, obs),
                                      dtype=np.float32)
            m.last_action = a
            self.n_forced += 1
        else:
            a = int(m.act(self.observation(agent_id, obs)))
        self._pending[agent_id] = a
        return a

    def push(self, agent_id, reward):
        super(QCDRestart, self).push(agent_id, reward)
        if self.deterministic:
            return
        a = self._pending.get(agent_id)
        if a is None:
            return
        u = self.streams[agent_id].push(a, reward)

        if self.arm == "glr":
            h = self.streams[agent_id].hist[a]
            if len(h) >= self.min_hist:
                stat = bernoulli_glr(h, self.stride)
                thr = glr_threshold(len(h), self.delta)
                if thr > 0:
                    self.glr_ratio_max = max(self.glr_ratio_max, stat / thr)
                if stat > thr:
                    self._restart(agent_id)
        elif self.arm == "rr":
            self.next_rr[agent_id] -= 1
            if self.next_rr[agent_id] <= 0:
                self._restart(agent_id)
                self.next_rr[agent_id] = self._draw_rr()
        else:                                            # master
            if self.masters[agent_id].step(
                    u, self._g_tilde.get(agent_id, u)):
                self._restart(agent_id)

    # ------------------------------------------------------------------ report
    def diagnostics(self):
        d = super(QCDRestart, self).diagnostics()
        tot = sum(self.restarts.values())
        d.update({
            "restarts": tot,
            "rst/agent": tot / max(self.n_av, 1),
            "u_spread": float(np.mean([s.spread() for s in
                                       self.streams.values()])),
        })
        if self.arm == "glr":
            d["glr_frac"] = self.glr_ratio_max
        if self.arm == "master":
            d["m_trig"] = sum(m.trig1 + m.trig2 for m in self.masters.values())
        if self.forced:
            d["forced"] = self.n_forced / max(self.n_decisions, 1)
        return d

    def close(self):
        tot = sum(self.restarts.values())
        spread = float(np.mean([s.spread() for s in self.streams.values()]))
        print("\n" + "=" * 74)
        print("[qcdr] CHANGE-DETECTION REPORT")
        print(f"  arm                      {self.arm}")
        print(f"  restarts (all agents)    {tot}")
        print(f"  restarts per agent       {tot / max(self.n_av, 1):.2f}")
        print(f"  mapped-stream spread     {spread:.4f}")
        if self.forced:
            print(f"  forced-exploration rate  "
                  f"{self.n_forced / max(self.n_decisions, 1):.4f} "
                  f"(alpha = {self.alpha:.4f})")
        if self.arm == "glr":
            print(f"  largest GLR / threshold  {self.glr_ratio_max:.4f}"
                  "   (>1 is a detection)")
        if self.arm == "master":
            t1 = sum(m.trig1 for m in self.masters.values())
            t2 = sum(m.trig2 for m in self.masters.values())
            sch = sum(m.sched_restarts for m in self.masters.values())
            c1 = max([m.closest1 for m in self.masters.values()] or [0.0])
            c2 = max([m.closest2 for m in self.masters.values()] or [0.0])
            print(f"  MASTER test 1 fired      {t1}   (closest approach "
                  f"{c1:.2e} x threshold)")
            print(f"  MASTER test 2 fired      {t2}   (closest approach "
                  f"{c2:.2e} x threshold)")
            print(f"  scheduled restarts       {sch}")
            if t1 == 0 and t2 == 0:
                print("  RESULT: neither test EVER fired, so MASTER ran as its own")
                print("          random-restart schedule. This reproduces "
                      "Theorem 4 of")
                print("          arXiv:2410.13772 on URB: the thresholds need "
                      "T >= 1.24e9")
                print(f"          and this run had T = {self.horizon}. It is the "
                      "finding, not")
                print("          a degenerate arm -- report it.")
            if sch == 0:
                print("  *** THE BLOCK SCHEDULE NEVER ADVANCED either, so nothing")
                print("  *** in this arm ever engaged. That IS degenerate.")
        if spread < 1e-4:
            print("  *** THE MAPPED REWARD STREAM IS CONSTANT. No change detector")
            print("  *** can fire on a constant. If normalize_stream is false,")
            print("  *** that is the documented cause (see the module docstring);")
            print("  *** otherwise the agents are not moving between routes.")
        elif self.arm in ("glr", "rr") and tot == 0:
            print("  *** NO AGENT EVER RESTARTED, so this arm is URB's own IQL")
            print("  *** with extra bookkeeping. For --arm glr check "
                  "'largest GLR /")
            print("  *** threshold' above: if it is far below 1 the detector "
                  "never came")
            print("  *** close, which is itself reportable.")
        print("=" * 74 + "\n", flush=True)


def add_args(parser):
    parser.add_argument('--arm', type=str, default=None, choices=list(QCDRestart.ARMS),
                        help="glr: Bernoulli-GLR detection + restart (default). "
                             "rr: random restart at Geo(eta_R) intervals, the "
                             "paper's order-optimal control. master: MASTER's "
                             "two tests, to check that they never fire here.")

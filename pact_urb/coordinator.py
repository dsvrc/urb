"""PACT for URB -- the per-day identify -> compensate cycle and every diagnostic.

`PACT_PIPELINE_SPEC` architecture, instantiated on the URB NS:

    obs_i ---> pi (host RL, UNCHANGED) ---> route_i
                                             |
    peers' EXECUTED exertion (t-1) --> psi ---+   exact arithmetic, no estimation
    own loading u_i(t) --------------> y -----+   the sensor (binding v/c)
                                             v
       RLS(psi, y)  --->  beta_hat            r+2 parameters
                                             v
       ell_hat  = beta_hat[peer] . psi   minus its standing level
       g        = max_trust if admissible else 0
       d_peer   = -g * ell_hat / (du/d offset)          the channel inverse
       d_ff     = -ff_gain * u_i (1 - g_dial) / (du/d offset)   analytic
                                             v
       offset_i = clip(d_peer + d_ff, +/- max_offset)

TWO PROPERTIES PRESERVED AT ALL COSTS (§1)
--------------------------------------------------------------------------------
  * **The host RL is untouched.** No loss terms, no critic changes, no extra action
    dimensions. The compensator only sets a departure offset in the SeverityLayer.
  * **The floor property.** When the gates say inadmissible the offset is exactly
    zero, so the executed day is byte-identical to the blind arm. A diverging
    estimate can fail to help; it must never do worse.

THE REGRESSOR IS ASYMMETRIC IN TIME (§4.2 [PAID])
--------------------------------------------------------------------------------
    psi_reg = [ 1 , own_col(t) , psi_peer(t-1) ]        target = u_i(t)

  * own column CURRENT -- the agent knows its own action exactly.
  * peer columns PREVIOUS -- it cannot observe peers' current choices. Using them
    would be an oracle; using yesterday's is a persistence assumption, sound
    because the coupling is slow by construction.

> Regressing y(t) on the whole of psi(t-1) lags the own-gain column by one step
> and corrupts the very coefficient the channel inverse divides by.
"""

import csv
import os
import time

import numpy as np

from pact_urb.core import (
    AgentRLS,
    NullTracker,
    PedestalEMA,
    analytic_feedforward,
    trust_gate,
)


class PactCoordinator(object):
    """Owns the declared basis, the per-agent estimators, and the compensator."""

    COLUMNS = [
        "day", "phase", "A", "g_mean", "dry", "sigma",
        # was the method ON AT ALL -- read these two first (§9)
        "applied_trust", "delta_nonzero_frac", "delta_clip_frac", "admissible_frac",
        # the local vs coordination split -- §7 honesty condition 1
        "ff_abs", "peer_abs", "peer_share",
        # the estimator
        "fit_gain_now", "beta_own", "beta_peer", "trP", "clamp_frac", "n_upd",
        "innov", "cond_psi",
        # the NS itself
        "u_mean", "u_max", "m_mean", "m_max", "excess_mean",
        # where the return is
        "tt_cav", "tt_hdv", "reward", "cav_adv", "tt_cav_roll",
        # the commons (T4)
        "route_switch_frac", "herd_index", "offset_mean_abs", "offset_std",
        "wall_s",
    ]

    def __init__(self, network, loading, severity, av_slots, cfg, run_dir=None,
                 exp_id=None):
        self.net = network
        self.loading = loading
        self.sev = severity
        self.av = np.asarray(av_slots, dtype=np.int64)
        self.n_av = len(self.av)
        self.cfg = dict(cfg)
        self.K = loading.K

        # ---- declared constants (guide II.1) ---------------------------------
        self.r = int(cfg.get("r", 1))                     # §2.2: DEFAULT r = 1
        self.mu = float(cfg.get("forget", 0.9995))        # §5.1 measured optimum
        self.p0 = float(cfg.get("p0", 10.0))
        self.max_trust = float(cfg.get("max_trust", 0.5))  # Phase-1 calibrated
        self.ff_gain = float(cfg.get("ff_gain", 1.0))
        self.fit_floor = float(cfg.get("fit_floor", 0.0))
        self.ready_updates = int(cfg.get("ready_updates", 100))
        self.window = int(cfg.get("fit_window", 200))
        self.warmup = int(cfg.get("fit_warmup", 50))
        self.tau = float(cfg.get("pedestal_tau", 400.0))
        self.max_offset = float(cfg.get("max_offset_s", 240.0))
        self.fd_step = float(cfg.get("fd_step_s", 30.0))
        self.arm = str(cfg.get("arm", "pact")).lower()
        assert self.arm in ("pact", "ff", "blind"), f"unknown arm {self.arm!r}"
        if self.arm == "blind":
            self.max_trust, self.ff_gain = 0.0, 0.0
        elif self.arm == "ff":
            self.max_trust = 0.0            # local term only: the info-matched base

        # ---- the DECLARED basis, built once, never fitted (§2.1) -------------
        self.W = network.build_operator(loading.gids, loading.O)
        self.W_av = self.W[np.ix_(self.av, np.arange(loading.N))]

        # ---- per-agent, per-channel scaling (§2.3 [PAID]) --------------------
        # range[i] = sum_j W[i,j] * Phi_j^max ; x_ref = 0.5*range (uniform peers);
        # psi lands in [-0.5, 0.5]. A single shared scale drove cond to 72,148
        # against 57, because an agent whose channel is ~100x smaller contributes a
        # near-zero column and the Gram goes singular. Each agent runs its own
        # estimator, so per-agent scaling leaks nothing.
        self.rng_scale = self.W_av @ np.nan_to_num(loading.phi_max, nan=0.0)
        self.rng_scale = np.where(self.rng_scale > 1e-12, self.rng_scale, 1.0)
        self.x_ref = 0.5 * self.rng_scale

        # ---- method state ------------------------------------------------------
        p = 2 + self.r                                   # [1, own, peer x r]
        self.p = p
        self.rls = [AgentRLS(p, self.mu, self.p0) for _ in range(self.n_av)]
        self.null = [NullTracker(self.window, self.warmup) for _ in range(self.n_av)]
        self.pedestal = PedestalEMA(self.tau, self.n_av)
        self._psi_prev = None
        self._have_prev = False
        self._offsets = np.zeros(loading.N)
        self._last_choice = None

        # ---- rolling / logging --------------------------------------------------
        self.roll_n = int(cfg.get("roll_episodes", 100))
        self._roll_cav = []
        self._t0 = time.time()
        self._fit_hist = []
        self._dbg = self._dbg_w = None
        self.debug_path = None
        d = cfg.get("debug_dir", run_dir)
        if d:
            os.makedirs(d, exist_ok=True)
            name = f"pact_ns_{exp_id}.csv" if exp_id else "pact_ns_debug.csv"
            self.debug_path = os.path.abspath(os.path.join(d, name))
            self._dbg = open(self.debug_path, "w", newline="", encoding="utf-8")
            self._dbg_w = csv.writer(self._dbg)
            self._dbg_w.writerow(self.COLUMNS)
            self._dbg.flush()
        self.banner()

    # ------------------------------------------------------------------ banner
    def banner(self):
        print("\n" + "=" * 74)
        print(f"[PACT] arm={self.arm}   r={self.r}   max_trust={self.max_trust}   "
              f"ff_gain={self.ff_gain}")
        print("=" * 74)
        print("  The coupling operator W is DECLARED from link-path incidence and")
        print("  link capacity -- the traffic counterpart of PTDF. Never fitted.")
        print(f"  basis        {self.n_av} agents, p={self.p} "
              f"[1, own, peer x {self.r}]")
        print(f"  estimator    RLS forget={self.mu} p0={self.p0}, windup bound ON")
        print(f"  gate         binary admissibility (fit_gain > {self.fit_floor}, "
              f">= {self.ready_updates} updates) x constant {self.max_trust}")
        print(f"  channel      departure offset, +/- {self.max_offset:.0f} s; the "
              f"divisor du/d(offset) comes from the OPERATOR, not the estimator")
        print(f"  pedestal     slow EMA tau={self.tau} -- compensate the "
              "fluctuation, not the standing level")
        if self.arm == "blind":
            print("  [ARM blind]  trust and feedforward BOTH 0 -> offsets are")
            print("               exactly zero. Byte-identical to the stock task.")
        elif self.arm == "ff":
            print("  [ARM ff]     analytic feedforward ONLY -- the LOCAL term. This")
            print("               is the information-matched baseline (§7.2): any")
            print("               method with the domain model can compute u*(1-g).")
        if self.debug_path:
            print(f"  trace        {self.debug_path}")
        print("=" * 74 + "\n", flush=True)

    # ------------------------------------------------------------------ per day
    def plan_offsets(self, prev_choice):
        """Compute this day's departure offsets BEFORE the day runs.

        Uses yesterday's executed peer choices (§4.2's persistence assumption) and
        the driver, which is a function of observable time. Returns (N,) seconds.
        """
        off = np.zeros(self.loading.N)
        if self.arm == "blind" or prev_choice is None:
            self._offsets = off
            return off

        g = self.sev.g
        # the divisor: DECLARED, from the operator (§6.1 [PAID] -- learning it
        # produced |t| = 0.024 and never once cleared a t > 3 bar)
        du_da = self.loading.du_d_offset(prev_choice, g=g, h=self.fd_step)
        du_av = du_da[self.av]

        # --- the analytic feedforward: closed form, no estimator (§7) ---------
        u_prev, binding = self.loading.loading(prev_choice, g=g)
        # C.1's excess: u * (1 - g) at the link that actually binds. `g` is
        # per-link, so it is read at each agent's own binding link.
        g_bind = (np.asarray(g)[binding] if np.ndim(g) > 0
                  else np.full(self.loading.N, float(g)))
        excess = u_prev[self.av] * (1.0 - g_bind[self.av])
        d_ff = analytic_feedforward(excess, du_av, self.ff_gain, self.max_offset)

        # --- the peer term: the coordination contribution --------------------
        d_peer = np.zeros(self.n_av)
        fg = self._mean_fit_gain()
        n_upd = int(np.mean([e.n_updates for e in self.rls])) if self.rls else 0
        trust, admissible = trust_gate(fg, n_upd, self.fit_floor,
                                       self.ready_updates, self.max_trust)
        if trust > 0 and self._have_prev:
            ell = np.array([
                float(self.rls[i].beta[2:] @ np.atleast_1d(self._psi_prev[i, 2:]))
                for i in range(self.n_av)
            ])
            ell_ctrl = self.pedestal.step(ell)       # §6.4 fluctuation, not level
            d_peer = analytic_feedforward(ell_ctrl, du_av, trust, self.max_offset)
        else:
            self.pedestal.step(np.zeros(self.n_av))

        self._trust_applied = trust
        self._admissible = admissible
        self._d_ff, self._d_peer = d_ff, d_peer

        delta = np.clip(d_ff + d_peer, -self.max_offset, self.max_offset)
        self._clip_frac = float(np.mean(np.abs(d_ff + d_peer) > self.max_offset))
        off[self.av] = delta
        self._offsets = off
        return off

    def observe(self, choice):
        """Identify from the day that just happened. §4.2's asymmetric regressor."""
        g = self.sev.g
        u, _ = self.loading.loading(choice, g=g, offsets=self._offsets)
        y = u[self.av]

        # peer column: W-weighted peer exertion, per-agent scaled to [-0.5, 0.5]
        phi = self.loading.phi_opt[np.arange(self.loading.N),
                                   np.asarray(choice, dtype=np.int64)]
        x_peer = (self.W_av @ np.nan_to_num(phi, nan=0.0) - self.x_ref) / self.rng_scale
        own = phi[self.av]

        psi = np.empty((self.n_av, self.p))
        psi[:, 0] = 1.0
        psi[:, 1] = own                                   # CURRENT (§4.2)
        if self._have_prev:
            psi[:, 2:] = self._psi_prev[:, 2:]            # PREVIOUS (§4.2)
            for i in range(self.n_av):
                full = self.rls[i].predict(psi[i])
                null = float(self.rls[i].beta[0] + self.rls[i].beta[1] * psi[i, 1])
                self.null[i].add(y[i], full, null)        # score PRIOR predictions
                self.rls[i].update(psi[i], y[i])
        else:
            psi[:, 2:] = 0.0

        nxt = np.empty((self.n_av, self.p))
        nxt[:, 0] = 1.0
        nxt[:, 1] = own
        nxt[:, 2] = x_peer
        self._psi_prev = nxt
        self._have_prev = True
        self._last_psi_cond = _cond(psi)
        return y

    def _mean_fit_gain(self):
        v = [n.fit_gain() for n in self.null]
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    # ------------------------------------------------------------------ logging
    def log(self, day, phase, choice, tt_cav, tt_hdv, reward, switch, herd):
        ns = self.sev.diagnostics()
        B = np.array([e.beta for e in self.rls])
        fg = self._mean_fit_gain()
        if np.isfinite(fg):
            self._fit_hist.append(fg)
        if np.isfinite(tt_cav):
            self._roll_cav.append(float(tt_cav))
            if len(self._roll_cav) > self.roll_n:
                del self._roll_cav[0]
        ff_abs = float(np.mean(np.abs(getattr(self, "_d_ff", np.zeros(1)))))
        pr_abs = float(np.mean(np.abs(getattr(self, "_d_peer", np.zeros(1)))))
        tot = ff_abs + pr_abs
        off = self._offsets[self.av]
        adv = (float(tt_hdv) / float(tt_cav)
               if np.isfinite(tt_hdv) and np.isfinite(tt_cav) and tt_cav > 1e-9
               else float("nan"))
        row = [
            int(day), phase, _r(ns["A"], 5), _r(ns["g_mean"], 6), ns["dry"],
            _r(self.sev.sigma, 3),
            _r(getattr(self, "_trust_applied", 0.0), 5),
            _r(float(np.mean(np.abs(off) > 1e-9)), 4),
            _r(getattr(self, "_clip_frac", 0.0), 4),
            _r(1.0 if getattr(self, "_admissible", False) else 0.0, 3),
            _r(ff_abs, 4), _r(pr_abs, 4),
            _r(pr_abs / tot if tot > 1e-12 else float("nan"), 4),
            _r(fg, 5), _r(B[:, 1].mean(), 6), _r(B[:, 2].mean(), 6),
            _r(float(np.mean([np.trace(e.P) for e in self.rls])), 4),
            _r(float(np.mean([e.n_clamped for e in self.rls])), 4),
            int(np.sum([e.n_updates for e in self.rls])),
            _r(float(np.mean([e.innov for e in self.rls])), 6),
            _r(getattr(self, "_last_psi_cond", float("nan")), 2),
            _r(ns["u_mean"], 5), _r(ns["u_max"], 5),
            _r(ns["m_mean"], 5), _r(ns["m_max"], 5), _r(ns["excess_mean"], 6),
            _r(tt_cav, 4), _r(tt_hdv, 4), _r(reward, 5), _r(adv, 5),
            _r(float(np.mean(self._roll_cav)) if self._roll_cav else np.nan, 4),
            _r(switch, 5), _r(herd, 5),
            _r(float(np.mean(np.abs(off))), 3), _r(float(np.std(off)), 3),
            _r(time.time() - self._t0, 3),
        ]
        if self._dbg_w is not None:
            self._dbg_w.writerow(row)
            self._dbg.flush()

        pe = int(self.cfg.get("print_every", 25))
        if pe > 0 and int(day) % pe == 0:
            print(f"[PACT d{int(day):5d} {phase:5s}] A={ns['A']:.3f} "
                  f"g={ns['g_mean']:.4f}{' DRY' if ns['dry'] else '    '} | "
                  f"trust={getattr(self, '_trust_applied', 0.0):.3f} "
                  f"fit_gain={_f(fg)} | ff={ff_abs:6.1f}s peer={pr_abs:6.1f}s "
                  f"({_f(pr_abs / tot if tot > 1e-12 else float('nan'), 2)} peer) | "
                  f"u={ns['u_mean']:.3f} m={ns['m_mean']:.4f} | "
                  f"tt_cav={_f(tt_cav, 3)} adv={_f(adv, 3)}", flush=True)

    def close(self):
        if self._dbg is not None:
            try:
                self._dbg.close()
            except Exception:
                pass
            self._dbg = self._dbg_w = None


# ==========================================================================
def _cond(psi):
    psi = np.asarray(psi, dtype=np.float64)
    if psi.shape[0] < psi.shape[1] + 2:
        return float("nan")
    M = psi.T @ psi / psi.shape[0]
    try:
        return float(np.linalg.cond(M))
    except np.linalg.LinAlgError:
        return float("nan")


def _r(v, n):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    return round(v, n) if np.isfinite(v) else ""


def _f(v, n=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "nan"
    return f"{v:.{n}f}" if np.isfinite(v) else "nan"

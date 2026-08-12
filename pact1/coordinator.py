"""The per-day PACT-1 cycle: identify -> forecast -> steer, plus every gate.

ONE DAY, IN ORDER
--------------------------------------------------------------------------------
    begin_episode()      forecast tomorrow's peer load from the EMA'd occupancy and
                         each agent's own beta_hat -> tt_hat[i, k] for every
                         candidate route. Pure arithmetic; no estimation here.

    route_context(id)    the policy asks for its own tt_hat + confidence, applies
                         the trust-weighted logit shift, and chooses a route.

    end_episode(...)     the day is simulated. Read every traveller's executed route
                         and every AV's realized travel time:
                           * build the TRUE waveforms from actual peer choices
                           * score the forecast that was actually used  (pred_r2)
                           * score the model on the true waveform       (fit_r2)
                           * RLS-update each agent on its own single row
                           * roll the occupancy EMA forward
                           * write one diagnostic row and run the gates

WHY THE TWO R-SQUARED COLUMNS ARE THE WHOLE INSTRUMENT PANEL
--------------------------------------------------------------------------------
``fit_r2``  -- beta_hat . psi_TRUE  vs realized excess. Answers "does the reduction
               hold in this city at all?" If this is near zero the linear
               road-class model is wrong and NOTHING downstream can work. This is
               the gate that stops an expensive run early.

``pred_r2`` -- the forecast actually used, vs realized excess. Answers "is peer load
               predictable a day ahead?" fit high + pred low means the physics is
               right but the fleet is flapping faster than rho can track; that is a
               finding about the domain, not a bug.

Keeping them apart is what stops a bad number being blamed on the wrong thing.
"""

import csv
import os
import time
from collections import deque

import numpy as np

from pact1.core import (
    AgentRLS,
    herd_index,
    predict_excess,
    relative_excess,
    rls_confidence,
    trust_from_w,
)

_BANNER_SHOWN = False


class RouteContext(object):
    """What the policy needs to steer. Deliberately tiny and read-only."""

    __slots__ = ("slot", "tt_hat", "d_hat", "conf", "beta_hat", "valid")

    def __init__(self, slot, tt_hat, d_hat, conf, beta_hat, valid):
        self.slot = slot
        self.tt_hat = tt_hat
        self.d_hat = d_hat
        self.conf = conf
        self.beta_hat = beta_hat
        self.valid = valid


class Pact1Coordinator(object):
    """Owns the basis, the per-agent estimators, the forecast, and the diagnostics."""

    # ------------------------------------------------------------------ init
    def __init__(self, basis, agent_table, av_ids, free_flow, cfg, run_dir,
                 log_name="pact1_debug.csv", exp_id=None):
        """
        Args:
            basis:       a built ``RouteBasis``.
            agent_table: dict id(str) -> {"od": (o, d), "start": float,
                                          "machine": bool}. EVERY traveller.
            av_ids:      ordered list of machine-agent ids (str). Defines slots.
            free_flow:   dict (o, d) -> [fft per action index] (from the env).
            cfg:         the ``pact1`` block of the algo config.
            run_dir:     where ``pact1_debug.csv`` goes.
        """
        self.basis = basis
        self.cfg = dict(cfg)
        self.K = int(basis.n_paths)
        self.r = int(basis.r)
        self.p = self.r + 1                       # + intercept

        # ---- declared constants (guide II.1: dial vs constant) ----------------
        self.rho = float(cfg.get("rho", 0.8))                  # occupancy EMA
        self.mu = float(cfg.get("forget", 0.999))              # RLS forgetting
        self.p0 = float(cfg.get("p0", 10.0))                   # RLS prior looseness
        self.g_max = float(cfg.get("g_max", 1.0))
        self.g_ema = float(cfg.get("g_ema", 0.9))
        self.g_bias = float(cfg.get("g_bias", 2.2))            # -> 0.90 at w=0
        self.kappa = float(cfg.get("kappa", 1.0))              # logit-shift scale
        self.use_conf = bool(cfg.get("confidence_gate", True))
        self.y_clip = float(cfg.get("y_clip", 10.0))
        self.peer_scope = str(cfg.get("peer_scope", "all")).lower()
        self.trust_mode = str(cfg.get("trust_mode", "learned")).lower()
        self.freeze_beta = cfg.get("freeze_beta", None)
        self.raw_feature_obs = bool(cfg.get("raw_feature_obs", False))
        assert self.peer_scope in ("all", "fleet"), (
            f"peer_scope must be 'all' or 'fleet' (got {self.peer_scope!r})"
        )
        assert self.trust_mode in ("learned", "off", "fixed"), (
            f"trust_mode must be 'learned'|'off'|'fixed' (got {self.trust_mode!r})"
        )
        self.trust_fixed = float(cfg.get("trust_fixed", 0.9))

        # ---- gates ------------------------------------------------------------
        self.gate_abort = bool(cfg.get("gate_abort", True))
        self.gate_after = int(cfg.get("gate_after_episodes", 300))
        self.gate_min_fit_r2 = float(cfg.get("gate_min_fit_r2", 0.05))
        self.gate_window = int(cfg.get("gate_window", 50))
        self.gate_live_after = int(cfg.get("gate_live_after_episodes", 20))
        self.gate_cond_warn = float(cfg.get("gate_cond_warn", 1e6))
        self._gate_fired = {"live": False, "fit": False, "cond": False}

        # ---- agent bookkeeping -------------------------------------------------
        self.av_ids = list(av_ids)
        self.n_av = len(self.av_ids)
        self.slot_of = {a: i for i, a in enumerate(self.av_ids)}

        all_ids = sorted(agent_table.keys(), key=lambda s: (len(s), s))
        self.all_ids = all_ids
        self.gidx_of = {a: i for i, a in enumerate(all_ids)}
        self.n_all = len(all_ids)

        starts = np.array([float(agent_table[a]["start"]) for a in all_ids])
        ods = [tuple(agent_table[a]["od"]) for a in all_ids]
        self.od_of_all = ods
        self.is_machine = np.array(
            [bool(agent_table[a]["machine"]) for a in all_ids], dtype=bool
        )

        # trip duration for the co-presence window: mean free-flow time over the
        # traveller's own route options. Exogenous (schedule + geometry), fixed.
        dur = np.zeros(self.n_all)
        for i, od in enumerate(ods):
            ff = free_flow.get(od)
            if ff is not None and len(ff):
                v = np.asarray(ff, dtype=np.float64)
                v = v[np.isfinite(v) & (v > 0)]
                if v.size:
                    dur[i] = float(v.mean())
        self.durations = dur
        self.dur_factor = float(cfg.get("dur_factor", 1.0))
        self.overlap_slack = float(cfg.get("overlap_slack", 0.0))

        from pact1.basis import build_time_overlap
        self.O = build_time_overlap(
            starts, dur * self.dur_factor, slack=self.overlap_slack
        )

        # peer scope -> which columns of O are live
        if self.peer_scope == "fleet":
            self.peer_global = np.array(
                [self.gidx_of[a] for a in self.av_ids], dtype=np.int64
            )
            self.peer_ids = list(self.av_ids)
        else:
            self.peer_global = np.arange(self.n_all, dtype=np.int64)
            self.peer_ids = list(all_ids)
        self.peer_slot_of = {a: j for j, a in enumerate(self.peer_ids)}
        self.n_peer = len(self.peer_ids)

        av_global = np.array([self.gidx_of[a] for a in self.av_ids], dtype=np.int64)
        self.O_rows = np.ascontiguousarray(self.O[np.ix_(av_global, self.peer_global)])
        self.n_overlap_mean = float(self.O_rows.sum(1).mean()) if self.n_av else 0.0

        # ---- route ids and free-flow times per AV -----------------------------
        self.agent_gids = np.full((self.n_av, self.K), -1, dtype=np.int64)
        self.agent_fft = np.full((self.n_av, self.K), np.nan, dtype=np.float64)
        bad_od = []
        for a, i in self.slot_of.items():
            od = tuple(agent_table[a]["od"])
            if od not in basis.od_index:
                bad_od.append((a, od))
                continue
            ff = free_flow.get(od, [np.nan] * self.K)
            for k in range(self.K):
                self.agent_gids[i, k] = basis.gid_of[(od, k)]
                self.agent_fft[i, k] = float(ff[k]) if k < len(ff) else np.nan
        if bad_od:
            raise ValueError(
                f"[PACT-1] {len(bad_od)} machine agents have an OD pair absent from "
                f"paths.csv, e.g. {bad_od[:3]}. The basis cannot describe them."
            )

        # peer -> gid lookup table, filled per episode from executed actions
        self.peer_od = [tuple(agent_table[a]["od"]) for a in self.peer_ids]
        self.peer_gid_table = np.full((self.n_peer, self.K), -1, dtype=np.int64)
        for j, od in enumerate(self.peer_od):
            if od in basis.od_index:
                for k in range(self.K):
                    self.peer_gid_table[j, k] = basis.gid_of[(od, k)]

        # ---- reference load: the basis's own zero point -------------------------
        # x_ref[m, i, k] is the class-m load agent i would see on route k if every
        # peer chose UNIFORMLY at random. It is a function of the network geometry
        # and the (fixed) travel schedule ONLY -- no run data touches it -- so it is
        # part of the declared model class, not something fitted.
        #
        # WHY IT IS HERE. Raw x has a large common mean (measured on a synthetic
        # city: mean 37, std 9 against an intercept column of 1), which makes the
        # design matrix badly conditioned -- cond(E[psi psi^T]) ~ 1.3e5 -- so the
        # intercept and the class channels trade off against each other and the
        # SPLIT becomes unidentifiable even when the prediction is fine (exactly the
        # failure mode of guide III.6). Centring on the geometric zero point and
        # scaling per class turns the regressor into "how much more class-m traffic
        # than usual", which is both better conditioned and the more physical
        # statement. beta absorbs the change of units exactly; nothing about the
        # method changes.
        xr = np.zeros((self.r, self.n_av, self.K))
        for kp in range(self.K):
            xr += basis.waveforms(
                self.O_rows, self.peer_gid_table[:, kp], self.agent_gids[:, kp],
                self.agent_gids,
            )
        self.x_ref = xr / float(self.K)
        s = np.abs(self.x_ref).mean(axis=(1, 2))
        self.x_scale = np.where(s > 1e-9, s, 1.0)

        # ---- method state ------------------------------------------------------
        self.rls = [AgentRLS(self.p, self.mu, self.p0) for _ in range(self.n_av)]
        self.X_ema = np.zeros((self.r, self.n_av, self.K))     # forecast occupancy
        self.d_hat = np.zeros((self.n_av, self.K))             # predicted excess
        self.tt_hat = np.array(self.agent_fft, copy=True)      # predicted travel time
        self.conf = np.full(self.n_av, 1.0 / (1.0 + self.p))   # cold estimator
        self.g_pol = np.full(
            self.n_av, self.g_max / (1.0 + np.exp(-self.g_bias))
        )
        self.g_app = np.zeros(self.n_av)
        self.last_y = np.zeros(self.n_av)                      # last realized excess
        self.last_action = np.full(self.n_av, -1, dtype=np.int64)
        self._have_prev = False

        # ---- per-episode scratch ------------------------------------------------
        self._ep = 0
        self._phase = "train"
        self._trust_log = np.zeros((self.n_av, 3))             # g_pol, g_app, |shift|
        self._trust_seen = np.zeros(self.n_av, dtype=bool)
        self._fit_hist = []
        self._last_cond = float("nan")
        self._t0 = time.time()

        # ---- rolling window for the "is it beating the baseline" read ----------
        # Per-episode travel time is noisy enough that a single row says nothing.
        # URB's own winrate criterion is "CAVs were on average faster than human
        # drivers", i.e. cav_adv = t_HDV / t_CAV > 1, so that is what gets smoothed
        # and printed -- it is config-independent and directly comparable to the
        # published table.
        self.roll_n = int(cfg.get("roll_episodes", 100))
        self._roll_cav = deque(maxlen=self.roll_n)
        self._roll_adv = deque(maxlen=self.roll_n)

        # ---- diagnostics file ----------------------------------------------------
        # Default location is the REPO ROOT, not results/<exp_id>/, so a long run can
        # be tailed without hunting for it. Name carries exp_id so parallel arms
        # never clobber each other.
        self.run_dir = run_dir
        self._dbg = self._dbg_w = None
        debug_dir = cfg.get("debug_dir", run_dir)
        if exp_id:
            log_name = f"pact1_debug_{exp_id}.csv"
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            path = os.path.join(debug_dir, log_name)
            self._dbg = open(path, "w", newline="", encoding="utf-8")
            self._dbg_w = csv.writer(self._dbg)
            self._dbg_w.writerow(self.COLUMNS)
            self._dbg.flush()
            self.debug_path = os.path.abspath(path)
        else:
            self.debug_path = None

        self.obs_aug_dim = self.K + self.p + 2 + (self.r if self.raw_feature_obs else 0)

    # ------------------------------------------------------------------ columns
    COLUMNS = [
        "episode", "phase", "n_av", "n_peer", "n_peer_valid",
        # estimator
        "beta_intercept", "beta_local", "beta_collector", "beta_arterial",
        "beta_spread", "innov", "conf", "n_upd",
        # THE gates
        "fit_r2", "pred_r2", "fit_mae", "pred_mae", "cond_psi",
        # excitation / liveness
        "x_local", "x_collector", "x_arterial", "x_std",
        # trust -- applied and policy-set reported SEPARATELY (guide III.11)
        "trust_pol", "trust_app", "trust_spread", "shift_absmean",
        # WHERE THE RETURN IS -- these are the columns that answer "is it winning".
        # tt_cav / tt_hdv are in RouteRL's own units, so they are directly
        # comparable to the t_CAV / t_HDV of URB's published table (St. Arnoult
        # reference: t_pre 3.15, QMIX 3.21, IPPO 3.33, greedy/AON 3.01).
        # cav_adv = t_HDV / t_CAV is URB's CAV-advantage; > 1 is a "won" run by the
        # benchmark's own winrate definition.
        "tt_cav", "tt_hdv", "cav_adv", "tt_cav_roll", "cav_adv_roll",
        "reward", "excess_mean", "clip_frac",
        # the commons signature (T4 / guide III.8)
        "route_switch_frac", "herd_index",
        "wall_s",
    ]

    # ------------------------------------------------------------------ copying
    def __deepcopy__(self, memo):
        """Never copied -- this is a shared service, not agent state.

        RouteRL snapshots ``all_agents`` with ``copy.deepcopy`` on every episode
        reset. Each machine agent carries ``agent.model``, and the model holds a
        reference to this coordinator, so without this the snapshot would try to
        deepcopy:

          * the open ``pact1_debug_*.csv`` handle -> TypeError: cannot pickle
            'TextIOWrapper' instances, which kills the run at episode 0; and
          * the whole basis -- three (R x R) Gram matrices, ~18 MB on
            saint_arnoult -- once per episode, for nothing.

        Returning ``self`` makes the snapshot share the coordinator instead, which
        is both correct (a per-day record of an agent has no business owning a copy
        of the fleet's estimator) and free.
        """
        memo[id(self)] = self
        return self

    def __copy__(self):
        return self

    # ================================================================== banner
    def banner(self, extra=None):
        global _BANNER_SHOWN
        if _BANNER_SHOWN:
            return
        _BANNER_SHOWN = True
        s = self.basis.summary()
        print("\n" + "=" * 78)
        print("[PACT-1 / URB]  online identification of the congestion coupling")
        print("=" * 78)
        print(
            "  The road-CLASS partition is known (it is painted on the street).\n"
            "  The marginal delay per peer vehicle on each class is NOT: it is a\n"
            "  drifting r-vector tracked ONLINE and PER-AGENT by RLS on that\n"
            "  traveller's own realized-vs-free-flow residual. The policy learns\n"
            "  ONE scalar: how much to trust it.\n"
            "  FLOOR: trust = 0 reproduces plain IPPO exactly, for ANY beta_hat."
        )
        print("-" * 78)
        print(
            f"  basis      r={self.r} classes {s['class_names']} "
            f"bounds={s['speed_bounds_mps']} m/s"
            + ("   *** TERTILE FALLBACK ***" if s["fallback_tertiles"] else "")
        )
        print(f"             length share  {s['class_length_share']}")
        print(f"             {s['n_od']} OD pairs x {s['n_paths']} routes = "
              f"{s['n_routes']} route options over {s['n_edges_used']} links "
              f"({s['basis_mb']} MB)")
        if s["missing_edge_frac"] > 0:
            print(f"             WARNING: {s['missing_edge_frac']:.4f} of route edges "
                  f"were not found in the network file")
        print(f"  estimator  p={self.p} (intercept + {self.r}) forget={self.mu} "
              f"p0={self.p0} conf_gate={self.use_conf}")
        print(f"  forecast   occupancy EMA rho={self.rho}  "
              f"(rho=0 => yesterday-only persistence)")
        print(f"  trust      mode={self.trust_mode} g_max={self.g_max} "
              f"bias={self.g_bias} -> init "
              f"{self.g_max / (1.0 + np.exp(-self.g_bias)):.3f}  "
              f"ema={self.g_ema} kappa={self.kappa}")
        print(f"  peers      scope={self.peer_scope} n_peer={self.n_peer} "
              f"mean co-present peers={self.n_overlap_mean:.1f}")
        print(f"  regressor  centred on the uniform-mix reference load and scaled "
              f"per class\n"
              f"             x_ref mean {np.round(self.x_ref.mean(axis=(1, 2)), 4).tolist()}  "
              f"x_scale {np.round(self.x_scale, 4).tolist()}")
        print(f"  agents     {self.n_av} machine of {self.n_all} travellers")
        if self.freeze_beta is not None:
            print(f"  [ABLATION] freeze_beta={self.freeze_beta}: estimator BYPASSED")
        if self.raw_feature_obs:
            print("  [ABLATION] raw_feature_obs: x_m appended to the observation")
        if self.trust_mode == "off":
            print("  [ARM] trust_mode=off -> this is the BLIND arm (plain IPPO through\n"
                  "        the identical wrapper, so nothing else can differ).")
        print(f"  gates      abort={self.gate_abort} fit_r2>={self.gate_min_fit_r2} "
              f"after {self.gate_after} eps (window {self.gate_window})")
        if self.debug_path:
            print(f"  trace      {self.debug_path}")
        if extra:
            for line in extra:
                print(f"  {line}")
        print("=" * 78 + "\n", flush=True)

    # ================================================================== startup gates
    def selfcheck(self, rng=None, n_sample=8, rtol=1e-9):
        """GATE 1+2, run BEFORE any episode. Costs nothing and catches the two
        failures that are invisible later: a miswired waveform and a broken floor.

        Returns a list of human-readable result lines; raises on failure when
        ``gate_abort`` is set.
        """
        from pact1.core import steer_logits

        rng = rng or np.random.RandomState(0)
        out = []

        # --- GATE 1: waveform arithmetic (vectorised vs the definition) ---------
        rows = rng.choice(self.n_av, size=min(n_sample, self.n_av), replace=False)
        peer_gid = self._random_peer_gid(rng)
        self_gid = np.array(
            [peer_gid[self.peer_slot_of[self.av_ids[i]]]
             if self.av_ids[i] in self.peer_slot_of else -1 for i in rows],
            dtype=np.int64,
        )
        O_sub = self.O_rows[rows]
        fast = self.basis.waveforms(O_sub, peer_gid, self_gid, self.agent_gids[rows])
        slow = self.basis.waveforms_bruteforce(
            O_sub, peer_gid, self_gid, self.agent_gids[rows]
        )
        err = float(np.max(np.abs(fast - slow)))
        scale = max(1e-12, float(np.max(np.abs(slow))))
        rel = err / scale
        ok1 = rel <= rtol
        out.append(
            f"GATE 1 waveform arithmetic : {'PASS' if ok1 else 'FAIL'} "
            f"(max rel err {rel:.3e} over {len(rows)} agents x {self.K} routes)"
        )

        # --- GATE 2: the floor property (trust 0 must not touch the policy) -----
        lg = rng.randn(self.K)
        tt = rng.randn(self.K) * 5.0 + 100.0
        ok2 = bool(np.array_equal(steer_logits(lg, tt, 0.0, self.kappa), lg))
        # and a non-zero trust must actually move something, or the channel is dead
        moved = float(np.abs(steer_logits(lg, tt, 1.0, self.kappa) - lg).max())
        ok2 = ok2 and moved > 1e-9
        out.append(
            f"GATE 2 floor property      : {'PASS' if ok2 else 'FAIL'} "
            f"(g=0 exact; g=1 moves logits by {moved:.4f})"
        )

        # --- GATE 3: zero-diagonal / N=1 reduces to no fleet load ---------------
        lone = np.array([-1] * self.n_peer, dtype=np.int64)
        j = self.peer_slot_of.get(self.av_ids[int(rows[0])], None)
        ok3 = True
        if j is not None:
            lone[j] = self.agent_gids[int(rows[0]), 0]
            w = self.basis.waveforms(
                self.O_rows[[int(rows[0])]], lone,
                np.array([lone[j]]), self.agent_gids[[int(rows[0])]]
            )
            ok3 = bool(np.max(np.abs(w)) < 1e-12)
        out.append(
            f"GATE 3 zero-diagonal (N=1) : {'PASS' if ok3 else 'FAIL'} "
            "(a lone traveller reads exactly zero peer load)"
        )

        for line in out:
            print("[PACT-1] " + line, flush=True)
        if not (ok1 and ok2 and ok3):
            msg = ("[PACT-1][GATE FAIL] startup arithmetic check failed. This is a "
                   "WIRING bug, not a tuning issue -- fix it, do not reinterpret.")
            if self.gate_abort:
                raise AssertionError(msg + "\n" + "\n".join(out))
            print(msg + "  (gate_abort=false: continuing)", flush=True)
        return out

    def check_fft_alignment(self, env_ffts, rtol=1e-3):
        """GATE 4: paths.csv route order vs the env's own free-flow times.

        A permuted route order is the one silent catastrophe this file cannot
        detect any other way: every waveform would describe the wrong road and
        every diagnostic would still look plausible.
        """
        n, worst, bad = self.basis.check_fft(env_ffts, rtol=rtol)
        ok = (n > 0) and not bad
        print(
            f"[PACT-1] GATE 4 route-order alignment: {'PASS' if ok else 'FAIL'} "
            f"({n} routes checked, max rel err {worst:.3e})",
            flush=True,
        )
        if not ok:
            msg = (
                "[PACT-1][GATE FAIL] paths.csv free-flow times do not match "
                f"env.get_free_flow_times(). {len(bad)} mismatches, e.g. {bad[:3]}.\n"
                "        The action index does NOT line up with the route order in "
                "paths.csv, so every waveform would point at the wrong road."
            )
            if self.gate_abort:
                raise AssertionError(msg)
            print(msg + "  (gate_abort=false: continuing)", flush=True)
        return ok

    def _random_peer_gid(self, rng):
        g = np.full(self.n_peer, -1, dtype=np.int64)
        for j in range(self.n_peer):
            valid = self.peer_gid_table[j]
            if valid[0] >= 0:
                g[j] = valid[rng.randint(self.K)]
        return g

    # ================================================================== per day
    def begin_episode(self, episode, phase="train"):
        """Forecast what each candidate route will cost tomorrow. Pure arithmetic."""
        self._ep = int(episode)
        self._phase = phase
        self._t0 = time.time()
        self._trust_log[:] = 0.0
        self._trust_seen[:] = False

        Xz = self.standardize(self.X_ema)
        for i in range(self.n_av):
            psi = np.concatenate(
                [np.ones((self.K, 1)), Xz[:, i, :].T], axis=1
            )                                            # (K, p)
            self.d_hat[i] = predict_excess(self._beta(i), psi)
        self.tt_hat = self.agent_fft * (1.0 + self.d_hat)

        if self.use_conf:
            self.conf = np.array(
                [rls_confidence(self.rls[i].P, self.p0, self.p)
                 for i in range(self.n_av)]
            )
        else:
            self.conf = np.ones(self.n_av)

    def _beta(self, i):
        if self.freeze_beta is not None:
            return np.full(self.p, float(self.freeze_beta))
        return self.rls[i].beta

    def standardize(self, x):
        """Raw waveform -> regressor units: (x - x_ref) / x_scale, per class.

        The ONE place this conversion happens. Identification and forecasting must
        agree exactly or beta means two different things in the two paths, so both
        go through here. The EMA commutes with it (both are affine), which is why
        ``X_ema`` can be kept in raw units for the liveness columns.
        """
        return (np.asarray(x, dtype=np.float64) - self.x_ref) / self.x_scale[:, None, None]

    def route_context(self, agent_id):
        """What the policy needs. ``None`` for a non-machine agent."""
        i = self.slot_of.get(str(agent_id))
        if i is None:
            return None
        valid = np.isfinite(self.tt_hat[i])
        return RouteContext(
            slot=i,
            tt_hat=self.tt_hat[i],
            d_hat=self.d_hat[i],
            conf=float(self.conf[i]) if self.trust_mode != "off" else 0.0,
            beta_hat=self._beta(i),
            valid=valid,
        )

    def trust_for(self, slot, w):
        """Resolve the applied trust from the policy's raw head output.

        g_applied = trust(w) * confidence(P), i.e. the policy's setting gated by the
        estimator's own covariance (guide III.5). ``trust_mode`` selects the arm:
        ``learned`` (default), ``fixed`` (constant, for the "is trust learned at
        all" ablation) or ``off`` (the blind arm -- g is hard 0, so the wrapper is
        provably inert).
        """
        if self.trust_mode == "off":
            self.g_pol[slot] = 0.0
            return 0.0, 0.0
        if self.trust_mode == "fixed":
            gp = self.trust_fixed
        else:
            gp = float(
                trust_from_w(w, self.g_pol[slot], self.g_ema, self.g_max, self.g_bias)
            )
        self.g_pol[slot] = gp
        return gp, gp * (float(self.conf[slot]) if self.use_conf else 1.0)

    def record_trust(self, slot, g_pol, g_app, shift_abs):
        self.g_app[slot] = g_app
        self._trust_log[slot] = (g_pol, g_app, shift_abs)
        self._trust_seen[slot] = True

    def augment_obs(self, agent_id, obs):
        """Append PACT-1 state to the observation, in NATIVE units.

        [ d_hat (K) , beta_hat (p) , trust (1) , last residual (1) ] and, under the
        ``raw_feature_obs`` ablation, the raw waveforms x_m (r) as well.

        The last block is guide III.12's trivial ablation: if a policy handed the
        raw peer-load features matches PACT-1, then the estimator is decoration.
        Run it early.
        """
        i = self.slot_of.get(str(agent_id))
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        if i is None:
            return np.concatenate([obs, np.zeros(self.obs_aug_dim)])
        parts = [obs, self.d_hat[i], self._beta(i),
                 [self.g_app[i]], [self.last_y[i]]]
        if self.raw_feature_obs:
            parts.append(self.X_ema[:, i, :].mean(axis=1))
        return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1)
                               for p in parts])

    # ================================================================== end of day
    def end_episode(self, av_records, peer_actions, reward_mean=np.nan,
                    tt_hdv=np.nan, episode=None, phase=None):
        """Identify from what actually happened, then roll the forecast forward.

        Args:
            av_records:   dict av_id(str) -> (action:int, travel_time:float)
            peer_actions: dict traveller_id(str) -> action:int, for EVERY traveller
                          in scope (machines, plus humans when peer_scope='all').
            reward_mean:  mean AV reward this episode (logging only).
            tt_hdv:       mean human travel time this episode (logging only).
            episode:      label for the trace row. Records can arrive a day or more
                          after the day they describe (RouteRL flushes every
                          ``save_every`` episodes), so the row is labelled with the
                          episode the DATA belongs to, not the loop counter.
            phase:        overrides the phase label for this row.
        """
        if episode is not None:
            self._ep = int(episode)
        if phase is not None:
            self._phase = phase
        # ---- peers' executed routes -> global route ids ------------------------
        peer_gid = np.full(self.n_peer, -1, dtype=np.int64)
        for aid, act in peer_actions.items():
            j = self.peer_slot_of.get(str(aid))
            if j is None or act is None:
                continue
            k = int(act)
            if 0 <= k < self.K:
                peer_gid[j] = self.peer_gid_table[j, k]
        n_peer_valid = int((peer_gid >= 0).sum())

        # ---- this AV's own executed route (for the zero-diagonal subtraction) --
        self_gid = np.full(self.n_av, -1, dtype=np.int64)
        act_of = np.full(self.n_av, -1, dtype=np.int64)
        tt_of = np.full(self.n_av, np.nan)
        for aid, (act, tt) in av_records.items():
            i = self.slot_of.get(str(aid))
            if i is None or act is None:
                continue
            k = int(act)
            if not (0 <= k < self.K):
                continue
            act_of[i] = k
            tt_of[i] = float(tt)
            j = self.peer_slot_of.get(str(aid))
            self_gid[i] = peer_gid[j] if j is not None else self.agent_gids[i, k]

        # ---- TRUE waveforms from the day that actually happened ---------------
        x_true = self.basis.waveforms(
            self.O_rows, peer_gid, self_gid, self.agent_gids
        )                                                    # (r, n_av, K)

        drove = np.nonzero(act_of >= 0)[0]
        fit_r2 = pred_r2 = fit_mae = pred_mae = cond_psi = float("nan")
        clip_frac = float("nan")
        excess_mean = float("nan")

        if drove.size:
            kk = act_of[drove]
            # psi on the route ACTUALLY driven -> one RLS row per traveller per day
            xz = self.standardize(x_true)
            psi = np.empty((drove.size, self.p))
            psi[:, 0] = 1.0
            for m in range(self.r):
                psi[:, m + 1] = xz[m, drove, kk]

            fft = self.agent_fft[drove, kk]
            raw = tt_of[drove] / np.maximum(fft, 1e-6) - 1.0
            y = relative_excess(tt_of[drove], fft, clip=self.y_clip)
            clip_frac = float(np.mean(np.abs(raw - y) > 1e-12))
            excess_mean = float(np.nanmean(y))

            # --- score BEFORE updating: both numbers are honest one-step-ahead --
            fit = np.array([float(self._beta(i) @ psi[t])
                            for t, i in enumerate(drove)])
            pred = self.d_hat[drove, kk]
            fit_r2, fit_mae = _score(y, fit)
            pred_r2, pred_mae = _score(y, pred)
            cond_psi = _cond(psi)
            self._last_cond = cond_psi

            # --- identify -------------------------------------------------------
            if self.freeze_beta is None:
                for t, i in enumerate(drove):
                    self.rls[i].update(psi[t][None, :], y[t : t + 1])
            self.last_y[drove] = y
            self._have_prev = True

        # ---- roll the occupancy forecast forward -------------------------------
        # Same accumulator shape as Ant's X: rho*X + (1-rho)*(what just happened).
        # On Ant rho is known env structure; here it is a declared FORECAST
        # smoother and therefore an ablation knob, not a physical constant.
        self.X_ema = self.rho * self.X_ema + (1.0 - self.rho) * x_true

        switch = float("nan")
        if self._have_prev and drove.size:
            prev = self.last_action[drove]
            m = prev >= 0
            if m.any():
                switch = float(np.mean(act_of[drove][m] != prev[m]))
        self.last_action[drove] = act_of[drove]

        counts = np.bincount(act_of[drove], minlength=self.K) if drove.size else None
        herd = herd_index(counts) if counts is not None else float("nan")

        self._write_row(
            n_peer_valid, drove.size, x_true, fit_r2, pred_r2, fit_mae, pred_mae,
            cond_psi, np.nanmean(tt_of), tt_hdv, reward_mean, excess_mean,
            clip_frac, switch, herd,
        )
        self._run_gates(fit_r2, x_true, cond_psi)

    # ------------------------------------------------------------------ logging
    def _write_row(self, n_peer_valid, n_drove, x_true, fit_r2, pred_r2, fit_mae,
                   pred_mae, cond_psi, tt_cav, tt_hdv, reward, excess_mean,
                   clip_frac, switch, herd):
        B = np.array([self._beta(i) for i in range(self.n_av)])
        seen = self._trust_seen
        tp = float(np.mean(self._trust_log[seen, 0])) if seen.any() else float("nan")
        ta = float(np.mean(self._trust_log[seen, 1])) if seen.any() else float("nan")
        tsd = (float(self._trust_log[seen, 1].max() - self._trust_log[seen, 1].min())
               if seen.any() else float("nan"))
        sh = float(np.mean(self._trust_log[seen, 2])) if seen.any() else float("nan")

        # --- the "is it winning" block ------------------------------------------
        cav_adv = (float(tt_hdv) / float(tt_cav)
                   if np.isfinite(tt_hdv) and np.isfinite(tt_cav) and tt_cav > 1e-9
                   else float("nan"))
        if np.isfinite(tt_cav):
            self._roll_cav.append(float(tt_cav))
        if np.isfinite(cav_adv):
            self._roll_adv.append(cav_adv)
        tt_cav_roll = float(np.mean(self._roll_cav)) if self._roll_cav else float("nan")
        cav_adv_roll = float(np.mean(self._roll_adv)) if self._roll_adv else float("nan")

        xm = [float(np.mean(np.abs(x_true[m]))) for m in range(self.r)]
        row = [
            self._ep, self._phase, int(n_drove), self.n_peer, int(n_peer_valid),
            _r(B[:, 0].mean(), 6), _r(B[:, 1].mean(), 6), _r(B[:, 2].mean(), 6),
            _r(B[:, 3].mean(), 6), _r(B[:, 1:].std(0).mean(), 6),
            _r(np.mean([e.innov for e in self.rls]), 6),
            _r(float(np.mean(self.conf)), 5),
            int(np.sum([e.n_updates for e in self.rls])),
            _r(fit_r2, 5), _r(pred_r2, 5), _r(fit_mae, 6), _r(pred_mae, 6),
            _r(cond_psi, 2),
            _r(xm[0], 6), _r(xm[1], 6), _r(xm[2], 6),
            _r(float(np.std(x_true)), 6),
            _r(tp, 5), _r(ta, 5), _r(tsd, 5), _r(sh, 5),
            _r(tt_cav, 4), _r(tt_hdv, 4), _r(cav_adv, 5),
            _r(tt_cav_roll, 4), _r(cav_adv_roll, 5),
            _r(reward, 5), _r(excess_mean, 5),
            _r(clip_frac, 5), _r(switch, 5), _r(herd, 5),
            _r(time.time() - self._t0, 3),
        ]
        if self._dbg_w is not None:
            self._dbg_w.writerow(row)
            self._dbg.flush()

        pe = int(self.cfg.get("print_every", 25))
        if pe > 0 and self._ep % pe == 0:
            # The verdict marker is URB's OWN winrate criterion: a run is "won" when
            # the CAV fleet is on average faster than the human drivers it replaced.
            # Smoothed over roll_episodes, because a single day says nothing.
            if not np.isfinite(cav_adv_roll):
                mark = "  ?  "
            elif cav_adv_roll > 1.0:
                mark = " WIN "
            else:
                mark = " lose"
            print(
                f"[PACT-1 ep {self._ep:5d} {self._phase:5s}] "
                f"fit_r2={_f(fit_r2)} pred_r2={_f(pred_r2)} cond={_f(cond_psi, 1)} | "
                f"beta=[{_f(B[:,0].mean(),3)} {_f(B[:,1].mean(),3)} "
                f"{_f(B[:,2].mean(),3)} {_f(B[:,3].mean(),3)}] conf={_f(np.mean(self.conf),3)} | "
                f"trust pol={_f(tp,3)} app={_f(ta,3)} | "
                f"tt_cav={_f(tt_cav,3)} roll{self.roll_n}={_f(tt_cav_roll,3)} "
                f"vs hdv {_f(tt_hdv,3)} adv={_f(cav_adv_roll,3)}[{mark}] | "
                f"switch={_f(switch,3)} herd={_f(herd,3)}",
                flush=True,
            )

    # ------------------------------------------------------------------ gates
    def _run_gates(self, fit_r2, x_true, cond_psi):
        ep = self._ep

        # GATE 5 -- liveness. An inert waveform means the whole method is a no-op,
        # and a return that looks fine is the failure nobody investigates.
        if not self._gate_fired["live"] and ep >= self.gate_live_after:
            self._gate_fired["live"] = True
            amp, sd = float(np.mean(np.abs(x_true))), float(np.std(x_true))
            live = amp > 1e-9 and sd > 1e-9
            print(
                f"[PACT-1][GATE 5 liveness] {'PASS' if live else 'FAIL'} at ep {ep}: "
                f"mean|x|={amp:.6f} std(x)={sd:.6f}",
                flush=True,
            )
            if not live:
                msg = (
                    "[PACT-1][GATE FAIL] the peer-load waveform is INERT: no peer is "
                    "loading anyone. Causes, in order of likelihood: the co-presence "
                    "matrix is empty (check start_time spread and dur_factor), "
                    "peer_scope excluded everyone, or the route set has no overlap."
                )
                if self.gate_abort:
                    raise AssertionError(msg)
                print(msg + "  (gate_abort=false)", flush=True)

        # GATE 6 -- does the reduction actually hold in this city? THE credit saver.
        if np.isfinite(fit_r2):
            self._fit_hist.append(float(fit_r2))
        if (not self._gate_fired["fit"] and ep >= self.gate_after
                and len(self._fit_hist) >= self.gate_window):
            self._gate_fired["fit"] = True
            recent = float(np.mean(self._fit_hist[-self.gate_window :]))
            ok = recent >= self.gate_min_fit_r2
            print(
                f"[PACT-1][GATE 6 reduction] {'PASS' if ok else 'FAIL'} at ep {ep}: "
                f"fit_r2 over the last {self.gate_window} episodes = {recent:.4f} "
                f"(need >= {self.gate_min_fit_r2})",
                flush=True,
            )
            if not ok:
                msg = (
                    f"[PACT-1][GATE FAIL] mean fit_r2={recent:.4f} < "
                    f"{self.gate_min_fit_r2} after {ep} episodes.\n"
                    "        The linear road-class model does not explain this "
                    "network's delays, so the reduction PACT-1 rests on does not "
                    "hold here and no amount of trust will help.\n"
                    "        Before spending more: try peer_scope='all', a larger "
                    "dur_factor (wider co-presence), or a different basis. STOPPING."
                )
                if self.gate_abort:
                    raise AssertionError(msg)
                print(msg + "  (gate_abort=false: continuing anyway)", flush=True)

        # GATE 7 -- conditioning. Report-only by design (guide III.6): a degenerate
        # regressor still PREDICTS fine, it just cannot DECOMPOSE theta. Claim
        # accordingly rather than aborting.
        if (not self._gate_fired["cond"] and ep >= self.gate_after
                and np.isfinite(cond_psi)):
            self._gate_fired["cond"] = True
            if cond_psi > self.gate_cond_warn:
                print(
                    f"[PACT-1][GATE 7 conditioning] WARN at ep {ep}: "
                    f"cond(E[psi psi^T])={cond_psi:.3e} > {self.gate_cond_warn:.1e}.\n"
                    "        beta*.psi remains predictable but the per-class SPLIT is "
                    "not identifiable. Report prediction, do NOT claim to have "
                    "decomposed the road-class sensitivities.",
                    flush=True,
                )
            else:
                print(
                    f"[PACT-1][GATE 7 conditioning] PASS at ep {ep}: "
                    f"cond={cond_psi:.3e} -- the split is identifiable.",
                    flush=True,
                )

    def close(self):
        if self._dbg is not None:
            try:
                self._dbg.close()
            except Exception:
                pass
            self._dbg = self._dbg_w = None


# ==========================================================================
#  small helpers
# ==========================================================================
def _score(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return float("nan"), float("nan")
    y, p = y[m], p[m]
    sse = float(np.sum((y - p) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 1e-12 else float("nan")
    return r2, float(np.mean(np.abs(y - p)))


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

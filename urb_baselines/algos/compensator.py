"""The shared decision rule for the two non-learning compensator arms (B10).

``paper/BASELINES.md`` B10 -- "Classical control: the non-learning compensators
(must, invertible cell)":

  * an **extended-state / disturbance observer** feedforward: "estimate the
    lumped disturbance from the agent's own residual with a first-order observer
    and cancel it next step. No peer information, no basis, one gain."
  * an **unstructured RLS on raw peer actions**: "regress the residual on the
    N-1 broadcast actions directly, no classes ... isolates the value of the
    declared basis (r parameters vs N-1)."

Both are the SAME controller with a different estimator, which is the whole point
of running them side by side. That controller lives here so that the two arms
differ in exactly one object.

THE CONTROLLER
--------------------------------------------------------------------------------
Linear ADRC's structure is: estimate the total disturbance, subtract it, and let
a nominal controller act on the cleaned plant. On URB the "plant" for traveller
``i`` choosing route ``k`` is

    travel_time_{i,k}(t) = freeflow_{i,k} + excess_{i,k}(t)

with ``freeflow`` known exactly (it is published by the environment) and
``excess`` the lumped disturbance -- congestion, weather, peers, everything. The
estimator produces a one-step-ahead ``excess_hat_{i,k}(t+1)``; the nominal
controller is then the obvious one,

    k* = argmin_k [ freeflow_{i,k} + excess_hat_{i,k} ]

i.e. drive the route that is predicted to be quickest. No reward, no value
function, no gradient: these arms never call an optimiser.

THE ONE ADAPTATION, AND WHY IT IS FORCED
--------------------------------------------------------------------------------
ADRC is a control law and has no exploration, which is fine for a continuous
plant that is always excited. On a discrete choice the observer for a route that
is never driven is never updated, so a controller that starts at the
free-flow-fastest route would stay there for four thousand days and the arm would
measure nothing. Exploration is therefore supplied by the HOST's own schedule --
the epsilon-greedy of ``scripts/iql.py``, with its own ``eps_init``,
``eps_decay`` and ``eps_min`` -- and nothing else is borrowed from the learner.
The estimators themselves are untouched, use no reward, and (for the ESO arm) no
peer information at all.

WHY "THE PREVIOUS DAY'S" PEER ACTIONS
--------------------------------------------------------------------------------
B10 describes the RLS arm as regressing on "the N-1 BROADCAST actions", because
in the invertible cell the peers publish their actions before the step. Stock URB
has no broadcast channel, so the regressor is the previous day's realised peer
action vector -- persistence. That is a property of the host, not a concession by
this arm: any arm on URB that wants tomorrow's peer load has to forecast it.
"""

import numpy as np

from urb_baselines.host import BaselineAlgorithm

__all__ = ["ExcessPredictorAlgorithm"]


class ExcessPredictorAlgorithm(BaselineAlgorithm):
    """argmin over ``freeflow + excess_hat``, with epsilon-greedy exploration.

    Subclasses implement the estimator:

        predict(agent_id) -> np.ndarray (K,)   one-step-ahead excess per route
        observe(agent_id, action, excess, info)  the day's realised residual
        estimator_banner() -> list[(k, v)]
        estimator_diagnostics() -> dict
    """

    needs_records = True

    def __init__(self, ctx):
        super(ExcessPredictorAlgorithm, self).__init__(ctx)
        P = ctx.params
        self.eps = float(P.get("eps_init", 1.0))
        self.eps_decay = float(P.get("eps_decay", 0.9985))
        self.eps_min = float(P.get("eps_min", 0.0))
        self.rng = np.random.RandomState(int(ctx.env_seed) * 31 + 7)

        # free-flow time per (agent, route). NaN marks a masked route slot; those
        # are never chosen and never fitted.
        ff = ctx.free_flow
        self.ff = np.full((self.n_av, self.n_actions), np.nan, dtype=np.float64)
        missing = []
        for i, a in enumerate(self.av_ids):
            od = tuple(ctx.agent_table[a]["od"])
            row = ff.get(od)
            if row is None:
                missing.append(od)
                continue
            n = min(len(row), self.n_actions)
            self.ff[i, :n] = np.asarray(row, dtype=np.float64)[:n]
        if missing:
            raise KeyError(
                f"[{self.name}] no free-flow row for OD pairs {sorted(set(missing))}. "
                f"The environment published {len(ff)} OD pairs; this arm's whole "
                f"decision rule is freeflow + excess_hat, so it cannot proceed "
                f"with a missing baseline.")
        self.valid = np.isfinite(self.ff)
        if not self.valid.any(axis=1).all():
            raise ValueError(f"[{self.name}] some agent has no valid route.")

        self._chosen = {}
        self.n_days = 0
        self.n_updates = 0
        self.n_missing_records = 0
        self.resid_abs = 0.0
        self.pred_err = []

    # ------------------------------------------------------------------ hooks
    def begin_episode(self, day, phase):
        self._chosen = {}

    def act(self, agent_id, obs):
        i = self.slot_of[agent_id]
        valid = np.flatnonzero(self.valid[i])
        if (not self.deterministic) and self.rng.rand() < self.eps:
            a = int(self.rng.choice(valid))
        else:
            pred = np.asarray(self.predict(agent_id), dtype=np.float64)
            total = self.ff[i] + pred
            total[~self.valid[i]] = np.inf
            a = int(np.argmin(total))
        self._chosen[agent_id] = a
        return a

    def end_episode(self, day, phase, info):
        """Feed the day's realised excess to the estimator.

        ``info.av_records`` is ``{agent_id: (action, travel_time)}`` from
        RouteRL's own per-day file, so the action fitted is the one the simulator
        executed rather than the one this arm believes it asked for.
        """
        self.n_days += 1
        if not info.av_records:
            self.n_missing_records += 1
            return
        self.pre_observe(info)
        tot = 0.0
        n = 0
        for a, (act, tt) in info.av_records.items():
            i = self.slot_of.get(a)
            if i is None or not (0 <= act < self.n_actions):
                continue
            base = self.ff[i, act]
            if not np.isfinite(base) or not np.isfinite(tt):
                continue
            excess = float(tt) - float(base)
            if n < 64:                      # one-step-ahead prediction error
                pred = float(np.asarray(self.predict(a))[act])
                self.pred_err.append(abs(excess - pred))
            self.observe(a, int(act), excess, info)
            tot += abs(excess)
            n += 1
        if n:
            self.resid_abs = tot / n
            self.n_updates += 1
        if phase == "train":
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def begin_test(self):
        self.deterministic = True

    # ------------------------------------------------------------------ api
    def pre_observe(self, info):
        """Called once per day before the per-agent ``observe`` calls."""

    def predict(self, agent_id):
        raise NotImplementedError

    def observe(self, agent_id, action, excess, info):
        raise NotImplementedError

    def estimator_banner(self):
        return []

    def estimator_diagnostics(self):
        return {}

    # ------------------------------------------------------------------ report
    def diagnostics(self):
        d = {"eps": float(self.eps), "|resid|": float(self.resid_abs)}
        if self.pred_err:
            d["|err|"] = float(np.mean(self.pred_err[-2000:]))
        d.update(self.estimator_diagnostics())
        return d

    def loss_records(self):
        """These arms have no loss. The one-step-ahead prediction error is what
        they are judged on as ESTIMATORS, so that is what is written out."""
        return [{"iteration": i, "agent_id": "all", "loss": v}
                for i, v in enumerate(self.pred_err, start=1)]

    def close(self):
        print("\n" + "=" * 74)
        print(f"[{self.name}] COMPENSATOR REPORT")
        print(f"  days                    {self.n_days}")
        print(f"  days with records       {self.n_days - self.n_missing_records}")
        print(f"  estimator updates       {self.n_updates}")
        print(f"  mean |excess| (last)    {self.resid_abs:.2f} s")
        if self.pred_err:
            k = min(len(self.pred_err), 2000)
            print(f"  mean |prediction err|   {np.mean(self.pred_err[-k:]):.2f} s "
                  f"(last {k} samples)")
        if self.n_updates == 0:
            print("  *** THE ESTIMATOR NEVER RECEIVED A ROW. This arm ran as pure")
            print("  *** free-flow shortest path plus epsilon-greedy noise and")
            print("  *** must NOT be reported as a compensator.")
        for k, v in self.estimator_banner():
            print(f"  {k:<22} {v}")
        print("=" * 74 + "\n", flush=True)

"""Unstructured RLS on the raw peer actions (B10) on URB.

``paper/BASELINES.md`` B10:

    "Per-agent unstructured RLS on raw peer actions (regress the residual on the
    N-1 broadcast actions directly, no classes): isolates the value of the
    *declared basis* (r parameters vs N-1). Prediction: fails at N = 22/38
    (under-excited, ill-conditioned)."

and the objection table: "Why not learn the coupling instead of declaring it" ->
"unstructured RLS ... and the learned-basis arm".

WHAT IT IS
--------------------------------------------------------------------------------
Exactly the same controller as the ESO arm (``algos/compensator.py``:
``argmin_k [freeflow_k + excess_hat_k]``, host epsilon-greedy), with the
observer replaced by exponentially-weighted recursive least squares on the RAW
peer actions -- no road classes, no incidence matrix, no declared operator:

    excess_hat_{i,k}(t) = phi(t-1)^T theta_{i,k}
    phi(t-1) = [ one-hot(a_j(t-1)) for every broadcasting peer j != i , 1 ]

    RLS:  e     = y - phi^T theta
          g     = P phi / (lambda + phi^T P phi)
          theta = theta + g e
          P     = (P - g phi^T P) / lambda

This is the textbook exponentially-weighted RLS: a gain vector from the inverse
covariance, a rank-one covariance downdate, and a forgetting factor. It is the
SAME estimator family PACT-1 uses; the only difference is what the regressor is,
which is precisely what the arm is supposed to isolate.

WHY ONE FILTER PER (AGENT, OWN-ROUTE)
--------------------------------------------------------------------------------
A single regression of the residual on the peers' actions has no dependence on
the agent's own choice and therefore cannot rank routes -- it would be inert. The
agent's own action is a categorical with no order, so the honest unstructured
encoding is a separate parameter vector per own-route, fitted from the days that
route was driven. This is exactly the "r parameters vs N-1" accounting B10 asks
for, with r replaced by K x (N-1) x K.

WHAT IS BEING MEASURED
--------------------------------------------------------------------------------
Parameters per agent: ``K * (K * n_peers + 1)``. On saint_arnoult with the
default ``peer_scope: fleet`` that is 4 * (4 * 87 + 1) = 1396 parameters fitted
from 4000 daily scalars, only a quarter of which touch any given own-route
filter. The banner prints that count and the running condition number of P, so
"under-excited, ill-conditioned" is a measurement in the log rather than a claim
in the text.

PEER SCOPE
--------------------------------------------------------------------------------
``fleet`` (default) is the BROADCAST set: only machine agents publish an action,
so it is the set B10's phrase "the N-1 broadcast actions" names. ``all`` adds the
human travellers, which makes the regressor complete but squares the memory --
the banner prints the cost before it is paid.
"""

import numpy as np

from urb_baselines.algos.compensator import ExcessPredictorAlgorithm

__all__ = ["UnstructuredRLS", "add_args"]


class UnstructuredRLS(ExcessPredictorAlgorithm):
    name = "urls"

    def __init__(self, ctx):
        super(UnstructuredRLS, self).__init__(ctx)
        cfg = self.cfg
        self.forget = float(cfg.get("forget", 0.999))
        self.p0 = float(cfg.get("p0", 10.0))
        self.scope = str(getattr(ctx.args, "peer_scope", None)
                         or cfg.get("peer_scope", "fleet")).lower()
        if self.scope not in ("fleet", "all"):
            raise ValueError("[urls] peer_scope must be 'fleet' or 'all'")

        if self.scope == "fleet":
            peers = list(self.av_ids)
        else:
            peers = sorted(ctx.agent_table.keys(), key=lambda x: int(x))
        self.peer_ids = peers
        self.peer_index = {p: j for j, p in enumerate(peers)}
        self.n_peer = len(peers)

        # phi = [one-hot(a_j) for every peer, including this agent's own slot
        # which is masked to zero, ... , 1]. Keeping the agent's own slot in the
        # layout (and zeroing it) means every agent shares one index map, which
        # is what makes the day's feature vector a single O(N) build.
        self.d = self.n_peer * self.n_actions + 1
        nbytes = (self.n_av * self.n_actions * self.d * self.d * 4)
        max_mb = float(cfg.get("max_p_mb", 2048.0))
        if nbytes / 1e6 > max_mb:
            raise MemoryError(
                f"[urls] the covariance stack would need {nbytes / 1e6:.0f} MB "
                f"(> max_p_mb={max_mb:.0f}). d={self.d} features x "
                f"{self.n_av} agents x {self.n_actions} own-routes.\n"
                f"        This is the arm's own point -- an unstructured "
                f"regressor on N-1 raw actions does not scale -- but if you want "
                f"to run it anyway, raise max_p_mb or set peer_scope='fleet'.")

        self.theta = np.zeros((self.n_av, self.n_actions, self.d),
                              dtype=np.float32)
        self.P = np.zeros((self.n_av, self.n_actions, self.d, self.d),
                          dtype=np.float32)
        eye = np.eye(self.d, dtype=np.float32) * self.p0
        self.P[:] = eye
        self.phi_prev = np.zeros(self.d, dtype=np.float32)
        self.phi_prev[-1] = 1.0
        self.have_phi = False
        self._phi_today = None
        self.n_fits = 0
        self.last_innov = 0.0
        self.trace_P = float(self.p0 * self.d)

        self.banner("UNSTRUCTURED RLS ON RAW PEER ACTIONS (B10)", [
            ("controller", "identical to the ESO arm "
                           "(algos/compensator.py)"),
            ("regressor", f"one-hot of every peer's PREVIOUS-day action + "
                          f"intercept = {self.d} features"),
            ("peer scope", f"{self.scope}  ({self.n_peer} peers"
                           + (" -- the broadcast set)" if self.scope == "fleet"
                              else " -- machines AND humans)")),
            ("filters", f"{self.n_av} agents x {self.n_actions} own-routes"),
            ("parameters/agent", f"{self.n_actions * self.d}"),
            ("covariance memory", f"{nbytes / 1e6:.0f} MB"),
            ("forgetting / P0", f"{self.forget} / {self.p0}"),
            ("no classes", "no incidence matrix, no road classes, no declared "
                           "operator -- that is the comparison"),
        ])

    # ------------------------------------------------------------------ estimator
    def pre_observe(self, info):
        """Build today's peer feature vector; it becomes tomorrow's regressor.

        ``info.peer_acts`` is every traveller's EXECUTED action from RouteRL's
        own per-day file, so what is regressed on is what was driven.
        """
        phi = np.zeros(self.d, dtype=np.float32)
        phi[-1] = 1.0
        src = info.peer_acts or info.actions
        for p, a in src.items():
            j = self.peer_index.get(p)
            if j is None or not (0 <= int(a) < self.n_actions):
                continue
            phi[j * self.n_actions + int(a)] = 1.0
        self._phi_today = phi

    def predict(self, agent_id):
        i = self.slot_of[agent_id]
        if not self.have_phi:
            return np.zeros(self.n_actions, dtype=np.float64)
        phi = self.phi_prev.copy()
        self._mask_self(phi, agent_id)
        return self.theta[i] @ phi           # (K, d) @ (d,) -> (K,)

    def _mask_self(self, phi, agent_id):
        """Zero the agent's own block: B10's regressor is the N-1 OTHERS."""
        j = self.peer_index.get(agent_id)
        if j is not None:
            phi[j * self.n_actions:(j + 1) * self.n_actions] = 0.0

    def observe(self, agent_id, action, excess, info):
        if not self.have_phi:
            return
        i = self.slot_of[agent_id]
        phi = self.phi_prev.copy()
        self._mask_self(phi, agent_id)

        P = self.P[i, action]
        th = self.theta[i, action]
        Pphi = P @ phi
        denom = self.forget + float(phi @ Pphi)
        if denom <= 1e-12:
            return
        g = Pphi / denom
        e = float(excess) - float(th @ phi)
        th += g * e
        # P <- (P - g phi^T P) / lambda, symmetrised so float32 rounding cannot
        # drift the covariance out of the PSD cone over four thousand updates.
        P -= np.outer(g, Pphi)
        P /= self.forget
        P += P.T.copy()
        P *= 0.5
        self.last_innov = abs(e)
        self.n_fits += 1

    def end_episode(self, day, phase, info):
        super(UnstructuredRLS, self).end_episode(day, phase, info)
        # today's peer actions become tomorrow's regressor (persistence: stock
        # URB has no broadcast channel -- see algos/compensator.py). Cleared
        # after use so a day with no records rolls nothing forward rather than
        # silently re-using a stale vector.
        if self._phi_today is not None:
            self.phi_prev = self._phi_today
            self.have_phi = True
            self._phi_today = None
        if self.n_fits and self.n_fits % 500 == 0:
            self.trace_P = float(np.trace(self.P[0, 0]))

    def estimator_diagnostics(self):
        return {"fits": self.n_fits, "trP": self.trace_P,
                "|th|": float(np.abs(self.theta).mean())}

    def estimator_banner(self):
        p0 = self.P[0, 0].astype(np.float64)
        try:
            ev = np.linalg.eigvalsh(p0)
            cond = float(ev.max() / max(ev.min(), 1e-30))
        except np.linalg.LinAlgError:
            cond = float("nan")
        return [("RLS updates", self.n_fits),
                ("features", self.d),
                ("cond(P) agent0 route0", f"{cond:.3e}"),
                ("trace(P) agent0 route0", f"{np.trace(p0):.3e}"),
                ("mean |theta|", f"{np.abs(self.theta).mean():.4f}"),
                ("note", "a condition number that never falls is the "
                         "under-excitation B10 predicts")]


def add_args(parser):
    parser.add_argument('--peer-scope', type=str, default=None,
                        choices=["fleet", "all"],
                        help="fleet (default): the machine agents, i.e. the "
                             "broadcast set. all: humans too -- complete, but "
                             "the covariance memory grows as the square.")

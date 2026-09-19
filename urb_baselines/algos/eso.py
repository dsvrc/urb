"""ESO / DOB -- the linear-ADRC disturbance observer arm (B10) on URB.

    Han, "From PID to Active Disturbance Rejection Control", IEEE TIE 2009;
    Gao, "Scaling and bandwidth-parameterization based controller tuning",
    ACC 2003 (the beta_1 = 2 w_0, beta_2 = w_0^2 observer parameterisation).
    Reference implementations: https://pyadrc.readthedocs.io/en/latest/ ,
    https://github.com/MRGilak/Active-Disturbance-Rejection-Controller ,
    ADRC toolbox https://arxiv.org/pdf/2112.01614

``paper/BASELINES.md`` B10 calls this "the single most important non-RL baseline
and it is cheap":

    "Per-agent extended-state / disturbance observer feedforward (ESO / DOB, i.e.
    linear ADRC): estimate the lumped disturbance from the agent's own residual
    with a first-order observer and cancel it next step. No peer information, no
    basis, one gain."

and predicts: "it cancels the *slow* component and lags the fast one ... whereas
PACT's channels are computed from the peers' broadcast actions *before* the
step."

WHAT THE OBSERVER IS
--------------------------------------------------------------------------------
The plant seen by traveller ``i`` on route ``k`` is ``tt = freeflow + f``, where
``f`` is the total disturbance and there is no additive control channel -- the
only decision is WHICH plant to drive. The extended state is therefore ``f``
itself and the ESO reduces to a linear observer driven by the innovation
``y - z1``:

  ``--arm eso1``  (default; B10's "first-order observer ... one gain")
        z1 <- z1 + beta * (y - z1)
        prediction: z1

  ``--arm eso2``  the standard linear ADRC pair, which also tracks the
        disturbance's RATE, so it extrapolates instead of lagging:
        z1 <- z1 + z2 + beta1 * (y - z1)
        z2 <- z2      + beta2 * (y - z1)
        prediction: z1 + z2
        with Gao's bandwidth parameterisation beta1 = 2*w0*h, beta2 = (w0*h)^2,
        h = 1 day.

ONE OBSERVER PER (AGENT, ROUTE), AND WHY IT HAS TO BE
--------------------------------------------------------------------------------
A single lumped disturbance per agent would be added to every route's free-flow
time equally and could not change the argmin -- the arm would be provably inert,
which is the failure mode this repo refuses to ship. The extended state is
therefore per route, updated only on the days that route is actually driven. It
still uses no peer information, no basis and one gain: exactly B10's
specification, instantiated on a discrete choice.

The observer sees only this traveller's own realised-minus-free-flow residual.
It never sees a peer action, a reward, or the driver.
"""

import numpy as np

from urb_baselines.algos.compensator import ExcessPredictorAlgorithm

__all__ = ["ESO", "add_args"]


class ESO(ExcessPredictorAlgorithm):
    name = "eso"
    ARMS = ("eso1", "eso2")

    def __init__(self, ctx):
        super(ESO, self).__init__(ctx)
        arm = getattr(ctx.args, "arm", None) or str(self.cfg.get("arm", "eso1"))
        if arm not in self.ARMS:
            raise ValueError(f"[eso] --arm must be one of {self.ARMS}, got {arm!r}")
        self.arm = arm
        self.beta = float(self.cfg.get("beta", 0.5))
        w0 = float(self.cfg.get("omega", 0.5))          # rad/day, h = 1 day
        self.beta1, self.beta2 = 2.0 * w0, w0 * w0
        self.z1 = np.zeros((self.n_av, self.n_actions), dtype=np.float64)
        self.z2 = np.zeros((self.n_av, self.n_actions), dtype=np.float64)
        self.n_obs_updates = 0

        self.banner("ESO / DOB -- linear ADRC disturbance observer (B10)", [
            ("arm", self.arm + ("   first-order, one gain" if arm == "eso1"
                                else "   second-order (tracks the rate)")),
            ("gain", f"beta={self.beta}" if arm == "eso1"
                     else f"omega={w0} -> beta1={self.beta1}, beta2={self.beta2}"),
            ("extended state", f"per (agent, route): {self.n_av} x "
                               f"{self.n_actions}"),
            ("inputs", "the agent's OWN realised minus free-flow residual. "
                       "No peers, no basis, no reward."),
            ("control law", "argmin_k [ freeflow_k + z_k ]"),
            ("exploration", f"host epsilon-greedy, eps0={self.eps}, "
                            f"decay={self.eps_decay}, min={self.eps_min}"),
        ])

    # ------------------------------------------------------------------ estimator
    def predict(self, agent_id):
        i = self.slot_of[agent_id]
        if self.arm == "eso1":
            return self.z1[i]
        return self.z1[i] + self.z2[i]          # one-step-ahead extrapolation

    def observe(self, agent_id, action, excess, info):
        i = self.slot_of[agent_id]
        innov = float(excess) - self.z1[i, action]
        if self.arm == "eso1":
            self.z1[i, action] += self.beta * innov
        else:
            self.z1[i, action] += self.z2[i, action] + self.beta1 * innov
            self.z2[i, action] += self.beta2 * innov
        self.n_obs_updates += 1

    def estimator_diagnostics(self):
        return {"|z1|": float(np.abs(self.z1).mean())}

    def estimator_banner(self):
        return [("observer updates", self.n_obs_updates),
                ("mean |z1|", f"{np.abs(self.z1).mean():.2f} s"),
                ("states ever updated",
                 f"{int((self.z1 != 0).sum())}/{self.z1.size}")]


def add_args(parser):
    parser.add_argument('--arm', type=str, default=None, choices=list(ESO.ARMS),
                        help="eso1 (default, B10's first-order one-gain observer) "
                             "or eso2 (the standard linear-ADRC pair, which also "
                             "tracks the disturbance rate)")

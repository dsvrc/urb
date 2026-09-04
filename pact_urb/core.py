"""PACT core -- pure numpy. No torch, no routerl, no SUMO.

Implements `PACT_PIPELINE_SPEC` components 4, 5 and 7. Every `[PAID]` note below
marks a design choice that a run failing taught, and the numbers are that spec's
measured values.
"""

import numpy as np

__all__ = ["AgentRLS", "NullTracker", "trust_gate", "analytic_feedforward",
           "PedestalEMA"]


# ==========================================================================
#  Component 4 -- the estimator
# ==========================================================================
class AgentRLS(object):
    """Per-agent recursive least squares with forgetting AND a windup bound.

    `mu = 0.9995` is the spec's measured optimum, not an assumption:

        mu      0.990  0.995  0.997  0.999  0.9995  0.9999
        corr    0.556  0.500  0.481  0.525  0.555   0.572
        dg_std  0.395  0.317  0.231  0.093  0.055   0.030

    Aggressive forgetting buys nothing and injects noise straight into the
    coefficient the inverse divides by. Re-measure per environment; the optimum
    follows the drift rate.

    ** THE COVARIANCE BOUND IS NOT OPTIONAL. ** [PAID, §5.2]

    > Forgetting inflates unexcited directions by 1/mu every update WITHOUT BOUND.
    > As the policy converges excitation dies and the estimator runs away: measured
    > se(own_gain) across quarters 29 -> 4.9e3 -> 8.4e5 -> 1.4e8. Return tracked it
    > exactly -- the method led in Q1-Q3 and lost 28.8 in Q4.

    Rescaling preserves RELATIVE uncertainty while stopping the absolute scale from
    diverging, so the estimate survives intact.
    """

    def __init__(self, dim, mu=0.9995, p0=10.0, p_max_mult=10.0):
        self.dim = int(dim)
        self.mu = float(mu)
        self.p0 = float(p0)
        self.p_max = float(p_max_mult) * self.p0 * self.dim
        self.P = np.eye(self.dim) * self.p0
        self.beta = np.zeros(self.dim)
        self.n_updates = 0
        self.n_clamped = 0
        self.innov = 0.0

    def update(self, psi, y):
        """One row. `psi` (dim,), `y` scalar."""
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        if psi.shape[0] != self.dim or not np.all(np.isfinite(psi)):
            return self.beta
        if not np.isfinite(y):
            return self.beta
        if float(psi @ psi) < 1e-18:
            return self.beta                      # dead row: no info, no windup

        Ppsi = self.P @ psi
        denom = self.mu + float(psi @ Ppsi)
        if denom < 1e-12:
            return self.beta
        k = Ppsi / denom
        e = float(y) - float(psi @ self.beta)
        self.beta = self.beta + k * e
        self.P = (self.P - np.outer(k, Ppsi)) / self.mu
        self.P = 0.5 * (self.P + self.P.T)        # symmetric; the gates read P

        # --- §5.2 the windup bound -----------------------------------------
        tr = float(np.trace(self.P))
        if not np.isfinite(tr):
            self.P = np.eye(self.dim) * self.p0   # diverged: restart the prior
            self.n_clamped += 1
        elif tr > self.p_max:
            self.P *= self.p_max / tr
            self.n_clamped += 1

        self.innov = abs(e)
        self.n_updates += 1
        return self.beta

    def predict(self, psi):
        return float(np.asarray(psi, dtype=np.float64).reshape(-1) @ self.beta)

    def se(self, psi):
        """Standard error of the prediction along `psi` -- what §6.3's significance
        floor tests, and the only uncertainty the compensator actually uses."""
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        v = float(psi @ (self.P @ psi))
        return float(np.sqrt(max(v, 0.0)))


# ==========================================================================
#  Component 4.3 -- the null model is MANDATORY
# ==========================================================================
class NullTracker(object):
    """Windowed R^2 lift of the peer channels over an intercept+own null.

    §4.3 [PAID]:

    > A pooled R^2 of 0.9998 looked like a triumph until the intercept-only model
    > was scored too and came in at 0.656. The honest quantity was 0.344. Score
    > one-step-ahead (prior) predictions, not posterior fits, and skip a warmup
    > before accumulating -- a cold start otherwise dominates both sums and their
    > difference is noise.

    And §8.2: use a WINDOWED lift. A cumulative one is held down forever by early
    negatives, so the compensator never re-arms after a bad start.
    """

    def __init__(self, window=200, warmup=50):
        self.window = int(window)
        self.warmup = int(warmup)
        self.n_seen = 0
        self._y, self._full, self._null = [], [], []

    def add(self, y, pred_full, pred_null):
        self.n_seen += 1
        if self.n_seen <= self.warmup:
            return
        for buf, v in ((self._y, y), (self._full, pred_full),
                       (self._null, pred_null)):
            buf.append(float(v))
            if len(buf) > self.window:
                del buf[0]

    def fit_gain(self):
        """Fraction of the NULL model's error that the peer channels remove.

            fit_gain = 1 - SSE(full) / SSE(null)

        ** NOT a difference of R-squared. ** [PAID]

        The R2 form `R2(full) - R2(null)` divides both terms by the target's own
        variance, and the target here is a loading ratio that barely moves once the
        policy converges. Measured on the first URB NS runs: SST went small while
        both SSE stayed finite, so each R2 blew up hugely negative and their
        DIFFERENCE came out at 42, 156 and 870 -- values that are impossible for a
        quantity bounded near [-1, 1]. The admissibility gate then read 0.975 and
        admitted on nonsense for an entire 4000-day run.

        Normalising by the null model's error instead is bounded above by 1,
        negative exactly when the peer channels make prediction worse, and stable
        however little the target varies -- because SSE(null) only approaches zero
        when the null is already perfect, in which case there is genuinely nothing
        left for the peer term to explain.
        """
        if len(self._y) < max(20, self.dim_min):
            return float("nan")
        y = np.asarray(self._y)
        sse_full = float(np.sum((y - np.asarray(self._full)) ** 2))
        sse_null = float(np.sum((y - np.asarray(self._null)) ** 2))
        scale = max(sse_null, 1e-12 * max(1.0, float(np.sum(y * y))))
        if sse_null <= 0.0 or scale <= 0.0:
            return float("nan")            # the null is already exact
        return float(np.clip(1.0 - sse_full / scale, -1.0, 1.0))

    dim_min = 20


# ==========================================================================
#  Component 7 -- the analytic driver feedforward
# ==========================================================================
def analytic_feedforward(excess, du_da, ff_gain=1.0, max_delta=np.inf):
    """The largest single performance term, and it is CLOSED FORM. §7.

        excess = u * (1 - g)        KNOWN: g is a function of observable time
        d_ff   = -ff_gain * excess / du_da

    No estimator, no gate, no confidence, nothing to converge. On POWER it took
    return from **237 -> 705** and accounted for **79%** of the correction.

    ** TWO HONESTY CONDITIONS, BOTH BINDING (§7): **

      1. This term is LOCAL. It needs no peer information, so it supports NO
         coordination claim. Log it separately from the peer term and report the
         split.
      2. The baseline must get it too. Any method with the domain model can
         compute `u*(1-g)`. If only the method's arm has it, the gap is
         INFORMATION, not mechanism -- hence the `ff` arm.
    """
    excess = np.asarray(excess, dtype=np.float64)
    du_da = np.asarray(du_da, dtype=np.float64)
    safe = np.where(np.abs(du_da) > 1e-12, du_da, np.nan)
    d = -float(ff_gain) * excess / safe
    return np.clip(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0),
                   -max_delta, max_delta)


# ==========================================================================
#  Component 6.4 -- compensate the fluctuation, not the pedestal
# ==========================================================================
class PedestalEMA(object):
    """Slow EMA of the predicted loading, subtracted before compensating. §6.4 [PAID]

    > Centring on the geometric reference is right for IDENTIFICATION. But a
    > trained policy does not act uniformly at random, so psi carries a large
    > constant offset -- measured mean [-0.50, -2.01] against temporal std
    > [0.30, 0.17], i.e. ~85% pedestal. Cancelling it burns the entire bounded
    > actuator budget on a constant and leaves the time-varying part -- the only
    > part the policy cannot anticipate -- uncorrected.

    The standing level is the policy's job; the fluctuation is the compensator's.
    `tau` is a declared constant and belongs in the ablation table.
    """

    def __init__(self, tau=400.0, n=1):
        self.tau = float(tau)
        self.level = np.zeros(int(n))
        self.ready = False

    def step(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not self.ready:
            self.level = x.copy()
            self.ready = True
        else:
            self.level += (x - self.level) / self.tau
        return x - self.level


# ==========================================================================
#  Component 8 -- the gates
# ==========================================================================
def trust_gate(fit_gain, n_updates, fit_floor=0.0, ready_updates=100,
               max_trust=0.5):
    """Separate WHETHER from HOW MUCH. §8.1 [PAID]

    > RLS returns the LEAST-SQUARES prediction, and for a least-squares predictor
    > the residual-minimising gain is exactly g* = 1, because the LS prediction IS
    > the conditional mean. T4 pulls g* below 1 and estimation noise pulls it
    > further -- to an INTERIOR optimum, not to nothing. Multiplying three
    > heuristic confidences produced applied_trust = 0.024 against fit_gain = 0.016:
    > ~40x below the theoretical gain. Switching to binary admissibility x a
    > calibrated constant took applied trust to 0.192 (12x).

    And §8.2: gate on the MEASURED lift, not on a covariance proxy. Covariance can
    look fine while the peer term carries no information at all.
    """
    admissible = (np.isfinite(fit_gain) and fit_gain > float(fit_floor)
                  and int(n_updates) >= int(ready_updates))
    return (float(max_trust) if admissible else 0.0), bool(admissible)

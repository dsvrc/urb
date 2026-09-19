"""Oracle-driver IPPO -- the information arm, not a published method.

``paper/BASELINES.md`` Tier 1 item 4 ("oracle-driver blind ... and URB if the
observation change is cheap") and the objection table row "Give the learner the
same information PACT has".

WHAT IT IS
--------------------------------------------------------------------------------
URB's own IPPO, unchanged in every other respect, with the exogenous driver
``A(day)`` appended to every observation. Nothing else is added: no estimator, no
channels, no constraint.

WHY IT EXISTS
--------------------------------------------------------------------------------
It is the control for two separate claims.

  * Against **PACT-1**: if simply knowing the weather were enough, PACT's
    identification would be decoration. This arm gives the policy the driver and
    nothing else, so the gap between it and PACT is the value of knowing how much
    the peers' load matters under that weather -- not of knowing the weather.
  * Against **LCPO** (``scripts/lcpo.py``): LCPO also receives ``A(day)``. This
    arm is that augmentation with LCPO's machinery removed, so the LCPO-minus-
    oracle gap is the local constraint rather than the context.

It is therefore deliberately NOT a citation. It is an upper bound on what an
observed driver alone can buy, run through the identical host.
"""

import numpy as np

from urb_baselines.algos.reference import PerAgentPPOAlgorithm

__all__ = ["OracleDriverIPPO"]


class OracleDriverIPPO(PerAgentPPOAlgorithm):
    name = "oracle_ippo"
    needs_context = True

    def __init__(self, ctx):
        self.context = ctx.context
        self.ctx_dim = self.context.dim
        self._ctx_now = np.zeros(self.ctx_dim, dtype=np.float32)
        super(OracleDriverIPPO, self).__init__(ctx)
        self.banner("ORACLE-DRIVER IPPO -- the information arm", [
            ("learner", "URB's IPPO, unchanged (scripts/ippo.py)"),
            ("observation", f"{self.obs_size} host + {self.ctx_dim} context "
                            f"{list(self.context.names)} = {self.input_size()}"),
            ("driver", "degenerate (constant 0)" if self.context.degenerate
                       else f"A(day), sigma={self.context.sigma}"),
            ("agents", f"{self.n_av} independent learners"),
        ])

    def input_size(self):
        return self.obs_size + self.ctx_dim

    def begin_episode(self, day, phase):
        self._ctx_now = np.asarray(self.context.observed(day), dtype=np.float32)

    def observation(self, agent_id, obs):
        return np.concatenate([np.asarray(obs, dtype=np.float32).reshape(-1),
                               self._ctx_now])

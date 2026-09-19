"""Domain randomisation over severity -- B9's "must, zero cost" robust arm.

    "resample sigma in [0, 3] per episode ... train the stock learner, evaluate
    at the committed sigma."   -- paper/BASELINES.md B9

    Wrapper pattern after RRLS (https://arxiv.org/html/2406.08406v1) and
    SafeRL-Lab/Robust-RL-Baselines: randomise the environment parameter, leave
    the learner alone.

WHAT IT IS
--------------------------------------------------------------------------------
URB's own IPPO -- byte for byte the learner of ``scripts/ippo.py``, told nothing
about severity -- trained under a severity parameter redrawn every single day,
then evaluated at the committed sigma with the randomisation detached.

WHAT IT TESTS
--------------------------------------------------------------------------------
B9's prediction: "robustness costs sigma=0 performance and still does not cancel;
the asymptote lies between blind and PACT." Domain randomisation buys a policy
that is good *on average* over the severity range; it cannot track A(t) within a
run, because the policy has no input that tells it which draw it is in. That is
the distinction between hedging against the drift and identifying it.

WHY IT IS NOT A FLAG ON ANOTHER ARM
--------------------------------------------------------------------------------
It changes the environment, not the algorithm, so it could have been a flag on
``ns_launch.py``. It is a separate script instead for the same reason every other
baseline is: an arm that exists only as a flag ends up sharing a config, a log
and an exp_id convention with the arm it was a flag on, and its result becomes
hard to quote. See ``urb_baselines/domain_random.py`` for why the certified
severity wrapper itself is left untouched.
"""

from urb_baselines import domain_random
from urb_baselines.algos.reference import PerAgentPPOAlgorithm

__all__ = ["DomainRandomisedIPPO", "add_args"]


class DomainRandomisedIPPO(PerAgentPPOAlgorithm):
    name = "dr_ippo"

    def __init__(self, ctx):
        super(DomainRandomisedIPPO, self).__init__(ctx)
        rng_spec = getattr(ctx.args, "dr_range", None) or self.cfg.get(
            "dr_range", [0.0, 3.0])
        if isinstance(rng_spec, str):
            rng_spec = [float(x) for x in rng_spec.split(",")]
        lo, hi = float(rng_spec[0]), float(rng_spec[1])
        self.committed = float(getattr(ctx.env, "_ns_sigma", 0.0))
        self.dr = domain_random.attach(
            ctx.env, lo, hi, seed=int(ctx.env_seed) * 7919 + 13,
            sigma_floor=float(self.cfg.get("sigma_floor", 1e-9)), verbose=True)
        self.banner("DOMAIN RANDOMISATION OVER SEVERITY (B9)", [
            ("learner", "URB's IPPO, unchanged (scripts/ippo.py)"),
            ("training sigma", f"~ U[{lo}, {hi}], redrawn every day"),
            ("evaluation sigma", f"{self.committed} (committed at launch)"),
            ("told sigma?", "NO -- the observation is the stock one"),
            ("agents", f"{self.n_av} independent learners"),
        ])

    def begin_test(self):
        # B9's second half: evaluate at the committed severity. Detaching rather
        # than pinning means sigma=0 recovers the stock task byte for byte.
        self.dr.fix(self.committed)
        super(DomainRandomisedIPPO, self).begin_test()

    def diagnostics(self):
        d = self.dr.draws
        return {"sigma": float(d[-1]) if d else float("nan"),
                "n_draws": len(d)}

    def close(self):
        self.dr.report()


def add_args(parser):
    parser.add_argument('--dr-range', type=str, default=None,
                        help="training severity range as LO,HI (default 0,3 -- "
                             "B9's range). The TEST phase always uses the sigma "
                             "the run was launched with.")

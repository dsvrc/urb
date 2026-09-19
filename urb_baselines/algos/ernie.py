"""ERNIE -- adversarially Regularized multiageNt reInforcement lEarning, on URB.

    Bukharin, Li, Yu, Zhang, Chen, Zuo, Zhang, Zhang, Zhao,
    "Robust Multi-Agent Reinforcement Learning via Adversarial Regularization:
    Theoretical Foundation and Stable Algorithms", NeurIPS 2023.
    https://arxiv.org/abs/2310.10810
    Code: https://github.com/abukharin3/ERNIE  (README's "simplest version";
          Algorithms/qcombo.py train_step / get_adv_reg_loss / unroll_perturb)

``paper/BASELINES.md`` B9 puts ERNIE on URB as the "chosen published robust
method (MARL-specific; claims robustness to changing transition dynamics)", with
the prediction that "robustness costs sigma=0 performance and still does not
cancel; the asymptote lies between blind and PACT."

THE METHOD, IN THE PAPER'S OWN EQUATIONS
--------------------------------------------------------------------------------
Eq. 4  R_pi(o_k; theta_k) = max_{||delta|| <= eps} D( pi(o_k + delta), pi(o_k) )
Eq. 5  min_theta  L(theta) + lambda * sum_n E[ R_pi(o_n; theta_n) ]
Eq. 6  the Stackelberg form: delta^K(o, theta) is the K-fold composition of the
       gradient-ascent operator, so the leader (theta) differentiates THROUGH the
       follower's response and the gradient gains the leader-follower term.

Section 5.1: "When optimizing stochastic policies (e.g., MAPPO), D can be taken
to be the KL divergence" -- so D is KL here. Appendices E.1-E.3: "In practice we
apply ERNIE to the individual Q-functions / the individual policies", i.e. the
regulariser is PER AGENT on that agent's own policy and its own observation,
which is exactly the shape of URB's IPPO.

Appendix F: "we use the l_2 norm to bound the attacks delta ... use a grid search
to find eps and lambda_pi." Both are therefore configuration, not constants, and
the values this repo runs are recorded in config/algo_config/ernie/config1.json.

ARMS
--------------------------------------------------------------------------------
``ernie``        Eq. 6, the Stackelberg form (the paper's headline method).
``ernie_no_st``  Eq. 4/5 with the attack detached -- the paper's own ablation,
                 plotted as "ERNIE w/o ST" in Figures 7 and 9.
``gaussian``     Appendix H's Gaussian baseline: the same penalty at a RANDOM
                 delta instead of the worst-case one. It separates "smoothness"
                 from "adversarial smoothness".
``ippo``         lambda = 0: the host's IPPO through this code path. A wiring
                 control -- it must match ``scripts/ippo.py`` up to RNG.

WHAT IS SKIPPED, AND WHY
--------------------------------------------------------------------------------
Section 5.3 (robustness to malicious ACTIONS, Eq. 7) regularises a GLOBAL
Q-function over joint actions, and Section 5.4 (the mean-field extension)
regularises a mean-field Q. URB's on-policy host has neither object: IPPO is
actor-only and per-agent. Both are recorded as N/A in the checklist rather than
approximated, because an approximation of a regulariser on a function that does
not exist is not the baseline.
"""

import numpy as np
import torch
import torch.nn.functional as F

from urb_baselines.algos.reference import PerAgentPPOAlgorithm, SingleStepPPO

__all__ = ["ERNIE", "ErniePPO", "add_args"]


class ErniePPO(SingleStepPPO):
    """URB's PPO with the ERNIE adversarial regulariser added to the loss.

    Only ``learn`` differs from ``SingleStepPPO``: the clipped surrogate and the
    entropy bonus are computed exactly as the host does, then ``lambda * R`` is
    added before ``backward``. Nothing about acting, storing or the batch trigger
    changes, so ``lambda = 0`` recovers the host learner.
    """

    def __init__(self, *a, ernie=None, **kw):
        super(ErniePPO, self).__init__(*a, **kw)
        e = dict(ernie or {})
        self.lam = float(e.get("lam", 0.5))
        self.eps = float(e.get("eps", 0.1))
        self.alpha = float(e.get("alpha", 0.0)) or self.eps
        self.steps = int(e.get("perturb_steps", 1))
        self.relative = bool(e.get("relative", True))
        self.scale_floor = float(e.get("scale_floor", 0.0))
        self.init_std = float(e.get("init_std", 1e-3))
        self.mode = str(e.get("mode", "ernie"))     # ernie | ernie_no_st | gaussian
        self.last_reg = 0.0
        self.last_delta = 0.0
        self.last_entropy = float("nan")

    # ------------------------------------------------------------------ attack
    def _dist(self, o):
        return F.softmax(self.policy_net(o), dim=-1)

    def _kl(self, p_pert, p_base):
        """KL( pi(o + delta) || pi(o) ), per sample. Eq. 4's D for a stochastic
        policy (Section 5.1)."""
        return (p_pert * (torch.log(p_pert + 1e-20)
                          - torch.log(p_base + 1e-20))).sum(-1)

    def _project(self, delta, scale):
        """Project into the l_2 ball of radius eps, in units of ``scale``.

        Appendix F: "we use the l_2 norm to bound the attacks delta". ``scale`` is
        the released code's ``torch.abs(obs)`` relative scaling -- URB's
        observation mixes a start time in seconds with small route counts, so an
        absolute epsilon would mean two completely different things in the two
        blocks. A coordinate that is exactly zero therefore receives no
        perturbation; that is the released behaviour, and ``relative: false``
        switches to absolute units.
        """
        u = delta / scale
        n = u.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1)
        factor = torch.clamp(self.eps / n, max=1.0)
        return u * factor * scale

    def _regularizer(self, obs):
        o = obs.detach()
        scale = (o.abs().clamp_min(self.scale_floor) if self.relative
                 else torch.ones_like(o))
        scale = scale.clamp_min(1e-12)

        if self.mode == "gaussian":
            # Appendix H: delta ~ N(0, I), then bounded the same way.
            delta = self._project(torch.randn_like(o) * scale, scale)
            p_base = self._dist(o)
            reg = self._kl(self._dist(o + delta), p_base).mean()
            self.last_reg = float(reg.item())
            self.last_delta = float(delta.abs().mean().item())
            return reg

        create_graph = (self.mode == "ernie")       # Eq. 6 vs Eq. 4
        delta = (torch.randn_like(o) * self.init_std * scale
                 if self.init_std > 0 else torch.zeros_like(o))
        delta = self._project(delta, scale).detach().requires_grad_(True)

        for _ in range(self.steps):
            attack_obj = self._kl(self._dist(o + delta), self._dist(o)).mean()
            grad = torch.autograd.grad(attack_obj, delta,
                                       create_graph=create_graph,
                                       retain_graph=create_graph)[0]
            gnorm = grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1)
            delta = delta + self.alpha * (grad / gnorm) * scale
            delta = self._project(delta, scale)
            if not create_graph:
                delta = delta.detach().requires_grad_(True)
        if not create_graph:
            delta = delta.detach()

        reg = self._kl(self._dist(o + delta), self._dist(o)).mean()
        self.last_reg = float(reg.detach().item())
        self.last_delta = float(delta.detach().abs().mean().item())
        with torch.no_grad():
            # A near-uniform policy has KL ~ 0 to ANY perturbation, so a
            # near-zero regulariser early in training means "the policy has not
            # committed yet", not "the attack failed". Logging the entropy
            # alongside it is what makes the two distinguishable in a run log.
            p = self._dist(o)
            self.last_entropy = float(
                -(p * torch.log(p + 1e-20)).sum(-1).mean().item())
        return reg

    # ------------------------------------------------------------------ learn
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        import random as _random
        step_loss = []
        for _ in range(self.num_epochs):
            batch = _random.sample(self.memory, self.batch_size)
            states, actions, old_log_probs, rewards = zip(*batch)
            states_tensor = torch.FloatTensor(np.asarray(states,
                                              dtype=np.float32)).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            old_lp = torch.FloatTensor(old_log_probs).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)

            logits = self.policy_net(states_tensor)
            probs = self.softmax(logits)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_tensor)

            ratio = torch.exp(new_log_probs - old_lp)
            advantage = self._advantage(rewards_tensor)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                1 + self.clip_eps) * advantage
            entropy = dist.entropy().mean()
            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            if self.lam > 0 and self.steps >= 0:
                loss = loss + self.lam * self._regularizer(states_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                           max_norm=1.0)
            self.optimizer.step()
            step_loss.append(loss.item())

        self.loss.append(sum(step_loss) / len(step_loss))
        self.memory.clear()


class ERNIE(PerAgentPPOAlgorithm):
    name = "ernie"
    ARMS = ("ernie", "ernie_no_st", "gaussian", "ippo")

    def __init__(self, ctx):
        arm = getattr(ctx.args, "arm", None) or str(
            dict(ctx.algo_cfg).get("arm", "ernie"))
        if arm not in self.ARMS:
            raise ValueError(f"[ernie] --arm must be one of {self.ARMS}, "
                             f"got {arm!r}")
        self.arm = arm
        cfg = dict(ctx.algo_cfg)
        self._ernie = dict(
            lam=0.0 if arm == "ippo" else float(cfg.get("lam", 0.5)),
            eps=float(cfg.get("eps", 0.1)),
            alpha=float(cfg.get("alpha", 0.0)),
            perturb_steps=int(cfg.get("perturb_steps", 1)),
            relative=bool(cfg.get("relative", True)),
            scale_floor=float(cfg.get("scale_floor", 0.0)),
            init_std=float(cfg.get("init_std", 1e-3)),
            mode="ernie" if arm == "ippo" else arm,
        )
        super(ERNIE, self).__init__(ctx)
        self.banner("ERNIE (NeurIPS 2023) -- adversarial regularisation", [
            ("arm", self.arm + {
                "ernie": "        Eq. 6, Stackelberg (differentiated attack)",
                "ernie_no_st": "  Eq. 4/5, detached attack (paper's 'w/o ST')",
                "gaussian": "     Appendix H, random delta",
                "ippo": "         lambda = 0: the host learner (control)",
            }[self.arm]),
            ("base learner", "URB's IPPO, unchanged (scripts/ippo.py)"),
            ("regulariser", "lambda * KL( pi(o+delta) || pi(o) ),  per agent"),
            ("lambda", self._ernie["lam"]),
            ("epsilon", f"{self._ernie['eps']}  (l_2, "
                        + ("relative to |o|" if self._ernie["relative"]
                           else "absolute") + ")"),
            ("attack", f"{self._ernie['perturb_steps']} PGD step(s), "
                       f"alpha={self._ernie['alpha'] or self._ernie['eps']}"),
            ("skipped", "Eq. 7 (joint-action Q) and 5.4 (mean-field Q): the "
                        "host has no such object"),
            ("agents", f"{self.n_av} independent learners"),
        ])

    def make_model(self, agent_id):
        return ErniePPO(self.input_size(), self.n_actions, device=self.device,
                        ernie=self._ernie, **self.host_hp)

    def diagnostics(self):
        ms = list(self.models.values())
        ent = [m.last_entropy for m in ms if np.isfinite(m.last_entropy)]
        return {"reg": float(np.mean([m.last_reg for m in ms])),
                "|d|": float(np.mean([m.last_delta for m in ms])),
                # A near-zero regulariser on a near-deterministic policy means
                # "there is nothing left to perturb", not "the attack failed";
                # without the entropy beside it the two are indistinguishable.
                "pol_ent": float(np.mean(ent)) if ent else float("nan")}


def add_args(parser):
    parser.add_argument('--arm', type=str, default=None, choices=list(ERNIE.ARMS),
                        help="ernie (Stackelberg, the paper's method), "
                             "ernie_no_st (the paper's own ablation), gaussian "
                             "(Appendix H baseline), ippo (lambda=0 control)")

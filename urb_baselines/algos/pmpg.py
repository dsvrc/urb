"""Independent learning in a performative Markov potential game.

    Sahitaj, Sasnauskas, Yalin, Mandal & Radanovic, "Independent Learning in
    Performative Markov Potential Games", arXiv:2504.20593 (2025).

WHY THIS ARM EXISTS
--------------------------------------------------------------------------------
This is the control for PACT_THEORY Theorem **B2**, the paper's most novel piece
of theory. B2 says the estimator is performatively stable only below a critical
trust ``g_stab``, and it leans on Perdomo, Zrnic, Mendler-Dunner & Hardt (2020)
plus the multi-player extension. This paper is the 2025 version of exactly the
object B2 needs: performative effects **inside a Markov potential game**, which
is the structure A1 already establishes URB to be. It contributes the missing
citation and, here, the missing arm.

The question it isolates is narrow and fair:

    Is PACT-1's benefit the identification, or is it just PERFORMATIVE STABILITY
    bought by deploying a policy, holding it fixed, and retraining on what it
    produced -- rather than chasing data its own updates keep moving?

If a plain independent learner on the same host, differing only in that it
retrains in deployments instead of continuously, recovers most of the gap, then
the schedule was doing the work. That is worth knowing before the theory section
is written, and it is cheap to find out.

WHAT THE PAPER GIVES, AND WHAT IS IMPLEMENTED
--------------------------------------------------------------------------------
* **PSE** -- a performatively stable equilibrium: each agent's policy is optimal
  for the environment its own deployment induces. Exists under a sensitivity
  assumption. Instrumented here rather than assumed: ``sens`` below is the
  measured sensitivity, the change in the induced population distribution per
  unit of policy change, which is the ``epsilon`` of B2's condition
  ``epsilon < gamma / beta_J``.

* **IPGA** -- independent policy gradient ascent, ``theta_i += alpha grad_i J_i``.

* **INPG** -- independent natural policy gradient, ``theta_i += alpha F_i^-1
  grad_i J_i``, which converges to a PSE in the LAST iterate. For a softmax
  policy the natural gradient step is exactly a shift of the logits by the
  advantage (Agarwal, Kakade, Lee & Mahajan 2021), so it is implemented in the
  equivalent mirror-descent form: build the target policy
  ``softmax(logits + eta * A)`` and minimise ``KL(target || pi_theta)``. That is
  the natural gradient, not an approximation of it, up to the network's ability
  to represent the target.

* **Repeated retraining** -- the paper's finite-time result for a special case.
  Deploy ``theta_i``; collect a whole window under it with NO updates; retrain on
  that window to convergence; redeploy. This is repeated risk minimisation, and
  it is the piece that makes the arm the control for B2.

WHY THIS IS NOT "IPPO WITH A DIFFERENT CADENCE"
--------------------------------------------------------------------------------
Two things differ from the host learner, both from the paper rather than tuning:

1. The update rule is IPGA (vanilla policy gradient) or INPG (natural gradient),
   not PPO's clipped surrogate. PPO's clip is a trust region on the *step*; the
   paper's algorithms are the two updates its theorems are about.
2. The reference policy for the whole window is the DEPLOYED policy, so every
   sample in a retrain is on-policy for it. The host's PPO reuses a memory that
   spans several of its own updates.

Everything else -- ``lr``, ``batch_size``, ``num_epochs``, widths, entropy
coefficient, ``normalize_advantage`` -- is ``ippo/config1.json`` verbatim (R2).

A NOTE WORTH KEEPING
--------------------------------------------------------------------------------
URB's own IPPO is already closer to repeated retraining than it looks: with
``batch_size = 64`` and one sample per agent per day, ``learn()`` returns early
until the 64th day and only then updates and clears. The default
``deploy_window = 64`` is chosen to match that, so the schedule arm and the host
differ in the UPDATE RULE first, and ``--arm cont`` isolates the schedule on its
own.
"""

import numpy as np
import torch

from urb_baselines.algos.reference import PerAgentPPOAlgorithm

__all__ = ["PerformativeMPG", "add_args"]


def _flat_params(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


class PerformativeMPG(PerAgentPPOAlgorithm):
    """IPGA / INPG with repeated retraining, on URB's own PPO host."""

    name = "pmpg"
    ARMS = ("inpg", "ipga", "cont")
    needs_records = True                   # the induced distribution needs peers

    def __init__(self, ctx):
        cfg = dict(ctx.algo_cfg)
        self.arm = str(getattr(ctx.args, "arm", None)
                       or cfg.get("arm", "inpg")).lower()
        if self.arm not in self.ARMS:
            raise ValueError(f"[URB-BL][pmpg] --arm must be one of {self.ARMS}")
        self.deploy_window = int(getattr(ctx.args, "deploy_window", None)
                                 or cfg.get("deploy_window", 64))
        self.retrain_epochs = int(cfg.get("retrain_epochs", 20))
        self.eta = float(cfg.get("eta", 0.5))        # NPG logit step size
        self.print_every = int(cfg.get("print_every", 100))

        super(PerformativeMPG, self).__init__(ctx)

        self.n_retrains = 0
        self.n_skipped = 0
        self.dtheta = []                 # ||theta_{i+1} - theta_i|| per retrain
        self.dmix = []                   # induced-distribution change per retrain
        self.sens = []                   # dmix / dtheta = the measured epsilon
        self.logit_shift = []            # mean |A| * eta, the size of the NPG step
        self.window_mix = np.zeros(self.n_actions, dtype=np.float64)
        self.window_days = 0
        self.prev_mix = None
        self.loss_rows = []

        self.banner("PERFORMATIVE MARKOV POTENTIAL GAME (B2 control)", [
            ("paper", "arXiv:2504.20593 (Sahitaj et al., 2025)"),
            ("arm", {"inpg": "independent NATURAL policy gradient + retraining",
                     "ipga": "independent policy gradient ascent + retraining",
                     "cont": "INPG, host cadence (isolates the schedule)"}[
                         self.arm]),
            ("deployment window", f"{self.deploy_window} days"
             if self.arm != "cont" else "N/A -- continuous updating"),
            ("retrain epochs", f"{self.retrain_epochs} passes over the window"),
            ("NPG logit step eta", f"{self.eta}" if self.arm != "ipga" else "N/A"),
            ("host hyperparameters", "ippo/config1.json verbatim (R2)"),
            ("measures", "sensitivity epsilon = d(induced mix) / d(theta)"),
        ])

    # ------------------------------------------------------------------ update
    def _retrain_one(self, model):
        """Repeated risk minimisation for one agent, against its DEPLOYED policy.

        The whole window was generated by ``pi_theta_dep``, so that is the
        reference distribution for every sample in it. IPGA maximises
        ``E[log pi(a|s) A]``; INPG minimises ``KL(softmax(l_dep + eta A e_a) ||
        pi_theta)``, which is the natural-gradient step for a softmax policy.
        """
        mem = model.memory
        if len(mem) < model.batch_size:
            self.n_skipped += 1
            return None
        states, actions, _old_lp, rewards = zip(*mem)
        S = torch.FloatTensor(np.asarray(states, dtype=np.float32)).to(self.device)
        A = torch.LongTensor(actions).to(self.device)
        R = torch.FloatTensor(rewards).to(self.device)
        adv = model._advantage(R)

        with torch.no_grad():
            dep_logits = model.policy_net(S)               # the deployed policy
        if self.arm != "ipga":
            shift = torch.zeros_like(dep_logits)
            shift.scatter_(1, A.unsqueeze(1), (self.eta * adv).unsqueeze(1))
            target = torch.softmax(dep_logits + shift, dim=-1)
            self.logit_shift.append(float((self.eta * adv).abs().mean().item()))

        losses = []
        for _ in range(max(self.retrain_epochs, 1)):
            logits = model.policy_net(S)
            logp = torch.log_softmax(logits, dim=-1)
            if self.arm == "ipga":
                # independent policy gradient ascent, unclipped
                sel = logp.gather(1, A.unsqueeze(1)).squeeze(1)
                ent = -(logp.exp() * logp).sum(-1).mean()
                loss = -(sel * adv).mean() - model.entropy_coef * ent
            else:
                # INPG as mirror descent: KL(target || pi_theta)
                ent = -(logp.exp() * logp).sum(-1).mean()
                loss = -(target * logp).sum(-1).mean() - model.entropy_coef * ent
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(),
                                           max_norm=1.0)
            model.optimizer.step()
            losses.append(float(loss.item()))
        model.memory.clear()
        v = sum(losses) / len(losses)
        model.loss.append(v)
        return v

    def learn(self, day):
        """Deployments, not steps. ``--arm cont`` falls back to the host cadence."""
        if self.arm == "cont":
            if all(len(m.memory) < m.batch_size for m in self.models.values()):
                return
            self._deploy_step()
            return
        # Repeated retraining: nothing happens inside a deployment window.
        if self.window_days < self.deploy_window:
            return
        self._deploy_step()

    def _deploy_step(self):
        before = {a: _flat_params(self.models[a].policy_net)
                  for a in self.av_ids}
        for a in self.av_ids:
            self._retrain_one(self.models[a])
        d = float(np.mean([
            float((_flat_params(self.models[a].policy_net) - before[a]
                   ).norm().item()) for a in self.av_ids]))
        self.dtheta.append(d)

        mix = (self.window_mix / self.window_days) if self.window_days else None
        if mix is not None and self.prev_mix is not None:
            # total variation between the population distributions induced by
            # two consecutive deployments -- the W1 proxy on a finite simplex
            dm = 0.5 * float(np.abs(mix - self.prev_mix).sum())
            self.dmix.append(dm)
            if d > 1e-12:
                self.sens.append(dm / d)
        if mix is not None:
            self.prev_mix = mix
        self.window_mix[:] = 0.0
        self.window_days = 0
        self.n_retrains += 1

    # ------------------------------------------------------------------ hooks
    def end_episode(self, day, phase, info):
        if phase != "train":
            return
        src = info.peer_acts or info.actions
        if src:
            for v in src.values():
                k = int(v)
                if 0 <= k < self.n_actions:
                    self.window_mix[k] += 1.0
            self.window_days += 1

    # ------------------------------------------------------------------ report
    def diagnostics(self):
        return {
            "retrains": self.n_retrains,
            "win_days": self.window_days,
            "dtheta": self.dtheta[-1] if self.dtheta else float("nan"),
            "sens": self.sens[-1] if self.sens else float("nan"),
            "npg_step": (self.logit_shift[-1] if self.logit_shift
                         else float("nan")),
        }

    def close(self):
        print("\n" + "=" * 74)
        print("[pmpg] PERFORMATIVE RETRAINING REPORT")
        print(f"  arm                      {self.arm}")
        print(f"  deployments retrained    {self.n_retrains}")
        print(f"  agent-retrains skipped   {self.n_skipped}  (window shorter "
              "than batch_size)")
        if self.dtheta:
            d = np.asarray(self.dtheta)
            print(f"  ||d theta|| first/last   {d[0]:.4e} / {d[-1]:.4e}")
            if len(d) > 2:
                ratios = d[1:] / np.maximum(d[:-1], 1e-12)
                print(f"  contraction ratio        median {np.median(ratios):.3f}"
                      "   (<1 is the paper's linear convergence to a PSE)")
        if self.sens:
            s = np.asarray(self.sens)
            print(f"  sensitivity epsilon      median {np.median(s):.4e}  "
                  f"max {s.max():.4e}")
            print("                           (B2's epsilon: change in the induced")
            print("                            population mix per unit of policy")
            print("                            change. Small means the deployment")
            print("                            loop is not chasing itself.)")
        if self.logit_shift:
            print(f"  mean NPG logit step      "
                  f"{float(np.mean(self.logit_shift)):.4f}")
        if self.n_retrains == 0:
            print("  *** NOTHING WAS EVER RETRAINED. Every deployment window ended")
            print("  *** with fewer samples than batch_size, so no agent updated")
            print("  *** once in the whole run -- this arm is a frozen random")
            print("  *** policy. Raise deploy_window or lower batch_size.")
        elif not self.dtheta or max(self.dtheta) <= 1e-12:
            print("  *** THE PARAMETERS NEVER MOVED across a retrain. The update")
            print("  *** rule produced an exactly zero step; check the advantage")
            print("  *** (a constant reward gives a zero standardised advantage).")
        elif self.arm != "ipga" and (not self.logit_shift
                                     or max(self.logit_shift) <= 1e-9):
            print("  *** THE NATURAL-GRADIENT SHIFT IS ZERO, so INPG's target is")
            print("  *** the deployed policy itself and this arm is not INPG.")
        print("=" * 74 + "\n", flush=True)


def add_args(parser):
    parser.add_argument('--arm', type=str, default=None,
                        choices=list(PerformativeMPG.ARMS),
                        help="inpg: natural policy gradient + repeated "
                             "retraining (default, the paper's last-iterate "
                             "result). ipga: plain policy gradient ascent + "
                             "retraining. cont: INPG on the host's own cadence, "
                             "which isolates the retraining schedule.")
    parser.add_argument('--deploy-window', type=int, default=None,
                        help="days per deployment before a retrain (default 64, "
                             "one full host batch per agent)")

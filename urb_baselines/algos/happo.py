"""HAPPO -- Heterogeneous-Agent PPO (ICLR 2022) on URB.

    Kuba, Chen, Wen, Wen, Sun, Wang, Yang,
    "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning",
    ICLR 2022; and Zhong et al., "Heterogeneous-Agent Reinforcement Learning",
    JMLR 2024.  https://arxiv.org/abs/2304.09870
    Code: https://github.com/PKU-MARL/HARL
          harl/algorithms/actors/happo.py      (the clipped surrogate x factor)
          harl/runners/on_policy_ha_runner.py  (the sequential update + factor)
          harl/algorithms/critics/v_critic.py  (clipped huber value loss)
          harl/common/valuenorm.py             (ValueNorm)
          harl/configs/algos_cfgs/happo.yaml   (defaults)

``paper/BASELINES.md`` B1: "The MARL literature's own answer to non-stationarity
is to constrain policy change (sequential updates, trust regions). If our drift
were learning-induced, these would help; the prediction is that they stabilise
learning and plateau at the same asymptote as MAPPO/MATD3 under the severity
parameter." HASAC is assigned to the MAPDN host (continuous actions); HAPPO is
the on-policy, discrete member of the pair and is assigned to URB.

THE THREE THINGS HAPPO ADDS TO IPPO
--------------------------------------------------------------------------------
1. a **centralised critic**: the value is computed from the joint observation
   rather than the agent's own.
2. a **sequential update**: agents are updated one at a time, in a random order
   (``fixed_order: False``).
3. the **factor** ``M = prod_{j already updated} pi_j^new(a_j) / pi_j^old(a_j)``,
   evaluated on the same data, multiplying the next agent's clipped surrogate.
   This is what makes the joint improvement monotone: agent i optimises against
   the policies its predecessors have JUST moved to, not the ones they had when
   the data was collected.

All three are implemented. (1) and (3) are exactly what URB's IPPO lacks, and
(3) is the part no other arm in this package has.

WHY "FP" AND NOT "EP"
--------------------------------------------------------------------------------
HARL supports two state types. ``EP`` uses one shared V(s) and one shared
advantage, which presumes a common reward. URB's task config runs
``av_behavior: selfish``, so travellers are paid their OWN negative travel time;
collapsing that to a mean would silently turn every arm's objective into social
welfare for this arm only. ``FP`` -- a centralised critic with a PER-AGENT value
head and per-agent advantages (``advantages[:, :, agent_id]`` in
``on_policy_ha_runner.py``) -- keeps each traveller's own objective and changes
only what the critic is allowed to see. That is the honest port and it is
supported by the reference implementation.

The centralised state is the concatenation of every machine agent's observation,
which is also what ``scripts/mappo_torchrl.py`` gives MAPPO on URB
(``MultiAgentMLP(..., centralised=True)``), so the two centralised arms see the
same object.

HOST HYPERPARAMETERS VS PAPER HYPERPARAMETERS
--------------------------------------------------------------------------------
The rule used throughout this package: anything that exists in URB's shared
config (lr, batch_size, num_epochs, clip_eps, entropy_coef, widths, num_hidden)
comes from the shared config, so an arm difference is never a tuning difference;
anything that exists only in the baseline's own paper (value_loss_coef,
huber_delta, use_valuenorm, max_grad_norm, critic_epoch, fixed_order,
action_aggregation) comes from that baseline's reference defaults -- here
``happo.yaml``. Every such value is printed in the banner and listed in
``docs/baselines/CHECKLIST_happo.md``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from urb_baselines.host import BaselineAlgorithm

__all__ = ["HAPPO", "HARLMLP", "ValueNorm", "add_args"]


class ValueNorm(object):
    """harl/common/valuenorm.py, scalar case.

    An exponentially debiased running mean/variance of the returns. URB pays
    -travel_time (order -1000) and a value head starting near zero cannot reach
    that under gradient clipping, so this is load-bearing rather than cosmetic --
    and it is HARL's own mechanism, on by default in ``happo.yaml``.
    """

    def __init__(self, beta=0.99999, epsilon=1e-5):
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.running_mean = 0.0
        self.running_mean_sq = 0.0
        self.debiasing_term = 0.0

    def _mean_var(self):
        d = max(self.debiasing_term, self.epsilon)
        m = self.running_mean / d
        msq = self.running_mean_sq / d
        return m, max(msq - m * m, 1e-2)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        w = self.beta
        self.running_mean = self.running_mean * w + float(x.mean()) * (1 - w)
        self.running_mean_sq = (self.running_mean_sq * w
                                + float((x ** 2).mean()) * (1 - w))
        self.debiasing_term = self.debiasing_term * w + (1 - w)

    def normalize(self, x):
        m, v = self._mean_var()
        return (x - m) / np.sqrt(v)

    def denormalize(self, x):
        m, v = self._mean_var()
        return x * np.sqrt(v) + m


def huber_loss(e, d):
    """harl/utils/models_tools.huber_loss."""
    a = (e.abs() <= d).float()
    b = (e.abs() > d).float()
    return a * e ** 2 / 2 + b * d * (e.abs() - d / 2)


class HARLMLP(nn.Module):
    """HARL's ``MLPBase`` + output head, which is part of HAPPO's configuration.

    ``happo.yaml``'s model block specifies ``use_feature_normalization: True``,
    ``initialization_method: orthogonal_`` and ``gain: 0.01`` on the output
    layer, and all three are load-bearing on URB.

    URB's observation is ``[start_time_in_seconds, four route counts]``: the
    first coordinate is of order 1000 and the rest are of order 1. Orthogonal
    weights are norm-preserving, so without the input LayerNorm the logits come
    out of initialisation at order 1000, the softmax is a point mass, the
    entropy is 0 and the policy gradient is exactly zero -- the actor never moves
    for the whole run. That was measured, not guessed: with the plain host MLP
    the actor's parameters changed by 1e-15 per update and the final policy was
    [0, 1, 0, 0].

    URB's own IPPO does not need this because PyTorch's default initialisation
    scales weights down with fan-in. So this block is HAPPO's own reference
    configuration rather than a tuning choice, and it is the only place in this
    package where an arm's network differs from the host's.
    """

    def __init__(self, in_size, out_size, num_hidden, widths, gain=0.01,
                 feature_norm=True):
        super(HARLMLP, self).__init__()
        self.feature_norm = nn.LayerNorm(in_size) if feature_norm else None
        self.input_layer = nn.Linear(in_size, widths[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(widths[x], widths[x + 1]) for x in range(num_hidden)])
        self.out_layer = nn.Linear(widths[-1], out_size)
        for m in [self.input_layer] + list(self.hidden_layers):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.out_layer.weight, gain=float(gain))
        nn.init.constant_(self.out_layer.bias, 0.0)

    def forward(self, x):
        if self.feature_norm is not None:
            x = self.feature_norm(x)
        x = torch.relu(self.input_layer(x))
        for h in self.hidden_layers:
            x = torch.relu(h(x))
        return self.out_layer(x)


class HAPPO(BaselineAlgorithm):
    name = "happo"

    def __init__(self, ctx):
        super(HAPPO, self).__init__(ctx)
        P = ctx.params
        cfg = dict(self.cfg)
        # happo.yaml defaults for everything URB's config does not already fix
        cfg.setdefault("critic_epoch", 5)
        cfg.setdefault("critic_num_mini_batch", 1)
        cfg.setdefault("actor_num_mini_batch", 1)
        cfg.setdefault("value_loss_coef", 1.0)
        cfg.setdefault("use_clipped_value_loss", True)
        cfg.setdefault("use_huber_loss", True)
        cfg.setdefault("huber_delta", 10.0)
        cfg.setdefault("use_valuenorm", True)
        cfg.setdefault("max_grad_norm", 10.0)
        cfg.setdefault("fixed_order", False)
        cfg.setdefault("critic_lr", None)
        cfg.setdefault("ppo_epoch", None)       # None -> host's num_epochs
        cfg.setdefault("episode_length", None)  # None -> host's batch_size
        if getattr(ctx.args, "fixed_order", False):
            cfg["fixed_order"] = True
        self.cfg = cfg

        self.lr = float(P["lr"])
        self.clip_param = float(P["clip_eps"])
        self.entropy_coef = float(P["entropy_coef"])
        self.ppo_epoch = int(cfg["ppo_epoch"] or P["num_epochs"])
        self.episode_length = int(cfg["episode_length"] or P["batch_size"])
        self.num_hidden = int(P["num_hidden"])
        self.widths = list(P["widths"])

        gain = float(cfg.setdefault("gain", 0.01))          # happo.yaml
        fnorm = bool(cfg.setdefault("use_feature_normalization", True))

        # ---- actors: one per agent (share_param: False in happo.yaml) ----
        self.actors = {
            a: HARLMLP(self.obs_size, self.n_actions, self.num_hidden,
                       self.widths, gain=gain,
                       feature_norm=fnorm).to(self.device)
            for a in self.av_ids
        }
        self.actor_opt = {a: optim.Adam(m.parameters(), lr=self.lr, eps=1e-5)
                          for a, m in self.actors.items()}

        # ---- centralised critic: joint obs -> one value PER AGENT (FP) ----
        self.state_size = self.obs_size * self.n_av
        self.critic = HARLMLP(self.state_size, self.n_av, self.num_hidden,
                              self.widths, gain=1.0,
                              feature_norm=fnorm).to(self.device)
        self.critic_opt = optim.Adam(
            self.critic.parameters(),
            lr=float(cfg["critic_lr"] or self.lr), eps=1e-5)
        self.vnorm = ValueNorm() if cfg["use_valuenorm"] else None

        # ---- day-indexed buffer ----
        self._reset_buffer()
        self._pending = {}
        self.rng = np.random.RandomState(int(ctx.torch_seed) * 7919 + 101)
        self.loss = []
        self.last = {"pg": 0.0, "v": 0.0, "factor": 1.0, "ratio": 1.0}

        self.banner("HAPPO (ICLR 2022 / JMLR 2024) -- sequential trust region", [
            ("actors", f"{self.n_av} heterogeneous (share_param: False)"),
            ("critic", f"centralised: {self.state_size} -> {self.n_av} values "
                       f"(FP, per-agent advantage)"),
            ("sequential update", "random order every iteration "
                                  "(fixed_order: False)"),
            ("factor", "prod of predecessors' new/old action probabilities"),
            ("collection", f"{self.episode_length} days per update"),
            ("ppo_epoch", f"{self.ppo_epoch} (host num_epochs); "
                          f"critic_epoch {cfg['critic_epoch']} (happo.yaml)"),
            ("value loss", f"clipped={cfg['use_clipped_value_loss']} "
                           f"huber(delta={cfg['huber_delta']}) "
                           f"valuenorm={cfg['use_valuenorm']}"),
            ("max_grad_norm", f"{cfg['max_grad_norm']} (happo.yaml)"),
            ("advantage", "r - V(s)  (one-step episode: GAE collapses exactly)"),
        ])

    # ------------------------------------------------------------------ buffer
    def _reset_buffer(self):
        self.b_obs = []       # [T][n_av, obs]
        self.b_act = []       # [T][n_av]
        self.b_logp = []      # [T][n_av]
        self.b_rew = []       # [T][n_av]

    # ------------------------------------------------------------------ hooks
    def begin_episode(self, day, phase):
        self._pending = {}

    def act(self, agent_id, obs):
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        t = torch.as_tensor(o, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actors[agent_id](t)
            log_pi = F.log_softmax(logits, dim=-1).view(-1)
        if self.deterministic:
            a = int(torch.argmax(log_pi).item())
        else:
            a = int(torch.distributions.Categorical(
                logits=logits.view(-1)).sample().item())
        self._pending[agent_id] = (o, a, float(log_pi[a].item()))
        return a

    def push(self, agent_id, reward):
        if agent_id in self._pending:
            o, a, lp = self._pending[agent_id]
            self._pending[agent_id] = (o, a, lp, float(reward))

    def end_episode(self, day, phase, info):
        if phase != "train":
            return
        # Every machine agent acts exactly once per day, so the buffer is
        # day-indexed and the factor is well defined pointwise across agents.
        if len(self._pending) != self.n_av:
            return
        obs = np.zeros((self.n_av, self.obs_size), dtype=np.float32)
        act = np.zeros(self.n_av, dtype=np.int64)
        logp = np.zeros(self.n_av, dtype=np.float32)
        rew = np.zeros(self.n_av, dtype=np.float32)
        for i, a in enumerate(self.av_ids):
            rec = self._pending.get(a)
            if rec is None or len(rec) < 4:
                return              # a traveller never completed: drop the day
            obs[i], act[i], logp[i], rew[i] = rec[0], rec[1], rec[2], rec[3]
        self.b_obs.append(obs)
        self.b_act.append(act)
        self.b_logp.append(logp)
        self.b_rew.append(rew)

    # ------------------------------------------------------------------ learn
    def learn(self, day):
        T = len(self.b_obs)
        if T < self.episode_length:
            return
        dev = self.device
        obs = torch.as_tensor(np.asarray(self.b_obs), device=dev)      # (T,N,o)
        act = torch.as_tensor(np.asarray(self.b_act), device=dev)      # (T,N)
        old_lp = torch.as_tensor(np.asarray(self.b_logp), device=dev)  # (T,N)
        rew_np = np.asarray(self.b_rew, dtype=np.float64)              # (T,N)
        self._reset_buffer()

        state = obs.reshape(T, -1)                                     # (T, N*o)
        with torch.no_grad():
            v_pred = self.critic(state).cpu().numpy()                  # (T,N)

        # One step and terminal, so returns = r for any gamma/lambda and GAE
        # collapses to the TD error. HARL denormalises the value prediction
        # before subtracting -- on_policy_ha_runner.train.
        returns = rew_np
        v_den = self.vnorm.denormalize(v_pred) if self.vnorm is not None else v_pred
        adv = returns - v_den
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)                  # FP path
        adv_t = torch.as_tensor(adv.astype(np.float32), device=dev)

        # ---- sequential actor update with the factor -------------------
        factor = torch.ones(T, device=dev)
        order = (list(range(self.n_av)) if self.cfg["fixed_order"]
                 else list(self.rng.permutation(self.n_av)))
        pg_losses, ratios = [], []
        for i in order:
            a_id = self.av_ids[i]
            model, opt = self.actors[a_id], self.actor_opt[a_id]
            o_i, a_i = obs[:, i, :], act[:, i]
            with torch.no_grad():
                lp_before = F.log_softmax(model(o_i), -1).gather(
                    1, a_i.unsqueeze(-1)).squeeze(-1)

            n_mb = max(1, int(self.cfg["actor_num_mini_batch"]))
            per = max(1, T // n_mb)
            for _ in range(self.ppo_epoch):
                perm = torch.randperm(T, device=dev)
                for m in range(n_mb):
                    idx = perm[m * per:(m + 1) * per]
                    if len(idx) == 0:
                        continue
                    logits = model(o_i[idx])
                    log_pi = F.log_softmax(logits, -1)
                    lp = log_pi.gather(1, a_i[idx].unsqueeze(-1)).squeeze(-1)
                    imp = torch.exp(lp - old_lp[idx, i])
                    surr1 = imp * adv_t[idx, i]
                    surr2 = torch.clamp(imp, 1 - self.clip_param,
                                        1 + self.clip_param) * adv_t[idx, i]
                    entropy = -(log_pi * log_pi.exp()).sum(-1).mean()
                    # happo.py: the factor multiplies the clipped surrogate.
                    pg = -(factor[idx] * torch.min(surr1, surr2)).mean()
                    loss = pg - entropy * self.entropy_coef
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(self.cfg["max_grad_norm"]))
                    opt.step()
                    pg_losses.append(float(pg.item()))
                    ratios.append(float(imp.mean().item()))

            with torch.no_grad():
                lp_after = F.log_softmax(model(o_i), -1).gather(
                    1, a_i.unsqueeze(-1)).squeeze(-1)
                factor = factor * torch.exp(lp_after - lp_before)

        # ---- critic ------------------------------------------------------
        ret_t = torch.as_tensor(returns.astype(np.float32), device=dev)
        vp_t = torch.as_tensor(v_pred.astype(np.float32), device=dev)
        if self.vnorm is not None:
            self.vnorm.update(returns)
            tgt = torch.as_tensor(
                self.vnorm.normalize(returns).astype(np.float32), device=dev)
        else:
            tgt = ret_t
        v_losses = []
        n_mb = max(1, int(self.cfg["critic_num_mini_batch"]))
        per = max(1, T // n_mb)
        for _ in range(int(self.cfg["critic_epoch"])):
            perm = torch.randperm(T, device=dev)
            for m in range(n_mb):
                idx = perm[m * per:(m + 1) * per]
                if len(idx) == 0:
                    continue
                values = self.critic(state[idx])
                clipped = vp_t[idx] + (values - vp_t[idx]).clamp(
                    -self.clip_param, self.clip_param)
                e_orig = tgt[idx] - values
                e_clip = tgt[idx] - clipped
                d = float(self.cfg["huber_delta"])
                if self.cfg["use_huber_loss"]:
                    lo, lc = huber_loss(e_orig, d), huber_loss(e_clip, d)
                else:
                    lo, lc = e_orig ** 2 / 2, e_clip ** 2 / 2
                v_loss = (torch.max(lo, lc) if self.cfg["use_clipped_value_loss"]
                          else lo).mean() * float(self.cfg["value_loss_coef"])
                self.critic_opt.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(self.cfg["max_grad_norm"]))
                self.critic_opt.step()
                v_losses.append(float(v_loss.item()))

        self.last = {
            "pg": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "v": float(np.mean(v_losses)) if v_losses else 0.0,
            "factor": float(factor.mean().item()),
            "ratio": float(np.mean(ratios)) if ratios else 1.0,
        }
        self.loss.append(self.last["pg"])

    # ------------------------------------------------------------------ misc
    def begin_test(self):
        self.deterministic = True
        for m in self.actors.values():
            m.eval()

    def diagnostics(self):
        return dict(self.last)

    def loss_records(self):
        return [{"iteration": i, "agent_id": "all", "loss": v}
                for i, v in enumerate(self.loss, start=1)]


def add_args(parser):
    parser.add_argument('--fixed-order', action='store_true',
                        help="update agents in index order instead of a fresh "
                             "random order each iteration (happo.yaml default "
                             "is random)")

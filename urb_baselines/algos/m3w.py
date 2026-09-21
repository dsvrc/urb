"""M3W -- a mixture-of-experts world model, planned with multi-agent MPPI.

    Zhao, Xu, Fu, Chai, Zhu & Zhao, "Learning and Planning Multi-Agent Tasks via
    a MoE-based World Model", NeurIPS 2025,
    code https://github.com/zhaozijie2022/m3w-marl.

WHY THIS ARM EXISTS
--------------------------------------------------------------------------------
``paper/BASELINES.md`` B12 (model-based MARL) is the last class still marked
"cite; run only if asked", and its stated prediction is the one this arm makes
falsifiable:

    "a world model absorbs the drift into its latent but has no separate,
     invertible object for the coupling gain."

M3W is the sharpest possible version of that comparison, because on a one-step
URB day its world model and PACT-1's estimator are predicting **the same
quantity**. PACT-1 predicts the delay on each candidate route from a declared
basis with an RLS fit; M3W predicts the same day's reward from a learned
SparseMoE over a latent, and then, instead of shifting logits by a trust-weighted
z-score, it PLANS: it samples joint route assignments, scores them under its own
model and takes an MPPI-weighted choice.

So the arm isolates two things at once, and the ablations separate them:
the model (declared vs learned) and the use of it (steer vs plan).

WHAT URB FORCES -- FIVE ROWS, ALL IN THE CHECKLIST
--------------------------------------------------------------------------------
1. **horizon = 1.** The planner's default horizon in the release is 3 steps of
   latent rollout. A URB day is one step and the episode ends, so a horizon of
   more than one day is a forecast over DAYS, not a rollout within an episode.
   The dynamics model is kept and predicts the next DAY's latent -- which is a
   real object here, and the one a drifting instance makes interesting -- and
   ``horizon`` may be raised to plan over days.

2. **Categorical MPPI.** The release samples actions from a Gaussian and clamps
   to [-1, 1]. URB's action is one of K routes, so the sampling distribution is a
   per-agent categorical and the MPPI refit is the elite-weighted empirical
   distribution with a probability floor in place of ``min_std``. This is the
   standard discrete instantiation of the same update, not a different planner.

3. **The snapshot is YESTERDAY (SNAP-1).** Planning needs a joint action, and a
   URB day is sequential -- there is no instant at which every agent's current
   observation exists. The joint plan is therefore computed at the start of the
   day from each agent's last observation. Each agent then refines only its OWN
   route at its own turn, using today's real observation with the peers held at
   the plan. Both halves are the receding-horizon idea; the checklist records
   which is which.

4. **The coupling enters through the neighbour set.** The reward model needs to
   know what the peers chose, or a plan cannot be better than a policy. Each
   candidate is scored with the route histogram of the agent's neighbours under
   the sampled joint action, from ``urb_baselines/graph.py`` -- the same graph
   DGN uses, built the same way.

5. **The reward is standardised before the two-hot.** This is the FOURTH member
   of the family in ``docs/baselines/README.md`` section 4, and the most silent
   of them. The release encodes rewards two-hot over ``num_bins = 101`` spanning
   ``[reward_min, reward_max] = [-10, 10]``. URB pays about -1000. Every reward
   in the run lands in the bottom bin, the reward model becomes a constant, every
   sampled joint action scores identically, and MPPI's softmax over identical
   values is the uniform distribution -- i.e. **the planner picks routes at
   random while its loss curve looks perfectly healthy**. ``normalize_reward:
   false`` reproduces it exactly; ``plan_spread`` is printed every
   ``print_every`` days so it cannot be missed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from urb_baselines import graph as graphmod
from urb_baselines.algos.reference import PerAgentDQNAlgorithm

__all__ = ["M3W", "SoftMoE", "SparseMoE", "add_args"]


# ==========================================================================
#  the two mixtures, as the release defines them
# ==========================================================================
class SimNorm(nn.Module):
    """TD-MPC2's simplicial normalisation, kept because M3W's latent uses it."""

    def __init__(self, dim):
        super(SimNorm, self).__init__()
        self.dim = int(dim)

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        return torch.softmax(x, dim=-1).view(*shp)


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Mish(),
                         nn.Linear(hidden, hidden), nn.Mish(),
                         nn.Linear(hidden, out_dim))


class SoftMoE(nn.Module):
    """Soft mixture: every expert runs, the gate is a softmax over all of them.

    M3W uses this for the DYNAMICS, on the argument that task dynamics are
    boundedly similar -- so knowledge should be shared, not routed away.
    """

    def __init__(self, in_dim, hidden, out_dim, n_experts):
        super(SoftMoE, self).__init__()
        self.gate = nn.Linear(in_dim, int(n_experts))
        self.experts = nn.ModuleList(
            [_mlp(in_dim, hidden, out_dim) for _ in range(int(n_experts))])

    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=-1)                 # (B, E)
        ys = torch.stack([e(x) for e in self.experts], dim=1)   # (B, E, out)
        return (w.unsqueeze(-1) * ys).sum(dim=1)


class SparseMoE(nn.Module):
    """Noisy top-k routing, the release's ``NoisyTopKRouter``, for the REWARD.

    Rewards differ far more across tasks than dynamics do, so M3W routes them
    sparsely to avoid gradient conflict. ``balance`` is the load-balancing term
    (the squared coefficient of variation of the gate mass), returned so the
    caller can add it with the paper's ``balance_coef``.
    """

    def __init__(self, in_dim, hidden, out_dim, n_experts, k=2, noisy=True):
        super(SparseMoE, self).__init__()
        self.k = min(int(k), int(n_experts))
        self.noisy = bool(noisy)
        self.w_gate = nn.Linear(in_dim, int(n_experts), bias=False)
        self.w_noise = nn.Linear(in_dim, int(n_experts), bias=False)
        self.experts = nn.ModuleList(
            [_mlp(in_dim, hidden, out_dim) for _ in range(int(n_experts))])
        self.n_experts = int(n_experts)

    def forward(self, x):
        logits = self.w_gate(x)
        if self.noisy and self.training:
            noise = torch.nn.functional.softplus(self.w_noise(x)) + 1e-2
            logits = logits + torch.randn_like(logits) * noise
        top_v, top_i = logits.topk(self.k, dim=-1)
        gates = torch.zeros_like(logits).scatter_(
            -1, top_i, torch.softmax(top_v, dim=-1))
        ys = torch.stack([e(x) for e in self.experts], dim=1)
        out = (gates.unsqueeze(-1) * ys).sum(dim=1)
        load = gates.sum(dim=0)
        balance = (load.std() / (load.mean() + 1e-8)) ** 2
        return out, balance


# ==========================================================================
#  two-hot reward coding
# ==========================================================================
class TwoHot(object):
    """``num_bins`` bins over ``[lo, hi]``; a scalar becomes two adjacent masses.

    The release's own coding. The only thing changed here is WHAT is coded: a
    standardised reward rather than a raw one, for the reason in the module
    docstring.
    """

    def __init__(self, lo, hi, bins, device):
        self.lo, self.hi, self.bins = float(lo), float(hi), int(bins)
        self.centres = torch.linspace(self.lo, self.hi, self.bins, device=device)
        self.width = (self.hi - self.lo) / max(self.bins - 1, 1)

    def encode(self, v):
        v = v.clamp(self.lo, self.hi)
        pos = (v - self.lo) / self.width
        lo_i = pos.floor().long().clamp(0, self.bins - 1)
        hi_i = (lo_i + 1).clamp(0, self.bins - 1)
        w_hi = (pos - lo_i.float()).clamp(0.0, 1.0)
        out = torch.zeros(v.shape[0], self.bins, device=v.device)
        out.scatter_(1, lo_i.unsqueeze(1), (1.0 - w_hi).unsqueeze(1))
        out.scatter_add_(1, hi_i.unsqueeze(1), w_hi.unsqueeze(1))
        return out

    def decode(self, logits):
        return (torch.softmax(logits, dim=-1) * self.centres).sum(-1)

    def clipped_fraction(self, v):
        return float(((v < self.lo) | (v > self.hi)).float().mean().item())


# ==========================================================================
#  the algorithm
# ==========================================================================
class M3W(PerAgentDQNAlgorithm):
    """MoE world model + multi-agent MPPI on URB's own DQN host."""

    name = "m3w"
    ARMS = ("m3w", "mlp", "greedy")
    # Neither flag is needed: the neighbour histogram is over MACHINE agents,
    # which ``info.actions`` always carries, and the observations are captured
    # in ``act`` because the planner needs them keyed by slot anyway.

    def __init__(self, ctx):
        cfg = dict(ctx.algo_cfg)
        self.arm = str(getattr(ctx.args, "arm", None)
                       or cfg.get("arm", "m3w")).lower()
        if self.arm not in self.ARMS:
            raise ValueError(f"[URB-BL][m3w] --arm must be one of {self.ARMS}")
        super(M3W, self).__init__(ctx)

        self.latent_dim = int(cfg.get("latent_dim", 128))
        self.simnorm_dim = int(cfg.get("simnorm_dim", 8))
        self.hidden = int(cfg.get("hidden", 128))
        self.n_dyn = int(cfg.get("num_dynamics_experts", 16))
        self.n_rew = int(cfg.get("num_reward_experts", 16))
        self.top_k = int(cfg.get("top_k", 2))
        self.num_bins = int(cfg.get("num_bins", 101))
        self.r_lo = float(cfg.get("reward_min", -10.0))
        self.r_hi = float(cfg.get("reward_max", 10.0))
        self.balance_coef = float(cfg.get("balance_coef", 0.0005))
        self.reward_coef = float(cfg.get("reward_coef", 0.1))
        self.dynamics_coef = float(cfg.get("dynamics_coef", 20.0))
        self.normalize_reward = bool(cfg.get("normalize_reward", True))
        self.horizon = int(getattr(ctx.args, "horizon", None)
                           or cfg.get("horizon", 1))
        self.num_samples = int(cfg.get("num_samples", 128))
        self.num_elites = int(cfg.get("num_elites", 16))
        self.plan_iter = int(cfg.get("iterations", 6))
        self.num_pi_trajs = int(cfg.get("num_pi_trajs", 8))
        self.temperature = float(cfg.get("temperature", 0.5))
        self.min_prob = float(cfg.get("min_prob", 0.02))
        self.wm_lr = float(cfg.get("wm_lr", 0.0005))
        self.train_every = int(cfg.get("train_every", 1))
        self.wm_batch = int(cfg.get("wm_batch", 256))
        self.warmup_days = int(cfg.get("warmup_days", 100))
        self.print_every = int(cfg.get("print_every", 100))

        self.graph = graphmod.build_from_ctx(
            ctx, mode=str(cfg.get("graph", "overlap")),
            n_neighbors=int(cfg.get("n_neighbors", 3)),
            slack=float(cfg.get("slack", 0.0)), tag="URB-BL/m3w",
            verbose=True)
        self.nbr_slots = np.full((self.n_av, int(cfg.get("n_neighbors", 3))),
                                 -1, dtype=np.int64)
        for a in self.av_ids:
            i = self.slot_of[a]
            for j, nb in enumerate(self.graph.neighbours(a)[:self.nbr_slots.shape[1]]):
                self.nbr_slots[i, j] = self.slot_of[nb]

        D, K = self.latent_dim, self.n_actions
        # feature = [z, a_onehot, neighbour route histogram]
        feat = D + K + K
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_size, self.hidden), nn.Mish(),
            nn.Linear(self.hidden, D), SimNorm(self.simnorm_dim)).to(self.device)
        if self.arm == "mlp":
            self.dyn = _mlp(D + K, self.hidden, D).to(self.device)
            self.rew = _mlp(feat, self.hidden, self.num_bins).to(self.device)
        else:
            self.dyn = SoftMoE(D + K, self.hidden, D, self.n_dyn).to(self.device)
            self.rew = SparseMoE(feat, self.hidden, self.num_bins, self.n_rew,
                                 self.top_k).to(self.device)
        self.wm_opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.dyn.parameters())
            + list(self.rew.parameters()), lr=self.wm_lr)
        self.twohot = TwoHot(self.r_lo, self.r_hi, self.num_bins, self.device)

        # (obs, slot, action, reward, nbr_hist, prev_obs)
        self.buf = []
        self.buf_cap = int(ctx.params["buffer_size"]) * 8
        self.last_obs = {a: np.zeros(self.obs_size, dtype=np.float32)
                         for a in self.av_ids}
        self.plan = np.zeros(self.n_av, dtype=np.int64)
        self.plan_probs = np.full((self.n_av, K), 1.0 / K, dtype=np.float64)
        self.r_n, self.r_mean, self.r_m2 = 0, 0.0, 0.0
        self.n_wm_updates = 0
        self.n_plans = 0
        self.day = 0
        self._today_obs = {}
        self.spread_track = []
        self.clip_track = []
        self.wm_loss = []

        self.banner("M3W (B12: model-based MARL, MoE world model + MPPI)", [
            ("paper", "NeurIPS 2025 (Zhao et al.), github zhaozijie2022/m3w-marl"),
            ("arm", {"m3w": "SoftMoE dynamics + SparseMoE reward + MPPI",
                     "mlp": "plain MLP world model + MPPI (ablation: no MoE)",
                     "greedy": "MoE world model, argmax instead of MPPI"}[
                         self.arm]),
            ("seed policy", "URB's own DQN (scripts/iql.py), unchanged"),
            ("latent", f"{D}  (SimNorm dim {self.simnorm_dim})"),
            ("experts", f"dynamics {self.n_dyn} soft / reward {self.n_rew} "
                        f"top-{self.top_k}"),
            ("planner", f"{self.num_samples} joint samples, {self.num_elites} "
                        f"elites, {self.plan_iter} iterations, T="
                        f"{self.temperature}"),
            ("horizon", f"{self.horizon} day(s)  -- a day is one step"),
            ("coupling", f"neighbour histogram, graph={self.graph.mode}"),
            ("reward coding", f"two-hot, {self.num_bins} bins on "
                              f"[{self.r_lo}, {self.r_hi}] of "
             + ("STANDARDISED reward" if self.normalize_reward
                else "*** RAW reward -- every bin saturates")),
        ])

    # ------------------------------------------------------------------ reward
    def _r_std(self, r):
        if not self.normalize_reward:
            return float(r)
        if self.r_n < 2:
            return 0.0
        sd = float(np.sqrt(self.r_m2 / self.r_n))
        return (float(r) - self.r_mean) / (sd + 1e-6)

    def _r_push(self, r):
        self.r_n += 1
        d = float(r) - self.r_mean
        self.r_mean += d / self.r_n
        self.r_m2 += d * (float(r) - self.r_mean)

    # ------------------------------------------------------------------ model
    def _nbr_hist(self, actions):
        """(n_av, K) route histogram of each agent's neighbours, normalised.

        ``actions`` is (S, n_av) or (n_av,). Returns (S, n_av, K) or (n_av, K).
        """
        a = np.atleast_2d(np.asarray(actions, dtype=np.int64))
        S = a.shape[0]
        out = np.zeros((S, self.n_av, self.n_actions), dtype=np.float32)
        for j in range(self.nbr_slots.shape[1]):
            sl = self.nbr_slots[:, j]
            ok = sl >= 0
            if not ok.any():
                continue
            picks = a[:, sl[ok]]                       # (S, n_ok)
            rows = np.repeat(np.arange(S)[:, None], picks.shape[1], axis=1)
            cols = np.repeat(np.where(ok)[0][None, :], S, axis=0)
            np.add.at(out, (rows.ravel(), cols.ravel(), picks.ravel()), 1.0)
        tot = out.sum(-1, keepdims=True)
        out = np.divide(out, tot, out=np.zeros_like(out), where=tot > 0)
        return out[0] if np.ndim(actions) == 1 else out

    def _score(self, z, act_idx, nbr_hist):
        """Predicted standardised reward for (latent, action, neighbour mix)."""
        onehot = torch.zeros(act_idx.shape[0], self.n_actions, device=self.device)
        onehot.scatter_(1, act_idx.unsqueeze(1), 1.0)
        feat = torch.cat([z, onehot, nbr_hist], dim=-1)
        out = self.rew(feat)
        logits = out[0] if isinstance(out, tuple) else out
        return self.twohot.decode(logits)

    # ------------------------------------------------------------------ plan
    @torch.no_grad()
    def _mppi(self):
        """Their ``plan()`` with a categorical in place of the Gaussian.

        Sample joint route assignments from the per-agent categorical, score them
        under the world model, keep the elites, reweight by
        ``exp(temperature * (v - max v))`` and refit. ``num_pi_trajs`` of the
        samples are seeded from the DQN's greedy action, which is the release's
        policy-guided half.
        """
        n, K, S = self.n_av, self.n_actions, self.num_samples
        was_training = self.rew.training
        self.rew.eval()                      # no noisy gating while planning
        obs = np.stack([self.last_obs[a] for a in self.av_ids])
        z = self.encoder(torch.from_numpy(obs).float().to(self.device))

        probs = np.full((n, K), 1.0 / K, dtype=np.float64)
        pi_seed = None
        if self.num_pi_trajs > 0:
            q = torch.stack([self.models[a].q_network(
                torch.from_numpy(self.last_obs[a]).float().unsqueeze(0
                ).to(self.device)).squeeze(0) for a in self.av_ids])
            pi_seed = q.argmax(-1).cpu().numpy().astype(np.int64)

        vals = None
        for it in range(max(self.plan_iter, 1)):
            samples = np.empty((S, n), dtype=np.int64)
            cdf = np.cumsum(probs, axis=1)
            u = np.random.rand(S, n, 1)
            samples[:] = (u > cdf[None, :, :]).sum(-1).clip(0, K - 1)
            if pi_seed is not None:
                samples[:self.num_pi_trajs] = pi_seed[None, :]
            hist = self._nbr_hist(samples)                      # (S, n, K)

            zz = z.unsqueeze(0).expand(S, n, self.latent_dim).reshape(S * n, -1)
            aa = torch.from_numpy(samples.reshape(-1)).to(self.device)
            hh = torch.from_numpy(hist.reshape(S * n, K)).to(self.device)
            r = self._score(zz, aa, hh).reshape(S, n)
            # horizon > 1 plans over DAYS: roll the latent forward with the
            # dynamics model and keep scoring the same sampled assignment,
            # which is the release's rollout with the day as the step.
            for _h in range(max(self.horizon, 1) - 1):
                onehot = torch.zeros(S * n, K, device=self.device)
                onehot.scatter_(1, aa.unsqueeze(1), 1.0)
                zz = self.dyn(torch.cat([zz, onehot], dim=-1))
                r = r + self._score(zz, aa, hh).reshape(S, n)
            # M3W scores a joint action by the MEAN of the per-agent returns.
            vals = r.mean(dim=1).cpu().numpy()

            if self.arm == "greedy":
                best = samples[int(np.argmax(vals))]
                probs = np.full((n, K), self.min_prob, dtype=np.float64)
                probs[np.arange(n), best] = 1.0
                probs /= probs.sum(1, keepdims=True)
                break
            e = min(self.num_elites, S)
            elite = np.argsort(-vals)[:e]
            v = vals[elite]
            w = np.exp(self.temperature * (v - v.max()))
            w = w / max(w.sum(), 1e-12)
            new = np.full((n, K), 0.0, dtype=np.float64)
            for wi, si in zip(w, elite):
                new[np.arange(n), samples[si]] += wi
            new = new + self.min_prob                    # the min_std analogue
            probs = new / new.sum(1, keepdims=True)

        if vals is not None:
            rng = float(np.max(vals) - np.min(vals))
            self.spread_track.append(rng)
        self.plan_probs = probs
        cdf = np.cumsum(probs, axis=1)
        u = np.random.rand(self.n_av, 1)
        self.plan = (u > cdf).sum(-1).clip(0, K - 1).astype(np.int64)
        self.n_plans += 1
        if was_training:
            self.rew.train()

    @torch.no_grad()
    def _refine(self, agent_id, obs):
        """The agent's own turn: re-score its K routes with TODAY's observation,
        peers held at the plan. The receding-horizon half of SNAP-1."""
        i = self.slot_of[agent_id]
        K = self.n_actions
        was_training = self.rew.training
        self.rew.eval()
        z = self.encoder(torch.from_numpy(
            np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device))
        hist = self._nbr_hist(self.plan)[i]
        zz = z.expand(K, self.latent_dim)
        aa = torch.arange(K, device=self.device)
        hh = torch.from_numpy(np.repeat(hist[None, :], K, axis=0)
                              ).to(self.device)
        r = self._score(zz, aa, hh).cpu().numpy()
        if was_training:
            self.rew.train()
        if self.deterministic or self.arm == "greedy":
            return int(np.argmax(r))
        w = np.exp(self.temperature * (r - r.max())) * self.plan_probs[i]
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return int(self.plan[i])
        return int(np.random.choice(K, p=w / s))

    # ------------------------------------------------------------------ hooks
    def begin_episode(self, day, phase):
        self.day = int(day)
        self._today_obs = {}
        if day >= self.warmup_days:
            self._mppi()

    def act(self, agent_id, obs):
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._today_obs[agent_id] = o
        if self.day < self.warmup_days:
            # Warm-up is the host's own DQN: a plan under an untrained world
            # model is noise, and the buffer needs data before the model is
            # worth planning with. The release calls this warmup_train.
            return super(M3W, self).act(agent_id, obs)
        a = self._refine(agent_id, o)
        m = self.models[agent_id]
        m.last_state = o
        m.last_action = int(a)
        return int(a)

    def end_episode(self, day, phase, info):
        if phase != "train":
            return
        acts = np.zeros(self.n_av, dtype=np.int64)
        for a in self.av_ids:
            k = info.actions.get(a)
            if k is not None:
                acts[self.slot_of[a]] = int(k)
        hist = self._nbr_hist(acts)
        for a in self.av_ids:
            o = self._today_obs.get(a)
            r = info.rewards.get(a)
            k = info.actions.get(a)
            if o is None or r is None or k is None:
                continue
            self._r_push(r)
            i = self.slot_of[a]
            self.buf.append((o.copy(), i, int(k), float(r), hist[i].copy(),
                             self.last_obs[a].copy()))
            self.last_obs[a] = o.copy()
        if len(self.buf) > self.buf_cap:
            del self.buf[:len(self.buf) - self.buf_cap]

    def learn(self, day):
        super(M3W, self).learn(day)          # the seed policy keeps learning
        if (day + 1) % max(self.train_every, 1) == 0:
            self._train_wm()

    def _train_wm(self):
        if len(self.buf) < self.wm_batch:
            return
        idx = np.random.choice(len(self.buf), self.wm_batch, replace=False)
        obs = torch.from_numpy(np.stack([self.buf[i][0] for i in idx])
                               ).float().to(self.device)
        prev = torch.from_numpy(np.stack([self.buf[i][5] for i in idx])
                                ).float().to(self.device)
        act = torch.from_numpy(np.asarray([self.buf[i][2] for i in idx])
                               ).long().to(self.device)
        rew = torch.from_numpy(np.asarray(
            [self._r_std(self.buf[i][3]) for i in idx], dtype=np.float32)
            ).to(self.device)
        hist = torch.from_numpy(np.stack([self.buf[i][4] for i in idx])
                                ).float().to(self.device)
        self.clip_track.append(self.twohot.clipped_fraction(rew))

        z_prev = self.encoder(prev)
        z_now = self.encoder(obs)
        onehot = torch.zeros(act.shape[0], self.n_actions, device=self.device)
        onehot.scatter_(1, act.unsqueeze(1), 1.0)

        # dynamics: yesterday's latent + today's action -> today's latent
        z_pred = self.dyn(torch.cat([z_prev, onehot], dim=-1))
        dyn_loss = ((z_pred - z_now.detach()) ** 2).mean()

        out = self.rew(torch.cat([z_now, onehot, hist], dim=-1))
        logits, balance = (out if isinstance(out, tuple)
                           else (out, torch.zeros((), device=self.device)))
        target = self.twohot.encode(rew)
        rew_loss = -(target * torch.log_softmax(logits, dim=-1)).sum(-1).mean()

        loss = (self.dynamics_coef * dyn_loss + self.reward_coef * rew_loss
                + self.balance_coef * balance)
        self.wm_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.dyn.parameters())
            + list(self.rew.parameters()), 10.0)
        self.wm_opt.step()
        self.n_wm_updates += 1
        self.wm_loss.append(float(loss.item()))

    def begin_test(self):
        super(M3W, self).begin_test()
        self.encoder.eval()
        self.dyn.eval()
        self.rew.eval()

    # ------------------------------------------------------------------ report
    def diagnostics(self):
        d = super(M3W, self).diagnostics()
        d.update({
            "wm_upd": self.n_wm_updates,
            "plans": self.n_plans,
            "plan_spread": (float(np.mean(self.spread_track[-50:]))
                            if self.spread_track else float("nan")),
            "clip": (float(np.mean(self.clip_track[-50:]))
                     if self.clip_track else float("nan")),
            "wm_loss": self.wm_loss[-1] if self.wm_loss else float("nan"),
        })
        return d

    def loss_records(self):
        rows = super(M3W, self).loss_records()
        for it, v in enumerate(self.wm_loss, start=1):
            rows.append({"iteration": it, "agent_id": "shared/world_model",
                         "loss": v})
        return rows

    def close(self):
        spread = (float(np.mean(self.spread_track)) if self.spread_track else 0.0)
        clip = float(np.mean(self.clip_track)) if self.clip_track else float("nan")
        print("\n" + "=" * 74)
        print("[m3w] WORLD MODEL / PLANNER REPORT")
        print(f"  arm                      {self.arm}")
        print(f"  world-model updates      {self.n_wm_updates}")
        print(f"  days planned             {self.n_plans}")
        print(f"  MPPI value spread        {spread:.4f}   (max - min over the "
              f"{self.num_samples} sampled joint actions)")
        print(f"  two-hot clipped fraction {clip:.4f}   (rewards outside "
              f"[{self.r_lo}, {self.r_hi}])")
        print(f"  final plan entropy       "
              f"{float(-(self.plan_probs * np.log(self.plan_probs + 1e-12)).sum(1).mean()):.4f}"
              f"   (uniform = {float(np.log(self.n_actions)):.4f})")
        if self.n_plans == 0:
            print("  *** NOTHING WAS EVER PLANNED: the run never got past the")
            print(f"  *** {self.warmup_days}-day warm-up, so this arm is URB's "
                  "own IQL.")
        elif self.n_wm_updates == 0:
            print("  *** THE WORLD MODEL WAS NEVER TRAINED, so the planner scored")
            print("  *** every route under a randomly initialised network.")
        elif spread < 1e-4:
            print("  *** EVERY SAMPLED JOINT ACTION SCORES THE SAME. The reward")
            print("  *** model is a constant, so MPPI's softmax is uniform and")
            print("  *** THE PLANNER IS CHOOSING ROUTES AT RANDOM -- while the")
            print("  *** loss curve above looks healthy. If the clipped fraction")
            print("  *** is near 1, that is the documented cause: URB's -1000")
            print("  *** reward is outside the two-hot support and every reward")
            print("  *** lands in the bottom bin (normalize_reward: false does")
            print("  *** exactly this on purpose).")
        print("=" * 74 + "\n", flush=True)


def add_args(parser):
    parser.add_argument('--arm', type=str, default=None, choices=list(M3W.ARMS),
                        help="m3w: SoftMoE dynamics + SparseMoE reward + MPPI "
                             "(default). mlp: the same planner over a plain MLP "
                             "world model, which isolates the mixture of "
                             "experts. greedy: the MoE model with an argmax "
                             "instead of MPPI, which isolates the planner.")
    parser.add_argument('--horizon', type=int, default=None,
                        help="planning horizon in DAYS (default 1: a URB day is "
                             "one step)")

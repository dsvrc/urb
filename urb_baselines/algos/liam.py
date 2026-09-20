"""LIAM -- Local Information Agent Modelling (NeurIPS 2021) on URB.

    Papoudakis, Christianos, Albrecht, "Agent Modelling under Partial
    Observability for Deep Reinforcement Learning", NeurIPS 2021.
    https://proceedings.neurips.cc/paper_files/paper/2021/file/
    a03caec56cd82478bf197475b48c05f9-Paper.pdf
    Code: https://github.com/uoe-agents/LIAM
          lb_foraging/models.py (Encoder / Decoder / PolicyNet)
          lb_foraging/agent.py  (class A2C: the two optimisers, eval_decoding)
          lb_foraging/storage.py (return standardisation)
          lb_foraging/run_tests.py (hyperparameters)

``paper/BASELINES.md`` B5, "Agent modelling (learning about the others from local
observations)": "Chosen: LIAM ... Scalable (encoder-decoder on the controlled
agent's own observations and actions), PyTorch." Its prediction is that agent
modelling "models *who the peers are* (their policies), which at sigma=0 is
stationary; it does not model *how much they matter*, which is what drifts."

The document assigns LIAM to the MAPDN host as a Tier-2 item. It is implemented
here because URB is the cell this repository runs and because the port is the
same size either way: LIAM's controlled agent uses only its OWN observations and
actions, which is exactly what a URB traveller has.

THE METHOD
--------------------------------------------------------------------------------
Three networks, two optimisers:

  ``Encoder``   LSTM over the controlled agent's own ``(o_t, a_{t-1})`` ->
                embedding ``m_t``.
  ``PolicyNet`` actor-critic on ``[o_t | m_t]``. The embedding is DETACHED here
                (``agent.py``: ``torch.cat((obs, embeddings.detach()), dim=-1)``),
                so the encoder receives NO policy gradient.
  ``Decoder``   ``m_t`` -> the modelled agents' observations (squared error) and
                their actions (cross-entropy). This reconstruction loss is the
                ONLY thing that trains the encoder.

That split is the method: the representation is shaped entirely by "can I
reconstruct what the others saw and did", and the policy is then given it as a
fixed feature.

THREE THINGS URB DECIDES
--------------------------------------------------------------------------------
1. **What the sequence is.** LIAM's LSTM runs over the steps of an episode. A URB
   episode is ONE step, so a per-episode LSTM would have nothing to encode. The
   sequence here is the sequence of DAYS and the hidden state is carried across
   them -- the same decision, for the same reason, as ``algos/rippo.py``.
2. **Who is modelled.** LIAM models 1-3 other agents in MPE and level-based
   foraging. URB has hundreds of travellers and only the machine agents have an
   observation at all, so the modelled set is the ``m`` nearest machine peers by
   departure time within the COUPLING graph -- route sets that share a link and
   trips that are co-present (``m = 3`` by default, matching LIAM's largest
   setting). Reconstructing all 87 peers would be a different method with a
   different name.

   It is deliberately NOT the same-OD set: measured on the seven networks URB
   ships, the median OD pair carries exactly ONE traveller, so a same-OD
   modelled set is almost entirely padding and the decoder would be
   reconstructing the agent's own observation three times over. The banner
   prints how many slots ended up padded, so this is checkable per city rather
   than assumed.
3. **Reward scale.** URB pays -travel_time. LIAM standardises returns with a
   running mean/variance (``storage.compute_returns`` + ``standardise_stream.py``)
   and that is reproduced verbatim, because a value head starting near zero
   cannot otherwise reach -1000 under gradient clipping.

WHAT IS NOT PORTED
--------------------------------------------------------------------------------
LIAM's ``backprop_embeddings`` flag (letting the policy gradient reach the
encoder) is exposed because the release exposes it, and defaults to ``False``,
the released default. GAE over a rollout is N/A: one terminal step makes
``returns = r`` and ``adv = r - V(s)`` for any gamma and lambda.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from urb_baselines.graph import build_from_ctx
from urb_baselines.host import BaselineAlgorithm

__all__ = ["LIAM", "add_args"]


class _Encoder(nn.Module):
    """models.Encoder: LSTM -> ReLU MLP -> embedding."""

    def __init__(self, in_dim, hidden, out_dim):
        super(_Encoder, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.embedding = nn.Linear(hidden, out_dim)

    def forward(self, x, hidden):
        h, hidden = self.lstm(x, hidden)
        return self.embedding(torch.relu(self.fc1(h))), hidden


class _Decoder(nn.Module):
    """models.Decoder: two separate heads, one per reconstruction target."""

    def __init__(self, in_dim, hidden, obs_out, act_out):
        super(_Decoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out1 = nn.Linear(hidden, obs_out)
        self.fc3 = nn.Linear(in_dim, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.out2 = nn.Linear(hidden, act_out)

    def forward(self, x):
        h1 = torch.relu(self.fc2(torch.relu(self.fc1(x))))
        h2 = torch.relu(self.fc4(torch.relu(self.fc3(x))))
        return self.out1(h1), F.softmax(self.out2(h2), dim=-1)


class _PolicyNet(nn.Module):
    """models.PolicyNet: a shared trunk with a policy head and a value head."""

    def __init__(self, in_dim, hidden, n_actions):
        super(_PolicyNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.fc2(torch.relu(self.fc1(x))))
        return F.softmax(self.policy(h), dim=-1), self.value(h)


class _RunningMeanStd(object):
    """standardise_stream.RunningMeanStd. ``shape=None`` is the scalar case."""

    def __init__(self, epsilon=1e-4, shape=None):
        z = 0.0 if shape is None else np.zeros(shape, dtype=np.float64)
        o = 1.0 if shape is None else np.ones(shape, dtype=np.float64)
        self.mean, self.var, self.count = z, o, float(epsilon)

    def update(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if np.ndim(self.mean) == 0:
            bm, bv, bc = float(arr.mean()), float(arr.var()), arr.size
        else:
            arr = arr.reshape(-1, np.shape(self.mean)[0])
            bm, bv, bc = arr.mean(0), arr.var(0), arr.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var, self.count = m2 / tot, tot

    def standardize(self, arr):
        return (np.asarray(arr, dtype=np.float64) - self.mean) \
            / np.sqrt(self.var + 1e-8)


class _Learner(object):
    """One controlled agent: encoder, decoder, actor-critic and day memory."""

    def __init__(self, obs_size, n_actions, n_modelled, device, hp, cfg):
        self.device = device
        self.hp = hp
        self.cfg = cfg
        self.n_actions = int(n_actions)
        self.n_modelled = int(n_modelled)
        self.deterministic = False
        hid = int(cfg["hidden_dim"])
        emb = int(cfg["embedding_dim"])

        self.actor_critic = _PolicyNet(obs_size + emb, hid, n_actions).to(device)
        self.encoder = _Encoder(obs_size + n_actions, hid, emb).to(device)
        self.decoder = _Decoder(emb, hid, n_modelled * obs_size,
                                n_modelled * n_actions).to(device)
        # agent.py: two optimisers, lr1 for the actor-critic and lr2 for the
        # encoder+decoder pair.
        self.opt1 = optim.Adam(self.actor_critic.parameters(), lr=hp["lr"])
        self.opt2 = optim.Adam(list(self.encoder.parameters())
                               + list(self.decoder.parameters()),
                               lr=float(cfg["encoder_lr"]))
        self.hidden = (torch.zeros(1, 1, hid, device=device),
                       torch.zeros(1, 1, hid, device=device))
        self.prev_action = np.zeros(n_actions, dtype=np.float32)
        self.rms = _RunningMeanStd()
        self.mem = []            # (obs, prev_onehot, action, reward, h0, c0)
        self.targets = []        # (modelled_obs, modelled_act)
        self.loss = []
        self.last = {"pg": 0.0, "v": 0.0, "rec_o": 0.0, "rec_a": 0.0}
        self._pending = None

    # ------------------------------------------------------------------ act
    def act(self, obs):
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        x = torch.as_tensor(np.concatenate([o, self.prev_action]),
                            device=self.device).view(1, 1, -1)
        h0 = (self.hidden[0].detach(), self.hidden[1].detach())
        with torch.no_grad():
            emb, hidden = self.encoder(x, h0)
            probs, _ = self.actor_critic(
                torch.cat([torch.as_tensor(o, device=self.device).view(1, 1, -1),
                           emb], dim=-1))
        p = probs.view(-1)
        a = (int(torch.argmax(p).item()) if self.deterministic
             else int(torch.distributions.Categorical(p).sample().item()))
        self._pending = (o, self.prev_action.copy(), a,
                         h0[0].cpu().numpy().reshape(-1),
                         h0[1].cpu().numpy().reshape(-1))
        self.hidden = (hidden[0].detach(), hidden[1].detach())
        self.prev_action = np.eye(self.n_actions, dtype=np.float32)[a]
        return a

    def push(self, reward):
        if self._pending is None:
            return
        o, pa, a, h0, c0 = self._pending
        self.mem.append((o, pa, a, float(reward), h0, c0))
        self._pending = None

    def set_targets(self, modelled_obs, modelled_act):
        self.targets.append((modelled_obs, modelled_act))

    # ------------------------------------------------------------------ learn
    def learn(self):
        T = min(len(self.mem), len(self.targets))
        if T < int(self.hp["batch_size"]):
            return
        dev = self.device
        obs = torch.as_tensor(np.asarray([m[0] for m in self.mem[:T]]), device=dev)
        pact = torch.as_tensor(np.asarray([m[1] for m in self.mem[:T]]), device=dev)
        act = torch.as_tensor(np.asarray([m[2] for m in self.mem[:T]],
                                         dtype=np.int64), device=dev)
        rew = np.asarray([m[3] for m in self.mem[:T]], dtype=np.float64)
        h0 = torch.as_tensor(np.asarray([self.mem[0][4]], dtype=np.float32),
                             device=dev).view(1, 1, -1)
        c0 = torch.as_tensor(np.asarray([self.mem[0][5]], dtype=np.float32),
                             device=dev).view(1, 1, -1)
        m_obs = torch.as_tensor(np.asarray([t[0] for t in self.targets[:T]]),
                                device=dev)
        m_act = torch.as_tensor(np.asarray([t[1] for t in self.targets[:T]]),
                                device=dev)
        self.mem, self.targets = self.mem[T:], self.targets[T:]

        # storage.compute_returns: one terminal step, so returns = r; the
        # standardiser is updated with the returns and they are normalised.
        self.rms.update(rew)
        ret = torch.as_tensor(
            ((rew - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
             ).astype(np.float32), device=dev).view(-1, 1)

        # the LSTM is replayed from the hidden state recorded at the first stored
        # day, so the embedding sequence is the one the agent actually had.
        x = torch.cat([obs, pact], dim=-1).view(T, 1, -1)
        emb, _ = self.encoder(x, (h0, c0))               # (T, 1, emb)
        emb_flat = emb.view(T, -1)

        pol_in = torch.cat(
            [obs, emb_flat if self.cfg["backprop_embeddings"] else emb_flat.detach()],
            dim=-1)
        probs, value = self.actor_critic(pol_in)
        log_prob = torch.log(probs.gather(1, act.view(-1, 1)) + 1e-20)
        entropy = -(probs * torch.log(probs + 1e-20)).sum(-1, keepdim=True)
        adv = (ret - value).detach()
        action_loss = -(adv * log_prob)
        value_loss = (ret - value).pow(2)
        loss1 = (value_loss + action_loss
                 - entropy * float(self.hp["entropy_coef"])).mean()

        # agent.py eval_decoding: 0.5 * ||o - o_hat||^2 summed over dims, and
        # -log(sum(p * onehot)) -- i.e. cross-entropy against the true action.
        out_o, out_a = self.decoder(emb_flat)
        rec_o = 0.5 * ((m_obs.view(T, -1) - out_o) ** 2).sum(-1)
        pa = out_a.view(T, self.n_modelled, self.n_actions)
        rec_a = -torch.log((pa * m_act.view(T, self.n_modelled,
                                            self.n_actions)).sum(-1) + 1e-20)
        loss2 = (rec_o + rec_a.sum(-1)).mean()

        self.opt1.zero_grad()
        self.opt2.zero_grad()
        loss1.backward(retain_graph=self.cfg["backprop_embeddings"])
        loss2.backward()
        g = float(self.cfg["max_grad_norm"])
        nn.utils.clip_grad_norm_(self.encoder.parameters(), g)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), g)
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), g)
        self.opt1.step()
        self.opt2.step()

        self.last = {"pg": float(action_loss.mean().item()),
                     "v": float(value_loss.mean().item()),
                     "rec_o": float(rec_o.mean().item()),
                     "rec_a": float(rec_a.mean().item())}
        self.loss.append(self.last["rec_o"] + self.last["rec_a"])


class LIAM(BaselineAlgorithm):
    name = "liam"
    needs_obs_trace = True

    def __init__(self, ctx):
        super(LIAM, self).__init__(ctx)
        P = ctx.params
        cfg = dict(self.cfg)
        cfg.setdefault("embedding_dim", 20)          # run_tests.py
        cfg.setdefault("hidden_dim", int(P["widths"][-1]))
        cfg.setdefault("encoder_lr", 7e-4)           # run_tests.py lr2
        cfg.setdefault("max_grad_norm", 0.5)         # run_tests.py
        cfg.setdefault("n_modelled", 3)
        cfg.setdefault("backprop_embeddings", False)  # run_tests.py default
        # NOT "od": on every URB network shipped with the benchmark the
        # median OD pair has exactly ONE traveller, so the same-OD modelled
        # set would be entirely padding and the decoder would be
        # reconstructing the agent's own observation three times.
        cfg.setdefault("graph", "overlap")
        if getattr(ctx.args, "n_modelled", None):
            cfg["n_modelled"] = int(ctx.args.n_modelled)
        self.cfg = cfg

        self.n_modelled = int(cfg["n_modelled"])
        self.graph = build_from_ctx(ctx, mode=str(cfg["graph"]),
                                    n_neighbors=self.n_modelled, tag="liam",
                                    verbose=False)
        # the modelled set, padded with the agent itself when the city does not
        # supply enough same-OD machine peers. A padded slot is a reconstruction
        # target the encoder already knows, so it contributes almost no gradient
        # -- which is honest, and the banner reports how often it happens.
        self.modelled = {}
        n_pad = 0
        for a in self.av_ids:
            nb = list(self.graph.neighbours(a))
            n_pad += max(0, self.n_modelled - len(nb))
            while len(nb) < self.n_modelled:
                nb.append(a)
            self.modelled[a] = nb[:self.n_modelled]

        hp = dict(batch_size=int(P["batch_size"]), lr=float(P["lr"]),
                  entropy_coef=float(P["entropy_coef"]))
        self.learners = {
            a: _Learner(self.obs_size, self.n_actions, self.n_modelled,
                        self.device, hp, cfg) for a in self.av_ids
        }
        self._obs_today = {}
        # Standardise the OBSERVATION reconstruction target, per dimension,
        # across the whole fleet. URB's observation is [start_time_in_seconds,
        # four counts]: the first coordinate is of order 1000 and CONSTANT for a
        # given traveller, so an unscaled squared error is ~1e6 and is minimised
        # by memorising three constants -- the decoder would never look at the
        # counts and the embedding would carry no model of anyone. LIAM's own
        # environments have roughly unit-scale observations, so this restores the
        # regime its loss was written for. The measured symptom without it:
        # rec_o = 6.5e6 while the action head sat at chance.
        self.tgt_rms = _RunningMeanStd(shape=self.obs_size)
        self.standardize_targets = bool(cfg.get("standardize_targets", True))

        self.banner("LIAM (NeurIPS 2021) -- agent modelling under partial obs.", [
            ("base learner", "A2C on [own obs | embedding]"),
            ("encoder", f"LSTM over DAYS of (o, prev action) -> "
                        f"{cfg['embedding_dim']}-dim embedding"),
            ("embedding gradient", "DETACHED into the policy"
                                   if not cfg["backprop_embeddings"]
                                   else "flows into the policy (ablation)"),
            ("decoder", f"embedding -> {self.n_modelled} peers' observations "
                        f"(MSE) and actions (CE)"),
            ("modelled set", f"{self.n_modelled} nearest machine peers by "
                             f"departure time in the '{self.graph.mode}' graph "
                             f"({n_pad} padded slots)"),
            ("optimisers", f"actor-critic lr={hp['lr']}, encoder+decoder "
                           f"lr={cfg['encoder_lr']} (run_tests.py lr2)"),
            ("returns", "standardised by a running mean/var "
                        "(standardise_stream.py)"),
            ("reconstruction target", "standardised per observation dimension"
                                      if cfg.get("standardize_targets", True)
                                      else "RAW (ablation)"),
            ("advantage", "r - V(s)  (one-step episode: GAE collapses exactly)"),
        ])

    # ------------------------------------------------------------------ hooks
    def begin_episode(self, day, phase):
        self._obs_today = {}

    def act(self, agent_id, obs):
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._obs_today[agent_id] = o
        return self.learners[agent_id].act(o)

    def push(self, agent_id, reward):
        self.learners[agent_id].push(reward)

    def end_episode(self, day, phase, info):
        """Build each agent's reconstruction targets from what its modelled
        peers actually observed and did today."""
        if not self._obs_today:
            return
        acts = info.actions
        eye = np.eye(self.n_actions, dtype=np.float32)
        if self.standardize_targets:
            self.tgt_rms.update(np.stack(list(self._obs_today.values())))
        for a, lr in self.learners.items():
            mo, ma = [], []
            for p in self.modelled[a]:
                o = self._obs_today.get(
                    p, np.zeros(self.obs_size, dtype=np.float32))
                mo.append(self.tgt_rms.standardize(o)
                          if self.standardize_targets else o)
                k = int(acts.get(p, 0))
                ma.append(eye[max(0, min(self.n_actions - 1, k))])
            lr.set_targets(np.stack(mo).astype(np.float32),
                           np.stack(ma).astype(np.float32))

    def learn(self, day):
        for lr in self.learners.values():
            lr.learn()

    def begin_test(self):
        self.deterministic = True
        for lr in self.learners.values():
            lr.actor_critic.eval()
            lr.encoder.eval()
            lr.deterministic = True

    def diagnostics(self):
        ls = list(self.learners.values())
        if not ls:
            return {}
        keys = ("pg", "v", "rec_o", "rec_a")
        return {k: float(np.mean([x.last[k] for x in ls])) for k in keys}

    def loss_records(self):
        rows = []
        for a, lr in self.learners.items():
            for it, v in enumerate(lr.loss, start=1):
                rows.append({"iteration": it, "agent_id": a, "loss": v})
        return rows

    def close(self):
        ls = list(self.learners.values())
        ro = float(np.mean([x.last["rec_o"] for x in ls])) if ls else float("nan")
        ra = float(np.mean([x.last["rec_a"] for x in ls])) if ls else float("nan")
        print("\n" + "=" * 74)
        print("[liam] AGENT-MODELLING REPORT")
        print(f"  modelled peers/agent    {self.n_modelled}")
        print(f"  final obs reconstruction (0.5*SSE)   {ro:.4f}")
        print(f"  final action reconstruction (CE)     {ra:.4f}")
        print(f"  chance-level action CE               {np.log(self.n_actions):.4f}")
        pad = sum(1 for a in self.av_ids
                  for p in self.modelled[a] if p == a)
        print(f"  padded modelled slots                {pad}/"
              f"{len(self.av_ids) * self.n_modelled}")
        if pad > 0.5 * len(self.av_ids) * self.n_modelled:
            print("  *** More than half the modelled slots are the agent ITSELF:")
            print("  *** this city does not supply enough peers under the chosen")
            print("  *** graph, so the decoder is mostly reconstructing what the")
            print("  *** encoder already has. Change `graph`, or report the arm")
            print("  *** as modelling fewer than n_modelled peers.")
        if ra >= np.log(self.n_actions) - 1e-3:
            print("  *** The decoder never beat chance on the peers' actions, so")
            print("  *** the embedding carries no model of them and this arm is")
            print("  *** A2C with a random feature appended. Report it as such.")
        print("=" * 74 + "\n", flush=True)


def add_args(parser):
    parser.add_argument('--n-modelled', type=int, default=None,
                        help="how many peers each agent models (LIAM's largest "
                             "setting is 3)")

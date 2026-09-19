"""R-IPPO -- the memory arm: URB's IPPO plus R-MAPPO's recurrent machinery.

    Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu,
    "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games",
    NeurIPS 2022 Datasets & Benchmarks.  https://arxiv.org/abs/2103.01955
    Code: https://github.com/marlbenchmark/on-policy
          onpolicy/algorithms/utils/rnn.py      (RNNLayer)
          onpolicy/utils/shared_buffer.py       (recurrent_generator)

``paper/BASELINES.md`` B2, "the cheapest reviewer objection: a history-conditioned
policy could infer the drift from its own past." Its chosen representative is
"GRU-MAPPO via BenchMARL's built-in GRU model (a configuration change, no code)",
with R-MAPPO named as the reference implementation. URB has no BenchMARL host and
neither ``ippo_torchrl.py`` nor ``mappo_torchrl.py`` exposes an RNN, so the
configuration change has to be written out -- which is this file. The prediction
is that memory does not recover, because the required quantity (the peers' next
actions through W) is not a function of the agent's own history.

WHAT IS TAKEN FROM R-MAPPO
--------------------------------------------------------------------------------
* ``RNNLayer``: one ``nn.GRU`` with orthogonal weights, zero biases and a
  ``LayerNorm`` on the output, applied between the observation encoder and the
  action head. Verbatim in structure.
* the chunked BPTT update: the collected sequence is cut into chunks of
  ``data_chunk_length`` consecutive steps, each chunk replayed from the hidden
  state recorded at its FIRST step, ``num_mini_batch`` chunk-minibatches per
  epoch. Verbatim in structure.

WHAT IS NOT, AND WHY
--------------------------------------------------------------------------------
R-MAPPO also has a centralised critic and GAE. Adding them here would confound
"memory helps" with "a centralised critic helps", and B2's question is only the
first. So the learner is URB's own actor-only PPO -- same clip, same entropy
coefficient, same advantage (the standardised reward), same batch trigger -- with
the GRU inserted and nothing else changed. The centralised-critic question is
answered by its own arm, ``scripts/happo.py``.

THE ONE THING URB CHANGES, AND IT IS THE POINT
--------------------------------------------------------------------------------
In R-MAPPO the recurrence runs over the steps of an episode and the ``mask``
zeroes the hidden state at every episode boundary. A URB episode is ONE step, so
that mask would erase the memory every single day and the recurrent arm would be
numerically identical to the feed-forward one. The sequence that matters here is
the sequence of DAYS, so the hidden state is carried across days and across the
train/test boundary, and is reset only at construction. A memory baseline whose
memory is one step long would answer no objection at all.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from urb_baselines.host import BaselineAlgorithm
from urb_baselines.nets import GRUCore, MLP

__all__ = ["RecurrentIPPO", "RecurrentActor", "add_args"]


class RecurrentActor(nn.Module):
    """encoder MLP -> GRU (+LayerNorm) -> logits. R-MAPPO's actor, minus the critic."""

    def __init__(self, obs_size, n_actions, num_hidden, widths):
        super(RecurrentActor, self).__init__()
        # the host MLP with its output head removed: the last hidden width is the
        # GRU's input, so the parameter count matches scripts/ippo.py's actor up
        # to the recurrent block itself.
        self.encoder = MLP(obs_size, widths[-1], num_hidden, list(widths))
        self.rnn = GRUCore(widths[-1], widths[-1])
        self.head = nn.Linear(widths[-1], n_actions)
        self.hidden_size = widths[-1]

    def forward(self, x, h):
        """``x``: (T, B, obs); ``h``: (1, B, hidden) -> logits (T, B, K), h'."""
        T, B = x.shape[0], x.shape[1]
        e = torch.relu(self.encoder(x.reshape(T * B, -1))).reshape(T, B, -1)
        out, h_new = self.rnn(e, h)
        return self.head(out), h_new


class _Learner(object):
    """One agent's recurrent actor, day memory and chunked PPO update."""

    def __init__(self, obs_size, n_actions, device, hp, cfg):
        self.device = device
        self.hp = hp
        self.cfg = cfg
        self.n_actions = int(n_actions)
        self.deterministic = False

        self.actor = RecurrentActor(obs_size, n_actions, hp["num_hidden"],
                                    hp["widths"]).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=hp["lr"])
        self.h = self.actor.rnn.zero_state(1, device)      # carried across DAYS

        self.m_obs, self.m_act, self.m_logp, self.m_rew, self.m_h = [], [], [], [], []
        self.loss = []
        self._pending = None

    # ------------------------------------------------------------------ act
    def act(self, obs):
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                            device=self.device).view(1, 1, -1)
        h_in = self.h
        with torch.no_grad():
            logits, h_new = self.actor(x, h_in)
            probs = F.softmax(logits.view(-1), dim=-1)
        if self.deterministic:
            a = int(torch.argmax(probs).item())
        else:
            a = int(torch.distributions.Categorical(probs).sample().item())
        lp = float(torch.log(probs[a] + 1e-20).item())
        self.h = h_new.detach()
        self._pending = (np.asarray(obs, dtype=np.float32), a, lp,
                         h_in.detach().cpu().numpy().reshape(-1))
        return a

    def push(self, reward):
        if self._pending is None:
            return
        o, a, lp, h = self._pending
        self.m_obs.append(o)
        self.m_act.append(a)
        self.m_logp.append(lp)
        self.m_rew.append(float(reward))
        self.m_h.append(h)
        self._pending = None

    # ------------------------------------------------------------------ learn
    def learn(self):
        """R-MAPPO's chunked update on URB's actor-only PPO objective.

        The trigger is the host's: update once ``batch_size`` days have
        accumulated, exactly as ``scripts/ippo.py`` does, so the recurrent arm
        updates on the same amount of fresh data as the feed-forward one.
        """
        T = len(self.m_obs)
        L = int(self.cfg.get("data_chunk_length", 10))
        if T < self.hp["batch_size"] or T < L:
            return
        n_chunks = T // L                    # MAPPO drops the ragged tail
        n_mb = max(1, int(self.cfg.get("num_mini_batch", 1)))

        obs = torch.as_tensor(np.asarray(self.m_obs, dtype=np.float32),
                              device=self.device)
        act = torch.as_tensor(np.asarray(self.m_act, dtype=np.int64),
                              device=self.device)
        old_lp = torch.as_tensor(np.asarray(self.m_logp, dtype=np.float32),
                                 device=self.device)
        rew = torch.as_tensor(np.asarray(self.m_rew, dtype=np.float32),
                              device=self.device)
        hin = torch.as_tensor(np.asarray(self.m_h, dtype=np.float32),
                              device=self.device)
        self.m_obs, self.m_act, self.m_logp, self.m_rew, self.m_h = [], [], [], [], []

        # URB's IPPO advantage: the standardised reward. There is no critic in
        # the host's on-policy learner and this arm does not add one.
        adv_all = ((rew - rew.mean()) / (rew.std() + 1e-8)
                   if self.hp["normalize_advantage"] else rew)

        step_loss = []
        for _ in range(int(self.hp["num_epochs"])):
            order = torch.randperm(n_chunks, device=self.device)
            per = max(1, n_chunks // n_mb)
            for m in range(n_mb):
                idx = order[m * per:(m + 1) * per]
                if len(idx) == 0:
                    continue
                starts = (idx * L)
                # (L, B, *) chunks, each replayed from the hidden state recorded
                # at its first step -- shared_buffer.recurrent_generator.
                ar = torch.arange(L, device=self.device)
                pos = starts.view(1, -1) + ar.view(-1, 1)      # (L, B)
                ob = obs[pos]                                  # (L, B, obs)
                ac = act[pos]
                olp = old_lp[pos]
                ad = adv_all[pos]
                h0 = hin[starts].unsqueeze(0).contiguous()     # (1, B, hidden)

                logits, _ = self.actor(ob, h0)
                log_pi = F.log_softmax(logits, dim=-1)
                new_lp = log_pi.gather(-1, ac.unsqueeze(-1)).squeeze(-1)
                ratio = torch.exp(new_lp - olp)
                surr1 = ratio * ad
                surr2 = torch.clamp(ratio, 1 - self.hp["clip_eps"],
                                    1 + self.hp["clip_eps"]) * ad
                entropy = -(log_pi * torch.exp(log_pi)).sum(-1).mean()
                loss = (-torch.min(surr1, surr2).mean()
                        - self.hp["entropy_coef"] * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                               max_norm=1.0)
                self.optimizer.step()
                step_loss.append(float(loss.item()))
        if step_loss:
            self.loss.append(sum(step_loss) / len(step_loss))


class RecurrentIPPO(BaselineAlgorithm):
    name = "rippo"

    def __init__(self, ctx):
        super(RecurrentIPPO, self).__init__(ctx)
        P = ctx.params
        hp = dict(batch_size=int(P["batch_size"]), lr=float(P["lr"]),
                  num_epochs=int(P["num_epochs"]), num_hidden=int(P["num_hidden"]),
                  widths=list(P["widths"]), clip_eps=float(P["clip_eps"]),
                  normalize_advantage=bool(P["normalize_advantage"]),
                  entropy_coef=float(P["entropy_coef"]))
        cfg = dict(self.cfg)
        cfg.setdefault("data_chunk_length", 10)      # MAPPO's default
        cfg.setdefault("num_mini_batch", 1)          # MAPPO's default
        if getattr(ctx.args, "data_chunk_length", None):
            cfg["data_chunk_length"] = int(ctx.args.data_chunk_length)
        self.cfg = cfg
        self.learners = {a: _Learner(self.obs_size, self.n_actions, self.device,
                                     hp, cfg) for a in self.av_ids}
        hid = hp["widths"][-1]
        self.banner("R-IPPO -- the memory arm (B2)", [
            ("learner", "URB's IPPO objective, unchanged"),
            ("memory", f"GRU({hid}) + LayerNorm between encoder and head"),
            ("recurrence over", "DAYS (a URB episode is one step; see the "
                                "module docstring)"),
            ("BPTT", f"chunks of {cfg['data_chunk_length']} days, "
                     f"{cfg['num_mini_batch']} minibatch(es), "
                     f"{hp['num_epochs']} epochs"),
            ("update trigger", f"{hp['batch_size']} days (host's)"),
            ("hidden reset", "never -- carried across days and phases"),
            ("agents", f"{self.n_av} independent learners"),
        ])

    def act(self, agent_id, obs):
        return self.learners[agent_id].act(obs)

    def push(self, agent_id, reward):
        self.learners[agent_id].push(reward)

    def learn(self, day):
        for lr in self.learners.values():
            lr.learn()

    def begin_test(self):
        self.deterministic = True
        for lr in self.learners.values():
            lr.actor.eval()
            lr.deterministic = True

    def diagnostics(self):
        hs = [float(np.abs(l.h.detach().cpu().numpy()).mean())
              for l in self.learners.values()]
        return {"|h|": float(np.mean(hs)) if hs else 0.0}

    def loss_records(self):
        rows = []
        for a, lr in self.learners.items():
            for it, v in enumerate(lr.loss, start=1):
                rows.append({"iteration": it, "agent_id": a, "loss": v})
        return rows


def add_args(parser):
    parser.add_argument('--data-chunk-length', type=int, default=None,
                        help="BPTT chunk length in days (R-MAPPO's default: 10)")

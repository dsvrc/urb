"""MF-Q -- Mean Field Q-learning (ICML 2018) on URB.

    Yang, Luo, Li, Zhou, Zhang, Wang,
    "Mean Field Multi-Agent Reinforcement Learning", ICML 2018.
    https://proceedings.mlr.press/v80/yang18d/yang18d.pdf
    Code: https://github.com/mlii/mfrl  (examples/battle_model/algo/q_learning.py
          class MFQ, algo/base.py ValueNet, main_MFQ_Ising.py boltzman_explore),
          PyTorch port https://github.com/deligentfool/mfrl_pytorch

``paper/BASELINES.md`` B4: "Our coupling is a weighted mean of neighbours'
exertions; mean-field RL is the classical way to handle exactly that." The
document's own recommendation is *not* to port it, on the grounds that the
information-matched blind arm already is a mean-field-style policy. That
recommendation is followed for the PAPER's accounting -- the information-matched
arm is still reported under that name -- and MF-Q is nevertheless implemented
here, because on URB it is one file on the host's own DQN and it removes the one
thing a reviewer can say for free: "you did not actually run mean-field RL."

Where it genuinely differs from the information-matched blind arm:
  * the mean action is a normalised distribution over the neighbourhood, not raw
    counts of the earlier same-OD travellers;
  * it covers travellers who depart LATER, which the URB observation cannot see;
  * it enters through a dedicated embedding branch rather than the trunk;
  * the neighbourhood includes the human travellers.

THE METHOD
--------------------------------------------------------------------------------
Q^j(s, a^j, abar^j) where ``abar^j = (1/|N(j)|) sum_{k in N(j)} onehot(a^k)`` is
the mean action of j's neighbourhood at the previous step. The action-value
network takes the mean action through its own embedding branch
(``prob_emb_linear`` in the release: ``Linear(K,64) -> ReLU -> Linear(64,32)``),
concatenated with the observation embedding.

WHY THE NEIGHBOURHOOD IS THE OD PAIR BY DEFAULT
--------------------------------------------------------------------------------
Averaging one-hot action vectors is only meaningful where the action index means
the same thing for everybody. On URB action ``k`` is "the k-th route OF MY OD
PAIR", so route 0 for OD A and route 0 for OD B are different roads and their
one-hot mean is not a quantity. The default neighbourhood is therefore the OD
pair (every traveller on it, humans included). ``field_scope: all`` reproduces
the release's group-wide mean and ``field_scope: graph`` uses the |B| = 3
neighbourhood of ``urb_baselines/graph.py`` -- DGN's Table 4 lists 3 for MFQ.

TWO THINGS URB REMOVES, AND THEY ARE THE HOST'S DOING
--------------------------------------------------------------------------------
* **the bootstrap.** MF-Q's target is ``r + gamma (1-done) v_MF(s')`` with the
  mean-field value ``v_MF(s') = sum_a pi(a|s',abar') Q_target(s',a,abar')``. A URB
  day is one step and terminates, so ``done`` is always 1 and the target is ``r``
  -- exactly what URB's own IQL uses. The target network and the soft update then
  have no role and are not built; both are recorded as N/A in the checklist
  rather than left in as decoration.
* **the Boltzmann temperature.** The paper's policy is ``softmax(Q / T)`` with a
  decaying T (``main_MFQ_Ising.py``). URB pays -travel_time, so Q is of order
  -1000 and a temperature tuned for O(1) rewards makes the policy either uniform
  or a point mass. Exploration therefore defaults to the HOST's epsilon-greedy
  schedule -- which is also what the released ``act`` effectively does, since it
  takes the argmax of the softmax. ``exploration: boltzmann`` restores the
  paper's rule with a configurable temperature, and the banner warns that the
  temperature must be on the scale of the Q differences, not of the paper's.
"""

import numpy as np
import torch
import torch.nn as nn

from urb_baselines.algos.reference import PerAgentDQNAlgorithm, SingleStepDQN
from urb_baselines.graph import NeighbourGraph
from urb_baselines.nets import MLP

__all__ = ["MFQ", "MeanFieldQNet", "add_args"]


class MeanFieldQNet(nn.Module):
    """``Q(o, abar)``: the host trunk on ``[obs_emb, prob_emb]``.

    ``prob_emb_linear`` is the release's branch verbatim
    (``Linear(K, 64) -> ReLU -> Linear(64, 32)``). The trunk is the host's MLP so
    that the capacity available to the observation matches every other arm.
    """

    def __init__(self, obs_size, n_actions, num_hidden, widths, mf_hidden=64,
                 mf_embed=32):
        super(MeanFieldQNet, self).__init__()
        self.obs_emb = nn.Linear(obs_size, widths[0])
        self.prob_emb = nn.Sequential(
            nn.Linear(n_actions, mf_hidden), nn.ReLU(),
            nn.Linear(mf_hidden, mf_embed))
        self.trunk = MLP(widths[0] + mf_embed, n_actions, num_hidden, list(widths))

    def forward(self, x):
        """``x`` is ``[obs | abar]``; the split is by the observation width.

        Taking one concatenated tensor keeps this a drop-in for the host's
        ``Network``, so ``SingleStepDQN`` needs no change at all.
        """
        o, p = x[..., :self.obs_emb.in_features], x[..., self.obs_emb.in_features:]
        h = torch.cat([torch.relu(self.obs_emb(o)), self.prob_emb(p)], dim=-1)
        return self.trunk(h)


class _MFDQN(SingleStepDQN):
    """The host's DQN with a Boltzmann option. Everything else is unchanged."""

    def __init__(self, *a, exploration="epsilon", temperature=1.0,
                 temp_decay=1.0, temp_min=1e-3, **kw):
        super(_MFDQN, self).__init__(*a, **kw)
        self.exploration = str(exploration)
        self.temperature = float(temperature)
        self.temp_decay = float(temp_decay)
        self.temp_min = float(temp_min)

    def act(self, state):
        if self.exploration != "boltzmann":
            return super(_MFDQN, self).act(state)
        t = torch.FloatTensor(np.asarray(state, dtype=np.float32)
                              ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_network(t).view(-1)
        if self.deterministic:
            action = int(torch.argmax(q).item())
        else:
            # main_MFQ_Ising.boltzman_explore: sample from softmax(Q / T).
            p = torch.softmax(q / max(self.temperature, 1e-8), dim=-1)
            action = int(torch.multinomial(p, 1).item())
        self.last_state = np.asarray(state, dtype=np.float32)
        self.last_action = action
        return action

    def decay_epsilon(self):
        super(_MFDQN, self).decay_epsilon()
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)


class MFQ(PerAgentDQNAlgorithm):
    name = "mfq"
    needs_records = True

    SCOPES = ("od", "all", "graph")

    def __init__(self, ctx):
        cfg = dict(ctx.algo_cfg)
        self.scope = str(getattr(ctx.args, "field_scope", None)
                         or cfg.get("field_scope", "od")).lower()
        if self.scope not in self.SCOPES:
            raise ValueError(f"[mfq] field_scope must be one of {self.SCOPES}")
        self.mf_hidden = int(cfg.get("mf_hidden", 64))
        self.mf_embed = int(cfg.get("mf_embed", 32))
        self.exploration = str(cfg.get("exploration", "epsilon"))
        self.temperature = float(cfg.get("temperature", 50.0))
        self.temp_decay = float(cfg.get("temperature_decay", 0.999))
        self.temp_min = float(cfg.get("temperature_min", 1.0))
        super(MFQ, self).__init__(ctx)

        self.graph = None
        if self.scope == "graph":
            self.graph = NeighbourGraph(
                ctx.agent_table, self.av_ids, mode=str(cfg.get("graph", "od")),
                n_neighbors=int(cfg.get("n_neighbors", 3)), verbose=False)

        # which travellers are in each agent's field, resolved once
        self.field = {}
        by_od = {}
        for a, rec in ctx.agent_table.items():
            by_od.setdefault(tuple(rec["od"]), []).append(a)
        everyone = list(ctx.agent_table.keys())
        for a in self.av_ids:
            if self.scope == "all":
                self.field[a] = everyone
            elif self.scope == "graph":
                self.field[a] = self.graph.neighbours(a) or [a]
            else:
                self.field[a] = by_od[tuple(ctx.agent_table[a]["od"])]

        # abar^j from the PREVIOUS day. Uniform until the first records arrive,
        # which is the release's zero-initialised former_act_prob in spirit and
        # is at least a valid distribution.
        self.abar = {a: np.full(self.n_actions, 1.0 / self.n_actions,
                                dtype=np.float32) for a in self.av_ids}
        self.n_field_updates = 0

        sizes = [len(v) for v in self.field.values()]
        self.banner("MF-Q (ICML 2018) -- mean field Q-learning", [
            ("base learner", "URB's IQL, unchanged (scripts/iql.py)"),
            ("Q", f"Q(o, abar): obs->Linear({self.host_hp['widths'][0]}) | "
                  f"abar->Linear({self.mf_hidden})-ReLU-Linear({self.mf_embed}) "
                  f"-> host MLP -> {self.n_actions}"),
            ("abar", "mean one-hot action of the neighbourhood, PREVIOUS day"),
            ("neighbourhood", f"{self.scope}  (sizes {min(sizes)}-{max(sizes)}, "
                              + ("machines AND humans)" if self.scope != "graph"
                                 else "machines only)")),
            ("target", "r  (one-step episode; no bootstrap, as in URB's IQL)"),
            ("target network", "N/A -- nothing to bootstrap"),
            ("exploration", self.exploration
             + (f"  T0={self.temperature} decay={self.temp_decay} "
                f"min={self.temp_min}" if self.exploration == "boltzmann"
                else "  (host schedule; see the module docstring)")),
        ])
        if self.exploration == "boltzmann":
            print("[mfq][NOTE] Boltzmann exploration is on. URB rewards are of "
                  "order -1000, so a temperature transplanted from the paper's "
                  "O(1) rewards gives either a uniform or a degenerate policy. "
                  "Check the 'T' diagnostic against the spread of Q.", flush=True)

    # ------------------------------------------------------------------ model
    def input_size(self):
        return self.obs_size + self.n_actions

    def make_model(self, agent_id):
        net = MeanFieldQNet(self.obs_size, self.n_actions,
                            self.host_hp["num_hidden"], self.host_hp["widths"],
                            self.mf_hidden, self.mf_embed)
        return _MFDQN(self.input_size(), self.n_actions, device=self.device,
                      net=net, exploration=self.exploration,
                      temperature=self.temperature, temp_decay=self.temp_decay,
                      temp_min=self.temp_min, **self.host_hp)

    def observation(self, agent_id, obs):
        return np.concatenate([np.asarray(obs, dtype=np.float32).reshape(-1),
                               self.abar[agent_id]])

    # ------------------------------------------------------------------ field
    def end_episode(self, day, phase, info):
        """Recompute abar from the day's EXECUTED actions.

        ``info.peer_acts`` comes from RouteRL's own per-day file and therefore
        includes the human travellers, which the URB observation never does.
        """
        acts = info.peer_acts or info.actions
        if not acts:
            return
        onehot = {}
        for a, k in acts.items():
            k = int(k)
            if 0 <= k < self.n_actions:
                v = np.zeros(self.n_actions, dtype=np.float32)
                v[k] = 1.0
                onehot[a] = v
        if not onehot:
            return
        for a in self.av_ids:
            vs = [onehot[p] for p in self.field[a] if p in onehot]
            if vs:
                self.abar[a] = np.mean(vs, axis=0).astype(np.float32)
        self.n_field_updates += 1

    def diagnostics(self):
        d = super(MFQ, self).diagnostics()
        m = next(iter(self.models.values()), None)
        d["field_upd"] = self.n_field_updates
        if m is not None and self.exploration == "boltzmann":
            d["T"] = float(m.temperature)
        # how far abar is from uniform: 0 means the field says nothing
        u = 1.0 / self.n_actions
        d["abar_dev"] = float(np.mean([np.abs(v - u).sum()
                                       for v in self.abar.values()]))
        return d

    def close(self):
        print("\n" + "=" * 74)
        print("[mfq] MEAN-FIELD REPORT")
        print(f"  days with a field update   {self.n_field_updates}")
        u = 1.0 / self.n_actions
        dev = float(np.mean([np.abs(v - u).sum() for v in self.abar.values()]))
        print(f"  mean |abar - uniform|_1    {dev:.4f}")
        if self.n_field_updates == 0:
            print("  *** THE MEAN ACTION WAS NEVER UPDATED: every Q was")
            print("  *** evaluated at the uniform distribution, so this arm is")
            print("  *** IQL with four constant inputs. Do NOT report it as MF-Q.")
        elif dev < 1e-6:
            print("  *** abar is exactly uniform everywhere, so the mean-field")
            print("  *** input carries no signal. Check field_scope.")
        print("=" * 74 + "\n", flush=True)


def add_args(parser):
    parser.add_argument('--field-scope', type=str, default=None,
                        choices=list(MFQ.SCOPES),
                        help="od (default: every traveller on the same OD pair -- "
                             "the only scope where a one-hot mean is meaningful), "
                             "all (the release's group-wide mean), graph (the "
                             "|B|=3 neighbourhood)")

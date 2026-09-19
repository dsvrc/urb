"""URB's own two bases, and the reference arms that verify the shared host.

``paper/BASELINES.md`` asks each baseline to be built on the base its own paper
uses: on-policy methods on URB's on-policy learner, off-policy methods on URB's
off-policy learner. Those two learners are ``scripts/ippo.py``'s ``PPO`` and
``scripts/iql.py``'s ``DQN``. They are reproduced here, unchanged in every
numerical detail, so that every baseline in this package literally subclasses or
composes the host's own code rather than a re-derivation of it.

The two things to know about both, because every adaptation in this package
turns on them:

* **No critic, no bootstrap.** A URB day is one step. ``PPO`` uses the
  (optionally standardised) reward itself as the advantage, and ``DQN`` uses
  ``target = reward``. That is URB's choice, not ours; a baseline that quietly
  added a value function or a discounted bootstrap would be solving a different
  problem from the arms it is compared against. Where a baseline's own
  definition REQUIRES a critic -- LCPO is an actor-critic method, HAPPO needs a
  centralised V -- the critic is part of that method and is declared as such in
  its checklist, together with the ablation that isolates it.

* **One decision per agent per day.** ``act`` is called once, ``push`` once, and
  the memory is indexed by day.

``IPPOReference`` and ``IQLReference`` are these learners driven through
``urb_baselines.host``. They exist so ``selftest.py`` gate H1 can assert that the
shared loop reproduces ``scripts/ippo.py`` / ``scripts/iql.py`` update for
update; they are not meant to be run as arms (URB already ships those).
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from urb_baselines.host import BaselineAlgorithm
from urb_baselines.nets import MLP

__all__ = ["SingleStepPPO", "SingleStepDQN", "PerAgentPPOAlgorithm",
           "PerAgentDQNAlgorithm", "IPPOReference", "IQLReference"]


# ==========================================================================
#  the on-policy base -- scripts/ippo.py, class PPO
# ==========================================================================
class SingleStepPPO(object):
    """Actor-only single-step PPO. Identical to ``scripts/ippo.py``'s ``PPO``.

    Kept deliberately verbatim, including the un-clipped ``advantage = reward``
    when ``normalize_advantage`` is off, so that any baseline that subclasses it
    differs from URB's IPPO by exactly the lines its own paper specifies.
    """

    def __init__(self, state_size, action_space_size, device="cpu", batch_size=16,
                 lr=0.003, num_epochs=4, num_hidden=2, widths=(32, 64, 32),
                 clip_eps=0.2, normalize_advantage=True, entropy_coef=0.3,
                 net=None):
        self.device = device
        self.state_size = int(state_size)
        self.action_space_size = int(action_space_size)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.clip_eps = float(clip_eps)
        self.normalize_advantage = bool(normalize_advantage)
        self.entropy_coef = float(entropy_coef)

        self.policy_net = (net if net is not None
                           else MLP(state_size, action_space_size, num_hidden,
                                    list(widths))).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=float(lr))
        self.softmax = nn.Softmax(dim=-1)

        self.loss = []
        self.memory = []
        self.deterministic = False
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None

    # ------------------------------------------------------------------ act
    def act(self, state):
        state_tensor = torch.FloatTensor(np.asarray(state, dtype=np.float32)
                                         ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            probs = self.softmax(logits)
        dist = torch.distributions.Categorical(probs)
        action = (torch.argmax(probs).item() if self.deterministic
                  else dist.sample().item())
        self.last_state = np.asarray(state, dtype=np.float32)
        self.last_action = int(action)
        # `dist` lives on self.device, so the action index must too: on CUDA a
        # bare torch.tensor(action) is created on the CPU and Categorical.log_prob
        # raises "Expected all tensors to be on the same device". This is the fix
        # scripts/ippo.py carries.
        self.last_log_prob = dist.log_prob(
            torch.tensor(action, device=self.device)).item()
        return int(action)

    # ------------------------------------------------------------------ store
    def push(self, reward):
        if self.last_state is None:
            return
        self.memory.append((self.last_state, self.last_action,
                            self.last_log_prob, float(reward)))
        self.last_state = self.last_action = self.last_log_prob = None

    # ------------------------------------------------------------------ learn
    def _advantage(self, rewards_tensor):
        if self.normalize_advantage:
            return ((rewards_tensor - rewards_tensor.mean())
                    / (rewards_tensor.std() + 1e-8))
        return rewards_tensor

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        step_loss = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
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

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                           max_norm=1.0)
            self.optimizer.step()
            step_loss.append(loss.item())

        self.loss.append(sum(step_loss) / len(step_loss))
        self.memory.clear()


# ==========================================================================
#  the off-policy base -- scripts/iql.py, class DQN
# ==========================================================================
class SingleStepDQN(object):
    """Single-step DQN. Identical to ``scripts/iql.py``'s ``DQN``.

    ``target_q_values = rewards`` -- there is nothing to bootstrap from in a
    one-step episode. Every off-policy baseline in this package inherits that
    convention; where the original method's target contains a ``gamma * max Q'``
    term, its checklist records the term as N/A and says why.
    """

    def __init__(self, state_size, action_space_size, device="cpu", eps_init=0.99,
                 eps_decay=0.998, eps_min=0.0, buffer_size=256, batch_size=16,
                 lr=0.003, num_epochs=1, num_hidden=2, widths=(32, 64, 32),
                 net=None):
        self.device = device
        self.state_size = int(state_size)
        self.action_space_size = int(action_space_size)
        self.epsilon = float(eps_init)
        self.eps_decay = float(eps_decay)
        self.eps_min = float(eps_min)
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)

        self.q_network = (net if net is not None
                          else MLP(state_size, action_space_size, num_hidden,
                                   list(widths))).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=float(lr))
        self.loss_fn = nn.MSELoss()

        self.loss = []
        self.deterministic = False
        self.last_state = None
        self.last_action = None

    def act(self, state):
        if (not self.deterministic) and np.random.rand() < self.epsilon:
            action = int(np.random.choice(self.action_space_size))
        else:
            state_tensor = torch.FloatTensor(np.asarray(state, dtype=np.float32)
                                             ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = int(torch.argmax(q_values).item())
        self.last_state = np.asarray(state, dtype=np.float32)
        self.last_action = action
        return action

    def push(self, reward):
        if self.last_state is None:
            return
        self.memory.append((self.last_state, self.last_action, float(reward)))
        self.last_state = self.last_action = None

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        step_loss = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards = zip(*batch)
            states_tensor = torch.FloatTensor(np.asarray(states,
                                              dtype=np.float32)).to(self.device)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

            current_q = self.q_network(states_tensor).gather(1, actions_tensor)
            target_q = rewards_tensor                     # single-step episode
            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            step_loss.append(loss.item())
        self.loss.append(sum(step_loss) / len(step_loss))
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# ==========================================================================
#  generic per-agent algorithms over those bases
# ==========================================================================
class PerAgentPPOAlgorithm(BaselineAlgorithm):
    """One independent ``SingleStepPPO`` per machine agent.

    Baselines subclass this and override the narrow thing their paper changes:
    ``make_model`` (a different network or extra inputs), ``observation`` (what
    the policy sees) or ``learn`` (a different update).
    """

    name = "ippo-base"

    def __init__(self, ctx):
        super(PerAgentPPOAlgorithm, self).__init__(ctx)
        P = ctx.params
        self.host_hp = dict(
            batch_size=int(P["batch_size"]), lr=float(P["lr"]),
            num_epochs=int(P["num_epochs"]), num_hidden=int(P["num_hidden"]),
            widths=list(P["widths"]), clip_eps=float(P["clip_eps"]),
            normalize_advantage=bool(P["normalize_advantage"]),
            entropy_coef=float(P["entropy_coef"]),
        )
        self.models = {a: self.make_model(a) for a in self.av_ids}

    # -- overridable ------------------------------------------------------
    def input_size(self):
        return self.obs_size

    def make_model(self, agent_id):
        return SingleStepPPO(self.input_size(), self.n_actions,
                             device=self.device, **self.host_hp)

    def observation(self, agent_id, obs):
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    # -- hooks ------------------------------------------------------------
    def act(self, agent_id, obs):
        return self.models[agent_id].act(self.observation(agent_id, obs))

    def push(self, agent_id, reward):
        self.models[agent_id].push(reward)

    def learn(self, day):
        for m in self.models.values():
            m.learn()

    def begin_test(self):
        self.deterministic = True
        for m in self.models.values():
            m.policy_net.eval()
            m.deterministic = True

    def loss_records(self):
        rows = []
        for a, m in self.models.items():
            for it, v in enumerate(m.loss, start=1):
                rows.append({"iteration": it, "agent_id": a, "loss": v})
        return rows


class PerAgentDQNAlgorithm(BaselineAlgorithm):
    """One independent ``SingleStepDQN`` per machine agent."""

    name = "iql-base"

    def __init__(self, ctx):
        super(PerAgentDQNAlgorithm, self).__init__(ctx)
        P = ctx.params
        self.host_hp = dict(
            eps_init=float(P["eps_init"]), eps_decay=float(P["eps_decay"]),
            eps_min=float(P["eps_min"]), buffer_size=int(P["buffer_size"]),
            batch_size=int(P["batch_size"]), lr=float(P["lr"]),
            num_epochs=int(P["num_epochs"]), num_hidden=int(P["num_hidden"]),
            widths=list(P["widths"]),
        )
        self.models = {a: self.make_model(a) for a in self.av_ids}

    def input_size(self):
        return self.obs_size

    def make_model(self, agent_id):
        return SingleStepDQN(self.input_size(), self.n_actions,
                             device=self.device, **self.host_hp)

    def observation(self, agent_id, obs):
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def act(self, agent_id, obs):
        return self.models[agent_id].act(self.observation(agent_id, obs))

    def push(self, agent_id, reward):
        self.models[agent_id].push(reward)

    def learn(self, day):
        for m in self.models.values():
            m.learn()

    def begin_test(self):
        self.deterministic = True
        for m in self.models.values():
            m.q_network.eval()
            m.deterministic = True

    def diagnostics(self):
        any_model = next(iter(self.models.values()), None)
        return {} if any_model is None else {"eps": float(any_model.epsilon)}

    def loss_records(self):
        rows = []
        for a, m in self.models.items():
            for it, v in enumerate(m.loss, start=1):
                rows.append({"iteration": it, "agent_id": a, "loss": v})
        return rows


class IPPOReference(PerAgentPPOAlgorithm):
    """URB's IPPO, through the shared host. Parity target for selftest gate H1."""
    name = "ippo"


class IQLReference(PerAgentDQNAlgorithm):
    """URB's IQL, through the shared host. Parity target for selftest gate H1."""
    name = "iql"

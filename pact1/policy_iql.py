"""PACT-1 on a value-based host: URB's IQL with the steering applied to Q-values.

WHY A SECOND HOST
--------------------------------------------------------------------------------
A URB episode is one decision, so the Q-learning target reduces exactly to the
immediate reward and IQL becomes a low-variance regression of each route's cost
over a replay buffer. That is a better-specified estimator for this horizon than
a single-sample policy gradient, and it is why URB's IQL leads the learners. This
host asks whether the declared-basis prior still buys anything on top of it.

THE SHIFT, AND WHY Q IS STANDARDISED FIRST
--------------------------------------------------------------------------------
``core.steer_logits`` subtracts ``g * kappa * z`` where ``z`` is a z-score, so the
shift is dimensionless and ``kappa`` is one declared constant. Logits are already
dimensionless; Q-values here are travel times (order -3.2 with a cross-route
spread of order 0.3), so subtracting a term of order 1.7 would swamp the host
entirely. Standardising Q first puts both terms on the same scale:

    score_k = zscore(Q_k) - g * kappa * z_k ,    action = argmax_k score_k

THE FLOOR PROPERTY IS STILL EXACT
--------------------------------------------------------------------------------
``zscore`` is a positive affine map, so ``argmax zscore(Q) == argmax Q`` for any
Q. At ``g = 0`` ``steer_logits`` returns its input untouched, so the greedy action
is the untouched host's, bit for bit, for any ``beta_hat``. Asserted in
``_assert_floor`` at construction rather than argued.

TRUST IS FIXED, NOT LEARNED
--------------------------------------------------------------------------------
There is no policy gradient in a greedy-argmax host, so ``w`` has nothing to
learn from and ``trust_mode: learned`` is refused rather than silently ignored.
Use ``fixed`` (the arm that is already strongest on URB) or ``off``.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pact1.core import steer_logits


def _zscore(v):
    """Population z-score; all-equal input maps to zeros (never NaN)."""
    v = np.asarray(v, dtype=np.float64)
    sd = float(v.std())
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(v)
    return (v - v.mean()) / sd


class Pact1DQN(object):
    """URB's single-step DQN with a trust-weighted prior over routes.

    Mirrors ``scripts/iql.py::DQN`` exactly -- same buffer, same target
    (``target = reward``, which is the Bellman target when the next state is
    terminal), same epsilon schedule -- and adds only the steering shift.
    """

    def __init__(self, state_size, action_space_size, net_cls, coordinator,
                 agent_id, device="cpu", eps_init=1.0, eps_decay=0.9985,
                 eps_min=0.0, buffer_size=2048, batch_size=32, lr=0.001,
                 num_epochs=1, num_hidden=1, widths=(64, 64), kappa=1.0):
        self.device = device
        self.action_space_size = int(action_space_size)
        self.coord = coordinator
        self.agent_id = agent_id
        self.slot = coordinator.slot_of[agent_id]
        self.kappa = float(kappa)

        self.epsilon = float(eps_init)
        self.eps_decay = float(eps_decay)
        self.eps_min = float(eps_min)
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.deterministic = False

        if str(getattr(coordinator, "trust_mode", "fixed")).lower() == "learned":
            raise ValueError(
                "[PACT-1] trust_mode='learned' needs a policy gradient to reach the "
                "trust head; this host selects by argmax over Q and has none. Use "
                "--arm fixed (or 'off' for the blind arm) with --host iql."
            )

        self.policy_net = net_cls(state_size, self.action_space_size,
                                  num_hidden, list(widths)).to(device)
        self.q_network = self.policy_net                 # readable alias
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.loss = list()
        self._pending = None
        self._assert_floor()

    # ------------------------------------------------------------------ gate
    def _assert_floor(self, n=64, seed=0):
        """g = 0 must not change the greedy action, for ANY predicted costs."""
        rng = np.random.RandomState(seed)
        K = self.action_space_size
        for _ in range(n):
            q = rng.randn(K) * rng.choice([0.01, 1.0, 100.0]) - 3.2
            tt = rng.randn(K) * 8 + 120
            base = int(np.argmax(q))
            got = int(np.argmax(steer_logits(_zscore(q), tt, 0.0, self.kappa)))
            if got != base:
                raise AssertionError(
                    f"[PACT-1] floor broken on the Q host: g=0 changed the action "
                    f"({base} -> {got}). Refusing to run.")

    # ------------------------------------------------------------------ act
    def act(self, state, ctx=None):
        eps = 0.0 if self.deterministic else self.epsilon
        if eps > 0.0 and np.random.rand() < eps:
            action = int(np.random.choice(self.action_space_size))
            g_pol = g_app = 0.0
            shift_abs = 0.0
        else:
            st = torch.FloatTensor(np.asarray(state, dtype=np.float32))
            st = st.unsqueeze(0).to(self.device)
            with torch.no_grad():
                q = self.policy_net(st).cpu().numpy()[0].astype(np.float64)
            base = _zscore(q)
            if ctx is None:
                g_pol = g_app = 0.0
                score = base
            else:
                g_pol, g_app = self.coord.trust_for(self.slot, 0.0)
                score = steer_logits(base, np.asarray(ctx.tt_hat, dtype=np.float64),
                                     g_app, self.kappa,
                                     valid=np.asarray(ctx.valid, dtype=bool))
            shift_abs = float(np.max(np.abs(score - base))) if len(base) else 0.0
            action = int(np.argmax(score))

        if ctx is not None:
            self.coord.record_trust(self.slot, g_pol, g_app, shift_abs)
        self._pending = (np.asarray(state, dtype=np.float32), action)
        return action

    # ------------------------------------------------------------------ learn
    def push(self, reward):
        if self._pending is None:
            return
        s, a = self._pending
        self.memory.append((s, a, float(reward)))
        self._pending = None

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        step_loss = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards = zip(*batch)
            s = torch.FloatTensor(np.stack(states)).to(self.device)
            a = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            # target = reward: the next state is terminal, so the bootstrap term
            # of the Bellman target is zero by definition at this horizon.
            loss = self.loss_fn(self.policy_net(s).gather(1, a), r)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            step_loss.append(float(loss.item()))
        self.loss.append(sum(step_loss) / len(step_loss))
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

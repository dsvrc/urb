"""URB's single-step PPO + ONE trust head. The PPO update itself is unchanged.

THE HOST ALGORITHM IS NOT MODIFIED (guide III.4)
--------------------------------------------------------------------------------
``Pact1PPO`` is ``scripts/ippo.py``'s PPO with two differences and no third:

  1. the policy network emits ``K + 1`` numbers instead of ``K``: the K route
     logits, and one extra scalar ``w`` -- the trust control.
  2. before the softmax, the logits are shifted by the estimator's prediction,
     scaled by trust:

         g        = g_max * sigmoid(w + bias)   (EMA-smoothed), gated by conf(P)
         z_k      = zscore over the agent's routes of the predicted travel time
         logits_k = logits_k - g * kappa * z_k

Everything else -- the clipped surrogate, the entropy bonus, the
reward-as-advantage normalisation, the optimiser -- is byte-for-byte the URB
baseline, so an arm difference cannot be an algorithm difference.

WHY NO EXTRA LOG-PROB TERM IS NEEDED (and why this is cleaner than Ant's version)
--------------------------------------------------------------------------------
Ant appended trust as a sampled action dimension, which PPO then had to model.
Here trust is DETERMINISTIC given the state, and the route log-probability already
depends on it through the shift:

    log pi(k | s) = log softmax( logits(s) - g(w(s)) * kappa * z )_k

so the ordinary policy gradient flows into ``w`` with no extra sampling, no extra
log-prob, and no change to the PPO objective. Trust is learned from the return, as
designed, at zero cost to the host.

THE FLOOR PROPERTY IS EXACT
--------------------------------------------------------------------------------
At ``g == 0`` the shift term is identically zero, so the distribution is the
unmodified policy's -- bit for bit, for ANY ``beta_hat``, however wrong. The
estimator sits entirely outside the worst-case decision path: it can fail to help,
it cannot drag this arm below plain IPPO. ``check_shift_parity`` gates it, together
with numpy/torch agreement, before a single episode is simulated.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================================================
#  The shift (torch). MUST stay numerically identical to core.steer_logits.
# ==========================================================================
def apply_shift(logits, tt_hat, g, kappa, valid=None):
    """Trust-weighted model-based prior over routes. Batched, differentiable in
    ``g`` (hence in the trust head), constant in ``tt_hat``.

    logits: (B, K)   tt_hat: (B, K)   g: (B, 1)   valid: (B, K) bool or None
    """
    tt = torch.nan_to_num(tt_hat, nan=0.0, posinf=0.0, neginf=0.0)
    ok = torch.isfinite(tt_hat)
    if valid is not None:
        ok = ok & valid.bool()
    okf = ok.to(tt.dtype)
    n = okf.sum(dim=1, keepdim=True)

    tt_m = tt * okf
    mean = tt_m.sum(dim=1, keepdim=True) / n.clamp(min=1.0)
    dev = (tt - mean) * okf
    var = (dev * dev).sum(dim=1, keepdim=True) / n.clamp(min=1.0)
    sd = torch.sqrt(var)

    usable = (n >= 2.0) & (sd > 1e-12)
    z = torch.where(
        ok & usable, dev / sd.clamp(min=1e-12), torch.zeros_like(dev)
    )
    return logits - g * float(kappa) * z


def check_shift_parity(kappa=1.0, n=256, seed=0, tol=1e-9):
    """GATE 2b: the torch shift used in training must equal the numpy shift that
    ``selftest`` gates, and ``g = 0`` must be an EXACT no-op in both.

    Returns (ok, max_abs_diff, floor_exact).
    """
    from pact1.core import steer_logits

    rng = np.random.RandomState(seed)
    K = 4
    worst, floor_ok = 0.0, True
    for _ in range(n):
        lg = rng.randn(K)
        tt = rng.randn(K) * 8.0 + 120.0
        g = float(rng.rand())
        if rng.rand() < 0.1:
            tt[:] = tt[0]                       # degenerate: std == 0
        if rng.rand() < 0.1:
            tt[rng.randint(K)] = np.nan         # masked route
        ref = steer_logits(lg, tt, g, kappa)
        got = apply_shift(
            torch.tensor(lg)[None, :], torch.tensor(tt)[None, :],
            torch.tensor([[g]], dtype=torch.float64), kappa,
        ).numpy()[0]
        worst = max(worst, float(np.max(np.abs(ref - got))))

        z = apply_shift(
            torch.tensor(lg)[None, :], torch.tensor(tt)[None, :],
            torch.zeros(1, 1, dtype=torch.float64), kappa,
        ).numpy()[0]
        if not np.array_equal(z, lg):
            floor_ok = False
    return (worst <= tol and floor_ok), worst, floor_ok


# ==========================================================================
class Pact1PPO(object):
    """Single-step PPO with a trust head. Mirrors ``scripts/ippo.py::PPO``."""

    def __init__(self, state_size, action_space_size, net_cls, coordinator,
                 agent_id, device="cpu", batch_size=16, lr=0.003, num_epochs=4,
                 num_hidden=2, widths=(32, 64, 32), clip_eps=0.2,
                 normalize_advantage=True, entropy_coef=0.3):
        self.device = device
        self.K = int(action_space_size)
        self.action_space_size = self.K
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip_eps = clip_eps
        self.normalize_advantage = normalize_advantage
        self.entropy_coef = entropy_coef

        self.coord = coordinator
        self.agent_id = str(agent_id)
        self.slot = coordinator.slot_of[self.agent_id]
        self.kappa = coordinator.kappa

        # K route logits + 1 trust control. The ONLY architectural change.
        self.policy_net = net_cls(
            state_size, self.K + 1, num_hidden, list(widths)
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.loss = list()
        self.memory = list()
        self.deterministic = False
        self._pending = None

    # ------------------------------------------------------------------ act
    def act(self, state, ctx=None):
        """Choose a route. ``ctx`` is the coordinator's RouteContext for this agent;
        with ``ctx=None`` the shift is skipped entirely (identical to plain IPPO)."""
        st = torch.as_tensor(
            np.asarray(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device)

        # The EMA state BEFORE this step. trust_for() advances it, so it must be
        # read first: learn() replays the shift from this frozen value, and reading
        # it afterwards would silently replay the wrong distribution.
        prev_g = float(self.coord.g_pol[self.slot])

        with torch.no_grad():
            out = self.policy_net(st)                       # (1, K+1)
            logits = out[:, : self.K]
            w = float(out[0, self.K].item())

            if ctx is None:
                g_pol = g_app = 0.0
                tt = torch.zeros_like(logits)
                valid = torch.ones_like(logits, dtype=torch.bool)
            else:
                g_pol, g_app = self.coord.trust_for(self.slot, w)
                tt = torch.as_tensor(
                    np.asarray(ctx.tt_hat, dtype=np.float32)
                ).unsqueeze(0).to(self.device)
                valid = torch.as_tensor(
                    np.asarray(ctx.valid, dtype=bool)
                ).unsqueeze(0).to(self.device)

            g_t = torch.full((1, 1), float(g_app), device=self.device,
                             dtype=logits.dtype)
            shifted = apply_shift(logits, tt, g_t, self.kappa, valid)
            probs = torch.softmax(shifted, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = (int(torch.argmax(probs, dim=-1).item()) if self.deterministic
                  else int(dist.sample().item()))
        log_prob = float(dist.log_prob(torch.tensor([action]).to(self.device)).item())

        if ctx is not None:
            self.coord.record_trust(
                self.slot, g_pol, g_app,
                float(torch.abs(shifted - logits).max().item()),
            )

        # Everything the PPO ratio needs to be recomputed EXACTLY at update time.
        # prev_g and conf are frozen at their values now, so learn() differentiates
        # only through w -- which is the whole point.
        self._pending = dict(
            state=np.asarray(state, dtype=np.float32),
            action=action,
            log_prob=log_prob,
            tt=(np.asarray(ctx.tt_hat, dtype=np.float32) if ctx is not None
                else np.zeros(self.K, dtype=np.float32)),
            valid=(np.asarray(ctx.valid, dtype=bool) if ctx is not None
                   else np.ones(self.K, dtype=bool)),
            prev_g=prev_g,
            conf=float(ctx.conf) if ctx is not None else 0.0,
            use_ctx=ctx is not None,
        )
        return action

    def push(self, reward):
        if self._pending is None:
            return
        p = self._pending
        p["reward"] = float(reward)
        self.memory.append(p)
        self._pending = None

    # ------------------------------------------------------------------ learn
    def _g_from_w(self, w, prev_g, conf, use_ctx):
        """Differentiable reconstruction of the applied trust.

        g_pol = ema*prev + (1-ema)*g_max*sigmoid(w + bias)      [EMA state frozen]
        g_app = g_pol * conf                                    [conf frozen]

        Frozen terms are exactly the values that were live when the action was
        taken, so the recomputed distribution matches the behaviour policy while
        remaining differentiable in w.
        """
        c = self.coord
        if c.trust_mode == "off":
            return torch.zeros_like(w)
        if c.trust_mode == "fixed":
            return torch.full_like(w, c.trust_fixed) * conf
        tgt = c.g_max * torch.sigmoid(torch.clamp(w + c.g_bias, -30.0, 30.0))
        g_pol = c.g_ema * prev_g + (1.0 - c.g_ema) * tgt
        return g_pol * conf * use_ctx

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        step_loss = list()

        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states = torch.as_tensor(
                np.stack([b["state"] for b in batch])
            ).float().to(self.device)
            actions = torch.as_tensor(
                np.array([b["action"] for b in batch])
            ).long().to(self.device)
            old_lp = torch.as_tensor(
                np.array([b["log_prob"] for b in batch])
            ).float().to(self.device)
            rewards = torch.as_tensor(
                np.array([b["reward"] for b in batch])
            ).float().to(self.device)
            tt = torch.as_tensor(
                np.stack([b["tt"] for b in batch])
            ).float().to(self.device)
            valid = torch.as_tensor(
                np.stack([b["valid"] for b in batch])
            ).bool().to(self.device)
            prev_g = torch.as_tensor(
                np.array([[b["prev_g"]] for b in batch])
            ).float().to(self.device)
            conf = torch.as_tensor(
                np.array([[b["conf"]] for b in batch])
            ).float().to(self.device)
            use_ctx = torch.as_tensor(
                np.array([[1.0 if b["use_ctx"] else 0.0] for b in batch])
            ).float().to(self.device)

            out = self.policy_net(states)
            logits = out[:, : self.K]
            g = self._g_from_w(out[:, self.K : self.K + 1], prev_g, conf, use_ctx)
            probs = torch.softmax(apply_shift(logits, tt, g, self.kappa, valid), dim=-1)

            dist = torch.distributions.Categorical(probs)
            new_lp = dist.log_prob(actions)

            ratio = torch.exp(new_lp - old_lp)
            if self.normalize_advantage:
                advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                advantage = rewards

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            entropy = dist.entropy().mean()
            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            step_loss.append(loss.item())

        self.loss.append(sum(step_loss) / len(step_loss))
        self.memory.clear()

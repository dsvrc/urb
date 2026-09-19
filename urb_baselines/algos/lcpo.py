"""LCPO -- Locally Constrained Policy Optimization (ICLR 2025) on URB.

    Hamadanian, Nasr-Esfahany, Schwarzkopf, Sen, Alizadeh,
    "Online Reinforcement Learning in Non-Stationary Context-Driven
    Environments", ICLR 2025.  https://openreview.net/pdf?id=l6QnSQizmN
    Code: https://github.com/pouyahmdn/LCPO
          windy-gym/agent/lcpo.py, windy-gym/agent/core_alg/core_lcpo.py,
          windy-gym/agent/core_alg/core_lcppo.py, */buffer/buffer_ood.py,
          windy-gym/agent/a2c.py (return normalisation, entropy tuning),
          windy-gym/run_config_phase1.py (the configuration actually run)

``paper/BASELINES.md`` B6 puts this on URB and calls it "the most direct recent
competitor: non-stationarity induced by an *exogenous observed context* -- which
is exactly our driver A(t) -- with a method designed for it", predicting that
"it prevents forgetting across wet/dry contexts but cannot cancel; knowing the
weather is not knowing the neighbours' load under it."

WHAT THE METHOD IS
--------------------------------------------------------------------------------
An on-policy actor-critic whose policy update is a TRPO step with TWO trust
regions:

  * an **in-distribution** limit ``kl_in`` on the states just collected -- the
    ordinary trust region, and
  * an **out-of-distribution** limit ``kl_out`` (1000x tighter) on *anchor*
    states drawn from a reservoir of everything the agent has ever seen.

The conjugate-gradient direction is computed against the **OOD** KL Hessian and
the step is scaled by ``sqrt(2 kl_out / v_out)``; the line search accepts only if
BOTH limits hold and the surrogate improved. So the update is free to move the
policy where the data is and is pinned where the data is not -- which is what
stops a wet-season update from destroying the dry-season policy.

When the reservoir cannot fill an anchor batch the method falls back to plain
policy gradient. That fallback is also this file's ``--arm a2c``: the same
critic, the same advantage, the same entropy term, with the local constraint
off. Running both isolates LCPO's contribution from its actor-critic base, which
URB's own IPPO does not have.

WHAT THE HOST FORCES
--------------------------------------------------------------------------------
A URB day is one step and terminates, so GAE collapses exactly: ``returns = r``
and ``adv = r - V(s)``. ``gamma`` and ``lambda`` are inert and are therefore not
configurable here -- exposing them would suggest they do something.

The context ``A(day)`` is appended to every observation, which is LCPO's stated
requirement ("the context process is observed"). The identical augmentation with
no LCPO machinery is ``scripts/oracle_ippo.py``, so the pair separates "told the
weather" from "told the weather AND constrained".

THREE PLACES WHERE A CONSTANT DID NOT TRANSFER, AND WHAT WAS DONE
--------------------------------------------------------------------------------
1. ``master_batch``. In LCPO this is an ALGORITHM hyperparameter, not a host one
   (``run_config_phase1.py`` gives LCPO 200 and TRPO 3200 in the same sweep), and
   it sets how long the "recent" window is. On URB it must be shorter than the
   context's coherence time or one window already contains the whole weather
   cycle and nothing can be out of distribution -- LCPO would then silently be
   A2C. The default is the declared rule ``max(8, period // 5)`` = 20 days for
   the stock 100-day cycle. Overridable; always printed.
2. ``lcpo_thresh``. The authors' filter threshold is in the units of THEIR
   context (a 6-D wind vector, ``--lcpo_thresh 1``), and URB's context is a
   rainfall intensity in [0, 1] whose squared distances never exceed 1. The
   default here is ``ood_dist="all"`` -- every reservoir anchor is used, which
   is exactly what ``param.py``'s own default (``l2``, ``thresh=-9``) does, needs
   no constant to transfer, and always engages. The filtered forms are
   implemented and selectable.
3. **Reward scale.** URB pays -travel_time, i.e. rewards of order -1000, and a
   value head starting near zero cannot reach that under gradient clipping. The
   authors' own answer is used: ``a2c.py`` divides rewards by
   ``sqrt(ret_rms.var + 1)`` from a running estimator over returns. Implemented
   verbatim. Host-style advantage standardisation is therefore NOT applied (the
   authors do not do it).

See ``docs/baselines/CHECKLIST_lcpo.md`` for the item-by-item audit.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from urb_baselines.host import BaselineAlgorithm
from urb_baselines.nets import MLP

__all__ = ["LCPO", "OutOfDSampler", "RunningMeanStd", "conjugate_gradients",
           "make_is_different", "add_args"]


# ==========================================================================
#  utils/rms.py  (stable-baselines3 RunningMeanStd, vendored by the authors)
# ==========================================================================
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(epsilon)

    def update(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            return
        self.update_from_moments(float(arr.mean()), float(arr.var()), arr.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, m_2 / tot, tot


# ==========================================================================
#  buffer/buffer_ood.py -- class OutOfDSampler
# ==========================================================================
class OutOfDSampler(object):
    """Reservoir over every observation, plus a FIFO window of the recent ones.

    * ``states_fifo`` is a RESERVOIR (uniform over the whole history once full),
      not a ring buffer. That is the point: the anchors have to represent
      contexts from long ago, and a ring buffer would hold only the recent past
      -- the distribution the in-distribution constraint already covers.
    * ``recent_states`` is a genuine ring of the last ``window`` observations and
      is the "base" the distance function compares against.
    * ``get`` resamples up to 5 times and returns ``[]`` rather than a short
      batch, so a caller that receives nothing falls back to plain policy
      gradient instead of constraining on two anchors.
    """

    def __init__(self, obs_len, window, capacity, distant):
        self.cap = max(1, int(capacity))
        self.win = max(1, int(window))
        self.obs_len = int(obs_len)
        self.is_distant = distant
        self.states_fifo = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.recent_states = np.zeros([self.win, self.obs_len], dtype=np.float32)
        self.num_samples_so_far = 0
        self.i_win = 0

    def get(self, rng, batch_size):
        if self.num_samples_so_far == 0:
            return []
        recents = self.recent_states[:min(self.num_samples_so_far, self.win)]
        alls = self.states_fifo[:min(self.num_samples_so_far, self.cap)]
        rets, resamples = [], 0
        while len(rets) < batch_size and resamples < 5:
            new_rets = alls[rng.choice(len(alls), size=batch_size)]
            rets.extend(new_rets[self.is_distant(new_rets, recents)])
            resamples += 1
        return rets[:batch_size] if len(rets) >= batch_size else []

    def add_many_exp(self, states, rng):
        states = np.asarray(states, dtype=np.float32)
        n = len(states)
        if n == 0:
            return
        if self.num_samples_so_far + n <= self.cap:
            self.states_fifo[self.num_samples_so_far:
                             self.num_samples_so_far + n] = states
        else:
            for i in range(n):
                if self.num_samples_so_far + i < self.cap:
                    self.states_fifo[self.num_samples_so_far + i] = states[i]
                else:
                    idx = rng.choice(self.num_samples_so_far + i + 1)
                    if idx < self.cap:
                        self.states_fifo[idx] = states[i]
        if self.i_win + n <= self.win:
            self.recent_states[self.i_win:self.i_win + n] = states
        else:
            head = self.win - self.i_win
            self.recent_states[self.i_win:] = states[:head]
            tail = states[head:]
            if len(tail) >= self.win:
                self.recent_states[:] = tail[-self.win:]
            elif len(tail):
                self.recent_states[:len(tail)] = tail
        self.i_win = (self.i_win + n) % self.win
        self.num_samples_so_far += n


# ==========================================================================
#  the OOD distance -- windy-gym/env/*.py, is_different
# ==========================================================================
def make_is_different(kind, thresh, ctx_slice, ridge=1e-8):
    """``is_different(data, base) -> bool[len(data)]``.

    ``"all"``    every reservoir anchor is used. This is what ``param.py``'s own
                 defaults (``lcpo_ood_type='l2'``, ``lcpo_thresh=-9``) do, since a
                 squared distance is never below -9, and it is the URB default:
                 no constant has to transfer and the constraint always engages.
    ``"l2"``     ``||ctx - mean(ctx_base)||^2 > thresh``. The authors' headline
                 configuration (``--lcpo_thresh 1``) in the units of their 6-D
                 wind context; URB's rainfall context lives in [0, 1] so a
                 threshold of 1 would accept nothing. Provided for completeness.
    ``"mahala"`` the straggler-mitigation form: a Mahalanobis log-likelihood of
                 the CONTEXT coordinates against the recent window, flagged below
                 ``thresh``. Scale-free, so it is the one that can transfer. Note
                 the authors divide by ``base.shape[-1]`` -- the FULL observation
                 width, not the context width -- and that is reproduced, because
                 the threshold means nothing without it.
    """
    kind = str(kind).lower()
    lo, hi = ctx_slice

    def _all(data, base):
        return np.ones(len(data), dtype=bool)

    def l2(data, base):
        mu = np.mean(base[:, lo:hi], axis=0)
        return ((data[:, lo:hi] - mu) ** 2).sum(axis=-1) > thresh

    def _mahala(data, base, cols):
        b = np.atleast_2d(np.asarray(base[:, cols[0]:cols[1]], dtype=np.float64))
        x = np.atleast_2d(np.asarray(data[:, cols[0]:cols[1]], dtype=np.float64))
        mu = b.mean(axis=0)
        cov = np.atleast_2d(np.cov(b, rowvar=False))
        # A ridge, because URB's context is one-dimensional and every day of the
        # dry half has A == 0 EXACTLY: a window inside the placebo regime has a
        # singular covariance and cholesky raises. With the ridge the statistic
        # is finite and a wet anchor is correctly enormous.
        cov = cov + ridge * np.eye(cov.shape[0])
        cen = x - mu
        try:
            lu = np.linalg.cholesky(cov)
            y = np.linalg.solve(lu, cen.T)
        except np.linalg.LinAlgError:
            return np.ones(len(x), dtype=bool)
        d = np.einsum("ij,ij->j", y, y)
        return (-d / base.shape[-1] / 2.0) < thresh

    def mahala(data, base):
        return _mahala(data, base, (lo, hi))

    def mahala_full(data, base):
        return _mahala(data, base, (0, base.shape[-1]))

    return {"all": _all, "l2": l2, "mahala": mahala,
            "mahala_full": mahala_full}[kind]


# ==========================================================================
#  TRPO machinery -- core_alg/core_trpo.py + core_alg/core_lcpo.py
# ==========================================================================
def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view(p.size()))
        i += n


def conjugate_gradients(hvp, b, nsteps=15, residual_tol=1e-10):
    x = torch.zeros_like(b)
    resid = b.clone()
    d = b.clone()
    rr = torch.dot(resid, resid)
    for _ in range(nsteps):
        q = hvp(d)
        denom = torch.dot(d, q)
        if abs(denom.item()) < 1e-20:
            break
        alpha = rr / denom
        x = x + alpha * d
        resid = resid - alpha * q
        new_rr = torch.dot(resid, resid)
        d = resid + (new_rr / rr) * d
        rr = new_rr
        if rr < residual_tol:
            break
    return x


def _get_qu(a, b, c):
    """Roots of ``a s^2 + b s + c`` -- the dual solve of ``core_lcpo.trpo_step``."""
    sqr = np.sqrt(max(b ** 2 - 4 * a * c, 0.0))
    return (-b + sqr) / 2 / a, (-b - sqr) / 2 / a


def linesearch(model, loss_func, kl_out_func, kl_in_func, max_kl_out, max_kl_in,
               x_old, fullstep, expected_improve_rate, max_backtracks=10,
               accept_ratio=0.1):
    """Backtracking line search that must satisfy BOTH trust regions."""
    fval = loss_func(True).data
    for stepfrac in .5 ** np.arange(max_backtracks):
        x_new = x_old + float(stepfrac) * fullstep
        set_flat_params(model, x_new)
        newfval = loss_func(True).data
        with torch.no_grad():
            new_kl_in = kl_in_func().mean()
            new_kl_out = kl_out_func().mean()
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * float(stepfrac)
        ratio = actual_improve / (expected_improve + 1e-20)
        if (float(ratio) > accept_ratio and float(actual_improve) > 0
                and float(new_kl_out) <= max_kl_out
                and float(new_kl_in) <= max_kl_in):
            return True, x_new
    return False, x_old


def trpo_step(model, get_loss, get_kl_out, get_kl_in, max_kl_out, max_kl_in,
              damping, solve_dual):
    """One LCPO step: direction from the OOD Hessian, both KLs enforced."""
    params = list(model.parameters())
    loss = get_loss()
    grads = torch.autograd.grad(loss, params)
    loss_grad = torch.cat([g.reshape(-1) for g in grads]).data

    def _hvp(kl_func):
        def f(v):
            kl = kl_func().mean()
            g1 = torch.autograd.grad(kl, params, create_graph=True)
            flat = torch.cat([g.reshape(-1) for g in g1])
            g2 = torch.autograd.grad((flat * v.detach()).sum(), params)
            return torch.cat([g.contiguous().reshape(-1)
                              for g in g2]).data + v * damping
        return f

    q_out = _hvp(get_kl_out)
    stepoutdir = conjugate_gradients(q_out, -loss_grad, 15)
    vout = -(loss_grad * stepoutdir).sum().item()
    if vout <= 0:
        # The reference has the same guard (``if vout == 0``) and also skips the
        # step. It fires when the local surrogate has no gradient -- a converged
        # or saturated policy -- and is reported separately from a line-search
        # rejection, because the two say completely different things about the
        # run: "there was nothing to do" versus "the trust region refused".
        return loss.item(), 0.0, "no_grad"
    lm = 0.0
    if solve_dual:
        q_in = _hvp(get_kl_in)
        stepindir = conjugate_gradients(q_in, -loss_grad, 15)
        vin = -(loss_grad * stepindir).sum().item()
        # Reproduced line for line from core_lcpo.trpo_step, including the
        # asymmetry in a_out[0] (`stepindir * q_out(stepoutdir)`, not
        # `q_out(stepindir)`). Off by default, as in the reference.
        a_in = [0.5 * vin, vout,
                0.5 * (stepoutdir * q_in(stepoutdir)).sum().item()]
        a_out = [0.5 * (stepindir * q_out(stepoutdir)).sum().item(), vin,
                 0.5 * vout]
        diff = np.array(a_in) / max_kl_in - np.array(a_out) / max_kl_out
        if abs(diff[0]) < 1e-20:
            fullstep = float(np.sqrt(2 * max_kl_out / vout)) * stepoutdir
        else:
            diff = diff / diff[0]
            if diff[1] ** 2 <= 4 * diff[2] or max(_get_qu(*diff)) <= 0:
                fullstep = float(np.sqrt(2 * max_kl_out / vout)) * stepoutdir
            else:
                s = max(_get_qu(*diff))
                l_out = float(np.sqrt(
                    max_kl_in / (a_in[0] * s ** 2 + a_in[1] * s + a_in[0])))
                fullstep = (l_out * float(s)) * stepindir + l_out * stepoutdir
                lm = float(s)
    else:
        fullstep = float(np.sqrt(2 * max_kl_out / vout)) * stepoutdir

    prev = get_flat_params(model)
    neggdotstepdir = (-loss_grad * fullstep).sum()
    ok, new_params = linesearch(model, get_loss, get_kl_out, get_kl_in,
                                max_kl_out, max_kl_in, prev, fullstep,
                                neggdotstepdir)
    set_flat_params(model, new_params)
    return loss.item(), lm, ("accepted" if ok else "rejected")


# ==========================================================================
#  per-agent learner
# ==========================================================================
class _Learner(object):
    """One agent's actor, critic, reservoir, return normaliser and update."""

    def __init__(self, obs_dim, n_actions, device, hp, cfg, seed, is_diff):
        self.device = device
        self.n_actions = int(n_actions)
        self.hp = hp
        self.cfg = cfg
        self.deterministic = False

        self.policy_net = MLP(obs_dim, n_actions, hp["num_hidden"],
                              hp["widths"]).to(device)
        self.value_net = MLP(obs_dim, 1, hp["num_hidden"], hp["widths"]).to(device)
        self.opt_p = optim.Adam(self.policy_net.parameters(), lr=hp["lr"])
        self.opt_v = optim.Adam(self.value_net.parameters(),
                                lr=float(cfg.get("value_lr") or hp["lr"]))

        self.master_batch = int(cfg["master_batch"])
        self.ood = OutOfDSampler(obs_dim, self.master_batch,
                                 int(cfg["ood_capacity"]), is_diff)
        self.rng = np.random.RandomState(int(seed))
        self.ret_rms = RunningMeanStd()

        # entropy: fixed host coefficient, or the authors' auto-tuned one
        self.entropy_factor = float(cfg.get("entropy_max", hp["entropy_coef"]))
        self.auto_target = float(cfg.get("auto_target_entropy") or 0.0)
        if self.auto_target > 0:
            self.entropy_max = float(cfg.get("entropy_max", 3e-2))
            self.entropy_factor = self.entropy_max
            self.log_entropy = torch.zeros(1, requires_grad=True, device=device)
            self.target_entropy = float(np.log(n_actions)) * self.auto_target
            self.opt_ent = optim.Adam([self.log_entropy],
                                      lr=float(cfg.get("ent_lr", 1e-3)),
                                      weight_decay=1e-4)
        else:
            self.entropy_factor = float(hp["entropy_coef"])

        self.obs_mem, self.act_mem, self.rew_mem = [], [], []
        self.loss = []
        self._pending = None
        self.n_updates = 0
        self.n_constrained = 0
        self.n_accepted = 0
        self.n_nograd = 0
        self.last_kl_in = 0.0
        self.last_kl_out = 0.0
        self.last_ood_n = 0

    # ------------------------------------------------------------------ act
    def act(self, obs):
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(self.policy_net(t), dim=-1)
        a = (int(torch.argmax(probs).item()) if self.deterministic
             else int(torch.distributions.Categorical(probs).sample().item()))
        self._pending = (np.asarray(obs, dtype=np.float32), a)
        return a

    def push(self, reward):
        if self._pending is None:
            return
        o, a = self._pending
        self.obs_mem.append(o)
        self.act_mem.append(a)
        self.rew_mem.append(float(reward))
        self._pending = None

    # ------------------------------------------------------------------ learn
    def learn(self, arm):
        if len(self.obs_mem) < self.master_batch:
            return
        obs = np.asarray(self.obs_mem, dtype=np.float32)
        acts = np.asarray(self.act_mem, dtype=np.int64)
        rews = np.asarray(self.rew_mem, dtype=np.float64)
        self.obs_mem, self.act_mem, self.rew_mem = [], [], []

        # a2c.py: ret_rms is updated with the DISCOUNTED RETURN, which on a
        # one-step episode is the reward itself; rewards are then divided by
        # sqrt(var + 1). This is the authors' entire answer to reward scale, and
        # URB pays -travel_time (order -1000), so it is load-bearing.
        self.ret_rms.update(rews)
        rews_n = rews / np.sqrt(self.ret_rms.var + 1.0)

        obs_t = torch.as_tensor(obs, device=self.device)
        act_t = torch.as_tensor(acts, device=self.device).unsqueeze(-1)
        rew_t = torch.as_tensor(rews_n.astype(np.float32), device=self.device)

        # One step, terminal: cumulative_rewards -> r, gae_advantage -> r - V(s),
        # exactly, for any gamma and lambda.
        values = self.value_net(obs_t).squeeze(-1)
        adv = (rew_t - values.detach()).unsqueeze(-1)

        self.ood.add_many_exp(obs, self.rng)
        anchors = self.ood.get(self.rng, len(obs))
        self.last_ood_n = len(anchors)
        self.n_updates += 1

        if len(anchors) > 0 and arm != "a2c":
            self.n_constrained += 1
            ood_t = torch.as_tensor(np.asarray(anchors, dtype=np.float32),
                                    device=self.device)
            if arm == "lcppo":
                pg = self._lcppo(obs_t, act_t, adv, ood_t)
            else:
                pg = self._locopo(obs_t, act_t, adv, ood_t)
        else:
            pg = self._policy_gradient(obs_t, act_t, adv)

        v_loss = F.mse_loss(self.value_net(obs_t).squeeze(-1), rew_t)
        self.opt_v.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1)
        self.opt_v.step()

        self._tune_entropy(obs_t)
        self.loss.append(float(pg))

    # ---- entropy ---------------------------------------------------------
    def _tune_entropy(self, obs_t):
        """core_pg.train_entropy -- only when auto_target_entropy > 0."""
        if self.auto_target <= 0:
            return
        with torch.no_grad():
            log_pi = F.log_softmax(self.policy_net(obs_t), dim=-1)
            pi = torch.exp(log_pi)
            gap = (pi * log_pi).sum(dim=-1) + self.target_entropy
        ent_loss = -(torch.exp(self.log_entropy) * self.entropy_max * gap).mean()
        self.opt_ent.zero_grad()
        ent_loss.backward()
        self.opt_ent.step()
        self.entropy_factor = float(torch.exp(self.log_entropy).item()
                                    * self.entropy_max)

    # ---- the three updates -----------------------------------------------
    def _policy_gradient(self, obs_t, act_t, adv):
        """core_pg.policy_gradient -- LCPO's own fallback, and ``--arm a2c``."""
        log_pi = F.log_softmax(self.policy_net(obs_t), dim=-1)
        log_pi_a = log_pi.gather(1, act_t)
        pi = torch.exp(log_pi)
        entropy = -(log_pi * pi).sum(dim=-1).mean()
        pg_loss = -(log_pi_a * adv).mean()
        loss = pg_loss - self.entropy_factor * entropy
        self.opt_p.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.opt_p.step()
        with torch.no_grad():
            new_log_pi = F.log_softmax(self.policy_net(obs_t), dim=-1)
            self.last_kl_in = float(
                (pi.detach() * (log_pi.detach() - new_log_pi)).sum(-1).mean())
        self.last_kl_out = 0.0
        return float(pg_loss.item())

    def _locopo(self, obs_t, act_t, adv, ood_t):
        """core_lcpo.locopo -- the paper's algorithm."""
        with torch.no_grad():
            log_pi_g0 = F.log_softmax(self.policy_net(ood_t), dim=-1)
            pi_g0 = torch.exp(log_pi_g0)
            log_pi_l0 = F.log_softmax(self.policy_net(obs_t), dim=-1)
            pi_l0 = torch.exp(log_pi_l0)
            log_pi_a0 = log_pi_l0.gather(1, act_t)
        ent_coef = self.entropy_factor

        def get_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                log_pi = F.log_softmax(self.policy_net(obs_t), dim=-1)
                log_pi_a = log_pi.gather(1, act_t)
                pi = torch.exp(log_pi)
                entropy = -(log_pi * pi).sum(dim=-1).mean()
                action_loss = -adv * torch.exp(log_pi_a - log_pi_a0)
                return action_loss.mean() - entropy * ent_coef

        def get_kl_out():
            log_pi = F.log_softmax(self.policy_net(ood_t), dim=-1)
            return (pi_g0 * (log_pi_g0 - log_pi)).sum(dim=-1)

        def get_kl_in():
            log_pi = F.log_softmax(self.policy_net(obs_t), dim=-1)
            return (pi_l0 * (log_pi_l0 - log_pi)).sum(dim=-1)

        loss, _lm, status = trpo_step(
            self.policy_net, get_loss, get_kl_out, get_kl_in,
            max_kl_out=float(self.cfg["trpo_kl_out"]),
            max_kl_in=float(self.cfg["trpo_kl_in"]),
            damping=float(self.cfg["trpo_damping"]),
            solve_dual=bool(self.cfg.get("trpo_dual", False)))
        self.n_accepted += int(status == "accepted")
        self.n_nograd += int(status == "no_grad")
        with torch.no_grad():
            self.last_kl_in = float(get_kl_in().mean())
            self.last_kl_out = float(get_kl_out().mean())
        return loss

    def _lcppo(self, obs_t, act_t, adv, ood_t):
        """core_lcppo.locoprpo -- the authors' PPO variant of the same idea.

        The out-of-distribution term is a soft cross-entropy anchor towards the
        OLD policy on the anchor states, weighted by ``kappa``, plus the early
        stop when the in-distribution KL exceeds ``1.5 * ppo_kl_in``.
        """
        n_iters = int(self.cfg.get("ppo_iters", 30))
        clip = float(self.cfg.get("ppo_clip", self.hp["clip_eps"]))
        kappa = float(self.cfg.get("lcppo_kappa", 1.0))
        kl_in_lim = float(self.cfg.get("ppo_kl_in", self.cfg["trpo_kl_in"]))
        ent_coef = self.entropy_factor
        with torch.no_grad():
            log_pi_l0 = F.log_softmax(self.policy_net(obs_t), dim=-1)
            pi_l0 = torch.exp(log_pi_l0)
            log_pi_a0 = log_pi_l0.gather(1, act_t)
            log_pi_g0 = F.log_softmax(self.policy_net(ood_t), dim=-1)
            pi_g0 = torch.exp(log_pi_g0)

        n = obs_t.shape[0]
        n_ood = ood_t.shape[0]
        perm = torch.randperm(n, device=self.device).repeat(3)
        # core_lcppo.locoprpo splits 3 shuffled passes over the batch into
        # ppo_iters minibatches. URB's master batch can be smaller than
        # ppo_iters, in which case there is less than one sample per minibatch;
        # the number of minibatches is then reduced rather than the reshape
        # failing.
        bs = max(1, len(perm) // n_iters)
        n_iters = max(1, min(n_iters, len(perm) // bs))
        perm = perm[:n_iters * bs].view(n_iters, bs)
        losses = []
        for i in range(n_iters):
            idx = perm[i]
            gidx = idx % n_ood
            memory_loss = -(pi_g0[gidx] * F.log_softmax(
                self.policy_net(ood_t[gidx]), dim=-1)).sum(-1).mean()
            log_pi = F.log_softmax(self.policy_net(obs_t[idx]), dim=-1)
            log_pi_a = log_pi.gather(1, act_t[idx])
            pi = torch.exp(log_pi)
            entropy = -(log_pi * pi).sum(dim=-1).mean()
            ratio = torch.exp(log_pi_a - log_pi_a0[idx])
            clipped = ratio.clamp(1 - clip, 1 + clip)
            pg_loss = -torch.min(clipped * adv[idx], ratio * adv[idx]).mean()
            loss = pg_loss - ent_coef * entropy + kappa * memory_loss

            with torch.no_grad():
                approx_kl = float(
                    (pi_l0[idx] * (log_pi_l0[idx] - log_pi)).sum(-1).mean())
            if approx_kl > 1.5 * kl_in_lim:
                break
            self.opt_p.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
            self.opt_p.step()
            losses.append(float(pg_loss.item()))

        self.n_accepted += int(bool(losses))
        with torch.no_grad():
            new_l = F.log_softmax(self.policy_net(obs_t), dim=-1)
            self.last_kl_in = float((pi_l0 * (log_pi_l0 - new_l)).sum(-1).mean())
            new_g = F.log_softmax(self.policy_net(ood_t), dim=-1)
            self.last_kl_out = float((pi_g0 * (log_pi_g0 - new_g)).sum(-1).mean())
        return float(np.mean(losses)) if losses else 0.0


# ==========================================================================
#  the algorithm
# ==========================================================================
class LCPO(BaselineAlgorithm):
    name = "lcpo"
    needs_context = True

    ARMS = ("lcpo", "lcppo", "a2c")

    def __init__(self, ctx):
        super(LCPO, self).__init__(ctx)
        arm = getattr(ctx.args, "arm", None) or str(self.cfg.get("arm", "lcpo"))
        if arm not in self.ARMS:
            raise ValueError(f"[lcpo] --arm must be one of {self.ARMS}, got {arm!r}")
        self.arm = arm
        self.context = ctx.context
        self.ctx_dim = self.context.dim
        self._ctx_now = np.zeros(self.ctx_dim, dtype=np.float32)

        P = ctx.params
        hp = dict(lr=float(P["lr"]), num_hidden=int(P["num_hidden"]),
                  widths=list(P["widths"]), clip_eps=float(P["clip_eps"]),
                  entropy_coef=float(P["entropy_coef"]))

        cfg = dict(self.cfg)
        # run_config_phase1.py, the LCPO continual configuration:
        #   --trpo_kl_in 1e-1 --trpo_kl_out 1e-4 --trpo_damping 1e-1
        #   --ood_subsample 100 --val_lr_rate 1e-3 --n_hid 64 64
        cfg.setdefault("trpo_kl_in", 1e-1)
        cfg.setdefault("trpo_kl_out", 1e-4)
        cfg.setdefault("trpo_damping", 1e-1)
        cfg.setdefault("trpo_dual", False)
        cfg.setdefault("ood_dist", "all")
        cfg.setdefault("lcpo_thresh", -9.0)
        cfg.setdefault("ood_subsample", 100)
        cfg.setdefault("auto_target_entropy", None)

        period = getattr(self.context.driver, "period", 100) \
            if not self.context.degenerate else 100
        cfg["master_batch"] = int(cfg.get("master_batch")
                                  or max(8, int(period) // 5))
        # ood_len = master_batch * num_epochs // ood_subsample (windy-gym/train.py),
        # i.e. one slot per collected transition, scaled down. Per agent on URB
        # the transitions are the training days.
        cfg["ood_capacity"] = int(cfg.get("ood_capacity") or max(
            cfg["master_batch"],
            int(ctx.training_eps) // max(1, int(cfg["ood_subsample"]))))
        self.cfg = cfg

        obs_dim = self.obs_size + self.ctx_dim
        # The context occupies the LAST ctx_dim coordinates; that is the slice
        # the OOD distance reads.
        is_diff = make_is_different(cfg["ood_dist"], float(cfg["lcpo_thresh"]),
                                    (self.obs_size, obs_dim),
                                    float(cfg.get("ood_cov_ridge", 1e-8)))
        self.learners = {
            a: _Learner(obs_dim, self.n_actions, self.device, hp, cfg,
                        seed=int(ctx.torch_seed) * 100003 + i, is_diff=is_diff)
            for i, a in enumerate(self.av_ids)
        }

        ent = ("auto, target %.2f x log K" % float(cfg["auto_target_entropy"])
               if cfg["auto_target_entropy"] else
               f"fixed {hp['entropy_coef']} (host)")
        self.banner("LCPO (ICLR 2025) -- locally constrained policy optimization", [
            ("arm", self.arm + {"lcpo": "   (the paper's TRPO form)",
                                "lcppo": "  (the authors' PPO variant)",
                                "a2c": "    (ABLATION: constraint off)"}[self.arm]),
            ("observation", f"{self.obs_size} host + {self.ctx_dim} context "
                            f"{list(self.context.names)} = {obs_dim}"),
            ("kl_in / kl_out", f"{cfg['trpo_kl_in']} / {cfg['trpo_kl_out']}"
                               f"   dual={bool(cfg['trpo_dual'])}"),
            ("OOD filter", f"{cfg['ood_dist']}  thresh={cfg['lcpo_thresh']}"
                           + ("   (every reservoir anchor used)"
                              if cfg["ood_dist"] == "all" else "")),
            ("reservoir", f"{cfg['ood_capacity']} obs/agent   window "
                          f"{cfg['master_batch']}"),
            ("master batch", f"{cfg['master_batch']} days "
                             f"(rule: max(8, period//5), period={period})"),
            ("reward scale", "r / sqrt(running Var(return) + 1)  [a2c.py]"),
            ("advantage", "r - V(s)  (one-step episode: GAE collapses exactly)"),
            ("entropy", ent),
            ("agents", f"{self.n_av} independent learners"),
        ])

    # ------------------------------------------------------------------ hooks
    def begin_episode(self, day, phase):
        self._ctx_now = np.asarray(self.context.observed(day), dtype=np.float32)

    def act(self, agent_id, obs):
        o = np.concatenate([np.asarray(obs, dtype=np.float32).reshape(-1),
                            self._ctx_now])
        return self.learners[agent_id].act(o)

    def push(self, agent_id, reward):
        self.learners[agent_id].push(reward)

    def learn(self, day):
        for lr in self.learners.values():
            lr.learn(self.arm)

    def begin_test(self):
        self.deterministic = True
        for lr in self.learners.values():
            lr.policy_net.eval()
            lr.deterministic = True

    def diagnostics(self):
        ls = list(self.learners.values())
        if not ls:
            return {}
        upd = sum(x.n_updates for x in ls)
        con = sum(x.n_constrained for x in ls)
        ng = sum(x.n_nograd for x in ls)
        tried = max(1, con - ng)
        return {
            "upd": upd,
            "ood_hit": (con / upd) if upd else 0.0,
            "accept": (sum(x.n_accepted for x in ls) / tried) if con else 0.0,
            "nograd": (ng / con) if con else 0.0,
            "kl_in": float(np.mean([x.last_kl_in for x in ls])),
            "kl_out": float(np.mean([x.last_kl_out for x in ls])),
            "ent": float(np.mean([x.entropy_factor for x in ls])),
        }

    def loss_records(self):
        rows = []
        for a, lr in self.learners.items():
            for it, v in enumerate(lr.loss, start=1):
                rows.append({"iteration": it, "agent_id": a, "loss": v})
        return rows

    def close(self):
        ls = list(self.learners.values())
        upd = sum(x.n_updates for x in ls)
        con = sum(x.n_constrained for x in ls)
        acc = sum(x.n_accepted for x in ls)
        ng = sum(x.n_nograd for x in ls)
        tried = max(0, con - ng)
        rate = (con / upd) if upd else 0.0
        print("\n" + "=" * 74)
        print("[lcpo] LOCAL-CONSTRAINT REPORT")
        print(f"  updates                 {upd}")
        print(f"  with OOD anchors        {con}  ({rate:.1%})")
        print(f"  no local gradient       {ng}"
              + (f"  ({ng / con:.1%} -- a converged or saturated policy, so "
                 f"there was nothing to step)" if con else ""))
        print(f"  line search accepted    {acc}"
              + (f"  ({acc / tried:.1%} of the {tried} that had one)"
                 if tried else ""))
        if self.arm == "a2c":
            print("  arm=a2c: the constraint was OFF by request. This row is")
            print("  LCPO's own no-context fallback, not LCPO.")
        elif rate < 0.05:
            print("  *** THE LOCAL CONSTRAINT ALMOST NEVER ENGAGED: this arm ran")
            print("  *** as plain policy gradient and must NOT be reported as")
            print("  *** LCPO. With ood_dist='all' that can only mean the")
            print("  *** reservoir never held master_batch anchors -- check")
            print("  *** training_eps against master_batch.")
        elif tried and acc == 0:
            print("  *** EVERY line search was rejected, so the constrained")
            print("  *** updates did nothing at all. Check trpo_damping and")
            print("  *** trpo_kl_out before reporting this row.")
        elif con and ng > 0.9 * con:
            print("  *** Nine out of ten updates had NO LOCAL GRADIENT. The")
            print("  *** policy saturated early; look at the entropy and at")
            print("  *** value_lr (the critic has to catch up with the reward")
            print("  *** scale before the advantage means anything).")
        print("=" * 74 + "\n", flush=True)


def add_args(parser):
    parser.add_argument('--arm', type=str, default=None, choices=list(LCPO.ARMS),
                        help="lcpo (paper: TRPO with dual trust regions), lcppo "
                             "(the authors' PPO variant), a2c (ABLATION: the same "
                             "actor-critic with the local constraint off)")

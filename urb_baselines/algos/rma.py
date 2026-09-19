"""RMA / UP-OSI -- the meta-RL / online system-identification arm (B8) on URB.

    Kumar, Fu, Pathak, Malik, "RMA: Rapid Motor Adaptation for Legged Robots",
    RSS 2021.  https://www.roboticsproceedings.org/rss17/p011.html
    Yu, Tan, Liu, Turk, "Preparing for the Unknown: Learning a Universal Policy
    with Online System Identification" (UP-OSI), RSS 2017.
    https://arxiv.org/abs/1702.02453

``paper/BASELINES.md`` B8 specifies this one as an IN-HOUSE implementation rather
than a port:

    "Chosen: an RMA/UP-OSI-style adaptation module implemented in-house (teacher
    policy conditioned on the true beta*(t)A(t) -- available inside the
    environment -- then a history encoder distilled to predict it)."

and calls it "the closest *methodological* neighbour of PACT-1: it also
conditions on an identified quantity, but the identifier is a learned history
encoder rather than an estimator on a declared basis", predicting that "it works
while excitation exists (training) and degrades in greedy play, and it never
returns to the host exactly (no floor property)."

THE TWO PHASES (RMA Section 4)
--------------------------------------------------------------------------------
Phase 1 -- the TEACHER. An environment-factor encoder ``mu`` maps the privileged
environment vector ``e_t`` to a latent ``z_t``, and the base policy
``pi(o_t, z_t)`` is trained end to end by RL. Here ``e_t`` is
``DriverContext.privileged(day)``: the rainfall and the realised capacity derate
of the network, which a traveller provably cannot observe (sigma is not
published, the per-link rain sensitivity belongs to the road network, and
``min_e g_e`` names a facility). It is known before anyone acts, so it is a
legitimate environment vector rather than hindsight.

Phase 2 -- the STUDENT. The base policy and ``mu`` are FROZEN. An adaptation
module ``phi`` predicts ``z_hat_t`` from the agent's own recent history, trained
by supervised regression ``||z_hat - z||^2``. RMA rolls out with ``z_hat`` while
doing this (on-policy student data); that is UP-OSI's DAgger iteration in the
same clothes and is the default here (``phase2_act: student``).

Test phase -- deployment with ``z_hat`` only. The privileged vector is never used
again, which is the property that makes this a system-identification baseline
rather than an oracle.

THE ONE ADAPTATION THAT MATTERS
--------------------------------------------------------------------------------
RMA's history is ``(x_{t-k:t-1}, a_{t-k:t-1})`` -- proprioceptive state and
action. That works because a legged robot's proprioception ENCODES how the plant
responded. URB's observation does not: it is a start time and a count of earlier
travellers, and contains no travel time at all. A history of observations and
actions therefore carries almost no information about the weather, and the arm
would be identifying from nothing. The realised reward (i.e. the negative travel
time) is the channel through which the drift is observable at all, so the history
is ``(o, onehot(a), r_standardised)``. ``history_features: "oa"`` drops the
reward and reproduces RMA's literal input -- run it as the ablation that shows
why the choice was made.

WHAT THIS ARM IS NOT
--------------------------------------------------------------------------------
It is not ``scripts/oracle_ippo.py``. That arm is given the OBSERVED driver
``A(t)`` forever. This one is given a strictly larger, unobservable quantity, but
only during phase 1, and has to recover it from its own history thereafter. At
``sigma = 0`` the privileged vector collapses to ``[A, 0, 0]`` because the dial is
provably off, and the two arms become the same thing; that is correct behaviour
under a dial that does nothing, and the banner says so.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from urb_baselines.algos.reference import PerAgentPPOAlgorithm, SingleStepPPO
from urb_baselines.nets import MLP

__all__ = ["RMA", "FactorEncoder", "AdaptationModule", "add_args"]


class FactorEncoder(nn.Module):
    """``mu``: the privileged environment vector -> the latent ``z``.

    RMA uses a 3-layer MLP (256, 128) to an 8-dimensional latent; the widths here
    are the host's so that no arm gets a bigger network than another by accident,
    and the latent dimension is RMA's 8.
    """

    def __init__(self, e_dim, z_dim, num_hidden, widths):
        super(FactorEncoder, self).__init__()
        self.net = MLP(e_dim, z_dim, num_hidden, list(widths))

    def forward(self, e):
        return self.net(e)


class AdaptationModule(nn.Module):
    """``phi``: the agent's last ``k`` days -> ``z_hat``.

    RMA's adaptation module is "a 3-layer 1-D CNN" over the recent history after
    a linear projection, then a flatten and a linear head. Reproduced in shape;
    the channel count follows the host width.
    """

    def __init__(self, feat_dim, k, z_dim, channels=32):
        super(AdaptationModule, self).__init__()
        self.k = int(k)
        self.proj = nn.Linear(feat_dim, channels)
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(channels * self.k, z_dim)

    def forward(self, h):
        """``h``: (B, k, feat) -> (B, z_dim)."""
        x = torch.relu(self.proj(h))               # (B, k, C)
        x = self.conv(x.transpose(1, 2))           # (B, C, k)
        return self.head(x.flatten(1))


class RMA(PerAgentPPOAlgorithm):
    name = "rma"
    needs_context = True

    def __init__(self, ctx):
        cfg = dict(ctx.algo_cfg)
        self.z_dim = int(cfg.get("z_dim", 8))                 # RMA's latent size
        self.k = int(cfg.get("history_len", 50))              # RMA's window
        self.hist_feats = str(getattr(ctx.args, "history_features", None)
                              or cfg.get("history_features", "oar"))
        if self.hist_feats not in ("oar", "oa"):
            raise ValueError("[rma] history_features must be 'oar' or 'oa'")
        pf = getattr(ctx.args, "phase1_frac", None)
        self.phase1_frac = float(pf if pf is not None
                                 else cfg.get("phase1_frac", 0.5))
        self.phase2_act = str(cfg.get("phase2_act", "student"))
        self._host_phase = "train"
        self.context = ctx.context
        self.e_dim = self.context.privileged_dim if hasattr(
            self.context, "privileged_dim") else 3
        super(RMA, self).__init__(ctx)

        P = ctx.params
        self.mu = FactorEncoder(self.e_dim, self.z_dim, int(P["num_hidden"]),
                                list(P["widths"])).to(self.device)
        self.mu_opt = optim.Adam(self.mu.parameters(), lr=float(P["lr"]))

        self.feat_dim = (self.obs_size + self.n_actions
                         + (1 if self.hist_feats == "oar" else 0))
        self.phi = AdaptationModule(self.feat_dim, self.k, self.z_dim,
                                    channels=int(cfg.get("phi_channels", 32))
                                    ).to(self.device)
        self.phi_opt = optim.Adam(self.phi.parameters(),
                                  lr=float(cfg.get("phi_lr", 1e-3)))

        self.phase1_days = int(round(self.phase1_frac * ctx.training_eps))
        self.phase = 1
        self.day = 0
        self.z_now = np.zeros(self.z_dim, dtype=np.float32)
        self.zhat = {a: np.zeros(self.z_dim, dtype=np.float32)
                     for a in self.av_ids}
        # per-agent circular history of (o, onehot(a), r)
        self.hist = {a: np.zeros((self.k, self.feat_dim), dtype=np.float32)
                     for a in self.av_ids}
        self.hist_n = {a: 0 for a in self.av_ids}
        self._pending = {}
        self.r_mean, self.r_var, self.r_count = 0.0, 1.0, 1e-4
        self.phi_loss = []
        self.n_phi_updates = 0
        self.last_phi = float("nan")
        self.buf_h, self.buf_z = [], []

        self.banner("RMA / UP-OSI -- teacher-student system identification (B8)", [
            ("base learner", "URB's IPPO, unchanged (scripts/ippo.py)"),
            ("privileged e_t", f"{self.e_dim}-dim: "
                               "[A, 1-mean_e g_e, 1-min_e g_e]"
                               + ("   *** all zero beyond A at sigma=0 ***"
                                  if self.context.sigma == 0 else "")),
            ("latent z", f"mu(e) -> {self.z_dim} dims (RMA's 8)"),
            ("policy input", f"{self.obs_size} obs + {self.z_dim} z = "
                             f"{self.input_size()}"),
            ("adaptation phi", f"Linear -> 3 x Conv1d -> Linear over the last "
                               f"{self.k} days of ({self.hist_feats})"),
            ("phase 1", f"days 0-{self.phase1_days - 1}: teacher, z = mu(e)"),
            ("phase 2", f"days {self.phase1_days}-{ctx.training_eps - 1}: "
                        f"policy+mu FROZEN, phi regressed to z, acting with "
                        f"{'z_hat' if self.phase2_act == 'student' else 'z'}"),
            ("test", "z_hat only -- the privileged vector is never used again"),
        ])

    # ------------------------------------------------------------------ input
    def input_size(self):
        return self.obs_size + self.z_dim

    def make_model(self, agent_id):
        return SingleStepPPO(self.input_size(), self.n_actions,
                             device=self.device, **self.host_hp)

    # ------------------------------------------------------------------ hooks
    def begin_episode(self, day, phase):
        self.day = int(day)
        self._host_phase = phase
        if phase == "train" and day >= self.phase1_days and self.phase == 1:
            self._enter_phase2()
        # The host calls this AFTER env.reset(), which is what makes the
        # severity layer resolve today's capacity state; reading it earlier
        # would give the teacher yesterday's `g` (urb_baselines/host.py).
        e = torch.as_tensor(
            np.asarray(self.context.privileged(day), dtype=np.float32),
            device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.z_now = self.mu(e).cpu().numpy().reshape(-1)
        if self.phase >= 2 or phase == "test":
            self._predict_z_all()
        self._pending = {}

    def _use_student(self):
        """Which latent the policy is fed right now.

        Always ``z_hat`` in the test phase -- that is the deployment RMA is
        judged on. During phase 2, ``z_hat`` as well by default: RMA trains the
        adaptation module on ON-POLICY data collected with ``z_hat``, which is
        also UP-OSI's DAgger iteration. ``phase2_act: teacher`` keeps the
        privileged latent during phase 2 for the ablation.
        """
        if self._host_phase == "test":
            return True
        return self.phase >= 2 and self.phase2_act == "student"

    def observation(self, agent_id, obs):
        z = self.zhat[agent_id] if self._use_student() else self.z_now
        return np.concatenate([np.asarray(obs, dtype=np.float32).reshape(-1),
                               np.asarray(z, dtype=np.float32)])

    def act(self, agent_id, obs):
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        a = self.models[agent_id].act(self.observation(agent_id, o))
        self._pending[agent_id] = (o, int(a))
        return a

    def push(self, agent_id, reward):
        super(RMA, self).push(agent_id, reward)
        rec = self._pending.pop(agent_id, None)
        if rec is None:
            return
        o, a = rec
        self._update_reward_stats(float(reward))
        feat = [o, np.eye(self.n_actions, dtype=np.float32)[a]]
        if self.hist_feats == "oar":
            r_n = (float(reward) - self.r_mean) / float(np.sqrt(self.r_var) + 1e-8)
            feat.append(np.array([r_n], dtype=np.float32))
        v = np.concatenate(feat).astype(np.float32)
        h = self.hist[agent_id]
        h[:-1] = h[1:]
        h[-1] = v
        self.hist_n[agent_id] = min(self.k, self.hist_n[agent_id] + 1)

    def end_episode(self, day, phase, info):
        """Collect one (history, z) regression pair per agent, for phase 2."""
        if self.phase < 2:
            return
        for a in self.av_ids:
            if self.hist_n[a] >= self.k:
                self.buf_h.append(self.hist[a].copy())
                self.buf_z.append(self.z_now.copy())

    def learn(self, day):
        if self.phase == 1:
            super(RMA, self).learn(day)
        else:
            # RMA phase 2: "we keep the base policy fixed and train only the
            # adaptation module." The PPO memories keep filling and are simply
            # dropped, so no stale on-policy data can leak into a later update.
            for m in self.models.values():
                m.memory.clear()
            self._train_phi()

    def begin_test(self):
        if self.phase < 2:
            print("[rma][WARN] the test phase was reached before phase 2 began "
                  "(phase1_frac too large, or training_eps too small). z_hat has "
                  "never been trained and this row is the TEACHER, not RMA.",
                  flush=True)
        self.deterministic = True
        for m in self.models.values():
            m.policy_net.eval()
            m.deterministic = True
        self.phi.eval()

    # ------------------------------------------------------------------ phases
    def _enter_phase2(self):
        self.phase = 2
        for m in self.models.values():
            for p in m.policy_net.parameters():
                p.requires_grad_(False)
            m.memory.clear()
        for p in self.mu.parameters():
            p.requires_grad_(False)
        print(f"\n[rma] ---- PHASE 2 at day {self.day}: base policy and mu "
              f"FROZEN; phi is regressed to z and the fleet acts on "
              f"{'z_hat' if self.phase2_act == 'student' else 'z'} ----\n",
              flush=True)

    def _update_reward_stats(self, r):
        self.r_count += 1
        d = r - self.r_mean
        self.r_mean += d / self.r_count
        self.r_var += (d * (r - self.r_mean) - self.r_var) / self.r_count

    def _predict_z_all(self):
        """One batched forward for the whole fleet, once per day.

        Per-agent forwards would be 88 x 4000 calls into a three-layer 1-D CNN,
        which is minutes of pure framework overhead for a quantity that is the
        same shape for everybody.
        """
        ready = [a for a in self.av_ids if self.hist_n[a] >= self.k]
        for a in self.av_ids:
            if self.hist_n[a] < self.k:
                self.zhat[a] = self.z_now.copy()      # not enough history yet
        if not ready:
            return
        h = torch.as_tensor(np.asarray([self.hist[a] for a in ready]),
                            device=self.device)
        with torch.no_grad():
            z = self.phi(h).cpu().numpy()
        for a, row in zip(ready, z):
            self.zhat[a] = row.reshape(-1)

    def _train_phi(self):
        """Supervised regression ``||phi(history) - mu(e)||^2`` (RMA Eq. 2)."""
        bs = int(self.host_hp["batch_size"])
        if len(self.buf_h) < bs:
            return
        for _ in range(int(self.host_hp["num_epochs"])):
            idx = np.random.randint(0, len(self.buf_h), size=bs)
            h = torch.as_tensor(np.asarray([self.buf_h[i] for i in idx]),
                                device=self.device)
            z = torch.as_tensor(np.asarray([self.buf_z[i] for i in idx]),
                                device=self.device)
            loss = F.mse_loss(self.phi(h), z)
            self.phi_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.phi.parameters(), 1.0)
            self.phi_opt.step()
        self.last_phi = float(loss.item())
        self.phi_loss.append(self.last_phi)
        self.n_phi_updates += 1
        cap = int(self.cfg.get("phi_buffer", 20000))
        if len(self.buf_h) > cap:
            self.buf_h = self.buf_h[-cap:]
            self.buf_z = self.buf_z[-cap:]

    # ------------------------------------------------------------------ report
    def diagnostics(self):
        d = {"phase": self.phase, "|z|": float(np.abs(self.z_now).mean())}
        if self.phase >= 2:
            d["phi_mse"] = self.last_phi
            zh = np.asarray([self.zhat[a] for a in self.av_ids])
            d["|zhat-z|"] = float(np.abs(zh - self.z_now).mean())
        return d

    def loss_records(self):
        rows = super(RMA, self).loss_records()
        rows += [{"iteration": i, "agent_id": "phi", "loss": v}
                 for i, v in enumerate(self.phi_loss, start=1)]
        return rows

    def close(self):
        zh = np.asarray([self.zhat[a] for a in self.av_ids])
        err = float(np.abs(zh - self.z_now).mean()) if len(zh) else float("nan")
        print("\n" + "=" * 74)
        print("[rma] ADAPTATION REPORT")
        print(f"  phase reached           {self.phase}")
        print(f"  phi updates             {self.n_phi_updates}")
        print(f"  final phi MSE           {self.last_phi}")
        print(f"  mean |z_hat - z| (last) {err:.5f}")
        print(f"  mean |z| (last)         {float(np.abs(self.z_now).mean()):.5f}")
        if self.phase < 2:
            print("  *** PHASE 2 NEVER STARTED: this row is the privileged")
            print("  *** TEACHER, not RMA. Do not report it as an adaptation arm.")
        elif self.n_phi_updates == 0:
            print("  *** phi WAS NEVER TRAINED. z_hat is a random projection and")
            print("  *** the test phase ran on noise.")
        if self.context.sigma == 0:
            print("  NOTE sigma=0: the dial is provably off, so e_t carries only")
            print("  A(t) and this arm is the oracle-driver arm with extra steps.")
            print("  That is the correct degenerate behaviour, not a bug.")
        print("=" * 74 + "\n", flush=True)


def add_args(parser):
    parser.add_argument('--phase1-frac', type=float, default=None,
                        help="fraction of the training days spent on the "
                             "privileged teacher before phi takes over "
                             "(default 0.5)")
    parser.add_argument('--history-features', type=str, default=None,
                        choices=["oar", "oa"],
                        help="oar (default): observation, action, standardised "
                             "reward. oa: RMA's literal (state, action) input -- "
                             "the ablation showing URB's observation carries no "
                             "response signal.")

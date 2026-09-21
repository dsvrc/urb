"""WISDOM -- wavelet predictive task representations for non-stationary RL.

    Wang, Li, He, Li, Bennis, Islam & Wang, "Wavelet Predictive Representations
    for Non-Stationary Reinforcement Learning", arXiv:2510.04507 (2025),
    code https://github.com/MinWangcs/WISDOM.

WHY THIS ARM EXISTS
--------------------------------------------------------------------------------
It is the closest *learned* competitor to what PACT-1 does by declaration. Both
arms believe the same thing -- that you cannot act well under drift unless you
carry an estimate of where the drift is going -- and they differ on exactly one
axis:

    PACT-1   a declared basis (road classes, shared per-lane length), an RLS
             estimator with forgetting, and a forecast that is an EMA over days.
    WISDOM   a learned latent task representation, decomposed into wavelet
             coefficients at several scales, with a TD operator that predicts
             the NEXT representation and an auto-regressive head on top.

So "you should have learned the coupling instead of declaring it" -- the
objection ``paper/BASELINES.md`` answers with the unstructured-RLS arm on the
estimator side -- gets its representation-learning answer here, from a method
published for this exact problem. ``urls`` asks whether the basis was necessary;
this asks whether the whole declared pipeline was.

WHAT THE THEORY PREDICTS
--------------------------------------------------------------------------------
C4 (excitation is bounded by policy indecision) and C6 (non-identifiability)
apply to a learned encoder just as they do to RLS, and nothing about a wavelet
basis relieves them: if the fleet converges, the data stops containing the
variation that would separate "the coupling got stronger" from "everyone moved".
The prediction is that WISDOM tracks the SMOOTH component of ``A(t)`` well --
that is exactly what a multi-scale decomposition is for, and the paper's own
result is that it beats flat representations on gradually-evolving tasks -- and
that it does not recover the per-route split, because that is not identifiable
from an agent's own reward stream at all.

THE FOUR ADAPTATIONS, EACH A ROW IN THE CHECKLIST
--------------------------------------------------------------------------------
1. **SAC -> the host's PPO.** The paper is SAC-based with continuous actions.
   URB is discrete and one step, so the base is URB's own on-policy learner
   (R2), and the representation enters exactly where the paper puts it: the
   policy is conditioned on ``[state, z_hat]``.

2. **The sequence is the sequence of DAYS.** The paper's task-representation
   sequence runs along a trajectory. A URB episode is one step, so a per-episode
   sequence has length one and every wavelet level would be empty. The sequence
   here is the agent's own history across days -- the same decision, for the
   same reason, that ``algos/rippo.py`` and ``algos/liam.py`` record.

3. **The encoder and the wavelet network are SHARED across agents; the actors
   are not.** The paper is single-agent. Ninety-odd private sequence models,
   each seeing one transition per day, would be ninety small-sample problems;
   sharing is the choice DGN's paper makes for the same reason and it keeps the
   arm's cost in the same range as the others. It is unconditional here -- there
   is no per-agent switch -- and the checklist records that as the adaptation.

4. **The context is standardised.** URB pays ``-travel_time`` (order -1000) and
   its observation leads with a start time in seconds (order 1000). Fed raw, the
   encoder's reconstruction is dominated by two constants and the latent becomes
   an agent identifier -- the LIAM failure mode in
   ``docs/baselines/README.md`` section 4, which was MEASURED, not guessed.
   ``normalize_context: false`` reproduces it; ``z_std`` in the diagnostics is
   the number that shows it happening.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from urb_baselines.algos.reference import PerAgentPPOAlgorithm, SingleStepPPO
from urb_baselines.nets import MLP, RunningNorm

__all__ = ["WISDOM", "WaveletNet", "add_args"]

_SQRT_HALF = float(np.sqrt(0.5))


class WaveletNet(nn.Module):
    """Learnable multi-level Haar DWT over a latent sequence, plus the heads.

    The paper initialises two dilated causal convolutions as the Haar filter
    pair, ``y0 = [1/sqrt2, 1/sqrt2]`` (low-pass) and ``y1 = [1/sqrt2, -1/sqrt2]``
    (high-pass), and lets them train. One level halves the sequence and produces
    an approximation ``u`` and a detail ``g``; ``levels`` of that give
    ``u_M, g_1 .. g_M``. The most recent coefficient of each band is taken --
    the paper downsamples the selected details -- and a linear map returns the
    stack to the latent's own space as ``z_hat``.

    Grouped convolutions keep each latent coordinate on its own filter, which is
    what makes this a wavelet transform of the SEQUENCE rather than a mixing
    layer that happens to have width two.
    """

    def __init__(self, latent_dim, levels):
        super(WaveletNet, self).__init__()
        self.latent_dim = int(latent_dim)
        self.levels = int(levels)
        self.low = nn.ModuleList()
        self.high = nn.ModuleList()
        for _ in range(self.levels):
            lo = nn.Conv1d(latent_dim, latent_dim, kernel_size=2, stride=2,
                           groups=latent_dim, bias=False)
            hi = nn.Conv1d(latent_dim, latent_dim, kernel_size=2, stride=2,
                           groups=latent_dim, bias=False)
            with torch.no_grad():
                lo.weight.copy_(torch.tensor([_SQRT_HALF, _SQRT_HALF]
                                             ).view(1, 1, 2).repeat(
                                                 latent_dim, 1, 1))
                hi.weight.copy_(torch.tensor([_SQRT_HALF, -_SQRT_HALF]
                                             ).view(1, 1, 2).repeat(
                                                 latent_dim, 1, 1))
            self.low.append(lo)
            self.high.append(hi)
        self.mix = nn.Linear(latent_dim * (self.levels + 1), latent_dim)
        self.ar = nn.Linear(latent_dim, latent_dim)      # auto-regressive head

    def forward(self, z_seq):
        """``z_seq``: (B, L, D) -> ``(z_hat, bands)``.

        ``bands`` is ``[g_1 .. g_M, u_M]``, each (B, D), kept so the report can
        show how much energy the multi-scale part actually carries.
        """
        x = z_seq.transpose(1, 2)                        # (B, D, L)
        bands = []                                       # levels = 0: identity
        for m in range(self.levels):
            if x.shape[-1] < 2:
                # Sequence exhausted: pad by repetition rather than silently
                # producing a shorter stack, so the mix layer's width is fixed.
                x = x.repeat(1, 1, 2)
            g = self.high[m](x)
            x = self.low[m](x)
            bands.append(g[:, :, -1])                    # most recent detail
        bands.append(x[:, :, -1])                        # approximation u_M
        z_hat = self.mix(torch.cat(bands, dim=-1))
        return z_hat, bands


class _Encoder(nn.Module):
    """``e_eta(z | C)``: a Gaussian latent from one day's context tuple."""

    def __init__(self, in_dim, latent_dim, num_hidden, widths):
        super(_Encoder, self).__init__()
        self.body = MLP(in_dim, 2 * latent_dim, num_hidden, list(widths))
        self.latent_dim = int(latent_dim)

    def forward(self, x):
        h = self.body(x)
        mu, logvar = h[..., :self.latent_dim], h[..., self.latent_dim:]
        return mu, torch.clamp(logvar, -8.0, 4.0)


class WISDOM(PerAgentPPOAlgorithm):
    """Wavelet predictive representations on URB's own PPO."""

    name = "wisdom"
    ARMS = ("wisdom", "flat", "noar")

    def __init__(self, ctx):
        cfg = dict(ctx.algo_cfg)
        self.arm = str(getattr(ctx.args, "arm", None)
                       or cfg.get("arm", "wisdom")).lower()
        if self.arm not in self.ARMS:
            raise ValueError(f"[URB-BL][wisdom] --arm must be one of {self.ARMS}")
        self.latent_dim = int(cfg.get("latent_dim", 8))
        self.seq_len = int(cfg.get("seq_len", 16))
        self.levels = int(getattr(ctx.args, "levels", None)
                          or cfg.get("levels", 2))
        self.wav_gamma = float(cfg.get("wav_gamma", 0.99))
        self.alpha_y = float(cfg.get("alpha_y", 1.0))
        self.kl_coef = float(cfg.get("kl_coef", 0.1))
        self.enc_lr = float(cfg.get("enc_lr", 0.0) or ctx.params["lr"])
        self.train_every = int(cfg.get("train_every", 4))
        self.wav_batch = int(cfg.get("wav_batch", 32))
        self.wav_epochs = int(cfg.get("wav_epochs", 1))
        self.polyak = float(cfg.get("polyak", 0.01))
        self.normalize_context = bool(cfg.get("normalize_context", True))
        self.print_every = int(cfg.get("print_every", 100))
        if self.arm == "flat":
            # levels = 0 makes WaveletNet the identity on the last latent: the
            # representation is z_t itself, with the TD and AR heads still on
            # top. That isolates the DECOMPOSITION from everything else.
            self.levels = 0

        super(WISDOM, self).__init__(ctx)

        D = self.latent_dim
        ctx_dim = self.obs_size + self.n_actions + 1        # obs, a_onehot, r
        self.enc = _Encoder(ctx_dim, D, self.host_hp["num_hidden"],
                            self.host_hp["widths"]).to(self.device)
        self.wav = WaveletNet(D, self.levels).to(self.device)
        self.wav_t = WaveletNet(D, self.levels).to(self.device)
        self.wav_t.load_state_dict(self.wav.state_dict())
        for p in self.wav_t.parameters():
            p.requires_grad_(False)
        self.opt = optim.Adam(
            list(self.enc.parameters()) + list(self.wav.parameters()),
            lr=self.enc_lr)

        self.obs_norm = RunningNorm(self.obs_size)
        self.r_norm = RunningNorm(1)
        self.hist = {a: [] for a in self.av_ids}     # list of context vectors
        self.zhat = {a: np.zeros(D, dtype=np.float32) for a in self.av_ids}
        self.n_updates = 0
        self.last = {"td": float("nan"), "ar": float("nan"),
                     "kl": float("nan"), "detail": float("nan")}
        self.z_track = []
        self.loss_rows = []

        self.banner("WISDOM (wavelet predictive task representations)", [
            ("paper", "arXiv:2510.04507 (Wang et al., 2025)"),
            ("arm", {"wisdom": "wavelet TD + auto-regressive head",
                     "flat": "no decomposition (ablation)",
                     "noar": "wavelet TD only, no AR term (ablation)"}[self.arm]),
            ("base learner", "URB's own PPO (scripts/ippo.py), unchanged"),
            ("latent dim", f"{D}"),
            ("sequence", f"{self.seq_len} DAYS (an episode is one step)"),
            ("levels M", f"{self.levels}   filters init to Haar, then learned"),
            ("Gamma", f"{self.wav_gamma} I"),
            ("policy input", f"observation ({self.obs_size}) + z_hat ({D})"),
            ("context", "standardised" if self.normalize_context
             else "*** RAW -- the latent will encode the start time"),
        ])

    # ------------------------------------------------------------------ inputs
    def input_size(self):
        return self.obs_size + self.latent_dim

    def make_model(self, agent_id):
        return SingleStepPPO(self.input_size(), self.n_actions,
                             device=self.device, **self.host_hp)

    def observation(self, agent_id, obs):
        # The policy's observation is standardised as well as the encoder's
        # context. Without it the policy input is a start time in seconds beside
        # a latent of order one, and the latent is below the network's
        # resolution -- the same scale failure the context has, one layer later.
        o = self.obs_norm(obs, self.normalize_context)
        return np.concatenate([o, self.zhat[agent_id]])

    # ------------------------------------------------------------------ latent
    def _windows(self, agents):
        """Stack each agent's last ``seq_len`` context vectors, left-padded."""
        L, C = self.seq_len, self.obs_size + self.n_actions + 1
        out = np.zeros((len(agents), L, C), dtype=np.float32)
        for i, a in enumerate(agents):
            h = self.hist[a]
            if not h:
                continue
            take = h[-L:]
            out[i, L - len(take):] = np.asarray(take, dtype=np.float32)
        return out

    @torch.no_grad()
    def _refresh_zhat(self):
        """One batched forward for the whole fleet, once a day.

        The window ends YESTERDAY -- today's transition does not exist when the
        agent has to act -- so ``z_hat`` is a genuine prediction of today's task,
        which is what the paper's predictive representation is for.
        """
        agents = self.av_ids
        w = torch.from_numpy(self._windows(agents)).to(self.device)
        B, L, C = w.shape
        mu, _ = self.enc(w.reshape(B * L, C))
        z = mu.reshape(B, L, self.latent_dim)
        zh, bands = self.wav(z)
        arr = zh.cpu().numpy()
        for i, a in enumerate(agents):
            self.zhat[a] = arr[i].astype(np.float32)
        self.z_track.append(float(z[:, -1, :].std().item()))
        if len(bands) > 1:
            det = sum(float((b ** 2).sum().item()) for b in bands[:-1])
            app = float((bands[-1] ** 2).sum().item())
            self.last["detail"] = det / max(det + app, 1e-12)

    # ------------------------------------------------------------------ hooks
    def begin_episode(self, day, phase):
        self._refresh_zhat()
        self._today = {}

    def act(self, agent_id, obs):
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._today[agent_id] = o
        self.obs_norm.update(o)
        return super(WISDOM, self).act(agent_id, obs)

    def end_episode(self, day, phase, info):
        if phase != "train":
            return
        for a in self.av_ids:
            o = self._today.get(a)
            r = info.rewards.get(a)
            k = info.actions.get(a)
            if o is None or r is None or k is None:
                continue
            self.r_norm.update([float(r)])
            onehot = np.zeros(self.n_actions, dtype=np.float32)
            onehot[int(k)] = 1.0
            ctx_vec = np.concatenate([
                self.obs_norm(o, self.normalize_context), onehot,
                self.r_norm([float(r)], self.normalize_context)])
            h = self.hist[a]
            h.append(ctx_vec.astype(np.float32))
            if len(h) > self.seq_len * 4:
                del h[0]
        if (day + 1) % max(self.train_every, 1) == 0:
            self._train()

    # ------------------------------------------------------------------ learn
    def _train(self):
        """The paper's ``J_phi``: wavelet TD + AR, plus the encoder's VIB term."""
        pool = [a for a in self.av_ids if len(self.hist[a]) >= self.seq_len + 1]
        if not pool:
            return
        idx = np.random.choice(len(pool), min(self.wav_batch, len(pool)),
                               replace=False)
        agents = [pool[i] for i in idx]
        L, C = self.seq_len, self.obs_size + self.n_actions + 1
        cur = np.zeros((len(agents), L, C), dtype=np.float32)
        nxt = np.zeros((len(agents), L, C), dtype=np.float32)
        for i, a in enumerate(agents):
            h = self.hist[a]
            cur[i] = np.asarray(h[-(L + 1):-1], dtype=np.float32)
            nxt[i] = np.asarray(h[-L:], dtype=np.float32)

        for _ in range(max(self.wav_epochs, 1)):
            cu = torch.from_numpy(cur).to(self.device)
            nx = torch.from_numpy(nxt).to(self.device)
            B = cu.shape[0]
            mu_c, lv_c = self.enc(cu.reshape(B * L, C))
            std = torch.exp(0.5 * lv_c)
            z_c = (mu_c + std * torch.randn_like(std)).reshape(B, L,
                                                               self.latent_dim)
            kl = (-0.5 * (1.0 + lv_c - mu_c.pow(2) - lv_c.exp())).sum(-1).mean()

            with torch.no_grad():
                mu_n, _ = self.enc(nx.reshape(B * L, C))
                z_n = mu_n.reshape(B, L, self.latent_dim)
                zh_n, _ = self.wav_t(z_n)

            zh_c, _ = self.wav(z_c)
            # Wavelet TD: F W(z_t) = z_t + Gamma W(z_{t+1})
            target = z_c[:, -1, :].detach() + self.wav_gamma * zh_n
            td = 0.5 * ((zh_c - target) ** 2).sum(-1).mean()
            # AR: predict the next representation from the current one. A unit
            # variance Gaussian NLL is an MSE up to a constant.
            ar = ((self.wav.ar(zh_c) - zh_n) ** 2).sum(-1).mean()
            loss = self.alpha_y * td + self.kl_coef * kl
            if self.arm != "noar":
                loss = loss + ar

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.enc.parameters()) + list(self.wav.parameters()), 1.0)
            self.opt.step()
            with torch.no_grad():
                for p, q in zip(self.wav_t.parameters(), self.wav.parameters()):
                    p.mul_(1.0 - self.polyak).add_(self.polyak * q)
            self.last.update({"td": float(td.item()), "ar": float(ar.item()),
                              "kl": float(kl.item())})
            self.loss_rows.append(float(loss.item()))
            self.n_updates += 1

    def begin_test(self):
        super(WISDOM, self).begin_test()
        self.enc.eval()
        self.wav.eval()

    # ------------------------------------------------------------------ report
    def diagnostics(self):
        return {
            "z_std": self.z_track[-1] if self.z_track else float("nan"),
            "wav_td": self.last["td"],
            "ar": self.last["ar"],
            "kl": self.last["kl"],
            "detail": self.last["detail"],
            "upd": self.n_updates,
        }

    def loss_records(self):
        rows = super(WISDOM, self).loss_records()
        for it, v in enumerate(self.loss_rows, start=1):
            rows.append({"iteration": it, "agent_id": "shared/wavelet",
                         "loss": v})
        return rows

    def close(self):
        zs = float(np.mean(self.z_track[-100:])) if self.z_track else 0.0
        det = self.last["detail"]
        print("\n" + "=" * 74)
        print("[wisdom] WAVELET REPRESENTATION REPORT")
        print(f"  arm                      {self.arm}")
        print(f"  representation updates   {self.n_updates}")
        print(f"  latent spread z_std      {zs:.4f}   (last 100 days)")
        print(f"  wavelet TD loss          {self.last['td']:.4f}")
        print(f"  auto-regressive loss     {self.last['ar']:.4f}")
        print(f"  encoder KL               {self.last['kl']:.4f}")
        print(f"  detail energy fraction   "
              + ("n/a (--arm flat)" if self.arm == "flat"
                 else f"{det:.4f}"))
        if self.n_updates == 0:
            print("  *** THE REPRESENTATION WAS NEVER TRAINED. No agent ever held")
            print("  *** seq_len+1 days of context, so the policy ran on a")
            print("  *** constant zero z_hat and this is IPPO with a wider input.")
        elif zs < 1e-4:
            print("  *** THE LATENT IS CONSTANT ACROSS AGENTS AND DAYS. The")
            print("  *** encoder collapsed, so z_hat carries no task information.")
            print("  *** If normalize_context is false, that is the documented")
            print("  *** cause: the start time and the reward are ~1000 and")
            print("  *** dominate every other coordinate (see the module")
            print("  *** docstring and LIAM's row in docs/baselines/README.md).")
        elif self.arm != "flat" and np.isfinite(det) and det < 1e-4:
            print("  *** ALL THE ENERGY IS IN THE APPROXIMATION BAND. The detail")
            print("  *** coefficients are ~0, so the decomposition has collapsed")
            print("  *** into a moving average and the multi-scale claim is not")
            print("  *** being exercised -- this is WISDOM-flat under another")
            print("  *** name. On a city whose latent barely moves day to day")
            print("  *** that can be true of the DATA; say which.")
        print("=" * 74 + "\n", flush=True)


def add_args(parser):
    parser.add_argument('--arm', type=str, default=None, choices=list(WISDOM.ARMS),
                        help="wisdom: wavelet TD + AR head (default). flat: no "
                             "wavelet decomposition, the representation is the "
                             "raw latent. noar: wavelet TD without the "
                             "auto-regressive term (the paper's own ablation).")
    parser.add_argument('--levels', type=int, default=None,
                        help="wavelet decomposition levels M (default 2)")

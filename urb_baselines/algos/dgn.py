"""DGN -- Graph Convolutional Reinforcement Learning (ICLR 2020) on URB.

    Jiang, Dun, Huang, Lu, "Graph Convolutional Reinforcement Learning",
    ICLR 2020.  https://openreview.net/forum?id=S8icDSeqfvy
    Code: https://github.com/PKU-RL/DGN  (Routing/routers_regularization.py is
          the closest scenario: a routing problem with a 3-neighbour graph),
          PyTorch port https://github.com/jiechuanjiang/pytorch_DGN

``paper/BASELINES.md`` B3 -- "Graph / communication-structured policies (the same
peer information, unstructured)":

    "PACT uses the peers' broadcast actions through a *declared* operator. The
    natural objection: a policy that receives the neighbours' actions through a
    learned graph network could learn the same coupling. Prediction: it learns
    the sigma=0 coupling but cannot track the drift, because the gain is not
    identified as a separate quantity."

B3's chosen representative is a GNN model flag in BenchMARL, which URB does not
have (neither ``ippo_torchrl.py`` nor ``mappo_torchrl.py`` exposes a graph
model), so the published method itself is ported. DGN is Q-learning based and
discrete, so it sits on URB's off-policy base.

THE ARCHITECTURE (paper Section 3.1-3.2, Table 4)
--------------------------------------------------------------------------------
    encoder MLP   ->  relation kernel 1  ->  relation kernel 2  ->  Q head
                      (multi-head attention over B_{+i})

* Eq. 2: ``alpha^m_ij = softmax_j( tau * (W_Q^m h_i) . (W_K^m h_j)^T )`` over
  ``j in B_{+i} = B_i u {i}``, with ``tau = 0.25`` (Table 4) = ``1/sqrt(dv)``
  for ``dv = 16``.
* Eq. 3: ``h'_i = sigma( concat_m [ sum_j alpha^m_ij W_V^m h_j ] )`` with sigma a
  one-layer ReLU MLP.
* "for each agent, the features of all the preceding layers are concatenated and
  fed into the Q network" -- so the head sees ``[h0 | h1 | h2]``, DenseNet-style.
* Adjacency ``C_i`` is the ego-gather: row 0 is the one-hot of ``i``, row j the
  one-hot of its (j-1)-th neighbour. ``|B| = 3`` (Table 4; Figure 9 shows 3
  beating 1, 2 and 4). Two convolutional layers (Table 4, Figure 8).
* **Temporal relation regularisation** (Section 3.3, Eq. 4): add
  ``lambda * (1/M) sum_m KL( G^k_m(O_{i,C}) || G^k_m(O'_{i,C}) )`` on the UPPER
  layer's attention distribution, with ``lambda = 0.03`` (Table 4). The paper is
  explicit that this term uses the CURRENT network at the next state, not the
  target network. It is part of DGN; the ablation without it is the paper's
  "DGN-R" and is ``--arm dgn_r`` here.
* Parameter sharing: "All agents share weights and gradients are accumulated."
  Kept -- the graph convolution is defined on shared weights, so per-agent
  networks would not be DGN.

THE SNAPSHOT PROBLEM, AND THE ONE ADAPTATION
--------------------------------------------------------------------------------
DGN's feature matrix ``F^t`` is every agent's observation AT THE SAME INSTANT. A
URB day has no such instant: travellers act in start-time order and each one's
observation counts the routes earlier same-OD travellers took *that day*, so
agent i's observation does not exist until its turn.

The rule used here, identically at execution and at training:

    F_i(t)[j] = observation agent j acted on during day t-1,   for j != i
    F_i(t)[i] = agent i's CURRENT observation

It is causal (nothing from the future or from later-departing peers), it is the
same construction in both phases (no train/execute asymmetry to explain away),
and the ego keeps its own live observation so the arm is not handicapped against
IQL, whose input is exactly that row. The cost is that neighbours are one day
stale, which is a property of the host and is stated in the checklist.

WHAT URB REMOVES
--------------------------------------------------------------------------------
The Q-loss target (paper Eq. 1) is ``y = r + gamma max_a' Q(O', a'; theta')``. A
URB day is one step and terminates, so ``y = r`` -- as in URB's own IQL. The
target network therefore has no role ("we only use target network to produce the
target Q value") and is not built; the soft update ``beta`` and ``gamma`` are
recorded as N/A in the checklist rather than left in as decoration. The temporal
relation regulariser is unaffected: the paper computes it with the CURRENT
network at both timesteps.

WHAT URB ADDS, AND WHY IT IS NOT OPTIONAL
--------------------------------------------------------------------------------
The regression target is standardised by a running mean/variance of the rewards.
This is NOT a performance tweak; without it the architecture stops being a graph
network at all, and that was measured rather than assumed.

MAgent, where DGN's Table 4 constants were chosen, pays rewards of order 1. URB
pays -travel_time, of order -1000. The Q head is a linear map of the concatenated
features, so to reach -1000 the encoder's features have to grow to about 50. The
attention logit is ``tau * (W_Q h) . (W_K h)``, which is QUADRATIC in that scale:
measured on this host, ``|h| ~ 54`` gives logits of order 8000, the first
relation kernel's softmax becomes a hard argmax (``[0, 0, 0, 1]``), every node
then copies the same neighbour, and the second kernel's attention collapses to
EXACTLY uniform -- so the temporal relation regulariser is identically zero and
the graph convolution has degenerated into "copy one neighbour".

Standardising the target restores the unit scale DGN's ``tau = 1/sqrt(dv)`` was
calibrated for. It is the same device HARL's ``ValueNorm``, LCPO's ``ret_rms``
and LIAM's ``standardise_stream`` all use for the same reason. Q values are then
in normalised units, which changes no argmax and therefore no action.
``normalize_target: false`` reproduces the degenerate behaviour above, and the
banner and the close() report both print the attention spread so the collapse is
visible if it ever happens again.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from urb_baselines.graph import build_from_ctx
from urb_baselines.host import BaselineAlgorithm
from urb_baselines.nets import MLP

__all__ = ["DGN", "DGNNet", "RelationKernel", "add_args"]


class RelationKernel(nn.Module):
    """Multi-head dot-product attention over ``B_{+i}``. Paper Eq. 2-3."""

    def __init__(self, din, dout, n_heads=8, dv=16):
        super(RelationKernel, self).__init__()
        self.n_heads = int(n_heads)
        self.dv = int(dv)
        inner = self.n_heads * self.dv
        # The released Keras kernel puts a ReLU on the Q/K/V projections.
        self.fcq = nn.Linear(din, inner)
        self.fck = nn.Linear(din, inner)
        self.fcv = nn.Linear(din, inner)
        self.out = nn.Linear(inner, dout)
        self.tau = 1.0 / np.sqrt(self.dv)      # = 0.25 at dv = 16 (Table 4)

    def forward(self, h, adj, mask):
        """``h``: (B, N, din); ``adj``: (N, 1+m) long; ``mask``: (N, 1+m) bool.

        Returns ``(B, N, dout)`` and the ego's attention ``(B, N, heads, 1+m)``.
        Row 0 of ``adj`` is the ego itself, so the query is ``h_i`` and the keys
        are ``h_j`` for ``j in B_{+i}``.
        """
        Bn, N, _ = h.shape
        m = adj.shape[1]
        q = torch.relu(self.fcq(h))                       # (B, N, inner)
        k = torch.relu(self.fck(h))
        v = torch.relu(self.fcv(h))

        kk = k[:, adj.reshape(-1), :].reshape(Bn, N, m, -1)
        vv = v[:, adj.reshape(-1), :].reshape(Bn, N, m, -1)
        qq = q.reshape(Bn, N, 1, self.n_heads, self.dv)
        kk = kk.reshape(Bn, N, m, self.n_heads, self.dv)
        vv = vv.reshape(Bn, N, m, self.n_heads, self.dv)

        att = (qq * kk).sum(-1) * self.tau                # (B, N, m, heads)
        att = att.masked_fill(~mask.view(1, N, m, 1), -9e15)
        att = torch.softmax(att, dim=2)
        out = (att.unsqueeze(-1) * vv).sum(dim=2)         # (B, N, heads, dv)
        out = out.reshape(Bn, N, self.n_heads * self.dv)
        return torch.relu(self.out(out)), att.permute(0, 1, 3, 2)


class DGNNet(nn.Module):
    """encoder -> 2 relation kernels -> Q head on [h0 | h1 | h2]."""

    def __init__(self, obs_size, n_actions, num_hidden, widths, n_heads=8, dv=16):
        super(DGNNet, self).__init__()
        h = int(widths[-1])
        self.encoder = MLP(obs_size, h, num_hidden, list(widths))
        self.att1 = RelationKernel(h, h, n_heads, dv)
        self.att2 = RelationKernel(h, h, n_heads, dv)
        self.q = nn.Linear(3 * h, n_actions)

    def forward(self, x, adj, mask):
        h0 = torch.relu(self.encoder(x))
        h1, a1 = self.att1(h0, adj, mask)
        h2, a2 = self.att2(h1, adj, mask)
        return self.q(torch.cat([h0, h1, h2], dim=-1)), a1, a2


class DGN(BaselineAlgorithm):
    name = "dgn"

    ARMS = ("dgn", "dgn_r")     # dgn_r = the paper's ablation, no temporal reg

    def __init__(self, ctx):
        super(DGN, self).__init__(ctx)
        P = ctx.params
        cfg = dict(self.cfg)
        cfg.setdefault("n_neighbors", 3)        # Table 4
        cfg.setdefault("n_heads", 8)            # Table 4
        cfg.setdefault("dv", 16)                # tau = 0.25 = 1/sqrt(dv)
        cfg.setdefault("reg_lambda", 0.03)      # Table 4
        cfg.setdefault("graph", "overlap")
        cfg.setdefault("normalize_target", True)
        if getattr(ctx.args, "graph", None):
            cfg["graph"] = str(ctx.args.graph)
        self.cfg = cfg
        self.normalize_target = bool(cfg["normalize_target"])
        self.r_mean, self.r_var, self.r_count = 0.0, 1.0, 1e-4
        arm = getattr(ctx.args, "arm", None) or str(cfg.get("arm", "dgn"))
        if arm not in self.ARMS:
            raise ValueError(f"[dgn] --arm must be one of {self.ARMS}")
        self.arm = arm
        self.reg_lambda = 0.0 if arm == "dgn_r" else float(cfg["reg_lambda"])

        # ---- the graph ------------------------------------------------
        self.graph = build_from_ctx(
            ctx, mode=str(cfg["graph"]), n_neighbors=int(cfg["n_neighbors"]),
            slack=float(cfg.get("slack", 0.0)), tag="dgn")
        mode = self.graph.mode
        self.adj = torch.as_tensor(self.graph.adj, dtype=torch.long,
                                   device=self.device)
        self.mask = torch.as_tensor(self.graph.mask, dtype=torch.bool,
                                    device=self.device)

        # ---- the network (shared across agents) -----------------------
        self.net = DGNNet(self.obs_size, self.n_actions, int(P["num_hidden"]),
                          list(P["widths"]), int(cfg["n_heads"]),
                          int(cfg["dv"])).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=float(P["lr"]))

        # ---- exploration: the host's schedule -------------------------
        self.eps = float(P["eps_init"])
        self.eps_decay = float(P["eps_decay"])
        self.eps_min = float(P["eps_min"])
        self.batch_size = int(P["batch_size"])
        self.num_epochs = int(P["num_epochs"])
        self.rng = np.random.RandomState(int(ctx.env_seed) * 13 + 5)

        # ---- day buffer: one observation matrix per day ---------------
        self.cap = int(P["buffer_size"])
        self.buf_obs = np.zeros((self.cap, self.n_av, self.obs_size),
                                dtype=np.float32)
        self.buf_act = np.zeros((self.cap, self.n_av), dtype=np.int64)
        self.buf_rew = np.zeros((self.cap, self.n_av), dtype=np.float32)
        self.buf_full = np.zeros(self.cap, dtype=bool)
        self.n_stored = 0
        self.prev_obs = None            # day t-1's acting observations
        self._today = np.zeros((self.n_av, self.obs_size), dtype=np.float32)
        self._seen = np.zeros(self.n_av, dtype=bool)
        # The per-day arrays are (re)built by begin_episode, but they are
        # allocated here too so that a hook the caller forgot is a no-op rather
        # than an AttributeError deep inside act(). Selftest gate CFG-2 is what
        # noticed.
        self._acts = np.zeros(self.n_av, dtype=np.int64)
        self._rews = np.zeros(self.n_av, dtype=np.float32)
        self._rew_seen = np.zeros(self.n_av, dtype=bool)
        self.loss = []
        self.last = {"q": 0.0, "reg": 0.0, "att_spread": 0.0}

        self.banner("DGN (ICLR 2020) -- graph convolutional RL", [
            ("arm", self.arm + ("   (paper: with temporal relation reg.)"
                                if arm == "dgn" else
                                "   (paper's ablation: no temporal reg.)")),
            ("base learner", "URB's IQL objective, unchanged (target = r)"),
            ("graph", f"{mode}, |B|={cfg['n_neighbors']} "
                      f"(Table 4; Figure 9 prefers 3)"),
            ("kernels", f"2 x multi-head attention, heads={cfg['n_heads']}, "
                        f"dv={cfg['dv']}, tau={1 / np.sqrt(int(cfg['dv'])):.2f}"),
            ("Q head", "on [h0 | h1 | h2] (all preceding layers)"),
            ("weights", "SHARED across agents (paper Figure 1)"),
            ("temporal reg.", f"lambda={self.reg_lambda} on the upper layer, "
                              "current network at both timesteps"),
            ("target network", "N/A -- one-step episode, target = r"),
            ("target scale", "standardised by a running mean/var of the reward"
                             if self.normalize_target else
                             "RAW (ablation: expect the attention to saturate)"),
            ("feature matrix", "neighbours: their day t-1 observation; ego: its "
                               "current one. Identical at act and train."),
            ("exploration", f"host epsilon-greedy eps0={self.eps}, "
                            f"decay={self.eps_decay}"),
        ])

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _load_routes(ctx):
        if not ctx.routes_csv:
            return None, None
        try:
            from urb_ns.network import load_route_table
            routes, ffts = load_route_table(ctx.routes_csv,
                                            int(ctx.params["number_of_paths"]))
        except Exception as exc:                          # noqa: BLE001
            print(f"[dgn][WARN] could not read the route table "
                  f"{ctx.routes_csv}: {type(exc).__name__}: {exc}", flush=True)
            return None, None
        dur = {od: float(np.mean(v)) for od, v in ffts.items()}
        return routes, dur

    @staticmethod
    def _durations_from_freeflow(ctx):
        out = {}
        for od, row in ctx.free_flow.items():
            r = np.asarray(row, dtype=np.float64)
            r = r[np.isfinite(r)]
            if r.size:
                out[od] = float(r.mean())
        return out

    # ------------------------------------------------------------------ graph
    def _features(self, ego_slot, own_obs):
        """``F_i(t)``: day t-1 for everyone, the ego's live observation for i."""
        base = (self.prev_obs if self.prev_obs is not None
                else np.zeros((self.n_av, self.obs_size), dtype=np.float32))
        f = base.copy()
        f[ego_slot] = own_obs
        return f

    # ------------------------------------------------------------------ hooks
    def begin_episode(self, day, phase):
        self._today = (self.prev_obs.copy() if self.prev_obs is not None
                       else np.zeros((self.n_av, self.obs_size),
                                     dtype=np.float32))
        self._seen[:] = False
        self._acts = np.zeros(self.n_av, dtype=np.int64)
        self._rews = np.zeros(self.n_av, dtype=np.float32)
        self._rew_seen = np.zeros(self.n_av, dtype=bool)

    def act(self, agent_id, obs):
        i = self.slot_of[agent_id]
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._today[i] = o
        self._seen[i] = True
        if (not self.deterministic) and self.rng.rand() < self.eps:
            a = int(self.rng.randint(self.n_actions))
        else:
            f = torch.as_tensor(self._features(i, o), device=self.device
                                ).unsqueeze(0)
            with torch.no_grad():
                q, _, _ = self.net(f, self.adj, self.mask)
            a = int(torch.argmax(q[0, i]).item())
        self._acts[i] = a
        return a

    def push(self, agent_id, reward):
        i = self.slot_of[agent_id]
        self._rews[i] = float(reward)
        self._rew_seen[i] = True

    def end_episode(self, day, phase, info):
        if phase == "train" and self._seen.all() and self._rew_seen.all():
            j = self.n_stored % self.cap
            self.buf_obs[j] = self._today
            self.buf_act[j] = self._acts
            self.buf_rew[j] = self._rews
            self.buf_full[j] = True
            self.n_stored += 1
            if self.normalize_target:
                # Welford over the day's rewards; see the module docstring for
                # why this is load-bearing for the attention and not a tweak.
                for r in self._rews:
                    self.r_count += 1
                    d = float(r) - self.r_mean
                    self.r_mean += d / self.r_count
                    self.r_var += (d * (float(r) - self.r_mean)
                                   - self.r_var) / self.r_count
        self.prev_obs = self._today.copy()

    # ------------------------------------------------------------------ learn
    def _sampleable_slots(self):
        """Ring slots whose day has BOTH a predecessor and a successor stored.

        ``buf_obs`` is a ring, so slot ``j+1`` holds the day after slot ``j``
        everywhere except at the write head, where it holds a day a full lap
        older. Sampling that one pair would regularise the attention between two
        days ``cap`` apart -- rare, silent and wrong -- so it is excluded rather
        than tolerated.
        """
        if self.n_stored <= self.cap:
            hi = self.n_stored - 1          # need slot+1 stored
            return np.arange(1, hi) if hi > 1 else np.empty(0, dtype=np.int64)
        head = self.n_stored % self.cap     # oldest day sits here
        newest = (head - 1) % self.cap      # its successor is a lap away
        return np.setdiff1d(np.arange(self.cap), np.array([head, newest]))

    def learn(self, day):
        """Paper Eq. 1 with ``y = r`` (one-step episode), plus Eq. 4's KL.

        A sample is a (day, ego) pair so that each ego's feature matrix is the
        one it would actually have seen -- the same construction as ``act``. DGN
        accumulates the gradient over all agents in a graph; here the graph's
        agents are drawn uniformly, which is the same expectation at a fraction
        of the memory.
        """
        cand = self._sampleable_slots()
        if cand.size < 1 or min(self.n_stored, self.cap) < self.batch_size:
            return
        dev = self.device
        losses = []
        b = torch.arange(self.batch_size, device=dev)
        for _ in range(self.num_epochs):
            days = cand[self.rng.randint(0, cand.size, size=self.batch_size)]
            egos = self.rng.randint(0, self.n_av, size=self.batch_size)
            f = self.buf_obs[days]                        # (B, N, obs) = day t
            f_prev = self.buf_obs[(days - 1) % self.cap].copy()
            f_next = self.buf_obs[(days + 1) % self.cap]
            bi = np.arange(self.batch_size)
            feat = f_prev
            feat[bi, egos] = f[bi, egos]                  # F_i(t)
            feat_next = f.copy()
            feat_next[bi, egos] = f_next[bi, egos]        # F_i(t+1)

            x = torch.as_tensor(feat, device=dev)
            a = torch.as_tensor(self.buf_act[days, egos], device=dev)
            r_np = self.buf_rew[days, egos].astype(np.float64)
            if self.normalize_target:
                r_np = (r_np - self.r_mean) / np.sqrt(self.r_var + 1e-8)
            r = torch.as_tensor(r_np.astype(np.float32), device=dev)

            q, _, att2 = self.net(x, self.adj, self.mask)
            ego_t = torch.as_tensor(egos, device=dev, dtype=torch.long)
            q_ego = q[b, ego_t]                           # (B, K)
            q_taken = q_ego.gather(1, a.view(-1, 1)).squeeze(1)
            q_loss = F.mse_loss(q_taken, r)               # y = r

            loss = q_loss
            reg = torch.zeros((), device=dev)
            if self.reg_lambda > 0:
                xn = torch.as_tensor(feat_next, device=dev)
                _, _, att2n = self.net(xn, self.adj, self.mask)
                p = att2[b, ego_t]                        # (B, heads, 1+m)
                pn = att2n[b, ego_t]
                valid = self.mask[ego_t].unsqueeze(1)     # (B, 1, 1+m)
                # Eq. 4: KL( G(O_t) || G(O_{t+1}) ), averaged over heads, with
                # the CURRENT network at both timesteps (Section 3.3).
                kl = (p * (torch.log(p + 1e-12) - torch.log(pn + 1e-12)))
                kl = (kl * valid).sum(-1).mean()
                reg = kl
                loss = loss + self.reg_lambda * reg

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(float(q_loss.item()))
            with torch.no_grad():
                # The KL's VALUE is tiny whenever the attention is near uniform,
                # which it is at initialisation -- but its GRADIENT is not, so a
                # near-zero `reg` does not mean the term is inert. `spread` is
                # the interpretable companion: how far from uniform the upper
                # layer's attention actually is, over the VALID slots only (a
                # padded slot always has attention 0 and would otherwise show up
                # as spread that is really just padding).
                big = att2.masked_fill(~self.mask.view(1, -1, 1, att2.shape[-1]),
                                       -1.0)
                small = att2.masked_fill(~self.mask.view(1, -1, 1,
                                                         att2.shape[-1]), 2.0)
                spread = float((big.max(-1).values
                                - small.min(-1).values).mean())
            self.last = {"q": float(q_loss.item()), "reg": float(reg.item()),
                         "att_spread": spread}
        if losses:
            self.loss.append(float(np.mean(losses)))
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    # ------------------------------------------------------------------ misc
    def begin_test(self):
        self.deterministic = True
        self.net.eval()

    def diagnostics(self):
        d = {"eps": float(self.eps)}
        d.update(self.last)
        return d

    def loss_records(self):
        return [{"iteration": i, "agent_id": "shared", "loss": v}
                for i, v in enumerate(self.loss, start=1)]

    def close(self):
        s = float(self.last.get("att_spread", 0.0))
        print("\n" + "=" * 74)
        print("[dgn] GRAPH CONVOLUTION REPORT")
        print(f"  days stored             {self.n_stored}")
        print(f"  upper-layer attention spread (valid slots)   {s:.4f}")
        print(f"  temporal relation KL (last)                  "
              f"{self.last.get('reg', 0.0):.3e}")
        if s < 1e-4:
            print("  *** THE ATTENTION IS UNIFORM. Every neighbour is weighted")
            print("  *** equally, so the relation kernel has degenerated into a")
            print("  *** mean and this is DGN-M, not DGN. If normalize_target is")
            print("  *** false, that is the documented cause (see the module")
            print("  *** docstring); otherwise check the feature magnitudes.")
        elif s > 0.95:
            print("  *** THE ATTENTION IS A HARD ARGMAX: each node copies one")
            print("  *** neighbour and the softmax is saturated. Same cause,")
            print("  *** opposite symptom -- check the feature magnitudes.")
        print("=" * 74 + "\n", flush=True)


def add_args(parser):
    parser.add_argument('--arm', type=str, default=None, choices=list(DGN.ARMS),
                        help="dgn (the paper's method, with temporal relation "
                             "regularisation) or dgn_r (the paper's own ablation "
                             "without it)")
    parser.add_argument('--graph', type=str, default=None,
                        choices=["overlap", "copresence", "od"],
                        help="neighbour structure (default overlap: route sets "
                             "share a link AND the trips are co-present)")

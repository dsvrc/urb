"""Deep Fictitious Play -- the equilibrium-seeking arm.

    Solving Continuous Mean Field Games: Deep Reinforcement Learning for
    Non-Stationary Dynamics, NeurIPS 2025 (arXiv:2510.22158).
    "builds upon a Fictitious Play (FP) methodology, leveraging DRL for
     best-response computation and supervised learning for average policy
     representation ... learns a representation of the time-dependent
     population distribution using a Conditional Normalizing Flow."

WHY THIS ARM EXISTS
--------------------------------------------------------------------------------
Nothing in the previous twelve is equilibrium-seeking. Every one of them is a
learner that improves its own return; none of them treats the fleet as a
population whose distribution is the thing being solved for. That leaves the
paper's sharpest objection unanswered:

    "your forecast just walks the fleet toward equilibrium -- fictitious play
     does that with no estimator, no declared basis and no trust head."

URB day-to-day route choice is the textbook setting for that objection: it is a
congestion game (PACT_THEORY A1: an exact potential game on the co-presence
graph), fictitious play is the classical solver for one, and its traffic
instantiation -- best-respond to the averaged load, average the result -- is the
method of successive averages that the assignment literature has used since the
1950s. This arm is that method, in its 2025 deep form.

WHAT THE THEORY PREDICTS
--------------------------------------------------------------------------------
B1 says the steered fleet is a z-normalised logit QRE with effective rationality
``g*kappa/sd(c)``. Fictitious play with a hard best response is the same object
at ``g -> inf``: it converges to the selfish equilibrium, not to the social
optimum, so B4's corollary applies to it too -- it closes the learning gap and
leaves the equilibrium gap exactly where it is. If this arm reaches equilibrium
FASTER than PACT-1 does, that is the honest result and the paper says so; what
it cannot do is beat the equilibrium, and that is the claim to check.

THE THREE ADAPTATIONS URB FORCES, EACH RECORDED IN THE CHECKLIST
--------------------------------------------------------------------------------
1. **The conditional normalizing flow is N/A.** It exists in the paper because
   the state space is continuous, so the population distribution has no finite
   parameterisation. URB's is a distribution over K candidate routes -- a point
   on the K-simplex, represented exactly by K floats. A flow here would be a
   density model of a categorical.

2. **The population is the index-level mix.** Mean-field games assume
   exchangeable agents. URB's OD pairs are singletons on six of seven cities
   (``docs/baselines/README.md`` section 4), so a same-OD population is one
   traveller and carries no field at all. ``field_scope`` therefore defaults to
   ``all``, the same decision and the same reason as ``algos/mfq.py`` FIELD-1.

3. **An FP iteration is a day.** The paper's outer loop is the FP iteration; the
   inner loop trains a best response to a frozen population. Here one day is one
   inner step and ``fp_every`` days are one FP iteration, so the averaging weight
   is ``1/(n+1)`` in FP iterations, not in days.

WHAT IS PLAYED
--------------------------------------------------------------------------------
During TRAINING the best response acts. That is FP's inner loop: the fleet plays
a best response to the averaged population, and the mix it produces is what gets
averaged in -- averaging best responses is what fictitious play is. It also
keeps the host's PPO on-policy, which it would not be if a separately-trained
average policy chose the action while the BR's log-probability was stored for it.

During TEST the AVERAGE policy acts, deterministically, because the average is
the iterate FP claims converges, not the latest best response.

``play: avg`` puts the average policy in charge during training as well, mixed
with the best response at rate ``br_frac`` (Heinrich & Silver's anticipatory
parameter, default 0.1), and ``--arm br`` removes the averaging altogether.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from urb_baselines import graph as graphmod
from urb_baselines.algos.reference import PerAgentPPOAlgorithm, SingleStepPPO
from urb_baselines.nets import MLP, RunningNorm

__all__ = ["DeepFictitiousPlay", "add_args"]


class _AveragePolicy(object):
    """The supervised average policy: cross-entropy onto the BR's own choices.

    This is the paper's "supervised learning for average policy": a classifier
    fitted to the best response's action on the states the fleet actually
    visited. Its reservoir is uniform over the agent's whole history, which is
    what makes it an AVERAGE rather than a copy of the latest BR -- the same
    device as NFSP's reservoir buffer.
    """

    def __init__(self, in_size, n_actions, device, lr, num_hidden, widths,
                 capacity, batch_size, rng):
        self.net = MLP(in_size, n_actions, num_hidden, list(widths)).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=float(lr))
        self.device = device
        self.n_actions = int(n_actions)
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.rng = rng
        self.buf_x = []
        self.buf_y = []
        self.seen = 0
        self.loss = []
        self.deterministic = False

    def add(self, x, y):
        """Reservoir sampling: every (state, BR action) ever seen is equally
        likely to be in the buffer, so the fit is to the time-average."""
        self.seen += 1
        if len(self.buf_x) < self.capacity:
            self.buf_x.append(np.asarray(x, dtype=np.float32))
            self.buf_y.append(int(y))
            return
        j = self.rng.randint(0, self.seen)
        if j < self.capacity:
            self.buf_x[j] = np.asarray(x, dtype=np.float32)
            self.buf_y[j] = int(y)

    def probs(self, x):
        t = torch.FloatTensor(np.asarray(x, dtype=np.float32)
                              ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.net(t), dim=-1).cpu().numpy().reshape(-1)

    def act(self, x):
        p = self.probs(x)
        if self.deterministic:
            return int(np.argmax(p))
        p = np.clip(p, 1e-9, None)
        return int(self.rng.choice(self.n_actions, p=p / p.sum()))

    def learn(self):
        if len(self.buf_x) < self.batch_size:
            return
        idx = self.rng.choice(len(self.buf_x), self.batch_size, replace=False)
        x = torch.FloatTensor(np.asarray([self.buf_x[i] for i in idx],
                                         dtype=np.float32)).to(self.device)
        y = torch.LongTensor([self.buf_y[i] for i in idx]).to(self.device)
        logits = self.net(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.loss.append(float(loss.item()))


class DeepFictitiousPlay(PerAgentPPOAlgorithm):
    """FP over days: BR by URB's PPO against the averaged population mix."""

    name = "dfp"
    ARMS = ("fp", "ema", "br")
    needs_records = True                   # the population mix needs peer actions

    def __init__(self, ctx):
        cfg = dict(ctx.algo_cfg)
        self.field_scope = str(getattr(ctx.args, "field_scope", None)
                               or cfg.get("field_scope", "all")).lower()
        self.avg_mode = str(getattr(ctx.args, "arm", None)
                            or cfg.get("arm", "fp")).lower()
        if self.avg_mode not in self.ARMS:
            raise ValueError(f"[URB-BL][dfp] --arm must be one of {self.ARMS}")
        self.normalize_obs = bool(cfg.get("normalize_obs", True))
        self.play = str(cfg.get("play", "br")).lower()
        if self.play not in ("br", "avg"):
            raise ValueError("[URB-BL][dfp] play must be 'br' or 'avg'")
        self.fp_every = int(cfg.get("fp_every", 10))
        self.br_frac = float(cfg.get("br_frac", 0.1))
        self.rho = float(cfg.get("rho", 0.99))
        self.avg_capacity = int(cfg.get("avg_capacity", 2000))
        self.print_every = int(cfg.get("print_every", 100))

        super(DeepFictitiousPlay, self).__init__(ctx)

        self.rng = np.random.RandomState(int(ctx.torch_seed) * 53 + 11)
        K = self.n_actions

        # WHOSE actions form the population. scope "all" is one global mix over
        # every traveller (the exchangeable mean field an MFG assumes); scope
        # "od" is one mix per OD pair, which on six of URB's seven cities is one
        # traveller's own one-hot action -- the measurement in mfq FIELD-1.
        self._key_of = {a: "*" for a in self.av_ids}
        self._od_members = None
        if self.field_scope == "od":
            self.graph = graphmod.build_from_ctx(
                ctx, mode="od", n_neighbors=1, tag="URB-BL/dfp", verbose=False)
            self._key_of = {a: tuple(self.graph.od_of(a) or ("?",))
                            for a in self.av_ids}
            self._od_members = {}
            for a in self.av_ids:
                for m in self.graph.field_members(a, scope="od"):
                    self._od_members[str(m)] = self._key_of[a]
        else:
            self.graph = None

        self.mu_bar = np.full(K, 1.0 / K, dtype=np.float64)   # reported mix
        self.mu_by_key = {k: np.full(K, 1.0 / K, dtype=np.float64)
                          for k in set(self._key_of.values())}
        self.mu_today = np.full(K, 1.0 / K, dtype=np.float64)
        self.fp_iter = 0
        self.n_mix_updates = 0
        self.avg = {a: _AveragePolicy(
            self.obs_size, K, self.device, self.host_hp["lr"],
            self.host_hp["num_hidden"], self.host_hp["widths"],
            self.avg_capacity, self.host_hp["batch_size"],
            np.random.RandomState(int(ctx.torch_seed) * 7 + i))
            for i, a in enumerate(self.av_ids)}
        self._played_br = {}
        self._br_action = {}
        self._state = {}
        self.obs_norm = RunningNorm(self.obs_size)
        self.mix_infl = []
        self.n_br_plays = 0
        self.n_plays = 0
        self.n_disagree = 0

        self.banner("DEEP FICTITIOUS PLAY (mean-field / equilibrium seeking)", [
            ("paper", "arXiv:2510.22158 (NeurIPS 2025)"),
            ("best response", "URB's own PPO (scripts/ippo.py), unchanged"),
            ("BR input", f"observation ({self.obs_size}) + population mix ({K})"),
            ("average policy", f"supervised, reservoir {self.avg_capacity}"),
            ("averaging", {"fp": f"FP 1/(n+1), one iteration = {self.fp_every} days",
                           "ema": f"EMA rho = {self.rho} (declared ablation)",
                           "br": "NONE -- best response only (ablation)"}[
                              self.avg_mode]),
            ("population scope", self.field_scope),
            ("who acts (train)", "the best response" if self.play == "br"
             else f"the average policy, br_frac = {self.br_frac}"),
            ("who acts (test)", "the average policy -- FP's solution concept"
             if self.avg_mode != "br" else "the best response (--arm br)"),
            ("normalizing flow", "N/A -- the mix is a point on the K-simplex"),
            ("observation", "standardised" if self.normalize_obs else
             "*** RAW -- the mix is invisible beside a start time in seconds"),
        ])

    # ------------------------------------------------------------------ inputs
    def input_size(self):
        return self.obs_size + self.n_actions

    def make_model(self, agent_id):
        return SingleStepPPO(self.input_size(), self.n_actions,
                             device=self.device, **self.host_hp)

    def observation(self, agent_id, obs):
        """The BR sees the mean field; the average policy sees only the state.

        The BR must be a best response TO THE POPULATION, so the averaged mix is
        part of its input. The average policy is a map from states to actions --
        conditioning it on a quantity that changes every day would make it a
        function of the mix rather than an average over history.
        """
        o = self.obs_norm(obs, self.normalize_obs)
        mu = self.mu_by_key.get(self._key_of.get(agent_id, "*"), self.mu_bar)
        return np.concatenate([o, mu.astype(np.float32)])

    def _state_only(self, obs):
        return self.obs_norm(obs, self.normalize_obs)

    # ------------------------------------------------------------------ hooks
    def begin_episode(self, day, phase):
        self._played_br.clear()
        self._br_action.clear()
        self._state.clear()

    def act(self, agent_id, obs):
        self.obs_norm.update(obs)
        br_in = self.observation(agent_id, obs)
        st = self._state_only(obs)
        self._state[agent_id] = st
        model = self.models[agent_id]
        a_br = int(model.act(br_in))          # also sets model.last_* for push()
        self._br_action[agent_id] = a_br

        self.n_plays += 1
        if self.avg_mode == "br":
            self._played_br[agent_id] = True
            self.n_br_plays += 1
            return a_br

        avg = self.avg[agent_id]
        a_avg = avg.act(st)
        if a_br != a_avg:
            self.n_disagree += 1

        # WHO ACTS, AND WHY IT IS THE BEST RESPONSE DURING TRAINING.
        # Fictitious play's inner loop computes a best response TO the averaged
        # population, and the fleet that generates the next empirical mix is the
        # one playing that best response -- averaging best responses is what FP
        # is. So the BR acts while training, on its own on-policy data, and the
        # average policy is fitted alongside it by supervised learning. The
        # AVERAGE is the solution concept, so it is what plays in the test
        # phase, and --play avg puts it in charge during training too (the
        # NFSP-style mixture, with br_frac as the anticipatory parameter).
        #
        # This ordering also keeps the host's PPO honest: if the average policy
        # chose, the stored log-probability would be the BR's for an action the
        # BR did not take, and the ratio would be an uncorrected off-policy one.
        if self.deterministic:
            self._played_br[agent_id] = False
            return int(a_avg)
        if self.play == "br" or self.rng.rand() < self.br_frac:
            self._played_br[agent_id] = True
            self.n_br_plays += 1
            return a_br
        self._played_br[agent_id] = False
        # The BR did not act, so its stored decision must be the action actually
        # taken, scored under the BR itself.
        model.last_action = int(a_avg)
        with torch.no_grad():
            logits = model.policy_net(torch.FloatTensor(br_in).unsqueeze(0
                                      ).to(self.device))
            lp = torch.log_softmax(logits, dim=-1)[0, int(a_avg)]
        model.last_log_prob = float(lp.item())
        return int(a_avg)

    def learn(self, day):
        super(DeepFictitiousPlay, self).learn(day)
        if self.avg_mode != "br":
            for a in self.av_ids:
                self.avg[a].learn()

    def end_episode(self, day, phase, info):
        if phase != "train":
            return
        # Today's empirical population mix over route indices. peer_acts covers
        # EVERY traveller, humans included -- the population of the mean-field
        # game is the whole fleet, not the controllable subset.
        src = info.peer_acts or info.actions
        if not src:
            return
        per_key = {k: np.zeros(self.n_actions, dtype=np.float64)
                   for k in self.mu_by_key}
        total = np.zeros(self.n_actions, dtype=np.float64)
        for a, v in src.items():
            k = int(v)
            if not (0 <= k < self.n_actions):
                continue
            total[k] += 1.0
            if self._od_members is None:
                per_key["*"][k] += 1.0
            else:
                key = self._od_members.get(str(a))
                if key is not None and key in per_key:
                    per_key[key][k] += 1.0
        if total.sum() <= 0:
            return
        self.mu_today = total / total.sum()
        self.n_mix_updates += 1

        # Teach the average policy what the BR would have done today.
        if self.avg_mode != "br":
            for a in self.av_ids:
                br = self._br_action.get(a)
                st = self._state.get(a)
                if br is not None and st is not None:
                    self.avg[a].add(st, br)

        def _blend(old, new, w):
            s = new.sum()
            return old if s <= 0 else (1.0 - w) * old + w * (new / s)

        if self.avg_mode == "ema":
            w = 1.0 - self.rho
        elif self.avg_mode == "fp":
            if (day + 1) % max(self.fp_every, 1) != 0:
                return
            self.fp_iter += 1
            w = 1.0 / (self.fp_iter + 1.0)
        else:                                   # --arm br: no averaging at all
            w = 1.0
        for key, cnt in per_key.items():
            self.mu_by_key[key] = _blend(self.mu_by_key[key], cnt, w)
        self.mu_bar = _blend(self.mu_bar, total, w)

    def begin_test(self):
        super(DeepFictitiousPlay, self).begin_test()
        for a in self.av_ids:
            self.avg[a].net.eval()
            self.avg[a].deterministic = True

    # ------------------------------------------------------------------ report
    def _mix_dev(self):
        return float(np.abs(self.mu_bar - 1.0 / self.n_actions).sum())

    @torch.no_grad()
    def _probe_mix(self):
        """How much does the mean field actually move the best response?

        Mean absolute change in the BR's action probabilities when the mix
        coordinate is replaced by a uniform one, over the agents seen today.
        Zero means the population input is decoration -- which is what happens
        when a probability vector shares an input layer with a start time in
        seconds, so this is the number that catches it.
        """
        vals = []
        for a, st in list(self._state.items())[:16]:
            mu = self.mu_by_key.get(self._key_of.get(a, "*"), self.mu_bar)
            uni = np.full(self.n_actions, 1.0 / self.n_actions)
            x = torch.FloatTensor(np.stack([
                np.concatenate([st, mu.astype(np.float32)]),
                np.concatenate([st, uni.astype(np.float32)])])).to(self.device)
            p = torch.softmax(self.models[a].policy_net(x), dim=-1)
            vals.append(float((p[0] - p[1]).abs().sum().item()))
        return float(np.mean(vals)) if vals else float("nan")

    def diagnostics(self):
        self.mix_infl.append(self._probe_mix())
        return {
            "fp_iter": self.fp_iter,
            "mix_dev": self._mix_dev(),
            "mix_infl": self.mix_infl[-1],
            "br_diff": self.n_disagree / max(self.n_plays, 1),
            "br_play": self.n_br_plays / max(self.n_plays, 1),
        }

    def loss_records(self):
        rows = super(DeepFictitiousPlay, self).loss_records()
        for a in self.av_ids:
            for it, v in enumerate(self.avg[a].loss, start=1):
                rows.append({"iteration": it, "agent_id": f"{a}/avg", "loss": v})
        return rows

    def close(self):
        dev = self._mix_dev()
        diff = self.n_disagree / max(self.n_plays, 1)
        trained = sum(len(self.avg[a].loss) for a in self.av_ids)
        print("\n" + "=" * 74)
        print("[dfp] FICTITIOUS PLAY REPORT")
        print(f"  FP iterations            {self.fp_iter}")
        print(f"  population mix updates   {self.n_mix_updates}")
        print(f"  mix  mu_bar              "
              f"[{', '.join(f'{v:.3f}' for v in self.mu_bar)}]")
        print(f"  |mu_bar - uniform|_1     {dev:.4f}")
        print(f"  BR != average policy     {diff:.1%} of decisions")
        infl = self._probe_mix()
        print(f"  mean-field influence     {infl:.4f}   (|dp| when the mix is")
        print("                           replaced by a uniform one)")
        print(f"  average-policy updates   {trained}")
        if self.n_mix_updates == 0:
            print("  *** THE POPULATION MIX WAS NEVER UPDATED: no peer actions")
            print("  *** ever arrived, so the best response was computed against")
            print("  *** a uniform population every day and this is not "
                  "fictitious")
            print("  *** play. Check that records are reaching the host.")
        elif dev < 1e-3:
            print("  *** THE POPULATION MIX IS UNIFORM. The mean-field input is a")
            print("  *** constant, so the best response is an ordinary IPPO with a")
            print("  *** wider input layer. On a city whose travellers each have")
            print("  *** their own OD pair the index-level mix can be genuinely")
            print("  *** flat -- report it as a property of the instance.")
        if np.isfinite(infl) and infl < 1e-6:
            print("  *** THE POPULATION MIX DOES NOT MOVE THE BEST RESPONSE AT")
            print("  *** ALL, so this arm is an ordinary IPPO with four dead")
            print("  *** input coordinates. If normalize_obs is false that is the")
            print("  *** documented cause: URB's observation leads with a start")
            print("  *** time in seconds, and a probability beside it is below")
            print("  *** the network's resolution (measured -- see the checklist).")
        if self.avg_mode != "br" and trained == 0:
            print("  *** THE AVERAGE POLICY WAS NEVER TRAINED, so what acted was")
            print("  *** an untrained classifier. This is not fictitious play.")
        elif self.avg_mode != "br" and diff < 1e-3:
            print("  *** THE AVERAGE POLICY AND THE BEST RESPONSE NEVER DISAGREE.")
            print("  *** FP has either converged or the BR stopped moving; check")
            print("  *** br_diff over time before calling it convergence.")
        print("=" * 74 + "\n", flush=True)


def add_args(parser):
    parser.add_argument('--arm', type=str, default=None,
                        choices=list(DeepFictitiousPlay.ARMS),
                        help="fp: FP 1/(n+1) averaging (default, the paper). "
                             "ema: exponential averaging of the population mix. "
                             "br: best response only, no averaging -- the "
                             "ablation that shows what the averaging buys.")
    parser.add_argument('--field-scope', type=str, default=None,
                        choices=["all", "od"],
                        help="whose actions form the population mix "
                             "(default all; od is a singleton on most URB cities)")

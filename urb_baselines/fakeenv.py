"""A tiny stand-in for URB, so every baseline can be exercised without SUMO.

WHY THIS EXISTS
--------------------------------------------------------------------------------
The real host needs ``routerl`` and a SUMO install, which means the first time
anybody finds out that a baseline has a shape bug is forty minutes into a run on
the machine that has them. This module reproduces the two things the algorithms
actually depend on -- URB's observation shape and the SEQUENTIAL SINGLE-STEP day
-- in about a hundred lines of numpy, so ``selftest.py`` can drive every arm for
a few hundred days on a laptop.

WHAT IT IS AND IS NOT
--------------------------------------------------------------------------------
It IS: the right observation (``[start_time, counts of the routes earlier
same-OD travellers took today]``), the right action space (K routes per OD), the
right episode structure (one decision per agent per day, reward at the end), the
right reward SCALE (negative travel times of order -1000, which is what breaks
naive value heads), a congestion coupling so that the peers' choices genuinely
matter, and a weather-like multiplier so context-driven arms have something to
condition on.

It is NOT a traffic model and no number produced with it means anything about
any city. It exists to answer "does this code run, learn something, and keep its
own invariants", not "is this method good".
"""

import numpy as np

__all__ = ["FakeURB", "make_ctx"]


class FakeURB(object):
    """A sequential single-step routing day with congestion coupling."""

    def __init__(self, n_agents=24, n_machines=12, n_od=3, n_paths=4, seed=0,
                 period=40, sigma=1.0):
        rng = np.random.RandomState(seed)
        self.rng = rng
        self.n_agents = int(n_agents)
        self.n_paths = int(n_paths)
        self.period = int(period)
        self.sigma = float(sigma)

        self.ids = [str(i) for i in range(n_agents)]
        self.machine = np.zeros(n_agents, dtype=bool)
        self.machine[rng.choice(n_agents, n_machines, replace=False)] = True
        self.od = [(int(i % n_od), int(i % n_od) + 100) for i in range(n_agents)]
        self.start = np.sort(rng.uniform(0, 3600, n_agents))
        self.order = np.argsort(self.start, kind="stable")

        ods = sorted(set(self.od))
        self.free_flow = {od: np.asarray(
            sorted(rng.uniform(400, 1200, n_paths)), dtype=np.float64)
            for od in ods}
        # capacity per (od, path): the congestion denominator
        self.cap = {od: rng.uniform(3, 8, n_paths) for od in ods}

        self.day = 0
        self.last_action = np.zeros(n_agents, dtype=np.int64)
        # The minimum of the NS severity wrapper's surface, so the arms that
        # attach to it (dr_ippo) can be constructed here. `reset` exists for the
        # same reason: DomainRandomiser wraps it.
        self._ns_sigma = float(sigma)
        self._ns_layer = None
        self._ns_day = 0
        self.agent_table = {
            self.ids[i]: {"od": self.od[i], "start": float(self.start[i]),
                          "machine": bool(self.machine[i])}
            for i in range(n_agents)
        }
        self.av_ids = [self.ids[i] for i in range(n_agents) if self.machine[i]]
        self.obs_size = 1 + n_paths

    # ------------------------------------------------------------------ driver
    def A(self, day):
        ph = (day % self.period) / self.period
        return float(np.sin(np.pi * min(ph / 0.5, 1.0)) ** 2) if ph < 0.5 else 0.0

    # ------------------------------------------------------------------ day
    def reset(self, *a, **kw):
        """RouteRL's per-day reset. Nothing to do here; it exists so that the
        arms which WRAP reset (domain randomisation) can be exercised."""
        return None

    def observe(self, slot, today_actions, acted):
        """URB's ``PreviousAgentStartPlusStartTime``: start time, then the routes
        taken so far TODAY by earlier travellers with the same OD."""
        o = np.zeros(1 + self.n_paths, dtype=np.float32)
        o[0] = self.start[slot]
        for j in range(self.n_agents):
            if j == slot or not acted[j]:
                continue
            if self.od[j] == self.od[slot] and self.start[j] < self.start[slot]:
                o[1 + today_actions[j]] += 1
        return o

    def run_day(self, choose):
        """``choose(agent_id, obs) -> action`` for machines; humans pick greedily
        by yesterday's experience. Returns ``(rewards, records)``."""
        acts = np.zeros(self.n_agents, dtype=np.int64)
        acted = np.zeros(self.n_agents, dtype=bool)
        for slot in self.order:
            aid = self.ids[slot]
            if self.machine[slot]:
                a = int(choose(aid, self.observe(slot, acts, acted)))
            else:
                a = int(self.rng.randint(self.n_paths))
            acts[slot] = max(0, min(self.n_paths - 1, a))
            acted[slot] = True

        # congestion: BPR-ish delay from the load on each (od, path), with a
        # weather multiplier that shrinks capacity -- the same shape as URB-NS.
        # `_ns_sigma` is what the severity wrapper carries and what the
        # randomisation arms rewrite in reset(); it is initialised to `sigma`,
        # so every non-randomising arm sees exactly the value it saw before.
        g = 1.0 - float(getattr(self, "_ns_sigma", self.sigma)) * 0.14             * self.A(self.day)
        tt = np.zeros(self.n_agents)
        load = {}
        for i in range(self.n_agents):
            load[(self.od[i], acts[i])] = load.get((self.od[i], acts[i]), 0) + 1
        for i in range(self.n_agents):
            od, k = self.od[i], acts[i]
            u = load[(od, k)] / (self.cap[od][k] * g)
            tt[i] = self.free_flow[od][k] * (1.0 + 0.15 * u ** 4)
        self.last_action = acts
        self.day += 1
        records = [{"id": int(self.ids[i]), "action": int(acts[i]),
                    "travel_time": float(tt[i])} for i in range(self.n_agents)]
        return {self.ids[i]: -float(tt[i]) for i in range(self.n_agents)}, records


class _FakeContextDriver(object):
    """Just enough of ``urb_ns.driver.WeatherDriver`` for DriverContext."""

    def __init__(self, env):
        self.period = env.period
        self.wet_frac = 0.5
        self.loss = 0.14
        self._env = env

    def A(self, day):
        return self._env.A(int(day))

    def is_dry(self, day):
        return self.A(day) == 0.0


class _FakeContext(object):
    """A DriverContext with the same surface, backed by FakeURB's own driver."""

    def __init__(self, env, features="a"):
        self.driver = _FakeContextDriver(env)
        self.sigma = env.sigma
        self.degenerate = False
        self.names = ("A",) if features == "a" else ("A", "sin_phase", "cos_phase")
        self.features = features
        self.source = "FakeURB"
        self._env = env

    @property
    def dim(self):
        return len(self.names)

    @property
    def privileged_dim(self):
        return 3

    def A(self, day):
        return self._env.A(int(day))

    def resolve_day(self, host_day):
        return int(host_day)

    def observed(self, day):
        a = self.A(day)
        if self.features == "a":
            return np.array([a], dtype=np.float64)
        ph = (int(day) % self.driver.period) / float(self.driver.period)
        return np.array([a, np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])

    def privileged(self, day):
        a = self.A(day)
        return np.array([a, self.sigma * 0.14 * a, self.sigma * 0.14 * a])

    def check_day(self, day):
        return True

    def report(self):
        pass


class _Args(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, k):          # any flag not set behaves as "not given"
        return None


def make_ctx(env, algo_cfg=None, params=None, with_context=False, **args):
    """Build a ``HostContext`` around a ``FakeURB``."""
    from urb_baselines.host import HostContext
    import torch

    P = {
        "training_eps": 400, "test_eps": 10, "human_learning_episodes": 5,
        "batch_size": 16, "update_every": 1, "lr": 0.003, "num_epochs": 2,
        "num_hidden": 1, "widths": [32, 32], "clip_eps": 0.2,
        "normalize_advantage": True, "entropy_coef": 0.01,
        "eps_init": 1.0, "eps_decay": 0.99, "eps_min": 0.0, "buffer_size": 256,
        "number_of_paths": env.n_paths, "save_every": 1, "plot_every": 10000,
        "smooth_by": 10, "plot_choices": "basic",
        "observations": "previous_agents_plus_start_time",
        "ratio_machines": 0.5, "should_humans_adapt": False,
        "av_behavior": "selfish", "human_model": "gawron", "human_alpha": 0.2,
        "human_beta": -1.5, "human_beta_randomness": 0.1,
        "human_deterministic": True, "path_gen_beta": -5, "num_samples": 10,
        "path_gen_workers": 1,
    }
    P.update(params or {})
    return HostContext(
        env=env, args=_Args(**args), params=P, algo_cfg=dict(algo_cfg or {}),
        device=torch.device("cpu"), av_ids=list(env.av_ids),
        agent_table=env.agent_table, agent_lookup={},
        machine_ids=set(env.av_ids), obs_size=env.obs_size,
        n_actions=env.n_paths, free_flow=dict(env.free_flow),
        context=_FakeContext(env) if with_context else None, routes_csv=None,
        records_folder=".", plots_folder=".", exp_id="selftest",
        network="fake", repo_root=".", env_seed=0, torch_seed=0,
        training_eps=int(P["training_eps"]), test_eps=int(P["test_eps"]),
        name="selftest",
    )

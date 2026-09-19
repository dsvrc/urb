"""The shared URB host every baseline runs through.

WHY THE LOOP IS SHARED AND THE ALGORITHM IS NOT
--------------------------------------------------------------------------------
``paper/BASELINES.md`` rule 4: "every baseline is launched through the same entry
point under the same severity parameter". The strongest version of that rule is
that they also run the same *code*: the same config merge, the same
``TrafficEnvironment`` arguments, the same human-learning phase, the same
mutation call, the same day loop, the same deterministic test phase and the same
``run_metrics_analysis``. Then a difference between two arms cannot be a
difference in how the host was driven, because there is only one host.

What is NOT shared is the method. Each baseline is a separate algorithm in
``urb_baselines/algos/<name>.py``, launched by its own ``scripts/<name>.py``,
configured by its own ``config/algo_config/<name>/``. No baseline is a flag on
another baseline, and none of them is a flag on PACT-1.

THE HOOK PROTOCOL
--------------------------------------------------------------------------------
``BaselineAlgorithm`` is what the loop calls. Everything except ``act`` has a
no-op default, so a method with the shape of URB's own IPPO or IQL writes three
methods and inherits the rest::

    __init__(ctx)                  after mutation; build networks here
    begin_episode(day, phase)      start of a day, AFTER env.reset()
    act(agent_id, obs) -> int      that agent's turn
    push(agent_id, reward)         that agent's reward, same day
    learn(day)                     after the day, if day % update_every == 0
    end_episode(day, phase, info)  after the day, exactly once
    begin_test()                   freeze / go deterministic
    loss_records() -> list[dict]   rows for losses.csv
    diagnostics() -> dict          scalars printed every print_every days
    close()

Three class flags change what the host does for you::

    needs_records = True     drain RouteRL's per-day CSVs into ``info``; also
                             force save_every to 1 and abort if no record ever
                             arrives (a method that learns from records and
                             receives none is inert, and its curve looks
                             perfectly healthy -- guide II.7)
    needs_context = True     build the DriverContext and refuse to run without an
                             exogenous driver
    needs_obs_trace = True   keep ``info.obs``: the observation each agent
                             actually acted on, this day

THE TWO URB FACTS EVERY ALGORITHM HERE IS WRITTEN AGAINST
--------------------------------------------------------------------------------
* a day is ONE step: ``gamma`` has nothing to discount, and URB's own IQL uses
  ``target = reward``. Every off-policy baseline here does the same.
* a day is SEQUENTIAL: travellers act in start-time order and see the routes
  earlier same-OD travellers took TODAY. There is no instant at which all current
  observations coexist, so anything defined on a simultaneous joint observation
  has to name its snapshot. Each one does, in its checklist.

ONE DIFFERENCE FROM ``scripts/ippo.py``, STATED
--------------------------------------------------------------------------------
``ippo.py`` calls ``model.learn()`` inside the agent loop, at the moment each
agent terminates; this host calls ``algo.learn(day)`` once, after the day. For
independent learners the two are identical up to the order of RNG draws (each
agent's update reads only its own memory). For HAPPO they are not: its sequential
update needs every agent's data for the day to be present before the first agent
is updated, which the in-loop form cannot provide. One loop, the correct one.
"""

import argparse
import ast
import json
import logging
import os
import random
import sys

import numpy as np
import torch

from urb_baselines.context import DriverContext
from urb_baselines.records import RecordSource, flatten_free_flow, split_records

__all__ = ["BaselineAlgorithm", "HostContext", "EpisodeInfo", "main",
           "load_config", "aid"]


# ==========================================================================
#  helpers
# ==========================================================================
def aid(v):
    """Normalise an agent id so ``5``, ``5.0`` and ``'5'`` all key the same dict."""
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return str(v)


def load_config(folder, name, what):
    """Load a URB config, and on a miss say what IS available.

    Same failure mode ``scripts/pact1.py`` hit: URB's README advertises configs
    this distribution does not ship, and the first symptom is a bare
    FileNotFoundError from deep inside the parameter block.
    """
    path = os.path.join(folder, f"{name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    try:
        avail = sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".json"))
    except OSError:
        raise FileNotFoundError(
            f"[URB-BL] {what} directory does not exist: {os.path.abspath(folder)}"
        ) from None
    raise FileNotFoundError(
        f"[URB-BL] {what} {name!r} not found in {os.path.abspath(folder)}.\n"
        f"        Available: {', '.join(avail) if avail else '(none)'}\n"
        f"        NOTE: every arm -- PACT-1 and every baseline -- must use the\n"
        f"        SAME task config, or the comparison measures the scenario\n"
        f"        rather than the method."
    )


class EpisodeInfo(object):
    """What the host knows at the end of a day.

    Attributes:
        day:        AV-day index, 0-based, continuing through the test phase.
        phase:      ``"train"`` or ``"test"``.
        rewards:    ``{agent_id: reward}`` for machine agents.
        actions:    ``{agent_id: action}`` the host actually issued.
        obs:        ``{agent_id: np.ndarray}`` the observation each agent acted on
                    (only when ``needs_obs_trace``).
        av_records: ``{agent_id: (action, travel_time)}``, this day, or ``{}``.
        peer_acts:  ``{agent_id: action}`` for EVERY traveller, humans included,
                    or ``{}``.
        tt_hdv:     mean human travel time, or NaN.
        batches:    every record batch drained this day, ``[(day_label, av,
                    peers, tt_hdv)]``. Normally length 0 or 1 because
                    ``needs_records`` forces ``save_every = 1``.
        context:    observed context vector, or ``None``.
    """

    __slots__ = ("day", "phase", "rewards", "actions", "obs", "av_records",
                 "peer_acts", "tt_hdv", "batches", "context")

    def __init__(self, day, phase):
        self.day = int(day)
        self.phase = str(phase)
        self.rewards = {}
        self.actions = {}
        self.obs = {}
        self.av_records = {}
        self.peer_acts = {}
        self.tt_hdv = float("nan")
        self.batches = []
        self.context = None


class HostContext(object):
    """Everything an algorithm needs to build itself. Handed to ``__init__``."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ==========================================================================
#  the protocol
# ==========================================================================
class BaselineAlgorithm(object):
    """Base class with no-op hooks. See the module docstring for the contract."""

    #: drain RouteRL's per-day records into ``EpisodeInfo``
    needs_records = False
    #: build the exogenous-driver context and refuse to run without one
    needs_context = False
    #: keep ``EpisodeInfo.obs`` populated
    needs_obs_trace = False
    #: short name used in log lines
    name = "baseline"

    def __init__(self, ctx):
        self.ctx = ctx
        self.cfg = dict(ctx.algo_cfg)
        self.params = ctx.params
        self.device = ctx.device
        self.av_ids = list(ctx.av_ids)
        self.slot_of = {a: i for i, a in enumerate(self.av_ids)}
        self.n_av = len(self.av_ids)
        self.n_actions = int(ctx.n_actions)
        self.obs_size = int(ctx.obs_size)
        self.deterministic = False

    # ---- hooks ----
    def begin_episode(self, day, phase):
        pass

    def act(self, agent_id, obs):
        raise NotImplementedError

    def push(self, agent_id, reward):
        pass

    def learn(self, day):
        pass

    def end_episode(self, day, phase, info):
        pass

    def begin_test(self):
        self.deterministic = True

    def loss_records(self):
        return []

    def diagnostics(self):
        return {}

    def close(self):
        pass

    # ---- convenience ----
    def banner(self, title, rows):
        print("\n" + "=" * 74)
        print(f"[URB-BL] {title}")
        print("=" * 74)
        for k, v in rows:
            print(f"  {k:<22} {v}")
        print("=" * 74 + "\n", flush=True)


# ==========================================================================
#  the host
# ==========================================================================
def _base_parser(name):
    p = argparse.ArgumentParser(
        description=f"{name} on URB (see docs/baselines/CHECKLIST_{name}.md)")
    p.add_argument('--id', type=str, required=True)
    p.add_argument('--env-conf', type=str, default="config1")
    p.add_argument('--task-conf', type=str, required=True)
    p.add_argument('--alg-conf', type=str, required=True)
    p.add_argument('--net', type=str, required=True)
    p.add_argument('--env-seed', type=int, default=42)
    p.add_argument('--torch-seed', type=int, default=42)
    p.add_argument('--mode', type=str, default="train", choices=["train", "dry"],
                   help="dry: build everything, run every startup gate, print the "
                        "banner and EXIT before any AV day is simulated, so a "
                        "miswire costs seconds instead of a run.")
    p.add_argument('--dry-human-eps', type=int, default=10)
    p.add_argument('--allow-no-driver', action='store_true',
                   help="context-driven arms only: run without the NS severity "
                        "wrapper, with a constant-zero context. The arm is then "
                        "degenerate and every banner says so.")
    p.add_argument('--routes', type=str, default=None,
                   help="route table (routes.csv); found automatically if omitted")
    return p


def _find_routes_csv(explicit, records_folder, network, repo_root):
    """Locate the generated route table.

    An explicit path that does not exist is a typo, not a request to use some
    other city's routes, so it is never substituted.
    """
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(
                f"[URB-BL] --routes was given explicitly but does not exist: "
                f"{os.path.abspath(explicit)}")
        return explicit
    for cand in (os.path.join(records_folder, "routes.csv"),
                 os.path.join(records_folder, "paths.csv"),
                 os.path.join(repo_root, "networks", network, "routes.csv")):
        if os.path.exists(cand):
            return cand
    return None


def main(name, algo_cls, extra_args=None, description=None):
    """Run one baseline end to end.

    Args:
        name:       algorithm slug. Also the ``config/algo_config/<name>/``
                    directory and the key of the algorithm's own config block.
        algo_cls:   a ``BaselineAlgorithm`` subclass; constructed as
                    ``algo_cls(ctx)`` after mutation.
        extra_args: ``callable(parser)`` adding algorithm-only flags.
    """
    parser = _base_parser(name)
    if description:
        parser.description = description
    if extra_args is not None:
        extra_args(parser)
    args = parser.parse_args()

    # routerl, pandas and tqdm are imported HERE rather than at module import
    # time for two reasons. (1) `scripts/ns_launch.py` replaces
    # `routerl.TrafficEnvironment` with the severity subclass before it runs the
    # target script; taking the name at call time makes it impossible for this
    # host to capture the unwrapped class by accident. (2) every algorithm module
    # imports this one, so a module-level routerl import would make
    # `urb_baselines/selftest.py` unrunnable on a machine without SUMO -- and an
    # offline gate nobody can run is an offline gate nobody runs.
    import pandas as pd
    from tqdm import tqdm
    from routerl import TrafficEnvironment

    exp_id, network = args.id, args.net
    env_seed, torch_seed = args.env_seed, args.torch_seed

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {name.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {args.alg_conf}")
    print(f"Environment config: {args.env_conf}")
    print(f"Task config: {args.task_conf}")
    print(f"Mode: {args.mode}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    print("Device is: ", device)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # ---------------------------------------------------------------- config
    params = {}
    params.update(load_config(f"../config/algo_config/{name}", args.alg_conf,
                              "algorithm config"))
    params.update(load_config("../config/env_config", args.env_conf,
                              "environment config"))
    params.update(load_config("../config/task_config", args.task_conf,
                              "task config"))
    params.pop("desc", None)
    algo_cfg = dict(params.pop(name, {}))

    P = params
    training_eps = int(P["training_eps"])
    test_eps = int(P["test_eps"])
    human_learning_episodes = int(P["human_learning_episodes"])
    update_every = int(P.get("update_every", 1))
    plot_every = int(P.get("plot_every", 50))
    number_of_paths = int(P["number_of_paths"])

    needs_records = bool(getattr(algo_cls, "needs_records", False))
    needs_context = bool(getattr(algo_cls, "needs_context", False))
    needs_obs_trace = bool(getattr(algo_cls, "needs_obs_trace", False))

    save_every = int(P.get("save_every", 5))
    if needs_records and save_every != 1:
        print(f"[URB-BL] save_every {save_every} -> 1: this arm learns from "
              f"RouteRL's per-day records and a batched flush would delay every "
              f"update by up to {save_every} days. DISK setting only -- it changes "
              f"no dynamics and nothing the arms are compared on.", flush=True)
        save_every = 1

    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, training_eps + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        data = ast.literal_eval(f.read())
    origins, destinations = data['origins'], data['destinations']

    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    if not os.path.exists(agents_csv_path):
        raise FileNotFoundError(f"Agents CSV file not found at {agents_csv_path}.")
    num_agents = len(pd.read_csv(agents_csv_path))
    os.makedirs(records_folder, exist_ok=True)
    new_agents_csv_path = os.path.join(records_folder, "agents.csv")
    with open(agents_csv_path, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
        f.write(content)
    max_start_time = pd.read_csv(new_agents_csv_path)['start_time'].max()

    num_machines = int(num_agents * float(P["ratio_machines"]))
    total_episodes = human_learning_episodes + training_eps + test_eps

    from utils import (clear_SUMO_files, print_agent_counts,
                       run_metrics_analysis, save_loss_records,
                       script_path_for_config)

    dump_config = dict(params)
    dump_config.update({
        "network": network, "env_seed": env_seed, "torch_seed": torch_seed,
        "env_config": args.env_conf, "task_config": args.task_conf,
        "alg_config": args.alg_conf,
        "script": script_path_for_config(os.path.abspath(sys.argv[0]), repo_root),
        "algorithm": name, name: algo_cfg,
        "num_agents": num_agents, "num_machines": num_machines,
        "cli": {k: v for k, v in vars(args).items()},
    })
    with open(os.path.join(records_folder, "exp_config.json"), 'w',
              encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4, default=str)

    # ---------------------------------------------------------------- env
    env = TrafficEnvironment(
        seed=env_seed,
        create_agents=False,
        create_paths=True,
        save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": num_machines,
            "human_parameters": {
                "model": P["human_model"],
                "alpha": P["human_alpha"],
                "beta": P["human_beta"],
                "beta_randomness": P["human_beta_randomness"],
                "deterministic": P["human_deterministic"],
            },
            "machine_parameters": {
                "behavior": P["av_behavior"],
                "observation_type": P["observations"],
            },
        },
        environment_parameters={"save_every": save_every},
        simulator_parameters={
            "network_name": network,
            "custom_network_folder": custom_network_folder,
            "sumo_type": "sumo",
            "simulation_timesteps": max_start_time,
        },
        plotter_parameters={
            "phases": phases, "phase_names": phase_names,
            "smooth_by": P["smooth_by"], "plot_choices": P["plot_choices"],
            "records_folder": records_folder, "plots_folder": plots_folder,
        },
        path_generation_parameters={
            "origins": origins, "destinations": destinations,
            "number_of_paths": number_of_paths,
            "beta": P["path_gen_beta"], "num_samples": P["num_samples"],
            "path_gen_workers": P["path_gen_workers"],
            "visualize_paths": False,
        },
    )

    env.start()
    env.reset()
    print_agent_counts(env)

    # ---------------------------------------------------------------- humans
    n_human = human_learning_episodes
    if args.mode == "dry":
        n_human = min(n_human, int(args.dry_human_eps))
        print(f"[URB-BL] --mode dry: {n_human} human-learning episodes (of "
              f"{human_learning_episodes}). This is a WIRING test, not a "
              f"behavioural one.", flush=True)
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for _ in range(n_human):
        env.step()
        pbar.update()

    env.mutation(disable_human_learning=not P["should_humans_adapt"],
                 mutation_start_percentile=-1)
    print_agent_counts(env)

    # ---------------------------------------------------------------- tables
    machine_ids = {aid(a.id) for a in env.machine_agents}
    agent_table = {}
    for a in env.all_agents:
        key = aid(a.id)
        agent_table[key] = {
            "od": (int(a.origin), int(a.destination)),
            "start": float(a.start_time),
            "machine": key in machine_ids,
        }
    av_ids = [aid(a.id) for a in env.machine_agents]
    agent_lookup = {aid(a.id): a for a in env.machine_agents}

    k_sizes = {int(a.action_space_size) for a in env.machine_agents}
    if len(k_sizes) != 1 or sorted(k_sizes)[0] != number_of_paths:
        raise AssertionError(
            f"[URB-BL] machine agents disagree on the action-space size "
            f"({sorted(k_sizes)}) or it differs from number_of_paths="
            f"{number_of_paths}. Every baseline here assumes one K for the fleet.")
    n_actions = number_of_paths
    obs_size = int(env.observation_space(env.possible_agents[0]).shape[0])

    free_flow = flatten_free_flow(env.get_free_flow_times())

    context = None
    if needs_context:
        context = DriverContext(
            env,
            features=str(algo_cfg.get("context_features", "a")),
            require=not args.allow_no_driver,
            verbose=True,
        )

    ctx = HostContext(
        env=env, args=args, params=params, algo_cfg=algo_cfg, device=device,
        av_ids=av_ids, agent_table=agent_table, agent_lookup=agent_lookup,
        machine_ids=machine_ids, obs_size=obs_size, n_actions=n_actions,
        free_flow=free_flow, context=context,
        routes_csv=_find_routes_csv(args.routes, records_folder, network,
                                    repo_root),
        records_folder=records_folder, plots_folder=plots_folder,
        exp_id=exp_id, network=network, repo_root=repo_root,
        env_seed=env_seed, torch_seed=torch_seed,
        training_eps=training_eps, test_eps=test_eps, name=name,
    )

    algo = algo_cls(ctx)
    print(f"[URB-BL] observation {obs_size}, actions {n_actions}, "
          f"{len(av_ids)} machine agents, {len(agent_table)} travellers",
          flush=True)

    if args.mode == "dry":
        print("\n[URB-BL] --mode dry: every startup gate passed. Exiting before "
              "any AV day was simulated.")
        print("[URB-BL] The arm's closing report is NOT printed in dry mode: "
              "nothing has run, so every mechanism would correctly report itself "
              "as never having engaged, and those warnings would mean nothing "
              "here.\n", flush=True)
        if context is not None:
            context.report()
        env.stop_simulation()
        return

    # ---------------------------------------------------------------- loop
    src = RecordSource(env, records_folder) if needs_records else None
    state = {"seen": False, "offset": None}
    grace = 12
    print_every = int(algo_cfg.get("print_every", 100))
    os.makedirs(plots_folder, exist_ok=True)
    pbar.set_description("AV learning")

    def run_day(day, phase):
        """One URB day. ``end_episode`` is called exactly once, at the end."""
        info = EpisodeInfo(day, phase)
        # ORDER MATTERS. `env.reset()` is what makes the severity wrapper resolve
        # THIS day's weather: it calls `SeverityLayer.begin_episode`, which sets
        # `_A` and the per-link capacity `_g`. Reading the privileged context
        # before that would hand RMA's teacher YESTERDAY's capacity state -- an
        # off-by-one that nothing downstream could detect, because A(day) comes
        # straight from the driver and would look perfectly correct beside it.
        env.reset()
        if context is not None:
            if not context.check_day(day):
                # A CONSTANT offset between the two counters is absorbed by
                # DriverContext (and printed once). Reaching here means the
                # offset CHANGED mid-run, which nothing can absorb: from this
                # point on the context and the dial describe different days.
                raise AssertionError(
                    f"[URB-BL][GATE FAIL] the severity wrapper's day counter "
                    f"desynchronised from the host's at AV-day {day} "
                    f"(env._ns_day = {getattr(env, '_ns_day', '?')}). Every "
                    f"context published after this point would describe a "
                    f"different day's weather from the one the fleet drove.")
            info.context = context.observed(day)
        algo.begin_episode(day, phase)

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()
            a = aid(agent_id)
            if termination or truncation:
                info.rewards[a] = float(reward)
                algo.push(a, float(reward))
                action = None
            else:
                action = int(algo.act(a, observation))
                info.actions[a] = action
                if needs_obs_trace:
                    info.obs[a] = np.asarray(observation,
                                             dtype=np.float64).reshape(-1)
            env.step(action)

        if needs_records:
            batches = src.drain(day)
            if batches and state["offset"] is None:
                # RouteRL numbers episodes from the start of the RUN, so its
                # labels include the human-learning days. Infer the offset once.
                state["offset"] = batches[-1][0] - day
                print(f"[URB-BL] RouteRL episode numbering offset by "
                      f"{state['offset']} (its ep{batches[-1][0]} == AV day "
                      f"{day})", flush=True)
            off = state["offset"] or 0
            for ep_label, recs in batches:
                av_rec, peer_act, tt_hdv, n_bad = split_records(recs, machine_ids)
                if not av_rec:
                    continue
                label = ep_label - off
                info.batches.append((label, av_rec, peer_act, tt_hdv))
                if not state["seen"]:
                    state["seen"] = True
                    both = [x for x in info.actions if x in peer_act]
                    agree = sum(1 for x in both if peer_act[x] == info.actions[x])
                    print(f"[URB-BL] first records at AV day {day} "
                          f"(labelled ep{ep_label}): {len(recs)} drained, "
                          f"{len(av_rec)} AV, {len(peer_act) - len(av_rec)} human, "
                          f"{n_bad} unparseable; action cross-check "
                          f"{agree}/{len(both)}", flush=True)
                    if both and agree < len(both):
                        print("[URB-BL][WARN] recorded and issued actions "
                              "disagree. Check the id normalisation before "
                              "trusting anything computed from these records.",
                              flush=True)
            for label, av_rec, peer_act, tt_hdv in info.batches:
                if label == day:            # today's, which is the normal case
                    info.av_records, info.peer_acts, info.tt_hdv = \
                        av_rec, peer_act, tt_hdv
                    break
            else:
                if info.batches:            # only a stale flush arrived
                    _, info.av_records, info.peer_acts, info.tt_hdv = \
                        info.batches[-1]
            if (not state["seen"]) and day >= grace:
                raise AssertionError(
                    f"[URB-BL][GATE FAIL] no AV travel-time records after "
                    f"{day + 1} AV days, so this arm has never received a single "
                    f"row and is silently inert.\n" + src.diagnose())

        algo.end_episode(day, phase, info)
        return info

    for day in range(training_eps):
        info = run_day(day, "train")
        if day % update_every == 0:
            algo.learn(day)
        if print_every and day % print_every == 0:
            diag = algo.diagnostics()
            r = list(info.rewards.values())
            line = f"[{name}] day {day:5d}"
            if r:
                line += f"  reward {float(np.mean(r)):9.2f}"
            if diag:
                line += "  " + "  ".join(
                    (f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
                    for k, v in diag.items())
            print(line, flush=True)
        if day % plot_every == 0:
            env.plot_results()
        pbar.update()

    # ---------------------------------------------------------------- test
    algo.begin_test()
    pbar.set_description("Testing")
    for i in range(test_eps):
        run_day(training_eps + i, "test")
        pbar.update()

    # ---------------------------------------------------------------- finish
    pbar.close()
    env.plot_results()
    rows = algo.loss_records()
    if rows:
        save_loss_records(records_folder, rows, columns=list(rows[0].keys()))

    algo.close()
    if context is not None:
        context.report()
    if hasattr(env, "severity_report"):
        # Nothing else in the repo calls this, and a silently inert dial looks
        # exactly like a clean null result. It costs one print.
        env.severity_report()
    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"),
                     os.path.join(records_folder, "episodes"),
                     remove_additional_files=True)
    run_metrics_analysis(exp_id, results_folder="../results")

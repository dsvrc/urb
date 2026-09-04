"""Run ANY custom-loop URB algorithm under the NS severity dial.

    python scripts/ns_run.py --host ippo --sigma 1.0 --id sai_ns_ippo_s1 \
        --alg-conf config1 --task-conf config1 --net saint_arnoult --env-seed 42

WHY THIS SCRIPT EXISTS
--------------------------------------------------------------------------------
`NS_FORM_SPEC` B.5:

> Severity is TASK physics -- it must reach every arm. A dial only the method's arm
> experienced is worthless as evidence.

URB's own scripts (`ippo.py`, `iql.py`, `baselines.py`) construct
`TrafficEnvironment` directly, so they run at sigma = 0 whatever you pass them.
This runner rebuilds their loop with the `SeverityLayer` interposed, so a baseline
and PACT face the identical dial, the identical weather, and the identical seed.

    TrafficEnvironment -> SeverityLayer -> host algorithm

HOSTS
--------------------------------------------------------------------------------
    ippo     URB's PPO             (scripts/ippo.py)
    iql      URB's DQN             (scripts/iql.py)
    aon      all-or-nothing        (baseline_models/aon.py)
    random   uniform route choice  (baseline_models/random.py)
    gawron   RouteRL's human model, used as an AV baseline

`sigma = 0` reproduces the stock task byte for byte, so the SAME command gives the
no-severity row -- which is what keeps the two tables comparable.

For the TorchRL hosts (mappo / qmix / vdn / *_torchrl) the reward path runs inside
a TorchRL collector rather than this loop, so they need a `Transform` instead;
they are NOT covered here and must not be quoted as NS baselines.
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import json
import logging
import random

import numpy as np
import pandas as pd
import torch

from routerl import Keychain as kc
from routerl import TrafficEnvironment
from tqdm import tqdm

from ippo import PPO
from iql import DQN
from utils import clear_SUMO_files, print_agent_counts, run_metrics_analysis
from utils import save_loss_records, script_path_for_config

from pact1.records import RecordSource, split_records
from urb_ns.driver import HCM_HEAVY_RAIN_LOSS, WeatherDriver
from urb_ns.network import RoadNetwork, load_route_table, parse_sumo_net
from urb_ns.severity_env import SeverityLayer

from ns_certify import build_loading, find_routes_csv
from pact_ns import RECORD_KEYS, _aid, load_config

LEARNERS = {"ippo", "iql"}
BASELINES = {"aon", "random", "gawron"}
HOSTS = sorted(LEARNERS | BASELINES)


def make_model(host, env, agent, obs_size, device, params, free_flow):
    """Build the host's per-agent model, with URB's own hyperparameters."""
    k = int(agent.action_space_size)
    if host == "ippo":
        return PPO(obs_size, k, device=device,
                   batch_size=params["batch_size"], lr=params["lr"],
                   num_epochs=params["num_epochs"],
                   num_hidden=params["num_hidden"], widths=params["widths"],
                   clip_eps=params["clip_eps"],
                   normalize_advantage=params["normalize_advantage"],
                   entropy_coef=params["entropy_coef"])
    if host == "iql":
        return DQN(obs_size, k, device=device,
                   eps_init=params.get("eps_init", 0.99),
                   eps_decay=params.get("eps_decay", 0.998),
                   eps_min=params.get("eps_min", 0.0),
                   buffer_size=params.get("buffer_size", 256),
                   batch_size=params["batch_size"], lr=params["lr"],
                   num_epochs=params["num_epochs"],
                   num_hidden=params["num_hidden"], widths=params["widths"])
    from baseline_models import get_baseline
    ik = free_flow.get((agent.origin, agent.destination), [0.0] * k)
    return get_baseline({**params, "model": host}, list(ik))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, choices=HOSTS)
    ap.add_argument("--id", required=True)
    ap.add_argument("--alg-conf", required=True)
    ap.add_argument("--env-conf", default="config1")
    ap.add_argument("--task-conf", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--env-seed", type=int, default=42)
    ap.add_argument("--torch-seed", type=int, default=42)
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="0 = stock URB byte for byte; 1 = the HCM heavy-rain "
                         "factor; >1 a labelled beyond-physical stress test")
    ap.add_argument("--routes", default=None)
    ap.add_argument("--probe", type=int, default=0)
    args = ap.parse_args()

    print("### STARTING EXPERIMENT ###")
    for k, v in (("Host", args.host.upper()), ("Experiment ID", args.id),
                 ("Network", args.net), ("Task config", args.task_conf),
                 ("sigma", args.sigma)):
        print(f"{k}: {v}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.env_seed)
    np.random.seed(args.env_seed)
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

    cfg_dir = ("baseline" if args.host in BASELINES else args.host)
    params = {}
    params.update(load_config(f"../config/algo_config/{cfg_dir}", args.alg_conf,
                              "algorithm config"))
    params.update(load_config("../config/env_config", args.env_conf,
                              "environment config"))
    params.update(load_config("../config/task_config", args.task_conf,
                              "task config"))
    params.pop("desc", None)
    ns_cfg = dict(params.pop("ns", {}))

    records_folder = f"../results/{args.id}"
    plots_folder = f"{records_folder}/plots"
    net_folder = f"../networks/{args.net}"
    training_eps = int(params["training_eps"])
    hle = int(params["human_learning_episodes"])
    test_eps = int(params["test_eps"])
    with open(os.path.join(net_folder, f"od_{args.net}.txt"), encoding="utf-8") as f:
        data = ast.literal_eval(f.read())

    os.makedirs(records_folder, exist_ok=True)
    with open(os.path.join(net_folder, "agents.csv"), encoding="utf-8") as f:
        content = f.read()
    with open(os.path.join(records_folder, "agents.csv"), "w",
              encoding="utf-8") as f:
        f.write(content)
    adf = pd.read_csv(os.path.join(net_folder, "agents.csv"))
    n_agents, max_start = len(adf), adf["start_time"].max()
    n_machines = int(n_agents * params["ratio_machines"])
    total_eps = hle + training_eps + test_eps

    with open(os.path.join(records_folder, "exp_config.json"), "w",
              encoding="utf-8") as f:
        json.dump({**params, "network": args.net, "env_seed": args.env_seed,
                   "torch_seed": args.torch_seed, "algorithm": f"ns_{args.host}",
                   "host": args.host, "sigma": args.sigma, "ns": ns_cfg,
                   "task_config": args.task_conf, "alg_config": args.alg_conf,
                   "num_agents": n_agents, "num_machines": n_machines,
                   "script": script_path_for_config(__file__)}, f, indent=4)

    env = TrafficEnvironment(
        seed=args.env_seed, create_agents=False, create_paths=True,
        save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": n_machines,
            "human_parameters": {
                "model": params["human_model"], "alpha": params["human_alpha"],
                "beta": params["human_beta"],
                "beta_randomness": params["human_beta_randomness"],
                "deterministic": params["human_deterministic"]},
            "machine_parameters": {"behavior": params["av_behavior"],
                                   "observation_type": params["observations"]}},
        environment_parameters={"save_every": 1},
        simulator_parameters={"network_name": args.net,
                              "custom_network_folder": net_folder,
                              "sumo_type": "sumo",
                              "simulation_timesteps": max_start},
        plotter_parameters={
            "phases": [1, hle, training_eps + hle],
            "phase_names": ["Human stabilization", "Mutation and AV learning",
                            "Testing phase"],
            "smooth_by": params["smooth_by"],
            "plot_choices": params["plot_choices"],
            "records_folder": records_folder, "plots_folder": plots_folder},
        path_generation_parameters={
            "origins": data["origins"], "destinations": data["destinations"],
            "number_of_paths": params["number_of_paths"],
            "beta": params["path_gen_beta"], "num_samples": params["num_samples"],
            "path_gen_workers": params["path_gen_workers"],
            "visualize_paths": False})
    env.start()
    env.reset()
    print_agent_counts(env)

    pbar = tqdm(total=total_eps, desc="Human learning")
    for _ in range(hle):
        env.step()
        pbar.update()
    env.mutation(disable_human_learning=not params["should_humans_adapt"],
                 mutation_start_percentile=-1)
    print_agent_counts(env)

    # ================================================================ severity
    network = RoadNetwork(
        parse_sumo_net(os.path.join(net_folder, f"{args.net}.net.xml")),
        *load_route_table(find_routes_csv(args.routes, args.net),
                          int(params["number_of_paths"])),
        int(params["number_of_paths"]),
        sat_flow=float(ns_cfg.get("sat_flow", 600.0)))
    free_flow = env.get_free_flow_times()
    machine_ids = {_aid(a.id) for a in env.machine_agents}
    ordered = sorted(env.all_agents, key=lambda a: int(a.id))
    slot_of = {_aid(a.id): i for i, a in enumerate(ordered)}
    ag_tbl = pd.DataFrame({
        "id": [int(a.id) for a in ordered],
        "origin": [int(a.origin) for a in ordered],
        "destination": [int(a.destination) for a in ordered],
        "start_time": [float(a.start_time) for a in ordered]})
    loading = build_loading(network, ag_tbl, network.ffts_by_od, 0.0,
                            args.env_seed)
    loading.is_machine = np.array([_aid(a.id) in machine_ids for a in ordered],
                                  dtype=bool)
    driver = WeatherDriver(int(ns_cfg.get("period", 100)),
                           float(ns_cfg.get("wet_frac", 0.5)),
                           float(ns_cfg.get("loss", HCM_HEAVY_RAIN_LOSS)),
                           bool(ns_cfg.get("mean_preserving", False)))
    severity = SeverityLayer(network, driver, loading, args.sigma,
                             av_behavior=params["av_behavior"], max_offset_s=0.0)

    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    for a in env.machine_agents:
        a.model = make_model(args.host, env, a, obs_size, device, params,
                             free_flow)
    lookup = {_aid(a.id): a for a in env.machine_agents}
    is_learner = args.host in LEARNERS
    src = RecordSource(env, records_folder)
    n_train = int(args.probe) if args.probe else training_eps
    pbar.set_description(f"{args.host} [sigma={args.sigma}]")

    def run_day(day, learn):
        severity.begin_episode(day)
        env.reset()
        chosen, last_state = {}, {}
        for agent_id in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            aid = _aid(agent_id)
            model = lookup[aid].model
            if term or trunc:
                r = severity.harm_reward(slot_of.get(aid, -1), reward)
                if is_learner:
                    model.push(r)
                    if learn and day % params["update_every"] == 0:
                        model.learn()
                elif learn and hasattr(model, "learn"):
                    model.learn(last_state.get(aid), chosen.get(aid), r)
                action = None
            else:
                action = model.act(obs)
                chosen[aid] = int(action)
                last_state[aid] = obs
            env.step(action)

        choice = np.zeros(loading.N, dtype=np.int64)
        for _lbl, recs in src.drain(day):
            _av, peer_act, _hdv, _bad = split_records(recs, machine_ids,
                                                      RECORD_KEYS, _aid)
            for aid, k in peer_act.items():
                if aid in slot_of:
                    choice[slot_of[aid]] = int(k)
        for aid, k in chosen.items():
            if aid in slot_of:
                choice[slot_of[aid]] = k
        severity.end_episode(choice)
        pbar.update()

    for day in range(n_train):
        run_day(day, learn=True)
        if not args.probe and day % params["plot_every"] == 0:
            env.plot_results()

    if args.probe:
        print(f"\n[NS-RUN] probe of {n_train} days complete.")
        env.stop_simulation()
        return

    for a in env.machine_agents:
        if hasattr(a.model, "policy_net"):
            a.model.policy_net.eval()
        if hasattr(a.model, "q_network"):
            a.model.q_network.eval()
        if hasattr(a.model, "epsilon"):
            a.model.epsilon = 0.0
        a.model.deterministic = True
    pbar.set_description("Testing")
    for ep in range(test_eps):
        run_day(training_eps + ep, learn=False)

    pbar.close()
    env.plot_results()
    if is_learner:
        save_loss_records(records_folder,
                          [{"iteration": i, "agent_id": a.id, "loss": v}
                           for a in env.machine_agents
                           for i, v in enumerate(getattr(a.model, "loss", []),
                                                 start=1)],
                          columns=["iteration", "agent_id", "loss"])
    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"),
                     os.path.join(records_folder, "episodes"),
                     remove_additional_files=True)
    run_metrics_analysis(args.id, results_folder="../results")


if __name__ == "__main__":
    main()

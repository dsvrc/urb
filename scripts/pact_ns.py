"""PACT on the URB NS. Runs like any other URB algorithm script.

    python scripts/pact_ns.py --id sai_pact_s1 --alg-conf config1 \
        --task-conf config1 --net saint_arnoult --env-seed 42 --torch-seed 0 \
        --sigma 1.0 --arm pact

ARMS -- and the middle one is not optional
--------------------------------------------------------------------------------
    --arm blind   host IPPO under the NS. Offsets exactly zero, so the executed
                  day is byte-identical to the stock task. THE baseline.
    --arm ff      host + the ANALYTIC FEEDFORWARD only. Local, needs no peer
                  information, and any method with the domain model can compute
                  it. `PACT_PIPELINE_SPEC` §7 honesty condition 2 makes this
                  MANDATORY: without it a PACT-vs-blind gap is INFORMATION, not
                  mechanism.
    --arm pact    feedforward + the identified peer term. The coordination claim
                  rests on `pact` beating `ff`, not on `pact` beating `blind`.

LAYERING (`NS_FORM_SPEC` B.5)
--------------------------------------------------------------------------------
    TrafficEnvironment -> SeverityLayer (every arm) -> PACT (this arm only)

`--sigma` is TASK physics and is applied identically in all three arms. Baselines
that do not know about the NS at all (qmix_torchrl, ippo) can be run under it via
`--arm blind`, which is the same host algorithm with the compensator disabled.

WHAT THE NS IS
--------------------------------------------------------------------------------
Rain reduces road capacity by the Highway Capacity Manual's capacity adjustment
factor; `sigma = 1` IS the published heavy-rain figure. Capacity enters as the
denominator of the volume-to-capacity ratio, so it multiplies every peer's
contribution to your loading and a lone traveller's cross-agent term stays exactly
zero. Half the weather cycle is dry, where the dial is provably inert -- run the
same sweep there and every row is byte-identical.

Read `results/ns_certificate/` for the structural certificate; it is computed with
no simulator and no method, and it should be committed before this script is run.
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
from iql import Network                                        # noqa: F401
from utils import clear_SUMO_files
from utils import print_agent_counts
from utils import run_metrics_analysis
from utils import save_loss_records
from utils import script_path_for_config

from pact1.records import RecordSource, split_records
from pact_urb.coordinator import PactCoordinator
from urb_ns.driver import HCM_HEAVY_RAIN_LOSS, WeatherDriver
from urb_ns.network import RoadNetwork, load_route_table, parse_sumo_net
from urb_ns.severity_env import SeverityLayer

from ns_certify import build_loading, find_routes_csv

ALGORITHM = "pact_ns"

RECORD_KEYS = {
    "id": [kc.AGENT_ID, "id"],
    "action": [kc.ACTION, "action"],
    "travel_time": [kc.TRAVEL_TIME, "travel_time"],
}


def _aid(v):
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return str(v)


def load_config(folder, name, what):
    path = os.path.join(folder, f"{name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    try:
        avail = sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".json"))
    except OSError:
        raise FileNotFoundError(f"[PACT-NS] no {what} directory: {folder}") from None
    raise FileNotFoundError(
        f"[PACT-NS] {what} {name!r} not found in {os.path.abspath(folder)}.\n"
        f"        Available: {', '.join(avail)}\n"
        "        Every arm must use the SAME task config or the comparison "
        "measures the scenario, not the method."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--alg-conf", required=True)
    ap.add_argument("--env-conf", default="config1")
    ap.add_argument("--task-conf", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--env-seed", type=int, default=42)
    ap.add_argument("--torch-seed", type=int, default=42)
    # --- NS / PACT ---
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="severity. 1.0 = the HCM heavy-rain factor; >1 is a "
                         "labelled beyond-physical stress test.")
    ap.add_argument("--arm", default="pact", choices=["pact", "ff", "blind"])
    ap.add_argument("--routes", default=None)
    ap.add_argument("--probe", type=int, default=0,
                    help="run N days and exit -- the cheap wiring check")
    args = ap.parse_args()

    print("### STARTING EXPERIMENT ###")
    for k, v in (("Algorithm", ALGORITHM.upper()), ("Experiment ID", args.id),
                 ("Network", args.net), ("Task config", args.task_conf),
                 ("sigma", args.sigma), ("arm", args.arm)):
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
    print("Device is: ", device)

    # ---------------------------------------------------------------- config
    params = {}
    alg = load_config(f"../config/algo_config/{ALGORITHM}", args.alg_conf,
                      "algorithm config")
    env_p = load_config("../config/env_config", args.env_conf, "environment config")
    task_p = load_config("../config/task_config", args.task_conf, "task config")
    params.update(alg)
    params.update(env_p)
    params.update(task_p)
    params.pop("desc", None)
    ns_cfg = dict(params.pop("ns", {}))
    pact_cfg = dict(params.pop("pact", {}))
    pact_cfg["arm"] = args.arm
    for k, v in params.items():
        globals()[k] = v

    records_folder = f"../results/{args.id}"
    plots_folder = f"{records_folder}/plots"
    custom_network_folder = f"../networks/{args.net}"
    phases = [1, human_learning_episodes,                       # noqa: F821
              int(training_eps) + human_learning_episodes]      # noqa: F821
    with open(os.path.join(custom_network_folder, f"od_{args.net}.txt"),
              encoding="utf-8") as f:
        data = ast.literal_eval(f.read())
    origins, destinations = data["origins"], data["destinations"]

    os.makedirs(records_folder, exist_ok=True)
    agents_src = os.path.join(custom_network_folder, "agents.csv")
    with open(agents_src, encoding="utf-8") as f:
        content = f.read()
    with open(os.path.join(records_folder, "agents.csv"), "w",
              encoding="utf-8") as f:
        f.write(content)
    agents_df = pd.read_csv(agents_src)
    num_agents = len(agents_df)
    max_start_time = agents_df["start_time"].max()
    num_machines = int(num_agents * ratio_machines)             # noqa: F821
    total_eps = human_learning_episodes + training_eps + test_eps   # noqa: F821

    dump = dict(params)
    dump.update({"network": args.net, "env_seed": args.env_seed,
                 "torch_seed": args.torch_seed, "algorithm": ALGORITHM,
                 "task_config": args.task_conf, "alg_config": args.alg_conf,
                 "script": script_path_for_config(__file__),
                 "num_agents": num_agents, "num_machines": num_machines,
                 "sigma": args.sigma, "arm": args.arm,
                 "ns": ns_cfg, "pact": pact_cfg})
    with open(os.path.join(records_folder, "exp_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(dump, f, indent=4)

    # ---------------------------------------------------------------- env
    env = TrafficEnvironment(
        seed=args.env_seed, create_agents=False, create_paths=True,
        save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": num_machines,
            "human_parameters": {
                "model": human_model, "alpha": human_alpha,          # noqa: F821
                "beta": human_beta,                                  # noqa: F821
                "beta_randomness": human_beta_randomness,            # noqa: F821
                "deterministic": human_deterministic},               # noqa: F821
            "machine_parameters": {"behavior": av_behavior,          # noqa: F821
                                   "observation_type": observations},  # noqa: F821
        },
        environment_parameters={"save_every": 1},
        simulator_parameters={
            "network_name": args.net,
            "custom_network_folder": custom_network_folder,
            "sumo_type": "sumo", "simulation_timesteps": max_start_time},
        plotter_parameters={
            "phases": phases,
            "phase_names": ["Human stabilization", "Mutation and AV learning",
                            "Testing phase"],
            "smooth_by": smooth_by, "plot_choices": plot_choices,   # noqa: F821
            "records_folder": records_folder, "plots_folder": plots_folder},
        path_generation_parameters={
            "origins": origins, "destinations": destinations,
            "number_of_paths": number_of_paths,                    # noqa: F821
            "beta": path_gen_beta, "num_samples": num_samples,     # noqa: F821
            "path_gen_workers": path_gen_workers,                  # noqa: F821
            "visualize_paths": False},
    )
    env.start()
    env.reset()
    print_agent_counts(env)

    pbar = tqdm(total=total_eps, desc="Human learning")
    for _ in range(human_learning_episodes):                       # noqa: F821
        env.step()
        pbar.update()
    env.mutation(disable_human_learning=not should_humans_adapt,   # noqa: F821
                 mutation_start_percentile=-1)
    print_agent_counts(env)

    # ================================================================ NS
    net_edges = parse_sumo_net(
        os.path.join(custom_network_folder, f"{args.net}.net.xml"))
    routes_csv = find_routes_csv(args.routes, args.net)
    routes_by_od, ffts_by_od = load_route_table(routes_csv,
                                                int(number_of_paths))  # noqa: F821
    network = RoadNetwork(net_edges, routes_by_od, ffts_by_od,
                          int(number_of_paths),                   # noqa: F821
                          sat_flow=float(ns_cfg.get("sat_flow", 600.0)))

    free_flow = env.get_free_flow_times()
    n, worst, bad = network.check_fft(free_flow)
    print(f"[PACT-NS] route-order alignment: {'PASS' if not bad else 'FAIL'} "
          f"({n} routes, max rel err {worst:.3e})")
    if bad:
        raise AssertionError(
            "[PACT-NS] routes.csv free-flow times do not match the env's. The "
            f"action index does not line up with the route table. e.g. {bad[:3]}")

    machine_ids = {_aid(a.id) for a in env.machine_agents}
    ordered = sorted(env.all_agents, key=lambda a: int(a.id))
    id_order = [_aid(a.id) for a in ordered]
    slot_of = {a: i for i, a in enumerate(id_order)}
    ag_tbl = pd.DataFrame({
        "id": [int(a.id) for a in ordered],
        "origin": [int(a.origin) for a in ordered],
        "destination": [int(a.destination) for a in ordered],
        "start_time": [float(a.start_time) for a in ordered],
    })
    loading = build_loading(network, ag_tbl, ffts_by_od, 0.0, args.env_seed)
    loading.is_machine = np.array([i in machine_ids for i in id_order], dtype=bool)
    av_slots = np.nonzero(loading.is_machine)[0]

    driver = WeatherDriver(
        period=int(ns_cfg.get("period", 100)),
        wet_frac=float(ns_cfg.get("wet_frac", 0.5)),
        loss=float(ns_cfg.get("loss", HCM_HEAVY_RAIN_LOSS)),
        mean_preserving=bool(ns_cfg.get("mean_preserving", False)))
    ok, _ = driver.certify(sigmas=[0.0, args.sigma])
    if not ok:
        raise AssertionError("[PACT-NS] the severity dial failed its B.1 gates")

    severity = SeverityLayer(
        network, driver, loading, args.sigma, av_behavior=av_behavior,  # noqa: F821
        max_offset_s=(0.0 if args.arm == "blind"
                      else float(pact_cfg.get("max_offset_s", 240.0))))

    pact_cfg.setdefault("debug_dir", repo_root)
    coord = PactCoordinator(network, loading, severity, av_slots, pact_cfg,
                            run_dir=records_folder, exp_id=args.id)

    # ---------------------------------------------------------------- models
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    for a in env.machine_agents:
        a.model = PPO(obs_size, int(a.action_space_size), device=device,
                      batch_size=batch_size, lr=lr, num_epochs=num_epochs,  # noqa: F821
                      num_hidden=num_hidden, widths=widths,              # noqa: F821
                      clip_eps=clip_eps,                                 # noqa: F821
                      normalize_advantage=normalize_advantage,           # noqa: F821
                      entropy_coef=entropy_coef)                         # noqa: F821
    lookup = {_aid(a.id): a for a in env.machine_agents}

    src = RecordSource(env, records_folder)
    n_train = int(args.probe) if args.probe else int(training_eps)  # noqa: F821
    prev_choice = None
    seen = False
    os.makedirs(plots_folder, exist_ok=True)
    pbar.set_description(f"AV learning [{args.arm} sigma={args.sigma}]")

    def run_day(day, phase, learn):
        nonlocal prev_choice, seen
        severity.begin_episode(day)
        coord.plan_offsets(prev_choice)
        env.reset()
        rewards, chosen = [], {}
        for agent_id in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            aid = _aid(agent_id)
            model = lookup[aid].model
            if term or trunc:
                r = severity.harm_reward(slot_of.get(aid, -1), reward)
                rewards.append(float(r))
                model.push(r)
                if learn and day % update_every == 0:               # noqa: F821
                    model.learn()
                action = None
            else:
                action = model.act(obs)
                chosen[aid] = int(action)
            env.step(action)

        choice = np.zeros(loading.N, dtype=np.int64)
        for aid, k in chosen.items():
            if aid in slot_of:
                choice[slot_of[aid]] = k
        records = src.drain(day)
        tt_cav = tt_hdv = float("nan")
        for _lbl, recs in records:
            av_rec, peer_act, tt_hdv_raw, _ = split_records(
                recs, machine_ids, RECORD_KEYS, _aid)
            for aid, k in peer_act.items():
                if aid in slot_of:
                    choice[slot_of[aid]] = int(k)
            if av_rec:
                seen = True
                tts = [severity.harm_travel_time(slot_of[a], t)
                       for a, (_k, t) in av_rec.items() if a in slot_of]
                tt_cav = float(np.mean(tts)) if tts else float("nan")
            if np.isfinite(tt_hdv_raw):
                tt_hdv = tt_hdv_raw

        severity.end_episode(choice, offsets=coord._offsets)
        coord.observe(choice)

        switch = herd = float("nan")
        if prev_choice is not None:
            switch = float(np.mean(choice[av_slots] != prev_choice[av_slots]))
        c = np.bincount(choice[av_slots], minlength=loading.K).astype(float)
        p = c / max(c.sum(), 1)
        herd = float((len(p) * (p * p).sum() - 1) / max(len(p) - 1, 1))
        coord.log(day, phase, choice, tt_cav, tt_hdv,
                  float(np.mean(rewards)) if rewards else float("nan"),
                  switch, herd)
        prev_choice = choice
        pbar.update()

    for day in range(n_train):
        run_day(day, "train", learn=True)
        if not args.probe and day % plot_every == 0:               # noqa: F821
            env.plot_results()

    if args.probe:
        print(f"\n[PACT-NS] probe of {n_train} days complete. "
              f"records seen: {seen}. trace: {coord.debug_path}")
        coord.close()
        env.stop_simulation()
        return

    for a in env.machine_agents:
        a.model.policy_net.eval()
        a.model.deterministic = True
    pbar.set_description("Testing")
    for ep in range(test_eps):                                     # noqa: F821
        run_day(int(training_eps) + ep, "test", learn=False)       # noqa: F821

    pbar.close()
    env.plot_results()
    save_loss_records(records_folder,
                      [{"iteration": i, "agent_id": a.id, "loss": v}
                       for a in env.machine_agents
                       for i, v in enumerate(a.model.loss, start=1)],
                      columns=["iteration", "agent_id", "loss"])
    coord.close()
    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"),
                     os.path.join(records_folder, "episodes"),
                     remove_additional_files=True)
    run_metrics_analysis(args.id, results_folder="../results")


if __name__ == "__main__":
    main()

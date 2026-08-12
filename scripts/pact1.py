"""PACT-1 on URB — online identification of the congestion coupling, learned trust.

Run exactly like any other URB algorithm script:

    python scripts/pact1.py --id sai_pact1_0 --alg-conf config1 --task-conf config4 \
        --net saint_arnoult --env-seed 42 --torch-seed 0

Extra, PACT-1-only flags (all optional):

    --mode train|dry|probe   dry   builds the basis, runs every startup gate,
                                   prints the banner and EXITS -- no AV training,
                                   so a miswire costs seconds instead of a run.
                             probe runs --probe-eps days with UNIFORMLY RANDOM AV
                                   routes (maximum excitation), reports whether the
                                   reduction holds in this city, and exits. This is
                                   the cheap go/no-go: ~50 days instead of 4000.
    --probe-eps N            days for probe mode (default 60)
    --arm NAME               overrides pact1.trust_mode; 'blind' runs plain IPPO
                             through the identical wrapper so nothing else differs.

WHAT THIS ARM IS
--------------------------------------------------------------------------------
Base URB, unmodified. No injected non-stationarity, no changed reward, no changed
network, no changed demand. The claim being tested is the REDUCTION (guide III.2):
that this city's congestion coupling lives in a low-dimensional, known basis and
can therefore be identified online, decentralized, from each vehicle's own
realized-vs-free-flow residual — and that steering on that estimate beats learning
from scalar reward alone.

Read ``pact1/README.md`` for the column-by-column meaning of the diagnostics and
what each gate decides.
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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

from routerl import TrafficEnvironment
from routerl import Keychain as kc
from tqdm import tqdm

from iql import Network
from utils import clear_SUMO_files
from utils import print_agent_counts
from utils import run_metrics_analysis
from utils import save_loss_records
from utils import script_path_for_config

import pact1 as _pact1_pkg

# This script is `scripts/pact1.py` and the package is `pact1/`, so the two share a
# name. The sys.path insert above puts the repo root ahead of scripts/, which
# resolves it -- but if that ever stops holding, `import pact1.basis` fails with
# "'pact1' is not a package", which reads like a missing install rather than a
# shadowing problem. Say what actually happened instead.
if not hasattr(_pact1_pkg, "__path__"):
    raise ImportError(
        f"[PACT-1] `import pact1` resolved to {getattr(_pact1_pkg, '__file__', '?')} "
        "-- this script, not the package.\n"
        f"        Run it as `python scripts/pact1.py` from the repo, or put "
        f"{repo_root!r} ahead of the scripts/ directory on sys.path."
    )

from pact1.basis import RouteBasis, load_paths_csv, parse_sumo_net
from pact1.coordinator import Pact1Coordinator
from pact1.policy import Pact1PPO, check_shift_parity

ALGORITHM = "pact1"


# ==========================================================================
#  RouteRL record plumbing
# ==========================================================================
def load_config(folder, name, what):
    """Load a URB config, and on a miss say what IS available.

    URB's own README advertises ``--task-conf config4``, which this distribution
    does not ship -- so the first thing a new script hits is a bare
    FileNotFoundError from deep inside the parameter block. Listing the candidates
    turns a two-minute puzzle into a one-line fix, and matters more here because
    the run is remote.
    """
    path = os.path.join(folder, f"{name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    try:
        avail = sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".json"))
    except OSError:
        raise FileNotFoundError(
            f"[PACT-1] {what} directory does not exist: {os.path.abspath(folder)}"
        ) from None
    raise FileNotFoundError(
        f"[PACT-1] {what} {name!r} not found in {os.path.abspath(folder)}.\n"
        f"        Available: {', '.join(avail) if avail else '(none)'}\n"
        f"        NOTE: whatever you pick, every arm -- PACT-1 and the QMIX/IPPO/\n"
        f"        greedy/AON baselines -- must use the SAME task config, or the\n"
        f"        comparison measures the scenario rather than the method."
    )


def _aid(v):
    """Normalise an agent id to a string of an int, so ``5``, ``5.0`` and ``'5'``
    all key the same dict. Mixed id types across RouteRL's records and the agent
    objects would otherwise silently drop travellers from identification."""
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return str(v)


def drain_episode_records(env, cursor):
    """Return (new_records, new_cursor) from ``env.travel_times_list``.

    RouteRL appends one record per completed trip. The list is not guaranteed to
    be reset per episode, so a cursor is used (the same approach URB's own
    centralized wrapper takes); a list that shrank means it WAS reset, and the
    cursor is rewound rather than silently skipping a whole day.
    """
    lst = getattr(env, "travel_times_list", None)
    if lst is None:
        lst = getattr(env, "last_episode_travel_times", None)
        return (list(lst) if lst else []), 0
    if len(lst) < cursor:
        cursor = 0
    return list(lst[cursor:]), len(lst)


def split_records(records, machine_ids):
    """-> (av_records, peer_actions, tt_hdv, n_bad).

    av_records:   id -> (action, travel_time)   machine agents only
    peer_actions: id -> action                  EVERY traveller that completed a trip
    """
    av, peers, hdv, bad = {}, {}, [], 0
    for rec in records:
        if not isinstance(rec, dict):
            bad += 1
            continue
        rid = rec.get(kc.AGENT_ID)
        act = rec.get(kc.ACTION)
        tt = rec.get(kc.TRAVEL_TIME)
        if rid is None or act is None:
            bad += 1
            continue
        aid = _aid(rid)
        try:
            peers[aid] = int(act)
        except (TypeError, ValueError):
            bad += 1
            continue
        if aid in machine_ids:
            if tt is not None:
                av[aid] = (int(act), float(tt))
        elif tt is not None:
            hdv.append(float(tt))
    return av, peers, (float(np.mean(hdv)) if hdv else float("nan")), bad


# ==========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    # --- PACT-1 only ---
    parser.add_argument('--mode', type=str, default="train",
                        choices=["train", "dry", "probe"])
    parser.add_argument('--probe-eps', type=int, default=60)
    parser.add_argument('--arm', type=str, default=None,
                        choices=["pact1", "blind", "fixed"])
    args = parser.parse_args()

    exp_id, alg_config = args.id, args.alg_conf
    env_config, task_config = args.env_conf, args.task_conf
    network, env_seed, torch_seed = args.net, args.env_seed, args.torch_seed

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    print(f"PACT-1 mode: {args.mode}")

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

    # ---------------------------------------------------------------- config
    params = dict()
    alg_params = load_config(f"../config/algo_config/{ALGORITHM}", alg_config,
                             "algorithm config")
    env_params = load_config("../config/env_config", env_config,
                             "environment config")
    task_params = load_config("../config/task_config", task_config, "task config")
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    params.pop("desc", None)
    pact_cfg = dict(params.pop("pact1", {}))
    for key, value in params.items():
        globals()[key] = value

    if args.arm is not None:
        pact_cfg["trust_mode"] = {"pact1": "learned", "blind": "off",
                                  "fixed": "fixed"}[args.arm]

    # ---- the ONE gate that costs nothing: run it before touching SUMO --------
    ok, worst, floor_ok = check_shift_parity(kappa=float(pact_cfg.get("kappa", 1.0)))
    print(f"[PACT-1] GATE 2b torch/numpy shift parity: {'PASS' if ok else 'FAIL'} "
          f"(max abs diff {worst:.3e}, floor exact={floor_ok})", flush=True)
    if not ok and bool(pact_cfg.get("gate_abort", True)):
        raise AssertionError(
            "[PACT-1][GATE FAIL] the torch shift used in training disagrees with the "
            "numpy shift that selftest.py verifies. Fix before spending a run."
        )

    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes,                          # noqa: F821
              int(training_eps) + human_learning_episodes]         # noqa: F821
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

    num_machines = int(num_agents * ratio_machines)                # noqa: F821
    total_episodes = human_learning_episodes + training_eps + test_eps  # noqa: F821

    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config.update({
        "network": network, "env_seed": env_seed, "torch_seed": torch_seed,
        "env_config": env_config, "task_config": task_config,
        "alg_config": alg_config, "script": script_path_for_config(__file__),
        "algorithm": ALGORITHM, "num_agents": num_agents,
        "num_machines": num_machines, "pact1": pact_cfg,
        "pact1_mode": args.mode, "pact1_arm": args.arm or "pact1",
    })
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    # ---------------------------------------------------------------- env
    env = TrafficEnvironment(
        seed=env_seed,
        create_agents=False,
        create_paths=True,
        save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": num_machines,
            "human_parameters": {
                "model": human_model,                              # noqa: F821
                "alpha": human_alpha,                              # noqa: F821
                "beta": human_beta,                                # noqa: F821
                "beta_randomness": human_beta_randomness,          # noqa: F821
                "deterministic": human_deterministic,              # noqa: F821
            },
            "machine_parameters": {
                "behavior": av_behavior,                           # noqa: F821
                "observation_type": observations,                  # noqa: F821
            },
        },
        environment_parameters={"save_every": save_every},         # noqa: F821
        simulator_parameters={
            "network_name": network,
            "custom_network_folder": custom_network_folder,
            "sumo_type": "sumo",
            "simulation_timesteps": max_start_time,
        },
        plotter_parameters={
            "phases": phases, "phase_names": phase_names,
            "smooth_by": smooth_by,                                # noqa: F821
            "plot_choices": plot_choices,                          # noqa: F821
            "records_folder": records_folder, "plots_folder": plots_folder,
        },
        path_generation_parameters={
            "origins": origins, "destinations": destinations,
            "number_of_paths": number_of_paths,                    # noqa: F821
            "beta": path_gen_beta,                                 # noqa: F821
            "num_samples": num_samples,                            # noqa: F821
            "path_gen_workers": path_gen_workers,                  # noqa: F821
            "visualize_paths": False,
        },
    )

    env.start()
    env.reset()
    print_agent_counts(env)

    # ---------------------------------------------------------------- humans
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for _ in range(human_learning_episodes):                       # noqa: F821
        env.step()
        pbar.update()

    env.mutation(disable_human_learning=not should_humans_adapt,   # noqa: F821
                 mutation_start_percentile=-1)
    print_agent_counts(env)

    # ================================================================ PACT-1
    net_path = os.path.join(custom_network_folder, f"{network}.net.xml")
    paths_csv = os.path.join(records_folder, "paths.csv")
    print(f"\n[PACT-1] building the basis from\n"
          f"         {os.path.abspath(net_path)}\n"
          f"         {os.path.abspath(paths_csv)}", flush=True)

    net_edges = parse_sumo_net(net_path)
    routes_by_od, ffts_by_od = load_paths_csv(paths_csv, int(number_of_paths))  # noqa: F821
    basis = RouteBasis(
        net_edges, routes_by_od, ffts_by_od, n_paths=int(number_of_paths),  # noqa: F821
        speed_bounds=tuple(pact_cfg.get("speed_bounds", (8.5, 14.0))),
        min_class_share=float(pact_cfg.get("min_class_share", 0.03)),
        max_basis_mb=float(pact_cfg.get("max_basis_mb", 4000.0)),
    )

    machine_ids = {_aid(a.id) for a in env.machine_agents}
    agent_table = {}
    for a in env.all_agents:
        aid = _aid(a.id)
        agent_table[aid] = {
            "od": (int(a.origin), int(a.destination)),
            "start": float(a.start_time),
            "machine": aid in machine_ids,
        }
    av_ids = [_aid(a.id) for a in env.machine_agents]

    free_flow = env.get_free_flow_times()
    coord = Pact1Coordinator(basis, agent_table, av_ids, free_flow, pact_cfg,
                             run_dir=records_folder)
    coord.banner(extra=[f"exp_id     {exp_id}   net={network}   seed={env_seed}"])
    coord.check_fft_alignment(free_flow,
                              rtol=float(pact_cfg.get("fft_rtol", 1e-3)))
    coord.selfcheck()

    if args.mode == "dry":
        print("\n[PACT-1] --mode dry: every startup gate passed. Exiting before any "
              "AV episode was simulated.\n", flush=True)
        coord.close()
        env.stop_simulation()
        return

    # ---------------------------------------------------------------- models
    k_sizes = {int(a.action_space_size) for a in env.machine_agents}
    assert len(k_sizes) == 1 and k_sizes.pop() == int(number_of_paths), (  # noqa: F821
        "[PACT-1] machine agents disagree on the action-space size; the shared "
        "route basis assumes one K for the whole fleet."
    )
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    aug_size = obs_size + coord.obs_aug_dim
    print(f"[PACT-1] observation {obs_size} + {coord.obs_aug_dim} augmented "
          f"= {aug_size}", flush=True)

    for a in env.machine_agents:
        a.model = Pact1PPO(
            aug_size, int(a.action_space_size), Network, coord, _aid(a.id),
            device=device, batch_size=batch_size, lr=lr,                # noqa: F821
            num_epochs=num_epochs, num_hidden=num_hidden, widths=widths,  # noqa: F821
            clip_eps=clip_eps, normalize_advantage=normalize_advantage,   # noqa: F821
            entropy_coef=entropy_coef,                                    # noqa: F821
        )
    agent_lookup = {_aid(a.id): a for a in env.machine_agents}

    # ---------------------------------------------------------------- loop
    probe = args.mode == "probe"
    n_train = int(args.probe_eps) if probe else int(training_eps)   # noqa: F821
    rng_probe = np.random.RandomState(env_seed + 7919)
    cursor = 0
    os.makedirs(plots_folder, exist_ok=True)
    pbar.set_description("PACT-1 probe" if probe else "AV learning")

    for episode in range(n_train):
        coord.begin_episode(episode, phase="probe" if probe else "train")
        env.reset()
        rewards, chosen = [], {}

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            aid = _aid(agent_id)
            model = agent_lookup[aid].model

            if termination or truncation:
                rewards.append(float(reward))
                if not probe:
                    model.push(reward)
                    if episode % update_every == 0:                     # noqa: F821
                        model.learn()
                action = None
            elif probe:
                # Maximum excitation, and the policy is bypassed entirely: probe
                # mode measures whether the reduction holds in the BEST case, so a
                # failure here is conclusive and cannot be blamed on the learner.
                action = int(rng_probe.randint(model.action_space_size))
                chosen[aid] = action
            else:
                action = model.act(
                    coord.augment_obs(aid, observation), coord.route_context(aid)
                )
                chosen[aid] = int(action)

            env.step(action)

        records, cursor = drain_episode_records(env, cursor)
        av_rec, peer_act, tt_hdv, n_bad = split_records(records, machine_ids)

        if episode == 0:
            if not av_rec:
                raise AssertionError(
                    "[PACT-1][GATE FAIL] no AV travel-time records were recovered "
                    "after episode 0, so the estimator would never receive a single "
                    "row and the whole method would be silently inert.\n"
                    f"        drained {len(records)} records, {n_bad} unparseable. "
                    "Check that RouteRL exposes `travel_times_list` in this version."
                )
            n_human_rec = len(peer_act) - len(av_rec)
            print(f"[PACT-1] episode 0 records: {len(records)} drained, "
                  f"{len(av_rec)} AV, {n_human_rec} human, {n_bad} unparseable",
                  flush=True)
            if coord.peer_scope == "all" and n_human_rec < 1:
                # Not fatal -- the method still works off fleet load alone -- but it
                # silently changes what is being identified, so it must never pass
                # unnoticed (guide II.7: a return that looks fine is the failure
                # nobody investigates).
                print(
                    "[PACT-1][WARN] peer_scope='all' but NO human records were "
                    "recovered, so the waveform sees fleet traffic only. The human "
                    "background will be absorbed into the intercept instead of the "
                    "class channels. Either accept this and report peer_scope as "
                    "effectively 'fleet', or check RouteRL's record schema.",
                    flush=True,
                )

        # trust the actions we actually issued over the recorded ones
        for aid, k in chosen.items():
            peer_act[aid] = k
            if aid in av_rec:
                av_rec[aid] = (k, av_rec[aid][1])

        coord.end_episode(
            av_rec, peer_act,
            reward_mean=float(np.mean(rewards)) if rewards else float("nan"),
            tt_hdv=tt_hdv,
        )

        if (not probe) and episode % plot_every == 0:                  # noqa: F821
            env.plot_results()
        pbar.update()

    if probe:
        hist = coord._fit_hist[-min(len(coord._fit_hist), 30):]
        r2 = float(np.mean(hist)) if hist else float("nan")
        print("\n" + "=" * 78)
        print(f"[PACT-1 PROBE] {n_train} days, uniformly random AV routes "
              "(maximum excitation).")
        print(f"               mean fit_r2 over the last {len(hist)} days = {r2:.4f}")
        print("               fit_r2 is the fraction of realized excess delay that a")
        print("               LINEAR road-class model of peer load explains. This is")
        print("               the best case: if it is near zero here, the reduction")
        print("               does not hold in this city and a full run cannot help.")
        print(f"               trace: {coord.debug_path}")
        print("=" * 78 + "\n", flush=True)
        coord.close()
        env.stop_simulation()
        return

    # ---------------------------------------------------------------- test
    for a in env.machine_agents:
        a.model.policy_net.eval()
        a.model.deterministic = True

    pbar.set_description("Testing")
    for episode in range(test_eps):                                    # noqa: F821
        coord.begin_episode(int(training_eps) + episode, phase="test")  # noqa: F821
        env.reset()
        rewards, chosen = [], {}
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            aid = _aid(agent_id)
            model = agent_lookup[aid].model
            if termination or truncation:
                rewards.append(float(reward))
                action = None
            else:
                action = model.act(
                    coord.augment_obs(aid, observation), coord.route_context(aid)
                )
                chosen[aid] = int(action)
            env.step(action)

        records, cursor = drain_episode_records(env, cursor)
        av_rec, peer_act, tt_hdv, _ = split_records(records, machine_ids)
        for aid, k in chosen.items():
            peer_act[aid] = k
            if aid in av_rec:
                av_rec[aid] = (k, av_rec[aid][1])
        coord.end_episode(
            av_rec, peer_act,
            reward_mean=float(np.mean(rewards)) if rewards else float("nan"),
            tt_hdv=tt_hdv,
        )
        pbar.update()

    # ---------------------------------------------------------------- finish
    pbar.close()
    env.plot_results()
    loss_records = []
    for a in env.machine_agents:
        for iteration, loss_value in enumerate(a.model.loss, start=1):
            loss_records.append(
                {"iteration": iteration, "agent_id": a.id, "loss": loss_value}
            )
    save_loss_records(records_folder, loss_records,
                      columns=["iteration", "agent_id", "loss"])

    coord.close()
    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"),
                     os.path.join(records_folder, "episodes"),
                     remove_additional_files=True)
    run_metrics_analysis(exp_id, results_folder="../results")


if __name__ == "__main__":
    main()

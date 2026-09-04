"""Certify the URB NS and compute the coordination gap. NO SUMO. NO TRAINING.

    python scripts/ns_certify.py --net saint_arnoult \
        --routes ../results/<exp_id>/routes.csv \
        --task-conf config1 --sigmas 0.5 1.0 1.5

`NS_FORM_SPEC` E.1 and `PACT_PIPELINE_SPEC` §11 both put this first, and for the
same reason:

> Compute the ceiling decomposition FIRST. If the coordination gap is small, this
> environment is a poor showcase -- say so and pick another. This is minutes of
> work and it bounds everything downstream.

WHAT IT RUNS
--------------------------------------------------------------------------------
  structural   the declared operator: zero-diagonal, finite, asymmetric, weights
               spanning orders of magnitude (PIPELINE 2.1 refuses a flat proxy)
  Phi          std/mean > 0.05, or the coupling is uncancellable but
               unidentifiable (A.5 counter-check)
  B.1 / B.4    the four dial requirements plus the placebo regime, asserted over
               the WHOLE driver domain, not just at the peak
  PART C       the ceiling decomposition per sigma -- THE number
  C.4          the coordination gap against AV share, the N-scaling prediction

Everything here is a measurement of the ENVIRONMENT, made with no policy, so none
of it can be tuned toward a method. Commit the output before writing method code
(E.2 pitfall 10: retuning after seeing a method fail plants the problem).

The route table comes from any previous URB run's records folder -- RouteRL writes
`routes.csv` there during path generation -- so this needs no simulator.
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import json

import numpy as np
import pandas as pd

from urb_ns.ceiling import decompose, format_table, gap_vs_n
from urb_ns.driver import HCM_HEAVY_RAIN_LOSS, WeatherDriver
from urb_ns.loading import LoadingModel
from urb_ns.network import (
    RoadNetwork,
    build_co_presence,
    load_route_table,
    parse_sumo_net,
)


def find_routes_csv(explicit, network):
    """Locate a route table. RouteRL's filename varies by version, so look rather
    than assume -- it writes `routes.csv`, URB's clustered pipeline writes
    `paths.csv`, and both carry the same schema."""
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise FileNotFoundError(f"[URB-NS] --routes not found: {explicit}")
    seen = []
    for root in (os.path.join("..", "results"), os.path.join("..", "networks",
                                                             network), ".."):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in {".git", "SUMO_output", "episodes",
                                        "plots", "__pycache__", "networks"}]
            for fn in filenames:
                if fn in ("routes.csv", "paths.csv"):
                    p = os.path.join(dirpath, fn)
                    seen.append((os.path.getmtime(p), p))
    if seen:
        seen.sort(reverse=True)
        print(f"[URB-NS] route table found: {os.path.abspath(seen[0][1])}")
        return seen[0][1]
    raise FileNotFoundError(
        "[URB-NS] no routes.csv / paths.csv found under ../results or ../networks.\n"
        "        Run any URB experiment once to generate one, then pass\n"
        "        --routes <path>. This script needs no simulator, only that file."
    )


def build_loading(net, agents, ffts_by_od, av_share, seed, dur_factor=1.0):
    """Assemble a `LoadingModel` with a given AV share.

    URB mutates the FIRST `ratio_machines` fraction by `mutation_start_percentile
    = -1`, so a deterministic prefix by id reproduces the same fleet the
    experiment scripts create.
    """
    n = len(agents)
    ods = [(int(r.origin), int(r.destination)) for r in agents.itertuples()]
    start = np.array([float(r.start_time) for r in agents.itertuples()])

    K = net.n_paths
    gids = np.full((n, K), -1, dtype=np.int64)
    ff = np.full((n, K), np.nan)
    bad = []
    for i, od in enumerate(ods):
        if od not in net.od_index:
            bad.append(od)
            continue
        for k in range(K):
            gids[i, k] = net.gid_of[(od, k)]
            ff[i, k] = float(ffts_by_od[od][k])
    if bad:
        raise ValueError(
            f"[URB-NS] {len(bad)} travellers have an OD absent from the route "
            f"table, e.g. {bad[:3]}. The operator cannot describe them."
        )

    n_av = int(round(n * float(av_share)))
    is_machine = np.zeros(n, dtype=bool)
    is_machine[:n_av] = True

    dur = np.nanmean(ff, axis=1) * 60.0 * float(dur_factor)   # minutes -> seconds
    O = build_co_presence(start, dur)
    return LoadingModel(net, gids, ff, start, O, is_machine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--routes", default=None,
                    help="path to routes.csv / paths.csv from any previous run")
    ap.add_argument("--task-conf", default="config1")
    ap.add_argument("--env-conf", default="config1")
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--shares", type=float, nargs="+",
                    default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--period", type=int, default=100)
    ap.add_argument("--wet-frac", type=float, default=0.5)
    ap.add_argument("--loss", type=float, default=HCM_HEAVY_RAIN_LOSS)
    ap.add_argument("--mean-preserving", action="store_true")
    ap.add_argument("--dur-factor", type=float, default=1.0)
    ap.add_argument("--sat-flow", type=float, default=None,
                    help="effective approach capacity, veh/h/lane. Default 1900 "
                         "(HCM midblock saturation flow). Urban networks are "
                         "junction-constrained, so ~600 (green ratio ~0.3) is the "
                         "defensible effective figure -- DECLARE whichever you use "
                         "and report the resulting v/c distribution.")
    ap.add_argument("--n-draws", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../results/ns_certificate")
    args = ap.parse_args()

    print("=" * 78)
    print("URB-NS CERTIFICATE -- no simulator, no training, no method")
    print("=" * 78)

    env_cfg = json.load(open(f"../config/env_config/{args.env_conf}.json"))
    task_cfg = json.load(open(f"../config/task_config/{args.task_conf}.json"))
    n_paths = int(env_cfg["number_of_paths"])
    ratio = float(task_cfg["ratio_machines"])
    print(f"net={args.net}  task={args.task_conf}  number_of_paths={n_paths}  "
          f"ratio_machines={ratio}")

    net_path = os.path.join("..", "networks", args.net, f"{args.net}.net.xml")
    agents_path = os.path.join("..", "networks", args.net, "agents.csv")
    routes_csv = find_routes_csv(args.routes, args.net)

    print("\n--- 1. THE MEDIUM AND THE DECLARED OPERATOR -----------------------")
    net_edges = parse_sumo_net(net_path)
    routes_by_od, ffts_by_od = load_route_table(routes_csv, n_paths)
    from urb_ns.network import SAT_FLOW
    sat = float(args.sat_flow) if args.sat_flow else SAT_FLOW
    net = RoadNetwork(net_edges, routes_by_od, ffts_by_od, n_paths, sat_flow=sat)
    s = net.summary()
    struct_ok = True
    if s["op_spread"] < 0.2:
        print("[URB-NS] FAIL operator weights are nearly flat "
              f"(spread {s['op_spread']:.3f}). PIPELINE 2.1: a flat geometric "
              "proxy measured fit_gain = -0.0045. Check link capacities.")
        struct_ok = False
    if s["asymmetry"] < 1e-6:
        print("[URB-NS] FAIL operator is symmetric; a real transfer operator is "
              "not. Check the per-row normalisation.")
        struct_ok = False
    if struct_ok:
        print(f"[URB-NS] PASS operator is asymmetric with weights spanning "
              f"orders of magnitude")

    agents = pd.read_csv(agents_path)
    loading = build_loading(net, agents, ffts_by_od, ratio, args.seed,
                            args.dur_factor)
    n_av = int(loading.is_machine.sum())
    co = loading.O.sum(1).mean()
    print(f"[URB-NS] travellers {loading.N} ({n_av} AV / "
          f"{loading.N - n_av} human), mean co-present peers {co:.1f}")

    print("\n--- 2. THE EXERTION FUNCTIONAL Phi --------------------------------")
    spread = loading.phi_spread()
    ok_phi = spread > 0.05
    print(f"[URB-NS] {'PASS' if ok_phi else 'FAIL'} std(Phi)/mean(Phi) = "
          f"{spread:.4f}  (A.5 wants > 0.05; POWER ran 0.28)")
    if not ok_phi:
        print("[URB-NS]        a constant Phi is uncancellable but UNIDENTIFIABLE")

    print("\n--- 3. THE SEVERITY DIAL (B.1) AND THE PLACEBO (B.4) --------------")
    driver = WeatherDriver(args.period, args.wet_frac, args.loss,
                           args.mean_preserving)
    ok_dial, _ = driver.certify(sigmas=[0.0] + list(args.sigmas))

    print("\n--- 3b. G5 -- IS THE MEDIUM LOADED ENOUGH FOR THE DIAL TO BITE? ---")
    sg_probe = max(args.sigmas)
    op = loading.operating_point(n_draws=args.n_draws, seed=args.seed,
                                 sigma_probe=sg_probe, loss=args.loss)
    print(f"[URB-NS] v/c over AVs (binding link): mean {op['vc_mean']:.3f} "
          f"p50 {op['vc_p50']:.3f} p90 {op['vc_p90']:.3f} max {op['vc_max']:.3f}")
    print(f"[URB-NS] harm function: BPR alpha={loading.bpr_alpha:.3f} "
          f"beta={loading.bpr_beta:.2f} "
          + ("(LOW-UTILISATION regime -- linear, correct for URB; a quartic beta "
             "is numerically dead here)" if loading.bpr_beta <= 1.5 else
             "(CONGESTED regime -- needs v/c near 1; "
             f"only {op['frac_above_0p6'] * 100:.1f}% of AV-trips are above 0.6)"))
    print(f"[URB-NS] travel-time response at sigma={sg_probe}: "
          f"mean x{op['tt_response_mean']:.4f}  p90 x{op['tt_response_p90']:.4f} "
          f" max x{op['tt_response_max']:.4f}")
    ok_bite = op["tt_response_p90"] > 1.02
    print(f"[URB-NS] {'PASS' if ok_bite else 'FAIL'} G5: the dial must move travel "
          "time by >2% for the loaded tail, or the NS is inert whatever sigma is")
    if not ok_bite:
        print("[URB-NS]        BPR is QUARTIC: below v/c ~ 0.6 a capacity loss is\n"
              "[URB-NS]        invisible. This INSTANCE is demand-starved, not the\n"
              "[URB-NS]        form. Per C.3 -- 'use it to choose the environment,\n"
              "[URB-NS]        not to excuse the outcome' -- try a network with more\n"
              "[URB-NS]        demand per link (nemours 729 agents, provins 523,\n"
              "[URB-NS]        ingolstadt) before writing any method code.")

    print("\n--- 4. PART C -- THE COORDINATION GAP -----------------------------")
    rows = []
    for sg in args.sigmas:
        d = decompose(loading, driver, sg, n_draws=args.n_draws, seed=args.seed)
        rows.append(d)
    print(format_table(rows, "URB ceiling decomposition (geometric reference)"))

    live = [d for d in rows if not d.get("degenerate")]
    gap = float(np.mean([d["coordination_gap"] for d in live])) if live else float("nan")
    print(f"\n[URB-NS] mean coordination gap over the sweep: {gap * 100:.1f}%")
    print("[URB-NS] POWER measured 9.5% at sigma=1 with Delta_own = 76.9%.")
    if np.isfinite(gap):
        if gap > 0.25:
            print("[URB-NS] => URB is a STRONG coordination showcase. C.4's "
                  "no-local-authority prediction holds: an AV cannot decongest "
                  "its own road, so most recoverable damage is peer damage.")
        elif gap > 0.10:
            print("[URB-NS] => moderate gap, larger than POWER's. Usable.")
        else:
            print("[URB-NS] => SMALL gap. Per C.3, lead with another environment "
                  "rather than excusing the outcome.")

    print("\n--- 5. C.4 -- THE N-SCALING PREDICTION (no training) --------------")
    sg = args.sigmas[len(args.sigmas) // 2]
    print(f"[URB-NS] at sigma={sg}: the gap should GROW with the AV share, "
          "because a finer partition leaves each agent less local authority")
    ns_rows = gap_vs_n(
        lambda sh: build_loading(net, agents, ffts_by_od, sh, args.seed,
                                 args.dur_factor),
        driver, sg, args.shares, seed=args.seed,
        n_draws=max(4, args.n_draws // 2),
    )
    live_n = [d for d in ns_rows if not d.get("degenerate")]
    if len(live_n) >= 2:
        x = np.array([d["av_share"] for d in live_n])
        y = np.array([d["coordination_gap"] for d in live_n])
        slope = float(np.polyfit(x, y, 1)[0])
        print(f"[URB-NS] gap vs AV share: slope {slope:+.4f} per unit share "
              f"({y[0] * 100:.1f}% -> {y[-1] * 100:.1f}%)  "
              f"=> C.4 {'CONFIRMED' if slope > 0 else 'REFUTED'}")

    os.makedirs(args.out, exist_ok=True)
    payload = {
        "net": args.net, "task_conf": args.task_conf,
        "network": s, "phi_spread": spread, "phi_ok": bool(ok_phi),
        "operator_ok": bool(struct_ok), "dial_ok": bool(ok_dial),
        "driver": {"period": args.period, "wet_frac": args.wet_frac,
                   "loss": args.loss, "mean_preserving": args.mean_preserving,
                   "anchor": "HCM heavy-rain capacity adjustment factor"},
        "ceiling": rows, "gap_vs_share": ns_rows,
    }
    path = os.path.join(args.out, f"certificate_{args.net}_{args.task_conf}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\n[URB-NS] certificate written to {os.path.abspath(path)}")
    print("[URB-NS] COMMIT THIS before writing method code (E.2 pitfall 10).")

    all_ok = struct_ok and ok_phi and ok_dial
    print("=" * 78)
    print(f"STRUCTURAL CERTIFICATE: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Read a running PACT-1 trace and say, in plain terms, whether it is working.

    python pact1/watch.py sai_pact1_0
    python pact1/watch.py sai_pact1_0 --baseline 3.21        # your QMIX t_CAV
    python pact1/watch.py pact1_debug_sai_pact1_0.csv --last 200

Safe to run at any time against a live file -- it only reads.

TWO QUESTIONS, KEPT APART
--------------------------------------------------------------------------------
1. IS THE METHOD WORKING?   fit_r2, liveness, conditioning, trust.
   A failure here is mechanical and usually fatal: stop and fix.

2. IS IT WINNING?           t_CAV against the human drivers it replaced.
   URB's own winrate criterion is "CAVs were on average faster than HDVs", i.e.
   cav_adv = t_HDV / t_CAV > 1. That is config-independent, so it is the honest
   read even when your task config differs from the published table.

The distinction matters: the method can be working perfectly (fit_r2 high, beta
tracking) and still not win, which is a finding about how much headroom routing has
in this city -- not a bug.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# The published St. Arnoult numbers, for orientation ONLY. They were produced under
# the paper's task config, which is NOT the one shipped here -- so compare against
# your OWN baseline run with a matched --task-conf, never against these.
REFERENCE = {
    "saint_arnoult": {"t_pre": 3.15, "QMIX": 3.21, "IPPO": 3.33, "IQL": 3.53,
                      "MAPPO": 3.51, "greedy": 3.01, "AON": 3.01, "random": 3.58},
    "provins": {"t_pre": 2.80, "QMIX": 3.14, "IPPO": 2.98, "greedy": 2.74,
                "AON": 2.76},
    "ingolstadt_custom": {"t_pre": 4.21, "QMIX": 4.87, "IPPO": 4.71,
                          "greedy": 4.24, "AON": 4.37},
}


def _resolve(arg):
    if os.path.exists(arg):
        return arg
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for cand in (
        os.path.join(root, f"pact1_debug_{arg}.csv"),
        os.path.join(root, arg),
        os.path.join(root, "results", arg, f"pact1_debug_{arg}.csv"),
        os.path.join(root, "results", arg, "pact1_debug.csv"),
    ):
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        f"[watch] could not find a trace for {arg!r}. Looked for "
        f"pact1_debug_{arg}.csv in {root} and under results/{arg}/."
    )


def _m(df, col, n=None):
    if col not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if n:
        s = s.tail(n)
    return float(s.mean()) if len(s) else float("nan")


def _f(v, n=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"{v:.{n}f}" if np.isfinite(v) else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="exp_id or path to pact1_debug_*.csv")
    ap.add_argument("--last", type=int, default=200,
                    help="window of recent episodes to average (default 200)")
    ap.add_argument("--baseline", type=float, default=None,
                    help="t_CAV of YOUR baseline run (e.g. QMIX) for comparison")
    ap.add_argument("--net", type=str, default=None,
                    help="network name, to print published reference numbers")
    args = ap.parse_args()

    path = _resolve(args.trace)
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"[watch] {path} has no rows yet.")

    n = args.last
    train = df[df["phase"] == "train"] if "phase" in df.columns else df
    test = df[df["phase"] == "test"] if "phase" in df.columns else df.iloc[0:0]

    print("=" * 78)
    print(f"PACT-1 trace  {path}")
    print(f"{len(df)} episodes logged  "
          f"({len(train)} train, {len(test)} test)   window = last {n}")
    print("=" * 78)

    # ---------------------------------------------------------------- 1. health
    fit = _m(train, "fit_r2", n)
    pred = _m(train, "pred_r2", n)
    cond = _m(train, "cond_psi", n)
    xabs = np.nanmean([_m(train, c, n) for c in
                       ("x_local", "x_collector", "x_arterial")])
    conf = _m(train, "conf", n)
    tp, ta = _m(train, "trust_pol", n), _m(train, "trust_app", n)
    tp0 = pd.to_numeric(train.get("trust_pol", pd.Series(dtype=float)),
                        errors="coerce").dropna()
    tp_drift = (float(tp0.iloc[-1] - tp0.iloc[0]) if len(tp0) > 1 else float("nan"))

    print("\n1. IS THE METHOD WORKING?")
    print(f"   fit_r2   {_f(fit)}   does the linear road-class model explain "
          f"realized delay?")
    print(f"   pred_r2  {_f(pred)}   is peer load predictable a day ahead?")
    print(f"   cond     {_f(cond, 1)}   can theta be DECOMPOSED, not just predicted?")
    print(f"   x |mean| {_f(xabs)}   waveform liveness (0 => inert)")
    print(f"   conf     {_f(conf)}   estimator self-belief")
    print(f"   trust    policy {_f(tp)} -> applied {_f(ta)}   "
          f"(policy trust moved {_f(tp_drift)} over the run)")

    notes = []
    if not np.isfinite(fit) or fit < 0.05:
        notes.append("STOP: fit_r2 below 0.05 -- the reduction does not hold in "
                     "this city. Nothing downstream can work.")
    elif fit < 0.2:
        notes.append("WEAK: fit_r2 is low. Try peer_scope='all' or a larger "
                     "dur_factor before drawing conclusions.")
    else:
        notes.append("OK: the reduction holds -- delay really is close to linear "
                     "in road-class peer load.")
    if np.isfinite(fit) and np.isfinite(pred) and fit - pred > 0.3:
        notes.append("fit >> pred: the physics is right but the fleet flaps faster "
                     "than rho tracks. Sweep rho. This is a finding, not a bug.")
    if np.isfinite(cond) and cond > 1e4:
        notes.append("cond is high: report PREDICTION, do not claim to have "
                     "decomposed the per-class sensitivities (guide III.6).")
    if np.isfinite(tp_drift) and abs(tp_drift) < 0.02:
        notes.append("trust barely moved: it is well-INITIALISED, not learned. "
                     "Say so explicitly (guide III.14.2).")
    for s in notes:
        print(f"   -> {s}")

    # ---------------------------------------------------------------- 2. winning
    print("\n2. IS IT WINNING?")
    src, tag = (test, "TEST") if len(test) >= 5 else (train, f"train (last {n})")
    w = None if len(test) >= 5 else n
    cav, hdv = _m(src, "tt_cav", w), _m(src, "tt_hdv", w)
    adv = _m(src, "cav_adv", w)
    print(f"   phase used: {tag}")
    print(f"   t_CAV    {_f(cav, 3)}")
    print(f"   t_HDV    {_f(hdv, 3)}")
    print(f"   cav_adv  {_f(adv, 4)}   (= t_HDV / t_CAV; > 1 means CAVs are faster)")

    if np.isfinite(adv):
        if adv > 1.0:
            print("   -> WON by URB's own winrate criterion: the fleet is faster "
                  "than the drivers it replaced.")
        else:
            print("   -> NOT won: the fleet is slower than the humans it replaced. "
                  "Every MARL method in the URB paper sits here on most networks.")

    if args.baseline is not None and np.isfinite(cav):
        d = args.baseline - cav
        print(f"\n   vs YOUR baseline t_CAV = {args.baseline:.3f}: "
              f"PACT-1 is {abs(d):.3f} {'FASTER' if d > 0 else 'SLOWER'} "
              f"({100.0 * d / args.baseline:+.2f}%)")

    if args.net and args.net in REFERENCE:
        print(f"\n   published {args.net} figures (paper's task config, NOT the one "
              "shipped here -- orientation only):")
        print("     " + "  ".join(f"{k}={v}" for k, v in REFERENCE[args.net].items()))
        print("     Compare against your OWN matched-config baseline instead.")

    # ---------------------------------------------------------------- 3. commons
    sw, herd = _m(train, "route_switch_frac", n), _m(train, "herd_index", n)
    print(f"\n3. THE COMMONS (T4 / guide III.8)")
    print(f"   route_switch_frac {_f(sw)}   herd_index {_f(herd)}")
    print("   Steering onto the route the model calls fast is what LOADS it. A herd "
          "index\n   rising alongside trust is the externality becoming visible -- "
          "log it, do not fight it.")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())

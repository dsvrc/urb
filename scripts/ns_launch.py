"""Run ANY URB script under the NS severity dial, without editing that script.

    python scripts/ns_launch.py --sigma 3 --net saint_arnoult -- \
        qmix_torchrl.py --id sai_s3_qmix --alg-conf config1 \
        --task-conf config1 --net saint_arnoult --env-seed 42 --torch-seed 0

Everything after `--` is the target script and its own arguments, verbatim.

HOW IT WORKS
--------------------------------------------------------------------------------
`routerl.TrafficEnvironment` is replaced with a severity-wrapped subclass BEFORE
the target script is executed. The target does `from routerl import
TrafficEnvironment` at its own import time, which is after the patch, so it
constructs the wrapped class without knowing anything has happened.

That is what makes the comparison legitimate (`NS_FORM_SPEC` B.5): MAPPO, QMIX,
VDN, IPPO, IQL and PACT all face the identical dial, identical weather and
identical seed, because none of them can tell the difference.

WHY A LAUNCHER RATHER THAN EDITING THE SCRIPTS
--------------------------------------------------------------------------------
Editing eight baseline scripts to accept a severity flag would make each of them a
modified baseline, and a reviewer would rightly ask what else changed. This way
URB's scripts are bit-for-bit the ones its authors shipped, and the only new
object is the environment they run in -- which is exactly where the NS belongs.

`--sigma 0` gives the stock task byte for byte, so the SAME command produces the
no-severity row. Use it for both tables and the two are guaranteed comparable.
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(_scripts_dir, ".."))

# *** ORDER MATTERS AND IS NOT COSMETIC. ***
# `scripts/pact1.py` and the `pact1/` package share a name. Targets need
# scripts/ on the path (`from iql import Network`), but if scripts/ comes FIRST
# then `import pact1` resolves to the script, not the package, and pact1.py's own
# guard aborts the run. The target cannot repair this itself: it only inserts
# repo_root `if repo_root not in sys.path`, and by then it already is -- just in
# the wrong position. So repo_root is forced to index 0 here, every time.
for _p in (repo_root, _scripts_dir):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, repo_root)          # repo_root FIRST -- packages beat scripts

import argparse
import json
import runpy

from urb_ns.driver import HCM_HEAVY_RAIN_LOSS
from urb_ns.severity_traffic_env import make_severity_env


def main():
    ap = argparse.ArgumentParser(
        description="Run a URB script under the NS severity dial.")
    ap.add_argument("--sigma", type=float, required=True,
                    help="0 = stock task byte for byte; 1 = the HCM heavy-rain "
                         "factor; >1 a labelled beyond-physical stress test")
    ap.add_argument("--net", required=True)
    ap.add_argument("--env-conf", default="config1")
    ap.add_argument("--routes", default=None,
                    help="route table; found automatically if omitted")
    ap.add_argument("--sat-flow", type=float, default=600.0)
    ap.add_argument("--period", type=int, default=100)
    ap.add_argument("--wet-frac", type=float, default=0.5)
    ap.add_argument("--loss", type=float, default=HCM_HEAVY_RAIN_LOSS)
    ap.add_argument("--mean-preserving", action="store_true")
    ap.add_argument("--av-behavior", default="selfish")
    ap.add_argument("target", nargs=argparse.REMAINDER,
                    help="-- <script.py> <its args...>")
    args = ap.parse_args()

    target = [t for t in args.target if t != "--"]
    if not target:
        ap.error("nothing to run: put the target script after `--`")
    script = target[0]
    if not os.path.exists(script):
        ap.error(f"target script not found: {script}")

    env_cfg = json.load(open(f"../config/env_config/{args.env_conf}.json"))
    from ns_certify import find_routes_csv
    routes = find_routes_csv(args.routes, args.net)

    ns_cfg = {
        "net_xml": os.path.join("..", "networks", args.net, f"{args.net}.net.xml"),
        "number_of_paths": int(env_cfg["number_of_paths"]),
        "sat_flow": args.sat_flow, "period": args.period,
        "wet_frac": args.wet_frac, "loss": args.loss,
        "mean_preserving": args.mean_preserving,
        "av_behavior": args.av_behavior, "routes_csv": routes,
    }

    # --- the patch, before the target imports anything ----------------------
    import routerl
    base = routerl.TrafficEnvironment
    wrapped = make_severity_env(base, ns_cfg, args.sigma, routes_csv=routes)
    routerl.TrafficEnvironment = wrapped
    for mod in ("routerl.environment", "routerl.environment.environment"):
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, "TrafficEnvironment"):
            m.TrafficEnvironment = wrapped

    print(f"[URB-NS] launching {script} under sigma={args.sigma}")
    print(f"[URB-NS] route table: {os.path.abspath(routes)}")

    sys.argv = list(target)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        routerl.TrafficEnvironment = base


if __name__ == "__main__":
    main()

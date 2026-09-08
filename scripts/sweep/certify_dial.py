"""Print and certify the severity dial for a given sigma. No SUMO, no training.

    python scripts/sweep/certify_dial.py --sigma 3

Called once by the sweep wrappers so the dial's numbers are recorded in the
sweep log itself, next to the runs they apply to. Exits non-zero if the driver
fails its own B.1 / B.4 requirements, which is a reason not to start a sweep.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from urb_ns.driver import HCM_HEAVY_RAIN_LOSS, WeatherDriver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--period", type=int, default=100)
    ap.add_argument("--wet-frac", type=float, default=0.5)
    ap.add_argument("--loss", type=float, default=HCM_HEAVY_RAIN_LOSS)
    ap.add_argument("--mean-preserving", action="store_true")
    args = ap.parse_args()

    drv = WeatherDriver(args.period, args.wet_frac, args.loss,
                        args.mean_preserving)
    ok, _ = drv.certify(sigmas=(0.0, 1.0, args.sigma))

    d = np.arange(drv.period)
    g = drv.g(d, args.sigma)
    print(f"[sweep] sigma={args.sigma}: peak capacity kept {g.min():.4f}, "
          f"{drv.capacity_loss_over_cycle(args.sigma) * 100:.2f}% of capacity "
          f"removed over the cycle, intra-cycle swing {drv.swing(args.sigma):.3f}x")
    print(f"[sweep] driver certified: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

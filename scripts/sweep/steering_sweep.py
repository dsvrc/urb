"""saint_arnoult: the trust phase diagram on the DECLARED model.  No SUMO, no torch.

Tier 0 item 1 of VALIDATION_URB.md.  Iterates the steered fleet to a fixed point over a
grid of trust values and records social cost, concentration and flapping, so Thm 5.24's
K = 2 phase diagram can be checked at K = 4 on the real network before any training.

    python verify_urb_steering.py                      # damped (rho = 0.8) and synchronous
    python verify_urb_steering.py --damp 0.0           # synchronous only: flapping is visible
    python verify_urb_steering.py --sigma 0            # no dial

What is assumed, so the paper can state it:
  * host logits are uninformative (the steering acts alone), which is Prop 5.22's setting;
  * predictions are EXACT at the current profile (beta_hat = beta*), so this isolates the
    steering/commons effect from estimation error;
  * humans are frozen at one reference draw, as in config1.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

UR = os.environ.get("URB_ROOT",
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# ORDER MATTERS: scripts/pact1.py shadows the pact1/ package, and that script imports
# routerl at module level.  repo root must come first so `pact1` is the package.
sys.path.insert(0, os.path.join(UR, "scripts"))
sys.path.insert(0, UR)
os.chdir(os.path.join(UR, "scripts"))

from urb_ns.network import RoadNetwork, load_route_table, parse_sumo_net
from urb_ns.driver import WeatherDriver
from pact1.core import steer_logits
from ns_certify import build_loading


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def zscore_rows(c):
    """Population z-score per row, zero where the spread is degenerate (as steer_logits does)."""
    m = c.mean(axis=1, keepdims=True)
    s = c.std(axis=1, keepdims=True)
    z = np.zeros_like(c)
    ok = (s > 1e-12).ravel()
    z[ok] = (c[ok] - m[ok]) / s[ok]
    return z


class Fleet:
    """Expected costs of every candidate route under a mixed profile."""

    def __init__(self, lm, net, g):
        self.lm, self.net, self.g = lm, net, np.asarray(g, dtype=np.float64)
        self.N, self.K = lm.gids.shape
        # (K, N, E) binary incidence of each agent's candidate routes
        self.inc = np.stack([net.A[lm.gids[:, k]] for k in range(self.K)])
        self.cap_g = np.maximum(net.cap * self.g, 1e-9)               # (E,)
        self.T = np.maximum(lm.trip_h, lm.min_trip_h)                 # (N, K) hours
        self.alpha, self.beta = lm.bpr_alpha, lm.bpr_beta

    def costs(self, P):
        """P: (N, K) mixed profile -> (N, K) expected travel time of each candidate."""
        contrib = np.einsum("nk,kne->ne", P, self.inc)                # (N, E) expected load
        veh = self.lm.O @ contrib - contrib                           # strict j != i
        out = np.empty((self.N, self.K))
        for k in range(self.K):
            flow = veh / self.T[:, k][:, None]
            ratio = flow / self.cap_g[None, :]
            mask = self.inc[k] > 0
            r = np.where(mask, ratio, -np.inf)
            u = r.max(axis=1)
            u = np.where(np.isfinite(u), np.maximum(u, 0.0), 0.0)
            out[:, k] = self.lm.ffts[:, k] * (1.0 + self.alpha * u ** self.beta)
        return out


def run(fleet, P0, av, g_trust, kappa, damp, iters, tol):
    """Iterate the steered fixed point.  Returns metrics at the end of the run."""
    P = P0.copy()
    prev, moves = None, []
    for t in range(iters):
        C = fleet.costs(P)
        z = zscore_rows(C)
        target = softmax(-g_trust * kappa * z)                        # uninformative host logits
        newP = P.copy()
        newP[av] = damp * P[av] + (1.0 - damp) * target[av]
        mv = float(np.abs(newP[av] - P[av]).sum(axis=1).mean())
        moves.append(mv)
        prev, P = P, newP
        if mv < tol and t > 5:
            break
    C = fleet.costs(P)
    social = float((P[av] * C[av]).sum(axis=1).mean())
    conc = (P[av] ** 2).sum(axis=1)
    K = P.shape[1]
    herd = float(((conc - 1.0 / K) / (1.0 - 1.0 / K)).mean())
    # oscillation, measured AFTER the transient: the largest late move, not its mean
    tail = moves[-10:] if len(moves) >= 10 else moves[len(moves) // 2:]
    return dict(social=social, herd=herd, osc=float(np.max(tail)),
                last=float(moves[-1]), converged=bool(moves[-1] < tol),
                iters=len(moves))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--damp", type=float, default=None,
                    help="EMA damping; default runs both 0.8 and 0.0")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--share", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--routes", default="../results/sample_results/routes.csv",
                    help="route table, resolved relative to scripts/")
    ap.add_argument("--sign-limit", type=float, default=5.0,
                    help="large trust, reported as the sign-steering / Wardrop end of the path")
    ap.add_argument("--alpha-scale", type=float, default=1.0,
                    help="multiplies the calibrated BPR alpha (2.28). Raising it raises the "
                         "congestible ratio rho, the light-traffic knob of App. C, which "
                         "controls whether an interior optimum can exist at all.")
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6,
                             0.7, 0.8, 0.9, 1.0])
    args = ap.parse_args()

    net_edges = parse_sumo_net(os.path.join("..", "networks", "saint_arnoult",
                                            "saint_arnoult.net.xml"))
    routes, ffts_by_od = load_route_table(args.routes, 4)
    net = RoadNetwork(net_edges, routes, ffts_by_od, 4, sat_flow=600.0, verbose=False)
    agents = pd.read_csv(os.path.join("..", "networks", "saint_arnoult", "agents.csv"))
    lm = build_loading(net, agents, ffts_by_od, args.share, args.seed)
    av = lm.is_machine
    N, K = lm.gids.shape

    drv = WeatherDriver(100, 0.5, 0.14, False)
    g = drv.g(25, args.sigma, sens=net.rain_sens)                     # driver peak, A = 1
    g = np.broadcast_to(np.asarray(g, dtype=np.float64), (net.E,)).copy()
    lm.bpr_alpha = lm.bpr_alpha * args.alpha_scale
    fleet = Fleet(lm, net, g)

    # parity: the vectorised shift must equal the repo's own steer_logits
    rng = np.random.RandomState(0)
    worst = 0.0
    for _ in range(200):
        tt = rng.randn(K) * 8 + 120
        gg = float(rng.rand())
        ref = steer_logits(np.zeros(K), tt, gg, args.kappa)
        got = -gg * args.kappa * zscore_rows(tt[None, :])[0]
        worst = max(worst, float(np.max(np.abs(ref - got))))
    assert worst < 1e-12, f"shift parity broken: {worst:.3e}"

    # humans frozen at the reference draw; AVs start uniform
    P0 = np.zeros((N, K))
    hum_choice = lm.uniform_reference_choice(np.random.RandomState(args.seed))
    P0[np.arange(N), hum_choice] = 1.0
    P0[av] = 1.0 / K

    print(f"network {net.summary()['n_links']} links, {N} travellers, {int(av.sum())} AVs, K={K}")
    print(f"sigma={args.sigma} at the driver peak: g min {g.min():.4f} mean {g.mean():.4f}; "
          f"kappa={args.kappa}; alpha={lm.bpr_alpha:.3f} (x{args.alpha_scale}); "
          f"shift parity max |diff| {worst:.2e}")

    damps = [0.8, 0.0] if args.damp is None else [args.damp]
    for damp in damps:
        print(f"\n--- damping rho = {damp}  ({'EMA-smoothed' if damp else 'synchronous'}) ---")
        print(f"{'g':>6} {'social cost':>12} {'vs g=0':>9} {'herd':>7} "
              f"{'late osc':>10} {'iters':>6}  state")
        base = None
        rows = []
        for gt in list(args.grid) + [args.sign_limit]:
            r = run(fleet, P0, av, gt, args.kappa, damp, args.iters, args.tol)
            if base is None:
                base = r["social"]
            state = "fixed point" if r["converged"] else "NOT settled"
            tag = "  <- sign-steering limit" if gt == args.sign_limit else ""
            print(f"{gt:6.2f} {r['social']:12.5f} {100*(r['social']/base-1):+8.3f}% "
                  f"{r['herd']:7.4f} {r['osc']:10.2e} {r['iters']:6d}  {state}{tag}")
            rows.append((gt, r))
        grid_rows = rows[:len(args.grid)]
        best = min(grid_rows, key=lambda x: x[1]["social"])
        interior = best[0] not in (args.grid[0], args.grid[-1])
        print(f"  minimum over the grid at g* = {best[0]:.2f} "
              f"({100*(best[1]['social']/base-1):+.3f}% vs uniform); "
              f"interior optimum = {interior}"
              + ("   [Thm 5.24 threshold visible]" if interior
                 else "   [monotone: no informational-Braess threshold here]"))


if __name__ == "__main__":
    main()

"""The medium and the DECLARED coupling operator.

`NS_FORM_SPEC` A.2.2: `W` must be declared, not fitted -- a property of the
environment a domain expert would write down. In traffic that is **link-path
incidence divided by link capacity**, which is the traffic-assignment counterpart
of PTDF and is computed here from `net.xml` and the generated route set.

    Operator[a, j] = 1[link a is on j's route] / capacity_a
                     "how much one vehicle of j loads link a"

    W[i, j]        = mean over a in E(i) of Operator[a, j]        for j != i
    W[i, i]        = 0                                            ASSERTED

`PACT_PIPELINE_SPEC` §2.1 is emphatic about the one thing not to do here:

> [PAID] A geometric proxy ("peer owns an element I watch" vs "everything else",
> equal weight inside each bucket) measured fit_gain = -0.0045 -- the peer
> channels made prediction WORSE than an intercept-only null. The true operator's
> weights span orders of magnitude and are strongly asymmetric. Refuse to fall
> back to a proxy; make a missing operator fatal.

So the `1/capacity_a` weights are carried through explicitly, and the asymmetry is
real: `W[i,j]` averages over *i*'s links, so route length enters the row but not
the column. Both properties are measured and reported by `summary()`.

CAPACITY
--------------------------------------------------------------------------------
    capacity_a [veh/h] = n_lanes(a) * SAT_FLOW

`SAT_FLOW = 1900 veh/h/lane` is the Highway Capacity Manual base saturation flow
rate for urban arterials. A DECLARED CONSTANT (guide II.1), never tuned.
"""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np

# HCM base saturation flow rate, veh/h per lane, urban arterial. DECLARED.
SAT_FLOW = 1900.0

_SPLIT = re.compile(r"[,\s]+")
_ROUTE_COLS = ("origins", "destinations", "path", "free_flow_time")


# ==========================================================================
def parse_sumo_net(net_path):
    """`<net>.net.xml` -> {edge_id: (length_m, n_lanes, speed_mps)}.

    Internal junction edges (ids starting with ':') carry no routable length in
    RouteRL's route strings and are skipped. iterparse keeps memory flat on the
    20k-edge networks.
    """
    if not os.path.exists(net_path):
        raise FileNotFoundError(f"[URB-NS] SUMO network not found: {net_path}")
    edges = {}
    for _, elem in ET.iterparse(net_path, events=("end",)):
        if elem.tag != "edge":
            continue
        eid = elem.get("id")
        if eid is None or eid.startswith(":"):
            elem.clear()
            continue
        lengths, speeds = [], []
        for lane in elem.findall("lane"):
            try:
                lengths.append(float(lane.get("length", "0") or 0.0))
                speeds.append(float(lane.get("speed", "0") or 0.0))
            except (TypeError, ValueError):
                continue
        if lengths:
            edges[eid] = (float(np.mean(lengths)), max(1, len(lengths)),
                          float(max(speeds)) if speeds else 0.0)
        elem.clear()
    if not edges:
        raise ValueError(f"[URB-NS] no routable edges parsed from {net_path}")
    return edges


def load_route_table(routes_csv, n_paths):
    """RouteRL's `routes.csv` / `paths.csv` -> per-OD edge lists and free-flow times.

    Action-index alignment is the one thing that can be silently wrong (a permuted
    order would point every operator row at the wrong road while looking healthy),
    so `RoadNetwork.check_fft` verifies the parsed free-flow times against the
    environment's own `get_free_flow_times()` and aborts on a mismatch.
    """
    import pandas as pd

    if not os.path.exists(routes_csv):
        raise FileNotFoundError(f"[URB-NS] route table not found: {routes_csv}")
    df = pd.read_csv(routes_csv)
    for c in _ROUTE_COLS:
        if c not in df.columns:
            raise ValueError(
                f"[URB-NS] route table missing column {c!r}; found {list(df.columns)}"
            )
    routes, ffts = {}, {}
    for (o, d), grp in df.groupby(["origins", "destinations"], sort=True):
        if "cluster" in grp.columns and grp["cluster"].notna().all():
            grp = grp.sort_values("cluster")
        key = (int(o), int(d))
        el, ff = [], []
        for row in grp.itertuples(index=False):
            el.append([e for e in _SPLIT.split(str(row.path).strip()) if e])
            ff.append(float(row.free_flow_time))
        if len(el) != n_paths:
            raise ValueError(
                f"[URB-NS] OD {key} has {len(el)} routes but number_of_paths="
                f"{n_paths}. Refusing to guess."
            )
        routes[key], ffts[key] = el, ff
    if not routes:
        raise ValueError(f"[URB-NS] route table parsed to zero OD pairs: {routes_csv}")
    return routes, ffts


# ==========================================================================
class RoadNetwork(object):
    """Link capacities, route incidence, and the declared coupling operator."""

    def __init__(self, net_edges, routes_by_od, ffts_by_od, n_paths,
                 sat_flow=SAT_FLOW, verbose=True):
        self.n_paths = int(n_paths)
        self.routes_by_od = routes_by_od
        self.ffts_by_od = ffts_by_od
        self.sat_flow = float(sat_flow)

        # ---- global route index: (o, d, k) -> gid -----------------------------
        self.od_list = sorted(routes_by_od.keys())
        self.od_index = {od: j for j, od in enumerate(self.od_list)}
        self.n_od = len(self.od_list)
        self.R = self.n_od * self.n_paths
        self.gid_of = {}
        for od, j in self.od_index.items():
            for k in range(self.n_paths):
                self.gid_of[(od, k)] = j * self.n_paths + k

        # ---- links actually used by some route --------------------------------
        used, missing = set(), set()
        for od in self.od_list:
            for path in routes_by_od[od]:
                for e in path:
                    (used if e in net_edges else missing).add(e)
        self.edge_ids = sorted(used)
        self.edge_index = {e: i for i, e in enumerate(self.edge_ids)}
        self.E = len(self.edge_ids)
        self.missing_edges = missing
        self.missing_frac = len(missing) / max(1, len(missing) + self.E)
        if self.E == 0:
            raise ValueError("[URB-NS] no route edge matched the SUMO network.")

        self.edge_length = np.array([net_edges[e][0] for e in self.edge_ids])
        self.edge_lanes = np.array([net_edges[e][1] for e in self.edge_ids],
                                   dtype=np.float64)
        self.edge_speed = np.array([net_edges[e][2] for e in self.edge_ids])

        # ---- CAPACITY: the medium's supply, veh/h ------------------------------
        self.cap = self.edge_lanes * self.sat_flow
        self.inv_cap = 1.0 / np.maximum(self.cap, 1e-9)

        # ---- per-link RAIN SENSITIVITY, mean 1.0 -------------------------------
        # HCM capacity adjustment factors are facility-specific: wet weather costs
        # proportionally MORE capacity on higher-speed roads, because the speed
        # reduction is larger where free-flow speed is higher. A uniform derating
        # would leave the bottleneck link unchanged at every severity, which makes
        # the ceiling decomposition sigma-invariant -- POWER's moves 6.3 -> 9.5 ->
        # 12.7% precisely because its binding element shifts. Declared from
        # geometry, never fitted.
        sp = np.maximum(self.edge_speed, 1e-6)
        self.rain_sens = sp / max(1e-9, float(np.average(sp, weights=self.cap)))
        self.rain_sens = np.clip(self.rain_sens, 0.4, 2.0)

        # ---- route -> link incidence (R x E), binary --------------------------
        A = np.zeros((self.R, self.E), dtype=np.float64)
        rlen = np.zeros(self.R)
        nlink = np.zeros(self.R)
        for od in self.od_list:
            for k in range(self.n_paths):
                gid = self.gid_of[(od, k)]
                idx = [self.edge_index[e] for e in routes_by_od[od][k]
                       if e in self.edge_index]
                if idx:
                    A[gid, idx] = 1.0
                    rlen[gid] = float(self.edge_length[idx].sum())
                    nlink[gid] = len(idx)
        self.A = A
        self.route_length = np.maximum(rlen, 1e-6)
        self.route_nlinks = np.maximum(nlink, 1.0)

        # ---- the DECLARED per-link operator: 1/capacity ------------------------
        # Operator[a, j] = A[route_j, a] * inv_cap[a]. Kept per-link because the
        # sensor reads the BINDING link (A.4) and the channel inverse is evaluated
        # on whichever link is binding at that step (PIPELINE §6.1).
        self.op = A * self.inv_cap[None, :]          # (R, E) veh/h^-1 per vehicle

        # ---- route-pair coupling, averaged over i's links ---------------------
        # G[c, c'] = mean over a in E(c) of Operator[a, c']
        #          = (1/|E(c)|) * sum_a A[c,a] A[c',a] inv_cap[a]
        # Asymmetric by construction: the 1/|E(c)| normalisation is i's, not j's.
        self.G = (A * self.inv_cap[None, :]) @ A.T          # (R, R)
        self.G /= self.route_nlinks[:, None]

        if verbose:
            self._report()

    # ------------------------------------------------------------------ report
    def _report(self):
        s = self.summary()
        print(f"[URB-NS] network: {s['n_od']} OD x {s['n_paths']} routes = "
              f"{s['n_routes']} options over {s['n_links']} links")
        print(f"[URB-NS] capacity veh/h: min {s['cap_min']:.0f} "
              f"median {s['cap_median']:.0f} max {s['cap_max']:.0f} "
              f"(lanes 1..{int(self.edge_lanes.max())}, SAT_FLOW={self.sat_flow:.0f})")
        print(f"[URB-NS] operator spread std/mean = {s['op_spread']:.3f}, "
              f"asymmetry {s['asymmetry']:.3f}  "
              f"(PIPELINE 2.1 wants BOTH clearly non-zero)")
        if s["missing_link_frac"] > 0:
            print(f"[URB-NS] WARNING {s['missing_link_frac']:.4f} of route edges "
                  f"were absent from the network file")

    # ------------------------------------------------------------------ checks
    def check_fft(self, env_ffts, rtol=1e-3):
        """Route ORDER vs the environment's own free-flow times. A permuted order
        would aim every operator row at the wrong road while every diagnostic
        still looked healthy, so this is verified rather than trusted."""
        n, worst, bad = 0, 0.0, []
        for od in self.od_list:
            ref = env_ffts.get(od)
            if ref is None:
                continue
            for k in range(min(self.n_paths, len(ref))):
                a, b = float(self.ffts_by_od[od][k]), float(ref[k])
                if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
                    continue
                rel = abs(a - b) / max(abs(b), 1e-9)
                n += 1
                worst = max(worst, rel)
                if rel > rtol:
                    bad.append((od, k, a, b))
        return n, worst, bad

    def assert_structure(self):
        """The structural checks PIPELINE §2.4 requires at construction."""
        out = []
        gd = float(np.max(np.abs(np.diag(self.G))))
        out.append(("G has a non-zero diagonal by design (self-overlap); the "
                    "agent-level operator is zeroed in build_operator", True))
        assert np.all(self.cap > 0), "non-positive link capacity"
        assert np.all(np.isfinite(self.G)), "non-finite entry in the route Gram"
        return gd, out

    # ------------------------------------------------------------------ operator
    def build_operator(self, agent_gids, co_presence):
        """The agent-level declared operator `W`, zero-diagonal.

            W[i, j] = co_presence[i, j] * mean_{k, k'} G[gid(i,k), gid(j,k')]
            W[i, i] = 0                                       ASSERTED

        The mean over route options is the GEOMETRIC REFERENCE (peers acting
        uniformly at random) -- no run data enters, so `W` stays a declared model
        class rather than a fit (PIPELINE §2.3).
        """
        agent_gids = np.asarray(agent_gids, dtype=np.int64)
        n, K = agent_gids.shape
        # mean over the K x K route-option pairs, vectorised
        Wm = np.zeros((n, n))
        for k in range(K):
            rows = self.G[agent_gids[:, k], :]                # (n, R)
            for kp in range(K):
                Wm += rows[:, agent_gids[:, kp]]              # (n, n)
        Wm /= float(K * K)
        W = Wm * np.asarray(co_presence, dtype=np.float64)
        np.fill_diagonal(W, 0.0)                              # ASSERTED, not argued
        assert np.all(np.diag(W) == 0.0), "W diagonal is not zero"
        return W

    # ------------------------------------------------------------------ summary
    def summary(self):
        off = self.G[~np.eye(self.R, dtype=bool)]
        nz = off[off > 0]
        spread = float(nz.std() / nz.mean()) if nz.size else 0.0
        asym = float(np.mean(np.abs(self.G - self.G.T)) / max(1e-12, np.mean(self.G)))
        return {
            "n_od": self.n_od, "n_paths": self.n_paths, "n_routes": self.R,
            "n_links": self.E,
            "missing_link_frac": round(self.missing_frac, 6),
            "cap_min": float(self.cap.min()), "cap_max": float(self.cap.max()),
            "cap_median": float(np.median(self.cap)),
            "op_spread": round(spread, 4),
            "asymmetry": round(asym, 4),
            "sat_flow": self.sat_flow,
        }


# ==========================================================================
def build_co_presence(start_times, durations, slack=0.0):
    """Fixed (N, N) co-presence: 1 where two trips are on the road together.

    Departure times come from `agents.csv` and never change; duration is the trip's
    mean free-flow time. So this is EXOGENOUS STRUCTURE, computable before the
    first episode and constant thereafter -- which is what keeps it a declared part
    of the model class rather than something fitted.

    It is also what makes the regressor vary: two travellers who never share the
    road never load each other.
    """
    s = np.asarray(start_times, dtype=np.float64)
    T = np.maximum(np.asarray(durations, dtype=np.float64), 0.0)
    e = s + T + float(slack)
    return ((s[:, None] <= e[None, :]) & (s[None, :] <= e[:, None])).astype(np.float64)

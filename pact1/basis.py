"""The KNOWN model class: SUMO network geometry -> road classes -> route overlap.

This is the ``{B_m}`` of the guide (I.7), instantiated for URB. Everything in this
file is built ONCE, before any episode runs, from three artefacts that exist prior
to training:

    <net>.net.xml          the road network (link lengths, lane counts, speeds)
    paths.csv              the generated route set (which links each route uses)
    agents.csv / the env   who travels, from where to where, at what time

THE HONEST CONTRACT (guide I.7)
--------------------------------------------------------------------------------
"Know the model CLASS, estimate the PARAMETERS." The class handed to the agent here
is: *there are r road classes, and congestion on each has its own marginal cost.*
That is public infrastructure — a road's lane count and speed limit are painted on
it. What is NOT handed over is ``beta*``: how much delay a peer vehicle on each
class actually costs today. That drifts (the operating point on the congestion
curve moves as the fleet's route mix moves) and must be tracked online.

Contrast with Ant, and say it plainly in the paper: on Ant the basis was invented
along with the coupling. Here the basis is the city.

THE FOUR RULES FOR A BASIS (guide I.7), CHECKED
--------------------------------------------------------------------------------
1. zero-diagonal  -- every ``x_m`` excludes the agent's own contribution, so the
                     ``sum_{j != i}`` signature holds exactly. Enforced in
                     ``RouteBasis.waveforms`` and gated in ``selftest.py``.
2. legacy point   -- N/A here: there is no injected coupling to reproduce. The
                     ``g = 0`` floor plays the same role and is gated exactly.
3. bounded        -- N/A here: theta is not something we set, it is measured.
4. slow timescale -- the basis is FIXED geometry; only beta* moves.
"""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np

# Road-class boundaries in m/s. Chosen to land on the standard OSM/SUMO speed
# tiers, not fitted to anything:
#     <= 8.5   ~ 30 km/h and below  -> residential / local streets
#     <= 14.0  ~ 50 km/h            -> urban main / collector
#     >  14.0  ~ 60 km/h and above  -> primary / trunk / fast
# DECLARED CONSTANTS. If a network is degenerate under them the builder falls back
# to length-weighted speed tertiles and says so, loudly, in the banner.
DEFAULT_SPEED_BOUNDS = (8.5, 14.0)
CLASS_NAMES = ("local", "collector", "arterial")

_SPLIT = re.compile(r"[,\s]+")


# ==========================================================================
#  SUMO network
# ==========================================================================
def parse_sumo_net(net_path):
    """Read ``<net>.net.xml`` -> {edge_id: (length_m, n_lanes, speed_mps)}.

    Internal junction edges (ids starting with ':') are skipped: they carry no
    routable length in RouteRL's route strings. Edge length is taken as the mean
    lane length and speed as the max lane speed, which is how SUMO itself reports
    an edge whose lanes differ.
    """
    if not os.path.exists(net_path):
        raise FileNotFoundError(f"[PACT-1] SUMO network not found: {net_path}")

    edges = {}
    # iterparse keeps memory flat on the 20k-edge networks.
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
            edges[eid] = (
                float(np.mean(lengths)),
                max(1, len(lengths)),
                float(max(speeds)) if speeds else 0.0,
            )
        elem.clear()

    if not edges:
        raise ValueError(f"[PACT-1] No routable edges parsed from {net_path}")
    return edges


_PATH_COLS = ("origins", "destinations", "path", "free_flow_time")
_PRUNE = {".git", "networks", "SUMO_output", "episodes", "plots", "__pycache__",
          "docs", "leaderboard", ".venv", "venv"}


def _looks_like_paths_csv(p):
    try:
        import pandas as pd
        head = pd.read_csv(p, nrows=1)
    except Exception:                                     # noqa: BLE001
        return False
    return all(c in head.columns for c in _PATH_COLS)


def find_paths_csv(explicit=None, candidates=(), search_roots=(), verbose=True):
    """Locate RouteRL's generated route table, wherever this version puts it.

    RouteRL writes the route set during path generation, but WHERE depends on the
    version and on which of the several folder parameters it honours -- URB's own
    clustered pipeline sidesteps the question by writing the file itself. Rather
    than hard-coding a guess that breaks on the next release, look in the obvious
    places and then walk a few bounded roots for any CSV with the right schema.

    Accepts ``routes.csv`` too: URB's exporter writes the identical table under
    both names because "RouteRL might use either".
    """
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise FileNotFoundError(f"[PACT-1] --paths-csv not found: {explicit}")

    for c in candidates:
        if c and os.path.exists(c) and _looks_like_paths_csv(c):
            if verbose:
                print(f"[PACT-1] route table: {os.path.abspath(c)}", flush=True)
            return c

    hits = []
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _PRUNE]
            for fn in filenames:
                if fn in ("paths.csv", "routes.csv"):
                    p = os.path.join(dirpath, fn)
                    if _looks_like_paths_csv(p):
                        hits.append((os.path.getmtime(p), p))
    if hits:
        hits.sort(reverse=True)                            # newest first
        if verbose:
            print(f"[PACT-1] route table found by search: "
                  f"{os.path.abspath(hits[0][1])}", flush=True)
            for _, p in hits[1:4]:
                print(f"[PACT-1]   (also saw {os.path.abspath(p)})", flush=True)
        return hits[0][1]

    looked = "\n".join(f"          {os.path.abspath(c)}" for c in candidates if c)
    roots = "\n".join(f"          {os.path.abspath(r)}" for r in search_roots if r)
    raise FileNotFoundError(
        "[PACT-1] could not find RouteRL's route table (paths.csv / routes.csv).\n"
        "        It must exist before the basis can be built -- it is what maps a\n"
        "        route index to the links that route uses.\n\n"
        f"        Checked these exact paths:\n{looked}\n"
        f"        And searched under:\n{roots}\n\n"
        "        Fix: find it and pass it explicitly, e.g.\n"
        "          find /workspace/dsvrc/urb -name 'paths.csv' -newermt '-1 hour'\n"
        "          python scripts/pact1.py ... --paths-csv /path/to/paths.csv"
    )


def load_paths_csv(paths_csv, n_paths):
    """Read RouteRL's ``paths.csv`` -> ({(o, d): [edge_list per action index]},
    {(o, d): [free_flow_time per action index]}).

    Schema written by RouteRL / URB: ``origins, destinations, path,
    free_flow_time`` and optionally ``cluster``. ``path`` is the edge sequence,
    which URB writes space-separated in one code path and comma-separated in
    another — so it is split on either.

    Action-index alignment is the one thing that can be silently wrong here (a
    permuted route order would make every waveform point at the wrong road while
    still looking perfectly healthy). It is NOT trusted: ``RouteBasis.check_fft``
    verifies the free-flow times parsed here against the environment's own
    ``get_free_flow_times()`` and aborts on any mismatch.
    """
    import pandas as pd

    if not os.path.exists(paths_csv):
        raise FileNotFoundError(
            f"[PACT-1] paths.csv not found: {paths_csv}\n"
            "        It is written by RouteRL during path generation; PACT-1 must be\n"
            "        constructed AFTER env.start()."
        )
    df = pd.read_csv(paths_csv)
    for col in ("origins", "destinations", "path", "free_flow_time"):
        if col not in df.columns:
            raise ValueError(
                f"[PACT-1] paths.csv is missing column {col!r}. Found: {list(df.columns)}"
            )

    routes, ffts = {}, {}
    for (o, d), grp in df.groupby(["origins", "destinations"], sort=True):
        if "cluster" in grp.columns and grp["cluster"].notna().all():
            grp = grp.sort_values("cluster")
        key = (int(o), int(d))
        edge_lists, ff = [], []
        for row in grp.itertuples(index=False):
            p = [e for e in _SPLIT.split(str(row.path).strip()) if e]
            edge_lists.append(p)
            ff.append(float(row.free_flow_time))
        if len(edge_lists) != n_paths:
            raise ValueError(
                f"[PACT-1] OD {key} has {len(edge_lists)} routes in paths.csv but the "
                f"env config declares number_of_paths={n_paths}. Refusing to guess."
            )
        routes[key] = edge_lists
        ffts[key] = ff
    if not routes:
        raise ValueError(f"[PACT-1] paths.csv parsed to zero OD pairs: {paths_csv}")
    return routes, ffts


def load_routes_rou_xml(rou_path, n_paths):
    """Fallback route source: SUMO's ``route.rou.xml``.

    Route ids are ``{origin}_{destination}_{index}``, so the action index is stated
    EXPLICITLY rather than implied by row order -- which means this source cannot
    suffer the permutation failure that ``paths.csv`` can. It carries no free-flow
    times, so those come from the environment.
    """
    if not os.path.exists(rou_path):
        raise FileNotFoundError(rou_path)
    routes = {}
    for _, elem in ET.iterparse(rou_path, events=("end",)):
        if elem.tag != "route":
            continue
        rid = elem.get("id") or ""
        edges = [e for e in _SPLIT.split((elem.get("edges") or "").strip()) if e]
        parts = rid.split("_")
        if len(parts) >= 3 and edges:
            try:
                o, d, k = int(parts[-3]), int(parts[-2]), int(parts[-1])
            except ValueError:
                elem.clear()
                continue
            routes.setdefault((o, d), {})[k] = edges
        elem.clear()
    if not routes:
        raise ValueError(f"[PACT-1] no usable <route> entries in {rou_path}")

    out = {}
    for od, by_k in routes.items():
        if sorted(by_k) != list(range(n_paths)):
            continue                       # incomplete OD: cannot be described
        out[od] = [by_k[k] for k in range(n_paths)]
    if not out:
        raise ValueError(
            f"[PACT-1] {rou_path} has no OD pair with a complete set of "
            f"{n_paths} routes."
        )
    return out


def load_route_table(n_paths, env_ffts, explicit=None, candidates=(),
                     search_roots=(), rou_candidates=()):
    """Get {(o,d): [edge lists]} + {(o,d): [free-flow times]} from whatever this
    RouteRL version left on disk.

    Returns (routes_by_od, ffts_by_od, source_tag). ``source_tag`` is
    ``'paths.csv'`` or ``'route.rou.xml'``; the caller uses it to decide whether
    the route-order gate is meaningful (it is not for the xml, where the index is
    encoded in the id and therefore correct by construction).
    """
    try:
        p = find_paths_csv(explicit=explicit, candidates=candidates,
                           search_roots=search_roots)
        r, f = load_paths_csv(p, n_paths)
        return r, f, "paths.csv"
    except FileNotFoundError as first:
        for rp in rou_candidates:
            if rp and os.path.exists(rp):
                try:
                    r = load_routes_rou_xml(rp, n_paths)
                except (ValueError, FileNotFoundError):
                    continue
                f = {}
                for od in list(r.keys()):
                    ff = env_ffts.get(od)
                    if ff is None or len(ff) < n_paths:
                        r.pop(od)
                        continue
                    f[od] = [float(v) for v in ff[:n_paths]]
                if not r:
                    continue
                print(f"[PACT-1] route table: {os.path.abspath(rp)} "
                      f"(paths.csv absent; free-flow times taken from the env)",
                      flush=True)
                return r, f, "route.rou.xml"
        raise first


# ==========================================================================
#  The basis
# ==========================================================================
class RouteBasis:
    """Road-class route/route overlap, i.e. the r known coupling channels.

        G_m[c, c'] = ( sum_{a in E_c & E_c' & class_m}  len_a / lanes_a ) / L_c

    Read it as: *if one peer drives route c', how much class-m per-lane road does it
    put in front of me when I drive route c* — normalised by my own route length so
    it is comparable across long and short trips.

    Agent i's waveform for candidate route c is then

        x_m,i(c) = sum_{j != i}  O_ij * G_m[c, route_j]

    with ``O`` the fixed temporal co-presence matrix. Both factors are exact
    arithmetic over declared quantities (network geometry + peers' executed route
    choices); nothing here is estimated.
    """

    def __init__(
        self,
        net_edges,
        routes_by_od,
        ffts_by_od,
        n_paths,
        speed_bounds=DEFAULT_SPEED_BOUNDS,
        min_class_share=0.03,
        max_basis_mb=4000.0,
        verbose=True,
    ):
        self.n_paths = int(n_paths)
        self.routes_by_od = routes_by_od
        self.ffts_by_od = ffts_by_od
        self.speed_bounds = tuple(speed_bounds)
        self.fallback_used = False

        # ---- global route index: (o, d, k) -> gid ----------------------------
        self.od_list = sorted(routes_by_od.keys())
        self.od_index = {od: j for j, od in enumerate(self.od_list)}
        self.n_od = len(self.od_list)
        self.R = self.n_od * self.n_paths
        self.gid_of = {}
        for od, j in self.od_index.items():
            for k in range(self.n_paths):
                self.gid_of[(od, k)] = j * self.n_paths + k

        # ---- edges actually used by some route -------------------------------
        used, missing = [], set()
        for od in self.od_list:
            for path in routes_by_od[od]:
                for e in path:
                    if e in net_edges:
                        used.append(e)
                    else:
                        missing.add(e)
        if not used:
            raise ValueError("[PACT-1] No route edge matched the SUMO network.")
        self.edge_ids = sorted(set(used))
        self.edge_index = {e: i for i, e in enumerate(self.edge_ids)}
        self.E = len(self.edge_ids)
        self.missing_edges = missing
        self.missing_frac = len(missing) / max(1, len(missing) + self.E)

        length = np.array([net_edges[e][0] for e in self.edge_ids], dtype=np.float64)
        lanes = np.array([net_edges[e][1] for e in self.edge_ids], dtype=np.float64)
        speed = np.array([net_edges[e][2] for e in self.edge_ids], dtype=np.float64)
        self.edge_length, self.edge_lanes, self.edge_speed = length, lanes, speed

        # ---- memory guard BEFORE allocating anything large -------------------
        mb = (self.R * self.E * 8 + 3 * self.R * self.R * 8) / 1e6
        if mb > float(max_basis_mb):
            raise MemoryError(
                f"[PACT-1] basis would need ~{mb:.0f} MB (R={self.R} routes, "
                f"E={self.E} edges) which exceeds max_basis_mb={max_basis_mb}. "
                "Raise pact1_cfg.max_basis_mb if the machine can take it."
            )
        self.basis_mb = mb

        # ---- route incidence A (R x E), binary --------------------------------
        A = np.zeros((self.R, self.E), dtype=np.float64)
        route_len = np.zeros(self.R, dtype=np.float64)
        for od in self.od_list:
            for k in range(self.n_paths):
                gid = self.gid_of[(od, k)]
                idx = [self.edge_index[e] for e in routes_by_od[od][k]
                       if e in self.edge_index]
                if idx:
                    A[gid, idx] = 1.0
                    route_len[gid] = float(length[idx].sum())
        self.A = A
        self.route_length = np.maximum(route_len, 1e-6)

        # ---- road classes ------------------------------------------------------
        self.edge_class = self._classify(speed, length, A, min_class_share, verbose)
        self.r = 3

        # ---- Gram matrices G_m --------------------------------------------------
        w = length / np.maximum(lanes, 1.0)          # per-lane metres of road
        inv_L = 1.0 / self.route_length
        self.G = []
        for m in range(self.r):
            wm = w * (self.edge_class == m)
            Gm = (A * wm[None, :]) @ A.T             # (R, R)
            Gm *= inv_L[:, None]                     # normalise by MY route length
            self.G.append(np.ascontiguousarray(Gm))

        # ---- structural normalisation ------------------------------------------
        # Both scales come from geometry alone (never from run data), so this is
        # part of the basis definition rather than a tuned knob. It only puts x in
        # a sane numerical range for the RLS; it cannot change what is identifiable.
        self.g_scale = np.ones(self.r)
        for m in range(self.r):
            nz = self.G[m][self.G[m] > 0]
            if nz.size:
                self.g_scale[m] = float(nz.mean())
                self.G[m] = self.G[m] / self.g_scale[m]

        self.class_share = self._class_share(A, length)

    # ------------------------------------------------------------------ classes
    def _classify(self, speed, length, A, min_class_share, verbose):
        cls = np.searchsorted(np.asarray(self.speed_bounds, dtype=np.float64), speed)
        share = self._class_share(A, length, cls)
        if share.min() >= float(min_class_share):
            return cls

        # Degenerate under the declared bounds (e.g. a network that is uniformly
        # 50 km/h). Fall back to length-weighted speed TERTILES so the basis stays
        # non-degenerate, and make the substitution impossible to miss in the log.
        self.fallback_used = True
        w = np.repeat(speed, np.maximum(1, (length / 10.0).astype(int)))
        q = np.quantile(w, [1.0 / 3.0, 2.0 / 3.0]) if w.size else np.array([0.0, 0.0])
        self.speed_bounds = (float(q[0]), float(q[1]))
        cls2 = np.searchsorted(np.asarray(self.speed_bounds), speed)
        share2 = self._class_share(A, length, cls2)
        if verbose:
            print(
                "\n[PACT-1][BASIS WARNING] declared speed bounds gave a degenerate "
                f"partition (class shares {np.round(share, 4).tolist()}, "
                f"min < {min_class_share}).\n"
                f"                        Fell back to length-weighted speed tertiles "
                f"at {np.round(self.speed_bounds, 2).tolist()} m/s "
                f"-> shares {np.round(share2, 4).tolist()}.\n"
                "                        This is a BASIS CHANGE. Report it; do not "
                "compare against a run that did not take this path.",
                flush=True,
            )
        if share2.min() < 1e-6:
            raise ValueError(
                "[PACT-1] Road-class partition is degenerate even under tertiles "
                f"(shares {share2.tolist()}). This network cannot support r=3; the "
                "reduction would be rank-deficient by construction."
            )
        return cls2

    def _class_share(self, A, length, cls=None):
        """Route-weighted share of length in each class (what the basis actually
        sees — an unused class of streets is irrelevant however big it is)."""
        if cls is None:
            cls = self.edge_class
        cover = (A.sum(0) > 0).astype(np.float64) * length
        tot = cover.sum()
        if tot <= 0:
            return np.zeros(3)
        return np.array([cover[cls == m].sum() / tot for m in range(3)])

    # ------------------------------------------------------------------ checks
    def check_fft(self, env_ffts, rtol=1e-3):
        """Verify paths.csv route ORDER against the environment's free-flow times.

        This is the alignment gate. If paths.csv rows were in a different order
        than the action index, every waveform would describe the wrong road while
        looking perfectly healthy — the single most dangerous silent failure in
        this whole file. Free-flow time is a per-route fingerprint, so comparing it
        catches any permutation.

        Returns (n_checked, max_rel_err, offenders).
        """
        n, worst, bad = 0, 0.0, []
        for od in self.od_list:
            ref = env_ffts.get(od)
            if ref is None:
                continue
            for k in range(self.n_paths):
                if k >= len(ref):
                    continue
                a, b = float(self.ffts_by_od[od][k]), float(ref[k])
                if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
                    continue
                rel = abs(a - b) / max(abs(b), 1e-9)
                n += 1
                if rel > worst:
                    worst = rel
                if rel > rtol:
                    bad.append((od, k, a, b))
        return n, worst, bad

    # ------------------------------------------------------------------ waveforms
    def waveforms(self, O_rows, peer_gid, self_gid, agent_gids):
        """Exact peer-load waveforms. THE arithmetic that the whole method rests on.

            x_m[i, k] = sum_{j != i}  O_rows[i, j] * G_m[ agent_gids[i, k], peer_gid[j] ]

        Args:
            O_rows:      (n_i, n_peers) temporal co-presence, rows = the agents we
                         are computing for, columns = every peer in scope.
            peer_gid:    (n_peers,) global route id each peer actually drove.
            self_gid:    (n_i,) global route id THIS agent drove, subtracted out so
                         the sum is strictly over j != i (the category-C signature).
                         Pass -1 for an agent that did not travel.
            agent_gids:  (n_i, K) the K candidate routes of each agent.

        Returns (r, n_i, K).
        """
        O_rows = np.asarray(O_rows, dtype=np.float64)
        peer_gid = np.asarray(peer_gid, dtype=np.int64)
        self_gid = np.asarray(self_gid, dtype=np.int64)
        agent_gids = np.asarray(agent_gids, dtype=np.int64)
        n_i, K = agent_gids.shape

        out = np.zeros((self.r, n_i, K), dtype=np.float64)
        ok = np.nonzero(peer_gid >= 0)[0]
        if ok.size == 0:
            return out
        pg = peer_gid[ok]                     # (n_ok,) route ids actually driven
        Ok = np.ascontiguousarray(O_rows[:, ok])   # (n_i, n_ok)

        # --- the sum over ALL peers in scope, including i itself ----------------
        for m in range(self.r):
            Gm = self.G[m]
            for k in range(K):
                sub = Gm[np.ix_(agent_gids[:, k], pg)]      # (n_i, n_ok)
                out[m, :, k] = np.einsum("ij,ij->i", Ok, sub)

        # --- zero-diagonal: subtract i's OWN contribution -----------------------
        # i appears once in the peer list (if it travelled), contributing
        # O_ii * G_m[gid(i,k), self_gid[i]] with O_ii == 1 by construction. Removing
        # it is what makes x_m a strict sum over j != i -- the category-C signature,
        # and the reason a lone traveller reads exactly zero fleet load.
        has_self = np.nonzero(self_gid >= 0)[0]
        if has_self.size:
            sg = self_gid[has_self]
            for m in range(self.r):
                Gm = self.G[m]
                for k in range(K):
                    out[m, has_self, k] -= Gm[agent_gids[has_self, k], sg]
        return out

    # ------------------------------------------------------------------ brute force
    def waveforms_bruteforce(self, O_rows, peer_gid, self_gid, agent_gids):
        """Dead-simple loop straight off the definition.

        Used ONLY by the startup arithmetic gate, to prove the vectorised path above
        is wired correctly (index order, self-exclusion, peer masking). It is
        gait-independent arithmetic, which is exactly why a hard abort on a mismatch
        is safe -- it can only mean a wiring bug. Never called during training.
        """
        O_rows = np.asarray(O_rows, dtype=np.float64)
        agent_gids = np.asarray(agent_gids, dtype=np.int64)
        peer_gid = np.asarray(peer_gid, dtype=np.int64)
        self_gid = np.asarray(self_gid, dtype=np.int64)
        n_i, K = agent_gids.shape

        out = np.zeros((self.r, n_i, K), dtype=np.float64)
        for m in range(self.r):
            for i in range(n_i):
                for k in range(K):
                    c = int(agent_gids[i, k])
                    s = 0.0
                    for j in range(peer_gid.shape[0]):
                        gj = int(peer_gid[j])
                        if gj < 0:
                            continue
                        s += float(O_rows[i, j]) * float(self.G[m][c, gj])
                    if int(self_gid[i]) >= 0:
                        s -= float(self.G[m][c, int(self_gid[i])])
                    out[m, i, k] = s
        return out

    def summary(self):
        return {
            "n_od": self.n_od,
            "n_routes": self.R,
            "n_paths": self.n_paths,
            "n_edges_used": self.E,
            "missing_edge_frac": round(self.missing_frac, 6),
            "speed_bounds_mps": [round(float(b), 3) for b in self.speed_bounds],
            "class_names": list(CLASS_NAMES),
            "class_length_share": [round(float(s), 4) for s in self.class_share],
            "g_scale": [round(float(s), 6) for s in self.g_scale],
            "fallback_tertiles": bool(self.fallback_used),
            "basis_mb": round(float(self.basis_mb), 1),
        }


# ==========================================================================
#  Temporal co-presence
# ==========================================================================
def build_time_overlap(start_times, durations, slack=0.0):
    """Fixed (N, N) co-presence matrix: 1 where two trips are on the road together.

        O_ij = 1  iff  [s_i, s_i + T_i]  and  [s_j, s_j + T_j]  overlap (+ slack)

    Departure times come from ``agents.csv`` and never change, and ``T`` is the
    agent's mean free-flow time over its own route options — so this is EXOGENOUS
    STRUCTURE, computable before the first episode and constant thereafter. That is
    what keeps it a declared part of the model class rather than something fitted.

    It is also what makes the regressor vary: two travellers who never share the
    road never load each other, so ``x`` differs across agents by construction and
    the design matrix does not collapse (guide I.3's counter-check).

    ``O_ii == 1`` by construction; the self term is subtracted explicitly in
    ``RouteBasis.waveforms``.
    """
    s = np.asarray(start_times, dtype=np.float64)
    T = np.maximum(np.asarray(durations, dtype=np.float64), 0.0)
    e = s + T + float(slack)
    return ((s[:, None] <= e[None, :]) & (s[None, :] <= e[:, None])).astype(np.float64)

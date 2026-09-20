"""Who counts as a neighbour, for the baselines that need a neighbourhood.

Three baselines are defined on a set of "the other agents that matter":

  * **DGN** (B3) convolves over ``B_i``, "determined by distance or other metrics
    depending on the environment". Its paper fixes ``|B| = 3`` and its Table 4
    shows 3 beating 1, 2 and 4.
  * **MF-Q** (B4) averages the ONE-HOT actions of ``N(j)``.
  * **LIAM** (B5) reconstructs the observations and actions of a fixed, small set
    of modelled agents.

URB gives three candidate structures, and the right one is NOT the same for all
three, so the choice is made per baseline and recorded in its checklist:

``od``          same origin-destination pair.
                The only structure under which averaging one-hot action vectors
                is meaningful: action ``k`` indexes the ``k``-th route OF AN OD
                PAIR, so route 0 for OD A and route 0 for OD B are different
                roads and their one-hot mean is not a quantity.

                **On every URB network shipped with the benchmark this set is a
                SINGLETON and is therefore useless.** Measured: the median OD
                pair has exactly one traveller on all seven networks, and the
                maximum is two on six of them (the exception is
                ``ingolstadt_custom``, where 306 OD pairs carry 1035
                travellers). No arm here defaults to ``od`` for that reason;
                it is kept because it is the semantically correct reading and
                because a future network may populate it.

``copresence``  trips that are on the road at the same time
                (``urb_ns.network.build_co_presence``, departure time plus mean
                free-flow duration). Exogenous: departure times come from
                ``agents.csv`` and never change.

``overlap``     OD pairs whose ROUTE SETS share at least one edge, intersected
                with ``copresence``. This is the actual coupling graph -- two
                travellers load each other exactly when they are on a shared link
                at a shared time -- and is what DGN gets by default, because
                DGN's whole claim is that convolving over the interaction graph
                recovers the coupling.

Ranking within a structure is by ``|start_time_i - start_time_j|``: closest in
time is most likely to genuinely share the road, and it is a total order, so the
neighbour set is deterministic given ``agents.csv``.

WHAT THIS FILE NEVER DOES
--------------------------------------------------------------------------------
It never uses a travel time, a reward or anything else the run produces. Every
structure here is a function of ``agents.csv`` and the generated route table, and
is therefore fixed before the first episode -- which is what lets a baseline say
"the graph was declared" rather than "the graph was fitted".
"""

import numpy as np

__all__ = ["NeighbourGraph", "build_from_ctx"]


# ==========================================================================
#  building one from a HostContext, with the fallbacks in ONE place
# ==========================================================================
def _load_routes(ctx, tag):
    """The generated route table, or ``(None, None)`` with a printed reason."""
    if not getattr(ctx, "routes_csv", None):
        return None, None
    try:
        from urb_ns.network import load_route_table
        routes, ffts = load_route_table(ctx.routes_csv,
                                        int(ctx.params["number_of_paths"]))
    except Exception as exc:                              # noqa: BLE001
        print(f"[{tag}][WARN] could not read the route table "
              f"{ctx.routes_csv}: {type(exc).__name__}: {exc}", flush=True)
        return None, None
    return routes, {od: float(np.mean(v)) for od, v in ffts.items()}


def _durations_from_freeflow(ctx):
    out = {}
    for od, row in ctx.free_flow.items():
        r = np.asarray(row, dtype=np.float64)
        r = r[np.isfinite(r)]
        if r.size:
            out[od] = float(r.mean())
    return out


def build_from_ctx(ctx, mode="overlap", n_neighbors=3, slack=0.0, tag="URB-BL",
                   verbose=True):
    """``NeighbourGraph`` from a ``HostContext``, with the route-table fallback.

    ``overlap`` needs the generated route table; when it is missing the mode
    falls back to ``copresence`` and says so, rather than raising -- a config
    that names a graph the city cannot build should degrade to the next best
    structure with a warning, not take the run down seventeen minutes in.

    Trip durations always come from the environment's own published free-flow
    times when the route table is unavailable, so ``copresence`` is never
    reduced to "departs at the same second".
    """
    routes = None
    dur = None
    mode = str(mode).lower()
    if mode == "overlap":
        routes, dur = _load_routes(ctx, tag)
        if routes is None:
            print(f"[{tag}][WARN] graph='overlap' needs the route table and none "
                  f"was found; falling back to 'copresence'. Pass --routes to "
                  f"use the coupling graph.", flush=True)
            mode = "copresence"
    if dur is None:
        dur = _durations_from_freeflow(ctx)
    return NeighbourGraph(ctx.agent_table, ctx.av_ids, mode=mode,
                          n_neighbors=int(n_neighbors), routes_by_od=routes,
                          durations=dur, slack=float(slack), verbose=verbose)


class NeighbourGraph(object):
    """Static neighbour structure over the machine agents.

    Args:
        agent_table: ``{aid: {"od": (o, d), "start": float, "machine": bool}}``
                     for EVERY traveller, humans included.
        av_ids:      ordered list of machine-agent ids -- the graph's node set.
        mode:        ``"od" | "copresence" | "overlap"``.
        n_neighbors: ``|B_i|``; the ego is stored separately, so a row is
                     ``1 + n_neighbors`` wide.
        routes_by_od: ``{(o, d): [[edge_id, ...] x K]}``, required for ``overlap``.
        durations:   ``{(o, d): seconds}`` mean free-flow trip time per OD, used
                     by ``copresence`` / ``overlap``. Missing ODs fall back to the
                     median, which only widens the window and never invents a
                     neighbour that departs after everyone has arrived.
        slack:       extra seconds added to a trip's road occupancy.
    """

    MODES = ("od", "copresence", "overlap")

    def __init__(self, agent_table, av_ids, mode="overlap", n_neighbors=3,
                 routes_by_od=None, durations=None, slack=0.0, verbose=True):
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(f"[URB-BL] graph mode must be one of {self.MODES}, "
                             f"got {mode!r}")
        if mode == "overlap" and not routes_by_od:
            raise ValueError(
                "[URB-BL] graph mode 'overlap' needs the route table. Pass "
                "routes_by_od (urb_ns.network.load_route_table) or choose "
                "mode='copresence', which needs no route geometry.")
        self.mode = mode
        self.n_neighbors = int(n_neighbors)
        self.av_ids = [str(a) for a in av_ids]
        self.n = len(self.av_ids)
        self.slot_of = {a: i for i, a in enumerate(self.av_ids)}
        self.agent_table = agent_table

        self.od = [tuple(agent_table[a]["od"]) for a in self.av_ids]
        self.start = np.array([float(agent_table[a]["start"])
                               for a in self.av_ids], dtype=np.float64)

        dur = dict(durations or {})
        if dur:
            med = float(np.median([v for v in dur.values() if np.isfinite(v)]))
        else:
            med = 0.0
        self.duration = np.array([float(dur.get(o, med)) for o in self.od],
                                 dtype=np.float64)

        self._od_pair_links = None
        if mode == "overlap":
            self._od_pair_links = {
                k: set().union(*[set(r) for r in v]) if v else set()
                for k, v in routes_by_od.items()
            }

        self.adj, self.mask, self.degree = self._build(slack)

        # Every traveller, machines and humans, grouped by OD. MF-Q's mean action
        # is over this, not over the machine set: a human on the same OD loads the
        # same roads and is part of the field by any reading of the method.
        self.od_members = {}
        for a, rec in agent_table.items():
            self.od_members.setdefault(tuple(rec["od"]), []).append(str(a))
        for v in self.od_members.values():
            v.sort(key=lambda x: float(agent_table[x]["start"]))

        if verbose:
            self.banner()

    # ------------------------------------------------------------------ build
    def _candidates(self, i, slack):
        """Boolean mask over nodes: who is eligible to be i's neighbour."""
        same_od = np.array([o == self.od[i] for o in self.od], dtype=bool)
        if self.mode == "od":
            ok = same_od
        else:
            # co-presence: [s_i, s_i + T_i + slack] overlaps [s_j, s_j + T_j + slack]
            e = self.start + np.maximum(self.duration, 0.0) + float(slack)
            ok = (self.start[i] <= e) & (self.start <= e[i])
            if self.mode == "overlap":
                mine = self._od_pair_links.get(self.od[i], set())
                if mine:
                    shares = np.array(
                        [bool(mine & self._od_pair_links.get(o, set()))
                         for o in self.od], dtype=bool)
                else:
                    shares = same_od
                ok = ok & shares
        ok[i] = False
        return ok

    def _build(self, slack):
        w = 1 + self.n_neighbors
        adj = np.zeros((self.n, w), dtype=np.int64)
        mask = np.zeros((self.n, w), dtype=bool)
        degree = np.zeros(self.n, dtype=np.int64)
        for i in range(self.n):
            adj[i, 0] = i               # DGN: row 0 of C_i is the one-hot of i
            mask[i, 0] = True
            cand = np.flatnonzero(self._candidates(i, slack))
            degree[i] = len(cand)
            if len(cand):
                order = np.argsort(np.abs(self.start[cand] - self.start[i]),
                                   kind="stable")
                pick = cand[order][:self.n_neighbors]
                adj[i, 1:1 + len(pick)] = pick
                mask[i, 1:1 + len(pick)] = True
                # Pad the remaining slots with the ego. They are masked out of the
                # attention softmax, so the value is irrelevant -- but a padded
                # index must still be a VALID node index, because it is used to
                # gather features before the mask is applied.
                if len(pick) < self.n_neighbors:
                    adj[i, 1 + len(pick):] = i
            else:
                adj[i, 1:] = i
        return adj, mask, degree

    # ------------------------------------------------------------------ api
    def neighbours(self, agent_id):
        """-> list of neighbour agent ids (machine agents only), ego excluded."""
        i = self.slot_of.get(str(agent_id))
        if i is None:
            return []
        return [self.av_ids[j] for j, m in zip(self.adj[i, 1:], self.mask[i, 1:])
                if m]

    def od_of(self, agent_id):
        i = self.slot_of.get(str(agent_id))
        return None if i is None else self.od[i]

    def field_members(self, agent_id, scope="od"):
        """Who is in the mean field of ``agent_id``.

        ``scope="od"``   every traveller on the same OD pair, humans included.
        ``scope="all"``  every traveller (MAgent's group-wide mean, the form the
                         MF-Q release actually runs; see algos/mfq.py).
        """
        if scope == "all":
            return list(self.agent_table.keys())
        od = self.od_of(agent_id)
        if od is None:
            rec = self.agent_table.get(str(agent_id))
            od = tuple(rec["od"]) if rec else None
        return list(self.od_members.get(od, []))

    # ------------------------------------------------------------------ report
    def banner(self):
        deg = self.degree
        filled = self.mask[:, 1:].sum(axis=1)
        print("\n" + "=" * 74)
        print("[URB-BL] NEIGHBOUR GRAPH")
        print(f"  mode           {self.mode}   |B| requested {self.n_neighbors}")
        print(f"  nodes          {self.n} machine agents")
        print(f"  candidates     min {int(deg.min())}  median "
              f"{int(np.median(deg))}  max {int(deg.max())}")
        print(f"  filled slots   {int(filled.sum())}/{self.n * self.n_neighbors} "
              f"({filled.mean():.2f} per agent)")
        print(f"  OD groups      {len(self.od_members)} "
              f"(sizes {min(len(v) for v in self.od_members.values())}"
              f"-{max(len(v) for v in self.od_members.values())}, "
              "all travellers)")
        if int(deg.min()) == 0:
            print("  NOTE: some agents have NO neighbour under this mode; their")
            print("        graph rows are the ego alone, so the convolution is the")
            print("        identity for them. That is a property of the city, not")
            print("        a bug -- but if it is most of the fleet, the graph arm")
            print("        is measuring nothing and you should say so.")
        print("=" * 74 + "\n", flush=True)

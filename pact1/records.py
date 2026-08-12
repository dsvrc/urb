"""Getting each day's travel times and executed routes out of RouteRL.

This is deliberately defensive, because it is the one place where a silent failure
costs a whole run: if no records arrive, the estimator never receives a row, the
compensator does nothing, and the arm quietly degenerates to plain IPPO while every
return curve still looks perfectly reasonable (guide II.7 -- "a return that looks
good is the failure nobody investigates").

TWO SOURCES, TRIED IN ORDER
--------------------------------------------------------------------------------
1. an in-memory list on the environment. URB's own centralized wrapper reads
   ``env.travel_times_list``, but through ``getattr(..., [])`` -- it was never
   guaranteed to exist, and in routerl 2.0.0 it does not. Several plausible names
   and holders are probed rather than assumed.

2. the per-episode CSVs RouteRL writes to ``<records>/episodes/ep<N>.csv``. This is
   the interface ``analysis/metrics.py`` itself depends on, so it is the stable
   one: columns ``id, action, travel_time, origin, destination, start_time,
   reward``.

WHY BATCHING PER EPISODE MATTERS
--------------------------------------------------------------------------------
RouteRL flushes every ``save_every`` days. Merging a multi-day flush into one
update would pair day t-4's travel time with day t's peer waveform -- a silent
misalignment that would show up only as an inexplicably low fit_r2. Each episode
file is a complete, self-consistent day (it carries every traveller's action AND
travel time), so they are kept apart and replayed in order.
"""

import os

import numpy as np

__all__ = ["RecordSource", "split_records"]

# Candidate key names, in priority order. The first list entry that a record
# actually has wins, so both record shapes work: in-memory dicts keyed by RouteRL's
# Keychain constants, and CSV rows keyed by plain column names.
DEFAULT_KEYS = {
    "id": ["id", "agent_id"],
    "action": ["action"],
    "travel_time": ["travel_time"],
}


def _get(rec, names):
    for n in names:
        if n in rec:
            v = rec[n]
            if v is not None:
                return v
    return None


class RecordSource(object):
    """Per-day records, from whatever this RouteRL version exposes."""

    MEM_ATTRS = ("travel_times_list", "last_episode_travel_times",
                 "episode_travel_times", "travel_times", "records")
    HOLDERS = (None, "recorder", "simulator", "unwrapped", "env")

    def __init__(self, env, records_folder):
        self.env = env
        self.episodes_dir = os.path.join(records_folder, "episodes")
        self.cursor = 0
        self.consumed = set()
        self.mode = None                       # resolved on first successful drain

    # ---------------------------------------------------------------- memory
    def _holders(self):
        for name in self.HOLDERS:
            obj = self.env if name is None else getattr(self.env, name, None)
            if obj is not None:
                yield obj, ("env" if name is None else f"env.{name}")

    def _drain_memory(self):
        for obj, tag in self._holders():
            for attr in self.MEM_ATTRS:
                lst = getattr(obj, attr, None)
                if not isinstance(lst, (list, tuple)) or not lst:
                    continue
                if attr == "travel_times_list":
                    # a cumulative list: take only what is new, and rewind if it
                    # was reset under us rather than skipping a whole day
                    if len(lst) < self.cursor:
                        self.cursor = 0
                    new = list(lst[self.cursor:])
                    self.cursor = len(lst)
                else:
                    new = list(lst)
                if new:
                    return new, f"{tag}.{attr}"
        return [], None

    # ---------------------------------------------------------------- csv
    def _drain_csv(self):
        if not os.path.isdir(self.episodes_dir):
            return [], None
        import pandas as pd

        found = []
        for fn in sorted(os.listdir(self.episodes_dir)):
            if not (fn.startswith("ep") and fn.endswith(".csv")) or fn in self.consumed:
                continue
            try:
                ep_no = int(fn[2:-4])
            except ValueError:
                continue
            try:
                df = pd.read_csv(os.path.join(self.episodes_dir, fn))
            except Exception:                              # noqa: BLE001
                continue                       # still being written; retry next day
            self.consumed.add(fn)
            if df.empty or not any(c in df.columns for c in DEFAULT_KEYS["id"]):
                continue
            found.append((ep_no, df.to_dict("records")))
        found.sort(key=lambda t: t[0])
        return found, ("episodes/ep*.csv" if found else None)

    # ---------------------------------------------------------------- api
    def drain(self, episode):
        """-> list of (episode_label, records). Usually length 0 or 1."""
        recs, mode = self._drain_memory()
        batches = [(episode, recs)] if recs else []
        if not batches:
            batches, mode = self._drain_csv()
        if mode and self.mode != mode:
            self.mode = mode
            print(f"[PACT-1] travel-time records are coming from {mode}", flush=True)
        return batches

    def diagnose(self):
        """Everything a future fix would need, printed once. Cheap insurance
        against another multi-minute round trip."""
        out = ["[PACT-1] record-source diagnosis:"]
        for obj, tag in self._holders():
            hits = [a for a in dir(obj)
                    if any(k in a.lower()
                           for k in ("travel", "record", "episode", "trip"))
                    and not a.startswith("__")]
            out.append(f"        {tag}: {hits[:14]}")
        out.append(f"        episodes dir: {os.path.abspath(self.episodes_dir)} "
                   f"exists={os.path.isdir(self.episodes_dir)}")
        if os.path.isdir(self.episodes_dir):
            out.append("        contents (first 8): "
                       f"{sorted(os.listdir(self.episodes_dir))[:8]}")
        return "\n".join(out)


def split_records(records, machine_ids, keys=None, aid=str):
    """-> (av_records, peer_actions, tt_hdv, n_bad).

    av_records:   id -> (action, travel_time)   machine agents only
    peer_actions: id -> action                  EVERY traveller that completed a trip
    """
    k = dict(DEFAULT_KEYS if keys is None else keys)
    av, peers, hdv, bad = {}, {}, [], 0
    for rec in records:
        if not isinstance(rec, dict):
            bad += 1
            continue
        rid = _get(rec, k["id"])
        act = _get(rec, k["action"])
        tt = _get(rec, k["travel_time"])
        if rid is None or act is None:
            bad += 1
            continue
        key = aid(rid)
        try:
            peers[key] = int(act)
        except (TypeError, ValueError):
            bad += 1
            continue
        try:
            tt = None if tt is None else float(tt)
        except (TypeError, ValueError):
            tt = None
        if tt is not None and not np.isfinite(tt):
            tt = None
        if key in machine_ids:
            if tt is not None:
                av[key] = (int(act), tt)
        elif tt is not None:
            hdv.append(tt)
    return av, peers, (float(np.mean(hdv)) if hdv else float("nan")), bad

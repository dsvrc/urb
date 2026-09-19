"""Each day's executed routes and realized travel times, out of RouteRL.

THIS IS PLUMBING, NOT METHOD. Nothing here knows about any baseline: it turns
RouteRL's per-episode output into ``{agent_id: action}`` and
``{agent_id: (action, travel_time)}``, and it is used by the three baselines that
are defined on realized outcomes rather than on the observation alone:

  * ``eso``   (B10) the per-agent disturbance observer, whose input is the
              agent's own realized-minus-free-flow residual;
  * ``urls``  (B10) the unstructured recursive-least-squares arm, which regresses
              that residual on the raw peer actions;
  * ``mfq``   (B4)  mean-field Q-learning, whose mean action must include the
              human travellers -- and only the records know what a human did.

It is deliberately a SEPARATE implementation from ``pact1/records.py`` even
though the two solve the same RouteRL problem. The baselines have to be free-
standing: "we compared against baselines that import the method's own package"
is not a sentence this paper should have to write. Neither file contains method
logic, so the duplication costs nothing but the lines.

TWO SOURCES, CSV PREFERRED AND THEN LOCKED
--------------------------------------------------------------------------------
1. ``<records>/episodes/ep<N>.csv``, the interface ``analysis/metrics.py`` itself
   reads: columns ``id, action, travel_time, origin, destination, start_time,
   reward``. Each file states which episode it describes, so a multi-day flush
   replays in order and never pairs one day's travel time with another day's
   actions.
2. an in-memory list on the environment, for RouteRL versions that expose one.

The source is locked on the first success. Serving a day from memory and then
serving it again when its file appears would feed every consumer the same rows
twice, which reads as a suspiciously confident estimate and nothing else.
"""

import os

import numpy as np

__all__ = ["RecordSource", "split_records", "flatten_free_flow"]

# Candidate key names in priority order. The first one a record actually has
# wins, so RouteRL's in-memory dicts (Keychain constants) and the CSV rows
# (plain column names) both parse through one code path. They happen to agree
# today -- Keychain.AGENT_ID == "id" -- which is exactly why this must be a list
# rather than a hard-coded string: the day they diverge, a silent KeyError-free
# miss would drop every traveller.
DEFAULT_KEYS = {
    "id": ["id", "agent_id"],
    "action": ["action"],
    "travel_time": ["travel_time"],
    "origin": ["origin", "origins"],
    "destination": ["destination", "destinations"],
}


def aid(v):
    """Normalise an agent id so ``5``, ``5.0`` and ``'5'`` all key the same dict."""
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return str(v)


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

    def __init__(self, env, records_folder, tag="URB-BL"):
        self.env = env
        self.tag = tag
        self.episodes_dir = os.path.join(records_folder, "episodes")
        self.cursor = 0
        self.consumed = set()
        self.mode = None

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
                    if len(lst) < self.cursor:      # reset under us: rewind
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
            if not (fn.startswith("ep") and fn.endswith(".csv")):
                continue
            if fn in self.consumed:
                continue
            try:
                ep_no = int(fn[2:-4])
            except ValueError:
                continue
            try:
                df = pd.read_csv(os.path.join(self.episodes_dir, fn))
            except Exception:                              # noqa: BLE001
                continue                    # still being written; retry tomorrow
            self.consumed.add(fn)
            if df.empty or not any(c in df.columns for c in DEFAULT_KEYS["id"]):
                continue
            found.append((ep_no, df.to_dict("records")))
        found.sort(key=lambda t: t[0])
        return found, ("episodes/ep*.csv" if found else None)

    # ---------------------------------------------------------------- api
    def drain(self, episode):
        """-> list of ``(episode_label, records)``, oldest first."""
        if self.mode == "memory":
            recs, _ = self._drain_memory()
            return [(episode, recs)] if recs else []
        if self.mode == "csv":
            return self._drain_csv()[0]

        batches, tag = self._drain_csv()
        if batches:
            self.mode = "csv"
            print(f"[{self.tag}] travel-time records: {tag} "
                  "(episode-labelled, authoritative)", flush=True)
            return batches

        recs, tag = self._drain_memory()
        if recs:
            self.mode = "memory"
            print(f"[{self.tag}] travel-time records: {tag} "
                  "(no episode CSVs found; labelling by loop episode)", flush=True)
            return [(episode, recs)]
        return []

    def diagnose(self):
        out = [f"[{self.tag}] record-source diagnosis:"]
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


def split_records(records, machine_ids, keys=None):
    """-> ``(av, peers, tt_hdv, n_bad)``.

    ``av``    ``{id: (action, travel_time)}`` for machine agents that completed.
    ``peers`` ``{id: action}`` for EVERY traveller that completed, humans included.
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


def flatten_free_flow(env_ffts, invalid_pad=1e9):
    """``{(o, d): [t_0 .. t_{K-1}]}`` -> the same, cleaned.

    RouteRL pads masked route slots with ``invalid_pad`` (1e9). A residual
    computed against 1e9 would be 1e9 and would blow up any estimator that saw
    it, so padded slots become NaN here and every consumer has to decide
    explicitly what to do with a route that does not exist.
    """
    out = {}
    for od, vals in dict(env_ffts).items():
        arr = np.asarray(vals, dtype=np.float64).reshape(-1).copy()
        arr[~np.isfinite(arr)] = np.nan
        arr[arr >= invalid_pad * 0.5] = np.nan
        key = (int(od[0]), int(od[1])) if isinstance(od, (tuple, list)) else od
        out[key] = arr
    return out

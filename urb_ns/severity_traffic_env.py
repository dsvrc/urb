"""The severity dial as a TRUE environment wrapper -- so EVERY algorithm gets it.

`NS_FORM_SPEC` B.5 asks for the dial to sit BELOW the method in the class
hierarchy:

    TrafficEnvironment
      +-- SeverityTrafficEnvironment    <- MAPPO / QMIX / VDN / IPPO / IQL / AON
            +-- PACT                    <- adds the compensator only

`SeverityLayer` (severity_env.py) is a helper the *script* drives, which works for
loop-based hosts but leaves the TorchRL family out -- their rewards flow through a
collector, not through a Python loop anyone controls. That asymmetry would be
fatal for the comparison:

> A dial only the method's arm experienced is worthless as evidence.

So the dial is applied HERE instead, at the one place every host reads a reward.
Every URB script -- `qmix_torchrl.py`, `mappo_torchrl.py`, `ippo.py`, whatever --
gets the identical NS with no change to its own code, because TorchRL's
`PettingZooWrapper` is constructed with `use_mask=True` (the AEC path) and
therefore reads rewards through `last()` exactly as the hand-written loops do.

HOW THE DAY IS RESOLVED
--------------------------------------------------------------------------------
Route choices arrive one agent at a time via `step()`. Rewards arrive later, once
every agent has acted. So the day is finalised LAZILY, on the first `last()` that
reports termination: at that moment the whole day's choices are known, the loading
and the harm are computed once, and every subsequent reward that day is served
from the same latched multipliers.

FAILING LOUDLY
--------------------------------------------------------------------------------
If the layer cannot be built -- no route table, no machine agents yet -- rewards
pass through UNHARMED and the wrapper counts it. `severity_report()` prints those
counts at the end. A silently inert dial is the one failure that would look like a
clean null result, so it is made impossible to miss.
"""

import os

import numpy as np

_BANNER = False


def make_severity_env(base_cls, ns_cfg, sigma, routes_csv=None, verbose=True):
    """Return a subclass of `base_cls` (RouteRL's TrafficEnvironment) with the dial.

    `sigma = 0` short-circuits everything: the subclass is returned but every
    reward passes through untouched, so the stock task is recovered byte for byte
    and the same command produces the no-severity row.
    """

    class SeverityTrafficEnvironment(base_cls):
        _ns_sigma = float(sigma)
        _ns_cfg = dict(ns_cfg or {})
        _ns_routes = routes_csv

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._ns_day = 0
            self._ns_layer = None
            self._ns_slot = {}
            self._ns_choice = None
            self._ns_final = False
            self._ns_tried = False
            self._ns_n_harmed = 0
            self._ns_n_passthrough = 0
            self._ns_n_days = 0
            self._ns_banner()

        # -------------------------------------------------------------- banner
        def _ns_banner(self):
            global _BANNER
            if _BANNER or not verbose:
                return
            _BANNER = True
            print("\n" + "=" * 74)
            print(f"[URB-NS] SEVERITY WRAPPER ACTIVE  sigma={self._ns_sigma}")
            print("=" * 74)
            if self._ns_sigma == 0:
                print("  sigma = 0 -> the dial is OFF and every reward passes")
                print("  through untouched. This is the stock task, byte for byte.")
            else:
                print("  Rain derates road capacity (HCM capacity adjustment")
                print("  factor). Capacity is the DENOMINATOR of v/c, so it")
                print("  multiplies every peer's contribution and a lone")
                print("  traveller's cross-agent term stays exactly zero.")
                print("  Half the weather cycle is dry: the dial is provably")
                print("  inert there, for every sigma (the placebo regime).")
            print("  This wrapper sits BELOW the method, so every algorithm")
            print("  faces the identical dial (NS_FORM_SPEC B.5).")
            print("=" * 74 + "\n", flush=True)

        # -------------------------------------------------------------- lazy build
        def _ns_build(self):
            """Build the loading model. Only possible after `mutation()`, since it
            needs the machine-agent set and the generated route table."""
            if self._ns_tried:
                return self._ns_layer is not None
            self._ns_tried = True
            try:
                import pandas as pd

                from urb_ns.driver import HCM_HEAVY_RAIN_LOSS, WeatherDriver
                from urb_ns.network import (
                    RoadNetwork, load_route_table, parse_sumo_net,
                )
                from urb_ns.severity_env import SeverityLayer

                machines = list(getattr(self, "machine_agents", []) or [])
                if not machines:
                    return False
                rc = self._ns_routes or self._ns_cfg.get("routes_csv")
                if not rc or not os.path.exists(rc):
                    raise FileNotFoundError(
                        f"route table not found ({rc!r}); pass --routes")
                net_xml = self._ns_cfg["net_xml"]
                n_paths = int(self._ns_cfg["number_of_paths"])

                routes, ffts = load_route_table(rc, n_paths)
                net = RoadNetwork(parse_sumo_net(net_xml), routes, ffts, n_paths,
                                  sat_flow=float(self._ns_cfg.get("sat_flow", 600.0)),
                                  verbose=False)

                ordered = sorted(self.all_agents, key=lambda x: int(x.id))
                self._ns_slot = {str(int(x.id)): i for i, x in enumerate(ordered)}
                mids = {str(int(x.id)) for x in machines}
                tbl = pd.DataFrame({
                    "id": [int(x.id) for x in ordered],
                    "origin": [int(x.origin) for x in ordered],
                    "destination": [int(x.destination) for x in ordered],
                    "start_time": [float(x.start_time) for x in ordered]})
                from ns_certify import build_loading
                lm = build_loading(net, tbl, ffts, 0.0, 0)
                lm.is_machine = np.array(
                    [str(int(x.id)) in mids for x in ordered], dtype=bool)

                drv = WeatherDriver(
                    int(self._ns_cfg.get("period", 100)),
                    float(self._ns_cfg.get("wet_frac", 0.5)),
                    float(self._ns_cfg.get("loss", HCM_HEAVY_RAIN_LOSS)),
                    bool(self._ns_cfg.get("mean_preserving", False)))
                self._ns_layer = SeverityLayer(
                    net, drv, lm, self._ns_sigma,
                    av_behavior=self._ns_cfg.get("av_behavior", "selfish"),
                    max_offset_s=0.0, verbose=verbose)
                self._ns_choice = np.zeros(lm.N, dtype=np.int64)
                self._ns_layer.begin_episode(self._ns_day)
                return True
            except Exception as exc:                       # noqa: BLE001
                print(f"\n[URB-NS][WARN] severity layer could not be built: "
                      f"{type(exc).__name__}: {exc}\n"
                      f"[URB-NS][WARN] rewards will pass through UNHARMED. "
                      f"Do NOT report this run as a severity arm.\n", flush=True)
                return False

        # -------------------------------------------------------------- hooks
        def reset(self, *a, **kw):
            out = super().reset(*a, **kw)
            if self._ns_sigma != 0 and self._ns_layer is not None:
                self._ns_final = False
                if self._ns_choice is not None:
                    self._ns_choice[:] = 0
                self._ns_layer.begin_episode(self._ns_day)
            return out

        def step(self, action=None, *a, **kw):
            # record WHICH route this agent chose, before the env consumes it
            if (self._ns_sigma != 0 and action is not None
                    and self._ns_choice is not None):
                aid = getattr(self, "agent_selection", None)
                slot = self._ns_slot.get(str(aid))
                if slot is not None:
                    try:
                        self._ns_choice[slot] = int(action)
                    except (TypeError, ValueError):
                        pass
            return super().step(action, *a, **kw)

        def last(self, *a, **kw):
            obs, reward, term, trunc, info = super().last(*a, **kw)
            if self._ns_sigma == 0:
                return obs, reward, term, trunc, info
            if self._ns_layer is None and not self._ns_build():
                self._ns_n_passthrough += 1
                return obs, reward, term, trunc, info
            if not (term or trunc):
                return obs, reward, term, trunc, info

            if not self._ns_final:
                # every agent has acted by now, so the day can be resolved once
                self._ns_layer.end_episode(self._ns_choice)
                self._ns_harm_records()
                self._ns_final = True
                self._ns_day += 1
                self._ns_n_days += 1

            slot = self._ns_slot.get(str(getattr(self, "agent_selection", None)), -1)
            harmed = self._ns_layer.harm_reward(slot, reward)
            self._ns_n_harmed += 1
            return obs, harmed, term, trunc, info

        # -------------------------------------------------------------- records
        def _ns_harm_records(self):
            """Scale this day's travel-time records in place, before RouteRL
            writes them out.

            Without this the harm reaches the REWARD but not the RECORD, and the
            two diverge in a way that quietly breaks three things at once:

              * `analysis/metrics.py` computes `t_CAV` from the episode CSVs, so
                the headline metric would report SUMO's unharmed time. Measured:
                AON scored identically at sigma 0 and sigma 3 to six decimals --
                a non-learning agent under a live dial, which is impossible unless
                the dial never reached the metric.
              * any method that learns from recorded travel times (PACT-1's
                estimator does) would train on a world its policy is not being
                paid for.
              * the episode CSVs are what every downstream analysis reads.

            Scaling here keeps reward, record and metric describing the SAME trip.
            """
            recs = None
            for attr in ("travel_times_list", "last_episode_travel_times"):
                v = getattr(self, attr, None)
                if isinstance(v, list) and v:
                    recs = v
                    break
            if recs is None:
                self._ns_n_rec_missing = getattr(self, "_ns_n_rec_missing", 0) + 1
                return
            n = 0
            for rec in reversed(recs):                 # this day's are at the end
                if not isinstance(rec, dict):
                    continue
                rid = rec.get("id", rec.get("agent_id"))
                if rid is None:
                    continue
                slot = self._ns_slot.get(str(int(rid))) if str(rid).lstrip(
                    "-").isdigit() else None
                if slot is None:
                    continue
                for key in ("travel_time", "reward"):
                    if key in rec:
                        try:
                            rec[key] = self._ns_layer.harm_travel_time(
                                slot, float(rec[key])) if key == "travel_time" \
                                else self._ns_layer.harm_reward(slot,
                                                                float(rec[key]))
                        except (TypeError, ValueError):
                            pass
                n += 1
                if n >= len(self._ns_slot):            # one day's worth
                    break
            self._ns_n_rec_harmed = getattr(self, "_ns_n_rec_harmed", 0) + n

        # -------------------------------------------------------------- report
        def severity_report(self):
            """Print what the dial actually did. §II.7: a silently inert
            disturbance is invisible in exactly the arm you most need to trust."""
            print("\n" + "=" * 74)
            print(f"[URB-NS] SEVERITY REPORT  sigma={self._ns_sigma}")
            print(f"  days harmed            {self._ns_n_days}")
            print(f"  rewards harmed         {self._ns_n_harmed}")
            print(f"  records harmed         {getattr(self, '_ns_n_rec_harmed', 0)}")
            print(f"  days with no records   "
                  f"{getattr(self, '_ns_n_rec_missing', 0)}")
            print(f"  rewards passed through {self._ns_n_passthrough}")
            if self._ns_sigma != 0 and getattr(self, "_ns_n_rec_harmed", 0) == 0:
                print("  *** RECORDS were never harmed: metrics.py will report")
                print("      SUMO's UNHARMED travel time. Do not compare t_CAV")
                print("      across sigma from this run. ***")
            if self._ns_sigma != 0 and self._ns_n_harmed == 0:
                print("  *** THE DIAL NEVER FIRED. This run is NOT a severity")
                print("      arm -- do not report it as one. ***")
            elif self._ns_layer is not None:
                d = self._ns_layer.diagnostics()
                print(f"  last day  A={d['A']:.3f} g={d['g_mean']:.4f} "
                      f"dry={d['dry']} m_mean={d['m_mean']:.5f} "
                      f"u_mean={d['u_mean']:.4f}")
            print("=" * 74 + "\n", flush=True)

    SeverityTrafficEnvironment.__name__ = "SeverityTrafficEnvironment"
    return SeverityTrafficEnvironment

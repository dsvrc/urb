"""The SeverityLayer -- the NS dial, applied to EVERY arm.

`NS_FORM_SPEC` B.5 is explicit about where this sits:

    TrafficEnvironment
      +-- SeverityLayer        <- MAPPO / IPPO / QMIX / any baseline use this
            +-- PACTLayer      <- adds the compensator only

and that `severity` is read from the TASK config, never from the method's block:

> A dial only the method's arm experienced is worthless as evidence.

WHAT IT DOES, PER DAY
--------------------------------------------------------------------------------
    1. advance the day counter (it PERSISTS across episodes -- the weather does
       not reset when the day does)
    2. A(day) -> per-link capacity multiplier g, HCM-anchored
    3. from the executed route choices, compute each traveller's binding v/c
    4. the harm: travel time is multiplied by BPR(u under g) / BPR(u at nominal)

The multiplier is **exactly 1.0** when `g == 1` -- at `sigma = 0` and on every dry
day -- so the stock task is recovered byte for byte and the placebo regime is free.

WHERE THE HARM IS APPLIED, AND WHY THAT IS HONEST
--------------------------------------------------------------------------------
SUMO does not model rain, so the delay is computed in this layer from the declared
loading model rather than inside the simulator. That is the same arrangement Ant
uses -- the torque disturbance lives in the environment wrapper, not in MuJoCo --
and it keeps the reward FUNCTION untouched: the agent is still paid its own
realized travel time, that travel time is simply longer because the road delivered
less. No penalty term is ever subtracted (I.2.3 / A.1).

Scope, stated plainly: with `should_humans_adapt: false` (URB's config1 and
config2) the human population is frozen, so applying the harm to human rewards
would have no behavioural effect. Human travel times are still reported with the
harm so system-level metrics are consistent.

REWARD MAPPING
--------------------------------------------------------------------------------
Verified against URB's own episode records: under `av_behavior: selfish` the
reward is exactly `-travel_time`. So the harmed reward is `reward * m_i`. Other
behaviours mix own/group/human travel times, and the layer refuses to guess --
`--task-conf` must use a behaviour whose mapping is declared here.
"""

import numpy as np

# Reward = -travel_time exactly. Verified from URB episode CSVs under this
# behaviour; any other behaviour mixes terms and the mapping must be derived
# before it can be used.
_LINEAR_REWARD_BEHAVIOURS = {"selfish"}


class SeverityLayer(object):
    """The dial. Construct once after mutation; call `begin_episode` each day."""

    def __init__(self, network, driver, loading, sigma, av_behavior="selfish",
                 max_offset_s=0.0, verbose=True):
        self.net = network
        self.driver = driver
        self.loading = loading
        self.sigma = float(sigma)
        self.av_behavior = str(av_behavior).lower()
        self.max_offset_s = float(max_offset_s)

        if self.sigma > 0 and self.av_behavior not in _LINEAR_REWARD_BEHAVIOURS:
            raise ValueError(
                f"[URB-NS] av_behavior={self.av_behavior!r}: the harm is applied to "
                "realized travel time, and only a behaviour whose reward is a known "
                "function of travel time can be harmed correctly. Declared: "
                f"{sorted(_LINEAR_REWARD_BEHAVIOURS)}. Refusing to guess."
            )

        self.day = 0                      # PERSISTS across episodes
        self._g = None
        self._m = np.ones(loading.N)
        self._u_g = np.zeros(loading.N)
        self._u_0 = np.zeros(loading.N)
        self._binding = np.zeros(loading.N, dtype=np.int64)
        self._A = 0.0
        if verbose:
            self.banner()

    # ------------------------------------------------------------------ banner
    def banner(self):
        d = self.driver
        print("\n" + "=" * 74)
        print("[URB-NS] SEVERITY LAYER -- the dial, applied to EVERY arm")
        print("=" * 74)
        print(f"  sigma          {self.sigma}"
              + ("   (sigma=1 = HCM heavy-rain capacity adjustment factor)"
                 if self.sigma <= 1.0 else
                 "   *** BEYOND-PHYSICAL STRESS TEST (sigma > 1) ***"))
        print(f"  driver         rain, period {d.period} d, wet {d.wet_frac:.0%} "
              f"of the cycle, loss@sigma=1 {d.loss:.3f}")
        print(f"  placebo        {int(d.is_dry(np.arange(d.period)).sum())}/"
              f"{d.period} days are exactly dry (g == 1.0000 there, all sigma)")
        print(f"  medium         v/c on the BINDING link; capacity = lanes x "
              f"{self.net.sat_flow:.0f} veh/h")
        print(f"  harm           travel time x BPR(u/g)/BPR(u), "
              f"alpha={self.loading.bpr_alpha:.3f} beta={self.loading.bpr_beta:.2f}")
        print(f"  compensation   departure offset, +/- {self.max_offset_s:.0f} s"
              + ("  (DISABLED -- blind arm)" if self.max_offset_s <= 0 else ""))
        print(f"  reward         {self.av_behavior}: reward = -travel_time, so the "
              "harm multiplies it")
        print("=" * 74 + "\n", flush=True)

    # ------------------------------------------------------------------ per day
    def begin_episode(self, day=None):
        """Advance to the next day and resolve the driver. Weather does not reset
        with the episode -- it is exogenous and persists (A.2.3)."""
        if day is not None:
            self.day = int(day)
        self._A = float(self.driver.A(self.day))
        self._g = np.asarray(
            self.driver.g(self.day, self.sigma, sens=self.net.rain_sens),
            dtype=np.float64)
        return self._A

    def end_episode(self, choice, offsets=None):
        """Resolve the day's harm from the EXECUTED choices.

        Returns `m` (N,) travel-time multipliers, exactly 1.0 wherever the dial is
        off. `offsets` is the compensation channel; `None` means every traveller
        departed as scheduled, which is the blind arm.
        """
        choice = np.asarray(choice, dtype=np.int64)
        off = None if offsets is None else np.asarray(offsets, dtype=np.float64)
        if off is not None and self.max_offset_s > 0:
            off = np.clip(off, -self.max_offset_s, self.max_offset_s)
        elif off is not None:
            off = np.zeros_like(off)

        m, u_g, binding = self.loading.delay_multiplier(
            choice, g=self._g, offsets=off)
        u_0, _ = self.loading.loading(choice, g=1.0, offsets=off)
        self._m, self._u_g, self._u_0, self._binding = m, u_g, u_0, binding
        self.day += 1
        return m

    # ------------------------------------------------------------------ apply
    def harm_reward(self, slot, reward):
        """Apply the day's harm to one traveller's reward.

        Under `selfish`, reward == -travel_time, so a travel time multiplied by
        `m` is a reward multiplied by `m`. At `m == 1` this returns the reward
        UNCHANGED, bit for bit -- the byte-identity B.1.1 asks for.
        """
        m = float(self._m[slot]) if 0 <= slot < len(self._m) else 1.0
        if m == 1.0:
            return reward
        return float(reward) * m

    def harm_travel_time(self, slot, tt):
        m = float(self._m[slot]) if 0 <= slot < len(self._m) else 1.0
        return float(tt) * m

    # ------------------------------------------------------------------ state
    @property
    def g(self):
        return self._g

    @property
    def A(self):
        return self._A

    def is_dry(self):
        return bool(self.driver.is_dry(self.day))

    def excess(self):
        """`u * (1 - g_binding)` -- the loading excess of C.1, per traveller. This
        is the quantity PACT's analytic feedforward cancels, and it is KNOWN from
        observable time plus the declared model, with no estimation at all."""
        gb = self._g[self._binding] if np.ndim(self._g) else float(self._g)
        return self._u_g * (1.0 - np.asarray(gb, dtype=np.float64))

    def diagnostics(self):
        av = self.loading.is_machine
        return {
            "day": int(self.day), "A": float(self._A),
            "g_min": float(np.min(self._g)) if self._g is not None else 1.0,
            "g_mean": float(np.mean(self._g)) if self._g is not None else 1.0,
            "dry": int(self.is_dry()),
            "m_mean": float(np.mean(self._m[av])),
            "m_max": float(np.max(self._m[av])),
            "u_mean": float(np.mean(self._u_g[av])),
            "u_max": float(np.max(self._u_g[av])),
            "excess_mean": float(np.mean(self.excess()[av])),
        }

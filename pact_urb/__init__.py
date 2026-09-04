"""PACT for the URB NS -- `PACT_PIPELINE_SPEC.md`, instantiated on urban routing.

    core.py         RLS with the windup bound, the null tracker, the analytic
                    feedforward, the pedestal EMA, the admissibility gate.
                    Pure numpy; `selftest.py` exercises all of it without SUMO.
    coordinator.py  the per-day identify -> compensate cycle and every diagnostic.
    selftest.py     the offline arithmetic gates §11 step 4 requires.

The compensation channel is the DEPARTURE OFFSET. It is continuous, additive,
exactly invertible in the linearised sense, and loop-coupled -- shifting when you
travel changes who you share the road with, so compensating feeds the medium it
compensates against. That is A.6, and it is why T4 applies here.
"""

from pact_urb.core import (
    AgentRLS,
    NullTracker,
    PedestalEMA,
    analytic_feedforward,
    trust_gate,
)

__all__ = ["AgentRLS", "NullTracker", "PedestalEMA", "analytic_feedforward",
           "trust_gate"]

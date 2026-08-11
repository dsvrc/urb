"""PACT-1 for URB — peer-action compensation with an online-identified coupling.

See ``pact1/README.md`` for the full description, the run command, and the meaning
of every diagnostic column and gate.

Layering (deliberate — it is what makes the method testable without SUMO):

    core.py        pure numpy. RLS, trust, confidence, steering. No torch, no
                   routerl, no SUMO. ``selftest.py`` exercises all of it offline.
    basis.py       the KNOWN model class: SUMO network geometry -> road-class
                   partition -> route/route overlap Gram matrices. Built once,
                   from files that exist before any episode runs.
    coordinator.py the per-day identify -> predict -> steer cycle, the diagnostic
                   CSV, and every gate.
    policy.py      URB's PPO + one trust head. The PPO update itself is unchanged.

Nothing in ``core`` or ``basis`` imports ``routerl``; the routerl-specific record
parsing lives in ``scripts/pact1.py`` and hands this package plain arrays.
"""

from pact1.core import (
    AgentRLS,
    predict_excess,
    relative_excess,
    rls_confidence,
    steer_logits,
    trust_from_w,
)

__all__ = [
    "AgentRLS",
    "predict_excess",
    "relative_excess",
    "rls_confidence",
    "steer_logits",
    "trust_from_w",
]

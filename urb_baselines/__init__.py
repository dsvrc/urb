"""URB baselines -- one published method per class, each its own algorithm.

WHAT THIS PACKAGE IS
--------------------------------------------------------------------------------
``paper/BASELINES.md`` picks one representative per *class* of method that a
reviewer would say answers the non-stationarity / adaptation / robustness /
structure objection. This package implements the ones whose host is URB.

Every baseline is a SEPARATE ALGORITHM with its own ``scripts/<name>.py`` entry
point and its own ``config/algo_config/<name>/`` block, exactly like
``scripts/pact1.py`` -- never a flag on PACT-1, and never a flag on another
baseline. What they DO share is the host: the same config loading, the same
``TrafficEnvironment`` construction, the same human-learning phase, the same
mutation, the same day loop, the same test phase and the same metrics call, all
in ``urb_baselines.host``. That is deliberate:

> If two arms differ, the difference must be the algorithm. Sharing the loop is
> the only way to make that a fact rather than a hope.

``host.py`` is verified against ``scripts/ippo.py`` by ``selftest.py`` gate H1:
the reference algorithm ``urb_baselines.algos.reference.IPPO`` is URB's own PPO
class driven through the shared loop, so any drift in the host shows up as a
parity failure rather than as a mysterious baseline result.

WHICH BASE EACH BASELINE IS BUILT ON
--------------------------------------------------------------------------------
On-policy methods are built on URB's on-policy base (the actor-only single-step
PPO of ``scripts/ippo.py``); off-policy methods are built on URB's off-policy
base (the single-step DQN of ``scripts/iql.py``). The table is in
``docs/baselines/README.md`` and is repeated in each algorithm's checklist.

TWO PROPERTIES OF URB THAT EVERY ADAPTATION TURNS ON
--------------------------------------------------------------------------------
1. **The episode is one step.** A day is: every traveller picks one of K routes,
   the day is simulated, everyone is paid once. There is no within-episode
   transition, so ``gamma`` and ``lambda`` have nothing to discount and
   bootstrapping has nothing to bootstrap from. URB's own IQL sets
   ``target = reward`` for exactly this reason, and every off-policy baseline
   here does the same, so the comparison is against the host's own convention
   rather than against a different problem.

2. **The day is sequential, not simultaneous.** Travellers act in start-time
   order and each one's observation counts the routes taken by EARLIER same-OD
   travellers that same day. So there is no instant at which all agents' current
   observations exist at once. Any method defined on a simultaneous joint
   observation (DGN's feature matrix, LIAM's modelled-agent targets, MF-Q's mean
   action) has to say which snapshot it uses; each one says so in its checklist
   and the choice is implemented in ``urb_baselines.graph`` /
   ``urb_baselines.records`` rather than inside the method.

MODULES
--------------------------------------------------------------------------------
    host.py            the shared loop + ``BaselineAlgorithm`` hook protocol
    nets.py            the host MLP (identical to ``scripts/iql.py``'s Network)
    context.py         the exogenous driver A(day) -- observed context / privileged z
    records.py         RouteRL's per-day travel-time + executed-action records
    graph.py           neighbour structures: same-OD, co-presence, route overlap
    domain_random.py   per-episode sigma resampling, attached to the env instance
    selftest.py        offline gates; no SUMO, no routerl, no torch training
    algos/             one module per baseline
"""

__all__ = ["host", "nets", "context", "records", "graph", "domain_random"]

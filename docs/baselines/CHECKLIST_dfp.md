# CHECKLIST — Deep Fictitious Play, on URB

**BASELINES.md** — a **new class**. None of the twelve original arms is
equilibrium-seeking: every one of them is a learner improving its own return, and
none treats the fleet as a population whose *distribution* is the object being
solved for. The objection that leaves open is the one a traffic reviewer reaches
for first:

> *"your forecast just walks the fleet toward equilibrium — fictitious play does
> that with no estimator, no declared basis and no trust head."*

**Paper** *Solving Continuous Mean Field Games: Deep Reinforcement Learning for
Non-Stationary Dynamics*, NeurIPS 2025, <https://arxiv.org/abs/2510.22158> ·
**related** Heinrich & Silver, *Deep Reinforcement Learning from Self-Play in
Imperfect-Information Games* (NFSP), whose reservoir and anticipatory parameter
the average policy uses.

**Implementation** `urb_baselines/algos/dfp.py` · **script** `scripts/dfp.py` ·
**config** `config/algo_config/dfp/config1.json`

---

## A. The specification, item by item

| id | item | status | where / why |
|---|---|---|---|
| **FP-1** | Best response computed by **DRL** against the current population | **IMPLEMENTED** | `SingleStepPPO` — URB's own learner (R2) — whose input is `[observation, population mix]` |
| **FP-2** | Average policy represented by **supervised learning** | **IMPLEMENTED** | `_AveragePolicy`: cross-entropy onto the BR's own action, over a reservoir |
| **FP-3** | Reservoir, so the fit is an average over history rather than a copy of the latest BR | **IMPLEMENTED** | `_AveragePolicy.add()` is reservoir sampling with capacity `avg_capacity = 2000` |
| **FP-4** | Fictitious-play averaging of the population distribution | **IMPLEMENTED** | `μ̄ ← (1 − w) μ̄ + w μ̂` with `w = 1/(n+1)` in FP **iterations**. Gate **DFP-1** checks the result equals the empirical mean of the daily mixes to 2e-4 |
| **FP-5** | Time-dependent population distribution | **IMPLEMENTED** | `μ̄` is updated every FP iteration and is an input to the BR, so it is time-dependent by construction |
| **FP-6** | Conditional Normalizing Flow for that distribution | **N/A** | the flow exists because the paper's state space is continuous, so the population distribution has no finite parameterisation. URB's is a point on the K-simplex and is represented **exactly** by K floats. A flow here would be a density model of a categorical |
| **FP-7** | The **average** is the solution concept | **IMPLEMENTED** | the test phase plays `_AveragePolicy` deterministically |

---

## B. What URB forced

### PLAY-1 — the best response acts during training, the average acts at test
FP's inner loop computes a best response *to* the averaged population, and the
fleet that produces the next empirical mix is the one playing that best response
— averaging best responses is what fictitious play **is**. So the BR acts while
training and the average policy is fitted alongside it.

This also keeps the host's PPO honest. If a separately-trained average policy
chose the action while the BR's log-probability was stored for it, the PPO ratio
would be an uncorrected off-policy one. `play: avg` puts the average policy in
charge during training as well (the NFSP-style mixture at rate `br_frac = 0.1`),
and in that path the stored decision is rewritten to the action actually taken
and rescored under the BR.

### FIELD-1 — the population is the **index-level** mix over every traveller
Mean-field games assume exchangeable agents. §4 of [README.md](README.md)
measures that URB's OD pairs are singletons on six of seven cities, so a same-OD
population is one traveller and carries no field at all. `field_scope` therefore
defaults to `all` — the same decision, for the same reason, as
[CHECKLIST_mfq.md](CHECKLIST_mfq.md) FIELD-1. `field_scope: od` is implemented
(one mix per OD pair) and is expected to be uninformative on these cities.

Humans are included in the mix: the population of a mean-field game is the whole
fleet, not the controllable subset. `info.peer_acts` is what carries that, which
is why the arm sets `needs_records`.

### ITER-1 — an FP iteration is `fp_every = 10` days
The paper's outer loop is the FP iteration and its inner loop trains a best
response against a frozen population. Here one day is one inner step, so the
`1/(n+1)` weight is in FP iterations, not in days.

### SCALE-1 — the observation must be standardised or the mean field is invisible
**Measured, not assumed.** URB's observation leads with a start time in seconds
(order 1000) and the population mix is a probability vector (order 0.25). Fed to
the same input layer, the mix is below the network's resolution: on
`fakeenv.FakeURB` the arm produced returns **identical to four significant
figures** with and without the mix, and identical to a plain IPPO with four dead
input coordinates.

`normalize_obs: true` standardises the observation (causal running mean/sd,
`nets.RunningNorm`) before the mix is concatenated. This is the same device
HAPPO needed for the same coordinate — see the `use_feature_normalization` row
in [CHECKLIST_happo.md](CHECKLIST_happo.md) — and §4 of
[README.md](README.md). Setting it false reproduces the degenerate behaviour.

The diagnostic that catches it is **`mix_infl`**: the mean absolute change in the
BR's action probabilities when the mix coordinate is replaced by a uniform one.
Zero means the population input is decoration.

---

## C. Arms

| arm | what it is |
|---|---|
| `fp` (default) | the paper's fictitious play, `1/(n+1)` averaging |
| `ema` | exponential averaging of the population mix with `rho = 0.99` — the adaptive variant, because a `1/n` average freezes under drift |
| `br` | best response only, no averaging at all: what the averaging buys |

---

## D. The gates

| gate | what it asserts |
|---|---|
| **DFP-1** | 60 FP iterations reproduce the empirical mean of the daily mixes to `2e-4`, and `μ̄` stays a distribution |
| **drive dfp** | 200 days on `fakeenv.FakeURB`: learns (−695 → −585, against IPPO's −701 → −690 on the same seed), completes FP iterations, updates the mix |
| **drive dfp --arm br** | the no-averaging ablation runs |
| **CFG-1 / CFG-2** | host block character-identical to `ippo/config1.json`; constructs and acts under its own shipped config |

---

## E. How to read a run

* `fp_iter`, `mix_dev` — FP iterations completed and how far the averaged mix is
  from uniform. **`***` if the mix was never updated** (no peer actions arrived)
  or if it is uniform (which on a singleton-OD city can be true of the data —
  say which).
* `mix_infl` — **`***` if ~0**: the population input does not move the best
  response, so the arm is an IPPO with four dead coordinates.
* `br_diff` — how often the average policy and the best response disagree.
  **`***` if identically zero**, and note that a falling `br_diff` over time is
  the closest thing here to an exploitability curve.

---

## F. What it tests

`PACT_THEORY` **B1** makes the relationship exact: the steered fleet is a
z-normalised logit QRE with effective rationality `gκ/sd(c)`, and fictitious play
with a hard best response is the same object at `g → ∞`. So FP converges to the
**selfish equilibrium**, and **B4**'s corollary applies to it unchanged: it can
close the learning gap and cannot touch the equilibrium gap.

If this arm reaches equilibrium faster than PACT-1 does, that is the honest
result and the paper reports it. What it cannot do is beat the equilibrium, and
that is the claim to check.

# CHECKLIST — ESO / DOB (linear ADRC) on URB

**References**
- Han, *From PID to Active Disturbance Rejection Control*, IEEE TIE 2009 — the
  extended state observer.
- Gao, *Scaling and bandwidth-parameterization based controller tuning*, ACC
  2003 — the `β₁ = 2ω₀`, `β₂ = ω₀²` observer parameterisation.
- Reference implementations: <https://pyadrc.readthedocs.io/en/latest/>,
  <https://github.com/MRGilak/Active-Disturbance-Rejection-Controller>,
  ADRC toolbox <https://arxiv.org/pdf/2112.01614>

**BASELINES.md** B10 · Tier 1 item 3 · objection row *"A disturbance observer
needs no structure"*. B10 calls this *"the single most important non-RL baseline
and it is cheap"*.

**Implementation** `urb_baselines/algos/eso.py` (+ the shared controller in
`urb_baselines/algos/compensator.py`) · **script** `scripts/eso.py` · **config**
`config/algo_config/eso/config1.json`

**Arms** `--arm eso1` (default) · `--arm eso2`

---

## A. What B10 specifies, item by item

B10's text is the specification:

> *"Per-agent extended-state / disturbance observer feedforward (ESO / DOB, i.e.
> linear ADRC): estimate the lumped disturbance from the agent's own residual
> with a first-order observer and cancel it next step. No peer information, no
> basis, one gain."*

| id | item | status | where / why |
|---|---|---|---|
| **SPEC-1** | **Per agent** | **IMPLEMENTED** | one extended state per machine agent (and per route — see ADAPT-1) |
| **SPEC-2** | The input is **the agent's own residual** | **IMPLEMENTED** | `excess = travel_time − freeflow[chosen route]`, from RouteRL's own per-day record, so the action fitted is the one the simulator executed |
| **SPEC-3** | **First-order observer** | **IMPLEMENTED** | `--arm eso1`: `z₁ ← z₁ + β(y − z₁)` |
| **SPEC-4** | **One gain** | **IMPLEMENTED** | `beta: 0.5`, and that is the only tunable in `eso1` |
| **SPEC-5** | **Cancel it next step** | **IMPLEMENTED** | the controller predicts `freeflow_k + ẑ_k` and drives the argmin |
| **SPEC-6** | **No peer information** | **IMPLEMENTED — and checkable** | the observer's only input is this traveller's own residual. It never reads `peer_acts`, a reward, or the driver. That is the whole point of running it beside `scripts/urls.py`, which reads nothing else *but* peer actions. |
| **SPEC-7** | **No basis** | **IMPLEMENTED** | no incidence matrix, no road classes, no declared operator |
| **SPEC-8** | It does not learn | **IMPLEMENTED** | there is no network, no optimiser and no reward anywhere in this arm |
| **ADRC-1** | Second-order linear ADRC: the ESO also tracks the disturbance's **rate**, so it extrapolates instead of lagging | **IMPLEMENTED** | `--arm eso2`: `z₁ ← z₁ + z₂ + β₁(y − z₁)`, `z₂ ← z₂ + β₂(y − z₁)`, prediction `z₁ + z₂` |
| **ADRC-2** | Gao's bandwidth parameterisation `β₁ = 2ω₀h`, `β₂ = (ω₀h)²` | **IMPLEMENTED** | `omega: 0.5`, `h = 1 day` |

---

## B. The two adaptations

### ADAPT-1 — one extended state per (agent, route), and why it has to be
The plant seen by traveller `i` on route `k` is

```
travel_time_{i,k}(t) = freeflow_{i,k} + excess_{i,k}(t)
```

with `freeflow` published exactly by the environment and `excess` the lumped
disturbance. There is no additive control channel: the only decision is **which
plant to drive**.

A single lumped disturbance per agent would be added to every route's free-flow
time **equally**, and could therefore not change the argmin. The arm would be
provably inert — free-flow shortest path with a constant offset — which is
exactly the failure mode this repository refuses to ship. The extended state is
therefore per route, updated only on the days that route is actually driven.

This does not weaken any of SPEC-1..8: it is still per agent, still driven by
that agent's own residual, still one gain, still no peers and no basis.

### ADAPT-2 — exploration comes from the host, because a control law has none
ADRC is a control law and has no exploration, which is fine for a continuous
plant that is always excited. On a discrete choice, the observer for a route that
is never driven is never updated, so a controller starting at the
free-flow-fastest route would stay there for four thousand days and the arm would
measure nothing.

Excitation is therefore supplied by the **host's own** ε-greedy schedule — the
`eps_init` / `eps_decay` / `eps_min` of `iql/config1.json`, the same three
numbers IQL, DGN and MF-Q use — and **nothing else** is borrowed from the
learner. The observer itself is untouched.

### The gains, and why they are declared rather than tuned
`beta = 0.5` is an observer time constant of about two days: far faster than the
100-day weather cycle and slower than daily noise. `omega = 0.5` is the `eso2`
bandwidth under Gao's parameterisation. Neither was tuned on returns; both are
declared in the config's `desc` and are one edit to sweep.

---

## C. Gates

| gate | what it proves |
|---|---|
| **drive eso --arm eso1**, **--arm eso2** | Both run, both improve, and the observer actually receives residuals (`n_obs_updates > 0`). An estimator that never received a row is the silent failure here, and `close()` prints a `***` block for it. |

---

## D. What to watch in a real run

- `|resid|` — the mean absolute realised excess delay, in seconds. This is the
  quantity the observer is tracking.
- `|err|` — the mean absolute **one-step-ahead prediction error**. This is what
  the arm is judged on as an estimator, and it is what `losses.csv` contains for
  this arm (there is no loss).
- `|z1|` — the mean absolute extended state.
- `states ever updated` in the closing report — how much of the `(agent, route)`
  table the exploration schedule actually reached.

**Prediction under test** (BASELINES.md B10): *"it cancels the slow component and
lags the fast one; on the feeder it cannot break the loop because its estimate is
one step behind the peers' reaction, whereas PACT's channels are computed from
the peers' broadcast actions before the step."* `eso2` is the sharper version of
the same test: it extrapolates the rate, so if the gap were purely lag, `eso2`
would close it.

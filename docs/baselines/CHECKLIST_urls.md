# CHECKLIST — unstructured RLS on raw peer actions, on URB

**BASELINES.md** B10 · objection rows *"A disturbance observer needs no
structure"* and *"Why not learn the coupling instead of declaring it"*

> *"Per-agent unstructured RLS on raw peer actions (regress the residual on the
> N−1 broadcast actions directly, no classes): isolates the value of the
> **declared basis** (r parameters vs N−1). Prediction: fails at N = 22/38
> (under-excited, ill-conditioned). ≈ 30 lines."*

**Implementation** `urb_baselines/algos/urls.py` (+ the shared controller in
`urb_baselines/algos/compensator.py`) · **script** `scripts/urls.py` · **config**
`config/algo_config/urls/config1.json`

This arm exists to isolate **one** thing. It is the same controller as
`scripts/eso.py` — the same `argmin_k [freeflow_k + excess_hat_k]`, the same host
ε-greedy — and the same estimator family PACT-1 uses (exponentially-weighted
recursive least squares, the same `forget` and `p0`). The **only** difference is
the regressor: raw peer actions instead of a declared road-class basis. That is
what makes the comparison mean what B10 says it means.

---

## A. What B10 specifies, item by item

| id | item | status | where / why |
|---|---|---|---|
| **SPEC-1** | Recursive least squares | **IMPLEMENTED** | `observe`: gain `g = Pφ/(λ + φᵀPφ)`, update `θ += g·e`, downdate `P ← (P − gφᵀP)/λ` |
| **SPEC-2** | Exponential forgetting | **IMPLEMENTED** | `forget: 0.999` — the same value `pact1/config1.json` declares, so the filter is not the difference |
| **SPEC-3** | `P₀ = p₀·I` | **IMPLEMENTED** | `p0: 10.0` — likewise the same as PACT-1's |
| **SPEC-4** | Per agent | **IMPLEMENTED** | one filter bank per machine agent |
| **SPEC-5** | The regressand is the agent's own residual | **IMPLEMENTED** | `excess = travel_time − freeflow[chosen route]`, from RouteRL's per-day record |
| **SPEC-6** | The regressor is the **raw N−1 peer actions** | **IMPLEMENTED** | one-hot of every broadcasting peer's action, plus an intercept |
| **SPEC-7** | **No classes** | **IMPLEMENTED** | no incidence matrix, no road classes, no declared operator, no network geometry of any kind |
| **SPEC-8** | The agent's **own** action is excluded from the regressor (it is the N−1 *others*) | **IMPLEMENTED** | `_mask_self` zeroes the agent's own block |
| **SPEC-9** | The parameter count is what is being measured | **IMPLEMENTED AND PRINTED** | the banner prints features, parameters per agent and the covariance memory; `close()` prints `cond(P)` and `trace(P)` |

---

## B. The three adaptations

### ADAPT-1 — one filter per (agent, own-route)
A single regression of the residual on the peers' actions has **no dependence on
the agent's own choice** and therefore cannot rank routes — the arm would be
inert. The agent's own action is a categorical with no order, so the honest
unstructured encoding is a separate parameter vector per own-route, fitted from
the days that route was driven.

This is exactly the "r parameters vs N−1" accounting B10 asks for, with `r`
replaced by `K × (K·n_peers + 1)`. On `saint_arnoult` with the default
`peer_scope: fleet` that is **4 × (4 × 87 + 1) = 1 396 parameters per agent**,
fitted from 4 000 daily scalars, only a quarter of which touch any given
own-route filter. The banner prints that number so the claim is a measurement,
not a sentence.

### ADAPT-2 — the regressor is the PREVIOUS day's peer actions
B10 says "the N−1 **broadcast** actions", because in the invertible cell the
peers publish their actions before the step. **Stock URB has no broadcast
channel.** The regressor is therefore the previous day's realised peer action
vector — persistence — which is the natural forecast and is what any arm on URB
that wants tomorrow's peer load has to do. It is a property of the host, not a
concession by this arm.

### ADAPT-3 — `peer_scope: fleet` by default
"Broadcast" names the set that publishes: on URB only the machine agents do. The
default is therefore the fleet (`n_av − 1` peers). `peer_scope: all` adds the
human travellers, which makes the regressor complete but squares the covariance
memory; `max_p_mb` refuses the run with an explanatory message rather than
thrashing — and the message notes that not scaling *is the arm's own point*.

Exploration comes from the host's ε-greedy schedule, for the same reason as the
ESO arm (see [CHECKLIST_eso.md](CHECKLIST_eso.md) ADAPT-2). The `P` update is
symmetrised after each downdate so float32 rounding cannot drift the covariance
out of the PSD cone over four thousand updates.

---

## C. Gates

| gate | what it proves |
|---|---|
| **drive urls** | The arm runs, improves, and RLS actually fits (`n_fits > 0`) — an estimator that never received a row is the silent failure. |

---

## D. What to watch in a real run

The closing report prints the numbers B10's prediction is about:

- **`cond(P)` for agent 0, route 0** — *"a condition number that never falls is
  the under-excitation B10 predicts"*. This is the headline quantity for this
  arm.
- `trace(P)` — the total remaining posterior uncertainty. It should fall as the
  filter becomes confident; if it stays at `p₀·d` the regressor is never excited.
- `fits` — the number of RLS updates.
- `|err|` — the one-step-ahead prediction error, to be read **against the ESO
  arm's**, since the controller is identical and only the estimator differs.
- `|theta|` — the mean absolute parameter.

**Prediction under test**: *"fails at N = 22/38 (under-excited,
ill-conditioned)"*. URB's `saint_arnoult` has 88 machine agents, so the regressor
is larger and the prediction is, if anything, sharper.

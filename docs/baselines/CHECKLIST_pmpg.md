# CHECKLIST — independent learning in a performative Markov potential game

**BASELINES.md** — a **new entry**, and the only arm here whose primary job is to
be a *control for a theorem* rather than a competitor.

`PACT_THEORY` **Theorem B2** (performative stability of the estimator) is the
paper's most novel theory, and it currently rests on Perdomo, Zrnic,
Mendler-Dünner & Hardt (2020) plus the multi-player extension. This paper is the
2025 version of exactly the object B2 needs — performative effects **inside a
Markov potential game**, which **A1** already establishes URB to be — so it
supplies both the missing citation and the missing arm.

**Paper** Sahitaj, Sasnauskas, Yalın, Mandal & Radanović, *Independent Learning
in Performative Markov Potential Games*, <https://arxiv.org/abs/2504.20593> ·
**base result** Perdomo et al. (2020), *Performative Prediction* ·
**NPG identity** Agarwal, Kakade, Lee & Mahajan (2021).

**Implementation** `urb_baselines/algos/pmpg.py` · **script** `scripts/pmpg.py` ·
**config** `config/algo_config/pmpg/config1.json`

**The question it isolates.** Is PACT-1's benefit the identification, or is it
performative stability bought by *deploying a policy, holding it fixed, and
retraining on what it produced* rather than chasing data its own updates keep
moving? If a plain independent learner that differs only in that schedule
recovers most of the gap, the schedule was doing the work.

---

## A. The specification, item by item

| id | item | status | where / why |
|---|---|---|---|
| **PSE-1** | Performatively stable equilibrium: each agent's policy is optimal for the environment its own deployment induces | **INSTRUMENTED** | not assumed. `sens` measures the sensitivity ε — the change in the induced population distribution per unit of policy change — which is the ε of B2's condition `ε < γ/β_J` |
| **PSE-2** | Existence under a sensitivity assumption | **N/A (theory)** | an existence theorem, not an algorithm |
| **IPGA-1** | `θ_i ← θ_i + α ∇_i J_i`, independent, unclipped | **IMPLEMENTED** | `--arm ipga`: `loss = −E[log π(a|s) Â] − entropy_coef·H`. Deliberately **not** PPO's clipped surrogate — the clip is a trust region on the step, and the paper's theorems are about these two updates |
| **INPG-1** | `θ_i ← θ_i + α F_i^{−1} ∇_i J_i`, converging to a PSE in the **last** iterate | **ADAPTED** | for a softmax policy the natural-gradient step is exactly a shift of the logits by `η·Â` (Agarwal et al. 2021), so it is applied in the equivalent mirror-descent form: build `target = softmax(ℓ_dep + η Â e_a)` and minimise `KL(target ‖ π_θ)`. Exact up to the network's ability to represent the target; gate **PMPG-1** checks the target is that object |
| **RRM-1** | Repeated retraining (repeated risk minimisation): deploy, collect a window under the frozen policy, retrain on it, redeploy | **IMPLEMENTED** | `learn()` does nothing inside a window; `_deploy_step()` retrains every agent and starts a new window |
| **RRM-2** | Retrain **to convergence** on the deployment's own data | **IMPLEMENTED** | `retrain_epochs = 20` full passes over the window, against the **deployed** policy as the reference, so every sample is on-policy for it |
| **MPG-1** | The game is a Markov potential game | **INHERITED** | `PACT_THEORY` A1 proves URB's CAV routing is an exact potential game on the co-presence graph; a one-state repeated game is a special case of an MPG, which the paper notes |

---

## B. What URB forced

### WINDOW-1 — `deploy_window = 64` days, and why that is not arbitrary
URB gives **one sample per agent per day**, and `ippo/config1.json` sets
`batch_size = 64`. A window shorter than 64 could not retrain at all —
`SingleStepPPO.learn` returns early below `batch_size` — so 64 is the smallest
window in which every agent has exactly one full host batch, i.e. one
deployment's worth of data. Nothing here was tuned on returns.

**Worth recording, because it changes how the comparison reads:** URB's own IPPO
is *already* closer to repeated retraining than it looks. With `update_every = 2`
the host calls `learn()` every other day, but the first 31 of those calls return
early and the memory is only consumed on the 64th day. So the schedule arm and
the host differ in the **update rule** first; `--arm cont` isolates the schedule
on its own.

### ADV-1 — the advantage is the standardised reward
A URB day is one step, so GAE collapses exactly to `adv = r − V(s)` for any γ, λ,
and URB's own PPO uses the standardised reward itself. `model._advantage()` is
the host's, unchanged, so IPGA and INPG differ from the host in the update rule
and in nothing else.

### SENS-1 — the sensitivity proxy
`ε̂ = TV(D(θ_i), D(θ_{i+1})) / ‖θ_{i+1} − θ_i‖`, where `D(θ)` is the empirical
population route mix over the deployment window. Total variation on a finite
simplex is the `W₁` of B2's ε-sensitivity assumption up to the metric on a
discrete set. Reported as `sens`, with the per-deployment `‖Δθ‖` and its
contraction ratio beside it: under the paper's condition the ratio is `< 1` and
roughly constant, which is what linear convergence to a PSE looks like.

---

## C. Not applicable, and not implemented

| item | status | why |
|---|---|---|
| Discounted state-occupancy measures in the analysis | **N/A** | one-step episode; there is nothing to discount |
| The paper's finite-time bound for the special case | **N/A (theory)** | a rate, not an algorithm; the contraction ratio in the report is the observable version |
| A centralised potential | **N/A** | the potential exists as a theoretical object (A1); no arm constructs it |
| Perdomo et al.'s repeated **gradient** descent variant | **SKIPPED** | the paper's algorithms are IPGA and INPG; repeated gradient descent would be a third schedule answering the same question as `--arm cont` |

---

## D. Arms

| arm | what it is |
|---|---|
| `inpg` (default) | natural policy gradient + repeated retraining — the paper's last-iterate result |
| `ipga` | plain policy gradient ascent + repeated retraining |
| `cont` | INPG on the host's own cadence: isolates the deployment schedule from the update rule |

---

## E. The gates

| gate | what it asserts |
|---|---|
| **PMPG-1** | the NPG target moves the taken action's probability in the direction of `sign(Â)` for **100%** of samples, and a retrain consumes its window |
| **drive pmpg**, **--arm ipga** | 200 days on `fakeenv.FakeURB`: learns, retrains 8 deployments, and the parameters actually move (`‖Δθ‖ ≈ 1.1e-3`) |
| **CFG-1 / CFG-2** | host block character-identical to `ippo/config1.json`; constructs and acts under its own shipped config |

---

## F. How to read a run

* `retrains` — **`***` if zero**: every window ended below `batch_size` and no
  agent updated once in the whole run, so the arm is a frozen random policy.
* `‖Δθ‖` first/last and the **contraction ratio** — `< 1` and roughly constant is
  the paper's linear convergence to a PSE.
* `sens` — B2's ε. Small means the deployment loop is not chasing itself.
* `npg_step` — **`***` if zero** for `inpg`/`cont`: the natural-gradient target
  is then the deployed policy itself and the arm is not INPG.

---

## G. What it tests

Two readings of the same result, and the arm separates them:

* if `cont` ≈ `inpg`, the deployment schedule is not what matters and B2's
  stability threshold is not being approached in practice;
* if `inpg` ≈ PACT-1 under drift, then the performative loop, not the
  identification, is carrying the result — which is exactly what B2 predicts can
  happen past the critical trust `g_stab`, and the arm to plot against
  `pred_gain − fit_gain` from the fixed-trust sweep.

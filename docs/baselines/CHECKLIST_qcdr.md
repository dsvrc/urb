# CHECKLIST — QCD restart bandits, on URB

**BASELINES.md** B7 *single-agent non-stationary RL, latent / change-point (drift
not observed)* · objection row *"Latent-variable non-stationary RL"*

B7's chosen representative is LILAC, which is SAC-based and continuous and was
assigned to MAPDN, so **on URB this class had no arm at all**. This fills it.

**Paper** Gerogiannis, Huang & Veeravalli, *Is Prior-Free Black-Box
Non-Stationary Reinforcement Learning Feasible?*, <https://arxiv.org/abs/2410.13772>
· **detector** Besson, Kaufmann, Maillard & Seznec, *Efficient Change-Point
Detection for Tackling Piecewise-Stationary Bandits* (JMLR 2022), which the paper
takes its Bernoulli-GLR from · **MASTER** Wei & Luo (2021), whose two tests the
`master` arm reproduces.

**No public code** was found for arXiv:2410.13772; the implementation is written
from the paper's equations, and every equation used is quoted in the rows below.

**Implementation** `urb_baselines/algos/qcdr.py` · **script** `scripts/qcdr.py` ·
**config** `config/algo_config/qcdr/config1.json`

**Why this paper is the right B7 representative on URB.** Its entire empirical
section is a *piecewise-stationary multi-armed bandit with 5 arms*, and URB on
these cities **is** a repeated K-armed bandit for every learner — §4 of
[README.md](README.md) measures that the observation is the constant
`[start_time, 0, 0, 0, 0]` for ~97% of travellers on every day. The method and
the host were written for the same object.

---

## A. The specification, item by item

| id | item | status | where / why |
|---|---|---|---|
| **GLR-1** | Bernoulli-GLR statistic `G_n = max_{1≤s<n} s·kl(μ_{1:s}, μ_{1:n}) + (n−s)·kl(μ_{s+1:n}, μ_{1:n})` | **IMPLEMENTED** | `bernoulli_glr()`; computed from one cumulative sum, so the full scan over split points is O(n). Checked against a hand computation at the true split by gate **QCDR-1** |
| **GLR-2** | Threshold `β(n, δ) = 2 log(3 n^{3/2} / δ)` | **IMPLEMENTED** | `glr_threshold()`, Besson et al. (2022) eq. for the Bernoulli GLR |
| **GLR-3** | `δ = 1/√T` when the number of change points is unknown | **IMPLEMENTED** | `delta: null` in the config resolves to `1/sqrt(training_eps)` at construction and is printed in the banner |
| **GLR-4** | Forced exploration at rate `α = √(N_C log T / T)` (their GLRklUCB) | **IMPLEMENTED** | `alpha: null` → `sqrt(log T / T)`, i.e. `N_C = 1`, the only value available without prior knowledge. `forced_exploration: false` is their QCD++ variant |
| **GLR-5** | On detection, clear the history and restart (their Algorithm 3) | **IMPLEMENTED** | `_restart()` rebuilds the agent's `SingleStepDQN` from scratch, flushes its replay memory and its per-route history, and resets ε to `eps_init` |
| **RR-1** | Random restart, intervals i.i.d. `Geo(η_R)` (Theorem 11) | **IMPLEMENTED** | `--arm rr`; `η_R = √η / log T` with `η = T^{−ξ}`, `ξ = 0.5` |
| **MAS-1** | MASTER Test 1, with the constants `54 (log₂T + 1) log(T/δ) ρ(2^m)` | **IMPLEMENTED** | `_Master.step()`. The constants are verbatim **on purpose**: Theorem 4 is a statement about those constants, and softening them answers a different question |
| **MAS-2** | MASTER Test 2, `18 (log₂T + 1) log(T/δ) ρ(t − t_n + 1)` | **IMPLEMENTED** | same |
| **MAS-3** | MASTER's multi-scale block schedule: order `m` restarted with probability `2^{−m}` | **IMPLEMENTED** | `_Master.step()`; this is what the paper says MASTER degenerates into once the tests never fire |
| **MAS-4** | `ρ(n)` = the base algorithm's regret rate | **ADAPTED** | `ρ(n) = √(K/n)`, the bandit instantiation the paper uses. On URB `K` is the number of candidate routes |
| **MAS-5** | `g̃` = the base learner's optimistic value estimate | **ADAPTED** | `max_k Q(s, k)` for the day's state, which is the role klUCB's index plays in their bandit instantiation. Computed **before** the action, so exploration does not change it |

---

## B. What URB forced

### SCALE-1 — the reward stream must be mapped into (0, 1), and the reference must be FROZEN
A Bernoulli GLR is defined for observations in `[0, 1]`. URB pays
`−travel_time`, of order −1000. Two things follow, and only the first is obvious.

1. **Raw, the detector can never fire.** Every value clips to the same boundary,
   the statistic is identically 0, and the run looks exactly like "there was no
   change point". This is the fourth member of the family in §4 of
   [README.md](README.md). `normalize_stream: false` reproduces it; gate
   **QCDR-2** measures it (GLR `0.00`, stream spread `0.0e+00`).

2. **A *running* standardisation is almost as bad.** It absorbs the shift it is
   meant to detect, because the mean it subtracts is dragged toward the new level
   by the new data. Measured in the same gate: a 300-second level change gives
   **GLR 15.5** under a running reference — below the threshold of 27.6, so it
   never fires — against **59.6** under a frozen one.

So the reference `(μ_ref, sd_ref)` is measured over the first `calib = 30` days
and then frozen, and re-measured after a restart. Re-calibrating on restart is
the right pairing: a restart declares the old regime over, so "normal" is
re-measured in the new one.

### HOST-1 — klUCB → URB's own DQN
The paper's arms are bandit algorithms (UCB, klUCB). Rule **R2** says the host's
learner and the paper's mechanism, so the learner is `scripts/iql.py`'s DQN with
the host block copied verbatim from `iql/config1.json`, and what comes from the
paper is the **detector and the restart**. The ε-greedy schedule is the host's
and is not touched; forced exploration is added on top of it, as the paper adds
it on top of klUCB's index.

### HIST-1 — the per-route history is windowed at `max_hist = 500`
A restart flushes the history anyway, so the window only bounds cost. It never
changes which split points are available within the window.

---

## C. Not applicable, and not implemented

| item | status | why |
|---|---|---|
| klUCB / UCB indices | **N/A** | the host's learner is a DQN with ε-greedy; an index policy would replace URB's own learner, which R2 forbids |
| `γ·max_a' Q(s', a')` in the target | **N/A** | a URB day is one step; URB's own IQL uses `target = reward` and every off-policy arm here does the same |
| Their regret plots | **N/A** | regret needs a known best arm; URB reports travel time against the benchmark's own ceilings |
| MASTER's full instance stack | **SKIPPED** | the paper's object of study is the two **tests** and the block schedule, and both are implemented. Maintaining a stack of nested MASTER instances would add machinery that Theorem 4 says is never reached |

---

## D. The gates

| gate | what it asserts |
|---|---|
| **QCDR-1** | the GLR does **not** fire on a stationary Bernoulli(0.5) stream (3.85 < 26.39) and **does** fire on a 0.2 → 0.8 change (49.29); and the max over split points is at least the value at the true split |
| **QCDR-2** | at URB's scale the raw stream cannot fire (GLR 0.00, spread 0) while the mapped one does (59.58 > 27.60) |
| **drive qcdr** | runs 200 days on `fakeenv.FakeURB`, learns, writes loss rows, and the mapped stream is not constant |
| **drive qcdr --arm rr**, **--arm master** | both ablations run and restart |
| **CFG-1 / CFG-2** | the host block is character-identical to `iql/config1.json`, and the arm constructs and acts under its own shipped config |

---

## E. How to read a run

The closing report prints, per arm:

* `restarts` and restarts per agent — **`***` if zero** for `glr` and `rr`,
  because the arm is then URB's own IQL with extra bookkeeping;
* `largest GLR / threshold` — how close the detector ever came. Far below 1 is
  itself a reportable result;
* `mapped-stream spread` — **`***` if ~0**, the scale failure above;
* for `--arm master`, the trigger counts for both tests and their closest
  approach. **Zero there is the finding, not a degenerate arm**: it reproduces
  Theorem 4 on URB (the thresholds need `T ≥ 1.24e9`; a run is 4000 days), and
  the report says so in place of a `***` line, so the sweep's `DEGENERATE` list
  stays meaningful.

---

## F. What it tests

The sharpest objection an identification method faces: *if the drift is not
observed, detect the change and restart — why estimate a coupling?*

`PACT_THEORY` **C4** says excitation is bounded by policy indecision, and a
restart re-injects exactly that. So this arm should track drift that a converged
estimator cannot, and pay for every restart with a cold start. Report it as the
trade-off it is; a result in either direction is publishable and neither is a
walkover.

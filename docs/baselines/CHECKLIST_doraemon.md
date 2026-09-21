# CHECKLIST — DORAEMON (domain randomisation via entropy maximisation), on URB

**BASELINES.md** B9 *robust RL* · objection row *"Domain randomisation / robust
RL"*

B9's "must" is `dr_ippo`: σ ~ `U[0, 3]` every day, the stock learner, evaluate at
the committed σ. Its weakness is the one this paper was written about — the range
is fixed in advance, so it is either too narrow to generalise or so wide that the
policy is conservative everywhere. **The two arms answer different objections**:
`dr_ippo` answers *"did you try domain randomisation"*, this one answers *"did
you try it with a curriculum that adapts to what the policy can actually do"*,
which is the form a sim-to-real reviewer has in mind.

**Paper** Tiboni, Klink, Peters, Tommasi, D'Eramo & Chalvatzaki, *Domain
Randomization via Entropy Maximization*, ICLR 2024,
<https://arxiv.org/abs/2311.01885> · **code**
<https://github.com/gabrieletiboni/doraemon> (read: `doraemon/doraemon.py`,
classes `DomainRandDistribution` and `DORAEMON`).

**Implementation** `urb_baselines/algos/doraemon.py` · **script**
`scripts/doraemon.py` · **config** `config/algo_config/doraemon/config1.json`

---

## A. The specification, item by item

| id | item | status | where / why |
|---|---|---|---|
| **OPT-1** | Eq. (4): `max_φ H(ν_φ)` s.t. `Ĝ(θ, φ) ≥ α` and `KL(ν_φ ‖ ν_{φ_i}) ≤ ε` | **IMPLEMENTED** | `solve_doraemon()`; both constraints are enforced on every candidate and gate **DOR-1** asserts the returned distribution satisfies both |
| **OPT-2** | Eq. (6): the backup problem `max_φ Ĝ` s.t. the same KL ball, when the incumbent is already infeasible | **IMPLEMENTED** | `_update_distribution()` runs it first and, if it still fails, commits it and skips the entropy step — Algorithm 1's control flow verbatim |
| **OPT-3** | Eq. (5): the importance-sampled success estimate `Ĝ = (1/K) Σ [ν_{φ'}(ξ_k)/ν_{φ_i}(ξ_k)] 1{σ(τ_k)=1}` | **IMPLEMENTED** | `_success_estimate()`; the log-ratio is clipped to ±30 before exponentiation |
| **DIST-1** | Uncorrelated univariate **Beta** distributions on a rescaled bounded support | **IMPLEMENTED** | `BetaDR`, one Beta rescaled to `[lo, hi]` |
| **DIST-2** | Entropy and KL in closed form | **IMPLEMENTED** | `BetaDR.entropy()` / `.kl_to()`; gate **DOR-1** checks both against numerical integration to `1e-4` |
| **HP-1** | `α = 0.5` | **IMPLEMENTED** | the paper states it selected 0.5 for **all** experiments; not tuned here |
| **HP-2** | `init_beta_param = 100` | **IMPLEMENTED** | the released code's default: start narrow, let the entropy objective widen it |
| **HP-3** | KL trust region `ε` | **ADAPTED** | the paper grid-searches it per environment over roughly 0.001–0.1; `kl_ub = 0.05` is the middle of the range they report, chosen **before** any URB run and not revisited |
| **ALG-1** | The RL training subroutine between distribution updates | **IMPLEMENTED** | URB's own IPPO, unchanged, host block from `ippo/config1.json` |
| **EVAL-1** | Evaluate at the committed parameter with randomisation off | **IMPLEMENTED** | `begin_test()` calls `dr.fix(committed)`, identical to `dr_ippo` — B9's second half |

---

## B. What URB forced

### DIM-1 — one randomisation dimension
The paper randomises a vector of dynamics parameters with one Beta per
coordinate. URB-NS has exactly one certified dial, σ, so the distribution is a
single rescaled Beta. Every quantity in the paper's formulae (entropy, KL,
importance weight) is per-coordinate and sums, so the one-dimensional case is
their own formulae with one term.

### SUCC-1 — the success indicator is relative excess delay, calibrated and frozen
The paper is explicit that `σ(τ)` is "defined through domain knowledge" and
allows "a lower bound on the expected return". URB's return is `−travel_time` in
seconds, which is comparable neither across travellers nor across networks. The
indicator here is

    success(day) ⟺ mean over CAVs of (tt_i / fft_i − 1) ≤ perf_lb

— relative excess delay, BPR's natural form, and the same target PACT-1's
regressor uses. `fft` comes from the environment's own published free-flow times
through `records.flatten_free_flow`, which turns RouteRL's `1e9` padding into
NaN so a masked route can never enter the mean.

With `perf_lb: null` the threshold is **calibrated** as the median day-metric
over the first `warmup_days = 200` days — while the distribution is still at its
initial value — and then **frozen**. The rule is declared, reproducible and reads
nothing from returns. A numeric `perf_lb` (or `--perf-lb`) overrides it.

### SOLVE-1 — a deterministic grid, not scipy
The released code calls `scipy.optimize.minimize` with two `NonlinearConstraint`s.
`scipy` is **not** in URB's `requirements.txt`, and a baseline must not add a
dependency to the host. With two parameters the same constrained problem is
solved by a coarse-to-fine grid over `(log a, log b)`: exact to grid resolution,
and deterministic, which a local optimiser from a moving start point is not. The
**objective and the constraints are the paper's**; only the solver differs.

`_digamma` is written out for the same reason (entropy and KL need ψ, and
`scipy.special` is unavailable); it is the standard recurrence-plus-asymptotic
series, accurate to ~1e-12 over the range the grid searches.

### EP-1 — an episode is a day
The paper's `K` trajectories per distribution update are `update_every_days = 100`
days here, and the policy update in between is the host's own.

### WRAP-1 — the certified dial is not touched
`_BetaRandomiser` **subclasses** `domain_random.DomainRandomiser` and overrides
only the draw. Everything else — wrapping `reset`, the `sigma_floor = 1e-9` and
its reason, `fix()` detaching for the test phase — is inherited unchanged, and
`urb_baselines/domain_random.py` itself is not modified, so `dr_ippo` and
selftest gate **DR-1** are unaffected. The reason the dial is left alone at all
is in [CHECKLIST_dr_ippo.md](CHECKLIST_dr_ippo.md) WHERE-1.

---

## C. Not applicable, and not implemented

| item | status | why |
|---|---|---|
| `bootstrap_values` (the J constraint from a context-aware value function) | **SKIPPED** | the released default is Monte-Carlo samples; URB's host is actor-only and has no `V(s, ξ)` to bootstrap from |
| `robust_estimate` / `alpha_ci` / `performance_lb_percentile` | **SKIPPED** | alternatives to the sample-mean constraint in the release; the paper's headline algorithm uses the mean |
| `reset_agent`, `train_until_performance_lb`, `stopAtRewardThreshold` | **SKIPPED** | training-subroutine controls tied to the release's SB3 loop; URB's day loop is the host's |
| `target_distr` (converge to a target instead of maximising entropy) | **SKIPPED** | the paper's headline objective is the entropy; the target-distribution variant answers a different question |
| Multi-dimensional `prior_constraint` | **N/A** | one dimension |

---

## D. Arms

| arm | what it is |
|---|---|
| `doraemon` (default) | adapt the Beta by entropy maximisation |
| `fixed` | keep the initial Beta(100, 100): what the adaptation buys, holding the family and the support fixed |

---

## E. The gates

| gate | what it asserts |
|---|---|
| **DOR-1** | the Beta entropy and KL match numerical integration to `1e-4`; `KL(p‖p) = 0`; the solve widens (`−0.827 → −0.678`) while staying inside the trust region (`KL = 0.025 ≤ 0.05`) and satisfying the success constraint; and the backup problem also stays inside it |
| **drive doraemon**, **--arm fixed** | 200 days on `fakeenv.FakeURB`: learns, calibrates the threshold, and performs 9 distribution updates ending at Beta(10, 10) — i.e. it actually widened |
| **DR-1** | unchanged, and still passing: the shared randomiser is not modified by this arm |
| **CFG-1 / CFG-2** | host block character-identical to `ippo/config1.json`; constructs and acts under its own shipped config |

---

## F. How to read a run

* the **calibration line**, printed once: the frozen success threshold;
* `beta_a`, `beta_b`, `H`, `sig_mu` — the distribution and where its mass is;
* `G` — the importance-weighted success rate at the last solve, against `α = 0.5`;
* the **trajectory table** in the closing report: `(day, a, b, H)` sampled
  through the run, which is the curriculum;
* **`***` if the distribution was never updated** (the warm-up never finished, or
  no day produced a usable travel-time record) or **if the entropy never moved**
  (the trust region admits nothing, or the constraint is satisfied/violated
  everywhere on the grid — `G` says which).

---

## G. What it tests

B9's prediction, sharpened. If hedging beat identifying, the best hedge would be
the widest one the policy can carry — so this is the strongest form of the
objection. If it still cannot track `A(t)` **within** a run, that is because no
severity *distribution* is a substitute for knowing which draw you are in, which
is the distinction the paper is making. Read it next to `dr_ippo`: same support,
same learner, same evaluation, and the only difference is whether the training
distribution adapted.

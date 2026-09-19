# CHECKLIST — LCPO (ICLR 2025) on URB

**Paper** Hamadanian, Nasr-Esfahany, Schwarzkopf, Sen, Alizadeh, *Online
Reinforcement Learning in Non-Stationary Context-Driven Environments*, ICLR 2025.
<https://openreview.net/pdf?id=l6QnSQizmN>

**Code read** <https://github.com/pouyahmdn/LCPO> @ `main`
- `windy-gym/agent/lcpo.py` — the agent: reservoir, then `train_lcpo`
- `windy-gym/agent/core_alg/core_lcpo.py` — `locopo`, `trpo_step`, `linesearch`
- `windy-gym/agent/core_alg/core_lcppo.py` — `locoprpo`, the authors' PPO variant
- `windy-gym/agent/core_alg/core_pg.py` — `policy_gradient`, `gae_advantage`,
  `cumulative_rewards`, `value_train`, `train_entropy`
- `windy-gym/agent/core_alg/core_trpo.py` — `conjugate_gradients`
- `windy-gym/buffer/buffer_ood.py` — `OutOfDSampler`
- `windy-gym/agent/a2c.py` — return normalisation, entropy tuning
- `windy-gym/env/pendulum.py` — `is_different` (l2 / mahala / mahala_full)
- `windy-gym/param.py`, `windy-gym/run_config_phase1.py` — defaults and the
  configuration the paper's continual experiments actually ran
- `straggler_mitigate/agent/lcpo.py`, `.../core_lcpo.py` — the second instance

**BASELINES.md** B6 · Tier 1 item 7 · objection row *"Non-stationary RL with an
observed context solves this"*

**Implementation** `urb_baselines/algos/lcpo.py` · **script** `scripts/lcpo.py` ·
**config** `config/algo_config/lcpo/config1.json`

**Arms** `--arm lcpo` (default, the paper's TRPO form) · `--arm lcppo` (the
authors' PPO variant) · `--arm a2c` (ablation: the same actor-critic with the
local constraint off — this is LCPO's own fallback path)

---

## A. The method, item by item

| id | item | source | status | where / why |
|---|---|---|---|---|
| **CORE-1** | On-policy actor-critic: separate policy and value networks, separate optimisers | `a2c.py`; `core_pg.value_train` | **IMPLEMENTED** | `_Learner.__init__`: `policy_net`, `value_net`, `opt_p`, `opt_v` |
| **CORE-2** | Collect `master_batch` transitions, then one training iteration | `a2c.run` | **ADAPTED** | `_Learner.learn` updates once `len(mem) >= master_batch`. See ADAPT-1 for the value. |
| **CORE-3** | Returns by `cumulative_rewards`, advantage by `gae_advantage` (γ, λ) | `core_pg` | **N/A (collapses exactly)** | A URB day is one step and terminates, so `returns = r` and `adv = r − V(s)` for *any* γ, λ. Both are therefore not exposed: printing them would imply they do something. |
| **CORE-4** | Value loss = MSE(V, returns), grad-clipped at 1, its own optimiser | `core_pg.value_train` | **IMPLEMENTED** | `_Learner.learn`, `F.mse_loss` + `clip_grad_norm_(…, 1)` |
| **CORE-5** | Reward normalisation `r / sqrt(ret_rms.var + 1)` from a running estimator over returns | `a2c.run` lines 172–173; `utils/rms.py` | **IMPLEMENTED (verbatim)** | `RunningMeanStd` + `_Learner.learn`. Load-bearing: URB pays −travel_time. |
| **CORE-6** | No advantage standardisation | `core_pg` (absent) | **IMPLEMENTED** | Not applied. CORE-5 is the authors' scale mechanism and is used instead. |
| **OOD-1** | Anchor buffer is a **reservoir** over all history (uniform once full), not a ring | `buffer_ood.add_many_exp` | **IMPLEMENTED (verbatim)** | `OutOfDSampler.add_many_exp`. Gate **LCPO-2** proves the buffer still holds day 39 after 200 days. |
| **OOD-2** | A separate FIFO **window** of the last `window` observations is the comparison base | `buffer_ood` | **IMPLEMENTED (verbatim)** | `OutOfDSampler.recent_states`. Gate **LCPO-2** proves it holds only days 198–199. |
| **OOD-3** | `is_different(data, base)` filters the anchors | `env/pendulum.is_different` | **ADAPTED** | All three forms (`l2`, `mahala`, `mahala_full`) implemented in `make_is_different`, plus `all`. Default is `all`. See ADAPT-2. |
| **OOD-4** | `get` resamples up to 5× and returns `[]` rather than a short batch | `buffer_ood.get` | **IMPLEMENTED (verbatim)** | `OutOfDSampler.get` |
| **OOD-5** | Empty anchor batch ⇒ fall back to plain policy gradient | `core_lcpo.train_lcpo` | **IMPLEMENTED** | `_Learner.learn`, `if len(anchors) > 0 … else self._policy_gradient(...)` |
| **OOD-6** | Reservoir capacity `master_batch * num_epochs // ood_subsample` | `windy-gym/train.py:160` | **IMPLEMENTED** | `ood_capacity = training_eps // ood_subsample` per agent — the same formula, with URB's transitions being training days. |
| **TRPO-1** | CG direction computed against the **OOD** KL Hessian | `core_lcpo.trpo_step`, `q_out_dot_v` | **IMPLEMENTED** | `trpo_step._hvp(get_kl_out)` → `conjugate_gradients` |
| **TRPO-2** | 15 CG steps, residual tol 1e-10 | `core_trpo.conjugate_gradients` | **IMPLEMENTED (verbatim)** | `conjugate_gradients(..., 15, 1e-10)` |
| **TRPO-3** | Damping added to the Hessian-vector product | `core_lcpo` | **IMPLEMENTED** | `+ v * damping`, `trpo_damping = 0.1` |
| **TRPO-4** | Step size `l_out = sqrt(2 · kl_out / v_out)` | `core_lcpo.trpo_step` | **IMPLEMENTED** | same expression |
| **TRPO-5** | Line search: ≤10 backtracks, `accept_ratio = 0.1`, requires improvement **and** `kl_out ≤ max_kl_out` **and** `kl_in ≤ max_kl_in` | `core_lcpo.linesearch` | **IMPLEMENTED (verbatim)** | `linesearch`. Gate **LCPO-1** asserts both bounds hold after a step, in both dual modes. |
| **TRPO-6** | `v_out == 0` ⇒ take no step | `core_lcpo.trpo_step` | **IMPLEMENTED (+ guard)** | `if vout <= 0: return "no_grad"`. `<= 0` rather than `== 0`: a negative would put a negative under the paper's square root. Counted and reported separately from a rejection. |
| **TRPO-7** | Dual solve over both constraints (`--trpo_dual`) | `core_lcpo.trpo_step` (windy-gym only) | **IMPLEMENTED (verbatim, incl. the asymmetry)** | `trpo_step(..., solve_dual=True)`. `a_out[0]` uses `stepindir · q_out(stepoutdir)`, exactly as the reference does; the comment says so. Off by default, as in the reference. |
| **TRPO-8** | Surrogate `−adv · exp(logπ − logπ_old)` minus `entropy_factor · entropy` | `core_lcpo.get_loss_lcpo` | **IMPLEMENTED** | `_locopo.get_loss` |
| **TRPO-9** | Both KLs are `KL(π_old ‖ π_new)` with `π_old` detached | `core_lcpo.get_kl_out/in` | **IMPLEMENTED** | `_locopo.get_kl_out`, `get_kl_in` |
| **PPO-1** | LCPPO: clipped surrogate + `kappa · CrossEntropy(logits_new(ood), π_old(ood))` | `core_lcppo.locoprpo` | **IMPLEMENTED** | `_Learner._lcppo` |
| **PPO-2** | LCPPO: 3 shuffled passes split into `ppo_iters` minibatches | `core_lcppo.locoprpo` | **IMPLEMENTED (+ guard)** | `.repeat(3)`; `n_iters` is reduced when the master batch is smaller than `ppo_iters`, which the reference's reshape would have crashed on. |
| **PPO-3** | LCPPO: early stop when approx KL > `1.5 · ppo_kl_in` | `core_lcppo.locoprpo` | **IMPLEMENTED (verbatim)** | `_lcppo`, `if approx_kl > 1.5 * kl_in_lim: break` |
| **ENT-1** | Auto-tuned entropy coefficient toward `auto_target_entropy · max_entropy` | `a2c.tune_entropy`, `core_pg.train_entropy` | **IMPLEMENTED, OFF BY DEFAULT** | `_Learner._tune_entropy`. Default uses the host's fixed `entropy_coef` so exploration pressure matches every other arm; set `auto_target_entropy: 0.1` to reproduce the paper's configuration (see ADAPT-3). |
| **CTX-1** | The context process must be **observed** by the agent | paper §1, §4 | **IMPLEMENTED** | `A(day)` appended to every observation (`LCPO.act`), sourced from `urb_baselines/context.py`. The run aborts without a driver unless `--allow-no-driver`. |
| **CTX-2** | The OOD metric is computed on the **context coordinates** | `env/pendulum.is_different` via `only_context` | **IMPLEMENTED** | The context occupies the last `ctx_dim` coordinates; `make_is_different` is given that slice. |
| **TEST-1** | Evaluation is deterministic | host convention | **ADAPTED** | `begin_test` switches to argmax, matching `scripts/ippo.py`'s test phase so every arm is evaluated the same way. |

---

## B. Adaptations, in full

### ADAPT-1 — `master_batch` is 20 days, by a declared rule
In LCPO, `master_batch` is an **algorithm** hyperparameter, not a host one:
`run_config_phase1.py` gives LCPO `--master_batch 200` and TRPO `--master_batch
3200` in the same sweep. It sets how long the "recent window" is, and therefore
what "out of distribution" can possibly mean.

URB's weather cycle is 100 days. A window of 64 days (the host's `batch_size`)
spans two-thirds of it, so *one window already contains every context* and
nothing in the past can be out of distribution — LCPO would silently be A2C. The
default is therefore the declared rule `max(8, period // 5)` = **20 days**, read
from the driver, printed in the banner, and overridable with `master_batch`.

This is a hyperparameter choice made by the experimenter from the problem's known
timescales, not information the agent uses at run time.

### ADAPT-2 — the OOD filter defaults to `all`, and the paper's threshold does not transfer
`param.py`'s own defaults are `lcpo_ood_type='l2'`, `lcpo_thresh=-9` — and a
squared distance is never below −9, so **the released default accepts every
reservoir anchor**. `run_config_phase1.py` instead runs `--lcpo_thresh 1`, which
is a distance in the units of *their* context: a 6-dimensional wind vector.

URB's context is a rainfall intensity in `[0, 1]`, whose squared distances never
exceed 1, so a threshold of 1 would accept **nothing**. There is no
scale-preserving way to carry the number across. The default here is `all` —
identical in effect to `param.py`'s own default, needing no constant to
transfer, and always engaging. `mahala` (the straggler-experiment form, scale-
free, `thresh = -3`) and `l2` are both implemented and selectable, including the
authors' `base.shape[-1]` normalisation by the *full* observation width, which is
reproduced because their threshold means nothing without it.

A ridge (`ood_cov_ridge`, 1e-8) is added to the Mahalanobis covariance because
every day of URB's dry half has `A == 0` **exactly**, so a window inside the
placebo regime is singular and `cholesky` raises. Gate **LCPO-3** covers that
case: against an all-dry window a dry anchor is in-distribution and every wet
one is not.

### ADAPT-3 — entropy coefficient
The authors auto-tune it (`--auto_target_entropy 0.1 --entropy_max 3e-2`). The
mechanism is implemented (`_tune_entropy`), but the default uses URB's shared
`entropy_coef` so that the exploration pressure is identical across every arm in
the comparison. Rule R2 in the README: host hyperparameters come from the shared
config. Set `auto_target_entropy: 0.1` in the config to run the authors' version.

### ADAPT-4 — learning rates
The authors use `lr_rate 4e-4` for the actor and `val_lr_rate 1e-3` for the
critic, i.e. the critic is 2.5× faster. The actor lr is URB's shared `lr`
(3e-4); the critic keeps its own `value_lr` (1e-3), giving a ratio of 3.3×. The
critic's rate is a paper hyperparameter with no URB equivalent, so R2 says it
comes from the paper.

---

## C. Nothing was skipped

Every mechanism in the paper and in both reference instantiations is present.
The only items marked N/A (CORE-3) are ones that *provably collapse* on a
one-step episode rather than ones that were left out.

---

## D. Gates

| gate | what it proves |
|---|---|
| **LCPO-1** | After a step, `kl_out ≤ 1e-4` **and** `kl_in ≤ 0.1` **and** the surrogate improved — in both `trpo_dual` modes. This is the one that would catch a wrong Hessian, a wrong step size or a line search that does not enforce both bounds. |
| **LCPO-2** | The anchor buffer is a reservoir and the window is a ring: after 200 days the sampler still returns day 39 while the window holds only days 198–199. A ring-buffer anchor set would silently make the constraint redundant. |
| **LCPO-3** | The Mahalanobis filter survives a singular (all-dry) window and discriminates correctly. |
| **drive lcpo / lcppo / a2c** | All three arms run, learn and produce loss rows; the constraint engages on 100% of updates. |

---

## E. What to watch in a real run

The banner prints the arm, both KL limits, the filter, the reservoir size and
`master_batch`. The per-100-day line prints:

- `ood_hit` — the fraction of updates that had an anchor batch. **If this is
  near zero the arm is not LCPO**, it is plain policy gradient, and `close()`
  prints a `***` block saying so.
- `nograd` — the fraction of constrained updates where the local surrogate had
  no gradient at all (a converged or saturated policy), so there was nothing to
  step. This is reported *separately* from a line-search rejection because the
  two mean completely different things.
- `accept` — of the updates that did have a gradient, the fraction whose line
  search accepted. All rejected ⇒ a `***` block.
- `kl_in`, `kl_out` — the realised movement. `kl_in ≈ kl_out` means the anchors
  are not distinguishable from the current data, so the out-of-distribution
  constraint is binding on the very states the update is trying to move. That is
  a property of the problem worth reporting, and it is visible rather than
  hidden.

**Prediction under test** (BASELINES.md B6): *"it prevents forgetting across
wet/dry contexts but cannot cancel; knowing the weather is not knowing the
neighbours' load under it."* The arm that isolates the first half from the
actor-critic base is `--arm a2c`; the arm that isolates the context from the
machinery is `scripts/oracle_ippo.py`.

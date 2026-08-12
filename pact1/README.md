# PACT-1 on URB

Online identification of the congestion coupling, with learned trust — on **base
URB, unmodified**. No injected non-stationarity, no changed reward, no changed
network, no changed demand.

---

## Run it

Exactly like any other URB algorithm script:

```bash
nohup python scripts/pact1.py --id sai_pact1_0 --alg-conf config1 --task-conf config4 --net saint_arnoult --env-seed 42 --torch-seed 0 > pact1_saint_arnoult_alg-conf_config1_task-conf_config4_env-config_config1.txt &
```

### Do these three things first — they cost minutes, not runs

**1. Offline self-test (seconds, no SUMO).**

```bash
python pact1/selftest.py
```

19 checks. It builds a synthetic city with a *known* linear congestion law and
asserts the estimator recovers it. If this passes, the arithmetic is sound and any
later failure is a statement about the city, not the code.

**2. Dry run (minutes — path generation only, no AV training).**

```bash
python scripts/pact1.py --id dry --alg-conf config1 --task-conf config4 --net saint_arnoult --env-seed 42 --mode dry
```

Builds the basis from the real network, runs every startup gate, prints the banner
and exits before a single AV episode.

**3. Probe (~60 days instead of 4000).**

```bash
python scripts/pact1.py --id probe_sai --alg-conf config1 --task-conf config4 --net saint_arnoult --env-seed 42 --mode probe --probe-eps 60
```

Runs with **uniformly random AV routes** — maximum excitation, the best case for
identification — and reports `fit_r2`. This is the go/no-go:

| probe `fit_r2` | reading |
|---|---|
| **> 0.3** | the reduction holds here. Run the full thing. |
| 0.1 – 0.3 | weak but real. Try `peer_scope: "all"`, a larger `dur_factor`. |
| **< 0.05** | a linear road-class model does not explain this city's delays. A full run cannot help — change the basis or the network, don't spend the compute. |

### Arms

```bash
--arm pact1    # default: trust learned by the policy
--arm blind    # trust forced to 0 -> plain IPPO through the IDENTICAL wrapper
--arm fixed    # constant trust: isolates "is trust learned, or just well-initialised?"
```

The blind arm is the honest baseline: same wrapper, same observation augmentation,
same seed, same everything — only `g = 0`. `selftest.py` proves it is bit-identical
to the untouched policy.

---

## What the method is

Base URB's congestion **is** a cross-agent, dynamics-mediated coupling: your travel
time rises because other vehicles chose your roads, computed by SUMO physics, with
the reward untouched. That half of the structure is native to the domain — it is
not injected.

A link performance function (BPR) says delay on a link is a function of flow on that
link, so to first order

```
tt_i / fft_i - 1  ≈  β₀ + Σ_m β_m · x_{m,i}
```

where `x_m` is the per-lane peer load on the class-`m` links of your route,
computable **exactly** from peers' executed route choices and the network geometry,
and `β*` is an `(r+1)`-vector holding every unknown. That is the guide's **T2
reduction** instantiated in a city we did not build: `r` parameters, independent of
`N` and of the number of links.

```
      peers' executed routes ──► x₁ x₂ x₃      exact arithmetic, no estimation
                                    │
      own realized tt vs free flow ─┤          the honest sensor
                                    ▼
            per-agent RLS with forgetting  ──►  β̂
                                    ▼
              g = trust(w) · confidence(P)      w = ONE policy output
                                    ▼
              logits_k ─= g · κ · z(tt̂_k)      the steering channel
```

### What transfers from Ant, and what does not

| | |
|---|---|
| **T2, the reduction** | ✅ the whole basis of this file |
| honest proprioceptive sensor | ✅ realized travel time vs own free-flow |
| per-agent RLS, decentralized | ✅ agent *i* never sees another's residual |
| inverted trust prior (§III.5) | ✅ `g_bias = 2.2` → 0.90, not 0.5 |
| covariance-gated reliance | ✅ `conf = 1/(1 + tr(P)/p₀)` |
| **floor property** | ✅ `g = 0` ⇒ untouched policy, **exactly**, for any `β̂` |
| T4, the compensation commons | ✅ steering onto a route *loads* it — Wardrop/Braess. Logged as `route_switch_frac` + `herd_index` |
| **T3, conjugacy / channel inverse** | ❌ **does not transfer.** You cannot subtract minutes off a congested road. The inverse is replaced by a trust-weighted logit shift. |

**Say this plainly in the paper.** The claim on URB is the reduction and the
identification, not exact compensation.

---

## Two URB-native improvements (neither changes the core)

**1. The regressor is centred on a geometric reference load.**
`x_ref[m,i,k]` is the class-`m` load agent `i` would see on route `k` if every peer
chose uniformly — a function of network geometry and the fixed travel schedule
only, so no run data touches it and it is part of the declared model class.

Why it matters, measured: raw `x` has a large common mean (mean 37 against an
intercept column of 1), which made `cond(E[ψψᵀ]) ≈ 1.3e5`. The intercept and the
class channels then trade off and the *split* becomes unidentifiable even though
prediction is fine — exactly the SMAC failure of guide §III.6. Centring and scaling
per class dropped it to **24** (class-segregated city) and **336** (class-mixed
city), and β recovery from *unidentifiable* to **0.001 / 0.012**. `selftest.py`
locks this in: remove the centring and the test fails loudly rather than the run
quietly reporting a decomposition it did not earn.

**2. Trust is learned with no change to the PPO objective.**
Ant appended trust as a sampled action dimension. Here trust is deterministic given
the state, and the route log-probability already depends on it through the shift:

```
log π(k|s) = log softmax( logits(s) − g(w(s))·κ·z )_k
```

so the ordinary policy gradient reaches `w` with no extra sampling, no extra
log-prob term, and no change to the clipped surrogate. `selftest.py` asserts the
trust head actually receives gradient.

---

## The gates — what stops a bad run

| # | when | checks | on failure |
|---|---|---|---|
| **1** | startup | vectorised waveform == the brute-force definition | **abort** — wiring bug |
| **2** | startup | `g=0` is an exact no-op; `g=1` moves the logits | **abort** |
| **2b** | before SUMO | torch shift == numpy shift | **abort** |
| **3** | startup | a lone traveller reads exactly zero peer load (zero-diagonal / N=1) | **abort** |
| **4** | startup | `paths.csv` route order == `env.get_free_flow_times()` | **abort** — the one silent catastrophe: a permuted route order points every waveform at the wrong road while looking perfectly healthy |
| **5** | ep 20 | the waveform is not inert | **abort** |
| **6** | ep 300 | rolling `fit_r2 ≥ 0.05` | **abort** — the reduction does not hold here; **this is the credit saver** |
| **7** | ep 300 | `cond(E[ψψᵀ])` | **warn only** — a degenerate regressor still predicts, it just cannot decompose θ. Claim accordingly (§III.6). |

Gates 1–4 and 2b run before any AV episode. Set `pact1.gate_abort: false` to
downgrade all of them to warnings.

---

## Reading the trace

One row per day, written to the **repo root** as `pact1_debug_<exp_id>.csv` and
flushed every episode, so a multi-hour run can be tailed from one place and
parallel arms never collide. Override with `pact1.debug_dir`.

```bash
python pact1/watch.py sai_pact1_0 --last 200 --baseline 3.21
```

`watch.py` reads the live file and answers the two questions separately: **is the
method working** (`fit_r2`, liveness, conditioning, trust) and **is it winning**
(`t_CAV` against the humans it replaced). Keeping them apart matters — the method
can work perfectly and still not win, which is a statement about how much headroom
routing has in this city, not a bug.

**The winning criterion is URB's own.** A run is "won" when the CAV fleet is on
average faster than the human drivers it replaced, i.e. `cav_adv = t_HDV / t_CAV >
1`. That is config-independent, which is what makes it the honest read when your
task config differs from the published table. `tt_cav` is in RouteRL's own units,
so it is directly comparable to the `t_CAV` column of URB's results table — but
compare against **your own baseline run with a matched `--task-conf`**, never
against the published numbers, which used a task config this distribution does not
ship.

The two columns that matter most:

| column | question | healthy |
|---|---|---|
| **`fit_r2`** | `β̂·ψ_true` vs realized excess — **does the reduction hold in this city?** | rises and stays up |
| **`pred_r2`** | the forecast actually used vs realized — **is peer load predictable a day ahead?** | below `fit_r2`, but positive |

`fit` high + `pred` low means the physics is right but the fleet is flapping faster
than `rho` can track. That is a finding about the domain, not a bug — and keeping
the two apart is what stops a bad number being blamed on the wrong thing.

| column | meaning |
|---|---|
| `beta_intercept/local/collector/arterial` | the identified marginal delay per class |
| `beta_spread` | cross-agent disagreement about β |
| `conf`, `innov`, `n_upd` | estimator self-belief, innovation, rows consumed |
| `cond_psi` | can θ be *decomposed*, not just predicted? (§III.6) |
| `x_local/collector/arterial`, `x_std` | excitation — the liveness signal |
| **`trust_pol`** vs **`trust_app`** | policy-set vs applied (= × confidence). **Report separately** (§III.11): if `trust_pol` never moves, trust was well-initialised, not learned — say so. |
| `shift_absmean` | how hard the model is actually steering |
| **`tt_cav`**, `tt_hdv` | mean AV / human travel time, in URB's own units |
| **`cav_adv`** | `t_HDV / t_CAV`. **> 1 = won**, by URB's winrate definition |
| `tt_cav_roll`, `cav_adv_roll` | the same, averaged over `roll_episodes` (100) — read these, not the per-day values |
| `reward`, `excess_mean` | where the return is |
| `clip_frac` | fraction of SUMO travel-time outliers clipped |
| **`route_switch_frac`**, **`herd_index`** | the T4 commons signature: steering onto the fast route *loads* it. Rising herd alongside rising trust is the externality becoming visible. |

---

## Configuration

`config/algo_config/pact1/config1.json`. Host hyperparameters are **identical** to
`ippo/config1.json`, so an arm difference cannot be an algorithm difference.

Everything under `pact1` is a declared constant or an arm switch. Per guide §II.1:
**if you find yourself tuning one to make a curve look better, it belongs in the
ablation table, not in the default.**

| knob | role |
|---|---|
| `rho` | occupancy-EMA forecast smoother. **On Ant this was known env structure; here it is a method hyperparameter** — declare it and sweep it (`{0.7, 0.8, 0.9}`) as the misspecification row. |
| `forget`, `p0` | RLS bias/variance dial. The tracking floor is `Θ(√(noise·drift))` and cannot be tuned away, only balanced (§III.9). |
| `g_bias` | **2.2 → trust 0.90 at `w=0`.** Do not set this to 0. Getting this backwards cost a 10M-step Ant run 1800 return (§III.5). |
| `kappa` | logit-shift scale. One declared constant; sweep it in the ablation table. |
| `peer_scope` | `all` (humans + fleet — a fleet observing link counts is realistic) or `fleet` (decentralized purity; note N=1 then vanishes exactly). |
| `dur_factor`, `overlap_slack` | co-presence window width. |
| `speed_bounds` | road-class boundaries in m/s. All seven shipped networks partition cleanly at the default `[8.5, 14.0]`. |
| `raw_feature_obs` | **§III.12's trivial ablation** — hand the policy the raw `x_m` with no estimator. If it matches PACT-1, the estimator is decoration. **Run it early.** |
| `freeze_beta` | bypass the estimator with a hand-set β. |

---

## Interfacing with RouteRL (things that bite)

Three RouteRL behaviours the implementation has to respect. All three are covered
by `selftest.py`, so they cannot regress silently.

**Records come from the episode CSVs, not from memory.** `env.travel_times_list`
exists, but RouteRL snapshots *and resets* it inside `_reset_episode`, which runs
during the last `env.step()` of a day — so whether a read afterwards returns today's
rows or yesterday's is a library timing detail. `results/<exp_id>/episodes/ep<N>.csv`
states which day it describes, so that is preferred and the source is **locked** on
first success (mixing them would feed the estimator the same day twice).

**Multi-day flushes are replayed per day.** RouteRL flushes every `save_every`
episodes. Pooling a flush would pair day *t−4*'s travel time with day *t*'s peer
waveform — invisible except as an inexplicably low `fit_r2`. `save_every` is
forced to 1 for this arm (disk only, no dynamics).

**The coordinator must never be deep-copied.** RouteRL deep-copies `all_agents`
every episode reset, and each agent carries `agent.model` → the coordinator → the
open debug-CSV handle. Without `__deepcopy__` returning `self`, that raises
`cannot pickle 'TextIOWrapper'` and would also copy ~18 MB of Gram matrices per
episode. Anything you attach to an agent must be deepcopy-safe and cheap.

**RouteRL's episode numbers include the human-learning days**, so the offset is
inferred once and trace rows are labelled in AV-training time.

## Honest limits — state all of these

1. **T3 does not transfer.** There is no channel inverse in base URB; the method
   steers, it does not compensate. The paper's claim here is the reduction.
2. **`rho` is a method hyperparameter on URB**, not known structure as on Ant.
3. **The linearity is an approximation.** BPR is a first-order model of a
   microsimulator with signals, intersections and spillback. `fit_r2` is the
   measurement of how good an approximation — report it, don't assume it.
4. **θ may be predicted without being decomposed** where `cond_psi` is large.
   Measure, then claim accordingly.
5. **Trust may be well-initialised rather than learned.** Report `trust_pol` and
   `trust_app` as separate columns and say which happened.
6. **No CTDE arm.** URB's IPPO is actor-only with no critic, so the Pigouvian
   prediction of §III.10 cannot be tested here without porting to
   `mappo_torchrl.py`. That is follow-up work, not a result.

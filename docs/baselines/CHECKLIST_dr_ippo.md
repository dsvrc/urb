# CHECKLIST — domain randomisation over severity, on URB

**BASELINES.md** B9 · Tier 1 item 1 · objection row *"Domain randomisation /
robust RL"*

> *"**Must (zero cost): domain randomisation over σ** — resample σ ∈ [0, 3] per
> episode (a NEW environment flag, ≈ 10 lines), train the stock learner,
> evaluate at the committed σ."*

**Reference for the wrapper pattern** RRLS (Robust RL Suite),
<https://arxiv.org/html/2406.08406v1> · SafeRL-Lab/Robust-RL-Baselines,
<https://github.com/SafeRL-Lab/Robust-RL-Baselines>

**Implementation** `urb_baselines/algos/dr_ippo.py` and
`urb_baselines/domain_random.py` · **script** `scripts/dr_ippo.py` · **config**
`config/algo_config/dr_ippo/config1.json`

---

## A. The specification, item by item

| id | item | status | where / why |
|---|---|---|---|
| **SPEC-1** | Resample σ **per episode** | **IMPLEMENTED** | `DomainRandomiser._reset` draws on every `env.reset()`, i.e. every day |
| **SPEC-2** | σ ∈ [0, 3] | **IMPLEMENTED** | `dr_range: [0.0, 3.0]`, overridable with `--dr-range` |
| **SPEC-3** | Train the **stock learner** | **IMPLEMENTED** | `DomainRandomisedIPPO` is `PerAgentPPOAlgorithm` with no override at all: byte for byte `scripts/ippo.py`'s learner, with the host block copied from `ippo/config1.json` |
| **SPEC-4** | The learner is **not told** σ | **IMPLEMENTED** | the observation is the stock one; nothing in this arm touches `context.py` |
| **SPEC-5** | Evaluate at the **committed** σ | **IMPLEMENTED** | `begin_test` calls `dr.fix(committed)`, where `committed` is the σ the run was launched with under `ns_launch.py` |
| **SPEC-6** | ≈ 10 lines, an environment change | **IMPLEMENTED** | `domain_random.py` is one wrapped `reset` |

---

## B. Two implementation decisions, both defensive

### WHERE-1 — the certified dial is not touched
B9 calls this "a NEW environment flag", which would naturally live in
`urb_ns/severity_traffic_env.py`. It does not, for one reason: that file is the
certified dial **every** arm runs under, PACT-1 included, and
`results/ns_certificate` is computed against it. Adding a randomisation branch
would mean every PACT-1 run from now on executes code written for a baseline, and
a reviewer would be right to ask what else changed.

The dial is therefore left byte-for-byte alone, and the randomisation is attached
to **one environment instance**, by the one script that wants it, at run time:
`SeverityTrafficEnvironment.reset()` resolves the day's weather by calling
`SeverityLayer.begin_episode`, which reads `layer.sigma`, so the only thing
domain randomisation has to do is set σ *before* that call.

### ZERO-1 — the zero floor, and why it is not a fudge
The severity wrapper short-circuits at `σ == 0`: `last()` returns before it
finalises the day, so `_ns_day` never advances and `_ns_final` is never reset.
That is correct for a run whose σ is 0 throughout, and **wrong for a run that
visits 0 for a single day**.

Draws are therefore floored at `sigma_floor = 1e-9`, where the capacity derate is
`1 − 1e-9·loss·A`, i.e. 1.0 to fifteen significant figures — numerically the same
task, on the bookkeeping path that a mid-run zero would fall off.

`fix(σ)` does **not** use the floor. It detaches the wrapper entirely and
restores the stock `reset`, so evaluating at `σ = 0` is byte-identical to a stock
run, which is what `NS_FORM_SPEC` B.1.1 requires.

---

## C. Gates

| gate | what it proves |
|---|---|
| **DR-1** | Over 200 resets: 200 draws, all in `(floor, hi]`, a sane mean, none on the zero short-circuit — and after `fix()` the randomiser stops drawing and σ is pinned. The two silent failures here are "it never redrew" and "it kept redrawing during the test phase", and both are asserted. |

---

## D. What to watch in a real run

- The banner states the training range, the floor and the evaluation σ.
- `sigma`, `n_draws` in the per-100-day line.
- The closing report gives min / mean / max of every draw and how many fell in
  the near-placebo tail (`σ < 0.05`), and prints a `***` block if **no** day was
  randomised — which would make the row plain IPPO under a different name.

**Note on launching.** The `--sigma` passed to `ns_launch.py` is the **committed**
severity used for the test phase; training resamples over `--dr-range` regardless.
So `run_baselines_sigma3.sh` gives "trained on U[0,3], evaluated at 3" and
`run_baselines_sigma0.sh` gives "trained on U[0,3], evaluated at 0" — which is
the pair B9 wants, because its prediction is about **both** ends:

> *"robustness costs σ=0 performance and still does not cancel; the asymptote lies
> between blind and PACT."*

The σ=0 row is where the cost shows up, and it is only meaningful because
`fix(0)` recovers the stock task exactly.

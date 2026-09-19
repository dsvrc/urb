# CHECKLIST — oracle-driver IPPO (the information arm) on URB

**BASELINES.md** Tier 1 item 4 (*"Oracle-driver blind ... and URB if the
observation change is cheap"*) · objection row *"Give the learner the same
information PACT has"*

**Implementation** `urb_baselines/algos/oracle_ippo.py` · **script**
`scripts/oracle_ippo.py` · **config** `config/algo_config/oracle_ippo/config1.json`

---

## A. What it is

URB's own IPPO — the learner of `scripts/ippo.py`, unchanged in every numerical
detail — with the exogenous driver `A(day)` appended to every observation.
Nothing else is added: no estimator, no channels, no constraint, no memory.

| id | item | status | where |
|---|---|---|---|
| **BASE-1** | The learner is URB's IPPO | **IMPLEMENTED** | `PerAgentPPOAlgorithm` with `observation()` overridden and nothing else |
| **BASE-2** | The host block is `ippo/config1.json` | **IMPLEMENTED** | verbatim |
| **CTX-1** | `A(day)` is appended to the observation | **IMPLEMENTED** | `OracleDriverIPPO.observation` |
| **CTX-2** | The context is only what B6 calls "publish A(t)" | **IMPLEMENTED** | `context_features: "a"`. `a_phase` additionally publishes the cycle phase and is a **declared ablation**, because it hands over the cycle *length* too, which A alone does not. |
| **CTX-3** | The run refuses to start without a driver | **IMPLEMENTED** | `--allow-no-driver` overrides, and every banner then says the arm is degenerate |

---

## B. Why it is not a citation

This arm is deliberately **not** a published method. It is the control for two
separate claims, and it is the only way to separate them:

**Against PACT-1.** If simply knowing the weather were enough, PACT's online
identification would be decoration. This arm gives the policy the driver and
nothing else, so the PACT-minus-oracle gap is the value of knowing *how much the
peers' load matters under that weather* — not of knowing the weather.

**Against LCPO.** `scripts/lcpo.py` also receives `A(day)` (it is LCPO's stated
requirement that the context be observed). This arm is that augmentation with
LCPO's machinery removed, so the LCPO-minus-oracle gap is the **local
constraint** rather than the context. Without this arm, an LCPO improvement could
be attributed to either.

It is therefore an upper bound on what an observed driver alone can buy, run
through the identical host.

---

## C. What is "observed" here, exactly

`A(day)` — rainfall intensity in `[0, 1]`, from `urb_baselines/context.py`. It is
published at σ = 0 as well: the weather happens, it just does not bite. That is
deliberate, so the σ=0 and σ=3 rows differ only in the dial.

What this arm does **not** get is the privileged vector: σ, the per-link rain
sensitivity and the identity of the worst-affected facility. That is
`scripts/rma.py`'s teacher, and the distinction is set out in
[CHECKLIST_rma.md](CHECKLIST_rma.md) PRIV-1.

---

## D. Gates

| gate | what it proves |
|---|---|
| **drive oracle_ippo** | Runs, learns, writes loss rows with the augmented observation. |

---

## E. What to watch in a real run

The banner prints the observation width before and after augmentation, and
whether the driver is live. If it says `degenerate (constant 0)`, the arm is
plain IPPO with a zero column and must not be reported as the information arm.

# CHECKLIST — WISDOM (wavelet predictive task representations), on URB

**BASELINES.md** B7/B8 — the **learned** competitor to PACT-1's declared basis.

`urls` (B10) asks whether the *basis* was necessary by regressing the residual on
raw peer actions. This asks the larger version of the same question: whether the
whole declared pipeline was necessary, using a method published for exactly this
problem. Both arms believe the same thing — that you cannot act well under drift
without an estimate of where the drift is going — and differ on one axis:

| | the estimate | the forecast |
|---|---|---|
| **PACT-1** | a declared basis (road classes, shared per-lane length), RLS with forgetting | an EMA over days |
| **WISDOM** | a learned latent task representation | wavelet coefficients at several scales, a TD operator that predicts the next representation, and an auto-regressive head |

**Paper** Wang, Li, He, Li, Bennis, Islam & Wang, *Wavelet Predictive
Representations for Non-Stationary Reinforcement Learning*,
<https://arxiv.org/abs/2510.04507> · **code**
<https://github.com/MinWangcs/WISDOM> (rlkit/CEMRL-derived; read `wisdom/`).

**Implementation** `urb_baselines/algos/wisdom.py` · **script**
`scripts/wisdom.py` · **config** `config/algo_config/wisdom/config1.json`

---

## A. The specification, item by item

| id | item | status | where / why |
|---|---|---|---|
| **ENC-1** | Context encoder `e_η(z | C)` over transition tuples `(s, a, s′, r)` | **ADAPTED** | `_Encoder` over `[obs, a_onehot, r]`. There is no `s′` **within** a URB episode — the day ends — so the successor information enters through the next day's tuple in the sequence, which is where the wavelet TD reads it |
| **ENC-2** | Variational information bottleneck: `J_η = E[KL(e_η(z|C) ‖ p(z))]` with a Gaussian prior | **IMPLEMENTED** | the `kl` term, weight `kl_coef = 0.1`; reparameterised sampling |
| **WAV-1** | Two learnable filters initialised to the Haar pair `y₀ = [1/√2, 1/√2]`, `y₁ = [1/√2, −1/√2]` | **IMPLEMENTED** | `WaveletNet`, grouped `Conv1d(kernel=2, stride=2)` per latent coordinate so the transform is of the **sequence**, not a mixing layer of width two. Gate **WAV-1** checks levels 1–2 against a hand-computed DWT (`1.9e-7`) and that energy is preserved |
| **WAV-2** | `M` levels of decomposition giving `u_M` and details `g_1..g_M` | **IMPLEMENTED** | `levels = 2`; iterated on the approximation branch |
| **WAV-3** | Selected details downsampled, combined with `u_M`, linearly mapped back | **ADAPTED** | the **most recent** coefficient of each band is taken (a causal downsample — the representation must be usable on the day it is built) and a `Linear` returns the stack to the latent space as `ẑ` |
| **TD-1** | Wavelet TD operator `F W(z_t) = z_t + Γ W(z_{t+1})`, with a target network | **IMPLEMENTED** | `td` in `_train()`; `Γ = wav_gamma·I = 0.99 I`; `wav_t` is polyak-updated at `polyak = 0.01`, the released value |
| **AR-1** | Auto-regressive term `−E[log Π_t P(ẑ_t | ẑ_{<t})]` | **ADAPTED** | a causal head `wav.ar` predicting the next representation, scored as a unit-variance Gaussian NLL, i.e. an MSE up to a constant. `--arm noar` drops it, which is the paper's own ablation |
| **POL-1** | The policy is conditioned on the representation | **IMPLEMENTED** | policy input is `[standardised observation, ẑ]` |
| **RL-1** | SAC | **ADAPTED** | URB is discrete and one step, so the base is URB's own on-policy learner (R2), `SingleStepPPO`, with the host block from `ippo/config1.json` |

---

## B. What URB forced

### SEQ-1 — the sequence is the sequence of DAYS
The paper's task-representation sequence runs along a trajectory. A URB episode
is **one step**, so a per-episode sequence has length one and every wavelet level
would be empty. The sequence is the agent's own history across days,
`seq_len = 16` of them — the same decision, for the same reason, as
[CHECKLIST_rippo.md](CHECKLIST_rippo.md) and [CHECKLIST_liam.md](CHECKLIST_liam.md).

The window ends **yesterday**: today's transition does not exist when the agent
has to act, so `ẑ` is a genuine one-day-ahead prediction, which is what a
*predictive* representation is for.

### SHARE-1 — the encoder and the wavelet network are shared; the actors are not
The paper is single-agent. Ninety-odd private sequence models, each seeing one
transition per day, would be ninety small-sample problems. Sharing is the choice
DGN's paper makes for the same reason, and it keeps this arm's cost in the same
range as the others. It is **unconditional** — there is no per-agent switch — and
one batched forward per day refreshes `ẑ` for the whole fleet.

### SCALE-1 — the context AND the policy observation must be standardised
URB pays `−travel_time` (~−1000) and its observation leads with a start time in
seconds (~1000). Two consequences, both measured:

1. **The encoder.** Fed raw, its capacity goes on two constants and the latent
   becomes an agent identifier — the LIAM failure mode measured in §4 of
   [README.md](README.md) (99.99% of the squared error was three constants).
2. **The policy.** A latent of order one beside a start time in seconds is below
   the network's resolution. On `fakeenv.FakeURB` the arm went from **−13 over
   200 days** (i.e. worse than its own start) to **+153** when the policy's
   observation was standardised as well; the encoder's context was already
   standardised in both runs. The same scale failure, one layer later.

`normalize_context` governs both, and `false` reproduces both. `z_std` in the
diagnostics is the number that shows the first; `RunningNorm` in
`urb_baselines/nets.py` is the shared, causal implementation.

### COST-1 — training cadence
`train_every = 4` days, `wav_batch = 32` agents per update, `wav_epochs = 1`. The
batch is over **agents**, each contributing one window. `enc_lr: null` resolves to
the host `lr`, so the representation is not given a learning rate the actors do
not have.

---

## C. Not applicable, and not implemented

| item | status | why |
|---|---|---|
| SAC's twin critics, target entropy, temperature | **N/A** | the host is actor-only and single-step; `target = reward` is URB's own convention |
| The paper's stochastic task-switching period `T_h ~ N(60, 20)` | **N/A** | URB-NS's drift is the certified dial `A(t)`, which every arm runs under. Injecting a second non-stationarity would make this arm face a different environment |
| Meta-World / SimGlucose / MuJoCo benchmarks | **N/A** | the host is URB |
| The CEMRL Gaussian-mixture encoder of the release | **SKIPPED** | the paper's own description is a Gaussian latent with a VIB term, and that is what is implemented; a mixture would add a component count nothing here selects |
| Per-agent encoders | **SKIPPED** | SHARE-1 |

---

## D. Arms

| arm | what it is |
|---|---|
| `wisdom` (default) | wavelet TD + auto-regressive head |
| `flat` | `levels = 0`: the representation is the raw latent, with the TD and AR heads still on top — isolates the **decomposition** |
| `noar` | wavelet TD without the AR term — the paper's own ablation |

---

## E. The gates

| gate | what it asserts |
|---|---|
| **WAV-1** | at initialisation the layer **is** a Haar DWT: levels 1–2 match a hand-computed transform of `[1, 3, 2, 8]` to `1.9e-7`, and energy is preserved (orthonormality). The filters are learnable, so if they do not *start* as a wavelet transform the arm is a convolution with the wrong name, and nothing downstream would say so |
| **drive wisdom**, **--arm flat** | 200 days on `fakeenv.FakeURB`: learns (−710 → −557), trains the representation 96 times, latent spread stays healthy (`z_std ≈ 4.0`, against `0.70` for `flat`) |
| **CFG-1 / CFG-2** | host block character-identical to `ippo/config1.json`; constructs and acts under its own shipped config |

---

## F. How to read a run

* `z_std` — **`***` if ~0**: the encoder collapsed and `ẑ` carries no task
  information. If `normalize_context` is false that is the documented cause.
* `detail` — the fraction of band energy in the **detail** coefficients.
  **`***` if ~0**: all the energy is in the approximation, the decomposition has
  collapsed into a moving average, and the multi-scale claim is not being
  exercised. On a city whose latent barely moves day to day that can be true of
  the *data* — say which.
* `wav_td`, `ar`, `kl`, `upd` — the three loss terms and the update count;
  **`***` if `upd` is 0**, because the policy then ran on a constant zero `ẑ`.

---

## G. What it tests

`PACT_THEORY` **C4** (excitation is bounded by policy indecision) and **C6**
(non-identifiability) apply to a learned encoder exactly as they do to RLS, and
nothing about a wavelet basis relieves them: once the fleet converges, the data
stops containing the variation that would separate "the coupling got stronger"
from "everyone moved".

The prediction is therefore split, and the run should be read for both halves:
WISDOM should track the **smooth component** of `A(t)` well — that is what a
multi-scale decomposition is for, and it is the paper's own result on gradually
evolving tasks — and it should **not** recover the per-route split, because that
is not identifiable from an agent's own reward stream at all.

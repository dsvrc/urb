# CHECKLIST — M3W (MoE world model + multi-agent MPPI), on URB

**BASELINES.md** B12 *model-based MARL* · objection row — B12 was the last class
still marked **"cite; run only if asked"**, with no arm on any host. This fills
it, and it makes B12's stated prediction falsifiable:

> *"a world model absorbs the drift into its latent but has no separate,
> invertible object for the coupling gain."*

**Paper** Zhao, Xu, Fu, Chai, Zhu & Zhao, *Learning and Planning Multi-Agent
Tasks via a MoE-based World Model*, NeurIPS 2025 ·
**code** <https://github.com/zhaozijie2022/m3w-marl> (read:
`m3w/models/world_models.py`, `m3w/runners/world_model_runner.py::plan`,
`configs/mujoco/.../config.json`) · the codebase extends HARL, TD-MPC2 and
Light-SoftMoE.

**Implementation** `urb_baselines/algos/m3w.py` · **script** `scripts/m3w.py` ·
**config** `config/algo_config/m3w/config1.json`

**Why it is the sharpest possible B12 arm.** On a one-step URB day M3W's world
model and PACT-1's estimator predict **the same quantity**. PACT-1 predicts the
delay on each candidate route from a declared basis with an RLS fit; M3W predicts
the same day's reward from a learned SparseMoE over a latent — and then, instead
of shifting logits by a trust-weighted z-score, it **plans**. The arm therefore
isolates two things at once, and its ablations separate them: the *model*
(declared vs learned) and the *use* of it (steer vs plan).

---

## A. The specification, item by item

| id | item | status | where / why |
|---|---|---|---|
| **WM-1** | Observation encoder to a latent with SimNorm | **IMPLEMENTED** | `SimNorm(simnorm_dim=8)` over a 128-dim latent, the release's values |
| **WM-2** | **SoftMoE** dynamics: every expert runs, gate is a softmax over all of them | **IMPLEMENTED** | `SoftMoE`, `num_dynamics_experts = 16` |
| **WM-3** | **SparseMoE** reward with noisy top-k routing | **IMPLEMENTED** | `SparseMoE`, the release's `NoisyTopKRouter`: `num_reward_experts = 16`, `top_k = 2`, softplus noise during training only |
| **WM-4** | Load-balancing term | **IMPLEMENTED** | squared coefficient of variation of the gate mass, weight `balance_coef = 0.0005` |
| **WM-5** | Two-hot reward coding, `num_bins = 101` on `[reward_min, reward_max]` | **IMPLEMENTED** | `TwoHot`; round-trips to `< 1e-4` (gate **M3W-1**). What is *coded* is adapted — see SCALE-1 |
| **WM-6** | Losses: `dynamics_coef·dyn + reward_coef·rew + balance_coef·balance` | **IMPLEMENTED** | `20 · MSE(latent) + 0.1 · CE(two-hot) + 0.0005 · balance`, the release's coefficients |
| **PL-1** | MPPI: sample, score under the model, keep `num_elites`, reweight by `exp(temperature·(v − max v))`, refit, repeat `iterations` | **IMPLEMENTED** | `_mppi()`; `num_samples = 128`, `num_elites = 16`, `iterations = 6`, `temperature = 0.5`, all the release's plan block |
| **PL-2** | `num_pi_trajs` samples seeded from a policy | **IMPLEMENTED** | 8 samples seeded from the host DQN's greedy action — the release's policy-guided half |
| **PL-3** | A joint action is scored by the **mean** of the per-agent returns | **IMPLEMENTED** | `vals = r.mean(dim=1)`, matching `torch.mean(torch.stack(g_returns))` in the release |
| **PL-4** | The final action is drawn from the elite mixture | **IMPLEMENTED** | the refitted per-agent categorical |
| **WARM-1** | `warmup_train` before planning | **IMPLEMENTED** | `warmup_days = 100`; before that the host's own DQN acts, because a plan under an untrained world model is noise |

---

## B. What URB forced — five rows

### HORIZON-1 — `horizon = 1` (the release plans 3 steps)
A URB day is one step and the episode ends, so a horizon above one is a forecast
over **days**, not a rollout within an episode. The dynamics model is kept and
does exactly that: it predicts the next **day's** latent from yesterday's latent
and today's action — a real and interesting object on a drifting instance — and
`--horizon H` rolls it forward `H` days inside the planner.

### DISCRETE-1 — categorical MPPI
The release samples from a Gaussian and clamps to `[-1, 1]`. URB's action is one
of K routes, so the sampling distribution is a per-agent categorical, the MPPI
refit is the elite-weighted empirical distribution, and `min_prob = 0.02` is the
floor that plays the role of the release's `min_std`. This is the standard
discrete instantiation of the same update, not a different planner.

### SNAP-1 — the snapshot is YESTERDAY, then each agent refines its own route
Planning needs a joint action, and a URB day is **sequential** — there is no
instant at which every agent's current observation exists (§4 of
[README.md](README.md)). So:

* at `begin_episode` the joint plan is computed from each agent's **last**
  observation;
* at its own turn each agent re-scores only **its own** K routes with **today's
  real** observation, peers held at the plan, and draws from the MPPI-weighted
  result.

Both halves are the receding-horizon idea. The first is the joint plan the method
requires; the second is what stops the arm from discarding the observation URB
actually gives it.

### GRAPH-1 — the coupling enters through the neighbour set
A plan cannot beat a policy unless the reward model knows what the peers chose.
Each candidate is scored with the **route histogram of the agent's neighbours**
under the sampled joint action, from `urb_baselines/graph.py` with
`graph = overlap`, `n_neighbors = 3` — the same graph DGN uses, built the same
way. Feature = `[z, a_onehot, neighbour histogram]`.

### SCALE-1 — the reward must be standardised before the two-hot **(the silent one)**
This is the **fourth** member of the family in §4 of [README.md](README.md), and
the most silent of them.

The release codes rewards two-hot over 101 bins spanning `[-10, 10]`. URB pays
about **−1000**. Measured in gate **M3W-1**: rewards of −1200, −1000, −800 and
−600 produce **identical** codes, all mass in bin 0, `clipped_fraction = 1.0`.
The consequences chain:

    every reward → bin 0  ⇒  the reward model is a constant
                          ⇒  every sampled joint action scores the same
                          ⇒  MPPI's softmax over identical values is uniform
                          ⇒  the planner picks routes at random

…and the loss curve looks perfectly healthy throughout, because a constant target
is easy to fit. `normalize_reward: true` standardises the reward before the
two-hot (support and bin count unchanged, now in standardised units, spread 0.94
in the same gate); `false` reproduces the failure exactly.

The two diagnostics that catch it are `plan_spread` (max − min of the MPPI values
over the sampled joint actions) and `clip` (the clipped fraction), both printed
every `print_every` days.

---

## C. Not applicable, and not implemented

| item | status | why |
|---|---|---|
| `γ · Q(z_H, a_H)` terminal bootstrap in `estimate_value` | **N/A** | one-step episode; URB's own IQL uses `target = reward` and every off-policy arm here does the same |
| The **multi-task** half of the paper (14 Bi-DexHands / 24 MA-MuJoCo tasks) | **N/A** | M3W's mixtures exist to reuse knowledge across *tasks*. URB is one task whose dynamics drift, so the experts specialise across **regimes** of the dial rather than across tasks. That is the honest reading and the paper should state it rather than claiming a multi-task result |
| The release's separate actor/critic (`world_model_actor.py`, `world_model_critic.py`) | **ADAPTED** | the seed policy is URB's own DQN (R2); the planner is the actor, which is the paper's own "without relying directly on an explicit policy model" |
| `use_twohot: false` path, `scale_tau`, `step_rho`, `wt_steps` | **SKIPPED** | release-side training-schedule controls tied to its SB3/HARL loop; URB's day loop is the host's |
| Bi-DexHands / MA-MuJoCo environments | **N/A** | the host is URB |

---

## D. Arms

| arm | what it is |
|---|---|
| `m3w` (default) | SoftMoE dynamics + SparseMoE reward + MPPI |
| `mlp` | the same planner over a plain MLP world model — isolates the **mixture of experts** |
| `greedy` | the MoE model with an argmax instead of MPPI — isolates the **planner** |

---

## E. The gates

| gate | what it asserts |
|---|---|
| **M3W-1** | the two-hot code is a distribution and round-trips to `< 1e-4`; URB's raw rewards all collapse into bin 0 with **identical** codes and `clipped_fraction = 1.0`; the standardised ones spread by 0.94 |
| **drive m3w**, **--arm mlp** | 200 days on `fakeenv.FakeURB`: learns (−639 → −551), trains the world model, plans 181 days, and `plan_spread > 0` — i.e. the planner is discriminating between joint actions |
| **GRAPH-1** | unchanged: the neighbour graph is deterministic and never crosses an OD in `od` mode |
| **CFG-1 / CFG-2** | host block character-identical to `iql/config1.json`; constructs and acts under its own shipped config |

---

## F. How to read a run

* `wm_upd`, `plans` — **`***` if either is zero**: the run never left the warm-up,
  or the planner scored every route under a randomly initialised network.
* `plan_spread` — **`***` if ~0**: every sampled joint action scores the same, so
  MPPI is a uniform draw and **the planner is choosing at random** while the loss
  curve looks healthy. If `clip` is near 1, SCALE-1 is the cause.
* `clip` — the fraction of rewards outside the two-hot support.
* **final plan entropy** against `log K` — a plan that stayed at uniform entropy
  planned nothing.

---

## G. Cost, stated rather than guessed

Planning is `num_samples × n_av` reward-model forwards per MPPI iteration:
`128 × 88 × 6 ≈ 68k` rows per day on `saint_arnoult`, plus the per-agent
refinement. That is the heaviest algorithm in this package by a clear margin and
it has **not** been measured on `saint_arnoult` yet — the cost table in
[README.md](README.md) §2 covers the original twelve only. Take the first
`d/min` line of a real run as the estimate, and if it is out of proportion,
`num_samples` and `iterations` are the two knobs that trade planning quality for
time, in that order.

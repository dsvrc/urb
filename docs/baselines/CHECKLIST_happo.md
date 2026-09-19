# CHECKLIST — HAPPO (ICLR 2022 / JMLR 2024) on URB

**Papers** Kuba et al., *Trust Region Policy Optimisation in Multi-Agent
Reinforcement Learning*, ICLR 2022 · Zhong et al., *Heterogeneous-Agent
Reinforcement Learning*, JMLR 2024, <https://arxiv.org/abs/2304.09870>

**Code read** <https://github.com/PKU-MARL/HARL> (local copy `HARL-main_corep/`)
- `harl/algorithms/actors/happo.py` — `update`, `train`
- `harl/runners/on_policy_ha_runner.py` — the sequential loop and the `factor`
- `harl/algorithms/critics/v_critic.py` — `cal_value_loss`, `update`
- `harl/common/valuenorm.py` — `ValueNorm`
- `harl/models/value_function_models/v_net.py`, `harl/models/base/mlp.py` — the
  model block (feature normalisation, orthogonal init, output gain)
- `harl/utils/models_tools.py` — `huber_loss`
- `harl/configs/algos_cfgs/happo.yaml` — every default quoted below

**BASELINES.md** B1 · Tier 2 · objection row *"Trust-region / sequential-update
MARL already handles non-stationarity"*

**Implementation** `urb_baselines/algos/happo.py` · **script** `scripts/happo.py`
· **config** `config/algo_config/happo/config1.json`

---

## A. The method, item by item

| id | item | source | status | where / why |
|---|---|---|---|---|
| **SEQ-1** | Agents are updated **one at a time**, not simultaneously | `on_policy_ha_runner.train` | **IMPLEMENTED** | `HAPPO.learn`, the `for i in order:` loop |
| **SEQ-2** | The order is a fresh random permutation each iteration (`fixed_order: False`) | `happo.yaml`; runner | **IMPLEMENTED** | `self.rng.permutation(self.n_av)`; `--fixed-order` switches to index order |
| **SEQ-3** | `factor = Π_{j updated} π_j^new(a_j)/π_j^old(a_j)`, evaluated on the same data | `on_policy_ha_runner.train` | **IMPLEMENTED** | `factor = factor * exp(lp_after − lp_before)` after each agent |
| **SEQ-4** | `factor` multiplies the clipped surrogate of the next agent | `happo.py:update` | **IMPLEMENTED** | `pg = -(factor[idx] * min(surr1, surr2)).mean()` |
| **SEQ-5** | `old_log_probs` are re-evaluated per agent **before** its update | runner | **IMPLEMENTED** | `lp_before` inside the loop |
| **CRIT-1** | Centralised V critic on a state the agent does not see | `v_critic.py` | **IMPLEMENTED** | input = concatenation of every machine agent's observation, matching `scripts/mappo_torchrl.py`'s `MultiAgentMLP(centralised=True)` on URB |
| **CRIT-2** | State type `FP` ⇒ per-agent value and per-agent advantage | runner, `advantages[:, :, agent_id]` | **IMPLEMENTED** | critic outputs `n_av` values; `adv[:, i]` per agent. See ADAPT-1 for why not `EP`. |
| **CRIT-3** | Advantage normalised over the batch | runner (`FP` branch) | **IMPLEMENTED** | `(adv − mean)/(std + 1e-5)` |
| **CRIT-4** | `ValueNorm`: exponentially debiased running mean/var; returns normalised, value predictions denormalised before subtraction | `valuenorm.py`; runner | **IMPLEMENTED** | class `ValueNorm`; `v_den = denormalize(v_pred)` |
| **CRIT-5** | Clipped value loss: `max(L(orig), L(clipped))` with the clip around the old prediction | `v_critic.cal_value_loss` | **IMPLEMENTED** | same expression |
| **CRIT-6** | Huber loss, `huber_delta = 10.0` | `models_tools.huber_loss`; `happo.yaml` | **IMPLEMENTED (verbatim)** | `huber_loss(e, d)` reproduced line for line |
| **CRIT-7** | `critic_epoch = 5`, `critic_num_mini_batch = 1` | `happo.yaml` | **IMPLEMENTED** | config |
| **PPO-1** | Clipped surrogate, `clip_param = 0.2` | `happo.py:update` | **IMPLEMENTED** | URB's `clip_eps` is also 0.2, so host and paper agree |
| **PPO-2** | Entropy bonus with `entropy_coef` | `happo.py:update` | **IMPLEMENTED** | host's `entropy_coef` (0.01) — identical to `happo.yaml`'s |
| **PPO-3** | `max_grad_norm = 10.0` | `happo.yaml` | **IMPLEMENTED** | a paper hyperparameter with no URB equivalent, so R2 takes it from the paper |
| **PPO-4** | `action_aggregation: prod` over action dimensions | `happo.yaml`; `happo.py` | **N/A** | URB's action is a single discrete choice; a product over one dimension is the identity |
| **PPO-5** | `share_param: False` ⇒ one actor per agent | `happo.yaml` | **IMPLEMENTED** | one `HARLMLP` per machine agent, which is also how `scripts/ippo.py` is built |
| **NET-1** | `use_feature_normalization: True` (LayerNorm on the input) | `happo.yaml`; `models/base/mlp.py` | **IMPLEMENTED — and load-bearing** | see ADAPT-2 |
| **NET-2** | `initialization_method: orthogonal_` | `happo.yaml` | **IMPLEMENTED** | `HARLMLP`, orthogonal weights, zero biases |
| **NET-3** | `gain: 0.01` on the actor's output layer | `happo.yaml` | **IMPLEMENTED** | `HARLMLP(..., gain=0.01)`; the critic uses gain 1.0, as `v_net.py` does |
| **NET-4** | Hidden sizes `[128, 128]` | `happo.yaml` | **ADAPTED** | URB's shared `widths: [64, 64]` is used instead. R2: network size is a host hyperparameter, and giving one arm a wider network than the others would be a confound. |
| **GAE-1** | GAE with `gamma = 0.99`, `gae_lambda = 0.95` | `happo.yaml` | **N/A (collapses exactly)** | one-step terminal episode ⇒ `returns = r`, `adv = r − V(s)` for any γ, λ |
| **RNN-1** | Optional recurrent policy, `data_chunk_length = 10` | `happo.yaml` | **N/A here** | `happo.yaml`'s default is `use_recurrent_policy: False`. The memory question has its own arm, `scripts/rippo.py`, which uses R-MAPPO's recurrent machinery. |
| **LR-1** | `use_linear_lr_decay: False` | `happo.yaml` | **IMPLEMENTED** | no decay, matching the default |
| **OPT-1** | Adam with `opti_eps = 1e-5`, `weight_decay = 0` | `happo.yaml` | **IMPLEMENTED** | `optim.Adam(..., eps=1e-5)` |

---

## B. Adaptations, in full

### ADAPT-1 — `FP`, not `EP`
HARL supports two state types. `EP` uses one shared `V(s)` and one shared
advantage series, which presumes a **common reward**. URB's task config runs
`av_behavior: selfish`, so each traveller is paid its own negative travel time.
Collapsing that to a fleet mean would silently change this arm's objective to
social welfare while every other arm optimised selfishly — a different problem,
not a different method.

`FP` keeps each traveller's own reward and advantage and changes only what the
*critic* is allowed to see. It is supported by the reference implementation
(`advantages[:, :, agent_id]` in `on_policy_ha_runner.train`), and it is the
honest reading of "centralised training, decentralised execution" for a
selfish-reward benchmark.

The centralised state is the concatenation of every machine agent's observation,
which is exactly what `scripts/mappo_torchrl.py` already gives MAPPO on URB, so
the two centralised arms see the same object.

### ADAPT-2 — the model block is HAPPO's, and that is not cosmetic
`happo.yaml`'s `model` block specifies feature normalisation, orthogonal
initialisation and an output gain of 0.01. Without the first two, HAPPO does not
train on URB **at all**, and the failure is silent.

URB's observation is `[start_time_in_seconds, four route counts]`: the first
coordinate is of order 1000 and the rest of order 1. Orthogonal weights are
norm-preserving, so out of initialisation the logits are of order 1000, the
softmax is a point mass, the entropy is 0 and the policy gradient is exactly
zero. Measured on `urb_baselines/fakeenv.py` before the fix:

```
  day    T    d|actor|   d|critic|      pg    factor   ratio
   15   16   7.437e-15   2.441e-03   1.2e-09  1.0000   1.0000
  191   16   5.778e-15   2.781e-04   9.0e-09  1.0000   1.0000
final policy of agent 0 on one state: [0. 1. 0. 0.]
```

The actor moved by 1e-15 per update for the whole run and ended as a hard point
mass; the critic trained normally, so every curve looked plausible. With the
`happo.yaml` model block in place the same run gives `factor 1.054`,
`ratio 1.0009`, and the reward improves.

URB's own IPPO does not need this because PyTorch's default initialisation
scales weights down with fan-in. This is therefore HAPPO's own reference
configuration rather than a tuning choice, and it is the **only** place in this
package where an arm's network differs from the host's. Selftest gate
**drive happo** asserts `factor ≠ 1`, so a regression here cannot pass silently.

### ADAPT-3 — collection length and epochs
`episode_length` (how many days are collected per training iteration) defaults to
the host's `batch_size` = 64, so HAPPO updates on exactly as much fresh data as
URB's IPPO does. `ppo_epoch` defaults to the host's `num_epochs` = 3 rather than
`happo.yaml`'s 5, for the same reason: the number of passes over the data is a
host hyperparameter under rule R2. Both are exposed in the config and printed.

---

## C. Nothing was skipped

Every mechanism in HAPPO is present. `action_aggregation`, GAE and the recurrent
option are marked N/A because the concept does not exist on a one-step,
single-discrete-action episode — not because they were left out.

---

## D. Gates

| gate | what it proves |
|---|---|
| **drive happo** | The arm runs, learns, and the sequential `factor` leaves 1 — i.e. the actors are actually moving. This is the gate that catches the NET-1 failure above. |

---

## E. What to watch in a real run

`pg` is the mean clipped surrogate and is ≈ 0 by construction once advantages are
standardised — it is not a sign of trouble. The two to read are:

- `factor` — if it stays at exactly 1.0000 for the whole run, no actor moved and
  the arm is broken (see ADAPT-2).
- `ratio` — the mean importance weight; it should sit slightly off 1.

**Prediction under test** (BASELINES.md B1): *"they stabilise learning and
plateau at the same asymptote as MAPPO/MATD3 under the severity parameter"* — the
comparison is against `mappo_torchrl` and `ippo` from
`scripts/sweep/run_sweep.sh`, at the same σ and seed.

# CHECKLIST — LIAM (NeurIPS 2021) on URB

**Paper** Papoudakis, Christianos, Albrecht, *Agent Modelling under Partial
Observability for Deep Reinforcement Learning*, NeurIPS 2021.
<https://proceedings.neurips.cc/paper_files/paper/2021/file/a03caec56cd82478bf197475b48c05f9-Paper.pdf>

**Code read** <https://github.com/uoe-agents/LIAM>
- `lb_foraging/models.py` — `Encoder`, `Decoder`, `PolicyNet`
- `lb_foraging/agent.py` — `class A2C`: `compute_embedding`, `act`, `evaluate`,
  `eval_decoding`, `update` (the two optimisers)
- `lb_foraging/storage.py` — `compute_returns` (return standardisation)
- `lb_foraging/standardise_stream.py` — `RunningMeanStd`
- `lb_foraging/run_tests.py` — the hyperparameters actually used

**BASELINES.md** B5 · objection row *"Model the other agents"*

**Implementation** `urb_baselines/algos/liam.py` · **script** `scripts/liam.py` ·
**config** `config/algo_config/liam/config1.json`

B5 assigns LIAM to the MAPDN host as a Tier-2 item. It is implemented here
because URB is the cell this repository runs, and because the port is the same
size either way: LIAM's controlled agent uses only its **own** observations and
actions, which is exactly what a URB traveller has.

---

## A. The method, item by item

| id | item | source | status | where / why |
|---|---|---|---|---|
| **ENC-1** | `Encoder`: LSTM → ReLU MLP → linear embedding | `models.Encoder` | **IMPLEMENTED (verbatim)** | `_Encoder` |
| **ENC-2** | The encoder's input is `(o_t, a_{t−1})` — the controlled agent's own observation and its previous action | `agent.compute_embedding` | **IMPLEMENTED** | `torch.cat([obs, prev_onehot])` |
| **ENC-3** | `embedding_dim = 20` | `run_tests.py` | **IMPLEMENTED** | `embedding_dim: 20` |
| **POL-1** | `PolicyNet`: shared 2-layer trunk, softmax policy head and scalar value head | `models.PolicyNet` | **IMPLEMENTED (verbatim)** | `_PolicyNet` |
| **POL-2** | The policy input is `[o_t | m_t]` | `agent.act` | **IMPLEMENTED** | `torch.cat([obs, emb])` |
| **POL-3** | The embedding is **detached** into the policy | `agent.evaluate`: `torch.cat((obs, embeddings.detach()), dim=-1)` | **IMPLEMENTED** | `emb_flat.detach()` unless `backprop_embeddings` |
| **POL-4** | `backprop_embeddings` is a flag, default `False` | `run_tests.py` | **IMPLEMENTED** | config, default `false` |
| **DEC-1** | `Decoder`: **two separate** 2-layer branches, one per reconstruction target | `models.Decoder` | **IMPLEMENTED (verbatim)** | `_Decoder` — note the branches do **not** share a trunk, which the reference is explicit about |
| **DEC-2** | Observation head: raw output, loss `0.5·Σ(o − ô)²` | `agent.eval_decoding` | **IMPLEMENTED (verbatim)** | `rec_o` |
| **DEC-3** | Action head: softmax, loss `−log(Σ p·onehot)` (cross-entropy against the true action) | `agent.eval_decoding` | **IMPLEMENTED (verbatim)** | `rec_a` |
| **OPT-1** | **Two optimisers**: `opt1` on the actor-critic, `opt2` on encoder+decoder | `agent.__init__`, `agent.update` | **IMPLEMENTED (verbatim)** | `opt1`, `opt2`; `loss1.backward()` then `loss2.backward()`, then both step |
| **OPT-2** | `lr1 = 3e-4` (actor-critic), `lr2 = 7e-4` (encoder+decoder) | `run_tests.py` | **IMPLEMENTED / ADAPTED** | `lr2 = encoder_lr: 7e-4` from the paper (no URB equivalent); `lr1` is URB's shared `lr`, which happens to be 3e-4 as well |
| **OPT-3** | Gradient clipping at `max_grad_norm = 0.5`, on all three networks separately | `agent.update`; `run_tests.py` | **IMPLEMENTED (verbatim)** | three `clip_grad_norm_` calls |
| **LOSS-1** | `loss1 = mean[(1−done)(value_loss + action_loss − entropy·coef)]` | `agent.update` | **IMPLEMENTED** | the `(1−done)` mask is N/A on a one-step episode (every step is terminal, and the mask in the reference exists to skip padded steps in a rollout) |
| **LOSS-2** | `loss2 = mean[(1−done)(rec_loss1 + rec_loss2)]` | `agent.update` | **IMPLEMENTED** | `(rec_o + rec_a.sum(-1)).mean()` |
| **RET-1** | Returns are standardised by a running mean/var; value predictions are **de**-standardised before the GAE | `storage.compute_returns` + `standardise_stream.py` | **IMPLEMENTED** | `_RunningMeanStd`; on a one-step episode the GAE collapses so only the standardisation survives, and it is load-bearing (URB pays −1000) |
| **GAE-1** | GAE with `gamma = 0.99`, `gae_lambda = 0.95` | `run_tests.py` | **N/A (collapses exactly)** | one-step terminal ⇒ `returns = r`, `adv = r − V(s)` |
| **ENT-1** | `entropy_coef = 0.001` | `run_tests.py` | **ADAPTED** | URB's shared `entropy_coef` (0.01). R2: exploration pressure is a host hyperparameter and must match the arms this is compared to. |
| **HID-1** | `hidden_dim1 = 128` | `run_tests.py` | **ADAPTED** | URB's shared width (64). R2. |
| **MOD-1** | The modelled set is the other agents in the task (1–3 in MPE / LBF) | paper §4; `opp_obs_dim`, `opp_act_dim` | **ADAPTED** | the 3 nearest **machine** peers by departure time within the coupling graph (`graph: overlap`), NOT the same-OD set. See ADAPT-1. |
| **SEQ-1** | The LSTM runs over the steps of an episode | `agent.act`, `storage` | **ADAPTED** | over **days**. Same decision, same reason, as `scripts/rippo.py`: a URB episode is one step, so a per-episode LSTM has nothing to encode. |

---

## B. Adaptations, in full

### ADAPT-1 — who is modelled
LIAM models 1–3 other agents. URB has hundreds of travellers, and **only the
machine agents have an observation at all** (human travellers in RouteRL have an
action but no observation vector). The modelled set is therefore the `m` nearest
machine peers by departure time, with `m = 3` — LIAM's largest setting.
Reconstructing all 87 peers would be a different method with a different name and
a decoder two orders of magnitude larger than the policy.

**Nearest within which structure?** Not the OD pair. Measured on the seven
networks URB ships, the median OD pair carries exactly **one** traveller (the
table is in [CHECKLIST_mfq.md](CHECKLIST_mfq.md) FIELD-1), so a same-OD modelled
set is almost entirely padding and the decoder ends up reconstructing the
agent's own observation three times over — a target the encoder already has, and
therefore no model of anyone. The default is `graph: overlap`: the coupling
graph, where two travellers are neighbours when their OD pairs' route sets share
a link **and** their trips are co-present. It needs the generated route table; if
none is found it falls back to `copresence` and says so.

Where a city still does not supply `m` peers, the slot is padded with the agent
itself. The banner reports how many slots were padded, and `close()` prints a
`***` block if more than half of them are — so "LIAM modelled nobody" cannot be
mistaken for "agent modelling did not help".

### SCALE-1 — the reconstruction target is standardised, and without it the decoder learns three constants
`DEC-2`'s loss is an unweighted squared error over the modelled agents'
observations. URB's observation is `[start_time_in_seconds, four counts]`, and
**start time is of order 1000 and constant for a given traveller**. Measured on
`urb_baselines/fakeenv.py` with the raw target:

```
  rec_o = 6.5e+06        (0.5 * SSE, dominated entirely by start times)
  rec_a = 2.46           (the action head, at/above chance = log 4 = 1.386)
```

99.99% of the squared error is memorising three constants, the decoder never
looks at the counts, and the embedding carries no model of anyone. After
standardising per observation dimension across the fleet:

```
  rec_o = 6.15
  rec_a = 2.44
```

LIAM's own environments have roughly unit-scale observations, so this restores
the regime its loss was written for. `standardize_targets: false` reproduces the
degenerate behaviour, and the drive gate asserts `rec_o < 1e4` so a regression
cannot pass silently.

---

## C. Nothing was skipped

Every module, both losses, both optimisers, the detachment and the return
standardisation are present. The items marked N/A (GAE-1, the `(1−done)` mask)
are ones that provably collapse on a one-step episode.

---

## D. Gates

| gate | what it proves |
|---|---|
| **drive liam** | The arm runs, learns, writes loss rows, and the observation reconstruction loss is in a sane range — i.e. SCALE-1 has not regressed. |

---

## E. What to watch in a real run

- `rec_a` against `log K = 1.386` — the action reconstruction cross-entropy
  versus chance. **If `rec_a` never falls below chance, the embedding contains no
  model of the peers** and the arm is A2C with a random feature appended.
  `close()` prints a `***` block saying exactly that. This is the single most
  important number for this baseline: it is the difference between "agent
  modelling did not help" and "agent modelling did not happen".
- `rec_o` — the observation reconstruction error, in standardised units.
- `pg`, `v` — the actor-critic terms.

**Prediction under test** (BASELINES.md B5): *"it models who the peers are
(their policies), which at σ=0 is stationary; it does not model how much they
matter, which is what drifts."* So the expected signature is `rec_a` falling
below chance — the model works — while the return does not recover under the
dial.

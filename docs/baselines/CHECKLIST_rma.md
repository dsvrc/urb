# CHECKLIST — RMA / UP-OSI (the system-identification arm) on URB

**Papers**
- Kumar, Fu, Pathak, Malik, *RMA: Rapid Motor Adaptation for Legged Robots*,
  RSS 2021. <https://www.roboticsproceedings.org/rss17/p011.html>
- Yu, Tan, Liu, Turk, *Preparing for the Unknown: Learning a Universal Policy
  with Online System Identification* (UP-OSI), RSS 2017.
  <https://arxiv.org/abs/1702.02453>

**BASELINES.md** B8 · Tier 2 · objection row *"Meta-RL / online system
identification"*

**Implementation** `urb_baselines/algos/rma.py` · **script** `scripts/rma.py` ·
**config** `config/algo_config/rma/config1.json`

---

## Note on status

B8 specifies this one as an **in-house implementation**, not a port:

> *"Chosen: an RMA/UP-OSI-style adaptation module implemented in-house (teacher
> policy conditioned on the true β\*(t)·A(t) — available inside the environment —
> then a history encoder distilled to predict it; ≈ 1 day in the existing
> hosts)."*

So the checklist below is against the two papers' **structure**, which is what
"RMA/UP-OSI-style" commits to, rather than against a released file. Where a
paper's own constant exists it is used and cited.

B8 also says why this arm matters: it is *"the closest methodological neighbour
of PACT-1: it also conditions on an identified quantity, but the identifier is a
learned history encoder rather than an estimator on a declared basis."*

---

## A. The method, item by item

| id | item | source | status | where / why |
|---|---|---|---|---|
| **P1-1** | Phase 1: an environment-factor encoder `μ` maps the privileged environment vector `e_t` to a latent `z_t` | RMA §4, Fig. 2 | **IMPLEMENTED** | `FactorEncoder` |
| **P1-2** | The base policy `π(o_t, z_t)` is trained end to end **with** `μ` by RL | RMA §4.1 | **IMPLEMENTED** | `z` is appended to the observation; `μ` is a live module during phase 1 |
| **P1-3** | `z` has dimension 8 | RMA §4.1 | **IMPLEMENTED** | `z_dim: 8` |
| **P1-4** | `μ` is a 3-layer MLP (256, 128) | RMA §4.1 | **ADAPTED** | URB's shared `widths: [64, 64]`. R2: network size is a host hyperparameter. |
| **P1-5** | `e_t` is a quantity the agent provably cannot observe | RMA §4.1 (mass, friction, payload, motor strength) | **IMPLEMENTED** | see PRIV-1 |
| **P2-1** | Phase 2: the base policy and `μ` are **frozen** | RMA §4.2 | **IMPLEMENTED** | `_enter_phase2` sets `requires_grad_(False)` on both and clears the PPO memories so no stale on-policy data can leak into a later update |
| **P2-2** | An adaptation module `φ` predicts `ẑ_t` from the agent's recent history | RMA §4.2; UP-OSI §4 | **IMPLEMENTED** | `AdaptationModule` |
| **P2-3** | `φ` is trained by supervised regression `‖ẑ − z‖²` | RMA §4.2, Eq. 2 | **IMPLEMENTED** | `_train_phi`, `F.mse_loss(self.phi(h), z)` |
| **P2-4** | `φ` is a 3-layer 1-D CNN over the history, after a linear projection, then a flatten and a linear head | RMA §4.2 | **IMPLEMENTED** | `Linear → 3×Conv1d(k=3) → Linear` |
| **P2-5** | The history is the last `k` steps | RMA §4.2 (k = 50) | **IMPLEMENTED** | `history_len: 50` |
| **P2-6** | Phase-2 data is collected **on policy** with `ẑ` | RMA §4.2; UP-OSI's DAgger iteration §4.2 | **IMPLEMENTED** | `phase2_act: student` (default). `teacher` keeps the privileged latent during phase 2, as the ablation. |
| **DEP-1** | Deployment uses `ẑ` only; the privileged vector is never used again | RMA §4.2 | **IMPLEMENTED** | `_use_student()` returns True for the whole test phase |
| **HIST-1** | The history is `(x_{t−k:t−1}, a_{t−k:t−1})` — state and action | RMA §4.2 | **ADAPTED** | `(o, onehot(a), r_standardised)`. See ADAPT-1. |
| **UP-1** | UP-OSI: the universal policy is conditioned on the model parameters `μ`, and OSI regresses them from a recent history window | UP-OSI §4.1–4.2 | **IMPLEMENTED** | the same two objects; RMA's latent encoding of `e` is the "universal policy" conditioning, and `φ` is OSI |
| **UP-2** | UP-OSI iterates: retrain OSI on data from the current UP-OSI policy | UP-OSI §4.3 | **IMPLEMENTED** | `phase2_act: student` is exactly this, run continuously rather than in outer rounds |
| **BASE-1** | The RL learner | RMA uses PPO | **IMPLEMENTED** | URB's own on-policy learner, `SingleStepPPO`, unchanged |
| **PHASE-1** | The split of the training budget between the two phases | — | **DECLARED** | `phase1_frac: 0.5`. Neither paper has a shared budget to copy (they are separate training runs), so the split is declared here and printed in the banner. |

---

## B. What is privileged, precisely

### PRIV-1 — the environment vector
`e_t = [A(t), 1 − mean_e g_e(t), 1 − min_e g_e(t)]` — the rainfall intensity and
the realised capacity derating of the road network, from
`urb_baselines/context.py`.

It is privileged for three independent reasons:
1. **σ is not published.** A traveller cannot observe how much the weather bites.
2. **The per-link rain sensitivity is a property of the road network**, not of
   the trip — it is read from the live severity layer, not reconstructed.
3. **`min_e g_e` names a facility** — the worst-affected link — which no
   traveller's observation contains.

It is legitimate rather than hindsight because all three are functions of the day
alone and are therefore known **before anyone acts** — the same status as mass and
friction in RMA.

**At σ = 0 it collapses to `[A, 0, 0]`,** because the dial is provably off. RMA's
teacher then knows nothing the oracle-driver arm does not, and RMA degenerates to
that arm. This is the correct behaviour under a dial that does nothing, and
`close()` prints a note saying so rather than leaving the reader to wonder.

This arm is **not** `scripts/oracle_ippo.py`: that one is given the *observed*
driver `A(t)` forever; this one is given a strictly larger, unobservable quantity
but only during phase 1, and must recover it from its own history thereafter.

---

## C. The one adaptation that matters

### HIST-1 — the history includes the realised reward
RMA's history is proprioceptive state and action. That works because a legged
robot's proprioception **encodes how the plant responded** — the joint positions
after a step tell you about the friction that step met.

URB's observation does not. It is `[start_time, counts of earlier same-OD
travellers' routes]`: a departure time that never changes and a count of what
other people did. It contains **no travel time at all**. A history of
observations and actions therefore carries almost no information about the
weather, and `φ` would be identifying from nothing.

The realised reward (the negative travel time) is the only channel through which
the drift is observable to a traveller at all, so the history is
`(o, onehot(a), r_standardised)`. The reward is standardised by a running
mean/variance so the CNN's input is not dominated by a −1000-scale coordinate.

`history_features: "oa"` drops the reward and reproduces RMA's literal input.
**Run it as the ablation** — it is the arm that shows why the choice was made,
and it is available by hand as `scripts/rma.py --history-features oa`. The sweep runs one arm per baseline and does not launch it.

---

## D. Gates

| gate | what it proves |
|---|---|
| **drive rma** | Phase 2 is actually reached, `φ` is actually trained, and `phi_mse` falls. The two ways this arm can silently be something else are "phase 2 never started" (it is then the privileged teacher, an oracle) and "`φ` was never trained" (the test phase then runs on a random projection); both are asserted here and both print a `***` block in `close()`. |

---

## E. What to watch in a real run

- `phase` — must reach 2. If `begin_test` is reached at phase 1 the arm prints a
  warning and the row is the **teacher**, not RMA.
- `phi_mse` — the regression loss. Should fall.
- `|zhat-z|` — the realised identification error, in latent units. This is the
  quantity B8's prediction is about.
- `|z|` — the scale of the latent, for reading `|zhat-z|` against.

**Prediction under test** (BASELINES.md B8): *"it works while excitation exists
(training) and degrades in greedy play, and it never returns to the host exactly
(no floor property)."* The first half is visible as `|zhat-z|` rising when the
policy stops exploring; the second is visible as the test-phase return relative
to the blind arm.

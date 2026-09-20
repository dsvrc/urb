# CHECKLIST — MF-Q (ICML 2018) on URB

**Paper** Yang, Luo, Li, Zhou, Zhang, Wang, *Mean Field Multi-Agent
Reinforcement Learning*, ICML 2018.
<https://proceedings.mlr.press/v80/yang18d/yang18d.pdf>

**Code read**
- <https://github.com/mlii/mfrl> — `examples/battle_model/algo/q_learning.py`
  (`class MFQ`), `algo/base.py` (`ValueNet`: the `prob` branch, `act`,
  `calc_target_q`), `main_MFQ_Ising.py` (`boltzman_explore` — the paper's own
  Boltzmann policy, sampled)
- <https://github.com/deligentfool/mfrl_pytorch> — `algo/q_learning.py`,
  `algo/base.py`, `senarios/senario_battle.py` (how `former_act_prob` is formed)

**BASELINES.md** B4 · objection row *"Mean-field RL is the standard for many
agents on a shared medium"*

**Implementation** `urb_baselines/algos/mfq.py` · **script** `scripts/mfq.py` ·
**config** `config/algo_config/mfq/config1.json`

---

## Note on why this exists at all

B4's recommendation is **not** to port mean-field RL, on the grounds that the
information-matched blind arm (the channels in the observation) already is a
mean-field-style policy. That recommendation stands for the paper's accounting —
report that arm under that name and cite MF-Q/MF-AC. MF-Q was implemented anyway
because on URB it is one file on the host's own DQN, and it removes a free
objection.

What it genuinely adds over the information-matched arm:

- the mean action is a **normalised distribution** over the neighbourhood, not
  raw counts of earlier same-OD travellers;
- it covers travellers who depart **later**, which the URB observation cannot see;
- it enters through a **dedicated embedding branch**, not the trunk;
- the neighbourhood includes the **human** travellers.

---

## A. The method, item by item

| id | item | source | status | where / why |
|---|---|---|---|---|
| **MF-1** | `Q^j(s, a^j, ā^j)` — the action value takes the neighbourhood's mean action as an extra input | Eq. 8–10 | **IMPLEMENTED** | `MeanFieldQNet` |
| **MF-2** | `ā^j = (1/|N(j)|) Σ_{k∈N(j)} onehot(a^k)` | Eq. 8 | **IMPLEMENTED** | `MFQ.end_episode` |
| **MF-3** | `ā` is taken from the **previous** step | `senario_battle.play`: `former_act_prob` is formed *after* the actions and used on the next step | **IMPLEMENTED** | `self.abar` is written at the end of day `t` and read during day `t+1` |
| **MF-4** | `ā` enters through its own embedding branch, `Linear(K, 64) → ReLU → Linear(64, 32)`, concatenated with the observation embedding | `algo/base.py`: `prob_emb_linear` | **IMPLEMENTED (verbatim)** | `MeanFieldQNet.prob_emb`, `mf_hidden: 64`, `mf_embed: 32` |
| **MF-5** | `ā` is initialised before any data arrives | `senario_battle`: `former_act_prob` starts at zeros | **ADAPTED** | initialised **uniform** rather than zero: a zero vector is not a distribution, and a uniform one is the honest "I know nothing". One line, stated here. |
| **NBR-1** | `N(j)` is the agent's neighbourhood | Eq. 8 | **ADAPTED** | default `field_scope: all` — the release's group-wide mean, because URB's OD pairs are singletons. See FIELD-1. |
| **POL-1** | Boltzmann policy `π_j(a|s, ā) ∝ exp(Q_j(s,a,ā)/T)`, **sampled** | Eq. 11; `main_MFQ_Ising.boltzman_explore` | **IMPLEMENTED, OFF BY DEFAULT** | `_MFDQN.act` with `exploration: "boltzmann"`. See ADAPT-2. |
| **POL-2** | Decaying temperature, floored at a minimum | `main_MFQ_Ising`: `current_t *= decay_rate`, floored | **IMPLEMENTED** | `temperature`, `temperature_decay`, `temperature_min` |
| **TGT-1** | `y = r + γ(1−done)·v^{MF}(s')` | Eq. 12 | **N/A** | A URB day is one step and terminates ⇒ `y = r`, as in URB's own IQL. |
| **TGT-2** | `v^{MF}(s') = Σ_{a'} π(a'|s', ā') Q_target(s', a', ā')` — the **expected** value under the Boltzmann policy, not a max | Eq. 12 | **N/A** | No bootstrap ⇒ no `v^{MF}` term. Recorded here because it is the detail a reader would check: MF-Q's target is *soft*, not `max`. |
| **TGT-3** | Target network + soft update `tau` | `algo/base.py`: `update()` | **N/A** | With `y = r` the target network has no role. Not built rather than left in as decoration. |
| **BUF-1** | Replay buffer | `tools.MemoryGroup` | **IMPLEMENTED** | URB's shared `buffer_size` deque, as in `scripts/iql.py` |
| **BUF-2** | The stored transition carries the `ā` that was used | `senario_battle`: `buffer['prob'] = former_act_prob` | **IMPLEMENTED** | `ā` is part of the observation vector the host's DQN stores, so it is carried automatically and cannot desynchronise |
| **NET-1** | CNN + feature embedding trunk (MAgent's visual observation) | `algo/base.py` | **ADAPTED** | URB's observation is a 5-vector, so the trunk is the host MLP. The `prob` branch is kept verbatim (MF-4) because that is the mean-field part. |
| **OPT-1** | Adam, lr 1e-4 | `q_learning.MFQ` | **ADAPTED** | URB's shared `lr` from `iql/config1.json`. R2. |
| **MFAC-1** | MF-AC, the actor-critic member of the same paper | §4.2 | **NOT IMPLEMENTED** | MF-Q is the value-based member and sits on URB's own off-policy base, which is the mapping rule R2 asks for. One representative per class. |

---

## B. Adaptations, in full

### FIELD-1 — the neighbourhood is the whole population, and the OD pair is a trap
Averaging one-hot action vectors is only strictly meaningful where the action
index means the same thing for everybody. On URB, action `k` is *"the k-th route
of my OD pair"*, so route 0 for OD A and route 0 for OD B are different roads and
their one-hot mean is not a quantity. That argues for the OD pair as the
neighbourhood, and it was the first default here.

**It is also moot, and the measurement is what settled it.** On every network URB
ships, the median OD pair carries exactly **one** traveller:

| network | travellers | distinct OD pairs | median / OD | max / OD |
|---|---|---|---|---|
| saint_arnoult | 222 | 215 | 1 | 2 |
| gretz_armainvilliers | 636 | 629 | 1 | 2 |
| nangis | 362 | 352 | 1 | 2 |
| nemours | 729 | 724 | 1 | 2 |
| provins | 523 | 517 | 1 | 2 |
| ingolstadt_custom(2) | 1035 | 306 | 1 | 85 |

A same-OD "mean field" is therefore one traveller's own one-hot action from
yesterday — not a mean field at all. The default is `field_scope: all`: the
group-wide mean, which is literally what the MAgent release computes
(`np.mean(..., keepdims=True)` over the whole handle) and which on URB is a
well-defined population statistic ("what fraction of everyone took their k-th
route"). It is a **weaker object than a per-road load**, and that weakness is
part of what this arm measures — it should be reported as such rather than
presented as a faithful neighbourhood mean.

Humans are included either way: a human loads the same roads and is part of the
field by any reading of the method, and `info.peer_acts` (RouteRL's own per-day
file) is what makes them visible.

Two alternatives remain selectable:
- `field_scope: od` — kept for a city that populates its OD pairs;
  `ingolstadt_custom` is the one that does. The arm prints a `[mfq][WARN]` when
  the median neighbourhood has two travellers or fewer, so choosing it on the
  wrong city cannot pass silently.
- `field_scope: graph` — the `|B| = 3` neighbourhood of `urb_baselines/graph.py`;
  DGN's Table 4 lists 3 neighbours for MFQ as well.

Under `all` every agent's field is the same set, so the mean is computed **once**
per day rather than per agent. Per-agent means would be O(machines × travellers)
— 428k dictionary lookups a day on `ingolstadt_custom`, 1.7 × 10⁹ over a full
run — for a quantity that is identical for everybody.

### ADAPT-2 — exploration defaults to the host's ε-greedy
The paper's policy is `softmax(Q/T)` with a decaying `T`. URB pays
`−travel_time`, so Q is of order −1000, while MAgent's rewards are of order 1: a
temperature transplanted from the paper gives either a uniform policy or a point
mass, with nothing in between at any value a reader could defend.

The default is therefore the **host's** ε-greedy schedule, the same one IQL and
DGN use — which is also, in practice, what the released `act` does: it takes the
`argmax` of the softmax, so the temperature is inert there and exploration comes
from `eps` being passed in from outside. `exploration: boltzmann` restores the
paper's rule, and the banner prints a note that the temperature must be on the
scale of the Q **differences**, not of the paper's. The `T` diagnostic is
printed alongside so it can be checked.

---

## C. Gates

| gate | what it proves |
|---|---|
| **drive mfq**, **drive mfq --field-scope all** | The arm runs, learns, the mean action is actually updated (`field_upd > 0`) and `ā` is not uniform (`abar_dev > 0`). Those two are precisely the ways this arm could silently reduce to IQL-with-four-constant-inputs. |
| **REC-1** | The record splitter keeps human travellers in `peer_acts`, which is what makes them part of the field. |

---

## D. What to watch in a real run

- `field_upd` — days on which `ā` was recomputed. Zero ⇒ every Q was evaluated at
  the uniform distribution and the arm is IQL with four constant inputs. `close()`
  prints a `***` block.
- `abar_dev` — `‖ā − uniform‖₁`, averaged over agents. Exactly zero ⇒ the
  mean-field input carries no signal; check `field_scope`. Note the **opposite**
  failure is also possible and looks healthy: a value near 1.5 under
  `field_scope: od` means `ā` is a single one-hot (one traveller), which is a
  large deviation from uniform and still not a mean field. The startup
  `[mfq][WARN]` is what catches that case.
- `T` — only when `exploration: boltzmann`; compare it with the spread of Q.

**Prediction under test** (BASELINES.md B4): the coupling on URB *is* a weighted
mean of neighbours' exertions, so this is the class most likely to capture part
of it — and the question is whether a mean action over a *route index* is the
same object as a mean load over a *shared link*. It is not, and the gap is what
this arm measures.

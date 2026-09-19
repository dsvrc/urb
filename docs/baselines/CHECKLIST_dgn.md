# CHECKLIST — DGN (ICLR 2020) on URB

**Paper** Jiang, Dun, Huang, Lu, *Graph Convolutional Reinforcement Learning*,
ICLR 2020. <https://openreview.net/forum?id=S8icDSeqfvy>

**Code read** <https://github.com/PKU-RL/DGN>
- `Routing/routers_regularization.py` — the closest published scenario: a routing
  problem with a 3-neighbour graph, including the temporal-regularisation wiring
  (`MLP`, `MultiHeadsAttModel`, `Q_Net`, `model_r`)
- `Battle/`, `Jungle/` — the MAgent instantiations

Also <https://github.com/jiechuanjiang/pytorch_DGN> (`Surviving/DGN.py`,
`Starcraft/DGN/model.py`) for the PyTorch masking idiom.

**BASELINES.md** B3 · objection row *"A GNN over the agents would learn the
coupling"*

**Implementation** `urb_baselines/algos/dgn.py` · **script** `scripts/dgn.py` ·
**config** `config/algo_config/dgn/config1.json`

**Arms** `--arm dgn` (the paper's method, with temporal relation regularisation)
· `--arm dgn_r` (the paper's own ablation without it)

B3's *chosen* representative is a GNN model flag in BenchMARL. URB has no
BenchMARL host and neither `ippo_torchrl.py` nor `mappo_torchrl.py` exposes a
graph model, so the published method itself is ported. DGN is Q-learning based
and discrete, so it sits on URB's off-policy base.

---

## A. The method, item by item

| id | item | source | status | where / why |
|---|---|---|---|---|
| **ARCH-1** | Three modules: observation encoder → convolutional layers → Q network | §3, Fig. 1 | **IMPLEMENTED** | `DGNNet` |
| **ARCH-2** | Encoder is an MLP for low-dimensional input | §3.1 | **IMPLEMENTED** | `MLP(obs, h, num_hidden, widths)` + ReLU |
| **ARCH-3** | **Two** convolutional layers (Table 4; Fig. 8 shows 2 beating 1) | Table 4 | **IMPLEMENTED** | `att1`, `att2` |
| **ARCH-4** | Relation kernel = multi-head dot-product attention over `B_{+i} = B_i ∪ {i}` | §3.2, Eq. 2 | **IMPLEMENTED** | `RelationKernel.forward`; row 0 of the gather is the ego, so the query is `h_i` |
| **ARCH-5** | `α^m_ij = softmax_j(τ · W_Q^m h_i · (W_K^m h_j)^T)` | Eq. 2 | **IMPLEMENTED** | gate **DGN-1** checks it against a hand computation, max error 0.0 |
| **ARCH-6** | `τ = 0.25` (= `1/sqrt(dv)` at `dv = 16`) | Table 4; the Keras kernel's `/np.sqrt(dv)` | **IMPLEMENTED** | `self.tau = 1/sqrt(dv)`, `dv: 16` |
| **ARCH-7** | `h'_i = σ(concat_m[Σ_j α^m_ij W_V^m h_j])`, σ = one-layer ReLU MLP | Eq. 3 | **IMPLEMENTED** | heads concatenated, then `relu(Linear(...))` |
| **ARCH-8** | ReLU on the Q/K/V projections | the released Keras kernel (`Dense(..., activation="relu")`) | **IMPLEMENTED** | `torch.relu(self.fcq(h))` etc. |
| **ARCH-9** | `M = 8` attention heads | Table 4 | **IMPLEMENTED** | `n_heads: 8` |
| **ARCH-10** | Q network takes the concatenation of **all preceding layers** (DenseNet-style) | §3.1; `Q_Net` in the Keras code concatenates `h1, h2, h3` | **IMPLEMENTED** | `self.q(cat([h0, h1, h2]))` |
| **ARCH-11** | Q network is an affine transformation | Table 4 | **IMPLEMENTED** | a single `nn.Linear` |
| **ARCH-12** | Encoder MLP units `(512, 128)` | Table 4 | **ADAPTED** | URB's shared `widths: [64, 64]`. R2: network size is a host hyperparameter; a wider network for one arm is a confound. |
| **ADJ-1** | `C_i` is `(|B_i|+1) × N`: row 0 the one-hot of `i`, row `j` the one-hot of its `(j−1)`-th neighbour | §3.1 | **IMPLEMENTED** | `NeighbourGraph.adj` is exactly that gather, stored as indices |
| **ADJ-2** | `|B| = 3` | Table 4; Fig. 9 prefers 3 over 1, 2, 4 | **IMPLEMENTED** | `n_neighbors: 3` |
| **ADJ-3** | Neighbours are "determined by distance or other metrics, depending on the environment" | §3 | **ADAPTED** | `graph: overlap` — route sets share a link **and** the trips are co-present. See ADAPT-1. |
| **ADJ-4** | The graph is fixed across two successive timesteps to ease learning | §3.1 | **IMPLEMENTED (stronger)** | URB's graph is fixed for the whole run: departure times come from `agents.csv` and route sets from the generated table, so it is exogenous structure, not something fitted |
| **ADJ-5** | Padded / absent neighbours must not receive attention | `pytorch_DGN`'s `-9e15*(1-mask)` | **IMPLEMENTED** | `masked_fill(..., -9e15)`; gate **DGN-1** asserts masked slots get **exactly** 0 and rows still sum to 1 |
| **LOSS-1** | `L(θ) = mean over samples and agents of (y_i − Q(O_{i,C}, a_i))²` | Eq. 1 | **IMPLEMENTED** | `F.mse_loss(q_taken, r)` |
| **LOSS-2** | `y_i = r_i + γ max_a' Q(O'_{i,C}, a'; θ')` | Eq. 1 | **N/A** | A URB day is one step and terminates ⇒ `y = r`, which is what URB's own IQL does. |
| **LOSS-3** | Target network, soft update `θ' = βθ + (1−β)θ'`, `β = 0.01` | Eq. 1, Table 4 | **N/A** | With no bootstrap the target network has no role. The paper is explicit that it is used *only* for the target Q value. Not built rather than left in as decoration. |
| **LOSS-4** | Gradients of all agents accumulated; **weights shared** | §3.1, Fig. 1 | **IMPLEMENTED** | one `DGNNet` for the whole fleet. The graph convolution is defined on shared weights; per-agent networks would not be DGN. |
| **REG-1** | Temporal relation regularisation: `+ λ (1/M) Σ_m KL(G^κ_m(O_{i,C}) ‖ G^κ_m(O'_{i,C}))` | Eq. 4, §3.3 | **IMPLEMENTED** | `learn`, the `if self.reg_lambda > 0` branch |
| **REG-2** | Applied to the **upper** convolutional layer (`κ = 2`) | §3.3, Table 4 | **IMPLEMENTED** | `att2`, not `att1` |
| **REG-3** | The **current** network is applied to the next state, *not* the target network | §3.3, explicitly | **IMPLEMENTED** | `self.net(xn, ...)` — the same weights |
| **REG-4** | `λ = 0.03` | Table 4 | **IMPLEMENTED** | `reg_lambda: 0.03` |
| **REG-5** | DGN **without** it is the paper's DGN-R ablation | §4 | **IMPLEMENTED** | `--arm dgn_r` |
| **EXP-1** | ε-greedy, `ε` and decay `0.6 / 0.996` | Table 4 | **ADAPTED** | URB's shared schedule from `iql/config1.json` (`eps_init 1.0`, `eps_decay 0.9985`). R2: the exploration schedule is a host hyperparameter and must match the arms it is compared to. |
| **BUF-1** | Replay buffer of `(O, A, O', R, C)` tuples, capacity 2×10⁵ | §3.1, Table 4 | **IMPLEMENTED** | a ring of per-day observation matrices; capacity is URB's shared `buffer_size` (2048 **days**, = 2048 × n_av transitions) |
| **BUF-2** | Batch size 10 (timesteps), 5 epochs per episode | Table 4 | **ADAPTED** | URB's shared `batch_size` (32) and `num_epochs` (1) per day, so the sample throughput matches IQL's exactly |
| **OPT-1** | Adam, lr 1e-4 | Table 4 | **ADAPTED** | URB's shared `lr` (1e-3, `iql/config1.json`). R2. |

---

## B. Adaptations, in full

### SNAP-1 — the feature matrix, which URB does not hand over
DGN's `F^t` is every agent's observation **at the same instant**. A URB day has
no such instant: travellers act in start-time order and each one's observation
counts the routes earlier same-OD travellers took *that day*, so agent `i`'s
observation does not exist until its turn.

The rule used here, **identically at execution and at training**:

```
F_i(t)[j] = the observation agent j acted on during day t−1,   for j ≠ i
F_i(t)[i] = agent i's CURRENT observation
```

Why this one:
- it is **causal** — nothing from the future, nothing from a later-departing peer;
- it is the **same construction in both phases**, so there is no train/execute
  asymmetry to explain away;
- the ego keeps its own live observation, which is exactly IQL's input, so the
  arm is not handicapped against the learner it is compared to.

The cost, stated: neighbours are one day stale. A sample is therefore a
`(day, ego)` pair rather than a whole graph, which is the same expectation at a
fraction of the memory (DGN accumulates the gradient over all agents in a graph;
here the graph's agents are drawn uniformly).

### ADAPT-1 — the neighbour structure
`graph: overlap` is the default: agents `i` and `j` are neighbours when their OD
pairs' route sets share at least one link **and** their trips are on the road at
the same time. That is the actual coupling graph, which is what B3's question is
about. It needs the generated route table; if none is found the arm falls back to
`copresence` and says so in the log. `od` and `copresence` are selectable with
`--graph`.

Ranking within a structure is by `|start_time_i − start_time_j|` — closest in
time is most likely to genuinely share the road, and it is a total order, so the
neighbour set is deterministic given `agents.csv`. Gate **GRAPH-1** checks
determinism and that `od` mode never crosses an OD pair.

### SCALE-1 — the regression target is standardised, and without it this is not a graph network
This is the one adaptation that changes the architecture's behaviour rather than
its inputs, so it is documented at length.

MAgent, where Table 4's constants were chosen, pays rewards of order 1. URB pays
`−travel_time`, of order −1000. The Q head is a linear map of the concatenated
features, so to reach −1000 the encoder's features must grow to about 50. The
attention logit is `τ · (W_Q h) · (W_K h)`, which is **quadratic** in that scale.
Measured on `urb_baselines/fakeenv.py` after 120 days with the raw target:

```
  h0 abs mean          5.41e+01      (features forced large by the target)
  layer-1 attention    [0, 0, 0, 1]  (a hard argmax: each node copies one neighbour)
  layer-2 attention    [0.25, 0.25, 0.25, 0.25]  (exactly uniform)
  temporal relation KL 0.0           (identically zero)
```

Every node copying the same neighbour makes the second kernel's keys identical,
which is why its attention is *exactly* uniform and the regulariser is
identically zero. The graph convolution has stopped being one, and every curve
still looks plausible.

Standardising the target by a running mean/variance of the rewards restores the
unit scale `τ = 1/sqrt(dv)` was calibrated for. After the fix, same run:

```
  h0 abs mean          4.01e+00
  layer-1 attention    [0.25, 0.25, 0.25, 0.25]   (untrained, not saturated)
  temporal relation KL 2.66e-09                    (live, not identically zero)
```

Q values are then in normalised units, which changes no argmax and therefore no
action. It is the same device HARL's `ValueNorm`, LCPO's `ret_rms` and LIAM's
`standardise_stream` all use, for the same reason. `normalize_target: false`
reproduces the degenerate behaviour above.

---

## C. Gates

| gate | what it proves |
|---|---|
| **DGN-1** | `softmax(τ q·k)` over `B_{+i}` matches a hand computation of Eq. 2 to 0.0; masked neighbour slots get **exactly** zero attention and the remaining rows still sum to 1. This is the gate that catches a transposed axis or a dropped mask. |
| **DGN-2** | The `(t−1, t, t+1)` sampler never straddles the ring buffer's write head, in both the not-yet-wrapped and wrapped regimes. Without it the temporal regulariser would occasionally compare two days a full lap apart — rare, silent and wrong. |
| **GRAPH-1** | The neighbour structure is deterministic and never crosses an OD pair in `od` mode. |
| **drive dgn**, **drive dgn --arm dgn_r** | Both arms run, learn and keep the attention out of saturation. |

---

## D. What to watch in a real run

- `att_spread` — how far the upper layer's attention is from uniform, over the
  **valid** slots only. Near 0 ⇒ the relation kernel has degenerated into a mean
  (that is DGN-M, the paper's other ablation, not DGN). Near 1 ⇒ the softmax has
  saturated into a hard argmax. Both print a `***` block in `close()`.
- `reg` — the temporal relation KL. Note that its **value** is tiny whenever the
  attention is near uniform, but its **gradient** is not, so a small `reg` does
  not mean the term is inert. Read it beside `att_spread`.
- `q` — the Q loss, in normalised units.

**Prediction under test** (BASELINES.md B3): *"it learns the σ=0 coupling but
cannot track the drift, because the gain is not identified as a separate
quantity."* The paper's own routing experiment is relevant context here: it notes
that in routing "data packets (agents) with different destinations seldom share
many links (cooperate continuously) along their paths", which is exactly the
regime `graph: overlap` measures on a real city — the banner prints the realised
degree distribution.

# CHECKLIST — R-IPPO (the memory arm) on URB

**Paper** Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu, *The Surprising
Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022 D&B.
<https://arxiv.org/abs/2103.01955>

**Code read** <https://github.com/marlbenchmark/on-policy>
- `onpolicy/algorithms/utils/rnn.py` — `RNNLayer`
- `onpolicy/utils/shared_buffer.py` — `recurrent_generator` (the chunked BPTT)

**BASELINES.md** B2 · Tier 1 item 2 · objection row *"Just give the policy
memory"*

**Implementation** `urb_baselines/algos/rippo.py` · **script** `scripts/rippo.py`
· **config** `config/algo_config/rippo/config1.json`

B2's *chosen* representative is "GRU-MAPPO via BenchMARL's built-in `GRU` model
(a configuration change, no code)", with R-MAPPO named as the reference
implementation. URB has no BenchMARL host and neither `ippo_torchrl.py` nor
`mappo_torchrl.py` exposes an RNN, so the configuration change has to be written
out. That is this arm: **URB's own IPPO with R-MAPPO's recurrent machinery, and
nothing else changed.**

---

## A. What is taken from R-MAPPO

| id | item | source | status | where / why |
|---|---|---|---|---|
| **RNN-1** | One `nn.GRU` layer between the observation encoder and the action head | `rnn.py: RNNLayer` | **IMPLEMENTED** | `RecurrentActor`: `encoder → GRUCore → head` |
| **RNN-2** | Orthogonal weight init, zero biases | `rnn.py` (`use_orthogonal: True` is the default) | **IMPLEMENTED (verbatim)** | `GRUCore.__init__` |
| **RNN-3** | `LayerNorm` on the GRU output | `rnn.py` | **IMPLEMENTED (verbatim)** | `GRUCore.norm` |
| **RNN-4** | `recurrent_N = 1` | `happo.yaml` / MAPPO defaults | **IMPLEMENTED** | `num_layers=1` |
| **BPTT-1** | The collected sequence is cut into chunks of `data_chunk_length` **consecutive** steps | `shared_buffer.recurrent_generator` | **IMPLEMENTED** | `starts = idx * L`, `pos = starts + arange(L)` |
| **BPTT-2** | Each chunk is replayed from the hidden state recorded at its **first** step | `recurrent_generator`: `rnn_states[ind]` | **IMPLEMENTED** | `h0 = hin[starts]` |
| **BPTT-3** | `data_chunk_length = 10` | MAPPO's default | **IMPLEMENTED** | config; `--data-chunk-length` overrides |
| **BPTT-4** | `num_mini_batch` chunk-minibatches per epoch | `recurrent_generator` | **IMPLEMENTED** | `num_mini_batch: 1`, MAPPO's default |
| **BPTT-5** | The ragged tail (`batch_size % data_chunk_length`) is dropped | `data_chunks = batch_size // data_chunk_length` | **IMPLEMENTED** | `n_chunks = T // L` |
| **MASK-1** | The hidden state is multiplied by the episode `mask`, zeroing it at a terminal step | `rnn.py: forward` | **ADAPTED — and this is the point of the arm** | see SEQ-1 |

---

## B. What is deliberately NOT taken, and why

| id | item | status | why |
|---|---|---|---|
| **NOT-1** | R-MAPPO's centralised critic | **NOT IMPLEMENTED (on purpose)** | Adding it would confound *"memory helps"* with *"a centralised critic helps"*, and B2's question is only the first. The centralised-critic question has its own arm: `scripts/happo.py`. |
| **NOT-2** | GAE with γ, λ | **N/A (collapses exactly)** | one-step terminal episode ⇒ the advantage is the host's, unchanged |
| **NOT-3** | MAPPO's `ppo_epoch = 15`, `lr = 5e-4` etc. | **ADAPTED** | The learner is URB's IPPO, so the clip, the entropy coefficient, the advantage, the learning rate, the number of epochs and the update trigger are all the host's. R2: the ONLY difference from `scripts/ippo.py` is the GRU. That is what makes this arm an answer to B2 rather than a second MAPPO. |

---

## C. The one adaptation

### SEQ-1 — the recurrence runs over DAYS, not within an episode
In R-MAPPO the recurrence runs over the steps of an episode and the `mask` zeroes
the hidden state at every episode boundary. **A URB episode is one step.** A
per-episode mask would therefore erase the memory every single day, the GRU
would see a length-1 sequence with a zero initial state, and the recurrent arm
would be numerically identical to the feed-forward one — a memory baseline whose
memory is one step long, which answers no objection at all.

The sequence that matters here is the sequence of **days**: B2's objection is
that *"a history-conditioned policy could infer the drift from its own past"*,
and the agent's past is its past days. So the hidden state is carried across days
and across the train/test boundary, and is reset only at construction.

The consequence for BPTT-2 is that the stored `h0` is genuinely the state the
agent had on the chunk's first day, so the replay is exact up to the policy
having moved since collection — which is the ordinary on-policy situation
R-MAPPO's generator is written for.

---

## D. Gates

| gate | what it proves |
|---|---|
| **drive rippo** | The arm runs, learns, writes loss rows, and the hidden state is non-trivial (`\|h\|` is reported). |

---

## E. What to watch in a real run

- `|h|` — the mean absolute hidden state. If it collapses to 0 the memory is
  carrying nothing.

**Prediction under test** (BASELINES.md B2): *"memory does not recover, because
the required quantity (the peers' next actions through W) is not a function of
the agent's own history; degradation persists."* The comparison is against
`ippo` from `scripts/sweep/run_sweep.sh` at the same σ and seed — and because
this arm is IPPO plus a GRU and nothing else, that comparison is clean.

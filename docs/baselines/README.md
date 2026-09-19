# URB baselines — index, rules and the verification register

This directory is the audit trail for `urb_baselines/`. Every baseline named in
[`paper/BASELINES.md`](../../../paper/BASELINES.md) whose host is URB is
implemented here as a **separate algorithm** — its own `scripts/<name>.py`, its
own `config/algo_config/<name>/`, its own module — exactly as `scripts/pact1.py`
is. No baseline is a flag on PACT-1 and none of them is a flag on another
baseline.

Each baseline has a checklist file in this directory that goes through its paper
and its released code item by item, and records for every item whether it is
**IMPLEMENTED**, **ADAPTED** (with what URB forced and why), **SKIPPED** (with
why) or **N/A** (with why the concept does not exist on this host). That is what
you read to check the work.

---

## 1. What was implemented

| # | baseline | class (BASELINES.md) | base | script | checklist |
|---|---|---|---|---|---|
| 1 | **LCPO** (ICLR 2025) | B6 non-stationary, observed context | on-policy | `scripts/lcpo.py` | [CHECKLIST_lcpo.md](CHECKLIST_lcpo.md) |
| 2 | **HAPPO** (ICLR 2022) | B1 trust region / sequential update | on-policy | `scripts/happo.py` | [CHECKLIST_happo.md](CHECKLIST_happo.md) |
| 3 | **ERNIE** (NeurIPS 2023) | B9 robust MARL | on-policy | `scripts/ernie.py` | [CHECKLIST_ernie.md](CHECKLIST_ernie.md) |
| 4 | **R-IPPO** (R-MAPPO machinery) | B2 memory | on-policy | `scripts/rippo.py` | [CHECKLIST_rippo.md](CHECKLIST_rippo.md) |
| 5 | **DGN** (ICLR 2020) | B3 graph / communication | off-policy | `scripts/dgn.py` | [CHECKLIST_dgn.md](CHECKLIST_dgn.md) |
| 6 | **MF-Q** (ICML 2018) | B4 mean field | off-policy | `scripts/mfq.py` | [CHECKLIST_mfq.md](CHECKLIST_mfq.md) |
| 7 | **LIAM** (NeurIPS 2021) | B5 agent modelling | on-policy | `scripts/liam.py` | [CHECKLIST_liam.md](CHECKLIST_liam.md) |
| 8 | **RMA / UP-OSI** (RSS 2021/2017) | B8 meta-RL / online sys-ID | on-policy | `scripts/rma.py` | [CHECKLIST_rma.md](CHECKLIST_rma.md) |
| 9 | **ESO / DOB** (linear ADRC) | B10 classical control | non-learning | `scripts/eso.py` | [CHECKLIST_eso.md](CHECKLIST_eso.md) |
| 10 | **Unstructured RLS** | B10 classical control | non-learning | `scripts/urls.py` | [CHECKLIST_urls.md](CHECKLIST_urls.md) |
| 11 | **Domain randomisation over σ** | B9 robust (the "must") | on-policy | `scripts/dr_ippo.py` | [CHECKLIST_dr_ippo.md](CHECKLIST_dr_ippo.md) |
| 12 | **Oracle-driver IPPO** | information arm | on-policy | `scripts/oracle_ippo.py` | [CHECKLIST_oracle_ippo.md](CHECKLIST_oracle_ippo.md) |

"on-policy" means it is built on URB's own on-policy learner, the single-step
actor-only PPO of `scripts/ippo.py`; "off-policy" means URB's own single-step DQN
from `scripts/iql.py`. Both are reproduced unchanged in
`urb_baselines/algos/reference.py`, and every baseline either subclasses or
composes them — that is the mapping `paper/BASELINES.md` asks for.

Things that were deliberately NOT implemented are in §6, with reasons.

---

## 2. How to run all of them

```bash
bash scripts/sweep/run_baselines_sigma3.sh 0
```

One command. It runs the offline gates first (about a minute, no SUMO), then
every Tier-1 baseline at torch seed 0 under σ = 3, through `ns_launch.py`, with
the same pinned route table, the same task config and the same environment seed
the PACT-1 sweep uses. `run_baselines_sigma0.sh` is the identical command with
the dial off, and is the no-severity row.

```bash
TIER=all ARMS=1 bash scripts/sweep/run_baselines_sigma3.sh 0   # + Tier 2 + ablations
ONLY="lcpo_0 eso_0"  bash scripts/sweep/run_baselines_sigma3.sh 0
DRYRUN=1             bash scripts/sweep/run_baselines_sigma3.sh 0   # print, run nothing
```

Every knob is documented in the header of
[`scripts/sweep/run_baselines.sh`](../../scripts/sweep/run_baselines.sh).
Completed arms leave a `.done` marker so the runner is resumable.

**Run the PACT-1 sweep too.** These baselines only mean something beside
`scripts/sweep/run_sweep.sh`, which must use the same `NET`, `ENV_SEED`,
`TASK_CONF` and route table. The runner bootstraps exactly the same pinned table
(`results/routegen_<net>_<seed>/routes.csv`), so running either one first is
fine.

### Before a long run

```bash
python urb_baselines/selftest.py                    # ~1 min, no SUMO, no routerl
python scripts/lcpo.py --id t --alg-conf config1 --task-conf config1 \
    --net saint_arnoult --mode dry                  # wiring, no AV day simulated
```

---

## 3. The four rules every baseline here follows

**R1 — one host, one loop.** Every arm runs through `urb_baselines/host.py`: the
same config merge, the same `TrafficEnvironment` arguments, the same
human-learning phase, the same `mutation()`, the same day loop, the same
deterministic test phase, the same `run_metrics_analysis`. A difference between
two arms therefore cannot be a difference in how the host was driven. The host
is checked against `scripts/ippo.py` / `scripts/iql.py` by selftest gate H2 and
by the two reference arms in `algos/reference.py`.

**R2 — host hyperparameters come from the shared config; paper hyperparameters
come from the paper.** Anything URB's config already fixes (`lr`, `batch_size`,
`num_epochs`, `clip_eps`, `entropy_coef`, `widths`, `num_hidden`,
`normalize_advantage`, the ε-greedy schedule) is copied verbatim from
`ippo/config1.json` or `iql/config1.json`, so an arm difference is never a
tuning difference. Anything that exists only in the baseline's own paper is
taken from that paper's own defaults, and every such value is sourced in the
`desc` block of its config file. Nothing anywhere was tuned on returns.

**R3 — adapt only where the host forces it, and say so.** Every deviation has a
row in that baseline's checklist marked ADAPTED, naming what URB forced. The two
things that force nearly all of them are in §4.

**R4 — an inert mechanism must be impossible to miss.** Every arm prints a
report at the end and a line beginning `***` if its own mechanism never
engaged — LCPO's constraint never binding, MF-Q's mean action never updating,
DGN's attention collapsing, RMA's phase 2 never starting, an estimator that
never received a row. The runner greps for those lines and lists the arm as
`DEGENERATE` in its summary. A baseline that quietly reduced to its own fallback
and still produced a plausible curve is the failure this project cannot afford.

---

## 4. The two properties of URB that force every adaptation

**A day is ONE step.** Every traveller picks one of K routes, the day is
simulated, everyone is paid once, the episode terminates. So:

- `gamma` and `lambda` have nothing to discount. GAE collapses *exactly* to
  `adv = r − V(s)` and `returns = r` for any values of either.
- There is nothing to bootstrap from. URB's own IQL uses `target = reward`, and
  every off-policy baseline here does the same. Target networks, soft updates
  and `max_a' Q(s', a')` terms are therefore recorded as **N/A** rather than
  left in as decoration.
- A per-episode RNN has a sequence of length one. Where a method's recurrence
  matters (R-IPPO, LIAM) the sequence is the sequence of **days** and the hidden
  state is carried across them; a per-episode reset would make the recurrent arm
  numerically identical to the feed-forward one.

**A day is SEQUENTIAL, not simultaneous.** Travellers act in start-time order,
and each one's observation counts the routes taken by *earlier same-OD
travellers that same day*. There is no instant at which all agents' current
observations exist. Any method defined on a simultaneous joint observation has
to name its snapshot, and each one does (DGN's is in
[CHECKLIST_dgn.md](CHECKLIST_dgn.md) item SNAP-1).

### The third thing, which is not about the episode: the reward scale

URB pays `−travel_time`, i.e. rewards of order **−1000**. Four of these methods
were published on benchmarks whose rewards are of order 1, and three of them
break on the difference in ways that are silent:

| arm | what breaks | the fix, and whose it is |
|---|---|---|
| HAPPO | orthogonal init on an unnormalised start time in seconds ⇒ logits ~1000 ⇒ point-mass softmax ⇒ **zero policy gradient, measured at 1e-15/update** | `use_feature_normalization: True` and `gain: 0.01` — both from `happo.yaml`, HARL's own model block |
| DGN | features must reach ~50 to output −1000; attention logits are quadratic in that ⇒ every softmax saturates to a hard argmax ⇒ **the graph convolution degenerates into "copy one neighbour"** | standardise the regression target — the same device as HARL's `ValueNorm`, LCPO's `ret_rms` and LIAM's `standardise_stream` |
| LIAM | the reconstruction target's start-time coordinate is ~1000 and constant per traveller ⇒ **99.99% of the squared error is memorising three constants** | standardise the reconstruction target per dimension |
| LCPO | — (the authors already handle it: `a2c.py` divides rewards by `sqrt(ret_rms.var + 1)`) | reproduced verbatim |

Each was **measured** on `urb_baselines/fakeenv.py`, not guessed; the numbers are
in the relevant checklist and in the module docstring. Each has an off switch so
the degenerate behaviour can be reproduced.

---

## 5. How to verify the work

1. **Read a checklist.** Each row cites the paper equation or the reference file
   and function it came from, and names the line of `urb_baselines/` that
   implements it.
2. **Run the gates.** `python urb_baselines/selftest.py` runs 30 of them in
   about a minute with no SUMO and no `routerl`. They are of two kinds:
   - *unit* gates check one equation against the paper — that LCPO's step
     respects **both** trust regions, that ERNIE's adversarial perturbation is
     measurably worse than a random one of the same budget (≈3.6×), that DGN's
     `softmax(τ q·k)` matches a hand computation and that masked neighbour slots
     get **exactly** zero attention, that the domain randomiser really redraws
     and really detaches;
   - *drive* gates run every arm for a few hundred days on a 100-line numpy
     stand-in with URB's observation shape, URB's sequential single-step day and
     URB's reward scale, and check it runs, learns, writes loss rows and keeps
     its own invariants.
   A pass there is not a result; it is the absence of a class of bug.
3. **Read a run's banner.** Every arm prints exactly what it was configured
   with, in the terms of its own paper, before it does anything.
4. **Read a run's closing report** and the runner's `DEGENERATE` list (rule R4).
5. **Diff the configs.** The host block of every `config1.json` should be
   character-identical to `ippo/config1.json` or `iql/config1.json`.

---

## 6. What was NOT implemented, and why

These are all consistent with `paper/BASELINES.md`; the reasons are repeated
here so the register is in one place.

| method | class | why not |
|---|---|---|
| **HASAC** | B1 | Off-policy and continuous-action; `BASELINES.md` assigns it to the MAPDN host and HAPPO to URB. Its trust-region claim is already represented on URB by HAPPO. |
| **LILAC** | B7 | SAC-based, continuous; assigned to MAPDN by B7 ("or skip and cite if time is short"). Its latent-per-episode idea is represented on URB by RMA's learned latent, which the checklist notes. |
| **TPA-for-AVC** | B11 | Domain-specific to active voltage control; its inputs are power-network quantities. Not portable to routing, and `BASELINES.md` assigns it to MAPDN. |
| **GNN-MAPPO / GRU-MAPPO via BenchMARL** | B2, B3 | BenchMARL is not a URB host. The underlying questions are answered by R-IPPO (memory) and DGN (graph), which are the published methods rather than model flags. |
| **DCG, MAT, QPLEX, AMAGO, MAMBA, MAMBPO, MBCD** | B3, B1, B8, B12, B7 | Tier 3 in `BASELINES.md` §D: cite, run only on request. |
| **M2TD3, RARL** | B9 | Continuous-action robust RL; no discrete host. ERNIE is the MARL-specific robust representative and is implemented. |
| **VariBAD, PEARL, Decision Adapter** | B8 | `BASELINES.md` cites rather than runs them; RMA/UP-OSI is the chosen representative and is implemented. |
| **LOLA, M-FOS, Meta-MAPG, POLA** | B5/E | Defined for two-player general-sum games with white-box or meta-game access to the opponent's learning. Not applicable at N = 89 selfish routers; `BASELINES.md` §E argues rather than runs them. |
| **ERNIE §5.3 (joint-action Q) and §5.4 (mean-field Q)** | — | Both regularise a *global* or *mean-field* Q-function. URB's on-policy host is actor-only and per-agent, so neither object exists. Recorded as N/A in [CHECKLIST_ernie.md](CHECKLIST_ernie.md) rather than approximated. |
| **MF-AC** | B4 | The actor-critic member of the same paper; MF-Q is the value-based one and sits on URB's own off-policy base, which is the mapping rule R2 asks for. |

`BASELINES.md` B4 recommends *not* porting mean-field RL at all, on the grounds
that the information-matched blind arm is already a mean-field-style policy.
That recommendation still stands for the paper's accounting — report that arm
under that name — and MF-Q was implemented anyway, because on URB it is one file
on the host's own DQN and it removes a free objection. What it adds over the
information-matched arm is listed at the top of `urb_baselines/algos/mfq.py`.

---

## 7. Layout

```
urb_baselines/
    host.py            the shared loop + the BaselineAlgorithm hook protocol
    nets.py            the host MLP (identical to scripts/iql.py's Network)
    context.py         the exogenous driver A(day): observed vs privileged
    records.py         RouteRL's per-day travel times and executed actions
    graph.py           neighbour structures: od / copresence / route overlap
    domain_random.py   per-episode sigma resampling, attached to one env instance
    fakeenv.py         a 100-line numpy stand-in for URB, for the gates
    selftest.py        the 30 offline gates
    algos/
        reference.py   URB's own PPO and DQN, and the two reference arms
        lcpo.py happo.py ernie.py rippo.py rma.py liam.py
        oracle_ippo.py dr_ippo.py compensator.py eso.py urls.py
        dgn.py mfq.py
scripts/
    lcpo.py happo.py ernie.py rippo.py rma.py liam.py oracle_ippo.py
    dr_ippo.py eso.py urls.py mfq.py dgn.py
    sweep/run_baselines.sh  run_baselines_sigma0.sh  run_baselines_sigma3.sh
config/algo_config/<name>/config1.json      one per baseline, sourced in `desc`
docs/baselines/CHECKLIST_<name>.md          one per baseline
```

Nothing in `urb_baselines/` imports `pact1/`, and nothing in `pact1/` was
touched. `urb_ns/` was not modified either: the severity dial every arm runs
under is byte-for-byte the certified one, and domain randomisation is attached
to a single environment *instance* at run time rather than built into it
(`urb_baselines/domain_random.py` explains why).

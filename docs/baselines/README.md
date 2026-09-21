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
| 13 | **QCD restart** (2024) | B7 change point, drift *unobserved* | off-policy | `scripts/qcdr.py` | [CHECKLIST_qcdr.md](CHECKLIST_qcdr.md) |
| 14 | **Deep FP** (NeurIPS 2025) | *new class*: equilibrium seeking | on-policy | `scripts/dfp.py` | [CHECKLIST_dfp.md](CHECKLIST_dfp.md) |
| 15 | **Performative MPG** (2025) | *new*: control for Theorem B2 | on-policy | `scripts/pmpg.py` | [CHECKLIST_pmpg.md](CHECKLIST_pmpg.md) |
| 16 | **DORAEMON** (ICLR 2024) | B9 robust, *adaptive* randomisation | on-policy | `scripts/doraemon.py` | [CHECKLIST_doraemon.md](CHECKLIST_doraemon.md) |
| 17 | **WISDOM** (2025) | B7/B8 learned predictive representation | on-policy | `scripts/wisdom.py` | [CHECKLIST_wisdom.md](CHECKLIST_wisdom.md) |
| 18 | **M3W** (NeurIPS 2025) | B12 model-based MARL | off-policy | `scripts/m3w.py` | [CHECKLIST_m3w.md](CHECKLIST_m3w.md) |

Arms 1-12 were implemented on 2026-09-19; **arms 13-18 on 2026-09-21**. The six
later ones close the three classes that had no URB arm at all -- B7
(change point, drift unobserved), B12 (model-based MARL) and equilibrium
seeking, which no class covered -- plus the adaptive form of B9 and the learned
form of the estimator. Each one's checklist opens with why that class was open.

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

**One arm per baseline, and only one.** The sweep runs each method in its own
primary configuration -- the one its checklist calls the method -- and nothing
else. The `--arm` flags still exist on the individual scripts and each is
covered by a selftest drive gate, so an ablation can be run by hand when a
specific question needs it, but the sweep does not launch any. "Which PART of
this method did the work" is a question for after the headline table exists, and
at roughly three hours per arm it is not how the SUMO budget should be spent.

One command. It runs the offline gates first (about a minute, no SUMO), then
every baseline at torch seed 0 under σ = 3, through `ns_launch.py`, with
the same pinned route table, the same task config and the same environment seed
the PACT-1 sweep uses. `run_baselines_sigma0.sh` is the identical command with
the dial off, and is the no-severity row.

```bash
ONLY="lcpo_0 eso_0"  bash scripts/sweep/run_baselines_sigma3.sh 0
DEVICE=cpu           bash scripts/sweep/run_baselines_sigma3.sh 0   # force CPU
DRYRUN=1             bash scripts/sweep/run_baselines_sigma3.sh 0   # print, run nothing
```

`DEVICE=cpu` does two things: it passes `--device cpu` to every baseline, and it
exports `CUDA_VISIBLE_DEVICES=""`, which also fixes `scripts/pact1.py`,
`scripts/ippo.py` and every other stock URB script — all of them pick the device
with the same `torch.device(0) if torch.cuda.is_available()` line and none of
them has a flag for it.

**If a run is killed** (Ctrl-C, SIGTERM, a scheduler) the runner stops rather
than launching the remaining arms into the same thing. Without that, one
interruption is reported as a list of failures.

Every knob is documented in the header of
[`scripts/sweep/run_baselines.sh`](../../scripts/sweep/run_baselines.sh).
Completed arms leave a `.done` marker so the runner is resumable.

**Run the PACT-1 sweep too.** These baselines only mean something beside
`scripts/sweep/run_sweep.sh`, which must use the same `NET`, `ENV_SEED`,
`TASK_CONF` and route table. The runner bootstraps exactly the same pinned table
(`results/routegen_<net>_<seed>/routes.csv`), so running either one first is
fine.

### Resuming, and what each arm costs

Completed arms leave a `.done` marker beside their log, and the runner skips
them, so re-running the same command after an interruption picks up where it
stopped. A failed arm writes **no** marker, so a re-run retries it. To force a
completed arm to re-run, delete its marker:

```bash
rm logs/bl_s3/lcpo_0.log.done
ONLY="lcpo_0" bash scripts/sweep/run_baselines_sigma3.sh 0
```

Every arm prints `d/min` and an `eta` for the remaining training days every
`print_every` days, so its cost is knowable from the first few hundred days
rather than at the end.

**The device is probed before SUMO starts.** `torch.cuda.is_available()` only
reports that a driver and a device exist; it says nothing about whether this
torch build has kernels for that GPU's compute capability. When it does not, the
error is `CUDA error: no kernel image is available for execution on the device`,
raised at the FIRST forward pass — which on URB is after the 200-day
human-learning phase, seventeen minutes in. One matmul and one backward at
startup turn that into an immediate message. `auto` then falls back to CPU and
says so; an explicit `--device cuda` that fails is fatal, because silently
running somewhere else is not what was asked for.

Measured ALGORITHM cost over a 4000-day run at `saint_arnoult`'s scale (222
travellers, 88 machines, K = 4), excluding the environment step — that part is
SUMO and is common to every arm. **These are the original twelve; arms 13–18 have
not been measured on this network yet.** Five of them are small additions to a
host learner and should land in the same band; `m3w` will not — it runs
`num_samples × n_av × iterations ≈ 68k` reward-model forwards per day and is the
heaviest algorithm here by a clear margin. Take the first `d/min` line of a real
run as its estimate rather than assuming.

| arm | min | ×IPPO | | arm | min | ×IPPO |
|---|---|---|---|---|---|---|
| eso | 0.2 | 0.04 | | rma | 5.7 | 1.1 |
| happo | 4.2 | 0.8 | | ernie | 6.7 | 1.3 |
| urls | 4.6 | 0.9 | | dgn | 7.6 | 1.4 |
| ippo (URB's own) | 5.3 | 1.0 | | rippo | 7.8 | 1.5 |
| oracle_ippo | 5.8 | 1.1 | | liam | 8.6 | 1.6 |
| iql (URB's own) | 15.4 | 2.9 | | lcpo | 16.9 | 3.2 |
| | | | | mfq | 19.2 | 3.6 |

The spread is **at most ~15 minutes** across the whole set, so an arm that takes
hours longer than another is not being slowed by its algorithm — look at the
`d/min` line, or at SUMO.

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
| **M3W** | two-hot reward coding over 101 bins spanning **[-10, 10]** ⇒ every URB reward lands in bin 0 ⇒ the reward model is a constant ⇒ every sampled joint action scores the same ⇒ **MPPI's softmax is uniform and the planner picks routes at random**, while the loss curve looks healthy | standardise the reward before the two-hot (`normalize_reward`); `plan_spread` and `clip` make it visible |
| **QCDR** | a Bernoulli GLR is defined on [0, 1]; raw, every value clips to the same boundary and the statistic is identically 0, so **the detector can never fire** — which looks exactly like "there was no change point". A *running* standardisation is nearly as bad: it absorbs the shift (measured: GLR 15.5 vs 59.6 for the same 300 s change) | map through a **frozen** reference + logistic (`normalize_stream`, `calib`); `u_spread` makes it visible |
| **DFP / WISDOM** | the observation leads with a start time in seconds, so a probability vector or a unit-scale latent beside it is **below the network's resolution** — measured: identical returns to 4 s.f. with and without the mean-field input, and −13 vs +153 for WISDOM over 200 gate days | standardise the policy observation (`nets.RunningNorm`), the device HAPPO needed for the same coordinate; `mix_infl` / `z_std` make it visible |

Each was **measured** on `urb_baselines/fakeenv.py`, not guessed; the numbers are
in the relevant checklist and in the module docstring. Each has an off switch so
the degenerate behaviour can be reproduced.

### The fourth thing, which is about the cities: URB's OD pairs are singletons

Measured over the seven networks URB ships (`networks/*/agents.csv`):

| network | travellers | distinct OD pairs | median / OD | with ≥1 earlier same-OD peer |
|---|---|---|---|---|
| saint_arnoult | 222 | 215 | 1 | 3.2% |
| gretz_armainvilliers | 636 | 629 | 1 | 1.1% |
| nangis | 362 | 352 | 1 | 2.8% |
| nemours | 729 | 724 | 1 | 0.7% |
| provins | 523 | 517 | 1 | 1.1% |
| ingolstadt_custom(2) | 1035 | 306 | 1 | 70.4% |

Two consequences, both properties of the benchmark rather than of any method
here, and both faced equally by every arm including PACT-1:

1. **Almost every traveller is alone on its OD pair.** Anything defined on the
   same-OD set is a singleton.
2. **URB's observation is `[start_time, 0, 0, 0, 0]` for ~97% of travellers on
   every day of the run** on all networks but `ingolstadt_custom`, because the
   four count coordinates tally *earlier same-OD* travellers. The observation is
   effectively a constant agent identifier, which makes URB on these cities a
   repeated **contextless** bandit for every learner.

Three defaults follow, each recorded in the relevant checklist:

| arm | was | is | why |
|---|---|---|---|
| MF-Q | `field_scope: od` | `field_scope: all` | a same-OD "mean field" is one traveller's own one-hot action ([FIELD-1](CHECKLIST_mfq.md)) |
| LIAM | `graph: od` | `graph: overlap` | a same-OD modelled set is almost all padding, so the decoder reconstructs what the encoder already has ([ADAPT-1](CHECKLIST_liam.md)) |
| ERNIE | `scale_floor: 0.0` | `scale_floor: 1.0` | relative scaling leaves an exactly-zero coordinate unperturbed, i.e. four of five here ([ADAPT-1](CHECKLIST_ernie.md)) |

A fourth consequence is a **finding rather than a fix**: DGN's temporal relation
regulariser compares the attention at day *t* and *t+1*, and on a city where the
node features are constant those two are the same, so the KL is ~0 by
construction. `att_spread` and `reg` are printed so this is visible; on
`ingolstadt_custom` it is live. Report it, do not patch it.

---

## 5. How to verify the work

1. **Read a checklist.** Each row cites the paper equation or the reference file
   and function it came from, and names the line of `urb_baselines/` that
   implements it.
2. **Run the gates.** `python urb_baselines/selftest.py` runs 33 of them in
   about two minutes with no SUMO and no `routerl` (53 with `--arms`). They are of two kinds:
   - *unit* gates check one equation against the paper — that LCPO's step
     respects **both** trust regions, that ERNIE's adversarial perturbation is
     measurably worse than a random one of the same budget (≈3.6×), that DGN's
     `softmax(τ q·k)` matches a hand computation and that masked neighbour slots
     get **exactly** zero attention, that the domain randomiser really redraws
     and really detaches;
   - *drive* gates run every arm for a few hundred days on a 100-line numpy
     stand-in with URB's observation shape, URB's sequential single-step day and
     URB's reward scale, and check it runs, learns, writes loss rows and keeps
     its own invariants;
   - *config* gates check the two things a hand-edited config can silently break:
     **CFG-1**, that every arm's host block is character-identical to
     `ippo/config1.json` or `iql/config1.json` (rule R2 — an arm difference is
     never a tuning difference), and **CFG-2**, that every arm actually
     constructs and acts under its **own shipped** `config1.json`. CFG-2 exists
     because a config and its code can drift apart — a config naming a graph the
     code could not build once raised after the 200-day human-learning phase,
     seventeen minutes into a run — and no other gate here would see it, since
     the drive gates use their own inline configs.
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
| **LILAC** | B7 | SAC-based, continuous; assigned to MAPDN by B7 ("or skip and cite if time is short"). On URB the class is now held by **QCDR** (arm 13), whose own experiments are piecewise-stationary bandits — the same object URB is for every learner — and its latent-per-episode idea is additionally represented by RMA's learned latent and WISDOM's. |
| **TPA-for-AVC** | B11 | Domain-specific to active voltage control; its inputs are power-network quantities. Not portable to routing, and `BASELINES.md` assigns it to MAPDN. |
| **GNN-MAPPO / GRU-MAPPO via BenchMARL** | B2, B3 | BenchMARL is not a URB host. The underlying questions are answered by R-IPPO (memory) and DGN (graph), which are the published methods rather than model flags. |
| **DCG, MAT, QPLEX, AMAGO, MAMBA, MAMBPO, MBCD** | B3, B1, B8, B12, B7 | Tier 3 in `BASELINES.md` §D: cite, run only on request. B12 itself is no longer empty — **M3W** (arm 18) holds it — so MAMBA and MAMBPO are cited as alternatives within a class that now has a representative, rather than as a class nobody ran. |
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
    nets.py            the host MLP (identical to scripts/iql.py's Network),
                       plus RunningNorm, the causal standardiser the scale
                       failures above are fixed with
    context.py         the exogenous driver A(day): observed vs privileged
    records.py         RouteRL's per-day travel times and executed actions
    graph.py           neighbour structures: od / copresence / route overlap
    domain_random.py   per-episode sigma resampling, attached to one env instance
    fakeenv.py         a 100-line numpy stand-in for URB, for the gates
    selftest.py        the offline gates (53 with --arms, 33 without)
    algos/
        reference.py   URB's own PPO and DQN, and the two reference arms
        lcpo.py happo.py ernie.py rippo.py rma.py liam.py
        oracle_ippo.py dr_ippo.py compensator.py eso.py urls.py
        dgn.py mfq.py
        qcdr.py dfp.py pmpg.py doraemon.py wisdom.py m3w.py
scripts/
    lcpo.py happo.py ernie.py rippo.py rma.py liam.py oracle_ippo.py
    dr_ippo.py eso.py urls.py mfq.py dgn.py
    qcdr.py dfp.py pmpg.py doraemon.py wisdom.py m3w.py
    sweep/run_baselines.sh  run_baselines_sigma0.sh  run_baselines_sigma3.sh
config/algo_config/<name>/config1.json      one per baseline, sourced in `desc`
docs/baselines/CHECKLIST_<name>.md          one per baseline
```

Nothing in `urb_baselines/` imports `pact1/`, and nothing in `pact1/` was
touched. `urb_ns/` was not modified either: the severity dial every arm runs
under is byte-for-byte the certified one, and domain randomisation is attached
to a single environment *instance* at run time rather than built into it
(`urb_baselines/domain_random.py` explains why).

## (MA)RL algorithms and baselines.

### (MA)RL algorithms

We deliver here scripts for the experiment runs. Each associated algorithm with selected implementations from `TorchRL`:
* ```ippo_torchrl.py``` uses Independent Proximal Policy Optimization algorithm,
* ```mappo_torchrl.py``` uses Multi Agent Proximal Policy Optimization algorithm,
* ```iql_torchrl.py``` uses Implicit Q-Learning algorithm,
* ```qmix_torchrl.py``` uses QMIX algorithm,
* ```vdn_torchrl.py``` uses Value Decomposition Network algorithm.

Moreover, we have two independent algorithms with our custom implementations:
* ```iql.py``` uses Independent Q-Learning,
* ```ippo.py``` uses Independent Proximal Policy Optimization,
* ```hyp_ippo.py``` uses Independent Proximal Policy Optimization with hypernetworks.

You can tune, adjust, hyperparameterize and modify all the provided implementations, or create own scripts.

### Output conventions

- At the end of each experiment script, metrics are automatically computed by calling `analysis/metrics.py`.
- For learning-based scripts, training losses are saved in a unified CSV file:
  `results/<exp_id>/losses/losses.csv`

### Baselines

In addition to RL algorithms, we provide baseline algorithms for comparison.
They can be executed with ```scripts/baselines.py``` or as standalone scripts ```scripts/<model_name>.py```, depending on the selected model.

The options consist of:

| Method   |  Description                                            | Location          | Execution     | Source          |
| -------- | ------------------------------------------------------- | ----------------- | ------------- | --------------- |
| `greedy` | Selects the route with the lowest recorded travel time based on past episodes. Uses a global structure to store per-agent past records. | `scripts/`     | Run as standalone script: `scripts/greedy.py` | Included in URB 
| `aon`    | Deterministically picks the shortest free-flow route regardless of congestion. | `baseline_models/` | Run via `scripts/baselines.py` with `--model aon` | Included in URB
| `random` | Fully undeterministic. | `baseline_models/` |     Run via `scripts/baselines.py` with  `--model random` | Included in URB |
| `gawron` | Human learning model based on [Gawron (1998)](https://kups.ub.uni-koeln.de/9257/); iteratively shifts cost expectations toward received rewards. | `baseline_models/` | Run via `scripts/baselines.py` with `--model gawron` | [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/993423d101f39ea67a1f7373e6856af95a0602d4/routerl/human_learning/learning_model.py#L42) |

### Published-method baselines (one per class)

Twelve further baselines, one representative per class of method a reviewer
associates with non-stationarity, adaptation, robustness, structure or classical
control. Each is a **separate algorithm** — its own script, its own
`config/algo_config/<name>/`, its own module under `urb_baselines/` — and all of
them run through one shared host so an arm difference cannot be a difference in
how the host was driven.

| script | method | class | base |
| --- | --- | --- | --- |
| `lcpo.py` | LCPO (ICLR 2025) | non-stationary RL, observed context | `ippo.py`'s PPO |
| `happo.py` | HAPPO (ICLR 2022) | trust region / sequential update | `ippo.py`'s PPO |
| `ernie.py` | ERNIE (NeurIPS 2023) | robust MARL | `ippo.py`'s PPO |
| `rippo.py` | R-MAPPO's GRU machinery | memory | `ippo.py`'s PPO |
| `dgn.py` | DGN (ICLR 2020) | graph / communication | `iql.py`'s DQN |
| `mfq.py` | MF-Q (ICML 2018) | mean field | `iql.py`'s DQN |
| `liam.py` | LIAM (NeurIPS 2021) | agent modelling | A2C |
| `rma.py` | RMA / UP-OSI (RSS 2021/2017) | meta-RL / online system ID | `ippo.py`'s PPO |
| `eso.py` | linear ADRC disturbance observer | classical control | non-learning |
| `urls.py` | unstructured RLS on raw peer actions | classical control | non-learning |
| `dr_ippo.py` | domain randomisation over severity | robust | `ippo.py`'s PPO |
| `oracle_ippo.py` | IPPO told the exogenous driver | information arm | `ippo.py`'s PPO |

Run all of them with one command:

```bash
bash scripts/sweep/run_baselines_sigma3.sh 0     # severity row, torch seed 0
bash scripts/sweep/run_baselines_sigma0.sh 0     # the same, dial off
```

Check them without SUMO in about a minute:

```bash
python urb_baselines/selftest.py
```

Every method is audited against its paper and its released code, item by item,
in **[`docs/baselines/`](../docs/baselines/README.md)** — including every
adaptation URB forced and everything deliberately skipped.


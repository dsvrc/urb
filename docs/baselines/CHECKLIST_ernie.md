# CHECKLIST — ERNIE (NeurIPS 2023) on URB

**Paper** Bukharin, Li, Yu, Zhang, Chen, Zuo, Zhang, Zhang, Zhao, *Robust
Multi-Agent Reinforcement Learning via Adversarial Regularization: Theoretical
Foundation and Stable Algorithms*, NeurIPS 2023.
<https://arxiv.org/abs/2310.10810>

**Code read** <https://github.com/abukharin3/ERNIE>
- `README.md` — "the simplest version of ERNIE" (the PGD attack, verbatim)
- `Algorithms/qcombo.py` — `train_step`, `get_adv_reg_loss`, `unroll_perturb`
- `Algorithms/configs/config_qcombo_adv.py` — `lam`, `perturb_alpha`,
  `perturb_num_steps`, `perturb_radius`, `stackelberg`, `normal_perturb`

**BASELINES.md** B9 · Tier 2 · objection row *"Domain randomisation / robust RL"*

**Implementation** `urb_baselines/algos/ernie.py` · **script** `scripts/ernie.py`
· **config** `config/algo_config/ernie/config1.json`

**Arms** `--arm ernie` (Eq. 6, Stackelberg — the paper's method) · `--arm
ernie_no_st` (Eq. 4/5, the paper's own ablation, plotted as "ERNIE w/o ST") ·
`--arm gaussian` (Appendix H's Gaussian baseline) · `--arm ippo` (λ = 0, a wiring
control)

---

## A. The method, item by item

| id | item | source | status | where / why |
|---|---|---|---|---|
| **REG-1** | `R_π(o_k; θ_k) = max_{‖δ‖≤ε} D(π_θ(o_k + δ), π_θ(o_k))` | Eq. 4 | **IMPLEMENTED** | `ErniePPO._regularizer` |
| **REG-2** | `min_θ L(θ) + λ Σ_n E[R_π(o_n; θ_n)]` | Eq. 5 | **IMPLEMENTED** | `ErniePPO.learn`, `loss = loss + self.lam * self._regularizer(states)` |
| **REG-3** | `D` is the **KL divergence** for stochastic policies (MAPPO); an `ℓ_p` norm for deterministic ones (MADDPG, Q-learning) | §5.1 | **IMPLEMENTED (KL)** | `_kl`. URB's on-policy learner has a stochastic categorical policy, so the KL branch is the applicable one. |
| **REG-4** | The regulariser is **per agent**, on that agent's own policy and observation | Appendix E.1–E.3: *"In practice we apply ERNIE to the individual policies"* | **IMPLEMENTED** | one `ErniePPO` per machine agent; the regulariser reads only that agent's minibatch |
| **ATK-1** | The inner max is solved by projected gradient ascent on `δ` | §5.2 | **IMPLEMENTED** | the `for _ in range(self.steps)` loop |
| **ATK-2** | `δ` is initialised from a small Gaussian (`1e-3`) | README; `qcombo.py` | **IMPLEMENTED (verbatim)** | `init_std = 1e-3` |
| **ATK-3** | The ascent step is scaled by `|o|` (relative perturbation) | `qcombo.py`: `+ alpha * grad * torch.abs(old_global_obs)` | **IMPLEMENTED** | `scale = o.abs()`; see ADAPT-1 |
| **ATK-4** | `δ` is bounded in the **ℓ2** norm | Appendix F: *"we use the l_2 norm to bound the attacks δ"* | **IMPLEMENTED** | `_project` normalises `δ/scale` to radius ε in ℓ2 |
| **ATK-5** | `perturb_num_steps = 1` | `config_qcombo_adv.py` | **IMPLEMENTED** | `perturb_steps: 1`; 3 and 5 also gated |
| **ATK-6** | Stackelberg: `δ^K` is the K-fold composition of the ascent operator, and `∂R/∂θ` includes the leader–follower term | Eq. 6 and the gradient decomposition after it | **IMPLEMENTED** | `create_graph=True` when `mode == "ernie"`, so the attack is differentiated through. The paper notes this is equivalent to their Hessian-vector form. |
| **ATK-7** | Non-Stackelberg variant: `δ` detached after the attack | README's "simplest version"; the paper's "ERNIE w/o ST" ablation (Figs. 7, 9) | **IMPLEMENTED** | `--arm ernie_no_st` |
| **ATK-8** | Gaussian baseline: `δ ~ N(0, I)` bounded the same way | Appendix H | **IMPLEMENTED** | `--arm gaussian` |
| **HP-1** | `λ` (regularisation weight) | `config_qcombo_adv.py`: `lam = 0.5` | **IMPLEMENTED** | `lam: 0.5` |
| **HP-2** | `ε` and `λ` are found by **grid search** per environment | Appendix F | **DECLARED, NOT SEARCHED** | `eps: 0.1` in relative units. No search was run and nothing was tuned on returns; the value is declared in the config's `desc` and is one edit to change. |
| **HP-3** | `perturb_alpha` | `config_qcombo_adv.py`: `0.001` (with a *raw*, unnormalised gradient) | **ADAPTED** | `alpha: 0` ⇒ `alpha = eps`. With one PGD step and an ℓ2-normalised direction that is the exact solution of the linearised inner maximisation (the ℓ2 FGSM attack). The released `0.001` belongs to their un-normalised step and does not carry over. |
| **BASE-1** | ERNIE is added to an existing learner; everything else is unchanged | §5.1, Appendix E | **IMPLEMENTED** | `ErniePPO` subclasses `SingleStepPPO` and overrides only `learn`; `--arm ippo` sets λ = 0 and must reproduce `scripts/ippo.py` up to RNG |
| **SKIP-1** | §5.3 — robustness to malicious **actions**: `R^A_ω(s, a) = max_{D(a,a')≤K} ‖Q(s,a;ω) − Q(s,a';ω)‖²` on a **global** Q over joint actions (Eq. 7, Algorithm 1) | §5.3, Appendix G | **N/A** | URB's on-policy host is actor-only and per-agent. There is no global Q over joint actions to regularise, and inventing one would be a different method. Not approximated. |
| **SKIP-2** | §5.4 — the mean-field extension: a Wasserstein-bounded regulariser on the mean-field state `d_s` | §5.4, Appendix B | **N/A** | Same reason: it regularises a mean-field Q. (URB *does* have a mean-field arm — `scripts/mfq.py` — but combining the two would be a new method, not this baseline.) |

---

## B. Adaptations, in full

### ADAPT-1 — relative perturbation scaling, and the one consequence
The released code perturbs by `alpha * grad * torch.abs(obs)`, i.e. each
coordinate is perturbed in proportion to its own magnitude. URB needs this more
than their traffic-grid does: its observation is
`[start_time_in_seconds, four route counts]`, mixing a quantity of order 1000
with quantities of order 1. An absolute ε would be a rounding error on the first
coordinate and an enormous perturbation on the others.

The consequence, stated plainly: a coordinate that is **exactly zero receives no
perturbation**. That is the released behaviour, not a choice made here.
`relative: false` switches to absolute units for anyone who wants the other
trade-off.

### ADAPT-2 — ℓ2 ball in relative units
Appendix F bounds `δ` in ℓ2; the released code scales by `|o|`. The two are
combined the only way that keeps both: the constraint is `‖δ / scale‖₂ ≤ ε`.
Gate **ERNIE-2** checks it holds for 1, 3 and 5 PGD steps.

---

## C. Gates

| gate | what it proves |
|---|---|
| **ERNIE-1** | On a *committed* policy, the adversarial `δ` produces a KL **3.6×** larger than a random `δ` of the same budget (7.4e-4 vs 2.0e-4), and the detached variant also beats random. If the inner maximisation were broken, this arm would be Gaussian smoothing wearing ERNIE's name — and nothing else would show it. |
| **ERNIE-2** | `‖δ / |o|‖₂ ≤ ε` for 1, 3 and 5 PGD steps. |
| **drive ernie**, **drive ernie --arm gaussian** | Both arms run, learn and write loss rows. |

---

## D. What to watch in a real run

Two diagnostics, and they must be read **together**:

- `reg` — the realised regulariser `KL(π(o+δ) ‖ π(o))`.
- `pol_ent` — the policy's entropy on the same batch.

A near-zero `reg` on a **near-deterministic** policy (`pol_ent → 0`) means *there
is nothing left to perturb*, not that the attack failed: a point mass has KL ≈ 0
to any small perturbation. URB's host IPPO does collapse toward a deterministic
policy (the entropy coefficient is 0.01 and the advantage is a standardised
reward with no baseline), so expect this, and expect ERNIE to track IPPO closely
when it happens. Reporting it is the point of logging both.

**Prediction under test** (BASELINES.md B9): *"robustness costs σ=0 performance
and still does not cancel; the asymptote lies between blind and PACT."*

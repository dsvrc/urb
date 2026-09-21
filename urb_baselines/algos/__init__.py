"""One module per baseline. See ``docs/baselines/README.md`` for the register.

    reference.py   URB's own on-policy (PPO) and off-policy (DQN) bases, and the
                   IPPO / IQL reference algorithms used by the host parity gate.

  on-policy (built on the PPO base of ``scripts/ippo.py``)
    lcpo.py        LCPO / LCPPO   -- ICLR 2025, non-stationary observed context
    happo.py       HAPPO          -- ICLR 2022, sequential trust region
    ernie.py       ERNIE          -- NeurIPS 2023, adversarial regularisation
    rippo.py       R-IPPO (GRU)   -- the R-MAPPO recurrent machinery
    rma.py         RMA / UP-OSI   -- teacher-student online system identification
    liam.py        LIAM           -- NeurIPS 2021, agent modelling
    oracle_ippo.py oracle-driver  -- the information arm: IPPO told A(t)
    dr_ippo.py     domain rand.   -- B9's "must": IPPO under resampled sigma
    eso.py         ESO / DOB      -- linear ADRC disturbance observer, no peers
    urls.py        unstructured RLS on the raw peer actions
    dfp.py         Deep FP        -- NeurIPS 2025, fictitious play / mean field
    pmpg.py        INPG + retrain -- performative Markov potential games, 2025
    doraemon.py    DORAEMON       -- ICLR 2024, entropy-maximising domain rand.
    wisdom.py      WISDOM         -- 2025, wavelet predictive representations

  off-policy (built on the DQN base of ``scripts/iql.py``)
    dgn.py         DGN            -- ICLR 2020, graph convolutional RL
    mfq.py         MF-Q           -- ICML 2018, mean-field Q-learning
    qcdr.py        QCD restart    -- 2024, change detection + restart bandits
    m3w.py         M3W            -- NeurIPS 2025, MoE world model + MPPI
"""

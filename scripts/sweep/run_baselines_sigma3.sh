#!/usr/bin/env bash
# The SEVERITY row: sigma = 3, a labelled beyond-physical stress test (sigma = 1
# is the HCM heavy-rain capacity adjustment factor; 3 is three times that).
#
#   bash scripts/sweep/run_baselines_sigma3.sh          all seeds, seed-major
#   bash scripts/sweep/run_baselines_sigma3.sh 2        this process: seed 2 only
#
# Identical to run_baselines_sigma0.sh in every respect except this line;
# everything else lives in run_baselines.sh so the two arms cannot drift apart.
export SIGMA=3
exec "$(dirname "${BASH_SOURCE[0]}")/run_baselines.sh" "$@"

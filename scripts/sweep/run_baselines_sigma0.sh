#!/usr/bin/env bash
# The NO-SEVERITY row: sigma = 0 gives the stock URB task byte for byte, so this
# is the same command that produces the severity table, with the dial off.
#
#   bash scripts/sweep/run_baselines_sigma0.sh          all seeds, seed-major
#   bash scripts/sweep/run_baselines_sigma0.sh 2        this process: seed 2 only
#
# Identical to run_baselines_sigma3.sh in every respect except this line.
export SIGMA=0
exec "$(dirname "${BASH_SOURCE[0]}")/run_baselines.sh" "$@"

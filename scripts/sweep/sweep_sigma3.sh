#!/usr/bin/env bash
# The SEVERITY arm: sigma = 3, a labelled beyond-physical stress test
# (sigma = 1 is the HCM heavy-rain capacity adjustment factor; 3 is 3x that).
#
#   bash scripts/sweep/sweep_sigma3.sh
#
# Identical to sweep_sigma0.sh in every respect except this line. Everything
# else lives in run_sweep.sh so the two arms cannot drift apart.
export SIGMA=3
exec "$(dirname "${BASH_SOURCE[0]}")/run_sweep.sh"

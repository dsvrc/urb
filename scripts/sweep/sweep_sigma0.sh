#!/usr/bin/env bash
# The CONTROL arm: sigma = 0, the stock URB task recovered byte for byte.
#
#   bash scripts/sweep/sweep_sigma0.sh
#
# Still launched through ns_launch.py rather than calling the scripts directly.
# That is deliberate: at sigma = 0 the wrapper short-circuits and every reward
# passes through untouched, so the task is identical to stock URB -- but the
# sys.path setup, the config resolution and the class the scripts construct are
# then identical to the severity arm too, so the two tables differ in the dial
# and in nothing else.
export SIGMA=0
exec "$(dirname "${BASH_SOURCE[0]}")/run_sweep.sh"

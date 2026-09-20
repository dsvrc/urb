#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# THE baseline runner. One command runs every baseline in docs/baselines/.
#
#   bash scripts/sweep/run_baselines_sigma3.sh          all seeds, seed-major
#   bash scripts/sweep/run_baselines_sigma3.sh 0        this process: seed 0
#   bash scripts/sweep/run_baselines_sigma0.sh 0        the no-severity row
#
# SIGMA is supplied by the two wrappers and is the ONLY thing that differs
# between them -- that is the whole point: any difference between the sigma=0
# and sigma=3 tables must be the dial and nothing else. Do not run this file
# directly.
#
# Every arm goes through scripts/ns_launch.py under the identical dial, the
# identical route table, the identical task config and the identical env seed,
# exactly as scripts/sweep/run_sweep.sh does for PACT-1 and the stock URB
# algorithms. Run that sweep too: these baselines are only meaningful beside it.
#
# Overridable from the environment:
#   NET ENV_SEED TORCH_SEEDS TASK_CONF ENV_CONF ALG_CONF ROUTES PY
#   TIER=1|2|all       which set to run (default 1; see docs/baselines/README.md)
#   ARMS=1             also run each baseline's declared ablation arms
#   ONLY="lcpo_0 eso_0"    run only these slugs
#   SKIP="dgn_0"           exclude these slugs
#   DEVICE=cpu|cuda|cuda:N|auto   where the networks run (default auto)
#   SELFTEST=0         skip the offline gate run (NOT recommended)
#   DRYRUN=1           print the commands, run nothing
# ---------------------------------------------------------------------------
set -uo pipefail        # deliberately NOT -e: one failed arm must not kill the run

# ...but an INTERRUPTION must. Without this the loop keeps launching arms after
# a Ctrl-C or a SIGTERM, each one is killed the instant it starts, and the
# summary reports eight "failures" that were really one interruption. Observed
# exactly that; see also the rc 130/143 check in run().
INTERRUPTED=0
on_signal () {
    INTERRUPTED=1
    echo
    echo "*** interrupted (signal) -- stopping the sweep. Completed arms keep"
    echo "*** their .done markers, so re-running resumes from here."
    exit 130
}
trap on_signal INT TERM

: "${SIGMA:?SIGMA must be set by a wrapper (run_baselines_sigma0.sh / _sigma3.sh)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

NET="${NET:-saint_arnoult}"
ENV_SEED="${ENV_SEED:-42}"
if [[ $# -gt 0 ]]; then TORCH_SEEDS="$*"; else TORCH_SEEDS="${TORCH_SEEDS:-0 1 2 3 4}"; fi
TASK_CONF="${TASK_CONF:-config1}"
ENV_CONF="${ENV_CONF:-config1}"
ALG_CONF="${ALG_CONF:-config1}"
TIER="${TIER:-1}"
ARMS="${ARMS:-0}"
DRYRUN="${DRYRUN:-0}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
SELFTEST="${SELFTEST:-1}"
DEVICE="${DEVICE:-auto}"

# DEVICE=cpu is enforced the blunt way as well as the polite way. The polite way
# is --device, which only this package's scripts understand. The blunt way hides
# the GPU from CUDA entirely, which ALSO fixes scripts/pact1.py, scripts/ippo.py
# and every other stock URB script -- all of them choose the device with the
# same `torch.device(0) if torch.cuda.is_available()` line and would otherwise
# crash on a GPU this torch build has no kernels for.
if [[ "$DEVICE" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    echo "[runner] DEVICE=cpu -> CUDA_VISIBLE_DEVICES='' (applies to every script)"
fi

# Dial parameters -- ns_launch.py's own defaults, passed explicitly so that
# every run's log documents the dial it actually ran under.
PERIOD="${PERIOD:-100}"
WET_FRAC="${WET_FRAC:-0.5}"
LOSS="${LOSS:-0.14}"
SAT_FLOW="${SAT_FLOW:-600}"

# The route table. PINNED ON PURPOSE, and it must be the SAME table
# run_sweep.sh used, or the baselines and PACT-1 are aimed at different route
# sets and the two tables cannot be put beside each other.
ROUTEGEN_ID="routegen_${NET}_${ENV_SEED}"
ROUTES_WAS_EXPLICIT=0
if [[ -n "${ROUTES:-}" ]]; then ROUTES_WAS_EXPLICIT=1; fi
ROUTES="${ROUTES:-../results/${ROUTEGEN_ID}/routes.csv}"
ROUTES_ABS="$REPO_ROOT/${ROUTES#../}"

TAG="bl_s${SIGMA%%.*}"
LOGDIR="$REPO_ROOT/logs/$TAG"
mkdir -p "$LOGDIR"

# ---------------------------------------------------------------- preflight
if [[ ! -f scripts/ns_launch.py ]]; then
    echo "FATAL: not a URB checkout: $REPO_ROOT" >&2
    exit 1
fi

if [[ -z "${PY:-}" ]]; then
    for cand in python python3 py; do
        if command -v "$cand" >/dev/null 2>&1 \
           && "$cand" -c "import numpy" >/dev/null 2>&1; then
            PY="$cand"; break
        fi
    done
fi
if [[ -z "${PY:-}" ]]; then
    echo "FATAL: no usable python found. Set PY=/path/to/python." >&2
    exit 1
fi

if [[ "$DRYRUN" != "1" ]] && ! "$PY" -c "import routerl, torch, pandas" >/dev/null 2>&1; then
    echo "FATAL: '$PY' cannot import the URB runtime stack." >&2
    "$PY" -c "import routerl, torch, pandas" 2>&1 | tail -3 | sed 's/^/       /' >&2
    exit 1
fi

# The offline gates. They take about a minute and they are the difference
# between finding a transposed attention tensor now and finding it in six hours.
if [[ "$SELFTEST" == "1" && "$DRYRUN" != "1" ]]; then
    echo "--- offline gates (no SUMO; see urb_baselines/selftest.py) ----------------"
    if ! "$PY" urb_baselines/selftest.py --days 150; then
        echo "FATAL: the baseline self-test failed. Fix that before spending a run." >&2
        exit 1
    fi
    echo
fi

echo "==========================================================================="
echo " URB BASELINES   sigma=$SIGMA   net=$NET   env-seed=$ENV_SEED (FIXED)"
echo " torch seeds : $TORCH_SEEDS"
echo " tier        : $TIER      extra ablation arms: $ARMS"
echo " configs     : alg=$ALG_CONF task=$TASK_CONF env=$ENV_CONF"
echo " route table : $ROUTES_ABS"
echo " logs        : $LOGDIR"
echo " started     : $(date -Iseconds 2>/dev/null || date)"
echo "==========================================================================="

# --- route table: use it, or build it once ---------------------------------
if [[ ! -f "$ROUTES_ABS" ]]; then
    if [[ "$ROUTES_WAS_EXPLICIT" == "1" ]]; then
        echo "FATAL: ROUTES was set explicitly but does not exist: $ROUTES_ABS" >&2
        exit 1
    fi
    if [[ "$DRYRUN" == "1" ]]; then
        echo "[bootstrap] would generate the route table at $ROUTES_ABS (DRYRUN)"
    else
        echo "--- bootstrapping the pinned route table ---------------------------------"
        echo "    generating via pact1.py --mode dry (path generation only)."
        echo "    Deterministic in (net=$NET, env-seed=$ENV_SEED, env-conf=$ENV_CONF),"
        echo "    so run_sweep.sh and this runner share one table."
        "$PY" scripts/pact1.py --id "$ROUTEGEN_ID" --alg-conf config1 \
            --task-conf "$TASK_CONF" --env-conf "$ENV_CONF" --net "$NET" \
            --env-seed "$ENV_SEED" --mode dry \
            > "$REPO_ROOT/logs/routegen.log" 2>&1
        if [[ ! -f "$ROUTES_ABS" ]]; then
            echo "FATAL: route generation did not produce $ROUTES_ABS" >&2
            tail -n 25 "$REPO_ROOT/logs/routegen.log" | sed 's/^/       | /' >&2
            exit 1
        fi
        echo "    done: $ROUTES_ABS"
        echo
    fi
fi

if [[ "$SIGMA" != "0" ]]; then
    echo "--- dial certification (no SUMO, no training) ---------------------------"
    if ! "$PY" scripts/sweep/certify_dial.py --sigma "$SIGMA" \
            --period "$PERIOD" --wet-frac "$WET_FRAC" --loss "$LOSS"; then
        echo "FATAL: the driver failed its own B.1/B.4 certification" >&2
        exit 1
    fi
    echo
fi

# ------------------------------------------------------------------- runner
FAILED=(); INERT=(); DEGENERATE=(); DONE=0; SKIPPED=0

wanted () {
    local slug="$1"
    if [[ -n "$SKIP" && " $SKIP " == *" $slug "* ]]; then return 1; fi
    if [[ -n "$ONLY" && " $ONLY " != *" $slug "* ]]; then return 1; fi
    return 0
}

run () {                        # run <slug> <target.py> [extra target args...]
    local slug="$1"; shift
    local script="$1"; shift
    # Once something has killed a run, every later call returns without
    # launching anything, so the loop below needs no change at its call sites.
    [[ "$INTERRUPTED" == "1" ]] && return 1
    wanted "$slug" || return 0

    local id="${TAG}_${slug}"
    local log="$LOGDIR/${slug}.log"

    if [[ -f "$log.done" ]]; then
        echo "SKIP   $id   (marker exists: $log.done)"
        SKIPPED=$((SKIPPED + 1)); return 0
    fi

    local cmd=( "$PY" scripts/ns_launch.py
        --sigma "$SIGMA" --net "$NET" --env-conf "$ENV_CONF" --routes "$ROUTES"
        --period "$PERIOD" --wet-frac "$WET_FRAC" --loss "$LOSS"
        --sat-flow "$SAT_FLOW" --av-behavior selfish
        -- "$script" --id "$id" --alg-conf "$ALG_CONF" --task-conf "$TASK_CONF"
           --net "$NET" --env-seed "$ENV_SEED" --routes "$ROUTES"
           --device "$DEVICE" "$@" )

    if [[ "$DRYRUN" == "1" ]]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi

    echo "RUN    $id   -> $log"
    local t0=$SECONDS
    "${cmd[@]}" > "$log" 2>&1
    local rc=$?
    local mins=$(( (SECONDS - t0) / 60 ))

    # 130 = SIGINT, 143 = SIGTERM. Neither is an algorithm failure: something
    # killed the run. Marching on would launch every remaining arm into the same
    # thing and report a list of failures that were one interruption.
    if [[ $rc -eq 130 || $rc -eq 143 ]]; then
        echo "ABORT  $id   killed by a signal (rc=$rc) after ${mins}m"
        echo "       Stopping the sweep. No .done marker was written, so"
        echo "       re-running this command retries this arm and the rest."
        INTERRUPTED=1
        return 1
    fi

    if [[ $rc -ne 0 ]]; then
        echo "FAIL   $id   rc=$rc after ${mins}m"
        tail -n 15 "$log" | sed 's/^/       | /'
        FAILED+=( "$id" ); return 0
    fi

    # A silently inert dial looks exactly like a clean null result.
    if [[ "$SIGMA" != "0" ]] && grep -qE \
        "THE DIAL NEVER FIRED|pass through UNHARMED|RECORDS were never harmed" "$log"; then
        echo "INERT  $id   the dial did not reach rewards/records -- NOT a severity arm"
        INERT+=( "$id" ); return 0
    fi

    # Every arm in this package prints a *** line when its own mechanism never
    # engaged. Those are the rows that must not be reported as that method.
    if grep -qE "^  \*\*\*" "$log"; then
        echo "DEGEN  $id   an arm mechanism reported itself inert:"
        grep -E "^  \*\*\*" "$log" | head -4 | sed 's/^/       | /'
        DEGENERATE+=( "$id" )
    fi

    if [[ -f "$REPO_ROOT/results/$id/routes.csv" ]] \
       && ! cmp -s "$REPO_ROOT/results/$id/routes.csv" "$ROUTES_ABS"; then
        echo "WARN   $id   generated routes.csv differs from the pinned table"
    fi

    echo "OK     $id   ${mins}m"
    : > "$log.done"; DONE=$((DONE + 1))
}

# ---------------------------------------------------------------------------
# TIER 1 -- the set docs/baselines/README.md calls the must-run column.
# TIER 2 -- everything else in the package.
# The ablation arms (ARMS=1) are each baseline's OWN declared ablation, not
# extra methods: lcpo/a2c isolates the local constraint from the critic,
# ernie/ernie_no_st is the paper's "w/o ST", dgn/dgn_r is the paper's "DGN-R",
# rma/oa is RMA's literal history input.
# ---------------------------------------------------------------------------
for S in $TORCH_SEEDS; do
    if [[ "$TIER" == "1" || "$TIER" == "all" ]]; then
        run "lcpo_$S"        lcpo.py        --torch-seed "$S"
        run "oracle_ippo_$S" oracle_ippo.py --torch-seed "$S"
        run "dr_ippo_$S"     dr_ippo.py     --torch-seed "$S"
        run "rippo_$S"       rippo.py       --torch-seed "$S"
        run "eso_$S"         eso.py         --torch-seed "$S"
        run "urls_$S"        urls.py        --torch-seed "$S"
        run "dgn_$S"         dgn.py         --torch-seed "$S"
        run "mfq_$S"         mfq.py         --torch-seed "$S"
    fi
    if [[ "$TIER" == "2" || "$TIER" == "all" ]]; then
        run "happo_$S"       happo.py       --torch-seed "$S"
        run "ernie_$S"       ernie.py       --torch-seed "$S"
        run "rma_$S"         rma.py         --torch-seed "$S"
        run "liam_$S"        liam.py        --torch-seed "$S"
    fi
    if [[ "$ARMS" == "1" ]]; then
        run "lcpo_a2c_$S"    lcpo.py        --torch-seed "$S" --arm a2c
        run "lcppo_$S"       lcpo.py        --torch-seed "$S" --arm lcppo
        run "ernie_nost_$S"  ernie.py       --torch-seed "$S" --arm ernie_no_st
        run "ernie_gauss_$S" ernie.py       --torch-seed "$S" --arm gaussian
        run "dgn_r_$S"       dgn.py         --torch-seed "$S" --arm dgn_r
        run "eso2_$S"        eso.py         --torch-seed "$S" --arm eso2
        run "rma_oa_$S"      rma.py         --torch-seed "$S" --history-features oa
        run "mfq_all_$S"     mfq.py         --torch-seed "$S" --field-scope all
    fi
done

# ------------------------------------------------------------------ summary
echo
echo "==========================================================================="
echo " baselines sigma=$SIGMA finished $(date -Iseconds 2>/dev/null || date)"
if [[ "$INTERRUPTED" == "1" ]]; then
echo "   *** INTERRUPTED: the remaining arms were not started. Completed arms"
echo "   *** kept their .done markers; re-run the same command to resume."
fi
echo "   completed $DONE   skipped $SKIPPED   failed ${#FAILED[@]}"
echo "   inert ${#INERT[@]}   degenerate ${#DEGENERATE[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then printf '   FAILED: %s\n' "${FAILED[*]}"; fi
if [[ ${#INERT[@]}  -gt 0 ]]; then
    printf '   INERT (do NOT report as severity arms): %s\n' "${INERT[*]}"; fi
if [[ ${#DEGENERATE[@]} -gt 0 ]]; then
    printf '   DEGENERATE (the arm said its own mechanism never engaged --\n'
    printf '               read the *** lines in its log before reporting): %s\n' \
        "${DEGENERATE[*]}"; fi
echo "   results in results/${TAG}_*,  logs in $LOGDIR"
echo "==========================================================================="

[[ ${#FAILED[@]} -eq 0 && ${#INERT[@]} -eq 0 ]]

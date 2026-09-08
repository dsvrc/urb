#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Common sweep runner. SIGMA is supplied by the two wrappers and is the ONLY
# thing that differs between them -- that is the whole point: any difference
# between the sigma=0 and sigma=3 tables must be the dial and nothing else.
#
# Do not run this directly. Run sweep_sigma0.sh or sweep_sigma3.sh.
#
# Overridable from the environment:
#   NET ENV_SEED TORCH_SEEDS TASK_CONF ENV_CONF ALG_CONF ROUTES PY
#   ONLY="pact1_0 qmix_0"   run only these slugs
#   SKIP="iql_trl_3"        exclude these slugs
#   REFS=0                  skip the non-learning reference baselines
#   EXTRA=1                 also run hyp_ippo and centralized_dqn (different
#                           episode budgets -- see the note where they are run)
#   DRYRUN=1                print the commands, run nothing
# ---------------------------------------------------------------------------
set -uo pipefail        # deliberately NOT -e: one failed arm must not kill the sweep

: "${SIGMA:?SIGMA must be set by a wrapper (0 or 3)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

NET="${NET:-saint_arnoult}"
ENV_SEED="${ENV_SEED:-42}"
TORCH_SEEDS="${TORCH_SEEDS:-0 1 2 3 4}"
TASK_CONF="${TASK_CONF:-config1}"
ENV_CONF="${ENV_CONF:-config1}"
ALG_CONF="${ALG_CONF:-config1}"
REFS="${REFS:-1}"
DRYRUN="${DRYRUN:-0}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"

# Dial parameters. These are ns_launch.py's own defaults, passed explicitly so
# that every run's log documents the dial it actually ran under.
PERIOD="${PERIOD:-100}"
WET_FRAC="${WET_FRAC:-0.5}"
LOSS="${LOSS:-0.14}"
SAT_FLOW="${SAT_FLOW:-600}"

# The route table the severity layer reads. PINNED ON PURPOSE.
# ns_launch.py's fallback picks the most recently MODIFIED routes.csv anywhere
# under results/ -- including one belonging to a different city -- and every run
# writes a new one. An unpinned sweep therefore aims the dial at a different
# route set on almost every launch. The path is relative to scripts/, because
# ns_launch.py chdir()s there before resolving it.
ROUTES="${ROUTES:-../results/sample_results/routes.csv}"
ROUTES_ABS="$REPO_ROOT/${ROUTES#../}"

TAG="s${SIGMA%%.*}"
LOGDIR="$REPO_ROOT/logs/$TAG"
mkdir -p "$LOGDIR"

# ---------------------------------------------------------------- preflight
if [[ ! -f scripts/ns_launch.py ]]; then
    echo "FATAL: not a URB checkout: $REPO_ROOT" >&2
    exit 1
fi

# Interpreter. On Git Bash under Windows a bare `python` often resolves to a
# broken shim ahead of the real install on PATH, so the first candidate that can
# actually import the stack wins. Override with PY=/path/to/python.
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

# Fail now rather than 104 times in a row an hour from now. Skipped under
# DRYRUN, whose whole purpose is to print the plan on a machine that cannot run it.
if [[ "$DRYRUN" != "1" ]] && ! "$PY" -c "import routerl, torch, torchrl, pandas" >/dev/null 2>&1; then
    echo "FATAL: '$PY' cannot import the URB runtime stack." >&2
    "$PY" -c "import routerl, torch, torchrl, pandas" 2>&1 | tail -3 | sed 's/^/       /' >&2
    echo "       Install it (pip install -r requirements.txt) or point PY at the" >&2
    echo "       interpreter that has it: PY=/path/to/python bash $0" >&2
    exit 1
fi

if [[ ! -f "$ROUTES_ABS" ]]; then
    echo "FATAL: pinned route table missing: $ROUTES_ABS" >&2
    echo "       Generate one first -- path generation only, no AV training:" >&2
    echo "         python scripts/pact1.py --id routegen --alg-conf config1 \\" >&2
    echo "             --task-conf $TASK_CONF --net $NET --env-seed $ENV_SEED --mode dry" >&2
    echo "       then re-run this sweep with ROUTES=../results/routegen/routes.csv" >&2
    exit 1
fi

echo "==========================================================================="
echo " URB sweep   sigma=$SIGMA   net=$NET   env-seed=$ENV_SEED (FIXED)"
echo " torch seeds : $TORCH_SEEDS"
echo " configs     : alg=$ALG_CONF task=$TASK_CONF env=$ENV_CONF"
echo " route table : $ROUTES_ABS"
echo " logs        : $LOGDIR"
echo " started     : $(date -Iseconds 2>/dev/null || date)"
echo "==========================================================================="

if [[ "$SIGMA" != "0" ]]; then
    echo
    echo "--- dial certification (no SUMO, no training) ---------------------------"
    if ! "$PY" scripts/sweep/certify_dial.py --sigma "$SIGMA" \
            --period "$PERIOD" --wet-frac "$WET_FRAC" --loss "$LOSS"; then
        echo "FATAL: the driver failed its own B.1/B.4 certification" >&2
        exit 1
    fi
    echo
fi

# ------------------------------------------------------------------- runner
FAILED=()
INERT=()
DONE=0
SKIPPED=0

wanted () {                     # slug -> is it selected by ONLY / SKIP?
    local slug="$1"
    if [[ -n "$SKIP" && " $SKIP " == *" $slug "* ]]; then return 1; fi
    if [[ -n "$ONLY" && " $ONLY " != *" $slug "* ]]; then return 1; fi
    return 0
}

run () {                        # run <slug> <target.py> [extra target args...]
    local slug="$1"; shift
    local script="$1"; shift
    wanted "$slug" || return 0

    local id="${TAG}_${slug}"
    local log="$LOGDIR/${slug}.log"

    if [[ -f "$log.done" ]]; then
        echo "SKIP   $id   (marker exists: $log.done)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    local cmd=( "$PY" scripts/ns_launch.py
        --sigma "$SIGMA" --net "$NET" --env-conf "$ENV_CONF" --routes "$ROUTES"
        --period "$PERIOD" --wet-frac "$WET_FRAC" --loss "$LOSS"
        --sat-flow "$SAT_FLOW" --av-behavior selfish
        -- "$script" --id "$id" --alg-conf "$ALG_CONF" --task-conf "$TASK_CONF"
           --net "$NET" --env-seed "$ENV_SEED" "$@" )

    if [[ "$DRYRUN" == "1" ]]; then
        printf '%q ' "${cmd[@]}"; echo
        return 0
    fi

    echo "RUN    $id   -> $log"
    local t0=$SECONDS
    "${cmd[@]}" > "$log" 2>&1
    local rc=$?
    local mins=$(( (SECONDS - t0) / 60 ))

    if [[ $rc -ne 0 ]]; then
        echo "FAIL   $id   rc=$rc after ${mins}m"
        tail -n 15 "$log" | sed 's/^/       | /'
        FAILED+=( "$id" )
        return 0
    fi

    # A silently inert dial is the one failure mode that looks exactly like a
    # clean null result, so it is checked rather than trusted.
    if [[ "$SIGMA" != "0" ]] && grep -qE \
        "THE DIAL NEVER FIRED|pass through UNHARMED|RECORDS were never harmed" "$log"; then
        echo "INERT  $id   the dial did not reach rewards/records -- NOT a severity arm"
        grep -nE "THE DIAL NEVER FIRED|pass through UNHARMED|RECORDS were never harmed" \
            "$log" | head -3 | sed 's/^/       | /'
        INERT+=( "$id" )
        return 0
    fi

    # The dial and (for PACT-1) the identification basis must describe the SAME
    # route set as the one the fleet actually drove.
    if [[ -f "$REPO_ROOT/results/$id/routes.csv" ]] \
       && ! cmp -s "$REPO_ROOT/results/$id/routes.csv" "$ROUTES_ABS"; then
        echo "WARN   $id   generated routes.csv differs from the pinned table --"
        echo "       the dial is aimed at a different route set than was driven."
        echo "       Investigate before reporting this row."
    fi

    echo "OK     $id   ${mins}m"
    : > "$log.done"
    DONE=$((DONE + 1))
}

# ------------------------------------------------------------ learning arms
# 4000 AV training days + 100 test days each, so every arm sees exactly 40 full
# weather cycles and a test phase spanning one clean wet/dry period.
for S in $TORCH_SEEDS; do
    run "pact1_$S"       pact1.py         --torch-seed "$S" --arm pact1
    run "pact1_blind_$S" pact1.py         --torch-seed "$S" --arm blind
    run "pact1_fixed_$S" pact1.py         --torch-seed "$S" --arm fixed
    run "ippo_$S"        ippo.py          --torch-seed "$S"
    run "iql_$S"         iql.py           --torch-seed "$S"
    run "mappo_$S"       mappo_torchrl.py --torch-seed "$S"
    run "qmix_$S"        qmix_torchrl.py  --torch-seed "$S"
    run "vdn_$S"         vdn_torchrl.py   --torch-seed "$S"
    run "ippo_trl_$S"    ippo_torchrl.py  --torch-seed "$S"
    run "iql_trl_$S"     iql_torchrl.py   --torch-seed "$S"

    # Opt-in (EXTRA=1). Kept out of the default sweep because their config1
    # episode budgets differ from the 4000 above -- hyp_ippo trains for 1000 and
    # centralized_dqn for 2000 plus a buffer-filling phase -- so they see a
    # different number of weather cycles and do not belong in the same column
    # as the arms above without saying so.
    if [[ "${EXTRA:-0}" == "1" ]]; then
        run "hyp_ippo_$S" hyp_ippo.py       --torch-seed "$S"
        run "cdqn_$S"     centralized_dqn.py --torch-seed "$S"
    fi
done

# ---------------------------------------------------------- reference arms
# No --torch-seed: baselines.py and greedy.py seed everything from --env-seed
# alone, so five "seeds" here would be five byte-identical runs. Varying
# --env-seed instead would regenerate the route set and break the pinned table
# that makes the sigma=0 / sigma=3 comparison exact. One run each, by design.
if [[ "$REFS" == "1" ]]; then
    run "aon"    baselines.py --model aon
    run "random" baselines.py --model random
    run "gawron" baselines.py --model gawron
    run "greedy" greedy.py
fi

# ------------------------------------------------------------------ summary
echo
echo "==========================================================================="
echo " sigma=$SIGMA finished $(date -Iseconds 2>/dev/null || date)"
echo "   completed $DONE   skipped $SKIPPED   failed ${#FAILED[@]}   inert ${#INERT[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then printf '   FAILED: %s\n' "${FAILED[*]}"; fi
if [[ ${#INERT[@]}  -gt 0 ]]; then printf '   INERT (do NOT report as severity arms): %s\n' "${INERT[*]}"; fi
echo "==========================================================================="

[[ ${#FAILED[@]} -eq 0 && ${#INERT[@]} -eq 0 ]]

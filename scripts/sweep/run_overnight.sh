#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Every outstanding verification and ablation, in one resumable overnight run.
# Each step writes ONE file into evidence/, plus a status line in evidence/INDEX.md.
#
#     bash scripts/sweep/run_overnight.sh              # tiers 0 and 1 (fits a night)
#     bash scripts/sweep/run_overnight.sh --dry        # print the plan, run nothing
#     TIERS="0" bash scripts/sweep/run_overnight.sh    # offline only, ~3 hours
#     TIERS="2" bash scripts/sweep/run_overnight.sh    # the training ablations (DAYS)
#
# Resumable: a step that finished leaves <file>.done and is skipped next time.
# Delete the marker to force a re-run of one step.
#
# Environment:
#   TIERS="0 1"     which tiers to run (0 offline, 1 cheap SUMO, 2 training)
#   EVIDENCE=dir    output directory (default evidence/)
#   NET, ENV_SEED, TASK_CONF, ENV_CONF, ROUTES, PY   as in run_sweep.sh
#   PAPER_DIR=dir   where paper/verification lives (default ../paper)
#   SPECTRAL_SWEEPS number of best-response sweeps for the PoA re-run (default 200)
#   TRUST_SEEDS     seeds for the tier-2 fixed-trust sweep (default "0")
# ---------------------------------------------------------------------------
set -uo pipefail          # NOT -e: one failed step must not kill the night

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

TIERS="${TIERS:-0 1}"
EVIDENCE="${EVIDENCE:-$REPO_ROOT/evidence}"
NET="${NET:-saint_arnoult}"
ENV_SEED="${ENV_SEED:-42}"
TASK_CONF="${TASK_CONF:-config1}"
ENV_CONF="${ENV_CONF:-config1}"
if [[ -z "${PAPER_DIR:-}" ]]; then
    for cand in "$REPO_ROOT/../paper" "$REPO_ROOT/../../paper" "$REPO_ROOT/paper"; do
        [[ -d "$cand/verification" ]] && { PAPER_DIR="$cand"; break; }
    done
fi
PAPER_DIR="${PAPER_DIR:-$REPO_ROOT/../paper}"
SPECTRAL_SWEEPS="${SPECTRAL_SWEEPS:-200}"
TRUST_SEEDS="${TRUST_SEEDS:-0}"
DRY=0
[[ "${1:-}" == "--dry" ]] && DRY=1

mkdir -p "$EVIDENCE"

# ---------------------------------------------------------------- interpreter
if [[ -z "${PY:-}" ]]; then
    for cand in python python3 py; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import numpy" >/dev/null 2>&1; then
            PY="$cand"; break
        fi
    done
fi
[[ -z "${PY:-}" ]] && { echo "FATAL: no usable python; set PY=/path/to/python" >&2; exit 1; }

HAS_SUMO=1
"$PY" -c "import routerl" >/dev/null 2>&1 || HAS_SUMO=0

# ---------------------------------------------------------------- route table
if [[ -z "${ROUTES:-}" ]]; then
    for cand in "../results/routegen_${NET}_${ENV_SEED}/routes.csv" \
                "../results/sample_results/routes.csv"; do
        [[ -f "$REPO_ROOT/${cand#../}" ]] && { ROUTES="$cand"; break; }
    done
fi
ROUTES="${ROUTES:-../results/sample_results/routes.csv}"
ROUTES_ABS="$REPO_ROOT/${ROUTES#../}"

echo "==========================================================================="
echo " URB overnight evidence run"
echo "   repo        $REPO_ROOT"
echo "   evidence    $EVIDENCE"
echo "   tiers       $TIERS        (0 offline, 1 cheap SUMO, 2 training)"
echo "   python      $PY           routerl available: $HAS_SUMO"
echo "   route table $ROUTES_ABS"
echo "   started     $(date -Iseconds 2>/dev/null || date)"
echo "==========================================================================="

if [[ ! -f "$ROUTES_ABS" ]]; then
    echo "FATAL: route table missing: $ROUTES_ABS" >&2
    echo "       Generate one: python scripts/pact1.py --id routegen_${NET}_${ENV_SEED} \\" >&2
    echo "           --alg-conf config1 --task-conf $TASK_CONF --net $NET \\" >&2
    echo "           --env-seed $ENV_SEED --mode dry" >&2
    exit 1
fi

DONE=0; SKIPPED=0; FAILED=(); MISSING=()

want_tier () { [[ " $TIERS " == *" $1 "* ]]; }

# step <nn> <slug> <tier> <needs_sumo 0|1> -- <command...>
step () {
    local nn="$1" slug="$2" tier="$3" needs="$4"; shift 5   # shift past the "--"
    want_tier "$tier" || return 0

    local out="$EVIDENCE/${nn}_${slug}.txt"
    if [[ -f "$out.done" ]]; then
        echo "SKIP   $nn $slug (done)"; SKIPPED=$((SKIPPED+1)); return 0
    fi
    # the plan prints even where the step cannot run, so it can be reviewed anywhere
    if [[ $DRY == 1 ]]; then
        printf 'PLAN   %s %s -> %s\n         ' "$nn" "$slug" "${out##*/}"
        printf '%q ' "$@"; echo; return 0
    fi
    if [[ "$needs" == "1" && "$HAS_SUMO" == "0" ]]; then
        echo "MISS   $nn $slug — needs routerl, not installed here"
        MISSING+=( "$nn $slug" ); return 0
    fi

    echo "RUN    $nn $slug -> ${out##*/}"
    local t0=$SECONDS
    {
        echo "# $slug"
        echo "# $(date -Iseconds 2>/dev/null || date)"
        printf '# '; printf '%q ' "$@"; echo
        echo
    } > "$out"
    "$@" >> "$out" 2>&1
    local rc=$? mins=$(( (SECONDS - t0) / 60 ))
    if [[ $rc -eq 0 ]]; then
        echo "OK     $nn $slug  ${mins}m"; : > "$out.done"; DONE=$((DONE+1))
    else
        echo "FAIL   $nn $slug  rc=$rc after ${mins}m"
        tail -n 8 "$out" | sed 's/^/       | /'
        FAILED+=( "$nn $slug" )
    fi
}

# ============================== TIER 0 — offline ===========================
step 01 selftest_urb_ns        0 0 -- "$PY" urb_ns/selftest.py
step 02 selftest_pact1         0 0 -- "$PY" pact1/selftest.py

step 03 ceiling_random_subset  0 0 -- "$PY" scripts/ns_certify.py \
    --net "$NET" --routes "$ROUTES" --task-conf "$TASK_CONF" --env-conf "$ENV_CONF" \
    --sigmas 1.0 --shares 0.1 0.2 0.4 0.6 0.8 1.0 --subset-draws 5 \
    --out "$EVIDENCE/ns_certificate_random"

step 04 ceiling_id_prefix      0 0 -- "$PY" scripts/ns_certify.py \
    --net "$NET" --routes "$ROUTES" --task-conf "$TASK_CONF" --env-conf "$ENV_CONF" \
    --sigmas 0.5 1.0 1.5 2.0 --shares 0.1 0.2 0.4 0.6 0.8 1.0 \
    --out "$EVIDENCE/ns_certificate_prefix"

step 05 steering_sigma0        0 0 -- "$PY" scripts/sweep/steering_sweep.py \
    --sigma 0 --routes "$ROUTES"
step 06 steering_sigma1        0 0 -- "$PY" scripts/sweep/steering_sweep.py \
    --sigma 1 --routes "$ROUTES"
step 07 steering_sigma3        0 0 -- "$PY" scripts/sweep/steering_sweep.py \
    --sigma 3 --routes "$ROUTES"
step 08 steering_sigma3_load20 0 0 -- "$PY" scripts/sweep/steering_sweep.py \
    --sigma 3 --routes "$ROUTES" --alpha-scale 20 --damp 0.8

# the PoA re-run lives in the paper tree; skip cleanly if it is not checked out here
if [[ -f "$PAPER_DIR/verification/verify_urb_spectral.py" ]]; then
    step 09 spectral_poa       0 0 -- env URB_ROOT="$REPO_ROOT" URB_SWEEPS="$SPECTRAL_SWEEPS" \
        "$PY" "$PAPER_DIR/verification/verify_urb_spectral.py"
else
    want_tier 0 && { echo "MISS   09 spectral_poa — $PAPER_DIR/verification not found"
                     MISSING+=( "09 spectral_poa" ); }
fi

# ============================== TIER 1 — cheap SUMO ========================
step 10 probe_sigma0 1 1 -- "$PY" scripts/ns_launch.py \
    --sigma 0 --net "$NET" --routes "$ROUTES" --env-conf "$ENV_CONF" \
    -- pact1.py --id ev_probe_s0 --alg-conf config1 --task-conf "$TASK_CONF" \
       --net "$NET" --env-seed "$ENV_SEED" --mode probe --probe-eps 60

step 11 probe_sigma3 1 1 -- "$PY" scripts/ns_launch.py \
    --sigma 3 --net "$NET" --routes "$ROUTES" --env-conf "$ENV_CONF" \
    -- pact1.py --id ev_probe_s3 --alg-conf config1 --task-conf "$TASK_CONF" \
       --net "$NET" --env-seed "$ENV_SEED" --mode probe --probe-eps 60

step 12 greedy_sigma3 1 1 -- "$PY" scripts/ns_launch.py \
    --sigma 3 --net "$NET" --routes "$ROUTES" --env-conf "$ENV_CONF" \
    -- greedy.py --id ev_greedy_s3 --alg-conf config1 --task-conf "$TASK_CONF" \
       --net "$NET" --env-seed "$ENV_SEED"

step 13 aon_sigma3 1 1 -- "$PY" scripts/ns_launch.py \
    --sigma 3 --net "$NET" --routes "$ROUTES" --env-conf "$ENV_CONF" \
    -- baselines.py --id ev_aon_s3 --alg-conf config1 --task-conf "$TASK_CONF" \
       --net "$NET" --env-seed "$ENV_SEED" --model aon

step 14 meanpreserving_sigma3 1 1 -- "$PY" scripts/ns_launch.py \
    --sigma 3 --net "$NET" --routes "$ROUTES" --env-conf "$ENV_CONF" --mean-preserving \
    -- ippo.py --id ev_meanpres_s3 --alg-conf config1 --task-conf "$TASK_CONF" \
       --net "$NET" --env-seed "$ENV_SEED" --torch-seed 0

# ============================== TIER 2 — training ==========================
n=20
for cfg in fixed_g01 fixed_g03 fixed_g05 fixed_g07 fixed_g09 fixed_g10; do
    for s in $TRUST_SEEDS; do
        step "$n" "trust_${cfg}_seed${s}" 2 1 -- "$PY" scripts/ns_launch.py \
            --sigma 3 --net "$NET" --routes "$ROUTES" --env-conf "$ENV_CONF" \
            -- pact1.py --id "ev_s3_${cfg}_${s}" --alg-conf "$cfg" \
               --task-conf "$TASK_CONF" --net "$NET" --env-seed "$ENV_SEED" --torch-seed "$s"
        n=$((n+1))
    done
done

# ============================== index ======================================
if [[ $DRY == 0 ]]; then
    {
        echo "# Evidence index"
        echo
        echo "Generated $(date -Iseconds 2>/dev/null || date) — tiers \`$TIERS\`,"
        echo "net \`$NET\`, env-seed $ENV_SEED, route table \`$ROUTES\`."
        echo
        echo "| file | status | size | last line |"
        echo "|---|---|---|---|"
        for f in "$EVIDENCE"/*.txt; do
            [[ -e "$f" ]] || continue
            b="$(basename "$f")"
            st="INCOMPLETE"; [[ -f "$f.done" ]] && st="ok"
            sz="$(wc -c < "$f" | tr -d ' ')"
            last="$(tail -n 1 "$f" | cut -c1-90 | sed 's/|/\\|/g')"
            echo "| \`$b\` | $st | ${sz}B | $last |"
        done
        echo
        echo "Re-run one step: delete its \`.txt.done\` marker and run the script again."
    } > "$EVIDENCE/INDEX.md"
fi

echo
echo "==========================================================================="
echo " finished $(date -Iseconds 2>/dev/null || date)"
echo "   completed $DONE   skipped $SKIPPED   failed ${#FAILED[@]}   unavailable ${#MISSING[@]}"
[[ ${#FAILED[@]}  -gt 0 ]] && printf '   FAILED: %s\n' "${FAILED[*]}"
[[ ${#MISSING[@]} -gt 0 ]] && printf '   UNAVAILABLE: %s\n' "${MISSING[*]}"
echo "   read:  $EVIDENCE/INDEX.md"
echo "==========================================================================="
[[ ${#FAILED[@]} -eq 0 ]]

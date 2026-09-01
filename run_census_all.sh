#!/usr/bin/env bash
#
# run_census_all.sh -- census every oact/BLASTNet job root, grouped so that
# Lustre sees ~8 concurrent scanners rather than 37. One process per group,
# each handling several roots; --workers is threads WITHIN a process, so the
# total concurrent I/O streams is (groups x workers).
#
# Run check_census_roots.py FIRST -- if a root nests its job dirs under
# jobs/ or jobs_parsl/, census finds nothing there and this script will
# happily report zero jobs for it.
#
# Usage:  bash run_census_all.sh [OUTDIR]

set -u

OUTDIR="${1:-$HOME/census_out}"
WORKERS="${WORKERS:-8}"
mkdir -p "$OUTDIR/shards"

echo "Writing shards to $OUTDIR/shards (workers=$WORKERS per group)"

# ---- oact_canada (1 root) ----
python -m oact_utilities.workflows.census \
    /lus/eagle/projects/oact/canada/nonact_06 \
    -o "$OUTDIR/shards/oact_canada.parquet" \
    --workers "$WORKERS" > "$OUTDIR/oact_canada.log" 2>&1 &

# ---- oact (2 roots) ----
python -m oact_utilities.workflows.census \
    /lus/eagle/projects/oact/act_226_michael \
    /lus/eagle/projects/oact/nonact_4_06 \
    -o "$OUTDIR/shards/oact.parquet" \
    --workers "$WORKERS" > "$OUTDIR/oact.log" 2>&1 &

# ---- oact_sandia (5 roots) ----
python -m oact_utilities.workflows.census \
    /lus/eagle/projects/oact/sandia/act_531_chunk10 \
    /lus/eagle/projects/oact/sandia/act_531_chunk12 \
    /lus/eagle/projects/oact/sandia/act_531_chunk14 \
    /lus/eagle/projects/oact/sandia/act_531_chunk16 \
    /lus/eagle/projects/oact/sandia/nonact_531_chunk05 \
    -o "$OUTDIR/shards/oact_sandia.parquet" \
    --workers "$WORKERS" > "$OUTDIR/oact_sandia.log" 2>&1 &

# ---- BLASTNet (10 roots) ----
python -m oact_utilities.workflows.census \
    /lus/eagle/projects/BLASTNet/act_222_santi \
    /lus/eagle/projects/BLASTNet/act_4_06_chunk_0 \
    /lus/eagle/projects/BLASTNet/nonact_222_santi \
    /lus/eagle/projects/BLASTNet/nonact_4_06 \
    /lus/eagle/projects/BLASTNet/act_226_michael \
    /lus/eagle/projects/BLASTNet/act_4_06_chunk_1 \
    /lus/eagle/projects/BLASTNet/nonact_226_michael \
    /lus/eagle/projects/BLASTNet/act_226_santi \
    /lus/eagle/projects/BLASTNet/nonact_226_santi \
    /lus/eagle/projects/BLASTNet/ritwik_226 \
    -o "$OUTDIR/shards/BLASTNet.parquet" \
    --workers "$WORKERS" > "$OUTDIR/BLASTNet.log" 2>&1 &

# ---- BLASTNet_sandia (4 roots) ----
python -m oact_utilities.workflows.census \
    /lus/eagle/projects/BLASTNet/sandia/act_406_chunk4 \
    /lus/eagle/projects/BLASTNet/sandia/act_531_chunk02 \
    /lus/eagle/projects/BLASTNet/sandia/act_531_chunk07 \
    /lus/eagle/projects/BLASTNet/sandia/nonact_531_chunk0 \
    -o "$OUTDIR/shards/BLASTNet_sandia.parquet" \
    --workers "$WORKERS" > "$OUTDIR/BLASTNet_sandia.log" 2>&1 &

# ---- BLASTNet_carpenter (6 roots) ----
python -m oact_utilities.workflows.census \
    /lus/eagle/projects/BLASTNet/carpenter/08052026-nonactinide-push \
    /lus/eagle/projects/BLASTNet/carpenter/08122026-nonactinide-push \
    /lus/eagle/projects/BLASTNet/carpenter/20260708-actinide-push \
    /lus/eagle/projects/BLASTNet/carpenter/20260722-actinide-transfer \
    /lus/eagle/projects/BLASTNet/carpenter/20260828-actinide-push \
    /lus/eagle/projects/BLASTNet/carpenter/wave1_carpenter_act \
    -o "$OUTDIR/shards/BLASTNet_carpenter.parquet" \
    --workers "$WORKERS" > "$OUTDIR/BLASTNet_carpenter.log" 2>&1 &

# ---- BLASTNet_barfoot (5 roots) ----
python -m oact_utilities.workflows.census \
    /lus/eagle/projects/BLASTNet/barfoot/20260706_wave \
    /lus/eagle/projects/BLASTNet/barfoot/20260722-actinide-push \
    /lus/eagle/projects/BLASTNet/barfoot/20260812-actinide-push \
    /lus/eagle/projects/BLASTNet/barfoot/20260828-actinide-push \
    /lus/eagle/projects/BLASTNet/barfoot/wave1 \
    -o "$OUTDIR/shards/BLASTNet_barfoot.parquet" \
    --workers "$WORKERS" > "$OUTDIR/BLASTNet_barfoot.log" 2>&1 &

# ---- BLASTNet_raider (4 roots) ----
python -m oact_utilities.workflows.census \
    /lus/eagle/projects/BLASTNet/raider/20260722-actinide-push \
    /lus/eagle/projects/BLASTNet/raider/20260724-actinide-push \
    /lus/eagle/projects/BLASTNet/raider/20260812-nonactinide-push \
    /lus/eagle/projects/BLASTNet/raider/20260828-actinide-push \
    -o "$OUTDIR/shards/BLASTNet_raider.parquet" \
    --workers "$WORKERS" > "$OUTDIR/BLASTNet_raider.log" 2>&1 &

echo "Launched 8 groups; waiting..."
wait

echo
echo "=== per-group results ==="
grep -H "Wrote\|Scanned in" "$OUTDIR"/*.log | sed "s|$OUTDIR/||"

fail=0
for f in "$OUTDIR"/*.log; do
    if grep -q "Traceback" "$f"; then echo "FAILED: $f"; fail=1; fi
done
[ "$fail" -eq 0 ] && echo "no tracebacks" || exit 1

echo
echo "=== merging shards ==="
# NOTE: the merged file goes OUTSIDE shards/, so a re-run of --merge on the
# shards directory can never re-ingest a previous merge output.
python -m oact_utilities.workflows.census --merge "$OUTDIR/shards" \
    -o "$OUTDIR/census_combined.parquet"

echo
echo "Combined table: $OUTDIR/census_combined.parquet"

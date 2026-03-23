#!/usr/bin/env bash
# globftp_upload.sh -- Generate FTP command files for recursive upload
# to LLNL GlobFTP (HPSS/globfs).
#
# The LLNL GlobFTP (DISCOM PFTP) server does not support recursive uploads.
# This script generates FTP command files split across N parallel sessions.
# You then open N authenticated ftp sessions and paste one session file
# into each.
#
# Usage:
#   bash globftp_upload.sh <local_dir> <remote_dir> [num_sessions] [jobs_per_batch] [exclude_file]
#
# Example:
#   bash globftp_upload.sh \
#     /p/lustre5/vargas58/oact/homoleptics/batch_226/nonact/jobs_parsl \
#     /p/globfs/vargas58/nonact/jobs_parsl \
#     4 50
#
# To skip already-uploaded directories:
#   1. In ftp, save a listing of what's already on the remote:
#        ftp> cd /p/globfs/vargas58/nonact/jobs_parsl
#        ftp> mls * /local/path/to/remote_listing.txt
#      Output is "job_0/orca.out" format -- the script extracts dir names automatically.
#   2. Pass that file as the 5th argument:
#        bash globftp_upload.sh <local> <remote> 4 50 remote_listing.txt
#
# Prerequisites:
#   - Run clean.py first to remove scratch files and reduce transfer size:
#       python -m oact_utilities.workflows.clean <db> <local_dir> --clean-all --execute

set -euo pipefail

LOCAL_DIR="${1:?Usage: $0 <local_dir> <remote_dir> [num_sessions] [jobs_per_batch] [exclude_file]}"
REMOTE_DIR="${2:?Usage: $0 <local_dir> <remote_dir> [num_sessions] [jobs_per_batch] [exclude_file]}"
NUM_SESSIONS="${3:-4}"
JOBS_PER_BATCH="${4:-50}"
EXCLUDE_FILE="${5:-}"

SCRIPT_DIR="${LOCAL_DIR}/../ftp_scripts"
mkdir -p "$SCRIPT_DIR"

echo "Local:           $LOCAL_DIR"
echo "Remote:          $REMOTE_DIR"
echo "Sessions:        $NUM_SESSIONS"
echo "Jobs per batch:  $JOBS_PER_BATCH"
echo "Script dir:      $SCRIPT_DIR"

# ---- Clean old files ----
rm -f "$SCRIPT_DIR"/session_*.txt "$SCRIPT_DIR"/session_*/batch_*

# ---- Generate job list (filtering out already-uploaded dirs) ----
echo ""
echo "Scanning local directories..."

ALL_LOCAL="$SCRIPT_DIR/all_local.txt"
ls "$LOCAL_DIR" | sort > "$ALL_LOCAL"
TOTAL_LOCAL=$(wc -l < "$ALL_LOCAL" | tr -d ' ')
echo "Found $TOTAL_LOCAL local job directories."

JOB_LIST="$SCRIPT_DIR/all_jobs.txt"
if [[ -n "$EXCLUDE_FILE" && -f "$EXCLUDE_FILE" ]]; then
    # Extract unique directory names from listing (handles "job_0/orca.out" format)
    # Strip paths to get just the directory part, deduplicate, sort
    sed 's/^[ \t]*//;s/[ \t]*$//' "$EXCLUDE_FILE" \
        | sed 's|/.*||' \
        | sort -u > "$SCRIPT_DIR/exclude_sorted.txt"
    comm -23 "$ALL_LOCAL" "$SCRIPT_DIR/exclude_sorted.txt" > "$JOB_LIST"
    SKIPPED=$((TOTAL_LOCAL - $(wc -l < "$JOB_LIST" | tr -d ' ')))
    echo "Excluding $SKIPPED directories already on remote."
else
    cp "$ALL_LOCAL" "$JOB_LIST"
fi

TOTAL=$(wc -l < "$JOB_LIST" | tr -d ' ')
echo "To upload: $TOTAL job directories."

if [[ "$TOTAL" -eq 0 ]]; then
    echo "Nothing to upload."
    exit 0
fi

# ---- FTP header (binary + prompt + ensure remote dir exists) ----
HEADER="$SCRIPT_DIR/header.txt"
{
    echo "binary"
    echo "prompt"
    echo "mkdir $REMOTE_DIR"
} > "$HEADER"

# ---- Split jobs across sessions ----
# Round-robin distribute jobs to sessions for even load
for i in $(seq 1 "$NUM_SESSIONS"); do
    mkdir -p "$SCRIPT_DIR/session_$i"
done

SESSION=1
while read -r job; do
    echo "$job" >> "$SCRIPT_DIR/session_${SESSION}/jobs.txt"
    SESSION=$(( (SESSION % NUM_SESSIONS) + 1 ))
done < "$JOB_LIST"

# ---- Generate command files per session, split into batches ----
LINES_PER_BATCH=$((JOBS_PER_BATCH * 4))

for i in $(seq 1 "$NUM_SESSIONS"); do
    SESSION_DIR="$SCRIPT_DIR/session_$i"
    SESSION_JOBS="$SESSION_DIR/jobs.txt"
    [[ -f "$SESSION_JOBS" ]] || continue

    SESSION_COUNT=$(wc -l < "$SESSION_JOBS" | tr -d ' ')

    # Generate full command file for this session
    SESSION_CMDS="$SESSION_DIR/commands.txt"
    {
        while read -r job; do
            echo "mkdir ${REMOTE_DIR}/${job}"
            echo "lcd ${LOCAL_DIR}/${job}"
            echo "cd ${REMOTE_DIR}/${job}"
            echo "mput *"
        done < "$SESSION_JOBS"
    } > "$SESSION_CMDS"

    # Split into batches with header prepended
    tail -n +1 "$SESSION_CMDS" \
        | split -l "$LINES_PER_BATCH" - "$SESSION_DIR/batch_"

    BATCH_COUNT=0
    for batch in "$SESSION_DIR"/batch_*; do
        cat "$HEADER" "$batch" > "${batch}.tmp"
        mv "${batch}.tmp" "$batch"
        BATCH_COUNT=$((BATCH_COUNT + 1))
    done

    echo "  Session $i: $SESSION_COUNT jobs -> $BATCH_COUNT batches"
done

# ---- Print instructions ----
echo ""
echo "============================================="
echo "  Ready to upload $TOTAL jobs"
echo "============================================="
echo ""
echo "Step 1: Open $NUM_SESSIONS terminal windows"
echo ""
echo "Step 2: In each terminal, authenticate to ftp:"
echo "  ftp globftp.llnl.gov"
echo "  [enter username + password + OTP]"
echo ""
echo "Step 3: Paste batch files into each session."
echo "  Each batch is $JOBS_PER_BATCH jobs. Wait for one to finish before pasting the next."
echo ""

for i in $(seq 1 "$NUM_SESSIONS"); do
    SESSION_DIR="$SCRIPT_DIR/session_$i"
    [[ -d "$SESSION_DIR" ]] || continue
    echo "  --- Terminal $i ---"
    for batch in "$SESSION_DIR"/batch_*; do
        echo "  cat $batch"
    done
    echo ""
done

echo "Tip: to verify what was uploaded, run in ftp:"
echo "  cd $REMOTE_DIR"
echo "  nlist"

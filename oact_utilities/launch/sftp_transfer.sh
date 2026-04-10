#!/usr/bin/env bash
# sftp_transfer.sh -- Parallel sftp transfer to goblin (globfs/HPSS)
#
# Transfers a directory of job folders to goblin using parallel sftp sessions.
# Supports restarts via a local manifest file that tracks completed jobs.
#
# Prerequisites:
#   - SSH keys set up between oslic and goblin (see below)
#   - Optionally, run clean.py first to reduce transfer size:
#       python -m oact_utilities.workflows.clean <db> <local_dir> --clean-all --execute
#
# SSH key setup (one-time, from any LC machine):
#   ssh-keygen -t rsa -b 4096   # leave passphrase blank
#   ssh-copy-id vargas58@goblin
#
# Usage:
#   bash sftp_transfer.sh <local_dir> <remote_dir> [num_sessions] [remote_host]
#
# Examples:
#   # Transfer jobs_parsl with 4 parallel sessions
#   bash sftp_transfer.sh \
#       /p/vast1/vargas58/oact/act_226/jobs_parsl \
#       /p/globfs/vargas58/act_226 \
#       4
#
#   # Transfer nonact batch with 8 sessions
#   bash sftp_transfer.sh \
#       /p/lustre5/vargas58/oact/homoleptics/batch_226/nonact/jobs_parsl \
#       /p/globfs/vargas58/nonact/jobs_parsl \
#       8
#
# Restart behavior:
#   A manifest file (<local_dir>/../sftp_done.txt) tracks completed jobs.
#   Re-running the same command skips already-transferred directories.
#   To force a full re-transfer, delete the manifest file.
#
# ============================================================================
# Validation (run after transfer)
# ============================================================================
#
# 1. Compare job directory counts:
#   ls <local_dir> | wc -l
#   ssh vargas58@goblin "ls <remote_dir>" | wc -l
#
# 2. Compare file counts (use -maxdepth 2 to exclude nested junk):
#   find <local_dir> -maxdepth 2 -type f | wc -l
#   ssh vargas58@goblin "find <remote_dir> -maxdepth 2 -type f | wc -l"
#
# 3. Compare actual byte sizes (du reports block sizes which differ across
#    filesystems -- use stat to get real file sizes):
#   find <local_dir> -maxdepth 2 -type f -print0 \
#       | xargs -0 stat --format='%s' \
#       | awk '{s+=$1}END{printf "%.2f GB\n", s/1073741824}'
#   ssh vargas58@goblin "find <remote_dir> -maxdepth 2 -type f -print0 \
#       | xargs -0 stat --format='%s'" \
#       | awk '{s+=$1}END{printf "%.2f GB\n", s/1073741824}'
#
# 4. Find missing files (on local but not remote):
#   find <local_dir> -maxdepth 2 -type f | sed 's|.*/<parent_dir>/||' | sort > /tmp/local_f.txt
#   ssh vargas58@goblin "find <remote_dir> -maxdepth 2 -type f" \
#       | sed 's|.*/<parent_dir>/||' | sort > /tmp/remote_f.txt
#   comm -23 /tmp/local_f.txt /tmp/remote_f.txt
#
# 5. Find extra files on remote (not on local):
#   comm -13 /tmp/local_f.txt /tmp/remote_f.txt | head -30
#
# ============================================================================
# Remote cleanup
# ============================================================================
#
# Remove orca_tmp_* scratch directories transferred by accident:
#   ssh vargas58@goblin 'find <remote_dir> -mindepth 2 -maxdepth 2 \
#       -type d -name "orca_tmp_*" -exec rm -rf {} +'
#
# Remove nested duplicate directories (from put -r without trailing slash):
#   ssh vargas58@goblin 'for d in <remote_dir>/*/; do
#       job=$(basename "$d")
#       [ -d "${d}${job}" ] && rm -rf "${d}${job}"
#   done'
#
# Check no nested files remain:
#   ssh vargas58@goblin "find <remote_dir> -mindepth 3 -type f | wc -l"

set -euo pipefail

LOCAL_DIR="${1:?Usage: $0 <local_dir> <remote_dir> [num_sessions] [remote_host]}"
REMOTE_DIR="${2:?Usage: $0 <local_dir> <remote_dir> [num_sessions] [remote_host]}"
NUM_SESSIONS="${3:-4}"
REMOTE_HOST="${4:-goblin}"

MANIFEST="${LOCAL_DIR}/../sftp_done.txt"
touch "$MANIFEST"

echo "Local:      $LOCAL_DIR"
echo "Remote:     $REMOTE_HOST:$REMOTE_DIR"
echo "Sessions:   $NUM_SESSIONS"
echo "Manifest:   $MANIFEST"

# ---- Build list of jobs to transfer (skip already-done) ----
ALL_JOBS=$(mktemp)
UPLOAD_JOBS=$(mktemp)

ls "$LOCAL_DIR" | sort > "$ALL_JOBS"
TOTAL_LOCAL=$(wc -l < "$ALL_JOBS" | tr -d ' ')

sort "$MANIFEST" | uniq > "${MANIFEST}.sorted"
comm -23 "$ALL_JOBS" "${MANIFEST}.sorted" > "$UPLOAD_JOBS"
rm -f "${MANIFEST}.sorted"

TOTAL_UPLOAD=$(wc -l < "$UPLOAD_JOBS" | tr -d ' ')
SKIPPED=$((TOTAL_LOCAL - TOTAL_UPLOAD))

echo "Total:      $TOTAL_LOCAL job directories"
echo "Skipped:    $SKIPPED (already transferred)"
echo "To upload:  $TOTAL_UPLOAD"

if [[ "$TOTAL_UPLOAD" -eq 0 ]]; then
    echo "Nothing to upload."
    rm -f "$ALL_JOBS" "$UPLOAD_JOBS"
    exit 0
fi

# ---- Split across sessions and transfer ----
CHUNK_PREFIX=$(mktemp -d)/chunk_
split -l $(( (TOTAL_UPLOAD + NUM_SESSIONS - 1) / NUM_SESSIONS )) "$UPLOAD_JOBS" "$CHUNK_PREFIX"

echo ""
echo "Starting $NUM_SESSIONS parallel sftp sessions..."

for chunk in "${CHUNK_PREFIX}"*; do
    (
    while read -r job; do
        # put -r <src>/ <dst> -- trailing slash on source avoids nesting
        sftp -b - "$REMOTE_HOST" <<EOF
mkdir $REMOTE_DIR/$job
put -r $LOCAL_DIR/$job/ $REMOTE_DIR/$job
EOF
        # Record success
        echo "$job" >> "$MANIFEST"
    done < "$chunk"
    ) &
done

wait
rm -f "$ALL_JOBS" "$UPLOAD_JOBS"

echo ""
echo "Transfer complete."
echo "Transferred: $(wc -l < "$MANIFEST" | tr -d ' ') / $TOTAL_LOCAL jobs"

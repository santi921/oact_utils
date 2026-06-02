#!/bin/bash
# Storage probe for DRAC clusters (Fir/Narval/Nibi/Rorqual/Trillium).
#
# Catalogs every mount the workflow could use: filesystem type, quota (bytes
# AND inode/file count, which is the limit conda blows through), free space.
# Use it to pick where the workflow DB and job dirs live.
#
# Usage:
#   bash probe_storage.sh                 # probe HOME, scratch, project, nearline
#   bash probe_storage.sh /some/path      # also report largest subdirs of /some/path
#
# Run it on a login node for the shared Lustre mounts. Run it again INSIDE an
# salloc/batch job to additionally capture $SLURM_TMPDIR (node-local NVMe),
# which only exists during a job.
#
# Safe and read-only. The optional `du` pass on an argument is the only heavy
# operation and is guarded with a warning (du on Lustre is slow).

set -u

hr() { printf '%s\n' "------------------------------------------------------------"; }

echo "DRAC storage probe"
hr
echo "host        : $(hostname)"
echo "user        : ${USER:-?}"
echo "date        : $(date)"
echo "SLURM_TMPDIR: ${SLURM_TMPDIR:-<not in a job>}"
hr

# --- 1. Canonical Alliance quota report (bytes + inodes per filesystem) ------
echo
echo "## diskusage_report (Alliance quota tool: watch the 'inode/file' columns)"
hr
if command -v diskusage_report >/dev/null 2>&1; then
    diskusage_report 2>&1 || echo "(diskusage_report returned nonzero)"
else
    echo "diskusage_report not found on this host."
fi

# --- 2. Per-mount detail: FS type, df, Lustre quota --------------------------
# Build the candidate list. project is a glob ($HOME/projects/* or
# $HOME/project/*) depending on cluster; resolve both.
declare -a MOUNTS=("$HOME" "$HOME/scratch")
for proot in "$HOME"/projects/* "$HOME"/project/*; do
    [ -d "$proot" ] && MOUNTS+=("$proot")
done
[ -d /nearline ] && MOUNTS+=("/nearline")
[ -n "${SLURM_TMPDIR:-}" ] && MOUNTS+=("$SLURM_TMPDIR")

echo
echo "## Per-mount detail"
for m in "${MOUNTS[@]}"; do
    [ -e "$m" ] || continue
    hr
    echo "path        : $m"
    # %T = filesystem type name (lustre, tmpfs, ext4, nfs, ...). 'lustre' is the
    # one that matters for SQLite locking decisions.
    fstype=$(stat -f -c '%T' "$m" 2>/dev/null || echo '?')
    echo "fs type     : $fstype"
    echo "df:"
    df -h "$m" 2>/dev/null | sed 's/^/    /'
    if [ "$fstype" = "lustre" ]; then
        echo "lustre user quota:"
        lfs quota -h -u "$USER" "$m" 2>/dev/null | sed 's/^/    /' \
            || echo "    (lfs quota unavailable)"
    fi
done
hr

# --- 3. Optional: largest subdirectories of a path the user names ------------
if [ "$#" -ge 1 ]; then
    target="$1"
    echo
    echo "## Largest immediate subdirectories of: $target"
    echo "   (du on Lustre is slow; this scans one level only)"
    hr
    if [ -d "$target" ]; then
        du -h --max-depth=1 "$target" 2>/dev/null | sort -h | tail -n 30
    else
        echo "Not a directory: $target"
    fi
    hr
fi

echo
echo "## Reads"
echo " - HOME is small (Fir: 50 GB / 500K files) and the file-count limit, not"
echo "   bytes, is usually what a conda env exhausts. Watch the inode column."
echo " - PROJECT is persistent + backed up: best home for a venv and final results."
echo " - SCRATCH is large but PURGES files older than ~60 days: never leave the"
echo "   live DB or final results here long-term."
echo " - \$SLURM_TMPDIR (only shown inside a job) is node-local NVMe: fastest for"
echo "   the live DB and ORCA scratch, but vanishes at job end (copy results back)."

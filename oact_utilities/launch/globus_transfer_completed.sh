#!/usr/bin/env bash
# globus_transfer_completed.sh -- submit a batch Globus transfer for completed jobs
#
# Usage:
#   bash globus_transfer_completed.sh <db_path> <dest_root>
#
# The script:
#   1. Starts Globus Connect Personal in the background if needed
#   2. Queries completed job directories from the workflow DB
#   3. Submits one batch Globus transfer task
#   4. Exits without waiting for transfer completion

set -euo pipefail

readonly CARPENTER_SOURCE_ENDPOINT_ID="b808a48a-4b2d-11f1-a9a0-02535127e3d7"
readonly BARFOOT_SOURCE_ENDPOINT_ID="1ea1ecb5-4d77-11f1-848e-0ea3589134b3"
readonly DESTINATION_ENDPOINT_ID="05d2c76a-e867-4f67-aa57-76edeb0beda0"

GLOBUS_CONNECT_PERSONAL_BIN="${GLOBUS_CONNECT_PERSONAL_BIN:-globusconnectpersonal}"
GLOBUS_CONNECT_STARTUP_WAIT="${GLOBUS_CONNECT_STARTUP_WAIT:-2}"
GLOBUS_TRANSFER_MIN_FILE_AGE_MINUTES="${GLOBUS_TRANSFER_MIN_FILE_AGE_MINUTES:-5}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_FILE_TO_CLEAN=""

usage() {
    echo "Usage: $0 <db_path> <dest_root>" >&2
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command '$cmd' was not found in PATH." >&2
        exit 1
    fi
}

normalize_dest_root() {
    local path="$1"
    if [[ "$path" == "/" ]]; then
        printf "/"
    else
        printf "%s" "${path%/}"
    fi
}

detect_source_endpoint_id() {
    local detected_hostname
    detected_hostname="$(hostname | tr '[:upper:]' '[:lower:]')"

    if [[ "$detected_hostname" == *carpenter* ]]; then
        printf "%s" "$CARPENTER_SOURCE_ENDPOINT_ID"
        return
    fi

    if [[ "$detected_hostname" == *barfoot* ]]; then
        printf "%s" "$BARFOOT_SOURCE_ENDPOINT_ID"
        return
    fi

    echo "Error: unsupported hostname '$detected_hostname'." >&2
    echo "Expected a hostname containing 'carpenter' or 'barfoot'." >&2
    exit 1
}

has_recently_modified_files() {
    local job_dir="$1"
    local min_file_age_minutes="$2"

    if [[ "$min_file_age_minutes" == "0" ]]; then
        return 1
    fi

    "$PYTHON_BIN" - "$job_dir" "$min_file_age_minutes" <<'PY'
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

job_dir = Path(sys.argv[1])
min_age_minutes = float(sys.argv[2])

if min_age_minutes <= 0:
    raise SystemExit(1)

threshold = time.time() - (min_age_minutes * 60.0)

for root, _, files in os.walk(job_dir):
    for file_name in files:
        file_path = Path(root) / file_name
        try:
            if file_path.stat().st_mtime > threshold:
                raise SystemExit(0)
        except OSError:
            continue

raise SystemExit(1)
PY
}

ensure_globus_connect_personal() {
    if pgrep -f "globusconnectpersonal" >/dev/null 2>&1; then
        echo "Globus Connect Personal is already running."
        return
    fi

    local start_log
    start_log="/tmp/globusconnectpersonal_$(date +%s).log"

    echo "Starting Globus Connect Personal in the background..."
    nohup "$GLOBUS_CONNECT_PERSONAL_BIN" -start >"$start_log" 2>&1 &
    sleep "$GLOBUS_CONNECT_STARTUP_WAIT"
    echo "Globus Connect Personal log: $start_log"
}

main() {
    if [[ $# -ne 2 ]]; then
        usage
        exit 1
    fi

    local db_path="$1"
    local dest_root
    dest_root="$(normalize_dest_root "$2")"

    if [[ ! -f "$db_path" ]]; then
        echo "Error: database not found at '$db_path'." >&2
        exit 1
    fi

    if [[ "$dest_root" != /* ]]; then
        echo "Error: destination root must be an absolute endpoint path." >&2
        exit 1
    fi

    require_cmd hostname
    require_cmd sqlite3
    require_cmd globus
    require_cmd "$GLOBUS_CONNECT_PERSONAL_BIN"
    require_cmd "$PYTHON_BIN"
    require_cmd pgrep
    require_cmd mktemp

    local source_endpoint_id
    source_endpoint_id="$(detect_source_endpoint_id)"

    ensure_globus_connect_personal

    local batch_file
    batch_file="$(mktemp "${TMPDIR:-/tmp}/globus-transfer-completed.XXXXXX")"
    BATCH_FILE_TO_CLEAN="$batch_file"
    trap 'rm -f -- "${BATCH_FILE_TO_CLEAN:-}"' EXIT

    local selected_count=0
    local valid_count=0
    local skipped_missing=0
    local skipped_recent=0
    local job_dir=""
    local job_name=""
    local dest_path=""
    local -a missing_examples=()
    local -a recent_examples=()

    while IFS= read -r job_dir; do
        [[ -z "$job_dir" ]] && continue
        selected_count=$((selected_count + 1))

        if [[ ! -d "$job_dir" ]]; then
            skipped_missing=$((skipped_missing + 1))
            if [[ ${#missing_examples[@]} -lt 3 ]]; then
                missing_examples+=("$job_dir")
            fi
            continue
        fi

        if has_recently_modified_files \
            "$job_dir" \
            "$GLOBUS_TRANSFER_MIN_FILE_AGE_MINUTES"; then
            skipped_recent=$((skipped_recent + 1))
            if [[ ${#recent_examples[@]} -lt 3 ]]; then
                recent_examples+=("$job_dir")
            fi
            continue
        fi

        job_dir="${job_dir%/}"
        job_name="${job_dir##*/}"
        dest_path="${dest_root}/${job_name}"
        printf '%q %q --recursive\n' "$job_dir" "$dest_path" >>"$batch_file"
        valid_count=$((valid_count + 1))
    done < <(
        sqlite3 -batch -noheader "$db_path" \
            "SELECT DISTINCT job_dir
             FROM structures
             WHERE status = 'completed'
               AND job_dir IS NOT NULL
               AND TRIM(job_dir) != ''
             ORDER BY job_dir;"
    )

    echo "Completed rows selected: $selected_count"

    if (( skipped_missing > 0 )); then
        echo "Skipped missing job directories: $skipped_missing" >&2
        for job_dir in "${missing_examples[@]}"; do
            echo "  missing: $job_dir" >&2
        done
    fi

    if (( skipped_recent > 0 )); then
        echo "Skipped recently modified job directories: $skipped_recent" >&2
        echo "  quiet period: ${GLOBUS_TRANSFER_MIN_FILE_AGE_MINUTES} minutes" >&2
        for job_dir in "${recent_examples[@]}"; do
            echo "  recent: $job_dir" >&2
        done
    fi

    if (( valid_count == 0 )); then
        echo "Error: no valid completed job directories were found to transfer." >&2
        exit 1
    fi

    local label
    label="completed-job-dir transfer $(basename "$db_path")"

    local task_id
    if ! task_id="$(
        globus transfer \
            "$source_endpoint_id" \
            "$DESTINATION_ENDPOINT_ID" \
            --batch "$batch_file" \
            --label "$label" \
            --jmespath 'task_id' \
            --format unix \
            --notify off
    )"; then
        echo "Error: Globus transfer submission failed." >&2
        exit 1
    fi

    echo "Valid job directories queued: $valid_count"
    echo "Submitted Globus task: $task_id"
}

main "$@"

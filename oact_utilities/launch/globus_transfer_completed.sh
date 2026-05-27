#!/usr/bin/env bash
# globus_transfer_completed.sh -- submit one bulk recursive Globus transfer.
#
# Usage:
#   bash globus_transfer_completed.sh <db_path> <dest_root>
#
# The script:
#   1. Starts Globus Connect Personal in the background if needed
#   2. Derives a common source root from job directories stored in the DB
#   3. Submits one recursive Globus transfer task for that root
#   4. Exits without waiting for transfer completion

set -euo pipefail

readonly CARPENTER_SOURCE_ENDPOINT_ID="b808a48a-4b2d-11f1-a9a0-02535127e3d7"
readonly BARFOOT_SOURCE_ENDPOINT_ID="1ea1ecb5-4d77-11f1-848e-0ea3589134b3"
readonly DESTINATION_ENDPOINT_ID="05d2c76a-e867-4f67-aa57-76edeb0beda0"

GLOBUS_CONNECT_PERSONAL_BIN="${GLOBUS_CONNECT_PERSONAL_BIN:-globusconnectpersonal}"
GLOBUS_CONNECT_STARTUP_WAIT="${GLOBUS_CONNECT_STARTUP_WAIT:-2}"

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
    require_cmd pgrep
    require_cmd python3

    local source_endpoint_id
    source_endpoint_id="$(detect_source_endpoint_id)"

    ensure_globus_connect_personal

    local temp_dir all_dirs_file analysis_file
    temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/globus-transfer-bulk.XXXXXX")"
    trap 'rm -rf -- "${temp_dir:-}"' EXIT
    all_dirs_file="$temp_dir/all_dirs.txt"
    analysis_file="$temp_dir/analysis.json"

    sqlite3 -batch -noheader "$db_path" \
        "SELECT DISTINCT job_dir
         FROM structures
         WHERE job_dir IS NOT NULL
           AND TRIM(job_dir) != ''
         ORDER BY job_dir;" >"$all_dirs_file"

    python3 - "$all_dirs_file" >"$analysis_file" <<'PY'
import json
import os
import sys
from pathlib import Path

all_dirs_path = Path(sys.argv[1])

dirs = []
for line in all_dirs_path.read_text().splitlines():
    stripped = line.strip()
    if stripped:
        dirs.append(Path(stripped))

if not dirs:
    print(json.dumps({"error": "no_job_dirs"}))
    raise SystemExit(0)

existing_dirs = [path.resolve() for path in dirs if path.is_dir()]
missing_dirs = [str(path) for path in dirs if not path.is_dir()]

if not existing_dirs:
    print(
        json.dumps(
            {
                "error": "no_existing_job_dirs",
                "missing_examples": missing_dirs[:3],
            }
        )
    )
    raise SystemExit(0)

if len(existing_dirs) == 1:
    source_root = existing_dirs[0].parent
else:
    source_root = Path(os.path.commonpath([str(path) for path in existing_dirs]))

payload = {
    "source_root": str(source_root),
    "total_db_dirs": len(dirs),
    "existing_dirs": len(existing_dirs),
    "missing_dirs": len(missing_dirs),
    "missing_examples": missing_dirs[:3],
}
print(json.dumps(payload))
PY

    local analysis_error
    analysis_error="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("error",""))' "$analysis_file")"
    if [[ -n "$analysis_error" ]]; then
        case "$analysis_error" in
            no_job_dirs)
                echo "Error: no job directories were found in the DB." >&2
                ;;
            no_existing_job_dirs)
                echo "Error: job directories were found in the DB, but none exist on disk." >&2
                python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); [print(f"  missing: {x}", file=sys.stderr) for x in data.get("missing_examples", [])]' "$analysis_file"
                ;;
            *)
                echo "Error: unexpected transfer analysis error '$analysis_error'." >&2
                ;;
        esac
        exit 1
    fi

    local source_root total_db_dirs existing_dirs missing_dirs
    source_root="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_root"])' "$analysis_file")"
    total_db_dirs="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_db_dirs"])' "$analysis_file")"
    existing_dirs="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["existing_dirs"])' "$analysis_file")"
    missing_dirs="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["missing_dirs"])' "$analysis_file")"

    echo "Source root: $source_root"
    echo "Distinct job directories in DB: $total_db_dirs"
    echo "Existing job directories under source root: $existing_dirs"

    if (( missing_dirs > 0 )); then
        echo "Missing DB job directories: $missing_dirs" >&2
        python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); [print(f"  missing: {x}", file=sys.stderr) for x in data.get("missing_examples", [])]' "$analysis_file"
    fi

    local label
    label="bulk recursive transfer $(basename "$db_path")"

    local task_id
    if ! task_id="$(
        globus transfer \
            "${source_endpoint_id}:${source_root%/}/" \
            "${DESTINATION_ENDPOINT_ID}:${dest_root}/" \
            --recursive \
            --label "$label" \
            --jmespath 'task_id' \
            --format unix \
            --sync-level size \
            --notify off
    )"; then
        echo "Error: Globus transfer submission failed." >&2
        exit 1
    fi

    echo "Submitted Globus task: $task_id"
}

main "$@"

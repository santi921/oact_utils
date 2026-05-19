#!/usr/bin/env bash
# globus_transfer_completed.sh -- submit one recursive Globus transfer for
# completed jobs while excluding DB-identified in-progress directories.
#
# Usage:
#   bash globus_transfer_completed.sh <db_path> <dest_root>
#
# The script:
#   1. Starts Globus Connect Personal in the background if needed
#   2. Derives a common source root from completed job directories in the DB
#   3. Excludes whole DB directories that are not transfer-eligible
#   4. Submits one recursive Globus transfer task
#   5. Exits without waiting for transfer completion

set -euo pipefail

readonly CARPENTER_SOURCE_ENDPOINT_ID="b808a48a-4b2d-11f1-a9a0-02535127e3d7"
readonly BARFOOT_SOURCE_ENDPOINT_ID="1ea1ecb5-4d77-11f1-848e-0ea3589134b3"
readonly DESTINATION_ENDPOINT_ID="05d2c76a-e867-4f67-aa57-76edeb0beda0"

GLOBUS_CONNECT_PERSONAL_BIN="${GLOBUS_CONNECT_PERSONAL_BIN:-globusconnectpersonal}"
GLOBUS_CONNECT_STARTUP_WAIT="${GLOBUS_CONNECT_STARTUP_WAIT:-2}"
GLOBUS_TRANSFER_MIN_UPDATE_AGE_MINUTES="${GLOBUS_TRANSFER_MIN_UPDATE_AGE_MINUTES:-${GLOBUS_TRANSFER_MIN_FILE_AGE_MINUTES:-5}}"

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
    require_cmd mktemp
    require_cmd python3

    if ! [[ "$GLOBUS_TRANSFER_MIN_UPDATE_AGE_MINUTES" =~ ^[0-9]+$ ]]; then
        echo "Error: GLOBUS_TRANSFER_MIN_UPDATE_AGE_MINUTES must be a nonnegative integer." >&2
        exit 1
    fi

    if ! sqlite3 -batch -noheader "$db_path" "SELECT updated_at FROM structures LIMIT 0;" >/dev/null 2>&1; then
        echo "Error: database is missing required column 'updated_at' for transfer quiet-period filtering." >&2
        exit 1
    fi

    local source_endpoint_id
    source_endpoint_id="$(detect_source_endpoint_id)"

    ensure_globus_connect_personal

    local temp_dir
    temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/globus-transfer-completed.XXXXXX")"
    trap 'rm -rf -- "$temp_dir"' EXIT

    local all_dirs_file eligible_dirs_file recent_dirs_file analysis_file
    all_dirs_file="$temp_dir/all_dirs.txt"
    eligible_dirs_file="$temp_dir/eligible_dirs.txt"
    recent_dirs_file="$temp_dir/recent_dirs.txt"
    analysis_file="$temp_dir/analysis.json"

    local recent_sql_clause=""
    local recent_sql_suffix=""
    if (( GLOBUS_TRANSFER_MIN_UPDATE_AGE_MINUTES > 0 )); then
        recent_sql_clause="AND updated_at <= datetime('now', '-${GLOBUS_TRANSFER_MIN_UPDATE_AGE_MINUTES} minutes')"
        recent_sql_suffix="AND updated_at > datetime('now', '-${GLOBUS_TRANSFER_MIN_UPDATE_AGE_MINUTES} minutes')"
    fi

    sqlite3 -batch -noheader "$db_path" \
        "SELECT DISTINCT job_dir
         FROM structures
         WHERE job_dir IS NOT NULL
           AND TRIM(job_dir) != ''
         ORDER BY job_dir;" >"$all_dirs_file"

    sqlite3 -batch -noheader "$db_path" \
        "SELECT DISTINCT job_dir
         FROM structures
         WHERE status = 'completed'
           AND job_dir IS NOT NULL
           AND TRIM(job_dir) != ''
           ${recent_sql_clause}
         ORDER BY job_dir;" >"$eligible_dirs_file"

    sqlite3 -batch -noheader "$db_path" \
        "SELECT DISTINCT job_dir
         FROM structures
         WHERE status = 'completed'
           AND job_dir IS NOT NULL
           AND TRIM(job_dir) != ''
           ${recent_sql_suffix}
         ORDER BY job_dir;" >"$recent_dirs_file"

    python3 - "$all_dirs_file" "$eligible_dirs_file" "$recent_dirs_file" >"$analysis_file" <<'PY'
import json
import sys
from pathlib import Path

all_dirs_path = Path(sys.argv[1])
eligible_dirs_path = Path(sys.argv[2])
recent_dirs_path = Path(sys.argv[3])


def read_lines(path: Path) -> list[Path]:
    values = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            values.append(Path(stripped))
    return values


all_dirs = read_lines(all_dirs_path)
eligible_dirs = read_lines(eligible_dirs_path)
recent_dirs = read_lines(recent_dirs_path)

if not eligible_dirs:
    print(json.dumps({"error": "no_eligible_dirs"}))
    raise SystemExit(0)

eligible_existing = [path.resolve() for path in eligible_dirs if path.is_dir()]
missing_eligible = [str(path) for path in eligible_dirs if not path.is_dir()]

if not eligible_existing:
    print(
        json.dumps(
            {
                "error": "no_existing_eligible_dirs",
                "missing_examples": missing_eligible[:3],
            }
        )
    )
    raise SystemExit(0)

if len(eligible_existing) == 1:
    source_root = eligible_existing[0].parent
else:
    source_root = Path(Path.commonpath([str(path) for path in eligible_existing]))

completed_rel = []
for path in eligible_existing:
    try:
        completed_rel.append(path.resolve().relative_to(source_root).as_posix())
    except ValueError:
        print(
            json.dumps(
                {
                    "error": "inconsistent_root",
                    "source_root": str(source_root),
                    "offending_path": str(path),
                }
            )
        )
        raise SystemExit(0)

exclude_rel = []
outside_root = []
for path in all_dirs:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(source_root).as_posix()
    except ValueError:
        outside_root.append(str(path))
        continue
    if relative not in completed_rel:
        exclude_rel.append(relative)

payload = {
    "source_root": str(source_root),
    "eligible_count": len(eligible_dirs),
    "valid_count": len(eligible_existing),
    "total_db_dirs": len(all_dirs),
    "skipped_missing": len(missing_eligible),
    "missing_examples": missing_eligible[:3],
    "skipped_recent": len(recent_dirs),
    "recent_examples": [str(path) for path in recent_dirs[:3]],
    "exclude_rel_paths": sorted(set(exclude_rel)),
    "outside_root_examples": outside_root[:3],
}
print(json.dumps(payload))
PY

    local analysis_error
    analysis_error="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("error",""))' "$analysis_file")"
    if [[ -n "$analysis_error" ]]; then
        case "$analysis_error" in
            no_eligible_dirs)
                echo "Error: no completed job directories were eligible for transfer." >&2
                ;;
            no_existing_eligible_dirs)
                echo "Error: completed job directories were found in the DB, but none exist on disk." >&2
                python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); [print(f"  missing: {x}", file=sys.stderr) for x in data.get("missing_examples", [])]' "$analysis_file"
                ;;
            inconsistent_root)
                echo "Error: eligible job directories do not share a stable source root." >&2
                python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); print(f"  source_root: {data.get(\"source_root\")}", file=sys.stderr); print(f"  offending_path: {data.get(\"offending_path\")}", file=sys.stderr)' "$analysis_file"
                ;;
            *)
                echo "Error: unexpected transfer analysis error '$analysis_error'." >&2
                ;;
        esac
        exit 1
    fi

    local source_root total_db_dirs eligible_count valid_count skipped_missing skipped_recent
    source_root="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_root"])' "$analysis_file")"
    total_db_dirs="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_db_dirs"])' "$analysis_file")"
    eligible_count="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["eligible_count"])' "$analysis_file")"
    valid_count="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["valid_count"])' "$analysis_file")"
    skipped_missing="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["skipped_missing"])' "$analysis_file")"
    skipped_recent="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["skipped_recent"])' "$analysis_file")"
    local outside_root_count
    outside_root_count="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("outside_root_examples", [])))' "$analysis_file")"

    echo "Source root: $source_root"
    echo "Distinct job directories in DB: $total_db_dirs"

    if (( skipped_missing > 0 )); then
        echo "Skipped missing completed job directories: $skipped_missing" >&2
        python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); [print(f"  missing: {x}", file=sys.stderr) for x in data.get("missing_examples", [])]' "$analysis_file"
    fi

    if (( skipped_recent > 0 )); then
        echo "Skipped recently updated completed job directories: $skipped_recent" >&2
        echo "  quiet period: ${GLOBUS_TRANSFER_MIN_UPDATE_AGE_MINUTES} minutes (from DB updated_at)" >&2
        python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); [print(f"  recent: {x}", file=sys.stderr) for x in data.get("recent_examples", [])]' "$analysis_file"
    fi

    if (( outside_root_count > 0 )); then
        echo "Warning: some DB job directories fall outside the inferred source root and will not be covered." >&2
        python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); [print(f"  outside-root: {x}", file=sys.stderr) for x in data.get("outside_root_examples", [])]' "$analysis_file"
    fi

    local -a exclude_args=()
    local exclude_db_count=0
    local rel_path
    while IFS= read -r rel_path; do
        [[ -z "$rel_path" ]] && continue
        exclude_args+=(--exclude "${rel_path}")
        exclude_args+=(--exclude "${rel_path}/*")
        exclude_db_count=$((exclude_db_count + 1))
    done < <(
        python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); [print(x) for x in data.get("exclude_rel_paths", [])]' "$analysis_file"
    )

    local label
    label="completed recursive transfer $(basename "$db_path")"

    local task_id
    if ! task_id="$(
        globus transfer \
            "${source_endpoint_id}:${source_root%/}/" \
            "${DESTINATION_ENDPOINT_ID}:${dest_root}/" \
            --recursive \
            "${exclude_args[@]}" \
            --label "$label" \
            --jmespath 'task_id' \
            --format unix \
            --sync-level size \
            --notify off
    )"; then
        echo "Error: Globus transfer submission failed." >&2
        exit 1
    fi

    echo "Eligible completed job directories after DB quiet period: $eligible_count"
    echo "Existing completed job directories covered by recursive root: $valid_count"
    echo "Excluded non-eligible DB paths: $exclude_db_count"
    echo "Submitted Globus task: $task_id"
}

main "$@"

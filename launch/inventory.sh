#!/usr/bin/env bash
#
# inventory.sh -- standalone, dependency-free port of
# oact_utilities/workflows/inventory.py. No Python, no conda, no package.
# Just coreutils + awk. Works with GNU find/stat (Linux HPC) or BSD find/stat
# (macOS); it detects which at startup.
#
# Walks every immediate subdirectory of a root as a "job directory" and reports,
# per directory and in aggregate: file/subdir counts, cumulative size, breakdown
# by file type, nested orca_tmp_* scratch dirs, and the largest dirs/files.
# Optionally deletes direct-child scratch (.tmp / core / .ges / orca_tmp_*/ ; .bas) from
# EVERY directory regardless of any DB -- only run that when nothing is executing.
#
# Usage:
#   ./inventory.sh JOBS_DIR
#   ./inventory.sh JOBS_DIR --top 40 --csv inventory.csv
#   ./inventory.sh JOBS_DIR --clean-tmp            # dry-run preview
#   ./inventory.sh JOBS_DIR --clean-tmp --execute  # actually delete
#
# This is a one-off helper, intentionally not part of the package.

set -o pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
ROOT=""
TOP=20
CSV=""
CAT_TMP=0
CAT_BAS=0
EXECUTE=0
WORKERS=8

usage() {
    sed -n '3,30p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --top)        TOP="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        --csv)        CSV="$2"; shift 2 ;;
        --clean-tmp)  CAT_TMP=1; shift ;;
        --clean-bas)  CAT_BAS=1; shift ;;
        --clean-all)  CAT_TMP=1; CAT_BAS=1; shift ;;
        --execute)    EXECUTE=1; shift ;;
        -h|--help)    usage 0 ;;
        -*)           echo "Unknown option: $1" >&2; usage 1 ;;
        *)            if [ -z "$ROOT" ]; then ROOT="$1"; else echo "Unexpected arg: $1" >&2; usage 1; fi; shift ;;
    esac
done

[ -n "$ROOT" ]   || { echo "Error: ROOT directory required." >&2; usage 1; }
ROOT="${ROOT%/}"
[ -d "$ROOT" ]   || { echo "Error: not a directory: $ROOT" >&2; exit 1; }
if [ "$EXECUTE" = 1 ] && [ "$CAT_TMP" = 0 ] && [ "$CAT_BAS" = 0 ]; then
    echo "Error: --execute requires a clean flag (--clean-tmp/--clean-bas/--clean-all)" >&2
    exit 1
fi
CLEANING=0
[ "$CAT_TMP" = 1 ] || [ "$CAT_BAS" = 1 ] && CLEANING=1

# ---------------------------------------------------------------------------
# Platform detection (GNU vs BSD find/stat)
# ---------------------------------------------------------------------------
if find . -maxdepth 0 -printf '' >/dev/null 2>&1; then GNU_FIND=1; else GNU_FIND=0; fi
if stat -c%s . >/dev/null 2>&1; then STAT_FLAVOR=gnu
elif stat -f%z . >/dev/null 2>&1; then STAT_FLAVOR=bsd
else STAT_FLAVOR=none; fi
if [ "$GNU_FIND" = 0 ] && [ "$STAT_FLAVOR" = none ]; then
    echo "Error: need GNU find (-printf) or a working stat (-c/-f)." >&2
    exit 1
fi

file_size() {  # bytes of a single regular file
    case "$STAT_FLAVOR" in
        gnu) stat -c%s "$1" 2>/dev/null || echo 0 ;;
        bsd) stat -f%z "$1" 2>/dev/null || echo 0 ;;
        *)   if [ "$GNU_FIND" = 1 ]; then find "$1" -maxdepth 0 -printf '%s' 2>/dev/null || echo 0; else echo 0; fi ;;
    esac
}

dir_size() {  # recursive byte sum of regular files under a dir (symlinks excluded)
    if [ "$GNU_FIND" = 1 ]; then
        find "$1" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}'
    else
        find "$1" -type f -print0 2>/dev/null | xargs -0 stat -f%z 2>/dev/null | awk '{s+=$1} END{print s+0}'
    fi
}

fmt() {  # human-readable size, identical rounding to the Python _format_size
    awk -v b="$1" 'BEGIN{
        if (b<1024) printf "%d B", b;
        else if (b<1048576) printf "%.1f KB", b/1024;
        else if (b<1073741824) printf "%.1f MB", b/1048576;
        else printf "%.1f GB", b/1073741824;
    }'
}

# ---------------------------------------------------------------------------
# Temp workspace
# ---------------------------------------------------------------------------
TMP="$(mktemp -d "${TMPDIR:-/tmp}/inventory.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
LISTING="$TMP/listing.tsv"   # type \t size \t relpath
TOTALS="$TMP/totals.tsv"
CLASSF="$TMP/class.tsv"
EXTF="$TMP/ext.tsv"
PJ="$TMP/perjob.tsv"

# ---------------------------------------------------------------------------
# Build the listing (type, size, absolute path). Symlinks are excluded by
# -type f/-type d and never followed (no -L), mirroring inventory.py. The walk
# is fanned out one job directory per worker via xargs -P, matching the Python
# ThreadPoolExecutor -- this is what hides per-stat latency on a network FS.
# Each worker writes its records to a private part file (no interleaving) and
# emits one line per finished job dir to drive the progress counter.
# ---------------------------------------------------------------------------
PARTS="$TMP/parts"
mkdir -p "$PARTS"

# Emit "type<TAB>size<TAB>abspath" for one job directory (passed absolute).
scan_one() {
    local job="$1" part="$PARTS/$$.$RANDOM"
    if [ "$GNU_FIND" = 1 ]; then
        {
            find "$job" -type d -printf 'd\t0\t%p\n'
            find "$job" -type f -printf 'f\t%s\t%p\n'
        } >> "$part" 2>/dev/null
    else
        find "$job" -type d 2>/dev/null | awk 'BEGIN{OFS="\t"}{print "d",0,$0}' >> "$part"
        find "$job" -type f -print0 2>/dev/null \
            | xargs -0 stat -f '%z%t%N' 2>/dev/null \
            | awk -F'\t' 'BEGIN{OFS="\t"}{print "f",$1,$2}' >> "$part"
    fi
    printf '%s\n' "$job"   # one progress tick per completed job dir
}
export -f scan_one
export PARTS GNU_FIND STAT_FLAVOR

TOTAL=$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
echo "Scanning $TOTAL job directories under $ROOT (workers: $WORKERS) ..." >&2

find "$ROOT" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null \
    | xargs -0 -P "$WORKERS" -n1 bash -c 'scan_one "$1"' _ \
    | awk -v t="$TOTAL" '
        { c++; printf "\r  scanned %d/%d job dirs", c, t > "/dev/stderr" }
        END { printf "\r  scanned %d/%d job dirs\n", c+0, t > "/dev/stderr" }'

find "$PARTS" -type f -exec cat {} + > "$LISTING" 2>/dev/null

# ---------------------------------------------------------------------------
# Aggregate everything in one awk pass.
# ---------------------------------------------------------------------------
awk -F'\t' -v TOT="$TOTALS" -v CLS="$CLASSF" -v EXT="$EXTF" -v PJ="$PJ" -v ROOT="$ROOT" '
BEGIN {
    OFS = "\t"
    rlen = length(ROOT)
    n = split("orca.out orca.out.gz orca.inp orca.inp.gz orca.engrad orca.engrad.gz orca.xyz orca_metrics.json generator_metrics.json orca.property.txt orca.property.txt.gz orca.gbw orca.densities flux_job.flux slurm_job.sh sella_status.txt sella.log opt.traj sella_config.json run_sella.py results.pkl sella_driver.log .do_not_rerun.json", a, " ")
    for (i = 1; i <= n; i++) ESS[a[i]] = 1
}
function strip(s) {
    # Drop the leading "<ROOT>/" so the rest of the logic sees job-relative paths.
    if (substr(s, 1, rlen + 1) == ROOT "/") return substr(s, rlen + 2)
    return s
}
function is_essential(name) {
    if (name in ESS) return 1
    if (name ~ /^orca_atom[0-9]+\.out$/) return 1
    if (name ~ /_trj\.xyz$/) return 1
    return 0
}
function classify(name, in_tmp) {
    if (in_tmp) return "scratch_tmp"
    if (is_essential(name)) return "essential"
    if (name ~ /\.tmp$/ || name ~ /\.tmp\.[0-9]+$/ || name == "core" || name ~ /^core\.[0-9]+$/ || name ~ /\.core$/ || name ~ /\.ges$/) return "scratch_tmp"
    if (name ~ /\.bas$/ || name ~ /\.bas[0-9]+$/) return "scratch_bas"
    return "other"
}
function ext_key(name,   m, k, last, prev) {
    if (name ~ /\.tmp$/ || name ~ /\.tmp\.[0-9]+$/) return ".tmp"
    if (name ~ /\.bas$/ || name ~ /\.bas[0-9]+$/)   return ".bas"
    if (name == "core" || name ~ /^core\.[0-9]+$/ || name ~ /\.core$/) return "core"
    if (index(name, ".") == 0) return "(no ext)"
    m = split(name, k, ".")
    last = "." tolower(k[m])
    if (last == ".gz" && m >= 3) { prev = "." tolower(k[m-1]); return prev last }
    return last
}
{
    type = $1; size = $2 + 0; rel = strip($3)
    np = split(rel, p, "/")
    base = p[np]
    if (type == "d") {
        if (np == 1) { seenjob[rel] = 1; njobs++; next }
        job = p[1]
        subdirs[job]++; total_subdirs++
        if (base ~ /^orca_tmp_[A-Za-z0-9_]+$/) { ntdirs[job]++; nested_dirs++ }
        next
    }
    # type == f
    if (np == 1) { loose_files++; loose_bytes += size; next }
    job = p[1]
    in_tmp = 0
    for (i = 1; i < np; i++) if (p[i] ~ /^orca_tmp_[A-Za-z0-9_]+$/) in_tmp = 1
    total_files++; total_bytes += size
    files[job]++; bytes[job] += size
    e = ext_key(base); ext_b[e] += size; ext_c[e]++
    c = classify(base, in_tmp); class_b[c] += size; class_c[c]++
    jclass[job, c] += size
    if (in_tmp) { nested_bytes += size; jntb[job] += size }
}
END {
    jwn = 0
    for (j in ntdirs) if (ntdirs[j] > 0) jwn++
    print total_files+0, total_subdirs+0, total_bytes+0, loose_files+0, loose_bytes+0, njobs+0, jwn+0, nested_dirs+0, nested_bytes+0 > TOT
    for (c in class_b) print c, class_c[c], class_b[c] > CLS
    for (e in ext_b)   print e, ext_c[e], ext_b[e] > EXT
    for (j in seenjob)
        print j, files[j]+0, subdirs[j]+0, bytes[j]+0, ntdirs[j]+0, jntb[j]+0, \
              jclass[j,"essential"]+0, jclass[j,"scratch_tmp"]+0, jclass[j,"scratch_bas"]+0, jclass[j,"other"]+0 > PJ
}
' "$LISTING"

# ---------------------------------------------------------------------------
# Read totals
# ---------------------------------------------------------------------------
IFS=$'\t' read -r T_FILES T_SUBDIRS T_BYTES L_FILES L_BYTES N_JOBS JWN NDIRS NBYTES < "$TOTALS"

class_val() { awk -F'\t' -v c="$1" -v col="$2" '$1==c{print $col; f=1} END{if(!f)print 0}' "$CLASSF"; }

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
echo
printf '======================================================================\n'
printf '  Job Directory File Inventory\n'
printf '======================================================================\n'
printf '  Root:            %s\n' "$ROOT"
printf '  Job directories: %s\n' "$N_JOBS"
printf '  Total files:     %s\n' "$T_FILES"
printf '  Total subdirs:   %s\n' "$T_SUBDIRS"
printf '  Total size:      %s\n' "$(fmt "$T_BYTES")"
if [ "$N_JOBS" -gt 0 ]; then
    printf '  Mean per job:    %s (%s files)\n' "$(fmt $((T_BYTES / N_JOBS)))" "$((T_FILES / N_JOBS))"
fi
if [ "$L_FILES" -gt 0 ]; then
    printf '  Loose files at root (not in a job dir): %s (%s)\n' "$L_FILES" "$(fmt "$L_BYTES")"
fi

printf '\n--- By classification (matches clean.py delete rules) ---\n'
printf '  class | files | size | %% of size\n'
ESS_BYTES=0
for c in essential scratch_tmp scratch_bas other; do
    cnt=$(class_val "$c" 2); by=$(class_val "$c" 3)
    [ "$c" = essential ] && ESS_BYTES=$by
    pct=$(awk -v p="$by" -v w="$T_BYTES" 'BEGIN{printf (w>0)?"%.1f%%":"0.0%%", (w>0)?100*p/w:0}')
    printf '  %s | %s | %s | %s\n' "$c" "$cnt" "$(fmt "$by")" "$pct"
done
RECL=$((T_BYTES - ESS_BYTES))
rpct=$(awk -v p="$RECL" -v w="$T_BYTES" 'BEGIN{printf (w>0)?"%.1f%%":"0.0%%", (w>0)?100*p/w:0}')
printf '  -> reclaimable (non-essential): %s (%s)\n' "$(fmt "$RECL")" "$rpct"

printf '\n--- By file type (top %s by size) ---\n' "$TOP"
printf '  type | files | size | %% of size\n'
sort -t$'\t' -k3,3 -rn "$EXTF" | head -n "$TOP" | while IFS=$'\t' read -r e cnt by; do
    pct=$(awk -v p="$by" -v w="$T_BYTES" 'BEGIN{printf (w>0)?"%.1f%%":"0.0%%", (w>0)?100*p/w:0}')
    printf '  %s | %s | %s | %s\n' "$e" "$cnt" "$(fmt "$by")" "$pct"
done

printf '\n--- Nested scratch dirs (orca_tmp_*) ---\n'
printf '  Job dirs with nested tmp dirs: %s of %s\n' "$JWN" "$N_JOBS"
printf '  Total nested tmp dirs:         %s\n' "$NDIRS"
printf '  Bytes inside nested tmp dirs:  %s\n' "$(fmt "$NBYTES")"

# ---------------------------------------------------------------------------
# Optional scratch deletion (DB-blind), run AFTER inventory so the report above
# reflects the pre-clean state.
# ---------------------------------------------------------------------------
if [ "$CLEANING" = 1 ]; then
    cats=""; [ "$CAT_BAS" = 1 ] && cats="bas"; [ "$CAT_TMP" = 1 ] && cats="${cats:+$cats, }tmp"
    mode="DRY RUN (preview only)"; [ "$EXECUTE" = 1 ] && mode="EXECUTE (deleting)"
    printf '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n' >&2
    printf '  SCRATCH DELETION [%s] -- categories: %s\n' "$mode" "$cats" >&2
    printf '  DB-BLIND: deletes matched scratch from EVERY directory on disk.\n' >&2
    printf '  Only --execute when the campaign is finished and nothing runs.\n' >&2
    printf '  Essential files (orca.out/inp/engrad/gbw, sella state, ...) are never matched.\n' >&2
    printf '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n' >&2

    if [ "$GNU_FIND" = 1 ]; then
        # ---- Fast parallel clean: one find+awk per job dir, no per-entry spawns ----
        # The awk reads a single recursive listing of the job dir and emits the
        # direct-child scratch to delete (F=file, D=orca_tmp dir) plus a TOTAL
        # count/bytes line. Because awk only prints in END, enumeration finishes
        # before any rm -- no delete-while-traversing hazard.
        CLEAN_AWK="$TMP/clean.awk"
        cat > "$CLEAN_AWK" <<'AWK'
BEGIN {
    OFS = "\t"
    n = split("orca.out orca.out.gz orca.inp orca.inp.gz orca.engrad orca.engrad.gz orca.xyz orca_metrics.json generator_metrics.json orca.property.txt orca.property.txt.gz orca.gbw orca.densities flux_job.flux slurm_job.sh sella_status.txt sella.log opt.traj sella_config.json run_sella.py results.pkl sella_driver.log .do_not_rerun.json", a, " ")
    for (i = 1; i <= n; i++) ESS[a[i]] = 1
}
function ess(nm) { return (nm in ESS) || nm ~ /^orca_atom[0-9]+\.out$/ || nm ~ /_trj\.xyz$/ }
{ ty[NR] = $1; sz[NR] = $2 + 0; rel[NR] = $3 }
END {
    for (i = 1; i <= NR; i++)
        if (index(rel[i], "/") == 0 && ty[i] == "d" && ct == 1 && rel[i] ~ /^orca_tmp_[A-Za-z0-9_]+$/)
            tmpdir[rel[i]] = 1
    count = 0; bytes = 0
    for (i = 1; i <= NR; i++) {
        r = rel[i]; s = sz[i]; t = ty[i]; p = index(r, "/")
        if (p == 0) {
            if (t == "f") {
                if (ess(r)) continue
                m = 0
                if (ct == 1 && (r ~ /\.tmp$/ || r ~ /\.tmp\.[0-9]+$/ || r == "core" || r ~ /^core\.[0-9]+$/ || r ~ /\.core$/ || r ~ /\.ges$/)) m = 1
                if (cb == 1 && (r ~ /\.bas$/ || r ~ /\.bas[0-9]+$/)) m = 1
                if (m) { count++; bytes += s; print "F", job "/" r }
            } else if (t == "d" && (r in tmpdir)) {
                count++; print "D", job "/" r   # bytes summed from its children below
            }
        } else {
            first = substr(r, 1, p - 1)
            if (t == "f" && (first in tmpdir)) bytes += s
        }
    }
    print "TOTAL", count, bytes
}
AWK
        PARTS_CLEAN="$TMP/clean_parts"; mkdir -p "$PARTS_CLEAN"
        clean_one() {
            local job="$1" tag a b count=0 bytes=0
            while IFS=$'\t' read -r tag a b; do
                if [ "$tag" = TOTAL ]; then count="$a"; bytes="$b"; continue; fi
                [ "$EXECUTE" = 1 ] || continue
                if [ "$tag" = F ]; then rm -f -- "$a"
                elif [ "$tag" = D ]; then if [ -L "$a" ]; then rm -f -- "$a"; else rm -rf -- "$a"; fi; fi
            done < <(find "$job" -mindepth 1 -printf '%y\t%s\t%P\n' 2>/dev/null \
                     | awk -F'\t' -v job="$job" -v ct="$CAT_TMP" -v cb="$CAT_BAS" -f "$CLEAN_AWK")
            printf '%s\t%s\n' "$count" "$bytes" >> "$PARTS_CLEAN/$$.$RANDOM"
            printf 'x\n'   # progress tick
        }
        export -f clean_one
        export CLEAN_AWK PARTS_CLEAN CAT_TMP CAT_BAS EXECUTE

        find "$ROOT" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null \
            | xargs -0 -P "$WORKERS" -n1 bash -c 'clean_one "$1"' _ \
            | awk -v t="$TOTAL" '
                { c++; printf "\r  processed %d/%d job dirs", c, t > "/dev/stderr" }
                END { printf "\r  processed %d/%d job dirs\n", c+0, t > "/dev/stderr" }'

        read -r C_MATCHED C_BYTES < <(find "$PARTS_CLEAN" -type f -exec cat {} + 2>/dev/null \
            | awk -F'\t' '{m+=$1; b+=$2} END{print m+0, b+0}')
        C_MATCHED=${C_MATCHED:-0}; C_BYTES=${C_BYTES:-0}
        [ "$EXECUTE" = 1 ] && C_FREED="$C_BYTES" || C_FREED=0
    else
        # ---- Fallback (no GNU find): slow per-entry loop with stat ----
        match_scratch() {  # $1=basename $2=is_dir(1/0); honors CAT_TMP/CAT_BAS
            local base="$1" isdir="$2"
            case "$base" in
                orca.out|orca.out.gz|orca.inp|orca.inp.gz|orca.engrad|orca.engrad.gz|orca.xyz|orca_metrics.json|generator_metrics.json|orca.property.txt|orca.property.txt.gz|orca.gbw|orca.densities|flux_job.flux|slurm_job.sh|sella_status.txt|sella.log|opt.traj|sella_config.json|run_sella.py|results.pkl|sella_driver.log|.do_not_rerun.json) return 1 ;;
            esac
            [[ "$base" =~ ^orca_atom[0-9]+\.out$ ]] && return 1
            [[ "$base" =~ _trj\.xyz$ ]] && return 1
            if [ "$CAT_TMP" = 1 ]; then
                if [ "$isdir" = 1 ]; then
                    [[ "$base" =~ ^orca_tmp_[A-Za-z0-9_]+$ ]] && return 0
                else
                    [[ "$base" =~ \.tmp$ ]] && return 0
                    [[ "$base" =~ \.tmp\.[0-9]+$ ]] && return 0
                    [ "$base" = core ] && return 0
                    [[ "$base" =~ ^core\.[0-9]+$ ]] && return 0
                    [[ "$base" =~ \.core$ ]] && return 0
                    [[ "$base" =~ \.ges$ ]] && return 0
                fi
            fi
            if [ "$CAT_BAS" = 1 ] && [ "$isdir" != 1 ]; then
                [[ "$base" =~ \.bas$ ]] && return 0
                [[ "$base" =~ \.bas[0-9]+$ ]] && return 0
            fi
            return 1
        }
        C_MATCHED=0; C_BYTES=0; C_FREED=0; C_DIRS=0
        shopt -s nullglob dotglob
        while IFS= read -r jobdir; do
            for entry in "$jobdir"/*; do
                base="${entry##*/}"
                if [ -d "$entry" ]; then isdir=1; else isdir=0; fi
                match_scratch "$base" "$isdir" || continue
                if [ -L "$entry" ]; then sz=0
                elif [ "$isdir" = 1 ]; then sz=$(dir_size "$entry")
                else sz=$(file_size "$entry"); fi
                C_MATCHED=$((C_MATCHED + 1)); C_BYTES=$((C_BYTES + sz))
                if [ "$EXECUTE" = 1 ]; then
                    if [ -L "$entry" ]; then rm -f -- "$entry" && C_FREED=$((C_FREED + sz))
                    elif [ "$isdir" = 1 ]; then rm -rf -- "$entry" && C_FREED=$((C_FREED + sz))
                    else rm -f -- "$entry" && C_FREED=$((C_FREED + sz)); fi
                fi
            done
            C_DIRS=$((C_DIRS + 1))
            printf '\r  processed %d/%d job dirs (%d matched)' "$C_DIRS" "$TOTAL" "$C_MATCHED" >&2
        done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
        printf '\n' >&2
        shopt -u nullglob dotglob
    fi

    verb="Would delete"; [ "$EXECUTE" = 1 ] && verb="Deleted"
    printf '\n--- Scratch deletion (%s) ---\n' "$cats"
    printf '  Entries matched (direct-child): %s\n' "$C_MATCHED"
    printf '  %s: %s\n' "$verb" "$(fmt "$C_BYTES")"
    if [ "$EXECUTE" = 1 ]; then
        printf '  Actually freed: %s\n' "$(fmt "$C_FREED")"
    else
        printf '  DRY RUN -- re-run with --execute to delete.\n'
    fi
fi

# ---------------------------------------------------------------------------
# Largest job directories / files (from the pre-clean listing)
# ---------------------------------------------------------------------------
printf '\n--- Largest %s job directories ---\n' "$TOP"
printf '  size | files | subdirs | nested_tmp | job_dir\n'
sort -t$'\t' -k4,4 -rn "$PJ" | head -n "$TOP" | while IFS=$'\t' read -r job f sub by nt ntb e t b o; do
    printf '  %s | %s | %s | %s | %s\n' "$(fmt "$by")" "$f" "$sub" "$nt" "$job"
done

printf '\n--- Largest %s individual files ---\n' "$TOP"
printf '  size | path\n'
awk -F'\t' -v ROOT="$ROOT" 'BEGIN{rlen=length(ROOT)} $1=="f"{p=$3; if(substr(p,1,rlen+1)==ROOT"/")p=substr(p,rlen+2); print $2"\t"p}' "$LISTING" \
    | sort -t$'\t' -k1,1 -rn | head -n "$TOP" | while IFS=$'\t' read -r sz path; do
    printf '  %s | %s\n' "$(fmt "$sz")" "$path"
done

# ---------------------------------------------------------------------------
# Optional per-job CSV
# ---------------------------------------------------------------------------
if [ -n "$CSV" ]; then
    {
        printf 'job_dir,n_files,n_subdirs,total_bytes,total_size,n_nested_tmp_dirs,nested_tmp_bytes,essential_bytes,scratch_tmp_bytes,scratch_bas_bytes,other_bytes,reclaimable_bytes\n'
        awk -F'\t' 'BEGIN{ OFS="," }
        function fmt(b){ if(b<1024)return b" B"; else if(b<1048576)return sprintf("%.1f KB",b/1024); else if(b<1073741824)return sprintf("%.1f MB",b/1048576); else return sprintf("%.1f GB",b/1073741824) }
        { recl=$4-$7; print "\""$1"\"",$2,$3,$4,"\""fmt($4)"\"",$5,$6,$7,$8,$9,$10,recl }
        ' "$PJ" | sort -t, -k4 -rn
    } > "$CSV"
    printf '\nPer-job inventory written to %s\n' "$CSV"
fi

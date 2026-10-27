"""Read-only inventory of an OpenAc-style job tree, run before writing an extractor.

Walks ``<root>/<category>/<group>/<job>/...`` (any depth), treats every directory holding an
``orca.inp`` as a job, and records what is on disk: the job-name tokens, the ``orca.inp``
header, which output / geometry / archive files exist, and whether the ORCA output terminated
normally and converged an optimization. Nothing is written inside ``root``.

Outputs:
    - a CSV with one row per job directory (``-o``)
    - a printed summary: per (category, group) counts, file-name pattern frequencies, the
      distinct ``!`` input lines, unparsed job names, sampled archive listings, and one full
      file listing per group

ORCA outputs (~200 MB each) are only tail-read for termination (``--tail-kb``). Geometry, cycle
count, energies and the final gradient come from ``orca.xyz``, ``orca_trj.xyz`` and
``orca.engrad``, which are small. ``--grep-converged`` adds a full-file grep for the
convergence banner.

Usage (on Perlmutter):
    python openac_inventory.py /pscratch/sd/s/santiago/openac -o openac_inventory.csv \
        --workers 16 > openac_inventory.log
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JOB_NAME_RE = re.compile(
    r"^(?P<lot>[^_]+)_(?P<formula>[A-Za-z0-9]+)_spin_(?P<spin>\d+)_(?P<functional>.+)_(?P<basis>[^_]+)$"
)
ORCA_ATOM_RE = re.compile(r"orca_atom\d+\.out")
ARCHIVE_SUFFIXES = (".tar.zst", ".tzst", ".tgz", ".tar.gz", ".tar")
GEOM_NAMES = (
    "orca.xyz",
    "orca_trj.xyz",
    "opt.traj",
    "sella.log",
    "orca.engrad",
    "orca.engrad.gz",
)


def iter_job_dirs(root: Path) -> Iterator[Path]:
    """Yield every directory under ``root`` that contains an ``orca.inp``."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if "orca.inp" in filenames or "orca.inp.gz" in filenames:
            yield Path(dirpath)


def read_text(path: Path) -> str:
    """Read a plain or gzipped text file, replacing undecodable bytes."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as f:
            return f.read()
    return path.read_text(errors="replace")


def parse_inp(path: Path) -> dict[str, object]:
    """Pull the simple-input line, charge, multiplicity, atom count and opt flags."""
    text = read_text(path)
    lines = text.splitlines()
    bang = " ".join(
        ln.strip()[1:].strip() for ln in lines if ln.strip().startswith("!")
    )
    rec: dict[str, object] = {
        "inp_simple": bang,
        "inp_has_opt": bool(
            re.search(r"\b(opt|looseopt|tightopt|verytightopt)\b", bang, re.I)
        ),
        "inp_has_geom_block": bool(re.search(r"^\s*%geom", text, re.I | re.M)),
        "inp_charge": None,
        "inp_mult": None,
        "inp_natoms": None,
        "inp_xyzfile": None,
    }
    for i, ln in enumerate(lines):
        s = ln.strip()
        m = re.match(r"^\*\s*xyzfile\s+(-?\d+)\s+(\d+)\s+(\S+)", s, re.I)
        if m:
            rec.update(inp_charge=int(m[1]), inp_mult=int(m[2]), inp_xyzfile=m[3])
            break
        m = re.match(r"^\*\s*xyz\s+(-?\d+)\s+(\d+)", s, re.I)
        if m:
            rec.update(inp_charge=int(m[1]), inp_mult=int(m[2]))
            n = 0
            for ln2 in lines[i + 1 :]:
                if ln2.strip().startswith("*"):
                    break
                n += bool(ln2.strip())
            rec["inp_natoms"] = n
            break
    return rec


CONVERGED = b"THE OPTIMIZATION HAS CONVERGED"
TRJ_E_RE = re.compile(r"\bE\s+(-?\d+\.\d+)")
# ORCA NormalOpt gradient tolerances (Eh/bohr); a proxy only, ORCA also checks energy and step.
TOL_MAXG, TOL_RMSG = 3e-4, 1e-4


def read_tail(path: Path, nbytes: int) -> str:
    """Last ``nbytes`` of a plain or gzipped file (gzip has no seek, so it is decompressed)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return f.read()[-nbytes:].decode(errors="replace")
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - nbytes))
        return f.read().decode(errors="replace")


def scan_out(path: Path, tail_kb: float, grep_converged: bool) -> dict[str, object]:
    """Termination and last error from the end of an ORCA output.

    Outputs here run 100-200 MB and the final property printout after the last optimization
    cycle alone can exceed 64 MB, so cycle counts, energies and convergence come from the small
    side files instead (see ``scan_trj`` / ``scan_engrad``). ``grep_converged`` additionally
    greps the whole file for the convergence banner with ``grep`` (C speed, but a full read).
    """
    last = read_tail(path, int(tail_kb * 1e3)).splitlines()[-10:]
    rec: dict[str, object] = {
        "out_file": path.name,
        "out_size_mb": round(path.stat().st_size / 1e6, 1),
        "out_terminated": "ORCA TERMINATED NORMALLY" in "\n".join(last),
        "out_error": "",
        "out_converged_banner": None,
    }
    if not rec["out_terminated"]:
        err = [
            ln.strip() for ln in last if "error" in ln.lower() or "abort" in ln.lower()
        ]
        rec["out_error"] = err[-1][:160] if err else ""
    if grep_converged and path.suffix != ".gz":
        res = subprocess.run(["grep", "-F", "-q", CONVERGED.decode(), str(path)])
        rec["out_converged_banner"] = res.returncode == 0
    return rec


def scan_xyz(path: Path) -> dict[str, object]:
    """Atom count and element string of ORCA's current/final geometry file."""
    lines = read_text(path).splitlines()
    n = int(lines[0].split()[0])
    return {
        "xyz_natoms": n,
        "xyz_elements": ";".join(ln.split()[0] for ln in lines[2 : 2 + n]),
    }


def scan_trj(path: Path) -> dict[str, object]:
    """Frame count (one per optimization cycle) and first / last frame energies from ``*_trj.xyz``."""
    lines = read_text(path).splitlines()
    energies: list[float] = []
    frames, i = 0, 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i].split()[0])
        m = TRJ_E_RE.search(lines[i + 1]) if i + 1 < len(lines) else None
        if m:
            energies.append(float(m[1]))
        frames += 1
        i += n + 2
    return {
        "trj_frames": frames,
        "trj_e_first_eh": energies[0] if energies else None,
        "trj_e_last_eh": energies[-1] if energies else None,
    }


def scan_engrad(path: Path) -> dict[str, object]:
    """Energy and gradient statistics from ``orca.engrad`` (gradient at the last geometry)."""
    vals = [
        ln.strip()
        for ln in read_text(path).splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    n = int(vals[0])
    grad = [float(v) for v in vals[2 : 2 + 3 * n]]
    norms = [sum(c * c for c in grad[3 * k : 3 * k + 3]) ** 0.5 for k in range(n)]
    maxg = max(abs(g) for g in grad)
    rmsg = (sum(g * g for g in grad) / len(grad)) ** 0.5
    return {
        "engrad_natoms": n,
        "engrad_energy_eh": float(vals[1]),
        "engrad_max_comp": maxg,
        "engrad_rms": rmsg,
        "engrad_max_atom_norm": max(norms),
        "grad_converged_normalopt": maxg <= TOL_MAXG and rmsg <= TOL_RMSG,
    }


def list_archive(path: Path) -> list[str]:
    """Member names of a tar / tar.gz / tar.zst archive (decompresses the stream; slow on big ones)."""
    if path.name.endswith((".tar.zst", ".tzst")):
        zstd = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
        res = subprocess.run(
            ["tar", "-tf", "-"], stdin=zstd.stdout, capture_output=True, text=True
        )
        zstd.wait()
    else:
        res = subprocess.run(["tar", "-tf", str(path)], capture_output=True, text=True)
    if res.returncode != 0:
        return [f"<tar failed: {res.stderr.strip()[:120]}>"]
    return res.stdout.split()


def file_pattern(name: str) -> str:
    """Collapse digit runs so ``orca.bas3`` and ``orca.bas12`` count as one pattern."""
    return re.sub(r"\d+", "N", name)


def inspect(
    job: Path, root: Path, tail_kb: float, grep_converged: bool
) -> dict[str, object]:
    """One inventory row for a job directory."""
    rel = job.relative_to(root)
    parts = rel.parts
    entries = sorted(os.listdir(job))
    files = [e for e in entries if (job / e).is_file()]
    subdirs = [e for e in entries if (job / e).is_dir()]
    rec: dict[str, object] = {
        "rel_path": str(rel),
        "depth": len(parts),
        "category": parts[0] if len(parts) > 0 else "",
        "group": parts[1] if len(parts) > 1 else "",
        "job": job.name,
        "n_files": len(files),
        "files": ";".join(files),
        "subdirs": ";".join(subdirs),
        "archives": ";".join(f for f in files if f.endswith(ARCHIVE_SUFFIXES)),
        "geom_files": ";".join(
            f for f in files if f in GEOM_NAMES or f.endswith(".xyz")
        ),
        "has_gbw": any(f.endswith((".gbw", ".gbw.gz")) for f in files),
        "inp_error": "",
    }
    m = JOB_NAME_RE.match(job.name)
    rec.update(
        name_parsed=bool(m),
        lot=m["lot"] if m else "",
        formula=m["formula"] if m else "",
        spin=int(m["spin"]) if m else None,
        functional=m["functional"] if m else "",
        basis=m["basis"] if m else "",
    )
    inp = job / ("orca.inp" if (job / "orca.inp").exists() else "orca.inp.gz")
    try:
        rec.update(parse_inp(inp))
    except Exception as exc:  # noqa: BLE001 -- inventory must not die on one bad file
        rec["inp_error"] = f"{type(exc).__name__}: {exc}"[:160]
    outs = [
        f
        for f in files
        if f.endswith((".out", ".out.gz"))
        and not ORCA_ATOM_RE.fullmatch(f.removesuffix(".gz"))
    ]
    # Restarted jobs leave one flux-<jobid>.out per attempt. Job IDs (base58, ASCII-ordered)
    # sort by submission time; mtimes do not survive a copy or tar extraction.
    flux = sorted((f for f in outs if f.startswith("flux-")), key=lambda f: (len(f), f))
    out = (
        "orca.out"
        if "orca.out" in outs
        else (
            "orca.out.gz"
            if "orca.out.gz" in outs
            else (flux[-1] if flux else (outs[0] if outs else None))
        )
    )
    if out:
        try:
            rec.update(scan_out(job / out, tail_kb, grep_converged))
        except Exception as exc:  # noqa: BLE001
            rec["out_error"] = f"{type(exc).__name__}: {exc}"[:160]
    else:
        rec["out_file"] = ""
    for fname, scanner in (
        ("orca.xyz", scan_xyz),
        ("orca_trj.xyz", scan_trj),
        ("orca.engrad", scan_engrad),
    ):
        hit = next((f for f in (fname, fname + ".gz") if f in files), None)
        if hit:
            try:
                rec.update(scanner(job / hit))
            except Exception as exc:  # noqa: BLE001
                rec[f"{fname}_error"] = f"{type(exc).__name__}: {exc}"[:160]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True, help="per-job CSV")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--list-archives",
        type=int,
        default=2,
        help="archives to list per group (0 = none)",
    )
    ap.add_argument(
        "--tail-kb",
        type=float,
        default=256.0,
        help="bytes of orca.out read from the end",
    )
    ap.add_argument(
        "--grep-converged",
        action="store_true",
        help="also grep each full orca.out for the convergence banner (reads every byte)",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    complete = root / "complete.txt"
    if complete.exists():
        lines = complete.read_text(errors="replace").splitlines()
        print(f"complete.txt: {len(lines)} lines; first 5:")
        for ln in lines[:5]:
            print(f"    {ln}")
        print()

    jobs = list(iter_job_dirs(root))
    print(f"{len(jobs)} job dirs (dirs holding orca.inp) under {root}\n")
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(
            pool.map(
                lambda j: inspect(j, root, args.tail_kb, args.grep_converged), jobs
            )
        )

    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows)")
    size = sum(float(r.get("out_size_mb") or 0) for r in rows)
    print(f"orca outputs on disk: {size / 1e3:.1f} GB\n")

    by_group: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_group[(str(r["category"]), str(r["group"]))].append(r)

    hdr = (
        f"{'category/group':<32} {'dirs':>5} {'depth':>6} {'parsed':>6} {'out':>5} {'term':>5} "
        f"{'opt_inp':>7} {'xyz':>5} {'trj':>5} {'engrad':>6} {'g_conv':>6} {'banner':>6} "
        f"{'arch':>5} {'gbw':>5}  functional/basis"
    )
    print(hdr)
    print("-" * len(hdr))
    for (cat, grp), rs in sorted(by_group.items()):
        fb = Counter(f"{r['functional']}/{r['basis']}" for r in rs if r["name_parsed"])
        n = lambda key, rs=rs: sum(bool(r.get(key)) for r in rs)  # noqa: E731
        print(
            f"{cat + '/' + grp:<32} {len(rs):>5} {','.join(sorted({str(r['depth']) for r in rs})):>6} "
            f"{n('name_parsed'):>6} {n('out_file'):>5} {n('out_terminated'):>5} {n('inp_has_opt'):>7} "
            f"{n('xyz_natoms'):>5} {n('trj_frames'):>5} {n('engrad_natoms'):>6} "
            f"{n('grad_converged_normalopt'):>6} {n('out_converged_banner'):>6} "
            f"{n('archives'):>5} {n('has_gbw'):>5}  {dict(fb.most_common(6))}"
        )

    frames = sorted(int(r["trj_frames"]) for r in rows if r.get("trj_frames"))
    maxg = sorted(
        float(r["engrad_max_comp"])
        for r in rows
        if r.get("engrad_max_comp") is not None
    )
    if frames:
        print(
            f"\ntrajectory frames per job: min {frames[0]}, median {frames[len(frames) // 2]}, "
            f"max {frames[-1]} ({len(frames)} jobs)"
        )
    if maxg:
        print(
            f"final max gradient component (Eh/bohr): median {maxg[len(maxg) // 2]:.2e}, "
            f"max {maxg[-1]:.2e}; <= {TOL_MAXG:g}: {sum(g <= TOL_MAXG for g in maxg)}/{len(maxg)}"
        )
    mism = [
        r["rel_path"]
        for r in rows
        if r.get("xyz_natoms")
        and r.get("inp_natoms")
        and r["xyz_natoms"] != r["inp_natoms"]
    ]
    print(f"orca.xyz atom count != orca.inp atom count: {len(mism)}")
    for p in mism[:10]:
        print(f"    {p}")
    side_err = Counter(
        f"{k}: {r[k]}"
        for r in rows
        for k in r
        if k.endswith("_error") and r[k] and k != "inp_error"
    )
    if side_err:
        print("side-file / output read errors:")
        for e, c in side_err.most_common(8):
            print(f"  {c:>6}  {e}")
    print("\nFile-name patterns (dirs containing each, top 50):")
    pat = Counter(
        p
        for r in rows
        for p in {file_pattern(f) for f in str(r["files"]).split(";") if f}
    )
    for p, n in pat.most_common(50):
        print(f"  {n:>6}  {p}")

    print("\nDistinct '!' input lines per group (top 3):")
    for (cat, grp), rs in sorted(by_group.items()):
        print(f"  {cat}/{grp}")
        for line, n in Counter(str(r.get("inp_simple", "")) for r in rs).most_common(3):
            print(f"      {n:>5}  {line[:150]}")

    unparsed = [r["rel_path"] for r in rows if not r["name_parsed"]]
    print(f"\nJob names not matching {JOB_NAME_RE.pattern}: {len(unparsed)}")
    for p in unparsed[:15]:
        print(f"    {p}")

    errs = Counter(str(r.get("out_error", "")) for r in rows if r.get("out_error"))
    print("\nMost common output errors (non-terminated):")
    for e, n in errs.most_common(10):
        print(f"  {n:>6}  {e}")

    if args.list_archives:
        print("\nSampled archive contents:")
        for _, rs in sorted(by_group.items()):
            for r in [r for r in rs if r["archives"]][: args.list_archives]:
                arch = str(r["archives"]).split(";")[0]
                members = list_archive(root / str(r["rel_path"]) / arch)
                print(f"  {r['rel_path']}/{arch}: {len(members)} members")
                for mname in members[:25]:
                    print(f"      {mname}")

    print("\nOne full listing per group:")
    for _, rs in sorted(by_group.items()):
        r = rs[0]
        print(f"  {r['rel_path']}")
        print(f"      files  : {str(r['files'])[:600]}")
        if r["subdirs"]:
            print(f"      subdirs: {str(r['subdirs'])[:300]}")


if __name__ == "__main__":
    main()

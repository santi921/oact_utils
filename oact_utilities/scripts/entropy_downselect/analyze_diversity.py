"""Quick diversity analysis of entropy-selected structures."""

from __future__ import annotations

import lmdb
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from pathlib import Path

LMDB_PATH = "/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/v2_seed_downselect/selected_structures.lmdb"
META_PATH = "/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/v2_seed_downselect/selected_metadata.parquet"
OUT_PATH = "/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/v2_seed_downselect/diversity_report.txt"


def main() -> None:
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        length = pickle.loads(txn.get(b"length"))

    metals, charges, spins, natoms_list = [], [], [], []
    formulas, element_sets, n_unique_elements = [], [], []
    all_element_counter: Counter = Counter()

    with env.begin() as txn:
        for i in range(length):
            atoms = pickle.loads(txn.get(f"{i}".encode("ascii")))
            info = atoms.info
            metals.append(info.get("metal", "?"))
            charges.append(info.get("charge", 0))
            spins.append(info.get("spin", 0))
            natoms_list.append(len(atoms))
            syms = atoms.get_chemical_symbols()
            sym_set = frozenset(syms)
            element_sets.append(sym_set)
            n_unique_elements.append(len(set(syms)))
            formulas.append(atoms.get_chemical_formula("hill"))
            for e in sym_set:
                all_element_counter[e] += 1
            if i % 100000 == 0:
                print(f"  scanned {i:,}...", flush=True)
    env.close()
    print(f"  scanned {length:,} done", flush=True)

    lines = []
    def p(s=""):
        lines.append(s)

    p("=" * 60)
    p(f"DIVERSITY ANALYSIS: {length:,} entropy-selected structures")
    p("=" * 60)

    mc = Counter(metals)
    p("\n--- Metal distribution ---")
    for m, c in mc.most_common():
        p(f"  {m:4s}: {c:>7,} ({100*c/length:.1f}%)")
    p(f"  Unique metals: {len(mc)}")

    cc = Counter(charges)
    p("\n--- Charge distribution ---")
    for ch, c in sorted(cc.items()):
        p(f"  {ch:+d}: {c:>7,} ({100*c/length:.1f}%)")

    sc = Counter(spins)
    p("\n--- Spin multiplicity distribution ---")
    for s, c in sorted(sc.items()):
        p(f"  {s:2d}: {c:>7,} ({100*c/length:.1f}%)")

    na = np.array(natoms_list)
    p(f"\n--- Structure size (natoms) ---")
    p(f"  min={na.min()}, max={na.max()}, mean={na.mean():.1f}, median={np.median(na):.0f}")
    bins = [0, 10, 20, 30, 50, 75, 100, 150, 200, 500]
    hist, _ = np.histogram(na, bins=bins)
    for i in range(len(hist)):
        p(f"  {bins[i]:>3d}-{bins[i+1]:>3d}: {hist[i]:>7,} ({100*hist[i]/length:.1f}%)")

    unique_formulas = len(set(formulas))
    unique_element_combos = len(set(element_sets))
    nue = np.array(n_unique_elements)
    p(f"\n--- Compositional diversity ---")
    p(f"  Unique formulas: {unique_formulas:,}")
    p(f"  Unique element combinations: {unique_element_combos:,}")
    p(f"  Elements per structure: min={nue.min()}, max={nue.max()}, mean={nue.mean():.1f}")
    p(f"  Total unique elements: {len(all_element_counter)}")
    p(f"  Top 20 elements:")
    for e, c in all_element_counter.most_common(20):
        p(f"    {e:3s}: {c:>7,} ({100*c/length:.1f}%)")

    cs_pairs = Counter(zip(charges, spins))
    p(f"\n--- Charge x Spin combinations ---")
    p(f"  Unique (charge, spin) pairs: {len(cs_pairs)}")
    p(f"  Top 10:")
    for (ch, sp), c in cs_pairs.most_common(10):
        p(f"    charge={ch:+d}, spin={sp:2d}: {c:>7,} ({100*c/length:.1f}%)")

    mcs = Counter(zip(metals, charges, spins))
    p(f"\n--- Metal x Charge x Spin ---")
    p(f"  Unique (metal, charge, spin) triples: {len(mcs)}")

    meta = pd.read_parquet(META_PATH)
    dc = meta["is_distorted"].value_counts()
    p(f"\n--- Distorted vs equilibrium ---")
    for k, v in dc.items():
        label = "distorted" if k else "equilibrium"
        p(f"  {label}: {v:>7,} ({100*v/length:.1f}%)")

    report = "\n".join(lines)
    Path(OUT_PATH).write_text(report)
    print(report)


if __name__ == "__main__":
    main()

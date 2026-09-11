"""Split a workflow DB into fixed-size child workflow DBs.

Each child is a fully independent SQLite workflow database with the parent's
exact schema, so it can be submitted to its own allocation or handed to a
collaborator without touching the others.

Two choices worth knowing about:

1. Rows are assigned to chunks at random (seeded). The AFIR corpus was built by
   walking ``*.xyz`` in alphabetical order, so row order tracks template name,
   which tracks family and metal. A sequential split would put every MOBH1x
   system in chunk 0 and skew each chunk's element and cost distribution; a
   seeded shuffle makes the chunks comparable. Frames of one trajectory are
   therefore spread across chunks, which is harmless because every frame is an
   independent ORCA job.
2. ``orig_index`` is carried over unchanged rather than renumbered. It stays
   globally unique across all children, so ``--job-dir-pattern`` templates that
   embed ``{orig_index}`` cannot collide between chunks sharing a job root, and
   any row can be traced back to its parent.

Usage:
    python -m examples.chunk_afir_workflow                    # use defaults
    python examples/chunk_afir_workflow.py --chunk-size 15000
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from pathlib import Path

DEFAULT_DB = Path("/Users/santiagovargas/dev/oact_utils/data/afir_v1/afir_v1.db")
DEFAULT_OUT = Path("/Users/santiagovargas/dev/oact_utils/data/afir_v1")


def read_schema(conn: sqlite3.Connection) -> tuple[str, list[str], list[str]]:
    """Return the parent's table DDL, its index DDLs, and its column names.

    The DDL is taken verbatim from ``sqlite_master`` rather than rebuilt from
    ``PRAGMA table_info``, which would silently drop DEFAULT clauses such as
    ``fail_count INTEGER DEFAULT 0``.

    Args:
        conn: Open connection to the parent database.

    Returns:
        Tuple of (table DDL, list of index DDLs, column names excluding ``id``).

    Raises:
        ValueError: If the parent has no ``structures`` table.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='structures'"
    ).fetchone()
    if row is None:
        raise ValueError("Parent database has no 'structures' table")
    table_ddl = row[0]

    index_ddls = [
        r[0]
        for r in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='structures'"
        )
        if r[0]
    ]

    # id is AUTOINCREMENT and is reassigned per child; every other column copies.
    columns = [
        r[1] for r in conn.execute("PRAGMA table_info(structures)") if r[1] != "id"
    ]
    return table_ddl, index_ddls, columns


def assign_chunks(ids: list[int], chunk_size: int, seed: int) -> list[list[int]]:
    """Shuffle row ids and cut them into fixed-size groups.

    Args:
        ids: All parent row ids.
        chunk_size: Rows per chunk. The final chunk holds the remainder.
        seed: Seed for the shuffle, so the split is reproducible.

    Returns:
        List of id lists, one per chunk.
    """
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i : i + chunk_size] for i in range(0, len(shuffled), chunk_size)]


def write_chunk(
    src_db: Path,
    dst_db: Path,
    ids: list[int],
    table_ddl: str,
    index_ddls: list[str],
    columns: list[str],
) -> int:
    """Copy the given parent rows into a new child workflow DB.

    Args:
        src_db: Parent database path.
        dst_db: Child database path to create.
        ids: Parent row ids to copy.
        table_ddl: Verbatim CREATE TABLE statement from the parent.
        index_ddls: Verbatim CREATE INDEX statements from the parent.
        columns: Column names to copy (all but ``id``).

    Returns:
        Number of rows written.

    Raises:
        FileExistsError: If the child path already exists.
    """
    if dst_db.exists():
        raise FileExistsError(f"Refusing to overwrite existing chunk: {dst_db}")

    conn = sqlite3.connect(str(dst_db))
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=OFF")  # bulk load; FULL is restored below
        conn.execute(table_ddl)
        for ddl in index_ddls:
            conn.execute(ddl)

        conn.execute("CREATE TEMP TABLE pick (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO pick (id) VALUES (?)", ((i,) for i in ids))

        conn.execute("ATTACH DATABASE ? AS src", (str(src_db),))
        collist = ", ".join(columns)
        conn.execute(
            f"INSERT INTO structures ({collist}) "
            f"SELECT {collist} FROM src.structures "
            f"WHERE id IN (SELECT id FROM pick) ORDER BY orig_index"
        )
        conn.commit()
        conn.execute("DETACH DATABASE src")
        written = conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
        conn.execute("PRAGMA synchronous=FULL")
        return int(written)
    finally:
        conn.close()


def chunk_workflow(
    db_path: Path, out_dir: Path, chunk_size: int, seed: int
) -> list[Path]:
    """Split ``db_path`` into child DBs of ``chunk_size`` rows each.

    Args:
        db_path: Parent workflow database.
        out_dir: Directory to write child DBs into.
        chunk_size: Rows per chunk.
        seed: Seed for the shuffle.

    Returns:
        Paths of the child databases, in order.

    Raises:
        FileNotFoundError: If the parent database does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        table_ddl, index_ddls, columns = read_schema(conn)
        ids = [r[0] for r in conn.execute("SELECT id FROM structures ORDER BY id")]
    finally:
        conn.close()

    groups = assign_chunks(ids, chunk_size, seed)
    width = max(2, len(str(len(groups) - 1)))
    print(f"Parent rows: {len(ids)}  ->  {len(groups)} chunks of <= {chunk_size}")

    out_paths: list[Path] = []
    for n, group in enumerate(groups):
        dst = out_dir / f"{db_path.stem}_chunk{n:0{width}d}.db"
        written = write_chunk(db_path, dst, group, table_ddl, index_ddls, columns)
        size_mb = dst.stat().st_size / 1e6
        print(f"  {dst.name}: {written:>6} rows  {size_mb:>7.1f} MB")
        out_paths.append(dst)

    return out_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Parent DB.")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory."
    )
    parser.add_argument("--chunk-size", type=int, default=15000, help="Rows per chunk.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chunk_workflow(args.db, args.out_dir, args.chunk_size, args.seed)

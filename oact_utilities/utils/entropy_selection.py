"""Shared core for greedy entropy-maximizing subset selection (D-optimal design).

Pure numpy + lmdb machinery extracted from the ``entropy_downselect`` script so it
can be reused by both the plain selector and the position-optimizing variant. No
fairchem dependency lives here.

The selection maximizes ``log det(C)`` of the whitened feature covariance via batch
greedy: a vectorized full scan builds a top-K candidate pool, then exact greedy within
the pool selects ``batch_size`` points with Sherman-Morrison inverse updates.

``run_selection`` accepts an optional ``commit_hook`` that lets a caller replace the
feature committed for a selected candidate (e.g. with a position-optimized feature)
without changing the selection mechanics.
"""

from __future__ import annotations

import os
import pickle
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm

# Hook called after a candidate wins the inner greedy step. Receives
# (best_global, rank, C, C_inv, mu, n_current) and returns (feature_to_commit, record).
CommitHook = Callable[
    [int, int, np.ndarray, np.ndarray, np.ndarray, int],
    "tuple[np.ndarray, dict | None]",
]

# Per-batch optimizer. Receives (items=[(rank, global_index), ...], C_inv, mu, n_current)
# and returns {rank: feature_to_commit} evaluated against the frozen batch-start state.
BatchOptimizeFn = Callable[
    ["list[tuple[int, int]]", np.ndarray, np.ndarray, int],
    "dict[int, np.ndarray]",
]


def debug_log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [pid={os.getpid()}] {message}", flush=True)


def load_features(
    features_dir: str,
    limit: int | None = None,
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Load and concatenate all *_features.npy files into a contiguous array.

    Returns:
        X: (N, D) float32 array
        file_boundaries: list of (stem, start_idx, count) tuples
    """
    feat_dir = Path(features_dir)
    npy_files = sorted(feat_dir.glob("*_features.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No *_features.npy files in {feat_dir}")

    shapes = []
    for f in npy_files:
        mmap = np.load(str(f), mmap_mode="r")
        shapes.append(mmap.shape)
        debug_log(f"  {f.name}: {mmap.shape}")

    D = shapes[0][1]
    total_rows = sum(s[0] for s in shapes)
    if limit is not None:
        total_rows = min(total_rows, limit)
    debug_log(f"Total: {total_rows:,} structures, {D} features")

    X = np.empty((total_rows, D), dtype=np.float32)
    file_boundaries: list[tuple[str, int, int]] = []
    offset = 0
    remaining = total_rows

    for f, shape in tqdm(
        zip(npy_files, shapes),
        desc="Loading features",
        total=len(npy_files),
    ):
        if remaining <= 0:
            break
        n = min(shape[0], remaining)
        stem = f.name.replace("_features.npy", "")
        mmap = np.load(str(f), mmap_mode="r")
        X[offset : offset + n] = mmap[:n]
        file_boundaries.append((stem, offset, n))
        offset += n
        remaining -= n

    nan_count = np.isnan(X).any(axis=1).sum()
    if nan_count:
        debug_log(f"WARNING: {nan_count} rows contain NaN values")

    return X, file_boundaries


def whiten(
    X: np.ndarray,
    ref: np.ndarray | None = None,
    reg: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten X in-place. Statistics computed from ref if provided, else from X.

    Returns:
        mean: (D,) float64 mean used for centering
        whitening_matrix: (D, D) float64
    """
    N, D = X.shape
    chunk = 1_000_000

    if ref is not None:
        debug_log(
            f"Computing whitening transform from reference ({ref.shape[0]:,} points)..."
        )
        ref_f64 = ref.astype(np.float64)
        mean = ref_f64.mean(axis=0)
        centered = ref_f64 - mean
        cov = centered.T @ centered / (ref.shape[0] - 1)
        cov += reg * np.eye(D, dtype=np.float64)
    else:
        debug_log("Computing population mean...")
        mean = np.zeros(D, dtype=np.float64)
        for i in range(0, N, chunk):
            mean += X[i : i + chunk].astype(np.float64).sum(axis=0)
        mean /= N

        debug_log("Computing population covariance...")
        mean_f32 = mean.astype(np.float32)
        cov = np.zeros((D, D), dtype=np.float64)
        for i in range(0, N, chunk):
            Xc = (X[i : i + chunk] - mean_f32).astype(np.float64)
            cov += Xc.T @ Xc
        cov /= N
        cov += reg * np.eye(D, dtype=np.float64)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    debug_log("Applying whitening transform to X...")
    mean_f32 = mean.astype(np.float32)
    W_f32 = W.astype(np.float32)
    for i in tqdm(range(0, N, chunk), desc="Whitening", total=(N + chunk - 1) // chunk):
        X[i : i + chunk] -= mean_f32
        X[i : i + chunk] = X[i : i + chunk] @ W_f32

    debug_log(
        f"Whitening done. Eigenvalue range: [{eigvals.min():.2e}, {eigvals.max():.2e}]"
    )
    return mean, W


def init_seed_from_external(
    X_seed: np.ndarray,
    reg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Initialize covariance from external seed features (already whitened).

    Returns:
        C: (D, D) float64 covariance matrix
        C_inv: (D, D) float64 inverse covariance
        mu: (D,) float64 mean of seed
        n_current: number of seed points
    """
    n, D = X_seed.shape
    X_f64 = X_seed.astype(np.float64)
    mu = X_f64.mean(axis=0)
    centered = X_f64 - mu
    C = centered.T @ centered / n + reg * np.eye(D, dtype=np.float64)
    C_inv = np.linalg.inv(C)

    log_det = float(np.linalg.slogdet(C)[1])
    debug_log(f"External seed: {n} points, initial log-det: {log_det:.4f}")
    return C, C_inv, mu, n


def compute_all_scores(
    X: np.ndarray,
    C_inv: np.ndarray,
    mu: np.ndarray,
    n_current: int,
    selected: np.ndarray,
    buf_Xc: np.ndarray | None = None,
    buf_Q: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized full scan: compute delta_log_det for all candidates.

    Pre-allocated buffers (buf_Xc, buf_Q) eliminate per-chunk memory allocation.

    Returns:
        scores: (N,) float64 -- delta_log_det for each candidate, -inf for selected
    """
    N, D = X.shape
    chunk = 2_000_000
    scores = np.full(N, -np.inf, dtype=np.float64)

    C_inv_f32 = C_inv.astype(np.float32)
    mu_f32 = mu.astype(np.float32)
    const_term = D * np.log(n_current / (n_current + 1))

    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        n = end - i
        mask = ~selected[i:end]
        if not mask.any():
            continue

        if buf_Xc is not None and buf_Q is not None:
            np.subtract(X[i:end], mu_f32, out=buf_Xc[:n])
            np.dot(buf_Xc[:n], C_inv_f32, out=buf_Q[:n])
            quads = np.einsum("ij,ij->i", buf_Q[:n], buf_Xc[:n])
        else:
            X_c = X[i:end] - mu_f32
            Q = X_c @ C_inv_f32
            quads = np.einsum("ij,ij->i", Q, X_c)

        chunk_scores = const_term + np.log1p(quads.astype(np.float64) / (n_current + 1))
        scores[i:end] = np.where(mask, chunk_scores, -np.inf)

    return scores


def compute_single_score(
    x: np.ndarray,
    C_inv: np.ndarray,
    mu: np.ndarray,
    n_current: int,
) -> float:
    """Compute delta_log_det for a single candidate."""
    D = len(mu)
    delta = x.astype(np.float64) - mu
    quad = delta @ C_inv @ delta
    return float(
        D * np.log(n_current / (n_current + 1)) + np.log1p(quad / (n_current + 1))
    )


def update_state(
    x: np.ndarray,
    C: np.ndarray,
    C_inv: np.ndarray,
    mu: np.ndarray,
    n_current: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Rank-1 covariance update with Sherman-Morrison inverse update.

    Returns:
        C_new, C_inv_new, mu_new, n_new, delta_log_det
    """
    D = len(mu)
    x64 = x.astype(np.float64)
    delta = x64 - mu
    n_new = n_current + 1
    scale = n_current / n_new
    alpha = n_current / (n_new**2)

    mu_new = mu + delta / n_new

    C_new = scale * C + alpha * np.outer(delta, delta)

    v = C_inv @ delta
    quad = delta @ v
    denom = scale + alpha * quad
    C_inv_new = (C_inv - (alpha / denom) * np.outer(v, v)) / scale

    delta_log_det = float(D * np.log(scale) + np.log1p(quad / n_new))

    return C_new, C_inv_new, mu_new, n_new, delta_log_det


def save_checkpoint(
    path: Path,
    selected_indices: list[int],
    C: np.ndarray,
    C_inv: np.ndarray,
    mu: np.ndarray,
    n_current: int,
    step: int,
    delta_log_dets: list[float],
    log_dets: list[float],
    seed_indices: np.ndarray,
    random_seed: int,
) -> None:
    """Atomic checkpoint write (write to .tmp then rename)."""
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez(
        str(tmp_path),
        selected_indices=np.array(selected_indices, dtype=np.int64),
        C=C,
        C_inv=C_inv,
        mu=mu,
        n_current=np.int64(n_current),
        step=np.int64(step),
        delta_log_dets=np.array(delta_log_dets, dtype=np.float64),
        log_dets=np.array(log_dets, dtype=np.float64),
        seed_indices=seed_indices,
        random_seed=np.int64(random_seed),
    )
    tmp_path.rename(path)


def load_checkpoint(path: Path) -> dict:
    """Load checkpoint from NPZ file."""
    data = np.load(str(path), allow_pickle=False)
    return {
        "selected_indices": data["selected_indices"].tolist(),
        "C": data["C"],
        "C_inv": data["C_inv"],
        "mu": data["mu"],
        "n_current": int(data["n_current"]),
        "step": int(data["step"]),
        "delta_log_dets": data["delta_log_dets"].tolist(),
        "log_dets": data["log_dets"].tolist(),
        "seed_indices": data["seed_indices"],
        "random_seed": int(data["random_seed"]),
    }


def run_selection(
    X: np.ndarray,
    n_select: int,
    seed_features: np.ndarray,
    reg: float,
    batch_size: int,
    pool_factor: int,
    checkpoint_every: int,
    output_dir: Path,
    resume: bool,
    random_seed: int,
    commit_hook: CommitHook | None = None,
    batch_optimize_fn: BatchOptimizeFn | None = None,
) -> dict:
    """Batch greedy selection: full scan -> top-K pool -> exact greedy within pool.

    Each batch does one vectorized full scan of all N candidates (~5-10s),
    then selects batch_size points via exact greedy within a filtered pool
    of the top candidates (< 1s). This avoids the heap-based lazy greedy
    which degrades badly when marginal gains are similar across candidates.

    Two optional hooks customize what feature is committed for selected candidates:

    - ``commit_hook(best_global, rank, C, C_inv, mu, n_current) -> (feature, record)``
      is called once *per pick* (sequential); ``feature`` (whitened, like ``X``) is
      committed in place of ``X[best_global]``. ``record`` is ignored (hooks persist
      their own output).
    - ``batch_optimize_fn(items, C_inv, mu, n_current) -> {rank: feature}`` is called
      once *per batch*, where ``items`` is the list of ``(rank, global_index)`` selected
      this batch (chosen by a working-copy greedy on the original features). It returns
      the optimized feature to commit for each rank, evaluated against the frozen
      batch-start covariance. This enables batched/parallel optimization.

    At most one of the two hooks should be set. Both may expose ``on_checkpoint(step)``,
    called at each checkpoint so the hook can flush streamed outputs in lockstep. Without
    either hook the original feature is committed (default behavior).
    """
    N, D = X.shape
    chunk = 2_000_000
    checkpoint_path = output_dir / "checkpoint.npz"

    if resume and checkpoint_path.exists():
        debug_log("Resuming from checkpoint...")
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt["random_seed"] != random_seed:
            raise ValueError(
                f"Checkpoint random_seed={ckpt['random_seed']} != {random_seed}"
            )
        selected_indices: list[int] = ckpt["selected_indices"]
        C = ckpt["C"]
        C_inv = ckpt["C_inv"]
        mu = ckpt["mu"]
        n_current = ckpt["n_current"]
        start_step = ckpt["step"]
        delta_log_dets: list[float] = ckpt["delta_log_dets"]
        log_dets: list[float] = ckpt["log_dets"]
        seed_indices = ckpt["seed_indices"]

        selected = np.zeros(N, dtype=bool)
        selected[seed_indices] = True
        for idx in selected_indices:
            selected[idx] = True

        debug_log(
            f"Resumed at step {start_step}, "
            f"{len(selected_indices)} already selected, "
            f"log-det={log_dets[-1]:.4f}"
        )
    else:
        C, C_inv, mu, n_current = init_seed_from_external(seed_features, reg)
        seed_indices = np.array([], dtype=np.int64)
        selected = np.zeros(N, dtype=bool)
        selected_indices = []
        start_step = 0
        delta_log_dets = []
        log_dets = [float(np.linalg.slogdet(C)[1])]

    buf_Xc = np.empty((chunk, D), dtype=np.float32)
    buf_Q = np.empty((chunk, D), dtype=np.float32)

    t0 = time.time()
    pbar = tqdm(total=n_select, initial=start_step, desc="Selecting")
    step = start_step

    def _do_checkpoint() -> None:
        hook = commit_hook if commit_hook is not None else batch_optimize_fn
        if hook is not None and hasattr(hook, "on_checkpoint"):
            hook.on_checkpoint(step)
        save_checkpoint(
            checkpoint_path,
            selected_indices,
            C,
            C_inv,
            mu,
            n_current,
            step,
            delta_log_dets,
            log_dets,
            seed_indices,
            random_seed,
        )
        debug_log(f"Checkpoint at step {step}")

    while step < n_select:
        step_before = step
        batch_n = min(batch_size, n_select - step)
        pool_size = min(batch_n * pool_factor, int((~selected).sum()))

        C_inv = np.linalg.inv(C)

        t_scan = time.time()
        scores = compute_all_scores(
            X,
            C_inv,
            mu,
            n_current,
            selected,
            buf_Xc,
            buf_Q,
        )
        scan_time = time.time() - t_scan

        top_k = np.argpartition(-scores, pool_size)[:pool_size]
        top_k = top_k[np.argsort(-scores[top_k])]
        pool_global = top_k.copy()
        pool_X = X[pool_global].astype(np.float64)
        pool_alive = np.ones(len(pool_global), dtype=bool)

        debug_log(
            f"Step {step}: scan {scan_time:.1f}s, "
            f"pool={pool_size}, batch={batch_n}, "
            f"log-det={log_dets[-1]:.4f}, "
            f"top-score={scores[top_k[0]]:.6f}"
        )

        if batch_optimize_fn is None:
            for _j in range(batch_n):
                if not pool_alive.any():
                    debug_log(f"Pool exhausted at step {step}")
                    break

                alive_idx = np.where(pool_alive)[0]
                deltas = pool_X[alive_idx] - mu
                quads = (deltas @ C_inv * deltas).sum(axis=1)

                best_pool = alive_idx[np.argmax(quads)]
                best_global = int(pool_global[best_pool])

                pool_alive[best_pool] = False
                selected[best_global] = True
                selected_indices.append(best_global)

                if commit_hook is not None:
                    rank = len(selected_indices) - 1
                    feat, _record = commit_hook(
                        best_global, rank, C, C_inv, mu, n_current
                    )
                else:
                    feat = X[best_global]

                C, C_inv, mu, n_current, dldet = update_state(
                    feat, C, C_inv, mu, n_current
                )
                delta_log_dets.append(dldet)
                log_dets.append(log_dets[-1] + dldet)
                step += 1
                pbar.update(1)

                if step % 1000 == 0:
                    pbar.set_postfix(
                        dlogdet=f"{dldet:.6f}", logdet=f"{log_dets[-1]:.2f}"
                    )

                if checkpoint_every > 0 and step % checkpoint_every == 0:
                    _do_checkpoint()
        else:
            # Batched optimization. Select the batch on a *working* covariance advanced
            # with the original features (for intra-batch diversity), keeping the real
            # covariance frozen; optimize the whole batch against that frozen batch-start
            # state; then commit the optimized features to the real covariance.
            Cw, Cw_inv, muw, nw = C, C_inv, mu, n_current
            batch_items: list[tuple[int, int]] = []
            for _j in range(batch_n):
                if not pool_alive.any():
                    debug_log(f"Pool exhausted at step {step}")
                    break
                alive_idx = np.where(pool_alive)[0]
                deltas = pool_X[alive_idx] - muw
                quads = (deltas @ Cw_inv * deltas).sum(axis=1)
                best_pool = alive_idx[np.argmax(quads)]
                best_global = int(pool_global[best_pool])
                pool_alive[best_pool] = False
                selected[best_global] = True
                selected_indices.append(best_global)
                batch_items.append((len(selected_indices) - 1, best_global))
                Cw, Cw_inv, muw, nw, _ = update_state(
                    X[best_global], Cw, Cw_inv, muw, nw
                )

            opt_feats = batch_optimize_fn(batch_items, C_inv, mu, n_current)
            dldet = 0.0
            for rank, _gidx in batch_items:
                C, C_inv, mu, n_current, dldet = update_state(
                    opt_feats[rank], C, C_inv, mu, n_current
                )
                delta_log_dets.append(dldet)
                log_dets.append(log_dets[-1] + dldet)
                step += 1
                pbar.update(1)

            if batch_items:
                pbar.set_postfix(dlogdet=f"{dldet:.6f}", logdet=f"{log_dets[-1]:.2f}")
            if checkpoint_every > 0 and (
                step // checkpoint_every != step_before // checkpoint_every
            ):
                _do_checkpoint()

    pbar.close()
    elapsed = time.time() - t0
    debug_log(
        f"Selection done: {len(selected_indices):,} structures in {elapsed:.1f}s "
        f"({len(selected_indices) / max(elapsed, 0.01):.0f} sel/s)"
    )

    return {
        "selected_indices": selected_indices,
        "delta_log_dets": delta_log_dets,
        "log_dets": log_dets,
        "seed_indices": seed_indices,
    }


def write_atoms_lmdb(
    atoms_by_rank: dict[int, object],
    output_path: Path,
    n_entries: int,
    chunk_size: int = 10_000,
) -> int:
    """Write rank-keyed Atoms objects to an LMDB (key "0", "1", ... + "length").

    Also writes a sibling ``metadata.npz`` (natoms array) next to the LMDB. Used by
    both the source-reading ``build_selected_lmdb`` and the position-optimizing
    selector (which supplies optimized Atoms directly).

    Returns:
        Number of structures written.
    """
    debug_log(f"Writing {n_entries:,} structures to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    lock_path = Path(str(output_path) + "-lock")
    lock_path.unlink(missing_ok=True)

    map_size = 50 * (1 << 30)
    env = lmdb.open(
        str(output_path), map_size=map_size, subdir=False, meminit=False, map_async=True
    )
    natoms_list = []

    for chunk_start in tqdm(
        range(0, n_entries, chunk_size),
        desc="Writing LMDB",
    ):
        chunk_end = min(chunk_start + chunk_size, n_entries)
        txn = env.begin(write=True)
        for rank in range(chunk_start, chunk_end):
            atoms = atoms_by_rank[rank]
            key = f"{rank}".encode("ascii")
            txn.put(key, pickle.dumps(atoms, protocol=-1))
            natoms_list.append(len(atoms))
        txn.commit()

    txn = env.begin(write=True)
    txn.put(b"length", pickle.dumps(n_entries, protocol=-1))
    txn.commit()
    env.sync()
    env.close()

    metadata_path = output_path.parent / "metadata.npz"
    np.savez(str(metadata_path), natoms=np.array(natoms_list, dtype=np.int32))
    debug_log(f"Written {n_entries:,} structures + metadata.npz")
    return n_entries


def build_selected_lmdb(
    selected_indices: np.ndarray,
    file_boundaries: list[tuple[str, int, int]],
    lmdb_dir: str,
    output_path: Path,
    chunk_size: int = 10_000,
) -> int:
    """Build an LMDB of selected structures in rank order.

    Reads Atoms objects from source LMDBs (grouped by file to minimize I/O),
    then writes them to a single output LMDB where key "0" is the most
    informative structure, key "1" the second, etc.

    Args:
        selected_indices: Global indices in selection rank order.
        file_boundaries: (stem, start_idx, count) tuples from load_features.
        lmdb_dir: Parent directory containing <stem>/data.lmdb files.
        output_path: Path for output .lmdb file.
        chunk_size: Entries per LMDB write transaction.

    Returns:
        Number of structures written.
    """
    lmdb_base = Path(lmdb_dir)
    n_selected = len(selected_indices)

    resolve = make_index_resolver(file_boundaries)

    debug_log(f"Resolving {n_selected:,} indices to source files...")
    by_source: dict[str, list[tuple[int, int]]] = {}
    for rank, gidx in enumerate(tqdm(selected_indices, desc="Mapping indices")):
        stem, local_idx = resolve(int(gidx))
        by_source.setdefault(stem, []).append((rank, local_idx))

    debug_log(f"Reading Atoms from {len(by_source)} source LMDBs...")
    atoms_by_rank: dict[int, object] = {}
    for stem, rank_local_pairs in tqdm(by_source.items(), desc="Reading LMDBs"):
        src_path = str(lmdb_base / stem / "data.lmdb")
        env = lmdb.open(src_path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            for rank, local_idx in rank_local_pairs:
                raw = txn.get(f"{local_idx}".encode("ascii"))
                atoms = pickle.loads(raw)
                atoms.info["selection_rank"] = rank
                atoms_by_rank[rank] = atoms
        env.close()

    return write_atoms_lmdb(atoms_by_rank, output_path, n_selected, chunk_size)


def make_index_resolver(
    file_boundaries: list[tuple[str, int, int]],
) -> Callable[[int], tuple[str, int]]:
    """Build a global_index -> (stem, local_idx) resolver from file boundaries."""
    boundary_lookup = []
    for stem, start, count in file_boundaries:
        boundary_lookup.append((start, start + count, stem))
    boundary_starts = np.array([b[0] for b in boundary_lookup])

    def _resolve(global_idx: int) -> tuple[str, int]:
        pos = int(np.searchsorted(boundary_starts, global_idx, side="right")) - 1
        start, _end, stem = boundary_lookup[pos]
        return stem, global_idx - start

    return _resolve

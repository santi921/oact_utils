# Corpus folder provenance

Maps each job-directory folder in `data/v4_model_dev/classifier_geom.db` back to the
workflow DB it was submitted from. The collaborator who ran these pushes did not label
the folders, so the mapping was reconstructed. Machine-readable version:
[corpus_folder_provenance.csv](corpus_folder_provenance.csv).

## Method

Job directory names encode the **source DB's** `orig_index`:

```
{hostname}_[{job_prefix}_]{formula}_q{charge}_m{spin}_idx{orig_index}
```

Match key is `(orig_index, charge, spin, formula)` parsed from `job_id`, compared against
`(orig_index, charge, spin, _format_formula(elements))` in each candidate DB. Scanned all
15 `examples/4_06` DBs, all 25 `v2_downselect` chunks, all 5 `optimized_batch_2` DBs, and
for the one folder those missed, every `.db` in the repo.

Do **not** use `classifier_geom.db.orig_index` for this. That column was re-indexed
globally across the corpus and does not correspond to any source DB.

## Mapping

| host | folder | jobs | source DB | matched | runner-up |
| --- | --- | --- | --- | --- | --- |
| barfoot | 20260722-actinide-push | 13564 | `v2_downselect/selected_v2_actinides_max100_chunk01.db` | 13564 | 0 |
| barfoot | 20260812-actinide-push | 13447 | `v2_downselect/selected_v2_actinides_max100_chunk13.db` | 13447 | 0 |
| barfoot | 20260828-actinide-push | 13360 | `optimized_batch_2/optimized_b2_actinides_max110_chunk01.db` | 13360 | 0 |
| barfoot | wave1 | 36122 | `examples/4_06/actinides_dod_47106_chunk1.db` | 36122 | 1 |
| carpenter | 08052026-nonactinide-push | 5804 | `oact_dbs/pull_04_27_26/ritwik_2026-02-21_09-49-34_actinides_dod.db` | 5804 | 170 |
| carpenter | 08122026-nonactinide-push | 14414 | `v2_downselect/selected_v2_non_actinides_max100_chunk01.db` | 14414 | 0 |
| carpenter | 20260708-actinide-push | - | **unresolved** | - | - |
| carpenter | 20260722-actinide-transfer | 13554 | `v2_downselect/selected_v2_actinides_max100_chunk11.db` | 13554 | 0 |
| carpenter | 20260828-actinide-push | 3713 | `optimized_batch_2/optimized_b2_actinides_max110_chunk02.db` | 3713 | 0 |
| carpenter | wave1_carpenter_act | 31400 | `examples/4_06/actinides_dod_47106_chunk0.db` | 31400 | 1 |
| raider | 20260722-actinide-push | 12019 | `v2_downselect/selected_v2_actinides_max100_chunk03.db` | 12019 | 0 |
| raider | 20260724-actinide-push | 3593 | `v2_downselect/selected_v2_actinides_max100_chunk17.db` | 3593 | 0 |
| raider | 20260812-nonactinide-push | 14345 | `v2_downselect/selected_v2_non_actinides_max100_chunk03.db` | 14345 | 0 |
| raider | 20260828-actinide-push | 12055 | `optimized_batch_2/optimized_b2_actinides_max110_chunk00.db` | 12055 | 0 |

Every folder matches its source at 100% of jobs. Runner-up is the best competing DB.

## Gotchas

- **`08052026-nonactinide-push` is mislabeled.** All 5804 jobs are actinide (Cf, Th, Pu,
  Ac) and come from an `actinides_dod` DB. It is also the only folder still using the
  legacy `{hostname}_job_{orig_index}` pattern, so a formula-based parser skips it
  entirely.
- **Two folders carry a `--job-prefix`** (`barfoot_20260820_...`,
  `carpenter_20260821_...`). A naive `idx` regex folds the prefix into the formula field
  and the match silently drops to zero. The prefix group must be optional.
- **`examples/4_06` and `ritwik_*` DBs carry the dropped-metal bug**
  (`parse_xyz_elements`): source `F;F;F;F` / natoms 4 vs corpus `Cf;F;F;F;F` / natoms 5.
  Those rows match on ligand composition but not on the full element string. Fall back to
  `idx+charge+spin` there.
- **Every `v2_downselect` chunk is exactly duplicated** - 30000 rows / 15000 distinct
  `orig_index`, each seen exactly twice with identical geometry. Effective size is half
  the row count. `optimized_batch_2` DBs are clean.
- **`20260708-actinide-push` is absent from the corpus** because its jobs are still
  `.tar.gz` archives under `carpenter/20260708-actinide-push/archives/jobs` and were
  deliberately deferred during the job-root merge. It needs an extraction-aware pass
  before it can be traced.
- `chunk11` was not an expected candidate but is the unambiguous source for
  `20260722-actinide-transfer` (13554/13554 including geometry).

## Verification

Geometry hashes corroborate the `optimized_batch_2` folders at ~99% on a 400-row sample.
Older folders diverge on geometry because `classifier_geom.db` stores post-calculation
coordinates while the source DBs hold inputs; those rest on the composite key, which is
still unambiguous (runner-up 0 or 1).

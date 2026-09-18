# Corpus folder provenance

Maps every job-directory group in `data/v4_model_dev/classifier_geom.db` (465,971 rows,
24 folders) back to the workflow DB it was submitted from. The collaborator who ran these
pushes did not label the folders, so the mapping was reconstructed. Machine-readable
version: [corpus_folder_provenance.csv](corpus_folder_provenance.csv).

## Method

Job directory names encode the **source DB's** `orig_index`, in one of two patterns:

```
{hostname}_[{job_prefix}_]{formula}_q{charge}_m{spin}_idx{orig_index}   # current
{hostname}_job_{orig_index}                                             # legacy
```

Match key is `(orig_index, charge, spin, formula)` parsed from `job_id`, compared against
`(orig_index, charge, spin, _format_formula(elements))` in each candidate DB, falling back
to `(orig_index, charge, spin)` where the source DB carries the dropped-metal bug. Scanned
all 180 DBs in the repo. All 37 groups resolve at 100% of jobs.

Two things to get right:

- **Do not use `classifier_geom.db.orig_index`.** It is a global re-index across the
  corpus and corresponds to no source DB.
- **The row key is `(folder, subroot)`, not `folder`.** Several folders span more than one
  source DB: `act_531` spans 7 downselect chunks, `nonact_531` and `act_4_06` and
  `entropy_grad` span 2 each, and `20260828-actinide-push` spans 3 HPC hosts. `subroot` is
  the path component under `job_path` that distinguishes them - the chunk directory for
  the parsl campaigns, the HPC hostname for the pushes.

## Mapping

| label | class | folder | subroot | jobs | source DB | runner-up |
| --- | --- | --- | --- | --- | --- | --- |
| `4_06_dod_chunk1_act` | act | wave1 | barfoot | 36,122 | `actinides_dod_47106_chunk1.db` | 611 |
| `4_06_dod_chunk0_act` | act | wave1_carpenter_act | carpenter | 31,400 | `actinides_dod_47106_chunk0.db` | 602 |
| `4_06_tuo_nonact` | nonact | nonact_4_06 | nonact_4_06 | 29,458 | `non_actinides_tuo_33744.db` | 29,458 |
| `4_06_tuo_chunk1_act` | act | act_4_06 | act_4_06_chunk_1 | 23,629 | `actinides_tuo_39829_chunk1.db` | 23,629 |
| `2_21_dod_act [ritwik_226]` | act | ritwik_226 | ritwik_226 | 18,988 | `santi_results_compiled_2026-02-21_09-49-34_actinides_dod.db` | 18,988 |
| `2_26_santi_act` | act | act_226_santi | act_226_santi | 18,669 | `new_2026-02-26_actinides_tuo_santi.db` | 18,669 |
| `2_26_michael_act` | act | act_226_michael | act_226_michael | 17,368 | `new_2026-02-26_actinides_tuo_michael.db` | 17,368 |
| `5_31_chunk01_nonact` | nonact | 08122026-nonactinide-push | carpenter | 14,414 | `selected_v2_non_actinides_max100_chunk01.db` | 99 |
| `5_31_chunk03_nonact` | nonact | 20260812-nonactinide-push | raider | 14,345 | `selected_v2_non_actinides_max100_chunk03.db` | 79 |
| `4_06_tuo_chunk0_act` | act | act_4_06 | act_4_06_chunk_0 | 14,193 | `actinides_tuo_39829_chunk0.db` | 14,193 |
| `5_31_chunk01_act` | act | 20260722-actinide-push | barfoot | 13,564 | `selected_v2_actinides_max100_chunk01.db` | 396 |
| `5_31_chunk11_act` | act | 20260722-actinide-transfer | carpenter | 13,554 | `selected_v2_actinides_max100_chunk11.db` | 44 |
| `5_31_chunk13_act` | act | 20260812-actinide-push | barfoot | 13,447 | `selected_v2_actinides_max100_chunk13.db` | 50 |
| `20260828_push_barfoot` | act | 20260828-actinide-push | barfoot | 13,360 | `optimized_b2_actinides_max110_chunk01.db` | 306 |
| `5_31_chunk05_nonact` | nonact | nonact_531 | nonact_531_chunk05 | 12,176 | `selected_v2_non_actinides_max100_chunk05.db` | 100 |
| `20260828_push_raider` | act | 20260828-actinide-push | raider | 12,055 | `optimized_b2_actinides_max110_chunk00.db` | 302 |
| `5_31_chunk03_act` | act | 20260722-actinide-push | raider | 12,019 | `selected_v2_actinides_max100_chunk03.db` | 57 |
| `5_31_chunk00_nonact` | nonact | nonact_531 | nonact_531_chunk0 | 11,815 | `selected_v2_non_actinides_max100_chunk00.db` | 405 |
| `4_06_dod_chunk4_act` | act | act_406 | act_406_chunk4 | 11,434 | `actinides_dod_47106_chunk4.db` | 11,434 |
| `afir_v1_chunk02` | act | afir_v1 | afir_v1_02 | 11,373 | `afir_v1_chunk02.db` | 11,373 |
| `5_31_chunk02_act` | act | act_531 | act_531_chunk02 | 10,889 | `selected_v2_actinides_max100_chunk02.db` | 10,889 |
| `5_31_chunk07_act` | act | act_531 | act_531_chunk07 | 10,620 | `selected_v2_actinides_max100_chunk07.db` | 10,620 |
| `5_31_chunk12_act` | act | act_531 | act_531_chunk12 | 10,516 | `selected_v2_actinides_max100_chunk12.db` | 40 |
| `5_31_chunk15_act` | act | act_531 | act_531_chunk15 | 10,102 | `selected_v2_actinides_max100_chunk15.db` | 191 |
| `5_31_chunk16_act` | act | act_531 | act_531_chunk16 | 10,021 | `selected_v2_actinides_max100_chunk16.db` | 317 |
| `2_21_tuo_act` | act | act_222_santi | act_222_santi | 9,841 | `santi_results_compiled_2026-02-21_09-49-34_actinides_tuo.db` | 9,841 |
| `entropy_0804_chunk0` | mixed | entropy_grad | entropy_grad_0 | 9,085 | `optimized_structures_entropy_0804_ishan_chunk0.db` | 9,085 |
| `5_31_chunk14_act` | act | act_531 | act_531_chunk14 | 9,045 | `selected_v2_actinides_max100_chunk14.db` | 35 |
| `2_21_tuo_nonact` | nonact | nonact_222_santi | nonact_222_santi | 7,811 | `santi_results_compiled_2026-02-21_09-49-34_non_actinides_tuo.db` | 7,811 |
| `entropy_0804_chunk1` | mixed | entropy_grad | entropy_grad_1 | 7,435 | `optimized_structures_entropy_0804_ishan_chunk1.db` | 7,435 |
| `5_31_chunk06_nonact` | nonact | nonact_531 | nonact_531_chunk06 | 7,357 | `selected_v2_non_actinides_max100_chunk06.db` | 7,357 |
| `5_31_chunk10_act` | act | act_531 | act_531_chunk10 | 6,203 | `selected_v2_actinides_max100_chunk10.db` | 21 |
| `2_21_dod_act [carpenter]` | act | 08052026-nonactinide-push | carpenter | 5,804 | `santi_results_compiled_2026-02-21_09-49-34_actinides_dod.db` | 5,804 |
| `20260828_push_carpenter` | act | 20260828-actinide-push | carpenter | 3,713 | `optimized_b2_actinides_max110_chunk02.db` | 86 |
| `5_31_chunk17_act` | act | 20260724-actinide-push | raider | 3,593 | `selected_v2_actinides_max100_chunk17.db` | 109 |
| `2_26_michael_nonact` | nonact | nonact_226_michael | nonact_226_michael | 495 | `new_2026-02-26_non_actinides_tuo_michael.db` | 495 |
| `2_26_santi_nonact` | nonact | nonact_226_santi | nonact_226_santi | 58 | `new_2026-02-26_non_actinides_tuo_santi.db` | 58 |

`runner-up` is the best competing DB's match count; the winner matches 100% in every row.
`class` is the actinide/non-actinide split of the rows that group supplied, taken from
`metal_class` rather than the DB filename.

## Gotchas

- **`08052026-nonactinide-push` is mislabeled.** All 5804 jobs are actinide (Cf, Th, Pu,
  Ac), from `..._actinides_dod.db`. It is also the only folder still on the legacy
  `job_{orig_index}` pattern, so a formula-based parser skips it entirely.
- **The two `entropy_0804` chunks are mixed provenance** (~53% actinide). Their filenames
  say neither, which is why `class` is derived from the data.
- **Two folders carry a `--job-prefix`** (`barfoot_20260820_...`, `carpenter_20260821_...`).
  A regex without an optional prefix group folds the date into the formula field and the
  match silently drops to zero.
- **`examples/4_06` and the 2026-02-21 `santi_results_compiled_*` DBs carry the
  dropped-metal bug** (`parse_xyz_elements`): source `F;F;F;F` / natoms 4 vs corpus
  `Cf;F;F;F;F` / natoms 5. Those rows match on ligand composition but not on the full
  element string.
- **Every `v2_downselect` chunk is exactly duplicated** - 30000 rows / 15000 distinct
  `orig_index`, each seen twice with identical geometry. Effective size is half the row
  count. `optimized_batch_2` DBs are clean.
- **The local `optimized_batch_2` DBs are all 100% `to_run`**, including the three
  actinide chunks that demonstrably ran (they supply 29,128 corpus rows). These are
  pre-submission snapshots that were never updated with results, so their `status` column
  says nothing about whether a chunk was run. Use the corpus match instead.
- **Neither `optimized_b2_non_actinides_max110` chunk appears in the corpus.** Best
  overlap with any group is 440/15,000 (3%), which is coincidental index collision. Those
  22,983 structures either were never submitted or landed in a root that was never
  ingested.
- **The same DB was used twice**: `santi_results_compiled_2026-02-21_09-49-34_actinides_dod.db`
  supplies both `ritwik_226` (18,988) and `08052026-nonactinide-push` (5,804), so its
  label is disambiguated by subroot.
- **`20260708-actinide-push` is absent from the corpus** because its jobs are still
  `.tar.gz` archives under `carpenter/20260708-actinide-push/archives/jobs` and were
  deliberately deferred during the job-root merge. It needs an extraction-aware pass
  before it can be traced.

## Verification

Geometry hashes corroborate the `optimized_batch_2` folders at ~99% on a 400-row sample.
Elsewhere the corpus stores post-calculation coordinates against source inputs, so
geometry disagrees by design and the composite key is the evidence.

## Consumers

`oact_utilities/notebooks/eda_v4_filtered_structures.ipynb` embeds this mapping as
`SOURCE_LABELS` / `SOURCE_CLASS` and keys its end-of-notebook plots on the short label
instead of the raw folder name.

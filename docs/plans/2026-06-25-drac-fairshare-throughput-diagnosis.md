# DRAC fairshare + throughput diagnosis

Living doc. Append a dated entry to the iteration log on each revisit.
Snapshot tables are VOLATILE (queue state moves hour to hour); fairshare
factors move slowly (RawUsage decays ~1-week half-life).

## Question

"No throughput on some DRAC clusters - am I over my fair share (pummeling it
too hard) and need to wait for it to recover?"

## Verdict (2026-06-25)

Fairshare is NOT the bottleneck. Ruled out by direct evidence:

- All four clusters sit at the SAME FairShare factor (~0.29-0.31). If fairshare
  caused the throughput gap, the good cluster would score high and the bad ones
  low. It is the inverse.
- The cluster that flows BEST (Nibi) has the LOWEST FairShare factor (0.289).
  The cluster among the stuck ones with the HIGHEST factor is Rorqual (0.312).
  Inverse correlation = fairshare is not the cause.
- Every cluster is 100% packed (0 idle Regular nodes at every walltime tier),
  so the whole campaign runs purely on backfill. The differentiator is
  backfill contention/turnover per cluster and per tier, which a fairshare
  score cannot capture.

Why you are "over target" but it does not matter: `def-yqw` is a DEFAULT
(opportunistic) account with a tiny target share (~0.14-0.30% of each cluster).
You go over target the moment you run any real campaign, so factor ~0.30 is the
normal floor, not a penalty box. Waiting decays you toward 0.5 only while idle
and snaps back as soon as you resume. Waiting is not a lever. Keep submitting.

## Per-cluster fairshare snapshot (2026-06-25)

Account `def-yqw_cpu`, user `santi921`. FairShare factor scale: 1.0 = allocation
untouched, 0.5 = exactly on target, ->0 = heavily over.

| Cluster | Node cores | Target share (NormShares) | Usage (NormUsage) | Over target | FairShare factor | Account LevelFS | Throughput (reported) |
| ------- | ---------- | ------------------------- | ----------------- | ----------- | ---------------- | --------------- | --------------------- |
| Narval  | 64         | 0.001898 (0.19%)          | 0.006447 (0.64%)  | 3.40x       | 0.296            | 0.294           | poor                  |
| Fir     | 192        | 0.001765 (0.18%)          | 0.005797 (0.58%)  | 3.28x       | 0.305            | 0.305           | poor                  |
| Rorqual | 192        | 0.002958 (0.30%)          | 0.010567 (1.06%)  | 3.57x       | 0.312            | 0.280           | poor                  |
| Nibi    | 192        | 0.001414 (0.14%)          | 0.004420 (0.44%)  | 3.13x       | 0.289            | 0.320           | GOOD                  |

`sprio` confirms the FAIRSHARE term is ~99.9% of job priority (AGE is
negligible: 217-875 against a fairshare term of ~1.44-1.56M), so fairshare is
the only priority lever - and it is roughly equal everywhere.

## Backfill contention snapshot (2026-06-24/25)

By-core queued jobs per walltime tier (the second number in partition-stats
`node:core`; our ORCA jobs are by-core / partial-node). 0 idle Regular nodes on
every cluster and tier.

| Cluster | 3h  | 12h | 24h  | 72h | 168h |
| ------- | --- | --- | ---- | --- | ---- |
| Narval  | 1   | 7   | 0    | 58  | 373  |
| Fir     | 72  | 574 | 1135 | 267 | 2467 |
| Rorqual | 408 | 62  | 1608 | 52  | 245  |
| Nibi    | 538 | 751 | 467  | 791 | 13   |

Least-contended by-core tier to aim each cluster's lane at:

| Cluster | Aim tier        | Note                                                        |
| ------- | --------------- | ----------------------------------------------------------- |
| Narval  | 3h or 12h       | both nearly empty (1, 7 queued) - pile in here              |
| Fir     | 3h (least-bad)  | genuinely contended; 12h+ is 574-2467 deep                  |
| Rorqual | 12h, not 3h     | 3h is the MOST crowded tier (408) - 12h (62) clears faster  |
| Nibi    | already flowing | non-shallow queue but high turnover; 168h tier is empty (13)|

Inference (label: inferred, not measured): Nibi flows despite the worst
fairshare and a non-shallow queue because gap-creation rate (turnover) on a
large packed cluster, not queue depth, governs opportunistic backfill. The
"always go shortest tier" rule is NOT universal - Rorqual's 3h tier is its most
congested. Pick the tier empirically per cluster.

## Levers that actually move throughput (in order)

1. Keep the queue deep with SHORT + NARROW by-core jobs. At factor 0.30 on a
   0-idle cluster, short walltime + small core count slot into the small gaps
   that open continuously as other jobs finish. Many narrow jobs beat few wide
   ones - a wide job needs many cores to free simultaneously, which rarely
   happens at 0 idle nodes.
2. Aim each cluster's lane at its least-contended tier (table above).
3. Do NOT chase the fairshare score, and do NOT idle to "recover" it.
4. Structural ceiling only: running under the sponsor's RAC (`rrg-`/`rpp-`)
   carries real priority but directly depresses their jobs - a sponsor
   conversation, not a wait. A separate `def-*` account only isolates the
   sponsor from our usage; it does not raise priority.

Correction to an earlier instinct: "massive cores for a short number of hours
to shove something in" is backwards. Short hours help backfill; massive cores
hurt it. Narrow + short wins on a packed cluster.

## Next step: chunk into atom/core-capped lanes

Goal: maximize effective throughput by routing molecules into disjoint lanes by
atom count, each with a core count and walltime tier tuned for backfill.

Prior measured completed wall time by size bucket (Rorqual chunk09, avg):

| Atoms  | Avg wall | Suggested cores/worker | Notes                              |
| ------ | -------- | ---------------------- | ---------------------------------- |
| <=20   | 0.8 h    | 8                      | comfortably <3h tier               |
| 21-30  | 1.4 h    | 8                      | <3h tier                           |
| 31-40  | 2.0 h    | 8-16                   | near 3h edge - watch the tail      |
| 41-50  | 2.8 h    | 16                     | 12h tier (3h too tight)            |
| 51-60  | 4.3 h    | 16                     | 12h tier                           |
| 61-80  | 5.6 h    | 32                     | 12-24h tier                        |
| 81-100 | 7.1 h    | 48                     | 24h tier; 81-100 RSS extrapolated  |

Existing 4-lane scheme (committed, Rorqual): <=40 = 24 workers x 8 cores;
41-60 = 12 x 16; 61-80 = 6 x 32; 81-100 = 4 x 48. Lanes run concurrently via
`--min-atoms`/`--max-atoms`. Full-pack mem model: per-job mem = cores/worker x
3900M, so cores/worker is the memory knob (8->31GB, 16->62GB, 32->125GB,
48->187GB). Set `--mem-per-cpu=3900M`.

Open tradeoffs to iterate on:
- Tier-vs-safety: short tiers backfill better, but under-requesting walltime
  gets the job KILLED and charged for a wasted partial run. Request the tier
  that comfortably covers the bucket's measured MAX wall, not the average.
- The ~13.9h tail in every >=31-atom bucket is SCF non-convergence (actinides),
  not size. Fix those with `omol_base`/KDIIS, NOT more walltime - do not size
  lanes around the tail.
- Per-cluster tier choice differs (table above). The same atom-cap lane may
  want a different walltime tier on Narval (3h) vs Rorqual (12h).

## Iteration log

- 2026-06-25: Initial diagnosis across Narval, Fir, Rorqual, Nibi. Fairshare
  ruled out (inverse correlation: Nibi worst factor 0.289 but best throughput).
  Bottleneck is backfill contention on 100%-packed clusters. Captured
  least-contended tier per cluster and the size->wall->cores lane direction.
  Next: design atom/core-capped lanes and validate start latency empirically
  per cluster/tier.

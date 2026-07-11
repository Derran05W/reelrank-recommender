# Phase 6 — Second-stage ranker: hnsw_ranker vs hnsw (similarity-only)

**Setup:** `configs/medium.json`, seed 42 — 10k users × 100k reels × 200 interactions/user =
2,000,000 impressions over 200,000 requests, Release build, single-threaded. Both arms share the
same master seed, dataset, behaviour stream, and HNSW graph (same single index-seed rng draw), so
every difference is attributable to the ranking stage alone. Runs:
`hnsw_ranker-seed42-20260711T032436`, `hnsw-seed42-20260711T032604`.

**Pipelines.** `hnsw`: one HNSW candidate source (k = 500) → merge/filter/cap → order by raw
similarity (TDD 16.4). `hnsw_ranker`: four sources (HNSW 500 + popular 100 + trending 100 +
creator-affinity 100, TDD 13) → merge/dedup/filter/cap 500 → `WeightedRanker` over the ten
TDD 14.1 features with the default TDD 14.2 weights (similarity 0.50, quality 0.10, trending
0.08, freshness 0.08, popularity 0.07, creator affinity 0.07, exploration 0.05 [inert until
Phase 8], duration match 0.05, − repetition 0.15, − impression fatigue 0.05).

## Headline: engagement vs alignment

| metric | hnsw | hnsw_ranker | Δ |
|---|---|---|---|
| **reward / impression (TDD 10.5)** | 0.02979 | **0.06710** | **+125% (2.25×)** |
| completion rate | 0.3268 | 0.3765 | +15.2% |
| like rate | 0.0627 | 0.0702 | +11.9% |
| share rate | 0.0232 | 0.0260 | +12.0% |
| follow rate | 0.00876 | 0.01024 | +16.8% |
| instant-skip rate | 0.5528 | 0.5052 | −8.6% |
| mean watch ratio | 0.3655 | 0.4107 | +12.4% |
| mean true affinity | 0.1062 | 0.0921 | −13.2% |
| oracle mean regret (affinity units) | 0.6766 | 0.6914 | +2.2% |

**Reading the trade-off:** the ranker more than doubles the composite reward — the objective the
system optimizes (watch + like + share + skip penalty) — by spending 30% of the score's weight on
quality/popularity/trending/duration signals that convert directly to engagement. The cost is a
13% drop in mean true affinity (and a 2.2% affinity-unit regret increase): pre-Phase-7 every user
shares the same static cold-start estimate, so raw similarity-to-estimate IS the affinity metric's
best friend, and any weight diverted from similarity dilutes it. The affinity-based oracle regret
is therefore expected to favour `hnsw`; reward is the criterion the TDD 3 question is scored on
(the plan's statistical exit test asserts reward, and it passes at both small — +20% — and medium
— +125% — scale). Mean watch seconds drop 11.6% while watch RATIO rises 12.4%: the duration-match
feature steers users toward reels short enough to finish — more completions from less watch time.

## Latency (wall-clock, TDD 18.7)

| stage (p50 / p95 ms) | hnsw | hnsw_ranker |
|---|---|---|
| retrieval | 0.128 / 0.136 | 1.000 / 1.044 |
| ranking | 0.004 / 0.005 | 0.116 / 0.124 |
| total | 0.148 / 0.156 | **1.196 / 1.244** |

The full ranked pipeline stays ~8× under the TDD 27 <10 ms p95 target. The added millisecond is
dominated by the retrieval stage: the popular and trending sources are honest O(catalog) scans
(100k reels each) per request, accepted for Phase 6 and documented in their headers; the
`WeightedRanker`'s ten-feature scoring of a 500-candidate pool costs only ~0.116 ms. Wall time:
88 s → 298 s for 2M impressions.

## Regression / determinism cross-check

The `hnsw` arm — re-run at the Phase 6 HEAD with the reworked orchestrator (nullptr-ranker path)
and the extended simulator (trending accumulators + creator-affinity bookkeeping) — reproduces the
Phase 5 published run **to full float precision** (mean_true_affinity 0.1062002732378534,
reward_per_impression 0.029788209575586488, completion_rate 0.3267985, identical live-retrieval
recall/distance-error). The Phase 6 additions are pure arithmetic on existing state with no rng
draws added/removed/reordered (D8), and the identity ranking path is byte-identical.

## Ranking explanations (TDD 14.4)

`explanation_example.json` is the committed output of
`inspect_user --explain-user 3 --warmup 20 --seed 42` at the default 100k-reel config: a 10-item
`hnsw_ranker` feed where every item carries all ten per-feature contributions summing exactly to
its score (the top item was surfaced by the Trending source with contributions
similarity 0.368 + popularity 0.063 + trending 0.062 + duration_match 0.044 + quality 0.040 +
freshness 0.035 − impression_penalty 0.005).

## Notes

- `metadata.json` records `reel_rank_dirty: true`: runs predate the phase commit (same provenance
  situation as Phases 1/4/5).
- Live retrieval metrics are identical across both arms by construction (same graph, same static
  query per user until Phase 7) — see the Phase 5 caveat about the single adversarially hard
  near-centroid query; re-read after Phase 7.
- `diversity_metrics.csv` still not emitted (no diversity until Phase 9, per the TDD 26 subset
  note from Phase 4).

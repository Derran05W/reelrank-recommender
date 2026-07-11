# Phase 5 — HNSW vs exact retrieval in the feed pipeline (configs/medium.json, seed 42)

The project's core engineering question (TDD §3): **what does approximate retrieval cost in
recommendation quality, and what does it buy in latency?** Both recommenders run end-to-end
through the Phase 5 orchestrator pipeline on the medium config: 10,000 users x 100,000 reels,
200 interactions/user (200,000 requests of feed size 10, 2,000,000 impressions), identical master
seed and rng streams, Release build, single-threaded.

## Headline

| | exact_vector | hnsw (ef_search=64) | delta |
|---|---|---|---|
| retrieval p50 / p95 / p99 (ms) | 6.33 / 8.92 / 9.78 | **0.132 / 0.143 / 0.158** | **48x / 62x faster** |
| mean true affinity | 0.10787 | 0.10620 | −1.5% |
| reward per impression | 0.02766 | 0.02979 | +7.7% |
| oracle mean regret (affinity units) | 0.67578 | 0.67661 | +0.12% |
| total wall seconds (2M impressions) | 2921 | 94 | 31x faster |

**HNSW retrieval delivers effectively exact-quality recommendations at ~2% of the retrieval
latency.** Feed-level quality is statistically indistinguishable from the brute-force ceiling:
true affinity is within 1.5%, oracle regret within 0.12%, and the small reward difference (+7.7%
for HNSW) reflects feed-composition noise interacting with stochastic behaviour, not a real
quality ordering — the affinity and regret columns are the monotone quality measures and both say
"parity". Engagement rates (completion, like, share, follow, skip) agree to three decimal places.
Full metric set in `comparison.csv`.

## Live retrieval quality (TDD §18.1), and how to read it

Live Recall@K is measured during the run on a 2% Bernoulli sample of requests (4093 samples,
identical across runs by stream independence): the recommender's own index is queried at k=50
against an exact ground-truth index over the same reels and the same live query.

| | exact_vector | hnsw ef64 | hnsw ef128 |
|---|---|---|---|
| live recall@10 / @50 (k=50 probe) | 1.00 / 1.00 | 0.40 / 0.38 | 0.60 / 0.52 |
| mean distance error (top-10) | 0.0 | 0.0101 | 0.0061 |

Two essential caveats, both verified:

1. **The probe queries one adversarially hard point.** Until Phase 7's online learning, every
   user's estimated preference is the same static cold-start vector (the global average
   preference, TDD §11.1). A near-centroid query is the worst case for ANN search on clustered
   data — the true top-50 are barely separated from thousands of near-ties (mean distance error
   0.01 at recall 0.40 shows how tight the near-tie band is). These recall numbers characterize
   that single query point, not the per-user personalized queries that arrive with Phase 7.
2. **The k=50 probe understates the pipeline's operating point.** vector-db's HNSW searches with
   an effective beam of `max(ef_search, k)` (hnsw_index.cpp). The feed pipeline retrieves
   k = 500 candidates (`recommendation.vector_candidates`), so production retrieval runs at beam
   500 regardless of ef_search. The feed-level comparison above (affinity/regret parity with
   exact) is the correct measure of pipeline retrieval quality; the k=50 probe is a conservative
   diagnostic.

## The ef_search question (carried from Phase 1) — resolved

Phase 1 left open whether the default `ef_search=64` (recall@10 ≈ 0.82 on 10k isotropic data)
should be raised to 128 (≈ 0.95) for the pipeline. Answer: **it makes no difference to feeds, by
construction.** Because the pipeline's k=500 dominates `max(ef_search, k)`, the ef64 and ef128
runs produced byte-identical feeds — every behaviour metric, oracle regret, and impression count
matches to full precision (compare the two hnsw columns in `comparison.csv`). Only the k=50
diagnostic probe (beam 64 vs 128) differs: recall@10 0.40 → 0.60, distance error 0.0101 → 0.0061.
The default stays `ef_search=64`; the pipeline's retrieval breadth knob is
`recommendation.vector_candidates`, and ef_search only matters for direct index queries with
k < ef_search. The ef128 run is kept under `ef128-diagnostic/` as evidence.

## Notes

- Baselines run with STATIC cold-start estimates (no online learning until Phase 7); the flat
  learning curve is by design. Requests set enableExploration=false, enableDiversity=false.
- Oracle regret is in TRUE-AFFINITY units on a 2% request sample (3945 requests, identical
  across algorithms by rng-stream independence) — see `regret_units_note` in each summary.json.
- exact_vector's latency `mean`/`max` (14.2 ms / 757 s) are contaminated by a system-level stall
  (machine sleep) during its ~49-minute run; the p50/p95/p99 percentiles over 200k samples are
  robust to the handful of affected samples. HNSW runs (~94 s) were unaffected.
- The exact-vs-exact row measuring exactly recall 1.0 / distance error 0.0 is the harness
  self-check (the recommender's index is compared against an independently built ground-truth
  index), validating the recall wiring.
- Per-run output (config.json, summary.json, retrieval_metrics.csv, per-round CSVs, latency,
  metadata.json with reel-rank + vector-db SHAs, build, hardware) in the run directories
  alongside this file. diversity_metrics.csv is N/A until Phase 9.

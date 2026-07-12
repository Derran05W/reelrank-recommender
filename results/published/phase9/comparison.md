# Phase 9 — Diversity re-ranking: engagement-vs-diversity trade-off

**Setup.** `configs/medium.json`, seed 42: 10,000 users × 100,000 reels × 200 interactions/user,
online learning enabled, 200,000 requests/arm. Four arms, differing ONLY in algorithm and two
config flags (each arm's fully-resolved config is committed in its `config.json`):

| arm | algorithm | diversity | use_mmr | exploration |
|---|---|---|---|---|
| `hnsw_ranker/` | hnsw_ranker (TDD 16.5) | (inert) | — | off for this algorithm |
| `constraints/` | hnsw_ranker_diversity (TDD 16.6) | on | false | off (isolation mode) |
| `constraints-mmr/` | hnsw_ranker_diversity | on | true | off (isolation mode) |
| `complete-system/` | hnsw_ranker_diversity | on | true | **on, ε=0.05** (complete initial system) |

Machine-readable table: `comparison.csv`. The four arms ran CONCURRENTLY (Release build), so
absolute latencies carry cache/memory contention; cross-arm quality comparisons are
like-for-like.

## Regression cross-checks (no-op contracts)

- `hnsw_ranker` at Phase 9 HEAD reproduces **Phase 7's published run byte-identically**
  (`recommendation_metrics.csv`, `learning_curve.csv`, `regret_curve.csv`,
  `retrieval_metrics.csv` all `cmp`-identical to
  `phase7/hnsw_ranker-seed42-20260711T042131/`) — with `diversity.enabled=true` in its config,
  proving the entire phase (request flag, metrics collection, reranker plumbing) is exactly inert
  for existing algorithms at 2M-impression scale.
- The integration suite additionally proves `hnsw_ranker_diversity` with
  `diversity.enabled=false` is byte-identical to `hnsw_ranker` (the Phase-8-style gate no-op),
  and that two identical complete-system runs are byte-identical (determinism).

## Repetition eliminated (exit criterion 1, live verification)

`repetition_rate` — feed items already shown to that user earlier in the run, or duplicated
within a feed — is **exactly 0 in all four arms** (2.0M / 1.26M / 1.29M / 1.30M impressions),
published per-round in every `diversity_metrics.csv`. The flagship property suite asserts the
same invariants (plus creator/topic caps and no-seen) across 24 seeds × both reranker modes.

## Diversity improves (exit criterion 2)

Feed-size-fair comparison (hard caps shorten feeds — see trade-off below — and raw per-feed
counts/HHI are mechanically depressed/inflated by shorter feeds, so per-item ratios and the
size-robust intra-list similarity are the honest measures):

| arm | unique topics /item | unique creators /item | intra-list similarity ↓ | topic HHI ↓ | creator HHI |
|---|---|---|---|---|---|
| hnsw_ranker | 0.231 | 0.811 | 0.6962 | 0.7181 | 0.1583 |
| constraints | 0.437 (+89%) | 0.927 (+14%) | 0.6149 (−12%) | 0.5656 (−21%) | 0.2391* |
| constraints-mmr | 0.439 (+90%) | 0.932 (+15%) | 0.6066 (−13%) | 0.5529 (−23%) | 0.2299* |
| complete-system | 0.438 (+89%) | 0.930 (+15%) | **0.5946 (−15%)** | **0.5433 (−24%)** | 0.2269* |

\* Raw creator HHI RISES despite the ≤2-per-creator cap because HHI's floor is 1/n and feeds
shrank from 10.0 to ~6.4 items; per-item unique-creator ratio (which improves 14–15%) is the
size-fair creator-diversity read. Raw per-feed values for every metric are in `comparison.csv`
and per-round in each arm's `diversity_metrics.csv`.

## Engagement trade-off (exit criterion 3)

| arm | reward/impression | mean true affinity | completion | instant-skip | mean feed size | est↔hidden cosine (final) | oracle regret |
|---|---|---|---|---|---|---|---|
| hnsw_ranker | **0.2141** | **0.2421** | 0.5356 | 0.3677 | 10.00 | **0.4247** | **0.5416** |
| constraints | 0.1848 (−13.7%) | 0.2132 (−11.9%) | 0.5047 | 0.3944 | 6.32 | 0.3850 | 0.5479 |
| constraints-mmr | 0.1816 (−15.2%) | 0.2096 (−13.4%) | 0.5007 | 0.3977 | 6.44 | 0.3814 | 0.5555 |
| complete-system | 0.1841 (−14.0%) | 0.2101 (−13.2%) | 0.5045 | 0.3942 | 6.50 | 0.3826 | 0.5545 |

Three distinct costs, quantified:

1. **Per-impression engagement −13.7%** (constraints vs baseline): diversity slots go to
   off-preference topics, the same engagement-vs-alignment shape as Phase 6's ranker trade-off
   (and the affinity-unit oracle regret again structurally favours similarity-heavy feeds — read
   its +1.2% with the Phase 4 caveat).
2. **Delivered volume −37%** (feeds 10.0 → 6.3 items): the headline operational finding. The
   orchestrator's pool cap keeps the `candidateLimit` MOST-SIMILAR candidates, which for a
   personalized query are topic-concentrated; the scaled topic cap (3 per 10-item feed) then
   binds on a ~2-topic pool at ~3+3, so the cap-feasible feed is ~6.3 items (round 0, with the
   shared near-centroid cold-start query, delivers full 10-item feeds; the shortfall appears
   exactly when personalization kicks in at round 1+). Documented short-feed-over-cap-relaxation
   behaviour (`constraint_reranker.hpp`); the knobs if fuller feeds are wanted are a diverse pool
   cap or a larger `candidateLimit`, both left to future work — NOT silently relaxing TDD 15.1's
   hard rules.
3. **Learning drag**: final est↔hidden cosine 0.4247 → 0.3850 (−9.3%) — diverse feeds feed the
   online learner more off-preference evidence, slowing convergence relative to a pure exploit
   loop. (Mirror image of Phase 8's finding that ε-exploration IMPROVED alignment: exploration
   diversifies the pool; hard caps instead REMOVE on-preference impressions.)

**MMR increment** (constraints-mmr vs constraints): a further −1.7% reward/impression buys
intra-list similarity 0.6149 → 0.6066 and topic HHI 0.5656 → 0.5529. MMR only re-orders the
constraint-selected set within a request; the small set-level differences arise through the
learning feedback loop (different presentation order → different interactions → different
queries). `use_mmr=true` stays the shipped default: the cost is small, ordering quality
(adjacent-item variety) is its actual objective, and TDD 15.2 recommends λ=0.75.

**Complete initial system** (all six TDD 13 sources incl. ungated fresh:100 — closing the
Phase 8 deferred deviation — plus ε=0.05 exploration with guaranteed slots, plus diversity):
best diversity of every arm (ILS 0.5946, topic HHI 0.5433), reward within 0.4% of
constraints-only, and +2.8% impressions vs constraints (exploration/fresh candidates diversify
the pool, making more items cap-feasible). Exploration remains compatible with hard caps: caps
take precedence over the guarantee (documented in `orchestrator.cpp`).

## Latency (TDD 18.7 / §27)

Whole-pipeline recommend() p50/p95: 2.47/3.04 ms (hnsw_ranker), 2.49/3.07 ms (constraints),
2.49/3.07 ms (constraints-mmr), 3.31/8.18 ms (complete-system — six sources incl. two
O(catalog) scans + exploration modes, and four-way run contention; the Phase 11
leaderboard-cache item covers the scans). Re-ranking itself is negligible: p95 ≤ 0.04 ms in
every diversity arm. §27's p95 < 10 ms holds in all arms.

## Reproduce

`build-release/apps/simulate --config <arm config.json> --algorithm <arm> --seed 42 --out <dir>`
— each arm directory's `config.json` is the fully-resolved config; deterministic CSVs are
byte-reproducible (D8). `metadata.json` records `reel_rank_dirty: true` (runs predate the phase
commit — same provenance situation as Phases 1/4–8).

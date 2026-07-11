# Phase 7 — Online preference learning: hnsw_ranker learning vs frozen estimates

**Setup:** `configs/medium.json`, seed 42 — 10k users × 100k reels × 200 interactions/user =
2,000,000 impressions over 200,000 requests (20 rounds of feed-size 10), Release build,
single-threaded. Both arms are the full Phase 6 `hnsw_ranker` pipeline (four candidate sources →
WeightedRanker, now over ELEVEN features incl. the newly-activated session-topic similarity at
weight 0.05). The ONLY difference between the arms is `learning.enabled`:

- **learning** (`hnsw_ranker-seed42-20260711T042131`): `OnlineUserStateUpdater` runs after every
  interaction — long-term update η=0.02 (TDD 11.2), session vector recomputed over the
  current-session events of the 20-interaction window with λ=0.90 (TDD 11.3), effective estimate
  = normalize(0.65·longTerm + 0.35·session) (TDD 8.3).
- **frozen** (`frozen-hnsw_ranker-seed42-20260711T042112`): every user's estimate stays at the
  cold-start global-average preference for the whole run (the pre-Phase-7 behaviour).

Round-0 reward is IDENTICAL in both arms (0.102379) — the first feeds are served before any
learning has occurred, confirming the arms share dataset, streams, and HNSW graph exactly (D8);
everything after is attributable to online learning alone.

## Headline: personalization pays on every axis

| metric (overall, 2M impressions) | frozen | learning | Δ |
|---|---|---|---|
| **reward / impression (TDD 10.5)** | 0.06753 | **0.21410** | **+217% (3.17×)** |
| mean true affinity | 0.09251 | 0.24211 | +162% (2.62×) |
| oracle mean regret (affinity units) | 0.68978 | 0.54161 | **−21.5%** |
| completion rate | 0.3782 | 0.5356 | +41.6% |
| like rate | 0.0703 | 0.1228 | +74.8% |
| share rate | 0.0260 | 0.0452 | +74.0% |
| follow rate | 0.0102 | 0.0354 | +247% |
| instant-skip rate | 0.5034 | 0.3677 | −27.0% |
| mean watch ratio | 0.4125 | 0.5710 | +38.4% |
| estimated↔hidden cosine (end of run) | 0.2161 (constant) | **0.4247** | +96% over the prior |

Unlike Phase 6's engagement-vs-alignment trade-off, learning improves reward AND true affinity
AND regret simultaneously — personalizing the query enlarges the pie instead of re-slicing it.

## Learning curve (TDD 18.5) — `learning_curve.csv`

| interactions/user | learning reward | learning est↔hidden cos | frozen reward | frozen cos |
|---|---|---|---|---|
| 10 (round 0) | 0.10238 | 0.2507 | 0.10238 | 0.2161 |
| 20 | 0.13527 | 0.2640 | 0.07227 | 0.2161 |
| 30 | 0.16516 | 0.2719 | 0.07241 | 0.2161 |
| 50 | 0.18604 | 0.2921 | 0.07271 | 0.2161 |
| 100 | 0.21544 | 0.3370 | 0.07261 | 0.2161 |
| 150 | 0.24597 | 0.3824 | 0.06245 | 0.2161 |
| 200 | 0.27670 | 0.4247 | 0.02966 | 0.2161 |

Two findings:
1. **New users improve monotonically** — reward per impression rises every round without
   exception (0.102 → 0.277), and alignment climbs from the 0.216 cold prior to 0.425.
2. **The frozen arm decays in two phases**: an immediate drop off the first feed (0.102 → 0.072
   at round 1 — the static query's very best matches are consumed at once and stay seen), a long
   ~0.072 plateau through 100 interactions, then a collapse to 0.030 by 200 as the seen-filter
   exhausts the whole neighbourhood of the shared cold-start estimate. Online learning does not
   merely add improvement on top of a flat baseline — it is what keeps a personalized feed from
   running out of good candidates.

## Cold-start metrics (TDD 18.5)

Mean oracle regret over the first N impressions per user (round-window approximation at feed
size 10: N=10 → round 0, N=25 → rounds 0–2 [30 impressions], N=50 → rounds 0–4, N=100 →
rounds 0–9; sampled requests, rate 0.02):

| window | frozen | learning | Δ |
|---|---|---|---|
| first 10 | 0.6937 | 0.6937 | 0 (identical feeds) |
| first ~25 | 0.6759 | 0.6316 | −6.6% |
| first 50 | 0.6795 | 0.6118 | −10.0% |
| first 100 | 0.6859 | 0.5844 | −14.8% |

Interactions to reach target reward: the learning arm exceeds the frozen arm's OVERALL mean
reward (0.0675) from its very first round (10 interactions, 0.102) and never drops below it; it
exceeds the frozen arm's BEST round (0.102, round 0) from round 1 (20 interactions).

## Live retrieval quality — Phase 5's open question, re-read (TDD 18.1)

| metric | frozen | learning |
|---|---|---|
| Recall@10 | 0.400 | **0.694** |
| Recall@50 | 0.380 | 0.625 |
| mean distance error (top-10) | 0.0101 | 0.0097 |

Phase 5 flagged its live recall numbers (0.40@10 at ef64) as "a single adversarially hard query"
— every user probed the same near-centroid cold-start estimate, ANN's worst case. As predicted,
once online learning personalizes the query the measured live recall rises to ~0.69@10 at
IDENTICAL HNSW parameters (m=16, efConstruction=200, ef_search=64, k=50 probes). The Phase 5
conclusion stands: feed-level quality parity was the right retrieval-quality measure, and the
low cold-start recall was a property of the query distribution, not the index.

## Latency (wall-clock, TDD 18.7)

total p50/p95: frozen 2.06/2.47 ms, learning 2.15/2.44 ms — no meaningful difference (the
updater is O(window·dim) per IMPRESSION on the simulation side; it never touches the recommend()
path). Caveat: all three runs of this phase executed CONCURRENTLY on the same machine, so
absolute latencies are inflated vs Phase 6's solo runs (retrieval p50 1.85 ms vs 1.00 ms) by
cache/memory contention; cross-arm comparisons within this phase are like-for-like. The §27
<10 ms p95 target still holds with 4× headroom.

## Regression / determinism cross-check (`phase6-regression-xcheck/`)

A third arm ran at Phase 7 HEAD with `learning.enabled=false` AND `session_topic_weight=0`:
`recommendation_metrics.csv`, `regret_curve.csv`, and `retrieval_metrics.csv` are
**byte-identical to Phase 6's published** `hnsw_ranker-seed42-20260711T032436`. Phase 7's
changes are therefore provably additive: with both new knobs off, the pipeline reproduces the
Phase 6 system bit-for-bit (only that arm's config/summary/metadata are committed here; its CSVs
are Phase 6's own).

The frozen-with-session-topic arm (above) differs from Phase 6 only marginally (reward 0.06753
vs 0.06710, affinity 0.09251 vs 0.09210): with a static session vector the session-topic feature
is nearly-redundant with similarity, so activating it at weight 0.05 barely moves the feed.

## Notes

- Both arms' `metadata.json` record `reel_rank_dirty: true` — the runs predate the phase commit
  (same provenance situation as Phases 1/4/5/6).
- The affinity-unit oracle regret (Phase 4 deviation) structurally favours similarity-style
  rankers; here BOTH arms are the same ranked pipeline, so the −21.5% regret delta is a clean
  like-for-like improvement.

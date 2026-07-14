# ReelRank — Results

This document answers ReelRank's core engineering question and each secondary question from the
technical design (TDD §3) with measured numbers, then audits the project against the MVP (TDD §32)
and strong-portfolio (TDD §33) completion checklists. Every claim points to a committed artifact
under [`../results/published/`](../results/published/); figures referenced by name live in
[`../results/published/figures/`](../results/published/figures/).

## Hardware and measurement context

All Phase 11 numbers (throughput, tail latency, the recall grid) were measured on an **Apple M5,
10 cores, 24 GB RAM, macOS (Darwin 25.5.0), AppleClang 21.0.0, Release `-O3`**, reel-rank at the
Phase 11 tree (`metadata.json` records merge SHA `537d883`, tree-identical to the squashed Phase 11
commit), vector-db `17e434a`, seed 42. **Phase 11 is the only phase whose latencies are
uncontaminated**: every published number came from a single serialized 78-minute run chain on an
idle machine. Phases 4–10 ran their experiment arms *concurrently* on one machine, so their
absolute latencies carry cache/memory contention — cross-arm comparisons *within* a phase are
like-for-like, but only Phase 11 should be quoted for absolute RPS/p99. Each run directory carries a
`metadata.json` with the full CPU/RAM/OS/compiler/flags/thread-count/dataset provenance required by
TDD §27.

---

## The core question

> **How closely can approximate vector retrieval match exhaustive personalized search while
> dramatically reducing recommendation latency?** (TDD §3)

**Answer: at the shipped configuration, HNSW retrieval is quality-indistinguishable from
brute-force exact search while cutting retrieval latency 48–62×.**

The controlled head-to-head is Phase 5
([`../results/published/phase5/`](../results/published/phase5/)): `exact_vector` versus `hnsw`
through the identical orchestrated pipeline on the medium config (10,000 users × 100,000 reels ×
200 interactions = 2,000,000 impressions), same seed, same rng streams, single-threaded.

| | exact_vector | hnsw (ef_search=64) | delta |
|---|---|---|---|
| mean true affinity | 0.10787 | 0.10620 | **−1.5%** |
| oracle mean regret (affinity units) | 0.67578 | 0.67661 | **+0.12%** |
| reward / impression | 0.02766 | 0.02979 | +7.7% (noise, in HNSW's favour) |
| retrieval p50 / p95 / p99 (ms) | 6.33 / 8.92 / 9.78 | **0.132 / 0.143 / 0.158** | **48× / 62× / 62× faster** |
| wall time, 2M impressions (s) | 2,921 | 94 | 31× faster |

The two monotone quality measures — mean true affinity and oracle regret — both say *parity*: true
affinity is within 1.5%, oracle regret within 0.12%. The +7.7% reward difference is feed-composition
noise interacting with stochastic behaviour, not a real quality ordering (engagement rates agree to
three decimals). HNSW buys effectively-exact recommendations at ~2% of the retrieval latency. See
[`hnsw_vs_exact.png`](../results/published/figures/hnsw_vs_exact.png).

**Retrieval quality holds on production-like data at scale.** At the *shipped default* HNSW
configuration (M=16, efConstruction=200, ef_search=64), Recall@10 on the generator's topic-clustered
(production-like) embeddings is **0.934 @ 10k** and **0.919 @ 100k**
([`../results/published/phase11/retrieval/clustered-100k/`](../results/published/phase11/retrieval/clustered-100k/),
`clustered-10k/`). Because the feed pipeline fetches k=500 candidates (beam 500 ≥ 256), effective
Recall@10 at the pipeline's operating point is **≥ 0.932** at 100k. (On isotropic-random unit
vectors — a documented near-worst-case ANN distribution — the same parameters score 0.47 @ef64;
this must **not** be quoted as system recall.)

**Pure-ANN latency stays sub-2 ms even at extreme scale.** HNSW search p95 is **≤ 1.8 ms at 1M
vectors** even at the widest grid point (M32, ef256), and 0.26 ms on clustered 100k at the k=500
operating point ([`../results/published/phase11/retrieval/`](../results/published/phase11/retrieval/)).
ANN is never the latency bottleneck (see the §27 table below and the secondary-question answers).

---

## TDD §27 performance targets — pass/fail

Measured in Phase 11 on the hardware above. Honest verdicts, with the one nuance called out.

| Benchmark | Target | Measured | Verdict |
|---|---|---|---|
| Small (10k reels, 1k users, 64d) | e2e p95 < 10 ms | 0.53 ms (T=1) … 1.26 ms (T=10) | **PASS** (8–19× headroom) |
| Small | HNSW Recall@10 > 90% | **0.934** @ default M16/efC200/ef64, clustered (0.965 @ef256) | **PASS** |
| Medium (100k reels, 10k users, 64d) | HNSW p95 retrieval < 10 ms | pure HNSW search p95 **0.26 ms** (clustered) / 0.84 ms (random) at k=500; full six-source candidate-generation *stage* p95 3.9 ms (T=1) … **11.4 ms (T=10)** | **PASS on the HNSW reading everywhere**; the candidate-generation *stage* crosses 10 ms at T≥8 — see nuance |
| Medium | e2e p95 < 25 ms | 4.2 ms (T=1) … 12.0 ms (T=10) | **PASS** (2–6× headroom) |
| Medium | HNSW Recall@10 > 90% | **0.919** @ default ef64 clustered (0.958 @128d); effective ≥ 0.932 at k=500. Random worst case 0.47 @ef64 | **PASS** on production-like data |
| Large stretch (1M reels, 100k users, 128d) | none ("establish after measuring") | build 640 s, peak RSS ~2.0 GB, e2e p95 43.9 ms (T=1) → 125.3 ms (T=10); pure HNSW p95 ≤ 1.8 ms | measured, no target to pass/fail |

**The one nuance (medium HNSW-p95 target).** The *pure HNSW search* passes at every thread count
(p95 0.26–0.84 ms at the k=500 operating point). What crosses 10 ms at T≥8 is the whole *six-source
candidate-generation stage* (11.4 ms @T=10), and profiling attributes that to the **O(catalog)
candidate-source scans** (popular / trending / fresh / exploration / creator-affinity), which sample
profiling clocks at **~69–72% of request self time** while HNSW is only ~11–13%
([`../results/published/phase11/profiling/BOTTLENECKS.md`](../results/published/phase11/profiling/BOTTLENECKS.md)).
The HNSW-specific target is met; the stage-level crossing is a linear-scan cost, not an ANN cost,
and the leaderboard-cache fix is designed but deliberately deferred because no §27 target actually
fails (see [`LIMITATIONS.md`](LIMITATIONS.md)).

---

## Secondary questions (TDD §3)

**How quickly can the system infer a new user's interests?**
Online learning drives new-user reward up monotonically every round — from 0.102 (round 0) to 0.277
(round 20) reward/impression, with estimated↔hidden preference cosine climbing 0.216 → 0.425 over
200 interactions
([`../results/published/phase7/`](../results/published/phase7/)). The dedicated cold-start
experiment injects 500 fresh users mid-simulation and measures their first 100 impressions: at the
shipped ε=0.05 exploration, injected users earn **+26% reward/impression by impression 100** versus
no exploration (0.222 vs 0.176), and ε=0.05 is the only arm whose new users reach the (deliberately
hard) pre-injection population-mean reward inside their 100-impression window (at impression 98)
([`../results/published/phase8/`](../results/published/phase8/),
[`cold_start.png`](../results/published/figures/cold_start.png)).

**How quickly can it adapt when a user's interests change?**
Phase 10 drifts the whole population's hidden preference at interaction 100 and compares learning
configurations ([`../results/published/phase10/`](../results/published/phase10/),
[`drift_recovery.png`](../results/published/figures/drift_recovery.png)). Adaptation is read from
**round-by-round dominance and post-drift slopes**, not instantaneous levels (the noise-free drift
target lifts every arm's affinity ceiling at drift, including the frozen arm, so absolute post-drift
levels are partly a reward-scale shift). On that unconfounded footing: the session-aware default
(0.65/0.35) beats long-term-only (1.0/0.0) on estimated↔hidden **alignment at every post-drift
round** (final **0.662 vs 0.472**) and on **drifted-cohort reward from one round after drift onward**
with a widening gap (final-2-round mean **0.431 vs 0.295, +46%**); the frozen static arm never adapts
(alignment flat at 0.335, late reward 0.174). Session-aware re-crosses 95% of its *own* pre-drift
reward baseline within 20 interactions, driven by the session window refilling with new-preference
events. (Raw cross-arm "time to 95% recovery" numbers are double-confounded and are **not** used for
comparison — see [`LIMITATIONS.md`](LIMITATIONS.md).)

**How should exploration be balanced against known preferences?**
The ε-greedy sweep (ε ∈ {0, 0.02, 0.05, 0.10}) shows moderate exploration is free or better for the
whole population: every ε>0 arm beats ε=0 on reward, affinity, alignment, and regret simultaneously
([`../results/published/phase8/`](../results/published/phase8/),
[`cumulative_regret.png`](../results/published/figures/cumulative_regret.png)). ε=0.05 (the TDD §12.7
suggested value, confirmed as the shipped default) wins whole-population reward (0.21527 vs 0.21277,
+1.2%), alignment (0.4245 vs 0.4179), and new-user regret at every window ≥25; ε=0.02 is marginally
better on *existing*-population cumulative regret; ε=0.10 over-explores (gives back reward). ε=0 is
byte-identical to the non-exploration pipeline at 2.05M-impression scale — a verified no-op that
isolates ε as the sole variable.

**How much recommendation quality is lost when HNSW replaces brute-force search?**
Essentially none at the shipped configuration: −1.5% mean true affinity and +0.12% oracle regret at
feed level (Phase 5, the core-question table above), with clustered Recall@10 of 0.919–0.934 at the
default HNSW parameters (Phase 11). The quality gap is within behavioural noise; the latency saving
is 48–62×. See [`hnsw_vs_exact.png`](../results/published/figures/hnsw_vs_exact.png) and
[`recall_vs_latency.png`](../results/published/figures/recall_vs_latency.png).

**How do HNSW parameters affect recall, latency, insertion cost, and memory?**
Recall rises monotonically with both M and ef_search across the Phase 1 and Phase 11 grids
([`../results/published/phase1/`](../results/published/phase1/),
[`../results/published/phase11/retrieval/`](../results/published/phase11/retrieval/),
[`recall_vs_efsearch.png`](../results/published/figures/recall_vs_efsearch.png),
[`latency_vs_efsearch.png`](../results/published/figures/latency_vs_efsearch.png)). At ~0.97
Recall@10, HNSW performs **~10,800 distance computations per query versus 100,000 for brute force**
(9.3× fewer; comps scale ~linearly with ef and sublinearly with M). efConstruction shows
diminishing returns past 200 (0.573/0.633/0.666/0.675 at efC 50/100/200/400 for 1.8/3/5.5/10× build
cost), justifying the efC=200 default. Insert throughput at 100k is 2,629–6,931 vec/s (M32→M8) and
falls to ~1,563 vec/s at 1M (graph-growth cost). A notable finding: on clustered data **128d beats
64d on recall (0.958 vs 0.919 @ef64)** — topic clustering keeps intrinsic dimensionality low,
reversing the distance-concentration penalty that crushes 128d on random data.

**How much does second-stage ranking improve over raw vector similarity?**
The feature-based `WeightedRanker` more than doubles the objective: reward/impression
**0.02979 → 0.06710, +125% (2.25×)**, with completions +15%, likes +12%, shares +12%, follows +17%,
instant-skips −8.6% ([`../results/published/phase6/`](../results/published/phase6/)). Honestly, this
came at a **−13.2% mean true affinity** cost at Phase 6 (0.1062 → 0.0921): pre-learning, every user
shares the same static cold-start estimate, so raw similarity *is* the affinity metric's best proxy
and any weight diverted from it dilutes affinity — a documented engagement-vs-alignment trade-off,
and reward is the TDD §3 criterion. That trade-off **shifts once online learning lands** (Phase 7):
personalizing the query lifts affinity +162% and reward another +217%, so ranking and learning
compound rather than trade.

**How does diversity-aware re-ranking affect engagement and feed repetition?**
Hard diversity constraints drive feed **repetition rate to exactly 0** in all arms (verified over
1.26–2.0M impressions) and raise per-item topic diversity **+89%** (intra-list similarity −12%, topic
HHI −21%) ([`../results/published/phase9/`](../results/published/phase9/),
[`diversity_vs_reward.png`](../results/published/figures/diversity_vs_reward.png)). The cost is
**−13.7% reward/impression** and — the phase's operational finding — feeds **shrinking from 10.0 to
~6.3 items** once personalization concentrates the candidate pool onto ~2 topics and the hard topic
cap binds. That short feed is documented behaviour under hard caps (TDD §15.1), not a defect; the
designed knobs (topic-aware pool cap, larger candidateLimit) are recorded as future work.

**How does the system behave under concurrent recommendation requests?**
Concurrent `const search()` on a frozen `HNSWVectorIndex` is **data-race-free** — zero
ThreadSanitizer findings on both an 8-thread probe and the full load driver under TSan — so the
design is one shared read-only index plus per-thread pipeline state (no replicas, no mutex)
([`../results/published/phase11/concurrency/`](../results/published/phase11/concurrency/)). Throughput
scales **349 → 1,821 RPS from T=1 to T=10 at 100k reels** (5.2×; 6.1× @10k, 3.9× @1M)
([`throughput_vs_clients.png`](../results/published/figures/throughput_vs_clients.png)). The ceiling
tightens as the working set grows (CPU ~975% at T=10 in every cell — cores busy waiting on DRAM),
the signature of a memory-bandwidth-bound workload; corpus scaling is linear (e2e p99 at T=10 goes
13.2 → 134.6 ms for 100k → 1M) because the linear candidate-source scans, not HNSW, dominate
([`p99_vs_corpus.png`](../results/published/figures/p99_vs_corpus.png)).

---

## MVP completion audit (TDD §32)

Every §32 item, with a one-line evidence pointer. **All items met.**

| MVP requirement | Status | Evidence |
|---|---|---|
| ≥ 100,000 synthetic reels | ✅ | medium config = 100k reels (Phase 5+); 1M generated in Phase 11 |
| ≥ 10,000 synthetic users | ✅ | medium config = 10k users (Phase 5+) |
| Reel embeddings indexed with existing HNSW | ✅ | `HNSWVectorIndex` wraps vector-db `HNSWIndex` directly (Phase 1) |
| Exact and HNSW retrieval comparable | ✅ | differential suite (Phase 1) + feed-level exact-vs-hnsw (Phase 5) |
| Recall@K measured | ✅ | `phase1/retrieval_metrics.csv`, live recall (Phase 5/7), Phase 11 grid |
| Retrieval p50/p95/p99 measured | ✅ | `phase5` timing block, `phase11/load` + `retrieval` CSVs |
| Random and popularity baselines exist | ✅ | `RandomRecommender`, `PopularityRecommender` (Phase 4) |
| Personalized > random | ✅ | mean true affinity 0.1058 vs 0.0451 (2.3×), Phase 4 exit test |
| HNSW quality compared with exact | ✅ | Phase 5 (−1.5% affinity, +0.12% regret, 48–62× faster) |
| Users update estimated preferences | ✅ | `OnlineUserStateUpdater`, η=0.02 + λ=0.90 session (Phase 7) |
| New-user quality improves over time | ✅ | monotone learning curve 0.102 → 0.277 (Phase 7); cold-start (Phase 8) |
| Deterministic under fixed seeds | ✅ | D8 portable RNG; byte-identical-CSV determinism tests every phase |
| All core modules have automated tests | ✅ | 533/533 tests green (unit/integration/property/differential) |
| Reproducible benchmark instructions | ✅ | committed `config.json` per run, `scripts/`, `configs/`, README + `scripts/demo.sh` |
| README explains architecture and limitations | ✅ | `README.md` + [`LIMITATIONS.md`](LIMITATIONS.md) |

## Strong-portfolio audit (TDD §33)

| Strong-portfolio item | Status | Evidence / note |
|---|---|---|
| Session-aware preferences | ✅ | 0.65 long-term / 0.35 session blend (Phase 7) |
| Preference drift | ✅ | scheduled drift + adaptation (Phase 10) |
| Diversity re-ranking | ✅ | constraint + MMR rerankers, repetition 0 (Phase 9) |
| Exploration | ✅ | ε-greedy sweep, ε=0.05 default (Phase 8) |
| Cold-start experiments | ✅ | mid-sim user/reel injection, first-100 windows (Phase 8) |
| One-million-reel benchmark | ⚠️ partial | 1M load + retrieval run (Phase 11) — but **random data only; clustered-1M recall NOT run** (time budget). Random-vector 1M recall (0.19) is a distribution artifact, not system recall. Recorded as a limitation. |
| Concurrent request benchmark | ✅ | TSan-verified, 349→1,821 RPS (Phase 11) |
| Feature-contribution inspection | ✅ | `explanation_example.json`, `inspect_user --explain-user` (Phase 6) |
| Recall–latency trade-off graphs | ✅ | `recall_vs_latency.png`, `recall_vs_efsearch.png`, `latency_vs_efsearch.png` |
| Recommendation reward comparisons | ✅ | every phase's `comparison.md` + `comparison.csv` (Phases 4–10) |
| Clean architecture diagrams | ✅ | `README.md` architecture diagram (mirrors TDD §7) |
| Documented hardware and build flags | ✅ | `metadata.json` per run (CPU/RAM/OS/compiler/flags/threads/dataset) |
| Honest limitations and future work | ✅ | [`LIMITATIONS.md`](LIMITATIONS.md) |
| **LSH comparison (TDD §7 sketch)** | ❌ gap | TDD §7 sketches an LSH alternative and vector-db ships an LSH index, but ReelRank benchmarks **exact vs HNSW only**. Never built. Recorded as a limitation. |

Two honest gaps: the clustered-1M recall benchmark was not run (isotropic-random 1M was), and no
LSH-vs-HNSW comparison was built. Both are documented in [`LIMITATIONS.md`](LIMITATIONS.md).

---

## Traceability

Every number above is copied from a committed artifact, not from memory. Latency claims name their
hardware/threading context: Phase 11 runs are serialized and uncontended (the reference for absolute
latency/throughput); Phases 4–10 arms ran concurrently, so their percentile comparisons are
like-for-like *within* a phase only. Recall claims name the data regime (clustered production-like
vs isotropic random). The published-artifact paths in this document are the source of truth.

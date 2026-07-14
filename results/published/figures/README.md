# Published figures — ReelRank §26 recommended graphs

The eleven canonical figures below are the TDD §26 recommended graphs, consolidated into one
committed set for the portfolio README and `docs/`. Every figure is regenerated deterministically
from committed published CSVs/JSON by `scripts/plot_results.py --canonical`; no figure contains data
that is not already committed under `results/published/`.

**Regenerate the whole set** (from the repo, with the pinned Python — see D15; `uv` is invoked as
`python3 -m uv`, matplotlib needs Python 3.12):

```sh
cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical
```

That writes all eleven PNGs into `results/published/figures/`. To regenerate one figure, add
`--only <name>` (the name is the filename without `.png`), e.g.
`--only recall_vs_efsearch,cold_start`.

## Hardware / provenance context (read before quoting a latency number)

All latency-bearing figures were produced on the machine recorded in the Phase 11 run metadata
(`results/published/phase11/load/*/metadata.json`, `results/published/phase11/retrieval/*/metadata.json`):

- **CPU** Apple M5, 10 logical cores · **RAM** 24 GiB · **OS** macOS (Darwin 25.5.0, arm64)
- **Build** Release, AppleClang 21.0.0, C++20 · reel-rank `537d883`, vector-db `17e434a`

**Contention caveat.** The Phase 11 retrieval and load sweeps are the project's only *serialized,
uncontended* latency runs (one measured index, warmup + timed iterations), so their absolute
millisecond numbers are directly comparable. The Phase 4–10 published runs executed their arms
**concurrently** on the same box, so those runs' *absolute* latencies carry scheduler contention;
percentile comparisons **within a single phase** are still like-for-like (all arms saw the same
contention). This is why `hnsw_vs_exact.png` compares retrieval **percentiles** (p50/p99), never the
mean/max — the Phase 5 exact arm's `latency_metrics.csv` mean (14.2 ms) and max (≈757 s) are
contaminated by a machine sleep during that long run, whereas its p50/p95/p99 (6.33 / 8.92 / 9.78 ms)
are robust.

---

## The figures

### 1. `recall_vs_efsearch.png`
- **Shows** Recall@10 vs efSearch at the canonical medium-benchmark cell (N=100k, d=64, efC=200),
  one line per data distribution at fixed graph degree **M=16**.
- **Takeaway** On clustered (production-like) embeddings Recall@10 clears the 0.90 §27 target from
  efSearch=64 (0.919) and reaches 0.932 at efSearch=256; the same M=16 graph on uniform-random data
  tops out at 0.845 — real, clustered content is materially easier to retrieve.
- **Series choice** M=16 is the only graph degree measured for the clustered sweep at this cell, and
  is the natural mid-degree for random (which also has M=8/32); both distributions are drawn at M=16
  so the lines isolate the *distribution* effect. Clustered is the basis for the §27 verdict.
- **Source** `results/published/phase11/retrieval/**/retrieval_metrics.csv` (rows: `vector_count=100000,
  dimensions=64, ef_construction=200, k=10, m=16`; clean-latency rows only, i.e.
  `distance_comps_per_query < 0`).
- **Regenerate** `cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical --only recall_vs_efsearch`

### 2. `latency_vs_efsearch.png`
- **Shows** Query p95 latency (ms) vs efSearch at the same cell/series as figure 1.
- **Takeaway** Clustered p95 stays under 0.14 ms across the whole efSearch sweep (0.037 ms at ef=16 →
  0.139 ms at ef=256); random costs ~3× more at high efSearch (0.459 ms at ef=256) for lower recall.
- **Source** same as figure 1 (`query_p95_ms` column; clean-latency rows only).
- **Regenerate** `cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical --only latency_vs_efsearch`

### 3. `recall_vs_latency.png`
- **Shows** Recall@10 vs query p95 latency Pareto at the canonical cell; color = distribution,
  marker = M, each point annotated with its efSearch.
- **Takeaway** The clustered M=16 frontier dominates the top-left (≈0.92 recall at ≈0.05 ms p95,
  ef=64); random needs M=32 and ~0.4 ms p95 to approach the 0.90 line — an order-of-magnitude worse
  recall/latency trade-off than clustered.
- **Source** same retrieval CSVs as figure 1, all M at the cell (clean-latency rows).
- **Regenerate** `cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical --only recall_vs_latency`

### 4. `reward_vs_interactions.png`
- **Shows** Mean reward per impression vs interactions/user, online-learning arm vs frozen arm (Phase 7).
- **Takeaway** Online preference learning lifts reward/impression to 0.214 over the run vs 0.068 frozen
  — **+217%** — and the learning curve keeps climbing to 0.276 by 200 interactions while frozen decays.
- **Source** `results/published/phase7/hnsw_ranker-seed42-*/learning_curve.csv` (learning) and
  `results/published/phase7/frozen-hnsw_ranker-seed42-*/learning_curve.csv` (frozen).
- **Regenerate** `cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical --only reward_vs_interactions`

### 5. `cumulative_regret.png`
- **Shows** Cumulative oracle regret (true-affinity units) vs interactions/user, both Phase 7 arms.
- **Takeaway** Learning ends at 2137 cumulative regret vs 2721 frozen (**−21.5%**); the gap widens
  monotonically after the arms diverge past ~25 interactions.
- **Source** `results/published/phase7/{hnsw_ranker,frozen-hnsw_ranker}-seed42-*/regret_curve.csv`
  (x-axis mapped through the paired `learning_curve.csv` round→interactions).
- **Regenerate** `cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical --only cumulative_regret`

### 6. `hnsw_vs_exact.png`
- **Shows** Two panels: recommendation-quality parity (mean true affinity, reward/impression, oracle
  regret) and retrieval-latency gap (p50/p99, log scale) for the Phase 5 exact-vector vs HNSW arms.
- **Takeaway** HNSW matches exact search on quality (affinity 0.106 vs 0.108, reward 0.030 vs 0.028,
  regret 0.677 vs 0.676) at **62× lower** retrieval p99 (0.158 ms vs 9.78 ms; p50 48× lower).
- **Source** `results/published/phase5/exact_vector-seed42-*/` and `.../hnsw-seed42-*/` — quality from
  `summary.json` (`metrics.*`, `oracle.mean_regret`), latency from the `retrieval` stage row of
  `latency_metrics.csv` (**percentiles only**; see the contention caveat above).
- **Regenerate** `cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical --only hnsw_vs_exact`

### 7. `diversity_vs_reward.png`
- **Shows** Reward per impression vs intra-list similarity (lower = more diverse), one labeled point
  per Phase 9 re-ranking arm, annotated with unique topics/feed.
- **Takeaway** The engagement↔diversity trade-off is explicit: ranker-only earns 0.214 reward at
  ILS 0.696 (2.31 topics/feed); the complete system gives up ~14% reward (0.184) to cut ILS to 0.595
  and raise diversity to 2.85 topics/feed.
- **Source** `results/published/phase9/{hnsw_ranker,constraints,constraints-mmr,complete-system}/summary.json`
  (`metrics.reward_per_impression`, `diversity.mean_intra_list_similarity`, `diversity.mean_unique_topics`).
- **Regenerate** `cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical --only diversity_vs_reward`

### 8. `cold_start.png`
- **Shows** Injected new-user mean reward vs impressions-since-injection, one line per ε-greedy
  exploration arm (Phase 8); a 5-impression centered rolling mean over the faint raw per-index series.
- **Takeaway** New users warm from ~0.10 to ~0.20 reward/impression within ~60 impressions regardless
  of ε; ε=0.05 leads late (new-user reward@100 = 0.222 vs 0.176 at ε=0) — modest exploration helps
  cold start without hurting steady state.
- **Source** `results/published/phase8/{eps000,eps002,eps005,eps010}/new_user_curve.csv`
  (`impression_index`, `mean_reward`; ε label from each arm's `config.json` `exploration.epsilon`).
- **Regenerate** `cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical --only cold_start`

### 9. `drift_recovery.png`
- **Shows** Mean reward per impression vs interactions/user for three Phase 10 arms (session-aware,
  long-term-only, frozen), with the drift onset (100 interactions) and each arm's 0.95×pre-drift
  guide marked.
- **Takeaway** After the hidden preference drifts at 100 interactions, the session-aware updater
  recovers to 0.95× its pre-drift reward within ~20 interactions and pulls ahead (0.44 by run end);
  the frozen arm never recovers (plateaus at ~0.19).
- **Source** `results/published/phase10/{session_aware,long_term_only,frozen}/learning_curve.csv`
  (`drifted_mean_reward` column) and `summary.json` (`adaptation.first_drift_interaction`,
  `adaptation.pre_drift_reward`).
- **Regenerate** `cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical --only drift_recovery`

### 10. `throughput_vs_clients.png`
- **Shows** Recommendations/second vs client threads, one line per (corpus, dimension) (Phase 11 load).
- **Takeaway** Throughput scales with concurrency — the 10k×64d corpus reaches 15,814 rec/s at 10
  threads (6.1× the single-thread 2,583); the 1M×64d corpus sustains 144 rec/s at 10 threads.
- **Source** `results/published/phase11/load/**/load_metrics.csv` (`threads`, `rps`, `corpus_reels`,
  `dimensions`).
- **Regenerate** `cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical --only throughput_vs_clients`

### 11. `p99_vs_corpus.png`
- **Shows** End-to-end and retrieval p99 latency vs corpus size (log-x) at the max thread count
  (10 threads) (Phase 11 load).
- **Takeaway** Tail latency grows sub-linearly then steepens with corpus: e2e p99 ≈ 1.3 ms at 10k,
  13.2 ms at 100k (d=64), 134.6 ms at 1M — retrieval dominates the end-to-end tail throughout.
- **Source** `results/published/phase11/load/**/load_metrics.csv` (`corpus_reels`, `threads`,
  `e2e_p99_ms`, `retrieval_p99_ms`).
- **Regenerate** `cd scripts && UV_PYTHON=3.12 python3 -m uv run plot_results.py --canonical --only p99_vs_corpus`

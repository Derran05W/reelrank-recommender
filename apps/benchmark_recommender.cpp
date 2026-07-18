// benchmark_recommender — Phase 11 multi-threaded closed-loop load driver (plan task 2, TDD
// 18.7/24.5/27, D13). T client threads, each with its OWN pipeline instance (candidate sources,
// rng, ranker, reranker, orchestrator — per-request mutable state is thread-local) sharing ONE
// frozen read-only HNSWVectorIndex, each drawing disjoint users, issuing full recommendation
// requests. No interaction simulation on the timed path.
//
// FROZEN OUTPUT CONTRACT (Phase 11 scaffolding — scripts/plot_results.py parses this schema;
// do not rename or reorder columns):
// load_metrics.csv, one row per (corpus_reels, dimensions, threads) measurement cell:
//   corpus_reels,users,dimensions,algorithm,threads,
//   warmup_requests_per_thread,timed_requests_per_thread,total_timed_requests,wall_seconds,rps,
//   e2e_p50_ms,e2e_p95_ms,e2e_p99_ms,e2e_mean_ms,e2e_max_ms,
//   retrieval_p50_ms,retrieval_p95_ms,retrieval_p99_ms,
//   ranking_p50_ms,ranking_p95_ms,ranking_p99_ms,
//   rerank_p50_ms,rerank_p95_ms,rerank_p99_ms,
//   cpu_utilization_pct,peak_rss_mb,index_build_seconds,index_insert_per_sec
// Plus config.json, summary.json, metadata.json per D12 in the same result directory.
//
// ---------------------------------------------------------------------------------------------
// DESIGN NOTES (Phase 11 package A).
//
// SHARED vs PER-THREAD (D13). The ONE shared object is the frozen HNSWVectorIndex; concurrent const
// search on it is verified data-race-free by apps/concurrency_check.cpp (TSan) and by running THIS
// app under TSan in --smoke (see results/published/phase11/concurrency/VERDICT.md). Everything with
// per-request mutable state is PER-THREAD: each thread builds its own six candidate sources
// (Popular/Trending/Creator/Fresh reuse a scratch_ buffer; Exploration holds an Rng* +
// lastFiredSlots_ — all documented "single-threaded core, D13"), its own WeightedRanker /
// DiversityReranker (const per call) and Orchestrator, and its own forked "load-thread-<t>" Rng.
// The const std::vector<Reel>& and std::vector<User>& are shared and READ-ONLY during the timed
// window (no simulation runs), so concurrent reads race with nothing. This mirrors
// FullRecommender's complete-initial-system wiring (src/recommendation/full_recommender.cpp)
// component-for-component; no production code is touched.
//
// SYNTHETIC WARM STATE. A real run personalizes each user over 200 interactions and warms every
// reel counter through simulation — far too expensive to stand up per benchmark cell. Instead we
// synthesize the recommender-visible state the request path actually reads, from forked rr::Rng
// streams (D8), and treat it as frozen. Reading HiddenUserState here is the same evaluation-side
// carve-out the harness uses (TDD 18.2); the recommender objects still never see hidden state, so
// D11 holds. Exactly which fields are populated and why is documented at synthesizeWarmState() —
// they are precisely the User fields (estimatedPreference via effectivePreference;
// sessionPreference for the session-topic feature; creatorAffinity for the creator source +
// feature; recentInteractions for duration-match + repetition; seenReels for the eligibility
// filter) and the Reel counters (impression/completion/like/share/skip for popularity; trending
// accumulators for trending; createdAt spread for freshness/fresh) that orchestrator.cpp + the six
// sources + feature_extractor.cpp read. Alignment of estimatedPreference to hiddenPreference is
// calibrated to cosine ~0.45 (Phase 7's published final est<->hidden alignment was 0.4245) and the
// achieved value is recorded in summary.json.
//
// cpu_utilization_pct = 100 * (Δuser_cpu + Δsys_cpu) / wall over the timed window, measured
// process-wide (rr::processCpuTimes) around the concurrent timed loop. It is a CORES-BUSY
// percentage and MAY EXCEED 100 (it approaches 100*threads at perfect scaling). peak_rss_mb is the
// process lifetime peak (rr::peakRssBytes), so one process invocation per (corpus, dim) cell-group
// keeps it meaningful — the load script drives exactly one process per cell-group.
//
// D2 containment holds: this file includes NO vector-db header; the index is reached only through
// rr::HNSWVectorIndex. D9: wall clock (rr::Stopwatch) only inside latency/CPU measurement.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <latch>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "rr/candidate_sources/creator_affinity_candidate_source.hpp"
#include "rr/candidate_sources/exploration_candidate_source.hpp"
#include "rr/candidate_sources/fresh_candidate_source.hpp"
#include "rr/candidate_sources/hnsw_candidate_source.hpp"
#include "rr/candidate_sources/popular_candidate_source.hpp"
#include "rr/candidate_sources/trending_candidate_source.hpp"
#include "rr/core/embedding.hpp"
#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/interaction.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/evaluation/metrics_collector.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/process_stats.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/candidate_generator.hpp"
#include "rr/recommendation/diversity_reranker.hpp"
#include "rr/recommendation/orchestrator.hpp"
#include "rr/recommendation/weighted_ranker.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/vindex/hnsw_vector_index.hpp"

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#elif defined(__linux__)
#include <sys/utsname.h>
#include <unistd.h>
#endif

// Repo/build facts injected by CMake (see apps/CMakeLists.txt).
#ifndef RR_REPO_DIR
#define RR_REPO_DIR "unknown"
#endif
#ifndef RR_VDB_DIR
#define RR_VDB_DIR "unknown"
#endif
#ifndef RR_BUILD_TYPE
#define RR_BUILD_TYPE "unknown"
#endif
#ifndef RR_COMPILER
#define RR_COMPILER "unknown"
#endif

namespace {

// The reel-generator creation window (src/simulation/reel_generator.cpp: createdAt is uniform over
// a fixed 30-day window of logical seconds). "now" for every request is placed one day past it so
// the freshness / trending / fresh windows are all non-degenerate (a ~6.7% recent slice qualifies
// as fresh; freshness scores span the full 31-day age range).
constexpr rr::Timestamp kCreationWindowSeconds = 30ULL * 24 * 60 * 60; // 2,592,000
constexpr rr::Timestamp kNow = kCreationWindowSeconds + 24ULL * 60 * 60;

// ---------------------------------------------------------------------------------------------
// Provenance / JSON helpers (same pattern as apps/benchmark_retrieval.cpp; percentiles come from
// rr::latencyStats so there is no duplicate percentile math here).
// ---------------------------------------------------------------------------------------------

std::string runCommand(const std::string &cmd) {
    std::string out;
    if (FILE *pipe = ::popen(cmd.c_str(), "r")) {
        char buf[512];
        while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
            out += buf;
        }
        ::pclose(pipe);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
        out.pop_back();
    }
    return out;
}

std::string jsonEscape(const std::string &s) {
    std::string r;
    r.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':
            r += "\\\"";
            break;
        case '\\':
            r += "\\\\";
            break;
        case '\n':
            r += "\\n";
            break;
        case '\r':
            r += "\\r";
            break;
        case '\t':
            r += "\\t";
            break;
        default:
            r += c;
            break;
        }
    }
    return r;
}

std::string nowStamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d-%H%M%S");
    return oss.str();
}

std::string nowIso() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

#if defined(__APPLE__)
std::string sysctlString(const char *name) {
    std::size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) {
        return "";
    }
    std::string buf(len, '\0');
    if (sysctlbyname(name, buf.data(), &len, nullptr, 0) != 0) {
        return "";
    }
    if (!buf.empty() && buf.back() == '\0') {
        buf.pop_back();
    }
    return buf;
}
std::uint64_t sysctlU64(const char *name) {
    std::uint64_t v = 0;
    std::size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, nullptr, 0) != 0) {
        return 0;
    }
    return v;
}
#endif

std::vector<std::size_t> parseThreadList(const std::string &s) {
    std::vector<std::size_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            out.push_back(static_cast<std::size_t>(std::stoul(tok)));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// Synthetic warm state (see the file header). Populates ONLY the recommender-visible fields the
// request path reads; alignment est<->hidden is calibrated to ~targetCos.
// ---------------------------------------------------------------------------------------------

// A unit vector that is `hidden` perturbed by per-component gaussian noise sized so that the
// expected cosine(noisy, hidden) ~= targetCos. For a d-dim unit `hidden`, E[cos] ~= 1/sqrt(1+sig^2
// d) so sig = sqrt((1/targetCos^2 - 1)/d).
rr::Embedding noisyUnit(const rr::Embedding &hidden, double sigma, rr::Rng &rng) {
    rr::Embedding v(hidden.size());
    for (std::size_t d = 0; d < hidden.size(); ++d) {
        v[d] = static_cast<float>(static_cast<double>(hidden[d]) + sigma * rng.gaussian());
    }
    rr::normalize(v);
    return v;
}

// Populate reel engagement + trending accumulators with a popularity-shaped (log-normal, long-tail)
// distribution; a ~15% recent slice gets nonzero trending accumulators dated near `kNow`.
void synthesizeReelWarmState(std::vector<rr::Reel> &reels, std::uint64_t seed,
                             double trendingHalfLifeSeconds) {
    rr::Rng rng = rr::forkRng(seed, "load-warm-reels");
    const auto twoHalfLives = static_cast<std::uint64_t>(2.0 * trendingHalfLifeSeconds);
    for (rr::Reel &r : reels) {
        // Log-normal impressions: median ~e^4.5 ~= 90, heavy right tail (a few reels go viral).
        const double impr = std::exp(rng.gaussian() * 1.3 + 4.5);
        const auto impressions = static_cast<std::uint64_t>(std::llround(impr));
        r.impressionCount = impressions;
        r.completionCount =
            static_cast<std::uint64_t>(std::llround(impr * rng.uniform(0.15, 0.55)));
        r.likeCount = static_cast<std::uint64_t>(std::llround(impr * rng.uniform(0.01, 0.12)));
        r.shareCount = static_cast<std::uint64_t>(std::llround(impr * rng.uniform(0.002, 0.02)));
        r.skipCount = static_cast<std::uint64_t>(std::llround(impr * rng.uniform(0.10, 0.45)));

        if (rng.bernoulli(0.15)) {
            const double tImpr = rng.uniform(5.0, 200.0);
            r.trendingImpressions = tImpr;
            r.trendingEngagement = tImpr * rng.uniform(0.5, 3.0);
            const std::uint64_t back = twoHalfLives > 0 ? rng.uniformInt(twoHalfLives) : 0;
            r.trendingUpdatedAt = kNow > back ? kNow - back : 0;
        }
    }
}

// Populate per-user recommender-visible state. Returns the mean achieved est<->hidden cosine over
// the population (reported in summary.json so the calibration is honest, not asserted).
double synthesizeUserWarmState(std::vector<rr::User> &users,
                               const std::vector<rr::HiddenUserState> &hidden,
                               const std::vector<rr::Reel> &reels, std::uint32_t creators,
                               std::size_t dim, std::uint64_t seed, std::size_t seenPerUser,
                               std::size_t affinityPerUser, std::size_t recentPerUser) {
    rr::Rng rng = rr::forkRng(seed, "load-warm-users");
    // sigma calibrated for E[cos] ~= 0.45 (estimated) / 0.45 (long-term) / 0.40 (session, noisier).
    const double sigmaEst = std::sqrt((1.0 / (0.45 * 0.45) - 1.0) / static_cast<double>(dim));
    const double sigmaLong = sigmaEst;
    const double sigmaSess = std::sqrt((1.0 / (0.40 * 0.40) - 1.0) / static_cast<double>(dim));

    double cosSum = 0.0;
    std::size_t cosCount = 0;
    const std::size_t n = users.size();
    for (std::size_t u = 0; u < n; ++u) {
        rr::User &user = users[u];
        rr::Embedding base;
        if (u < hidden.size() && hidden[u].hiddenPreference.size() == dim) {
            base = hidden[u].hiddenPreference;
        } else {
            base.assign(dim, 0.0f);
            for (std::size_t d = 0; d < dim; ++d) {
                base[d] = static_cast<float>(rng.gaussian());
            }
            rr::normalize(base);
        }
        user.estimatedPreference = noisyUnit(base, sigmaEst, rng);
        user.longTermPreference = noisyUnit(base, sigmaLong, rng);
        user.sessionPreference = noisyUnit(base, sigmaSess, rng);
        cosSum += static_cast<double>(rr::dot(user.estimatedPreference, base));
        ++cosCount;

        // seenReels: a random slice of the catalog the user has already been shown.
        user.seenReels.clear();
        const std::size_t seenTarget = std::min(seenPerUser, reels.size() / 4);
        for (std::size_t s = 0; s < seenTarget; ++s) {
            user.seenReels.insert(
                rr::ReelId{static_cast<std::uint32_t>(rng.uniformInt(reels.size()))});
        }

        // creatorAffinity: a few dozen liked creators (affinity in (0,1]).
        user.creatorAffinity.clear();
        const std::size_t affTarget = std::min<std::size_t>(affinityPerUser, creators);
        for (std::size_t a = 0; a < affTarget; ++a) {
            const auto c = static_cast<std::uint32_t>(rng.uniformInt(creators));
            user.creatorAffinity[rr::CreatorId{c}] = static_cast<float>(rng.uniform(0.15, 1.0));
        }

        // recentInteractions: a short recent history so duration-match + repetition are
        // non-trivial. ~60% completed/liked (duration-positive), timestamps clustered just before
        // kNow.
        user.recentInteractions.clear();
        const std::size_t recentTarget = std::min(recentPerUser, reels.size());
        for (std::size_t e = 0; e < recentTarget; ++e) {
            const auto ridx = static_cast<std::uint32_t>(rng.uniformInt(reels.size()));
            const rr::Reel &reel = reels[ridx];
            rr::InteractionEvent ev{};
            ev.userId = user.id;
            ev.reelId = reel.id;
            ev.creatorId = reel.creatorId;
            const double roll = rng.uniform01();
            ev.type = roll < 0.45   ? rr::InteractionType::CompleteWatch
                      : roll < 0.60 ? rr::InteractionType::Like
                      : roll < 0.85 ? rr::InteractionType::PartialWatch
                                    : rr::InteractionType::InstantSkip;
            ev.watchRatio = static_cast<float>(rng.uniform(0.1, 1.0));
            ev.watchSeconds = reel.durationSeconds * ev.watchRatio;
            ev.reward = 0.0f;
            const std::uint64_t back = static_cast<std::uint64_t>(recentTarget - e) *
                                       60ULL; // ~1 min apart, most recent last
            ev.timestamp = kNow > back ? kNow - back : 0;
            ev.sessionId = rr::SessionId{0};
            user.recentInteractions.push_back(ev);
        }

        user.totalInteractions = 200;
        user.currentSessionLength = static_cast<std::uint64_t>(rng.uniformInt(8) + 1);
    }
    return cosCount > 0 ? cosSum / static_cast<double>(cosCount) : 0.0;
}

// ---------------------------------------------------------------------------------------------
// Per-thread pipeline: replicates FullRecommender's complete-initial-system wiring exactly (source
// order hnsw, popular, trending, fresh, creator, exploration; guaranteed slots + reranker into the
// Orchestrator). Non-copyable (holds references); constructed in-place inside each worker thread.
// ---------------------------------------------------------------------------------------------
class ThreadPipeline {
  public:
    ThreadPipeline(const rr::HNSWVectorIndex &index, const std::vector<rr::Reel> &reels,
                   const rr::ExperimentConfig &cfg, rr::Rng threadRng)
        : rng_(std::move(threadRng)), // FIRST so it outlives exploration_ (which holds &rng_)
          hnsw_(index), popular_(reels, cfg.recommendation.popularCandidates),
          trending_(reels, cfg.recommendation.trendingCandidates,
                    cfg.ranking.trendingHalfLifeSeconds),
          creator_(reels, cfg.recommendation.creatorAffinityCandidates),
          fresh_(reels, cfg.recommendation.freshCandidates, cfg.exploration.freshWindowSeconds),
          exploration_(reels, cfg.exploration.epsilon, cfg.recommendation.explorationCandidates,
                       cfg.exploration.freshWindowSeconds, &rng_),
          ranker_(reels, cfg.ranking), reranker_(reels, cfg.diversity),
          orchestrator_({&hnsw_, &popular_, &trending_, &fresh_, &creator_, &exploration_}, reels,
                        &ranker_, &exploration_, cfg.exploration.guaranteedSlots, &reranker_) {}

    ThreadPipeline(const ThreadPipeline &) = delete;
    ThreadPipeline &operator=(const ThreadPipeline &) = delete;

    rr::RecommendationResponse recommend(const rr::User &user,
                                         const rr::RecommendationRequest &request) {
        return orchestrator_.recommend(user, request);
    }

  private:
    rr::Rng rng_;
    rr::HNSWCandidateSource hnsw_;
    rr::PopularCandidateSource popular_;
    rr::TrendingCandidateSource trending_;
    rr::CreatorAffinityCandidateSource creator_;
    rr::FreshCandidateSource fresh_;
    rr::ExplorationCandidateSource exploration_;
    rr::WeightedRanker ranker_;
    rr::DiversityReranker reranker_;
    rr::Orchestrator orchestrator_;
};

rr::RecommendationRequest makeRequest(const rr::User &user, std::size_t requestIndex,
                                      const rr::ExperimentConfig &cfg) {
    rr::RecommendationRequest req{};
    req.userId = user.id;
    req.sessionId = rr::SessionId{static_cast<std::uint32_t>(requestIndex)};
    req.feedSize = cfg.recommendation.feedSize;
    req.candidateLimit = cfg.recommendation.vectorCandidates;
    req.enableExploration = cfg.exploration.enabled;
    req.enableDiversity = cfg.diversity.enabled;
    req.requestTime = kNow;
    return req;
}

// Contiguous, disjoint user shard for thread `t` of `threads` (D13 "per-thread users"). Never
// empty: if there are fewer users than threads, threads wrap onto shared users (still a valid
// closed loop).
struct Shard {
    std::size_t begin;
    std::size_t size;
};
Shard shardFor(std::size_t t, std::size_t threads, std::size_t users) {
    if (users == 0) {
        return {0, 0};
    }
    const std::size_t chunk = users / threads;
    if (chunk == 0) {
        return {t % users, 1}; // more threads than users: one shared user each
    }
    const std::size_t begin = t * chunk;
    const std::size_t size = (t + 1 == threads) ? (users - begin) : chunk; // remainder to last
    return {begin, size};
}

// Per-thread timed samples.
struct ThreadSamples {
    std::vector<double> e2e;
    std::vector<double> retrieval;
    std::vector<double> ranking;
    std::vector<double> rerank;
    std::uint64_t feedChecksum = 0; // anti-DCE + a cheap cross-run content fingerprint
};

// One measured (threads) cell.
struct Row {
    std::size_t threads;
    std::size_t warmup;
    std::size_t timedPerThread;
    std::size_t totalTimed;
    double wallSeconds;
    double rps;
    rr::LatencyStats e2e;
    rr::LatencyStats retrieval;
    rr::LatencyStats ranking;
    rr::LatencyStats rerank;
    double cpuUtilPct;
};

void runWorker(std::size_t t, std::size_t threads, const rr::HNSWVectorIndex &index,
               const std::vector<rr::Reel> &reels, const std::vector<rr::User> &users,
               const rr::ExperimentConfig &cfg, std::uint64_t seed, std::size_t warmup,
               std::size_t timed, std::latch &warmupDone, std::atomic<bool> &go,
               ThreadSamples &out) {
    ThreadPipeline pipe(index, reels, cfg, rr::forkRng(seed, "load-thread-" + std::to_string(t)));
    const Shard shard = shardFor(t, threads, users.size());

    // Warmup (untimed): fills allocator arenas / branch predictors before the timed window.
    for (std::size_t w = 0; w < warmup && shard.size > 0; ++w) {
        const rr::User &user = users[shard.begin + (w % shard.size)];
        volatile auto resp = pipe.recommend(user, makeRequest(user, w, cfg));
        (void)resp;
    }

    warmupDone.count_down();
    go.wait(false); // released once every thread has warmed up (see main)

    out.e2e.reserve(timed);
    out.retrieval.reserve(timed);
    out.ranking.reserve(timed);
    out.rerank.reserve(timed);
    for (std::size_t i = 0; i < timed && shard.size > 0; ++i) {
        const rr::User &user = users[shard.begin + (i % shard.size)];
        const rr::RecommendationRequest req = makeRequest(user, i, cfg);
        rr::Stopwatch sw;
        rr::RecommendationResponse resp = pipe.recommend(user, req); // D9: only this is timed
        out.e2e.push_back(sw.elapsedMs());
        out.retrieval.push_back(resp.retrievalLatencyMs);
        out.ranking.push_back(resp.rankingLatencyMs);
        out.rerank.push_back(resp.rerankingLatencyMs);
        if (!resp.reels.empty()) {
            out.feedChecksum += resp.reels.front().reelId.value + resp.reels.size();
        }
    }
}

Row runCell(std::size_t threads, const rr::HNSWVectorIndex &index,
            const std::vector<rr::Reel> &reels, const std::vector<rr::User> &users,
            const rr::ExperimentConfig &cfg, std::uint64_t seed, std::size_t warmup,
            std::size_t timed) {
    std::vector<ThreadSamples> perThread(threads);
    std::latch warmupDone(static_cast<std::ptrdiff_t>(threads));
    std::atomic<bool> go{false};

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
        pool.emplace_back(runWorker, t, threads, std::cref(index), std::cref(reels),
                          std::cref(users), std::cref(cfg), seed, warmup, timed,
                          std::ref(warmupDone), std::ref(go), std::ref(perThread[t]));
    }

    // Barrier on warmup completion, THEN start the timed window + CPU sampling together.
    warmupDone.wait();
    const rr::CpuTimes cpu0 = rr::processCpuTimes();
    rr::Stopwatch wall;
    go.store(true);
    go.notify_all();
    for (std::thread &th : pool) {
        th.join();
    }
    const double wallMs = wall.elapsedMs();
    const rr::CpuTimes cpu1 = rr::processCpuTimes();

    // Merge samples.
    std::vector<double> e2e, retrieval, ranking, rerank;
    std::uint64_t sink = 0;
    for (const ThreadSamples &s : perThread) {
        e2e.insert(e2e.end(), s.e2e.begin(), s.e2e.end());
        retrieval.insert(retrieval.end(), s.retrieval.begin(), s.retrieval.end());
        ranking.insert(ranking.end(), s.ranking.begin(), s.ranking.end());
        rerank.insert(rerank.end(), s.rerank.begin(), s.rerank.end());
        sink += s.feedChecksum;
    }
    (void)sink;

    Row row{};
    row.threads = threads;
    row.warmup = warmup;
    row.timedPerThread = timed;
    row.totalTimed = e2e.size();
    row.wallSeconds = wallMs / 1000.0;
    row.rps = row.wallSeconds > 0.0 ? static_cast<double>(row.totalTimed) / row.wallSeconds : 0.0;
    row.e2e = rr::latencyStats(e2e);
    row.retrieval = rr::latencyStats(retrieval);
    row.ranking = rr::latencyStats(ranking);
    row.rerank = rr::latencyStats(rerank);
    const double cpuSec =
        (cpu1.userSeconds - cpu0.userSeconds) + (cpu1.systemSeconds - cpu0.systemSeconds);
    row.cpuUtilPct = row.wallSeconds > 0.0 ? 100.0 * cpuSec / row.wallSeconds : 0.0;
    return row;
}

// --- smoke correctness checks (--smoke only). Returns false on the FIRST violation. ---
bool smokeChecks(const rr::HNSWVectorIndex &index, const std::vector<rr::Reel> &reels,
                 const std::vector<rr::User> &users, const rr::ExperimentConfig &cfg,
                 std::uint64_t seed) {
    const std::size_t feedSize = cfg.recommendation.feedSize;
    // Run a handful of requests on one pipeline and validate every invariant.
    ThreadPipeline pipe(index, reels, cfg, rr::forkRng(seed, "load-thread-0"));
    for (std::size_t i = 0; i < 40 && i < users.size(); ++i) {
        const rr::User &user = users[i];
        const rr::RecommendationResponse resp = pipe.recommend(user, makeRequest(user, i, cfg));
        if (resp.reels.empty()) {
            std::cerr << "SMOKE FAIL: empty feed for user " << i << "\n";
            return false;
        }
        if (resp.reels.size() > feedSize) {
            std::cerr << "SMOKE FAIL: feed exceeds feedSize\n";
            return false;
        }
        if (!(resp.retrievalLatencyMs >= 0.0 && resp.rankingLatencyMs >= 0.0 &&
              resp.rerankingLatencyMs >= 0.0 && resp.totalLatencyMs >= 0.0)) {
            std::cerr << "SMOKE FAIL: per-stage latency not populated\n";
            return false;
        }
        std::unordered_set<std::uint32_t> seenInFeed;
        for (const rr::RankedReel &rr : resp.reels) {
            if (!seenInFeed.insert(rr.reelId.value).second) {
                std::cerr << "SMOKE FAIL: duplicate reel within feed\n";
                return false;
            }
            if (user.seenReels.contains(rr.reelId)) {
                std::cerr << "SMOKE FAIL: already-seen reel served\n";
                return false;
            }
        }
    }
    return true;
}

// Collect feed id sequences per thread for a determinism cross-check (content identical across two
// same-seed runs; timing differs). Rebuilds pipelines fresh each call.
std::vector<std::vector<std::vector<std::uint32_t>>>
collectFeeds(const rr::HNSWVectorIndex &index, const std::vector<rr::Reel> &reels,
             const std::vector<rr::User> &users, const rr::ExperimentConfig &cfg,
             std::uint64_t seed, std::size_t threads, std::size_t nReq) {
    std::vector<std::vector<std::vector<std::uint32_t>>> feeds(threads);
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            ThreadPipeline pipe(index, reels, cfg,
                                rr::forkRng(seed, "load-thread-" + std::to_string(t)));
            const Shard shard = shardFor(t, threads, users.size());
            for (std::size_t i = 0; i < nReq && shard.size > 0; ++i) {
                const rr::User &user = users[shard.begin + (i % shard.size)];
                const rr::RecommendationResponse resp =
                    pipe.recommend(user, makeRequest(user, i, cfg));
                std::vector<std::uint32_t> ids;
                ids.reserve(resp.reels.size());
                for (const rr::RankedReel &r : resp.reels) {
                    ids.push_back(r.reelId.value);
                }
                feeds[t].push_back(std::move(ids));
            }
        });
    }
    for (std::thread &th : pool) {
        th.join();
    }
    return feeds;
}

void writeMetadata(const std::filesystem::path &outDir, std::size_t reels, std::size_t usersN,
                   std::size_t dim, const std::vector<std::size_t> &threadList,
                   const std::string &algorithm) {
    const std::string repoDir = RR_REPO_DIR;
    const std::string vdbDir = RR_VDB_DIR;
    const std::string rrSha = runCommand("git -C '" + repoDir + "' rev-parse HEAD 2>/dev/null");
    const std::string rrDirtyRaw =
        runCommand("git -C '" + repoDir + "' status --porcelain 2>/dev/null");
    const bool rrDirty = !rrDirtyRaw.empty();
    const std::string vdbSha = runCommand("git -C '" + vdbDir + "' rev-parse HEAD 2>/dev/null");

#if defined(__APPLE__)
    const std::string cpu = sysctlString("machdep.cpu.brand_string");
    const std::uint64_t ramBytes = sysctlU64("hw.memsize");
    const std::uint64_t logicalCores = sysctlU64("hw.ncpu");
    const std::uint64_t physicalCores = sysctlU64("hw.physicalcpu");
#else
    const std::string cpu = "unknown";
    const std::uint64_t ramBytes = 0;
    const std::uint64_t logicalCores = 0;
    const std::uint64_t physicalCores = 0;
#endif
    std::string osSys, osRel, osVer, osMach;
#if defined(__APPLE__) || defined(__linux__)
    struct utsname uts{};
    if (uname(&uts) == 0) {
        osSys = uts.sysname;
        osRel = uts.release;
        osVer = uts.version;
        osMach = uts.machine;
    }
#endif
    std::ofstream j(outDir / "metadata.json");
    j << "{\n";
    j << "  \"generated_at\": \"" << nowIso() << "\",\n";
    j << "  \"tool\": \"benchmark_recommender (ReelRank Phase 11)\",\n";
    j << "  \"algorithm\": \"" << jsonEscape(algorithm) << "\",\n";
    j << "  \"git\": {\n";
    j << "    \"reel_rank_sha\": \"" << jsonEscape(rrSha) << "\",\n";
    j << "    \"reel_rank_dirty\": " << (rrDirty ? "true" : "false") << ",\n";
    j << "    \"vector_db_sha\": \"" << jsonEscape(vdbSha) << "\",\n";
    j << "    \"vector_db_dir\": \"" << jsonEscape(vdbDir) << "\"\n";
    j << "  },\n";
    j << "  \"build\": {\n";
    j << "    \"type\": \"" << jsonEscape(RR_BUILD_TYPE) << "\",\n";
    j << "    \"compiler\": \"" << jsonEscape(RR_COMPILER) << "\",\n";
    j << "    \"cxx_standard\": 20\n";
    j << "  },\n";
    j << "  \"hardware\": {\n";
    j << "    \"cpu_model\": \"" << jsonEscape(cpu) << "\",\n";
    j << "    \"logical_cores\": " << logicalCores << ",\n";
    j << "    \"physical_cores\": " << physicalCores << ",\n";
    j << "    \"ram_bytes\": " << ramBytes << ",\n";
    j << "    \"ram_gib\": " << std::fixed << std::setprecision(1)
      << (static_cast<double>(ramBytes) / (1024.0 * 1024.0 * 1024.0)) << "\n";
    j << "  },\n";
    j << "  \"os\": {\n";
    j << "    \"sysname\": \"" << jsonEscape(osSys) << "\",\n";
    j << "    \"release\": \"" << jsonEscape(osRel) << "\",\n";
    j << "    \"version\": \"" << jsonEscape(osVer) << "\",\n";
    j << "    \"machine\": \"" << jsonEscape(osMach) << "\"\n";
    j << "  },\n";
    j << "  \"dataset\": {\n";
    j << "    \"reels\": " << reels << ",\n";
    j << "    \"users\": " << usersN << ",\n";
    j << "    \"dimensions\": " << dim << "\n";
    j << "  },\n";
    j << "  \"thread_counts\": [";
    for (std::size_t i = 0; i < threadList.size(); ++i) {
        if (i) {
            j << ',';
        }
        j << threadList[i];
    }
    j << "]\n";
    j << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    // ---- args -----------------------------------------------------------------------------------
    std::size_t reelsN = 100000;
    std::size_t usersN = 10000;
    std::size_t dim = 64;
    std::vector<std::size_t> threadList = {1, 2, 4, 8};
    std::size_t requestsPerThread = 2000;
    std::size_t warmup = 200;
    std::uint64_t seed = 42;
    std::string outRoot = "results/phase11/load";
    bool smoke = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << what << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--reels") {
            reelsN = std::stoull(next("--reels"));
        } else if (a == "--users") {
            usersN = std::stoull(next("--users"));
        } else if (a == "--dim") {
            dim = std::stoull(next("--dim"));
        } else if (a == "--threads") {
            threadList = parseThreadList(next("--threads"));
        } else if (a == "--requests-per-thread") {
            requestsPerThread = std::stoull(next("--requests-per-thread"));
        } else if (a == "--warmup") {
            warmup = std::stoull(next("--warmup"));
        } else if (a == "--seed") {
            seed = std::stoull(next("--seed"));
        } else if (a == "--out") {
            outRoot = next("--out");
        } else if (a == "--smoke") {
            smoke = true;
        } else if (a == "--help" || a == "-h") {
            std::cout << "usage: benchmark_recommender [--reels N] [--users N] [--dim D] "
                         "[--threads L] [--requests-per-thread N] [--warmup N] [--seed S] "
                         "[--out DIR] [--smoke]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }
    if (smoke) {
        reelsN = 2000;
        usersN = 200;
        dim = 64;
        threadList = {1, 2, 4};
        requestsPerThread = 60;
        warmup = 10;
    }
    if (threadList.empty()) {
        threadList = {1};
    }

    // ---- resolve the full config (medium.json values as base; TDD 27 candidate counts) ----------
    rr::ExperimentConfig cfg;
    cfg.algorithm = rr::RecommendationAlgorithm::HnswRankerDiversity;
    cfg.simulation.reels = static_cast<std::uint32_t>(reelsN);
    cfg.simulation.users = static_cast<std::uint32_t>(usersN);
    cfg.simulation.dimensions = static_cast<std::uint32_t>(dim);
    cfg.simulation.seed = seed;
    // medium.json: hnsw m16 efc200 efs64; feed 10; popular/fresh/trending/creator 100; exploration
    // 50; epsilon 0.05; guaranteed slots 2; diversity max_per_creator 2 / max_per_topic 3 / mmr
    // 0.75. (All are the struct defaults, so only the corpus-dependent candidate budget is
    // overridden.)
    cfg.recommendation.feedSize = 10;
    cfg.recommendation.vectorCandidates = (reelsN <= 10000) ? 200u : 500u; // TDD 27
    cfg.exploration.enabled = true;
    cfg.diversity.enabled = true;

    const std::string algorithm = rr::toString(cfg.algorithm);
    const std::string expId = "load-seed" + std::to_string(seed) + "-r" + std::to_string(reelsN) +
                              "-d" + std::to_string(dim) + "-" + nowStamp();
    const std::filesystem::path outDir = std::filesystem::path(outRoot) / expId;
    std::filesystem::create_directories(outDir);

    std::cerr << "benchmark_recommender: reels=" << reelsN << " users=" << usersN << " dim=" << dim
              << " candidateLimit=" << cfg.recommendation.vectorCandidates << " threads=" <<
        [&] {
            std::string s;
            for (std::size_t i = 0; i < threadList.size(); ++i) {
                if (i)
                    s += ',';
                s += std::to_string(threadList[i]);
            }
            return s;
        }()
              << " req/thread=" << requestsPerThread << (smoke ? "  [SMOKE]\n" : "\n")
              << "  out=" << outDir.string() << "\n";

    // ---- dataset (real Phase 2 generators: topic-clustered reels) + synthetic warm state
    // ---------
    std::cerr << "  generating dataset ..." << std::endl;
    rr::GeneratedDataset ds = rr::generateDataset(cfg.simulation, seed);
    std::cerr << "  synthesizing warm state ..." << std::endl;
    synthesizeReelWarmState(ds.reels, seed, cfg.ranking.trendingHalfLifeSeconds);
    const std::size_t seenPerUser = smoke ? 20 : 100;
    const std::size_t affinityPerUser = 30;
    const std::size_t recentPerUser = std::min<std::size_t>(20, cfg.learning.recentWindow);
    const double meanEstCos =
        synthesizeUserWarmState(ds.users, ds.hiddenStates, ds.reels, cfg.simulation.creators, dim,
                                seed, seenPerUser, affinityPerUser, recentPerUser);
    std::cerr << "  est<->hidden mean cosine = " << std::fixed << std::setprecision(4) << meanEstCos
              << "\n";

    // ---- build the ONE shared frozen index ------------------------------------------------------
    std::cerr << "  building shared HNSW index ..." << std::endl;
    const std::uint64_t indexSeed = rr::splitmix64(seed ^ rr::fnv1a64("load-index"));
    rr::HNSWVectorIndex index(dim, cfg.hnsw, indexSeed);
    std::size_t activeReels = 0;
    rr::Stopwatch buildTimer;
    for (const rr::Reel &r : ds.reels) {
        if (r.active) {
            index.insert(r.id, r.embedding);
            ++activeReels;
        }
    }
    const double indexBuildSeconds = buildTimer.elapsedMs() / 1000.0;
    const double insertPerSec =
        indexBuildSeconds > 0.0 ? static_cast<double>(activeReels) / indexBuildSeconds : 0.0;
    std::cerr << "  index built: " << activeReels << " reels in " << std::setprecision(2)
              << indexBuildSeconds << " s (" << static_cast<long long>(insertPerSec)
              << " inserts/s). FROZEN.\n";
    // The index is READ-ONLY from here: no insert / setEfSearch while threads search (D13).

    // ---- smoke: correctness + determinism cross-check -------------------------------------------
    if (smoke) {
        std::cerr << "  running smoke correctness checks ..." << std::endl;
        if (!smokeChecks(index, ds.reels, ds.users, cfg, seed)) {
            return 1;
        }
        const auto feedsA = collectFeeds(index, ds.reels, ds.users, cfg, seed, 4, 25);
        const auto feedsB = collectFeeds(index, ds.reels, ds.users, cfg, seed, 4, 25);
        if (feedsA != feedsB) {
            std::cerr << "SMOKE FAIL: feed contents differ across two same-seed runs "
                         "(non-deterministic)\n";
            return 1;
        }
        std::cerr << "  smoke checks PASS (feeds well-formed, no dups, no seen served, per-stage "
                     "latency populated, content-deterministic across runs).\n";
    }

    // ---- sweep the thread list ------------------------------------------------------------------
    std::vector<Row> rows;
    for (std::size_t threads : threadList) {
        std::cerr << "  [T=" << threads << "] running " << requestsPerThread
                  << " timed req/thread ..." << std::endl;
        Row row = runCell(threads, index, ds.reels, ds.users, cfg, seed, warmup, requestsPerThread);
        rows.push_back(row);
        std::cerr << "    RPS=" << std::fixed << std::setprecision(0) << row.rps
                  << "  e2e p50/p95/p99=" << std::setprecision(3) << row.e2e.p50Ms << "/"
                  << row.e2e.p95Ms << "/" << row.e2e.p99Ms << " ms"
                  << "  cpu=" << std::setprecision(0) << row.cpuUtilPct << "%\n";
    }

    const double peakRssMb = static_cast<double>(rr::peakRssBytes()) / (1024.0 * 1024.0);

    // ---- load_metrics.csv (FROZEN schema) -------------------------------------------------------
    {
        std::ofstream csv(outDir / "load_metrics.csv");
        csv << "corpus_reels,users,dimensions,algorithm,threads,warmup_requests_per_thread,"
               "timed_requests_per_thread,total_timed_requests,wall_seconds,rps,e2e_p50_ms,"
               "e2e_p95_ms,e2e_p99_ms,e2e_mean_ms,e2e_max_ms,retrieval_p50_ms,retrieval_p95_ms,"
               "retrieval_p99_ms,ranking_p50_ms,ranking_p95_ms,ranking_p99_ms,rerank_p50_ms,"
               "rerank_p95_ms,rerank_p99_ms,cpu_utilization_pct,peak_rss_mb,index_build_seconds,"
               "index_insert_per_sec\n";
        csv << std::fixed;
        for (const Row &r : rows) {
            csv << reelsN << ',' << usersN << ',' << dim << ',' << algorithm << ',' << r.threads
                << ',' << r.warmup << ',' << r.timedPerThread << ',' << r.totalTimed << ','
                << std::setprecision(4) << r.wallSeconds << ',' << std::setprecision(1) << r.rps
                << ',' << std::setprecision(4) << r.e2e.p50Ms << ',' << r.e2e.p95Ms << ','
                << r.e2e.p99Ms << ',' << r.e2e.meanMs << ',' << r.e2e.maxMs << ','
                << r.retrieval.p50Ms << ',' << r.retrieval.p95Ms << ',' << r.retrieval.p99Ms << ','
                << r.ranking.p50Ms << ',' << r.ranking.p95Ms << ',' << r.ranking.p99Ms << ','
                << r.rerank.p50Ms << ',' << r.rerank.p95Ms << ',' << r.rerank.p99Ms << ','
                << std::setprecision(1) << r.cpuUtilPct << ',' << std::setprecision(2) << peakRssMb
                << ',' << std::setprecision(3) << indexBuildSeconds << ',' << std::setprecision(1)
                << insertPerSec << '\n';
        }
    }

    // ---- config.json (fully resolved) -----------------------------------------------------------
    {
        std::ofstream j(outDir / "config.json");
        j << "{\n";
        j << "  \"experiment_id\": \"" << expId << "\",\n";
        j << "  \"algorithm\": \"" << algorithm << "\",\n";
        j << "  \"seed\": " << seed << ",\n";
        j << "  \"reels\": " << reelsN << ",\n";
        j << "  \"users\": " << usersN << ",\n";
        j << "  \"dimensions\": " << dim << ",\n";
        j << "  \"feed_size\": " << cfg.recommendation.feedSize << ",\n";
        j << "  \"candidate_limit\": " << cfg.recommendation.vectorCandidates << ",\n";
        j << "  \"popular_candidates\": " << cfg.recommendation.popularCandidates << ",\n";
        j << "  \"trending_candidates\": " << cfg.recommendation.trendingCandidates << ",\n";
        j << "  \"fresh_candidates\": " << cfg.recommendation.freshCandidates << ",\n";
        j << "  \"creator_affinity_candidates\": " << cfg.recommendation.creatorAffinityCandidates
          << ",\n";
        j << "  \"exploration_candidates\": " << cfg.recommendation.explorationCandidates << ",\n";
        j << "  \"hnsw\": {\"m\": " << cfg.hnsw.m
          << ", \"ef_construction\": " << cfg.hnsw.efConstruction
          << ", \"ef_search\": " << cfg.hnsw.efSearch << "},\n";
        j << "  \"epsilon\": " << cfg.exploration.epsilon << ",\n";
        j << "  \"guaranteed_slots\": " << cfg.exploration.guaranteedSlots << ",\n";
        j << "  \"diversity\": {\"enabled\": " << (cfg.diversity.enabled ? "true" : "false")
          << ", \"max_per_creator\": " << cfg.diversity.maxPerCreator
          << ", \"max_per_topic\": " << cfg.diversity.maxPerTopic
          << ", \"mmr_lambda\": " << cfg.diversity.mmrLambda << "},\n";
        j << "  \"warmup_requests_per_thread\": " << warmup << ",\n";
        j << "  \"timed_requests_per_thread\": " << requestsPerThread << ",\n";
        j << "  \"thread_counts\": [";
        for (std::size_t i = 0; i < threadList.size(); ++i) {
            if (i)
                j << ',';
            j << threadList[i];
        }
        j << "]\n}\n";
    }

    // ---- summary.json (headline aggregates + warm-state provenance) -----------------------------
    {
        std::ofstream j(outDir / "summary.json");
        j << std::fixed;
        j << "{\n";
        j << "  \"experiment_id\": \"" << expId << "\",\n";
        j << "  \"algorithm\": \"" << algorithm << "\",\n";
        j << "  \"reels\": " << reelsN << ",\n";
        j << "  \"users\": " << usersN << ",\n";
        j << "  \"dimensions\": " << dim << ",\n";
        j << "  \"active_reels_indexed\": " << activeReels << ",\n";
        j << "  \"index_build_seconds\": " << std::setprecision(3) << indexBuildSeconds << ",\n";
        j << "  \"index_insert_per_sec\": " << std::setprecision(1) << insertPerSec << ",\n";
        j << "  \"peak_rss_mb\": " << std::setprecision(2) << peakRssMb << ",\n";
        j << "  \"warm_state\": {\n";
        j << "    \"est_hidden_mean_cosine\": " << std::setprecision(4) << meanEstCos << ",\n";
        j << "    \"est_hidden_target_cosine\": 0.45,\n";
        j << "    \"seen_reels_per_user\": " << seenPerUser << ",\n";
        j << "    \"creator_affinity_per_user\": " << affinityPerUser << ",\n";
        j << "    \"recent_interactions_per_user\": " << recentPerUser << ",\n";
        j << "    \"note\": \"synthetic warm state; recommender never reads HiddenUserState "
             "(D11)\"\n";
        j << "  },\n";
        j << "  \"cells\": [\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const Row &r = rows[i];
            j << "    {\"threads\": " << r.threads << ", \"rps\": " << std::setprecision(1) << r.rps
              << ", \"e2e_p50_ms\": " << std::setprecision(4) << r.e2e.p50Ms
              << ", \"e2e_p95_ms\": " << r.e2e.p95Ms << ", \"e2e_p99_ms\": " << r.e2e.p99Ms
              << ", \"retrieval_p95_ms\": " << r.retrieval.p95Ms
              << ", \"ranking_p95_ms\": " << r.ranking.p95Ms
              << ", \"rerank_p95_ms\": " << r.rerank.p95Ms
              << ", \"cpu_utilization_pct\": " << std::setprecision(1) << r.cpuUtilPct << "}"
              << (i + 1 < rows.size() ? "," : "") << "\n";
        }
        j << "  ]\n}\n";
    }

    writeMetadata(outDir, reelsN, usersN, dim, threadList, algorithm);

    std::cerr << "benchmark_recommender: DONE; wrote " << rows.size() << " cells to "
              << outDir.string() << "\n";
    std::cout << "RESULT_DIR=" << outDir.string() << "\n";
    return 0;
}

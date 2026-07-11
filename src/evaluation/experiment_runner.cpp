#include "rr/evaluation/experiment_runner.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/evaluation/cold_start.hpp"
#include "rr/evaluation/oracle.hpp"
#include "rr/evaluation/results_writer.hpp"
#include "rr/evaluation/retrieval_evaluator.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/learning/online_user_state_updater.hpp"
#include "rr/recommendation/effective_preference.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/recommendation/vector_index.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/simulator.hpp"

namespace rr {

namespace {

// yyyymmddThhmmss experiment-id stamp (D12). Wall-clock is permitted here: this labels the output
// directory for provenance and never feeds the simulation (D9).
std::string experimentTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%S", &tm);
    return buf;
}

} // namespace

ExperimentRunner::ExperimentRunner(ExperimentConfig config, std::filesystem::path outputRoot,
                                   BuildProvenance provenance)
    : config_(std::move(config)), outputRoot_(std::move(outputRoot)),
      provenance_(std::move(provenance)) {}

ExperimentResult ExperimentRunner::run() {
    Stopwatch wall; // total wall time (D9: provenance only, confined to summary.timing).
    const uint64_t seed = config_.simulation.seed;

    // 1. Dataset + cold-start prior (TDD 11.1): every user's three preference vectors start at the
    //    global-average hidden preference. From here they either stay frozen (learning disabled) or
    //    update online after every interaction (Phase 7).
    GeneratedDataset ds = generateDataset(config_.simulation, seed);
    applyColdStart(ds.users, globalAveragePreference(ds.hiddenStates));

    // Online preference learning (Phase 7, TDD 8.3/11.2/11.3). Constructed ONCE over the immutable
    // reel catalog; apply() runs after each Simulator::step below. It consumes no rng/clock, so
    // invoking it is stream-neutral (D8) - the recommender/behaviour/oracle streams are untouched.
    // When config.learning.enabled is false the updater is never invoked and estimates stay frozen.
    const OnlineUserStateUpdater updater(ds.reels, config_.learning);
    const bool learningEnabled = config_.learning.enabled;

    // 2. Simulator, recommender, and oracle each on an INDEPENDENT named rng stream (D8) so adding
    //    the oracle never perturbs the behaviour or recommender streams.
    Simulator sim(config_.behaviour, config_.reward, forkRng(seed, "behaviour"),
                  config_.learning.recentWindow, config_.ranking.trendingHalfLifeSeconds);
    RecommenderDeps deps{ds.reels, ds.users, config_};
    std::unique_ptr<Recommender> recommender =
        makeRecommender(config_.algorithm, deps, forkRng(seed, "recommender"));
    Rng oracleRng = forkRng(seed, "oracle");

    // Live retrieval evaluation (TDD 18.1). Its own INDEPENDENT rng stream (D8) so adding it never
    // perturbs behaviour/recommender/oracle. Bernoulli(retrievalSampleRate) is drawn for EVERY
    // request below (like the oracle) to keep the stream aligned across runs; the exact
    // ground-truth evaluator is built lazily only when the recommender is vector-based AND the rate
    // is positive.
    Rng retrievalRng = forkRng(seed, "retrieval");
    const VectorIndex *annIndex = recommender->retrievalIndex();
    const bool retrievalApplicable = annIndex != nullptr;
    std::optional<RetrievalEvaluator> retrievalEval;
    if (retrievalApplicable && config_.evaluation.retrievalSampleRate > 0.0) {
        retrievalEval.emplace(config_.simulation.dimensions, ds.reels);
    }

    // 3. Interleaved loop: ceil(interactionsPerUser / feedSize) rounds, each a request per user.
    const size_t feedSize = config_.recommendation.feedSize;
    const size_t interactionsPerUser = config_.simulation.interactionsPerUser;
    const size_t rounds =
        feedSize > 0 ? (interactionsPerUser + feedSize - 1) / feedSize : 0; // ceil

    MetricsCollector metrics;
    struct RegretAcc {
        size_t sampled = 0;
        double sum = 0.0;
    };
    std::vector<RegretAcc> regretByRound(rounds);
    struct RetrievalAcc {
        size_t samples = 0;
        double recall10Sum = 0.0;
        double recall50Sum = 0.0;
        double distErrorSum = 0.0;
    };
    std::vector<RetrievalAcc> retrievalByRound(rounds);
    // Estimate<->hidden alignment (TDD 18.5), one mean per round measured after the round
    // completes.
    std::vector<double> alignmentByRound(rounds, 0.0);

    std::vector<double> latencies;
    std::vector<double> retrievalLatencies;
    std::vector<double> rankingLatencies;
    std::vector<double> rerankingLatencies;
    const size_t latencyReserve = rounds * ds.users.size();
    latencies.reserve(latencyReserve);
    retrievalLatencies.reserve(latencyReserve);
    rankingLatencies.reserve(latencyReserve);
    rerankingLatencies.reserve(latencyReserve);

    std::vector<size_t> doneByUser(ds.users.size(), 0);
    size_t requestCount = 0;
    size_t impressionCount = 0;

    for (size_t round = 0; round < rounds; ++round) {
        for (size_t u = 0; u < ds.users.size(); ++u) {
            User &user = ds.users[u];
            const size_t remaining = interactionsPerUser - doneByUser[u];
            if (remaining == 0) {
                continue; // budget exhausted (defensive; ceil rounds never hit this at round start)
            }

            // Build the request. Baselines: no exploration / no diversity (documented in output).
            RecommendationRequest req{};
            req.userId = user.id;
            req.sessionId = user.recentInteractions.empty()
                                ? SessionId{0}
                                : user.recentInteractions.back().sessionId;
            req.feedSize = feedSize;
            req.candidateLimit = config_.recommendation.vectorCandidates;
            req.enableExploration = false;
            req.enableDiversity = false;
            req.requestTime = sim.now();

            // Time ONLY the recommend() call (D9 wall clock only here).
            Stopwatch sw;
            RecommendationResponse resp = recommender->recommend(req);
            latencies.push_back(sw.elapsedMs());
            // Per-stage latencies come from the response itself (0 for recommenders that do not
            // populate them, e.g. Random/Popularity). These feed the TDD 18.7 stage percentiles.
            retrievalLatencies.push_back(resp.retrievalLatencyMs);
            rankingLatencies.push_back(resp.rankingLatencyMs);
            rerankingLatencies.push_back(resp.rerankingLatencyMs);
            ++requestCount;

            // Oracle sampling: draw for EVERY request so the oracle stream stays aligned across
            // runs regardless of the outcome (D8).
            const bool sampled = oracleRng.bernoulli(config_.evaluation.oracleSampleRate);
            if (sampled) {
                std::vector<ReelId> feedIds;
                feedIds.reserve(resp.reels.size());
                for (const RankedReel &ranked : resp.reels) {
                    feedIds.push_back(ranked.reelId);
                }
                // seen-set snapshot is the pre-feed state: the oracle scores what was available
                // BEFORE this feed is consumed.
                const OracleResult oracle =
                    computeOracleRegret(ds.hiddenStates[u].hiddenPreference, ds.reels,
                                        user.seenReels, feedIds, feedSize);
                regretByRound[round].sampled += 1;
                regretByRound[round].sum += oracle.regret;
            }

            // Retrieval sampling (TDD 18.1): draw for EVERY request (independent stream) so it
            // stays aligned across runs; evaluate only when the recommender is vector-based. The
            // evaluator is deterministic and consumes no simulation rng, so it cannot perturb any
            // stream.
            const bool retrievalSampled =
                retrievalRng.bernoulli(config_.evaluation.retrievalSampleRate);
            if (retrievalSampled && retrievalEval.has_value()) {
                const RetrievalSample rs =
                    retrievalEval->evaluate(*annIndex, effectivePreference(user));
                RetrievalAcc &acc = retrievalByRound[round];
                acc.samples += 1;
                acc.recall10Sum += rs.recallAt10;
                acc.recall50Sum += rs.recallAt50;
                acc.distErrorSum += rs.distanceError;
            }

            // Consume the feed in order, capped so per-user interactions never exceed the budget.
            const size_t toConsume = std::min(remaining, resp.reels.size());
            for (size_t k = 0; k < toConsume; ++k) {
                const ReelId reelId = resp.reels[k].reelId;
                Reel &reel = ds.reels[reelId.value];
                const Creator &creator = ds.creators[reel.creatorId.value];
                const StepResult step = sim.step(user, ds.hiddenStates[u], reel, creator);

                // Online preference update (Phase 7): runs AFTER Simulator::step has recorded the
                // interaction into user.recentInteractions, updating ONLY the three preference
                // vectors (long-term / session / cached estimate). Gated by learning.enabled so the
                // frozen arm keeps the cold-start estimates. Stream-neutral (no rng/clock).
                if (learningEnabled) {
                    updater.apply(user, reel, step.event);
                }

                ImpressionSample s;
                s.watchRatio = step.event.watchRatio;
                s.watchSeconds = step.event.watchSeconds;
                s.instantSkip = step.outcome.instantSkip;
                s.completed = step.outcome.completed;
                s.liked = step.outcome.liked;
                s.shared = step.outcome.shared;
                s.followed = step.outcome.followed;
                s.reward = step.event.reward;
                // Evaluation-only hidden-state read (TDD 18.2): true affinity of the shown reel.
                s.trueAffinity = dot(ds.hiddenStates[u].hiddenPreference, reel.embedding);
                s.userId = user.id.value;
                s.sessionId = step.event.sessionId.value;
                metrics.add(round, s);
                ++impressionCount;
            }
            doneByUser[u] += toConsume;
        }

        // End-of-round estimate<->hidden alignment (TDD 18.5): mean over ALL users of
        // cos(estimatedPreference, hiddenPreference). Both are unit-length, so cosine == dot.
        // Evaluation-only hidden-state read (TDD 18.2 carve-out, same as trueAffinity above); the
        // aggregate never reaches a recommender. Consumes no rng, so it is stream-neutral (D8).
        if (!ds.users.empty()) {
            double cosineSum = 0.0;
            for (size_t u = 0; u < ds.users.size(); ++u) {
                cosineSum +=
                    dot(ds.users[u].estimatedPreference, ds.hiddenStates[u].hiddenPreference);
            }
            alignmentByRound[round] = cosineSum / static_cast<double>(ds.users.size());
        }
    }

    // 4. Assemble the result.
    ExperimentResult result;
    result.config = config_;
    result.seed = seed;
    result.experimentId = std::string(toString(config_.algorithm)) + "-seed" +
                          std::to_string(seed) + "-" + experimentTimestamp();
    result.directory = outputRoot_ / result.experimentId;
    result.userCount = ds.users.size();
    result.reelCount = ds.reels.size();
    result.requestCount = requestCount;
    result.impressionCount = impressionCount;
    result.overall = metrics.overall();
    result.oracleSampleRate = config_.evaluation.oracleSampleRate;
    result.learningEnabled = learningEnabled;
    result.retrievalApplicable = retrievalApplicable;
    result.retrievalSampleRate = config_.evaluation.retrievalSampleRate;

    double cumulativeRegret = 0.0;
    size_t totalSampled = 0;
    double totalRegretSum = 0.0;
    size_t totalRetrievalSamples = 0;
    double totalRecall10Sum = 0.0;
    double totalRecall50Sum = 0.0;
    double totalDistErrorSum = 0.0;
    result.rounds.reserve(rounds);
    for (size_t r = 0; r < rounds; ++r) {
        RoundMetrics rm;
        rm.round = r;
        rm.metrics = (r < metrics.roundCount()) ? metrics.roundSummary(r) : MetricsSummary{};
        rm.sampledRequests = regretByRound[r].sampled;
        rm.meanRegret = regretByRound[r].sampled > 0
                            ? regretByRound[r].sum / static_cast<double>(regretByRound[r].sampled)
                            : 0.0;
        cumulativeRegret += regretByRound[r].sum;
        rm.cumulativeRegret = cumulativeRegret;

        const RetrievalAcc &racc = retrievalByRound[r];
        rm.retrievalSamples = racc.samples;
        if (racc.samples > 0) {
            const double n = static_cast<double>(racc.samples);
            rm.meanRecallAt10 = racc.recall10Sum / n;
            rm.meanRecallAt50 = racc.recall50Sum / n;
            rm.meanDistanceError = racc.distErrorSum / n;
        }
        rm.meanEstimatedHiddenCosine = alignmentByRound[r];
        result.rounds.push_back(rm);

        totalSampled += regretByRound[r].sampled;
        totalRegretSum += regretByRound[r].sum;
        totalRetrievalSamples += racc.samples;
        totalRecall10Sum += racc.recall10Sum;
        totalRecall50Sum += racc.recall50Sum;
        totalDistErrorSum += racc.distErrorSum;
    }
    result.sampledRequestCount = totalSampled;
    result.meanRegret = totalSampled > 0 ? totalRegretSum / static_cast<double>(totalSampled) : 0.0;
    result.cumulativeRegret = totalRegretSum;
    result.finalEstimatedHiddenCosine =
        result.rounds.empty() ? 0.0 : result.rounds.back().meanEstimatedHiddenCosine;

    result.retrievalSampleCount = totalRetrievalSamples;
    if (totalRetrievalSamples > 0) {
        const double n = static_cast<double>(totalRetrievalSamples);
        result.retrievalRecallAt10 = totalRecall10Sum / n;
        result.retrievalRecallAt50 = totalRecall50Sum / n;
        result.retrievalDistanceError = totalDistErrorSum / n;
    }

    result.latency = latencyStats(latencies);
    result.retrievalLatency = latencyStats(retrievalLatencies);
    result.rankingLatency = latencyStats(rankingLatencies);
    result.rerankingLatency = latencyStats(rerankingLatencies);
    result.totalWallSeconds = wall.elapsedMs() / 1000.0;

    // 5. Write the §26 output layout.
    std::filesystem::create_directories(result.directory);
    RunMetadata meta = collectRunMetadata(provenance_, result.experimentId);
    meta.userCount = ds.users.size();
    meta.reelCount = ds.reels.size();
    meta.creatorCount = ds.creators.size();
    meta.topicCount = ds.topics.size();
    meta.dimensions = config_.simulation.dimensions;
    ResultsWriter::writeAll(result, meta);

    return result;
}

} // namespace rr

#include "rr/evaluation/experiment_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/evaluation/cold_start.hpp"
#include "rr/evaluation/diversity_metrics.hpp"
#include "rr/evaluation/event_driven_runner.hpp"
#include "rr/evaluation/oracle.hpp"
#include "rr/evaluation/oracle_satisfaction_recommender.hpp"
#include "rr/evaluation/results_writer.hpp"
#include "rr/evaluation/retrieval_evaluator.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/learning/online_user_state_updater.hpp"
#include "rr/learning/tolerance_estimator.hpp"
#include "rr/recommendation/effective_preference.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/recommendation/vector_index.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/drift_scheduler.hpp"
#include "rr/simulation/simulator.hpp"

namespace rr {

namespace {

// Cold-start tracking window (TDD 18.5): the harness measures per-injected-user reward and oracle
// regret for a new user's first kColdStartMaxImpressions impressions (indices [0, 100)), which
// covers every reported window (first 10/25/50/100). A named constant, NOT a config knob: the TDD
// fixes these windows, so there is nothing to tune on the config surface.
constexpr std::size_t kColdStartMaxImpressions = 100;

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
    // Phase 18 (D20): the event-queue scheduler dispatches to the EventDrivenRunner; the legacy
    // round-robin loop below is retained permanently as the default and the D17 golden path.
    if (config_.simulation.scheduler == "event_queue") {
        EventDrivenRunner eventRunner(config_, outputRoot_, provenance_);
        return eventRunner.run();
    }
    Stopwatch wall; // total wall time (D9: provenance only, confined to summary.timing).
    const uint64_t seed = config_.simulation.seed;

    // 1. Dataset + cold-start prior (TDD 11.1): every user's three preference vectors start at the
    //    global-average hidden preference. From here they either stay frozen (learning disabled) or
    //    update online after every interaction (Phase 7).
    GeneratedDataset ds = generateDataset(config_.simulation, config_.realism, seed);
    // Cold-start prior FROZEN at run start (TDD 11.1): the global-average hidden preference
    // computed ONCE here, before the round loop. Mid-simulation injected users (Phase 8) are
    // initialized to this SAME prior, so injection TIMING never silently changes the prior a cold
    // user starts from.
    const Embedding coldStartPrior = globalAveragePreference(ds.hiddenStates);
    applyColdStart(ds.users, coldStartPrior);

    // Online preference learning (Phase 7, TDD 8.3/11.2/11.3). Constructed ONCE over the immutable
    // reel catalog; apply() runs after each Simulator::step below. It consumes no rng/clock, so
    // invoking it is stream-neutral (D8) - the recommender/behaviour/oracle streams are untouched.
    // When config.learning.enabled is false the updater is never invoked and estimates stay frozen.
    const OnlineUserStateUpdater updater(ds.reels, config_.learning, config_.realism.contentV2);
    // Phase 17: observables-only tolerance estimation, invoked after each step under the
    // personalized-diversity gate (rng/clock-free — stream-neutral, D8).
    const ToleranceEstimator toleranceEstimator(ds.reels, config_.diversity);
    const bool personalizedDiversity = config_.realism.personalizedDiversity;
    const bool learningEnabled = config_.learning.enabled;

    // Scheduled hidden-preference drift (Phase 10, TDD 11.4). Constructed ONCE over the generated
    // topic set; construction VALIDATES the drift config (empty/invalid mix, unknown topic, bad
    // cohort range) and throws std::invalid_argument -> propagates as a setup error (fail fast,
    // D10). The scheduler reads/writes ONLY HiddenUserState and is rng-free and clock-free, so
    // invoking maybeApply before each Simulator::step is stream-neutral (D8): when
    // config.drift.events is empty maybeApply is a guaranteed no-op and the whole run stays
    // byte-identical to a pre-Phase-10 run (the regression contract). `drifted[u]` is the per-user
    // cohort flag the adaptation metrics split on; it is grown in lockstep with doneByUser when
    // Phase-8 injection appends users.
    const DriftScheduler drift(config_.drift, ds.topics);
    const bool driftConfigured = drift.configured();
    std::vector<uint8_t> drifted(ds.users.size(), 0);
    if (driftConfigured) {
        for (std::size_t u = 0; u < ds.users.size(); ++u) {
            drifted[u] = drift.everApplies(ds.users[u].id) ? 1 : 0;
        }
    }

    // 2. Simulator, recommender, and oracle each on an INDEPENDENT named rng stream (D8) so adding
    //    the oracle never perturbs the behaviour or recommender streams. Under
    //    realism.latent_reactions (Phase 14) the V2 constructor additionally forks the NEW
    //    "satisfaction" stream (D19) — gate-off runs fork exactly the V1 streams and stay
    //    byte-identical (D17).
    const bool latentReactions = config_.realism.latentReactions;
    const bool sessionDynamics = config_.realism.sessionDynamics;
    Simulator sim =
        sessionDynamics
            ? Simulator(config_.behaviour, config_.behaviourV2, config_.sessionDynamics,
                        config_.reward, forkRng(seed, "behaviour"), forkRng(seed, "satisfaction"),
                        forkRng(seed, "session-exit"), forkRng(seed, "external-interruption"),
                        config_.learning.recentWindow, config_.ranking.trendingHalfLifeSeconds)
        : latentReactions
            ? Simulator(config_.behaviour, config_.behaviourV2, config_.reward,
                        forkRng(seed, "behaviour"), forkRng(seed, "satisfaction"),
                        config_.learning.recentWindow, config_.ranking.trendingHalfLifeSeconds)
            : Simulator(config_.behaviour, config_.reward, forkRng(seed, "behaviour"),
                        config_.learning.recentWindow, config_.ranking.trendingHalfLifeSeconds);
    RecommenderDeps deps{ds.reels, ds.users, config_};
    // Phase 15: the oracle-satisfaction arm (V2 TDD 4.4 arm 4) reads hidden state, so it is
    // constructed HERE under the evaluation carve-out (D18); the recommendation-side factory
    // rejects it. All other algorithms go through the factory unchanged.
    std::unique_ptr<Recommender> recommender =
        config_.algorithm == RecommendationAlgorithm::OracleSatisfaction
            ? std::make_unique<OracleSatisfactionRecommender>(
                  config_, ds.reels, ds.users, ds.creators, ds.hiddenStates, ds.hiddenReelStates,
                  forkRng(seed, "recommender"))
            : makeRecommender(config_.algorithm, deps, forkRng(seed, "recommender"));
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
    // Hidden-user-welfare metric group (Phase 15, V2 TDD §6, D22): the welfare-group module
    // accumulates the per-impression latent stream (satisfaction/regret) + observable watch time +
    // the reel's hidden archetype index, all under the D18 EVALUATION CARVE-OUT — none of these
    // ever reach a recommender-visible structure. Fed ONLY under realism.latent_reactions; reduced
    // into ExperimentResult::welfare (per-round + overall
    // satisfaction/regret/satisfaction-per-minute + per-archetype exposure) after the round loop
    // and emitted as gate-on-only CSVs/summary blocks. Sized to the round count and the catalog
    // size so the buckets are deterministic. Constructed unconditionally (cheap); left un-fed and
    // un-reduced when the gate is off (byte-identical, D17).
    WelfareMetrics welfareMetrics(rounds, config_.realism.archetypes.size());
    // Session-health metric group (Phase 16, V2 TDD §4.9/§6, D22): reduces the exit-aware loop's
    // collected SessionRecords (one per probabilistic exit) into per-round + overall session
    // statistics + session utility U_s, all under the D18 EVALUATION CARVE-OUT — SessionRecord's
    // hidden-derived values never reach a recommender-visible structure. Fed ONLY under
    // realism.session_dynamics (via the stepV2 out-slot in the consume loop below); reduced into
    // ExperimentResult::sessionHealth after the round loop and emitted as a gate-on-only
    // CSV/summary block. Sized to the round count and given the session-dynamics lambdas (for U_s);
    // constructed unconditionally (cheap), left un-fed and un-reduced when the gate is off
    // (byte-identical, D17).
    SessionHealthMetrics sessionHealth(rounds, config_.sessionDynamics);
    // Estimate<->hidden alignment (TDD 18.5), one mean per round measured after the round
    // completes.
    std::vector<double> alignmentByRound(rounds, 0.0);

    // Drift cohort split (Phase 10, TDD 18.6): per-round reward and end-of-round est<->hidden
    // alignment, separated into the drifted vs control cohort. All inert (never read) when drift is
    // not configured. Reward accumulators pool this round's impressions; alignment accumulators are
    // filled by the end-of-round alignment loop below.
    struct CohortAcc {
        std::size_t count = 0;
        double sum = 0.0;
    };
    std::vector<CohortAcc> driftedRewardByRound(rounds);
    std::vector<CohortAcc> controlRewardByRound(rounds);
    std::vector<CohortAcc> driftedAlignByRound(rounds);
    std::vector<CohortAcc> controlAlignByRound(rounds);
    // Per-feed diversity (Phase 9, TDD 18.4). Accumulated on EVERY request (unsampled: the
    // computation is trivial) from the feed AS PRESENTED, before its impressions are stepped. Its
    // means feed diversity_metrics.csv + the summary.json diversity block. Deterministic
    // (rng/clock-free), so it is stream-neutral (D8) and part of the byte-identical guarantee.
    DiversityAccumulator diversityByRound(rounds);

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

    // --- Phase 8 mid-simulation injection state (TDD 18.5) --------------------------------------
    // Injection is "configured" when either count > 0. When it is NOT configured every branch below
    // is inert, so the run is byte-identical to a pre-Phase-8 run (the regression contract).
    const uint32_t newReels = config_.simulation.newReels;
    const uint32_t newReelsAt = config_.simulation.newReelsAt;
    const uint32_t newUsers = config_.simulation.newUsers;
    const uint32_t newUsersAt = config_.simulation.newUsersAt;
    const bool injectionConfigured = (newReels > 0) || (newUsers > 0);

    // First dense index of each injected block once injected (kNone until then, and forever when
    // that entity type is not injected).
    constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
    std::size_t firstInjectedReelIndex = kNone;
    std::size_t firstInjectedUserIndex = kNone;

    // New-user cold-start tracking (TDD 18.5): for each per-user impression index (0-based, capped
    // at kColdStartMaxImpressions), the pooled reward and forced-oracle regret over injected users.
    struct ColdStartAcc {
        std::size_t users = 0;
        double rewardSum = 0.0;
        double regretSum = 0.0;
    };
    std::vector<ColdStartAcc> newUserByImpression(kColdStartMaxImpressions);

    // New-reel exposure (TDD 18.5): impressions landing on injected reels, per round and
    // cumulative- distinct; plus total impressions per round for the round's exposure share.
    std::vector<std::size_t> injectedImpressionsByRound(rounds, 0);
    std::vector<std::size_t> impressionsByRound(rounds, 0);
    std::vector<std::size_t> distinctInjectedExposedByRound(rounds, 0);
    std::unordered_set<uint32_t> exposedInjectedReels;

    // Target-reward baseline (TDD 18.5): mean reward per impression over impressions consumed
    // BEFORE the new users are injected (rounds < newUsersAt) - the established reward level a cold
    // user is trying to reach.
    double preInjectionRewardSum = 0.0;
    std::size_t preInjectionImpressions = 0;

    for (size_t round = 0; round < rounds; ++round) {
        // Mid-simulation injection at the START of the round (Phase 8, TDD 18.5). ORDER within a
        // round is REELS FIRST, then users, so a user injected on the same round can already be
        // recommended the freshly-injected reels in its very first feed.
        if (newReels > 0 && round == newReelsAt) {
            firstInjectedReelIndex = appendReels(ds, config_.simulation, seed, newReels, sim.now());
            // Grow the recommender's vector index (D2 insert-only) so injected reels are
            // retrievable.
            recommender->onReelsAppended(firstInjectedReelIndex);
            // Keep live Recall@K ground truth honest against the grown catalog.
            if (retrievalEval.has_value()) {
                retrievalEval->appendReels(ds.reels, firstInjectedReelIndex);
            }
        }
        if (newUsers > 0 && round == newUsersAt) {
            firstInjectedUserIndex = appendUsers(ds, config_.simulation, seed, newUsers);
            // Apply the FROZEN run-start prior to the injected users only (TDD 11.1); the generator
            // leaves their estimate vectors empty. From here the OnlineUserStateUpdater evolves
            // them per interaction (when learning is enabled), exactly like an original cold user.
            for (std::size_t u = firstInjectedUserIndex; u < ds.users.size(); ++u) {
                ds.users[u].estimatedPreference = coldStartPrior;
                ds.users[u].longTermPreference = coldStartPrior;
                ds.users[u].sessionPreference = coldStartPrior;
            }
            // Grow per-user bookkeeping; injected users start with a spent budget of 0, so they
            // receive requests from THIS round onward - naturally only the remaining rounds, which
            // is exactly their cold-start window.
            doneByUser.resize(ds.users.size(), 0);
            // Extend the drift cohort flags for the injected users (Phase 10): cohort membership is
            // a deterministic hash of the userId, so an injected user drifts iff its id lands in a
            // configured cohort, exactly like an original user.
            drifted.resize(ds.users.size(), 0);
            if (driftConfigured) {
                for (std::size_t u = firstInjectedUserIndex; u < ds.users.size(); ++u) {
                    drifted[u] = drift.everApplies(ds.users[u].id) ? 1 : 0;
                }
            }
        }

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
            // Config-driven from Phase 8 (was hard-false). VERIFIED: no existing recommender,
            // orchestrator, or candidate source reads req.enableExploration today - only Package
            // A's new exploration source will - so every existing algorithm's output stays
            // byte-identical to a pre-Phase-8 run regardless of exploration.enabled. (grep: the
            // only readers are this assignment and a summary.json note string.)
            req.enableExploration = config_.exploration.enabled;
            // Config-driven from Phase 9 (was hard-false). VERIFIED by whole-repo grep: no existing
            // recommender, orchestrator, or candidate source reads req.enableDiversity today - the
            // sibling package's new orchestrator diversity gate will be its ONLY reader - so every
            // existing algorithm's output stays byte-identical regardless of diversity.enabled.
            // (The only current readers of the field are this assignment and a summary.json note
            // string.)
            req.enableDiversity = config_.diversity.enabled;
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

            // Feed reel ids, extracted ONCE for both the diversity metrics and the oracle below.
            std::vector<ReelId> feedReelIds;
            feedReelIds.reserve(resp.reels.size());
            for (const RankedReel &ranked : resp.reels) {
                feedReelIds.push_back(ranked.reelId);
            }

            // Per-feed diversity (Phase 9, TDD 18.4). Measured HERE - after recommend() returns but
            // BEFORE the impression loop below steps the simulator - so user.seenReels is still the
            // pre-feed ("as of presentation") set. VERIFIED: seenReels is mutated only inside
            // Simulator::step (user.seenReels.insert(reel.id)), which runs in the consume loop
            // further down, so nothing shown by THIS feed is in the seen-set yet. Unsampled: the
            // computation is trivial (feedSize dot products) and diversity is reported per feed.
            diversityByRound.add(round, feedReelIds, ds.reels, user.seenReels);

            // Oracle sampling: draw for EVERY request so the oracle stream stays aligned across
            // runs regardless of the outcome (D8). This Bernoulli draw is UNCONDITIONAL and
            // unchanged from pre-Phase-8, so the global sampled-regret aggregate is byte-identical
            // when injection is off.
            const bool sampled = oracleRng.bernoulli(config_.evaluation.oracleSampleRate);

            // Cold-start forcing (TDD 18.5/19): for an INJECTED user we evaluate the oracle on
            // EVERY request while its own impression count is still inside the tracked window, even
            // when the Bernoulli gate did not fire. computeOracleRegret is rng-free (only the gate
            // above draws from the "oracle" stream), so forcing consumes NO rng and every stream
            // stays aligned.
            const bool injectedUser =
                firstInjectedUserIndex != kNone && u >= firstInjectedUserIndex;
            const bool forceColdStart = injectedUser && doneByUser[u] < kColdStartMaxImpressions;

            double coldStartRequestRegret = 0.0;
            if (sampled || forceColdStart) {
                // seen-set snapshot is the pre-feed state: the oracle scores what was available
                // BEFORE this feed is consumed. `feedReelIds` was extracted above.
                const OracleResult oracle =
                    computeOracleRegret(ds.hiddenStates[u].hiddenPreference, ds.reels,
                                        user.seenReels, feedReelIds, feedSize);
                // Gate-fired samples feed the GLOBAL aggregate exactly as before (injected users'
                // gate-fired samples are INCLUDED, unchanged). Forced evaluations feed ONLY the
                // cold-start accumulators below - never double-counted into the global aggregate.
                if (sampled) {
                    regretByRound[round].sampled += 1;
                    regretByRound[round].sum += oracle.regret;
                }
                coldStartRequestRegret = oracle.regret;
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
            const size_t preFeedDone =
                doneByUser[u]; // impression index base for injected-user curve
            const size_t toConsume = std::min(remaining, resp.reels.size());
            // Phase 16 exit-aware consumption (V2 TDD 4.8, realism.session_dynamics): the actual
            // number of impressions simulated from THIS feed, which is < toConsume when the session
            // exits mid-feed (the remaining items are never simulated — no draws, deterministic).
            // With session dynamics off it always equals toConsume, so the budget bookkeeping below
            // is byte-identical to the pre-Phase-16 loop.
            size_t consumed = 0;
            for (size_t k = 0; k < toConsume; ++k) {
                const ReelId reelId = resp.reels[k].reelId;
                Reel &reel = ds.reels[reelId.value];
                const Creator &creator = ds.creators[reel.creatorId.value];

                // Scheduled hidden-preference drift (Phase 10, TDD 11.4): fire BEFORE the step,
                // keyed on the DOMAIN counter user.totalInteractions (Simulator::step increments it
                // by exactly 1 per impression, so every count is hit once -> each event fires
                // exactly once per cohort user). The impression about to be simulated (0-based
                // index == totalInteractions) is therefore the FIRST interaction under the new
                // preference. maybeApply is rng/clock-free, so when drift is unconfigured this is a
                // guaranteed no-op and stream-neutral (D8).
                drift.maybeApply(ds.hiddenStates[u], static_cast<uint32_t>(user.totalInteractions));

                // Phase 14 (realism.latent_reactions): the V2 step computes the hidden
                // LatentReaction and populates the V2 event fields; the latent flows ONLY into
                // the welfare accumulators below (evaluation carve-out, D18). Gate-off takes the
                // V1 step, byte-identical to pre-Phase-14 (D17).
                LatentReaction latent;
                StepResult stepStorage;
                // Phase 16: filled by stepV2 with the COMPLETED session when this impression closes
                // one under realism.session_dynamics; read only when the exit observable fires.
                SessionRecord closedRec;
                if (latentReactions) {
                    const RankedReel &served = resp.reels[k];
                    StepV2Inputs v2;
                    v2.hiddenReel = &ds.hiddenReelStates[reelId.value];
                    v2.positionInFeed = static_cast<uint32_t>(k);
                    v2.requestId = static_cast<uint64_t>(requestCount);
                    // requestTimestamp is the logical time the FEED REQUEST was served — the
                    // single `req.requestTime` captured once above (== sim.now() before this feed
                    // is consumed), the same value handed to the recommender. Using it (rather than
                    // a per-impression sim.now(), which drifts forward as earlier items in the same
                    // feed advance the clock) keeps requestTimestamp <= startTimestamp for every
                    // item in the feed and makes all items of one feed share a request time.
                    v2.requestTimestamp = req.requestTime;
                    // Provenance election (documented rule): a feed item can carry several
                    // candidate-source labels (it was retrieved by more than one source, Phase 8
                    // semantics). We elect a SINGLE representative for the event: Exploration wins
                    // if present (so the exploration flag and the provenance agree, and exploration
                    // exposure is never masked by a co-occurring organic source); otherwise the
                    // first (highest-priority) label the orchestrator recorded. Empty source list
                    // (identity/no-orchestrator paths) defaults to the benign VectorHNSW.
                    v2.fromExploration = false;
                    v2.sourceProvenance =
                        served.sources.empty() ? CandidateSource::VectorHNSW : served.sources[0];
                    for (const CandidateSource s : served.sources) {
                        if (s == CandidateSource::Exploration) {
                            v2.fromExploration = true;
                            v2.sourceProvenance = CandidateSource::Exploration;
                        }
                    }
                    // Phase 16 (realism.session_dynamics): pass a SessionRecord out-slot so the
                    // simulator can hand back a COMPLETED session when this impression closes one
                    // (probabilistic exit, V2 TDD 4.8). nullptr when session dynamics is off — the
                    // stepV2 default, byte-identical to the P14/P15 call.
                    SessionRecord *recSlot = sessionDynamics ? &closedRec : nullptr;
                    stepStorage =
                        sim.stepV2(user, ds.hiddenStates[u], reel, creator, v2, latent, recSlot);
                    // Welfare-group feed (evaluation carve-out, D18): the hidden latent +
                    // observable watch time + the reel's hidden archetype index. watchSeconds is
                    // the satisfaction-per-minute denominator; archetypeIndex drives the exposure
                    // breakdown. None of these reach any recommender-visible surface.
                    WelfareImpression wi;
                    wi.immediateSatisfaction = latent.immediateSatisfaction;
                    wi.regret = latent.regret;
                    wi.watchSeconds = stepStorage.outcome.watchSeconds;
                    wi.archetypeIndex = ds.hiddenReelStates[reelId.value].archetypeIndex;
                    welfareMetrics.add(round, wi);
                } else {
                    stepStorage = sim.step(user, ds.hiddenStates[u], reel, creator);
                }
                const StepResult &step = stepStorage;

                // Online preference update (Phase 7): runs AFTER Simulator::step has recorded the
                // interaction into user.recentInteractions, updating ONLY the three preference
                // vectors (long-term / session / cached estimate). Gated by learning.enabled so the
                // frozen arm keeps the cold-start estimates. Stream-neutral (no rng/clock).
                if (learningEnabled) {
                    updater.apply(user, reel, step.event);
                }
                if (personalizedDiversity) {
                    toleranceEstimator.apply(user, reel, step.event);
                }

                ImpressionSample s;
                s.watchRatio = step.event.watchRatio;
                s.watchSeconds = step.event.watchSeconds;
                s.instantSkip = step.outcome.instantSkip;
                s.completed = step.outcome.completed;
                s.liked = step.outcome.liked;
                s.shared = step.outcome.shared;
                s.followed = step.outcome.followed;
                // Realism V2 engagement signals (Phase 15). Always false under the V1 step (gate
                // off), so the derived rates stay 0 and never touch a V1 CSV — they surface only in
                // the gate-on welfare CSV / summary blocks (D22).
                s.commented = step.outcome.commented;
                s.saved = step.outcome.saved;
                s.profileVisited = step.outcome.profileVisited;
                s.reward = step.event.reward;
                // Evaluation-only hidden-state read (TDD 18.2): true affinity of the shown reel.
                s.trueAffinity = dot(ds.hiddenStates[u].hiddenPreference, reel.embedding);
                s.userId = user.id.value;
                s.sessionId = step.event.sessionId.value;
                metrics.add(round, s);
                ++impressionCount;

                // Drift cohort reward split (Phase 10, TDD 18.6): pool this impression's reward
                // into the drifted or control accumulator. Inert (guarded) when drift is
                // unconfigured.
                if (driftConfigured) {
                    CohortAcc &racc =
                        drifted[u] ? driftedRewardByRound[round] : controlRewardByRound[round];
                    racc.count += 1;
                    racc.sum += step.event.reward;
                }

                // --- Phase 8 injection metrics (all inert when injection is not configured)
                // -------
                ++impressionsByRound[round];
                // Target-reward baseline: reward of impressions consumed BEFORE new users appear.
                if (newUsers > 0 && round < newUsersAt) {
                    preInjectionRewardSum += step.event.reward;
                    ++preInjectionImpressions;
                }
                // New-reel exposure (TDD 18.5): an impression landing on an injected reel.
                if (firstInjectedReelIndex != kNone && reelId.value >= firstInjectedReelIndex) {
                    ++injectedImpressionsByRound[round];
                    exposedInjectedReels.insert(reelId.value);
                }
                // New-user cold-start curve (TDD 18.5): reward + forced-oracle regret indexed by
                // this injected user's OWN impression index. Only indices in [0,
                // kColdStartMaxImpressions) are tracked, and every such index implies
                // forceColdStart fired this request (idx >= preFeedDone), so coldStartRequestRegret
                // is a real forced evaluation, not the 0 default.
                if (injectedUser) {
                    const std::size_t idx = preFeedDone + k;
                    if (idx < kColdStartMaxImpressions) {
                        ColdStartAcc &acc = newUserByImpression[idx];
                        acc.users += 1;
                        acc.rewardSum += step.event.reward;
                        acc.regretSum += coldStartRequestRegret;
                    }
                }

                // Phase 16 exit-aware consumption (V2 TDD 4.8): this impression WAS simulated (its
                // metrics counted above), so bump the consumed count; then, under
                // realism.session_dynamics, if it CLOSED the session, collect the completed
                // SessionRecord for the session-health group and STOP consuming this feed — the
                // remaining feed items are never simulated (no draws → deterministic). The user's
                // NEXT request opens a fresh session; interactionsPerUser still counts IMPRESSIONS,
                // so an exit merely fragments the budget across sessions. Guarded by
                // sessionDynamics, so a gate-off run consumes the whole feed exactly as the
                // pre-Phase-16 loop did.
                ++consumed;
                if (sessionDynamics && stepStorage.event.observedExitAfterImpression) {
                    sessionHealth.add(round, closedRec);
                    break;
                }
            }
            doneByUser[u] += consumed;
        }

        // End-of-round estimate<->hidden alignment (TDD 18.5): mean over ALL users of
        // cos(estimatedPreference, hiddenPreference). Both are unit-length, so cosine == dot.
        // Evaluation-only hidden-state read (TDD 18.2 carve-out, same as trueAffinity above); the
        // aggregate never reaches a recommender. Consumes no rng, so it is stream-neutral (D8).
        if (!ds.users.empty()) {
            double cosineSum = 0.0;
            for (size_t u = 0; u < ds.users.size(); ++u) {
                const double cos =
                    dot(ds.users[u].estimatedPreference, ds.hiddenStates[u].hiddenPreference);
                cosineSum += cos;
                // Drift cohort alignment split (Phase 10, TDD 18.6). The population sum above is
                // unchanged (same terms, same order), so the overall alignment stays
                // byte-identical.
                if (driftConfigured) {
                    CohortAcc &aacc =
                        drifted[u] ? driftedAlignByRound[round] : controlAlignByRound[round];
                    aacc.count += 1;
                    aacc.sum += cos;
                }
            }
            alignmentByRound[round] = cosineSum / static_cast<double>(ds.users.size());
        }

        // Cumulative-distinct injected reels exposed through the END of this round (TDD 18.5).
        distinctInjectedExposedByRound[round] = exposedInjectedReels.size();
    }

    // Phase 16 run-end drain (integration): sessions still open when the run ends become
    // RunEnded records (reported as open_sessions; excluded from every exit-rate denominator).
    // Attributed to the LAST round, matching the collected-where-closed convention.
    if (sessionDynamics && rounds > 0) {
        for (const SessionRecord &openRec : sim.drainOpenSessions()) {
            sessionHealth.add(rounds - 1, openRec);
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

        // Per-round diversity means (Phase 9, TDD 18.4).
        const DiversitySummary dsum = diversityByRound.roundSummary(r);
        rm.diversityFeeds = dsum.feeds;
        rm.meanUniqueTopics = dsum.meanUniqueTopics;
        rm.meanUniqueCreators = dsum.meanUniqueCreators;
        rm.meanIntraListSimilarity = dsum.meanIntraListSimilarity;
        rm.meanTopicConcentration = dsum.meanTopicConcentration;
        rm.meanCreatorConcentration = dsum.meanCreatorConcentration;
        rm.repetitionCount = dsum.totalRepeats;
        rm.repetitionRate = dsum.repetitionRate;

        // Drift cohort split (Phase 10, TDD 18.6): means (0.0 with a 0 count; the count lets the
        // writer print `nan` for an empty cohort). All left 0 when drift is not configured.
        const CohortAcc &dr = driftedRewardByRound[r];
        const CohortAcc &cr = controlRewardByRound[r];
        const CohortAcc &da = driftedAlignByRound[r];
        const CohortAcc &ca = controlAlignByRound[r];
        rm.driftedImpressions = dr.count;
        rm.driftedMeanReward = dr.count > 0 ? dr.sum / static_cast<double>(dr.count) : 0.0;
        rm.controlImpressions = cr.count;
        rm.controlMeanReward = cr.count > 0 ? cr.sum / static_cast<double>(cr.count) : 0.0;
        rm.driftedAlignUsers = da.count;
        rm.driftedAlignment = da.count > 0 ? da.sum / static_cast<double>(da.count) : 0.0;
        rm.controlAlignUsers = ca.count;
        rm.controlAlignment = ca.count > 0 ? ca.sum / static_cast<double>(ca.count) : 0.0;

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

    // Hidden-user-welfare reduction (Phase 15, V2 TDD §6, D22): the welfare-group module reduces
    // the per-impression latent/observable/archetype stream into per-round + overall
    // satisfaction/regret/satisfaction-per-minute and the per-archetype exposure breakdown.
    // Populated ONLY under realism.latent_reactions (evaluation carve-out, D18); a gate-off run
    // leaves welfare.configured false and emits no welfare block/CSVs (byte-identical to
    // pre-Phase-14, D17). Deterministic (a pure reduction of sums the simulation already produced).
    // Archetype indices resolve to catalog names in index order (hidden identity → evaluation-only
    // label; the ranker never sees it).
    result.welfare.configured = latentReactions;
    if (latentReactions) {
        std::vector<std::string> archetypeNames;
        archetypeNames.reserve(config_.realism.archetypes.size());
        for (const ArchetypeSpec &a : config_.realism.archetypes) {
            archetypeNames.push_back(a.name);
        }
        result.welfare = welfareMetrics.reduce(archetypeNames);
        result.welfare.configured = true;
    }

    // Session-health reduction (Phase 16, V2 TDD §4.9/§6, D22): the session-health module reduces
    // the exit-aware loop's collected SessionRecords into per-round + overall session statistics +
    // session utility U_s (evaluation carve-out, D18). Populated ONLY under
    // realism.session_dynamics; a gate-off run leaves sessionHealth.configured false and emits no
    // session_health block/CSV (byte-identical to pre-Phase-16, D17). Deterministic (a pure
    // reduction of the collected records). Under the P16 scaffold stub the loop closes zero
    // sessions (stepV2 never fires the exit observable), so this reduces to an all-zero,
    // well-formed report — the populated path is exercised once package A lands the exit model.
    result.sessionHealth.configured = sessionDynamics;
    if (sessionDynamics) {
        result.sessionHealth = sessionHealth.reduce();
        result.sessionHealth.configured = true;

        // Realize the hidden-welfare group's harmful_fatigue column (Phase 15 shipped it as a
        // NOT-YET-MODELED constant 0). Under session dynamics it is the mean end-of-session harmful
        // fatigue over closed sessions — overall from the session-health overall mean, per round
        // from the round mean. This is the ONLY place the welfare group's harmful_fatigue is
        // filled; under P15-only gates this block does not run and the column stays 0, keeping the
        // P15 welfare output byte-identical (documented in results_writer). result.welfare is
        // always populated here because session_dynamics requires latent_reactions (D17 gate
        // dependency).
        result.welfare.harmfulFatigue = result.sessionHealth.harmfulFatigueMean;
        for (std::size_t r = 0; r < result.welfare.byRound.size(); ++r) {
            if (r < result.sessionHealth.byRound.size()) {
                result.welfare.byRound[r].harmfulFatigue =
                    result.sessionHealth.byRound[r].harmfulFatigueMean;
            }
        }
    }

    result.retrievalSampleCount = totalRetrievalSamples;
    if (totalRetrievalSamples > 0) {
        const double n = static_cast<double>(totalRetrievalSamples);
        result.retrievalRecallAt10 = totalRecall10Sum / n;
        result.retrievalRecallAt50 = totalRecall50Sum / n;
        result.retrievalDistanceError = totalDistErrorSum / n;
    }

    // Overall diversity (Phase 9, TDD 18.4): pooled means over EVERY feed in the run.
    const DiversitySummary diversityOverall = diversityByRound.overall();
    result.diversityFeedCount = diversityOverall.feeds;
    result.meanUniqueTopics = diversityOverall.meanUniqueTopics;
    result.meanUniqueCreators = diversityOverall.meanUniqueCreators;
    result.meanIntraListSimilarity = diversityOverall.meanIntraListSimilarity;
    result.meanTopicConcentration = diversityOverall.meanTopicConcentration;
    result.meanCreatorConcentration = diversityOverall.meanCreatorConcentration;
    result.totalRepetitions = diversityOverall.totalRepeats;
    result.repetitionRate = diversityOverall.repetitionRate;

    result.latency = latencyStats(latencies);
    result.retrievalLatency = latencyStats(retrievalLatencies);
    result.rankingLatency = latencyStats(rankingLatencies);
    result.rerankingLatency = latencyStats(rerankingLatencies);
    result.totalWallSeconds = wall.elapsedMs() / 1000.0;

    // 4b. Cold-start / injection report (Phase 8, TDD 18.5). Assembled + written ONLY when
    // injection
    //     is configured; a non-configured run leaves coldStart.configured == false and writes NO
    //     injection files/keys (byte-identical regression contract).
    ColdStartReport &cs = result.coldStart;
    cs.configured = injectionConfigured;
    if (injectionConfigured) {
        cs.newUsers = newUsers;
        cs.newUsersAt = newUsersAt;
        cs.newReels = newReels;
        cs.newReelsAt = newReelsAt;

        // Pooled new-user regret over the first N impressions (indices [0, N)); -1 when no
        // injected- user impression fell in the window.
        const auto pooledRegret = [&](std::size_t n) -> double {
            const std::size_t lim = std::min(n, newUserByImpression.size());
            std::size_t users = 0;
            double regretSum = 0.0;
            for (std::size_t i = 0; i < lim; ++i) {
                users += newUserByImpression[i].users;
                regretSum += newUserByImpression[i].regretSum;
            }
            return users > 0 ? regretSum / static_cast<double>(users) : -1.0;
        };
        cs.meanRegretFirst10 = pooledRegret(10);
        cs.meanRegretFirst25 = pooledRegret(25);
        cs.meanRegretFirst50 = pooledRegret(50);
        cs.meanRegretFirst100 = pooledRegret(100);

        // Target = pre-injection mean reward per impression; undefined if there were none (e.g. new
        // users injected at round 0). Interactions-to-target = smallest K (1-based) at which
        // injected users' cumulative mean reward over their first K impressions reaches the target;
        // -1 when the target is undefined or never reached within the tracked window.
        cs.targetDefined = preInjectionImpressions > 0;
        cs.targetReward = cs.targetDefined
                              ? preInjectionRewardSum / static_cast<double>(preInjectionImpressions)
                              : 0.0;
        cs.interactionsToTargetReward = -1;
        if (cs.targetDefined) {
            std::size_t users = 0;
            double rewardSum = 0.0;
            for (std::size_t i = 0; i < newUserByImpression.size(); ++i) {
                users += newUserByImpression[i].users;
                rewardSum += newUserByImpression[i].rewardSum;
                if (users > 0 && rewardSum / static_cast<double>(users) >= cs.targetReward) {
                    cs.interactionsToTargetReward = static_cast<long>(i) + 1;
                    break;
                }
            }
        }

        // New-user curve rows up to the largest impression index any injected user actually reached
        // (header-only when none reached). Deterministic.
        std::size_t maxReached = 0;
        bool anyReached = false;
        for (std::size_t i = 0; i < newUserByImpression.size(); ++i) {
            if (newUserByImpression[i].users > 0) {
                maxReached = i;
                anyReached = true;
            }
        }
        if (anyReached) {
            cs.newUserCurve.reserve(maxReached + 1);
            for (std::size_t i = 0; i <= maxReached; ++i) {
                const ColdStartAcc &acc = newUserByImpression[i];
                NewUserCurvePoint p;
                p.impressionIndex = i;
                p.usersAtIndex = acc.users;
                p.meanReward = acc.users > 0 ? acc.rewardSum / static_cast<double>(acc.users) : 0.0;
                p.meanRegret = acc.users > 0 ? acc.regretSum / static_cast<double>(acc.users) : 0.0;
                cs.newUserCurve.push_back(p);
            }
        }

        // New-reel exposure rows (one per round) + run totals.
        std::size_t cumImpr = 0;
        cs.newReelExposure.reserve(rounds);
        for (std::size_t r = 0; r < rounds; ++r) {
            cumImpr += injectedImpressionsByRound[r];
            NewReelExposurePoint p;
            p.round = r;
            p.injectedImpressions = injectedImpressionsByRound[r];
            p.injectedImpressionsCum = cumImpr;
            p.distinctInjectedExposedCum = distinctInjectedExposedByRound[r];
            p.shareOfRoundImpressions = impressionsByRound[r] > 0
                                            ? static_cast<double>(injectedImpressionsByRound[r]) /
                                                  static_cast<double>(impressionsByRound[r])
                                            : 0.0;
            cs.newReelExposure.push_back(p);
        }
        cs.totalInjectedImpressions = cumImpr;
        cs.distinctInjectedExposed = exposedInjectedReels.size();
        cs.injectedImpressionShare = impressionCount > 0 ? static_cast<double>(cumImpr) /
                                                               static_cast<double>(impressionCount)
                                                         : 0.0;
    }

    // 4c. Preference-drift adaptation report (Phase 10, TDD 18.6). Assembled + written ONLY when
    //     drift is configured; a non-drift run leaves adaptation.configured == false and writes NO
    //     adaptation keys/columns (byte-identical regression contract). All reads are of the
    //     already-assembled per-round cohort aggregates, so this is pure post-processing.
    AdaptationReport &ad = result.adaptation;
    ad.configured = driftConfigured;
    if (driftConfigured) {
        ad.feedSize = feedSize;
        ad.firstDriftInteraction = drift.firstDriftInteraction();
        // driftRound = floor(firstDriftInteraction / feedSize): the first round whose feeds can be
        // affected by the drift (the impression at index firstDriftInteraction is served here).
        ad.driftRound = feedSize > 0 ? static_cast<long>(ad.firstDriftInteraction / feedSize) : 0;

        for (uint8_t f : drifted) {
            if (f) {
                ++ad.driftedUsers;
            }
        }
        ad.controlUsers = drifted.size() - ad.driftedUsers;

        const long driftRound = ad.driftRound;
        const long nRounds = static_cast<long>(result.rounds.size());

        // Pre-drift reward baseline: mean of the drifted cohort's per-round mean reward over up to
        // the 3 rounds immediately before driftRound (rounds with drifted impressions only). -1
        // when driftRound == 0 or none of the window rounds had drifted impressions.
        if (driftRound > 0) {
            const long lo = std::max<long>(0, driftRound - 3);
            double sum = 0.0;
            long n = 0;
            for (long r = lo; r < driftRound && r < nRounds; ++r) {
                if (result.rounds[r].driftedImpressions > 0) {
                    sum += result.rounds[r].driftedMeanReward;
                    ++n;
                }
            }
            if (n > 0) {
                ad.preDriftReward = sum / static_cast<double>(n);
            }
        }

        // Trough = min drifted-cohort round reward over rounds >= driftRound; rewardDrop against
        // the pre-drift baseline (only when both are defined).
        for (long r = driftRound; r < nRounds; ++r) {
            if (result.rounds[r].driftedImpressions == 0) {
                continue;
            }
            const double rew = result.rounds[r].driftedMeanReward;
            if (ad.troughRound < 0 || rew < ad.troughReward) {
                ad.troughReward = rew;
                ad.troughRound = r;
            }
        }
        if (ad.preDriftReward >= 0.0 && ad.troughRound >= 0) {
            ad.rewardDrop = ad.preDriftReward - ad.troughReward;
        }

        // Recovery: first round >= driftRound with drifted reward >= 0.95 * preDriftReward.
        if (ad.preDriftReward > 0.0) {
            const double target = 0.95 * ad.preDriftReward;
            for (long r = driftRound; r < nRounds; ++r) {
                if (result.rounds[r].driftedImpressions > 0 &&
                    result.rounds[r].driftedMeanReward >= target) {
                    ad.recoveryRound = r;
                    ad.recoveryInteractions = (r - driftRound + 1) * static_cast<long>(feedSize);
                    break;
                }
            }
        }

        // Alignment adaptation: pre-drift baseline (round driftRound-1), post-drift minimum, and
        // the 0.95 * pre-drift threshold crossing (the "new preference detected" reading).
        if (driftRound > 0 && driftRound - 1 < nRounds &&
            result.rounds[driftRound - 1].driftedAlignUsers > 0) {
            ad.preDriftAlignment = result.rounds[driftRound - 1].driftedAlignment;
        }
        for (long r = driftRound; r < nRounds; ++r) {
            if (result.rounds[r].driftedAlignUsers == 0) {
                continue;
            }
            const double al = result.rounds[r].driftedAlignment;
            if (ad.postDriftAlignmentMinRound < 0 || al < ad.postDriftAlignmentMin) {
                ad.postDriftAlignmentMin = al;
                ad.postDriftAlignmentMinRound = r;
            }
        }
        if (ad.preDriftAlignment > 0.0) {
            const double target = 0.95 * ad.preDriftAlignment;
            for (long r = driftRound; r < nRounds; ++r) {
                if (result.rounds[r].driftedAlignUsers > 0 &&
                    result.rounds[r].driftedAlignment >= target) {
                    ad.alignmentRecoveryRound = r;
                    break;
                }
            }
        }

        // Cumulative regret during adaptation: sum of per-round SAMPLED regret over
        // [driftRound, recoveryRound|last]. Whole-population aggregate, true-affinity units.
        if (driftRound < nRounds) {
            const long end = ad.recoveryRound >= 0 ? ad.recoveryRound : nRounds - 1;
            double regretSum = 0.0;
            for (long r = driftRound; r <= end && r < nRounds; ++r) {
                regretSum += regretByRound[static_cast<std::size_t>(r)].sum;
            }
            ad.adaptationWindowRegret = regretSum;
        }
    }

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

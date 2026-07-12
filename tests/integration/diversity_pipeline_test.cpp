// Phase 9 integration tests: the full diversity pipeline end-to-end (TDD 16.6). The flagship
// per-feed invariants live in tests/property/diversity_property_test.cpp against the reranker
// directly; here the SAME caps are asserted through the real generated-dataset pipeline with
// live seen-state, plus the two regression contracts: diversity-off is byte-identical to
// hnsw_ranker (mirroring Phase 8's epsilon=0 no-op), and the complete initial system
// (ranker + exploration + diversity) is deterministic.
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/evaluation/cold_start.hpp"
#include "rr/evaluation/experiment_runner.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/constraint_reranker.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/simulator.hpp"

using namespace rr;

namespace {

namespace fs = std::filesystem;

std::string readFile(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Small but real dataset: enough reels per creator/topic that the caps genuinely bind.
ExperimentConfig pipelineConfig() {
    ExperimentConfig config{};
    config.simulation.seed = 20260712;
    config.simulation.reels = 1500;
    config.simulation.users = 40;
    config.simulation.creators = 60; // ~25 reels/creator: the creator cap must do real work
    config.simulation.topics = 8;    // ~190 reels/topic: the topic cap must do real work
    config.simulation.dimensions = 32;
    config.recommendation.feedSize = 10;
    config.recommendation.vectorCandidates = 300;
    return config;
}

// The same-shape arm config the byte-identical comparisons run under (pattern from
// ranker_pipeline_test.cpp).
ExperimentConfig armConfig(RecommendationAlgorithm algo) {
    ExperimentConfig c;
    c.simulation.seed = 20260712;
    c.simulation.users = 200;
    c.simulation.reels = 2000;
    c.simulation.creators = 100;
    c.simulation.topics = 16;
    c.simulation.dimensions = 32;
    c.simulation.interactionsPerUser = 30;
    c.recommendation.feedSize = 10;
    c.recommendation.vectorCandidates = 300;
    c.evaluation.oracleSampleRate = 0.05;
    c.algorithm = algo;
    return c;
}

// The deterministic, algorithm-output-dependent CSVs (D8/TDD 24.6). config.json and summary.json
// legitimately differ across arms (algorithm name, flags), so equivalence claims compare these.
const char *const kDeterministicCsvs[] = {"recommendation_metrics.csv", "learning_curve.csv",
                                          "regret_curve.csv",           "retrieval_metrics.csv",
                                          "latency_metrics.csv",        "diversity_metrics.csv"};

fs::path runArm(const ExperimentConfig &config, const fs::path &root) {
    fs::remove_all(root);
    ExperimentRunner runner(config, root);
    return runner.run().directory;
}

} // namespace

// End-to-end through the REAL pipeline with live seen-state: generate a dataset, run the
// hnsw_ranker_diversity recommender round-by-round with the Simulator consuming every feed, and
// assert the TDD 15.1 hard rules on every feed served: no duplicate ids, nothing the user had
// seen at presentation time, and the creator/topic caps.
TEST(DiversityPipelineTest, FullPipelineFeedsRespectCapsWithLiveSeenState) {
    ExperimentConfig config = pipelineConfig();
    config.algorithm = RecommendationAlgorithm::HnswRankerDiversity;
    config.exploration.enabled = false; // diversity-isolation mode
    const uint64_t seed = config.simulation.seed;

    GeneratedDataset ds = generateDataset(config.simulation, seed);
    applyColdStart(ds.users, globalAveragePreference(ds.hiddenStates)); // as the harness does
    RecommenderDeps deps{ds.reels, ds.users, config};
    auto recommender = makeRecommender(RecommendationAlgorithm::HnswRankerDiversity, deps,
                                       forkRng(seed, "recommender"));
    Simulator sim(config.behaviour, config.reward, forkRng(seed, "behaviour"),
                  config.learning.recentWindow, config.ranking.trendingHalfLifeSeconds);

    const std::size_t feedSize = config.recommendation.feedSize;
    const std::size_t topicCap =
        ConstraintReranker::topicCap(config.diversity.maxPerTopic, feedSize);
    const std::size_t rounds = 8;
    std::size_t feedsChecked = 0;

    for (std::size_t round = 0; round < rounds; ++round) {
        for (User &user : ds.users) {
            RecommendationRequest req{};
            req.userId = user.id;
            req.sessionId = user.recentInteractions.empty()
                                ? SessionId{0}
                                : user.recentInteractions.back().sessionId;
            req.feedSize = feedSize;
            req.candidateLimit = config.recommendation.vectorCandidates;
            req.enableExploration = false;
            req.enableDiversity = true;
            req.requestTime = sim.now();

            const RecommendationResponse resp = recommender->recommend(req);
            ASSERT_FALSE(resp.reels.empty());

            std::unordered_set<uint32_t> ids;
            std::unordered_map<uint32_t, uint32_t> perCreator;
            std::unordered_map<uint32_t, uint32_t> perTopic;
            for (const RankedReel &item : resp.reels) {
                EXPECT_TRUE(ids.insert(item.reelId.value).second)
                    << "duplicate reel " << item.reelId.value << " in feed";
                EXPECT_EQ(user.seenReels.count(item.reelId), 0u)
                    << "seen reel " << item.reelId.value << " served again";
                const Reel &reel = ds.reels[item.reelId.value];
                EXPECT_LE(++perCreator[reel.creatorId.value], config.diversity.maxPerCreator)
                    << "creator cap violated for creator " << reel.creatorId.value;
                EXPECT_LE(++perTopic[reel.primaryTopic.value], topicCap)
                    << "topic cap violated for topic " << reel.primaryTopic.value;
            }
            ++feedsChecked;

            // Consume the feed so seen-state, counters, and the clock advance realistically.
            for (const RankedReel &item : resp.reels) {
                Reel &reel = ds.reels[item.reelId.value];
                sim.step(user, ds.hiddenStates[user.id.value], reel,
                         ds.creators[reel.creatorId.value]);
            }
        }
    }
    EXPECT_EQ(feedsChecked, rounds * ds.users.size());
}

// The no-op regression contract (mirrors Phase 8's epsilon=0 check): hnsw_ranker_diversity with
// diversity.enabled=false and exploration.enabled=false must reproduce hnsw_ranker byte-for-byte
// on every deterministic CSV — proving the reranker gate, the FullRecommender's isolation-mode
// source list, and the request flag are all exactly inert.
TEST(DiversityPipelineTest, DiversityOffIsByteIdenticalToHnswRanker) {
    ExperimentConfig baseline = armConfig(RecommendationAlgorithm::HnswRanker);
    ExperimentConfig off = armConfig(RecommendationAlgorithm::HnswRankerDiversity);
    off.diversity.enabled = false;
    off.exploration.enabled = false;

    const fs::path rootA = fs::temp_directory_path() / "rr_div_off_baseline";
    const fs::path rootB = fs::temp_directory_path() / "rr_div_off_arm";
    const fs::path dirA = runArm(baseline, rootA);
    const fs::path dirB = runArm(off, rootB);

    for (const char *name : kDeterministicCsvs) {
        if (std::string(name) == "latency_metrics.csv") {
            continue; // wall-clock file, never byte-compared (D8 carve-out)
        }
        EXPECT_EQ(readFile(dirA / name), readFile(dirB / name)) << name << " differs";
    }
    fs::remove_all(rootA);
    fs::remove_all(rootB);
}

// Diversity ON must actually change feeds (the gate is live, not vacuously inert) and the run
// stays fully deterministic: two identical complete-system runs (ranker + exploration + fresh +
// diversity, TDD 16.6 "final initial system") produce byte-identical deterministic CSVs.
TEST(DiversityPipelineTest, CompleteSystemIsDeterministicAndDiversityGateIsLive) {
    ExperimentConfig complete = armConfig(RecommendationAlgorithm::HnswRankerDiversity);
    complete.exploration.enabled = true; // complete-initial-system mode

    const fs::path rootA = fs::temp_directory_path() / "rr_div_complete_a";
    const fs::path rootB = fs::temp_directory_path() / "rr_div_complete_b";
    const fs::path dirA = runArm(complete, rootA);
    const fs::path dirB = runArm(complete, rootB);
    for (const char *name : kDeterministicCsvs) {
        if (std::string(name) == "latency_metrics.csv") {
            continue;
        }
        EXPECT_EQ(readFile(dirA / name), readFile(dirB / name)) << name << " differs";
    }

    // Gate liveness: the same config with diversity disabled diverges on the feed-derived CSV.
    ExperimentConfig noDiversity = complete;
    noDiversity.diversity.enabled = false;
    const fs::path rootC = fs::temp_directory_path() / "rr_div_complete_c";
    const fs::path dirC = runArm(noDiversity, rootC);
    EXPECT_NE(readFile(dirA / "diversity_metrics.csv"), readFile(dirC / "diversity_metrics.csv"))
        << "diversity gate appears inert in the complete system";

    fs::remove_all(rootA);
    fs::remove_all(rootB);
    fs::remove_all(rootC);
}

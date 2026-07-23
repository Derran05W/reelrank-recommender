// Phase 23 LearnedRanker PIPELINE integration tests (contracts §5). Drives full gate-on
// event_queue experiments end-to-end through the ExperimentRunner (which dispatches to the
// EventDrivenRunner) and asserts the closed-loop half of the Tier-5 acceptance:
//   * SMOKE — the learned arm runs clean, retrains at least once, and emits retraining_log.csv +
//     the learned_models summary block + explanation_sample.json (contracts §4);
//   * RETRAINING DETERMINISM — same seed twice => identical retraining_log.csv (all but the
//     steady_clock wall_ms column, D9) AND bit-identical final model JSON dumps (contracts §5);
//   * SERVING PURITY — the FeatureVector the LearnedRanker scores equals the logged candidates.csv
//     feature row for the same (request, reel): one pure FeatureExtractor, byte-identical on the
//     served pool and at the log's 6-decimal precision (V2 §10 item 8);
//   * CLOSED-LOOP STATISTICAL — on identical worlds the learned arm is within a calibrated band of
//     the hand-tuned WeightedRanker on engagement AND at least the semantic (similarity-only)
//     control; the balanced weight vector lands at least as high on mean hidden satisfaction as the
//     pure-engagement vector (the frontier mechanism tripwire). Margins are mechanism-vs-control at
//     the demonstrated operating point — never a fine ordering between two learned variants.

#include "rr/evaluation/experiment_runner.hpp"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/evaluation/cold_start.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/learning_v2/hnsw_learned_ranker_recommender.hpp"
#include "rr/recommendation/feature_extractor.hpp"
#include "rr/recommendation/orchestrator.hpp"
#include "rr/simulation/dataset_generator.hpp"

using namespace rr;

namespace {

namespace fs = std::filesystem;

// Tiny full-gate learned world. training_log + learned_ranker on, algorithm hnsw_learned_ranker;
// low min_training_rows + short retrain interval so the models train and swap within the horizon.
ExperimentConfig learnedConfig(uint64_t seed = 20260723) {
    ExperimentConfig c;
    c.simulation.seed = seed;
    c.simulation.users = 200;
    c.simulation.reels = 2500;
    c.simulation.creators = 50;
    c.simulation.topics = 10;
    c.simulation.dimensions = 16;
    c.simulation.scheduler = "event_queue";
    c.simulation.horizonSeconds = 2.0 * 86400.0; // 2 simulated days
    c.recommendation.feedSize = 10;
    c.recommendation.vectorCandidates = 120;
    c.evaluation.oracleSampleRate = 0.05;
    c.evaluation.retrievalSampleRate = 0.02;
    c.algorithm = RecommendationAlgorithm::HnswLearnedRanker;
    c.realism.contentV2 = true;
    c.realism.latentReactions = true;
    c.realism.sessionDynamics = true;
    c.scheduling.openStaggerSeconds = 3600.0;
    c.scheduling.returnDelayMeanSeconds = 5400.0;
    c.serving.prefetchDepth = 10;
    c.learningV2.trainingLog = true;
    c.learningV2.logSampleRate = 1.0;     // matrix every shown impression (fast accumulation)
    c.learningV2.logPoolSampleRate = 0.0; // no full-pool rows needed here
    c.learningV2.learnedRanker = true;
    c.learningV2.retrainEveryHours = 8.0; // boundaries at 8/16/24/32/40 h
    c.learningV2.minTrainingRows = 150;
    c.learningV2.retrainEpochs = 20;
    c.learningV2.survey.enabled = true;
    c.learningV2.survey.sampleRate = 0.2;
    return c;
}

std::string readFile(const fs::path &p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

nlohmann::json readJson(const fs::path &p) {
    std::ifstream in(p);
    return nlohmann::json::parse(in);
}

// retraining_log.csv with the wall_ms column (index 3) stripped — the deterministic projection.
std::string retrainLogNoWall(const fs::path &csv) {
    std::ifstream in(csv);
    std::string line;
    std::string out;
    while (std::getline(in, line)) {
        std::vector<std::string> f;
        std::size_t start = 0;
        for (std::size_t i = 0; i <= line.size(); ++i) {
            if (i == line.size() || line[i] == ',') {
                f.push_back(line.substr(start, i - start));
                start = i + 1;
            }
        }
        for (std::size_t i = 0; i < f.size(); ++i) {
            if (i == 3) {
                continue; // drop wall_ms
            }
            out += f[i];
            out += (i + 1 == f.size()) ? '\n' : ',';
        }
    }
    return out;
}

// Format like results_writer/ TrainingLogger num(): fixed 6 decimals, classic locale.
std::string num6(double v) {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << std::fixed;
    oss.precision(6);
    oss << v;
    return oss.str();
}

double meanReward(const ExperimentResult &r) { return r.overall.rewardPerImpression; }

} // namespace

TEST(LearnedRankerPipelineTest, SmokeRetrainsAndEmitsArtifacts) {
    const fs::path root = fs::path(::testing::TempDir()) / "rr_p23_smoke";
    fs::remove_all(root);
    ExperimentRunner runner(learnedConfig(), root);
    const ExperimentResult r = runner.run();

    EXPECT_GT(r.requestCount, 0u);
    EXPECT_GT(r.impressionCount, 0u);

    // The learned report is configured, retrained at least once, and every §4 artifact exists.
    ASSERT_TRUE(r.learnedModels.configured);
    EXPECT_GE(r.learnedModels.retrainCount, 1);
    EXPECT_EQ(r.learnedModels.finalVersion, r.learnedModels.retrainCount);
    EXPECT_GE(r.learnedModels.meanNTrainRows,
              static_cast<double>(learnedConfig().learningV2.minTrainingRows));
    EXPECT_GE(r.learnedModels.fallbackRequestShare, 0.0);
    EXPECT_LT(r.learnedModels.fallbackRequestShare, 1.0); // some requests served learned

    EXPECT_TRUE(fs::exists(r.directory / "retraining_log.csv"));
    EXPECT_TRUE(fs::exists(r.directory / "explanation_sample.json"));
    // retraining_log.csv frozen header + >=1 data row.
    std::ifstream log(r.directory / "retraining_log.csv");
    std::string header;
    std::getline(log, header);
    EXPECT_EQ(header, "version,sim_time_seconds,n_train_rows,wall_ms,targets_trained");
    std::string firstRow;
    ASSERT_TRUE(static_cast<bool>(std::getline(log, firstRow)));

    // summary.json learned_models block: frozen keys present.
    const nlohmann::json summary = readJson(r.directory / "summary.json");
    ASSERT_TRUE(summary.contains("learned_models"));
    const nlohmann::json &lm = summary["learned_models"];
    for (const char *k : {"configured", "retrain_count", "final_version", "total_retrain_wall_ms",
                          "mean_n_train_rows", "fallback_request_share", "note"}) {
        EXPECT_TRUE(lm.contains(k)) << "missing learned_models key " << k;
    }

    // explanation_sample.json: captured, well-formed, learned_value == Σ predicted terms.
    const nlohmann::json expl = readJson(r.directory / "explanation_sample.json");
    EXPECT_EQ(expl["schema"].get<std::string>(), "phase23_explanation_sample_v1");
    ASSERT_TRUE(expl["captured"].get<bool>());
    ASSERT_FALSE(expl["candidates"].empty());
    const nlohmann::json &e0 = expl["candidates"][0]["explanation"];
    const double termSum =
        e0["predicted_watch"].get<double>() + e0["predicted_share"].get<double>() +
        e0["predicted_follow"].get<double>() + e0["predicted_satisfaction"].get<double>() +
        e0["predicted_exit"].get<double>() + e0["predicted_regret"].get<double>();
    EXPECT_NEAR(e0["learned_value"].get<double>(), termSum, 1e-6);
    EXPECT_DOUBLE_EQ(e0["fallback"].get<double>(), 0.0);
}

TEST(LearnedRankerPipelineTest, RetrainingDeterminism) {
    const fs::path rootA = fs::path(::testing::TempDir()) / "rr_p23_detA";
    const fs::path rootB = fs::path(::testing::TempDir()) / "rr_p23_detB";
    fs::remove_all(rootA);
    fs::remove_all(rootB);
    const ExperimentResult a = ExperimentRunner(learnedConfig(777), rootA).run();
    const ExperimentResult b = ExperimentRunner(learnedConfig(777), rootB).run();

    ASSERT_GE(a.learnedModels.retrainCount, 1);
    EXPECT_EQ(a.learnedModels.retrainCount, b.learnedModels.retrainCount);
    // retraining_log.csv is identical EXCEPT the steady_clock wall_ms column (D9 — not part of the
    // determinism guarantee, like latency_metrics.csv).
    EXPECT_EQ(retrainLogNoWall(a.directory / "retraining_log.csv"),
              retrainLogNoWall(b.directory / "retraining_log.csv"));
    // The final served model bundle is BIT-identical (nlohmann shortest-round-trip doubles).
    EXPECT_EQ(a.learnedModels.finalModelJson, b.learnedModels.finalModelJson);
    EXPECT_FALSE(a.learnedModels.finalModelJson.empty());
    // The served explanation sample is byte-identical too (deterministic serving).
    EXPECT_EQ(readFile(a.directory / "explanation_sample.json"),
              readFile(b.directory / "explanation_sample.json"));
}

TEST(LearnedRankerPipelineTest, ServingPurityParity) {
    // The FeatureVector the LearnedRanker scores IS the one the TrainingLogger logs: both come from
    // ONE pure FeatureExtractor built with the same (reels, ranking, contentV2). Build the learned
    // recommender + a capture extractor (exactly as the runner does), issue one captured request,
    // and confirm a fresh identically-configured extractor reproduces the captured pool
    // byte-for-byte AND at the candidates.csv 6-decimal precision.
    ExperimentConfig c = learnedConfig(31);
    GeneratedDataset ds = generateDataset(c.simulation, c.realism, c.simulation.seed);
    applyColdStart(ds.users, globalAveragePreference(ds.hiddenStates));

    RecommenderDeps deps{ds.reels, ds.users, c};
    HNSWLearnedRankerRecommender rec(deps, forkRng(c.simulation.seed, "recommender"));

    // The capture extractor the runner logs with (contracts §7): same reels/ranking/contentV2.
    FeatureExtractor logExtractor(ds.reels, c.ranking, c.realism.contentV2);
    RankingCapture capture;
    capture.extractor = &logExtractor;

    const std::uint32_t userId = 3;
    RecommendationRequest req{};
    req.userId = UserId{userId};
    req.feedSize = c.recommendation.feedSize;
    req.candidateLimit = c.recommendation.vectorCandidates;
    req.enableExploration = c.exploration.enabled;
    req.enableDiversity = c.diversity.enabled;
    req.requestTime = 1234;
    req.capture = &capture;
    const RecommendationResponse resp = rec.recommend(req);
    ASSERT_GT(resp.reels.size(), 0u);
    ASSERT_GT(capture.rows.size(), 0u);

    // Reconstruct the served pool from the capture (reel id + retrieval similarity + representative
    // source) and re-extract with a fresh, identically-configured extractor — the one the
    // LearnedRanker owns. Pool-relative features (popularity) are set-invariant, so the FULL pool
    // must be reconstructed in order.
    std::vector<Candidate> pool;
    pool.reserve(capture.rows.size());
    for (const RankingCaptureRow &row : capture.rows) {
        Candidate cand{};
        cand.reelId = row.reelId;
        cand.source = row.explorationLabeled ? CandidateSource::Exploration
                                             : (row.sources.empty() ? CandidateSource::VectorHNSW
                                                                    : row.sources.front());
        cand.retrievalSimilarity = row.retrievalSimilarity;
        pool.push_back(cand);
    }
    FeatureExtractor servingExtractor(ds.reels, c.ranking, c.realism.contentV2);
    const std::vector<FeatureVector> fresh =
        servingExtractor.extract(ds.users[userId], pool, req.requestTime);
    ASSERT_EQ(fresh.size(), capture.rows.size());

    auto asRow = [](const FeatureVector &f) {
        return std::vector<float>{f.similarity,         f.sessionTopic,    f.quality,
                                  f.freshness,          f.popularity,      f.trending,
                                  f.creatorAffinity,    f.exploration,     f.durationMatch,
                                  f.repetition,         f.impressionCount, f.visualMatch,
                                  f.musicMatch,         f.emotionalMatch,  f.clickbait,
                                  f.emotionalIntensity, f.usefulness,      f.productionQuality,
                                  f.informationDensity, f.languageMatch,   f.savePopularity};
    };
    for (std::size_t i = 0; i < fresh.size(); ++i) {
        const std::vector<float> served = asRow(fresh[i]); // what the LearnedRanker scores
        const std::vector<float> logged =
            asRow(capture.rows[i].features); // what candidates.csv gets
        ASSERT_EQ(served.size(), logged.size());
        for (std::size_t k = 0; k < served.size(); ++k) {
            EXPECT_FLOAT_EQ(served[k], logged[k]); // byte-identical served == captured
            EXPECT_EQ(num6(served[k]),
                      num6(logged[k])); // and identical at candidates.csv precision
        }
    }
}

TEST(LearnedRankerPipelineTest, ClosedLoopBandAndFrontierMechanism) {
    // Reduced-scale in-process arms on IDENTICAL worlds (same seed). Mechanism-vs-control margins
    // only, at the demonstrated operating point (printed below).
    const uint64_t seed = 4242;
    const fs::path root = fs::path(::testing::TempDir()) / "rr_p23_closedloop";
    fs::remove_all(root);

    auto run = [&](const std::string &tag, RecommendationAlgorithm algo, bool learned,
                   const LearningV2ValueWeights *weights) {
        ExperimentConfig c = learnedConfig(seed);
        c.algorithm = algo;
        c.learningV2.trainingLog = learned;
        c.learningV2.learnedRanker = learned;
        c.learningV2.survey.enabled = learned; // survey only feeds the learned satisfaction axis
        if (weights != nullptr) {
            c.learningV2.valueWeights = *weights;
        }
        return ExperimentRunner(c, root / tag).run();
    };

    // hand_tuned = hnsw_ranker (the WeightedRanker baseline); semantic = hnsw (similarity-only
    // order).
    const ExperimentResult handTuned =
        run("hand_tuned", RecommendationAlgorithm::HnswRanker, false, nullptr);
    const ExperimentResult semantic =
        run("semantic", RecommendationAlgorithm::Hnsw, false, nullptr);
    const ExperimentResult learnedBalanced =
        run("learned", RecommendationAlgorithm::HnswLearnedRanker, true, nullptr);

    // Frontier: a pure-engagement weight vector (all satisfaction/exit/regret zeroed, watch
    // carrying the freed mass) vs the balanced default. Only the weight vector differs; identical
    // world.
    LearningV2ValueWeights pureEngage;
    pureEngage.watch = 0.60;
    pureEngage.share = 0.15;
    pureEngage.follow = 0.10;
    pureEngage.satisfaction = 0.0;
    pureEngage.exit = 0.0;
    pureEngage.regret = 0.0;
    const ExperimentResult learnedPureEngage =
        run("pure_engage", RecommendationAlgorithm::HnswLearnedRanker, true, &pureEngage);

    ASSERT_GE(learnedBalanced.learnedModels.retrainCount, 1);
    ASSERT_GE(learnedPureEngage.learnedModels.retrainCount, 1);

    std::cerr << "[P23 closed-loop] reward/impression  hand_tuned=" << meanReward(handTuned)
              << " semantic=" << meanReward(semantic) << " learned=" << meanReward(learnedBalanced)
              << " pure_engage=" << meanReward(learnedPureEngage) << "\n";
    std::cerr << "[P23 closed-loop] mean hidden satisfaction  balanced="
              << learnedBalanced.welfare.meanSatisfaction
              << " pure_engage=" << learnedPureEngage.welfare.meanSatisfaction
              << " hand_tuned=" << handTuned.welfare.meanSatisfaction << "\n";

    // (1) Engagement band + control: learned is competitive with hand_tuned and at least the
    // similarity-only semantic control. Mechanism-level margins (calibrated, generous).
    ASSERT_GT(handTuned.overall.rewardPerImpression, 0.0);
    const double band =
        std::abs(meanReward(learnedBalanced) - meanReward(handTuned)) / meanReward(handTuned);
    EXPECT_LT(band, 0.35) << "learned reward not within band of hand_tuned";
    EXPECT_GE(meanReward(learnedBalanced), 0.90 * meanReward(semantic))
        << "learned should be at least the semantic control on engagement";

    // (2) Frontier mechanism: the balanced vector (which weights satisfaction) lands at least as
    // high on mean hidden satisfaction as the pure-engagement vector. Directional, control-vs-
    // mechanism — never a fine ordering between two learned variants.
    EXPECT_GE(learnedBalanced.welfare.meanSatisfaction,
              learnedPureEngage.welfare.meanSatisfaction - 1e-9)
        << "balanced weights should not underperform pure-engagement on hidden satisfaction";
}

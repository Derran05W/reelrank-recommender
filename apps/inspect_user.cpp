// inspect_user — dumps topic/reel/creator/user distribution summaries as JSON (Phase 2, task 7),
// and, with --explain-user, a ranked hnsw_ranker feed for one user with the per-feature
// contribution breakdown of every item (Phase 6, task 3 / TDD 14.4).
//
// A first-cut eyeballing tool: generates a full synthetic dataset via rr::generateDataset and
// reports a nearest-topic histogram (for reels and for users, by nearest topic centre) and a
// quality histogram (reel intrinsicQuality, creator baseQuality), plus basic scale counts. This is
// an inspection tool, not a benchmark (no timing claims) — see apps/benchmark_retrieval.cpp for
// that.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rr/core/embedding.hpp"
#include "rr/evaluation/cold_start.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/recommender.hpp"
#include "rr/recommendation/recommender_factory.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/simulator.hpp"

namespace {

// Index of the topic whose centre is closest (max dot product; embeddings are unit vectors so
// this ranks identically to nearest-by-Euclidean-distance, design decision D3) to `e`.
size_t nearestTopic(const rr::Embedding &e, const std::vector<rr::Topic> &topics) {
    size_t best = 0;
    float bestDot = -2.0f;
    for (size_t i = 0; i < topics.size(); ++i) {
        const float d = rr::dot(e, topics[i].centre);
        if (d > bestDot) {
            bestDot = d;
            best = i;
        }
    }
    return best;
}

std::vector<uint64_t> histogram(const std::vector<float> &values, int numBins) {
    std::vector<uint64_t> bins(static_cast<size_t>(numBins), 0);
    for (float v : values) {
        int bin = std::clamp(static_cast<int>(v * static_cast<float>(numBins)), 0, numBins - 1);
        ++bins[static_cast<size_t>(bin)];
    }
    return bins;
}

} // namespace

// --explain-user mode (Phase 6, task 3): cold-start the population, warm the simulation up with
// `warmupRounds` uniformly-random impressions per user (so popularity/trending counters and the
// per-user creator-affinity/duration signals are non-trivial), then serve ONE hnsw_ranker feed
// for the chosen user and dump it in the TDD 14.4 explanation shape. Deterministic for a given
// (config, seed): warm-up picks come from the dedicated "inspect_warmup" stream (D8).
nlohmann::json explainUser(const rr::ExperimentConfig &config, uint64_t seed, uint32_t userIdArg,
                           uint32_t warmupRounds) {
    rr::GeneratedDataset ds = rr::generateDataset(config.simulation, seed);
    if (userIdArg >= ds.users.size()) {
        std::cerr << "--explain-user " << userIdArg << " out of range (users: " << ds.users.size()
                  << ")\n";
        std::exit(2);
    }
    rr::applyColdStart(ds.users, rr::globalAveragePreference(ds.hiddenStates));

    rr::Simulator sim(config.behaviour, config.reward, rr::forkRng(seed, "behaviour"),
                      config.learning.recentWindow, config.ranking.trendingHalfLifeSeconds);
    rr::Rng warmupRng = rr::forkRng(seed, "inspect_warmup");
    for (uint32_t round = 0; round < warmupRounds; ++round) {
        for (rr::User &user : ds.users) {
            const uint32_t reelIdx = warmupRng.uniformInt(static_cast<uint32_t>(ds.reels.size()));
            rr::Reel &reel = ds.reels[reelIdx];
            if (!reel.active) {
                continue;
            }
            sim.step(user, ds.hiddenStates[user.id.value], reel, ds.creators[reel.creatorId.value]);
        }
    }

    rr::RecommenderDeps deps{ds.reels, ds.users, config};
    auto rec = rr::makeRecommender(rr::RecommendationAlgorithm::HnswRanker, deps,
                                   rr::forkRng(seed, "recommender"));

    rr::RecommendationRequest req{};
    req.userId = rr::UserId{userIdArg};
    req.feedSize = config.recommendation.feedSize;
    req.candidateLimit = config.recommendation.vectorCandidates;
    req.requestTime = sim.now();
    const rr::RecommendationResponse response = rec->recommend(req);

    nlohmann::json feed = nlohmann::json::array();
    for (const rr::RankedReel &item : response.reels) {
        const rr::Reel &reel = ds.reels[item.reelId.value];
        nlohmann::json contributions;
        for (const auto &[key, value] : item.featureContributions) {
            contributions[key] = value;
        }
        nlohmann::json sources = nlohmann::json::array();
        for (rr::CandidateSource s : item.sources) {
            sources.push_back(static_cast<int>(s));
        }
        feed.push_back({{"rank", item.rank},
                        {"reel_id", item.reelId.value},
                        {"creator_id", reel.creatorId.value},
                        {"primary_topic", reel.primaryTopic.value},
                        {"score", item.score},
                        {"sources", sources},
                        {"contributions", contributions}});
    }

    return nlohmann::json{{"seed", seed},
                          {"user_id", userIdArg},
                          {"warmup_rounds", warmupRounds},
                          {"request_time", req.requestTime},
                          {"algorithm", "hnsw_ranker"},
                          {"feed", feed}};
}

int main(int argc, char **argv) {
    std::string configPath;
    uint64_t seed = 42;
    std::string outPath;        // empty => stdout
    int64_t explainUserId = -1; // >= 0 => explanation mode
    uint32_t warmupRounds = 30;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << what << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--config") {
            configPath = next("--config");
        } else if (a == "--seed") {
            seed = std::stoull(next("--seed"));
        } else if (a == "--out") {
            outPath = next("--out");
        } else if (a == "--explain-user") {
            explainUserId = std::stoll(next("--explain-user"));
        } else if (a == "--warmup") {
            warmupRounds = static_cast<uint32_t>(std::stoul(next("--warmup")));
        } else if (a == "--help" || a == "-h") {
            std::cout << "usage: inspect_user [--config path] [--seed N] [--out path]\n"
                         "                    [--explain-user ID [--warmup ROUNDS]]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }

    rr::ExperimentConfig experiment; // defaults: 10k users / 100k reels / 5k creators / 32 topics.
    if (!configPath.empty()) {
        experiment = rr::loadExperimentConfig(configPath);
    }
    const rr::SimulationConfig &config = experiment.simulation;

    if (explainUserId >= 0) {
        const nlohmann::json out =
            explainUser(experiment, seed, static_cast<uint32_t>(explainUserId), warmupRounds);
        if (outPath.empty()) {
            std::cout << out.dump(2) << "\n";
        } else {
            std::filesystem::path p(outPath);
            if (p.has_parent_path()) {
                std::filesystem::create_directories(p.parent_path());
            }
            std::ofstream f(p);
            f << out.dump(2) << "\n";
            std::cerr << "wrote " << p.string() << "\n";
        }
        return 0;
    }

    rr::GeneratedDataset dataset = rr::generateDataset(config, seed);

    std::vector<uint64_t> reelTopicHistogram(dataset.topics.size(), 0);
    std::vector<float> reelQualities;
    reelQualities.reserve(dataset.reels.size());
    for (const auto &r : dataset.reels) {
        ++reelTopicHistogram[nearestTopic(r.embedding, dataset.topics)];
        reelQualities.push_back(r.intrinsicQuality);
    }

    std::vector<uint64_t> userTopicHistogram(dataset.topics.size(), 0);
    for (const auto &h : dataset.hiddenStates) {
        ++userTopicHistogram[nearestTopic(h.hiddenPreference, dataset.topics)];
    }

    std::vector<float> creatorQualities;
    creatorQualities.reserve(dataset.creators.size());
    for (const auto &c : dataset.creators) {
        creatorQualities.push_back(c.baseQuality);
    }

    nlohmann::json out;
    out["seed"] = seed;
    out["counts"] = {{"topics", dataset.topics.size()},
                     {"creators", dataset.creators.size()},
                     {"reels", dataset.reels.size()},
                     {"users", dataset.users.size()}};
    out["reel_nearest_topic_histogram"] = reelTopicHistogram;
    out["user_nearest_topic_histogram"] = userTopicHistogram;
    out["reel_quality_histogram"] = histogram(reelQualities, 10);
    out["creator_quality_histogram"] = histogram(creatorQualities, 10);

    if (outPath.empty()) {
        std::cout << out.dump(2) << "\n";
    } else {
        std::filesystem::path p(outPath);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream f(p);
        f << out.dump(2) << "\n";
        std::cerr << "wrote " << p.string() << "\n";
    }

    return 0;
}

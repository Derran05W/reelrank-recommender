#include "rr/learning/online_user_state_updater.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/evaluation/cold_start.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/effective_preference.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/simulation/simulator.hpp"

// Statistical / integration tests for online preference learning (Phase 7 exit criteria):
//   (a) convergence  - estimate moves toward hidden preference under liked content;
//   (b) divergence   - estimate moves away from a disliked-content centroid;
//   (c) reward improves - a greedy estimate-driven recommender earns more reward over time.
// Small fixed-seed datasets keep the suite fast (< ~15 s) and deterministic. Ground-truth hidden
// access here is the test driver's evaluation carve-out (TDD 18.2), never the recommender's.

using namespace rr;

namespace {

SimulationConfig convergenceConfig() {
    SimulationConfig c;
    c.reels = 2000;
    c.users = 50;
    c.creators = 200;
    c.topics = 32;
    c.dimensions = 64;
    return c;
}

// cos of two unit vectors.
float cosine(const Embedding &a, const Embedding &b) { return dot(a, b); }

// Reel indices sorted by ground-truth affinity dot(hidden, embedding). ascending=false gives the
// most-liked reels first; ascending=true the most-disliked. (Test-side hidden access only.)
std::vector<uint32_t> reelsByAffinity(const GeneratedDataset &ds, const HiddenUserState &hidden,
                                      bool ascending) {
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(ds.reels.size());
    for (uint32_t i = 0; i < ds.reels.size(); ++i) {
        scored.emplace_back(dot(hidden.hiddenPreference, ds.reels[i].embedding), i);
    }
    std::sort(scored.begin(), scored.end(), [ascending](const auto &x, const auto &y) {
        return ascending ? x.first < y.first : x.first > y.first;
    });
    std::vector<uint32_t> out;
    out.reserve(scored.size());
    for (const auto &p : scored) {
        out.push_back(p.second);
    }
    return out;
}

// Normalized centroid of the given reels' embeddings.
Embedding centroidOf(const GeneratedDataset &ds, const std::vector<uint32_t> &ids, size_t count) {
    const size_t dim = ds.reels[ids[0]].embedding.size();
    std::vector<double> acc(dim, 0.0);
    for (size_t k = 0; k < count; ++k) {
        const Embedding &e = ds.reels[ids[k]].embedding;
        for (size_t d = 0; d < dim; ++d) {
            acc[d] += static_cast<double>(e[d]);
        }
    }
    Embedding centroid(dim, 0.0f);
    for (size_t d = 0; d < dim; ++d) {
        centroid[d] = static_cast<float>(acc[d]);
    }
    normalize(centroid);
    return centroid;
}

} // namespace

// (a) Convergence: fresh cold-started users fed their most-liked unseen content converge toward
// their hidden preference. Two complementary, robust signals are checked at interaction-count
// checkpoints, averaged over >= 20 users:
//   * the LONG-TERM component's cosine to hidden rises monotonically across every checkpoint -- the
//     clean, slow-moving learning trajectory (eta = 0.02);
//   * the EFFECTIVE ESTIMATE's cosine rises fast and stays elevated. The estimate saturates within
//     ~25 interactions (its 0.35 session weight locks onto liked content in the first session) and
//     then plateaus, wobbling by a few hundredths as sessions rotate -- so its intermediate trend
//     is "climb then sustain", not strict monotonicity. We therefore assert a large early gain and
//     a sustained high plateau rather than a monotone estimate series.
TEST(LearningConvergence, EstimateConvergesTowardHiddenUnderPositiveFeedback) {
    const uint64_t seed = 20260711;
    GeneratedDataset ds = generateDataset(convergenceConfig(), seed);
    applyColdStart(ds.users, globalAveragePreference(ds.hiddenStates));

    LearningConfig learningCfg;
    RankingConfig rankingCfg;
    OnlineUserStateUpdater updater(ds.reels, learningCfg);
    Simulator sim(BehaviourConfig{}, RewardConfig{}, forkRng(seed, "behaviour"),
                  learningCfg.recentWindow, rankingCfg.trendingHalfLifeSeconds);

    const int U = 20;
    const int N = 200;
    const std::array<int, 5> checkpoints{0, 25, 50, 100, 200};
    std::array<double, 5> estSum{};
    std::array<double, 5> longTermSum{};

    for (int u = 0; u < U; ++u) {
        User &user = ds.users[u];
        const HiddenUserState &hidden = ds.hiddenStates[u];
        const std::vector<uint32_t> liked = reelsByAffinity(ds, hidden, /*ascending=*/false);

        size_t ptr = 0;
        size_t cpIdx = 0;
        auto record = [&]() {
            estSum[cpIdx] += cosine(effectivePreference(user), hidden.hiddenPreference);
            longTermSum[cpIdx] += cosine(user.longTermPreference, hidden.hiddenPreference);
            ++cpIdx;
        };
        record(); // step 0 (cold start)
        for (int s = 1; s <= N; ++s) {
            while (user.seenReels.count(ReelId{liked[ptr]})) {
                ++ptr;
            }
            Reel &reel = ds.reels[liked[ptr]];
            const Creator &creator = ds.creators[reel.creatorId.value];
            StepResult step = sim.step(user, hidden, reel, creator);
            updater.apply(user, reel, step.event);
            if (cpIdx < checkpoints.size() && s == checkpoints[cpIdx]) {
                record();
            }
        }
    }

    std::array<double, 5> estMean{};
    std::array<double, 5> longTermMean{};
    for (size_t i = 0; i < estMean.size(); ++i) {
        estMean[i] = estSum[i] / U;
        longTermMean[i] = longTermSum[i] / U;
    }

    // The long-term component's cosine to hidden increases monotonically across every checkpoint:
    // the clean convergence trajectory.
    for (size_t i = 0; i + 1 < longTermMean.size(); ++i) {
        EXPECT_GT(longTermMean[i + 1], longTermMean[i])
            << "long-term cos at checkpoint " << checkpoints[i + 1] << "=" << longTermMean[i + 1]
            << " not above checkpoint " << checkpoints[i] << "=" << longTermMean[i];
    }

    // The estimate learns quickly: a large gain over the first 25 interactions...
    EXPECT_GT(estMean[1] - estMean[0], 0.2)
        << "estimate cos cold-start=" << estMean[0] << " at 25 interactions=" << estMean[1];
    // ... and stays materially converged at every later checkpoint (no collapse back to prior).
    for (size_t i = 2; i < estMean.size(); ++i) {
        EXPECT_GT(estMean[i], estMean[0] + 0.25)
            << "estimate cos at checkpoint " << checkpoints[i] << "=" << estMean[i]
            << " not sustained above cold-start=" << estMean[0];
    }
    // ... for a substantial total gain from cold start to 200 interactions.
    EXPECT_GT(estMean.back() - estMean.front(), 0.2)
        << "start mean cos=" << estMean.front() << " end mean cos=" << estMean.back();
}

// (b) Divergence: users fed ONLY their most-disliked content move the estimate AWAY from the
// disliked-content centroid (cosine to that centroid decreases).
TEST(LearningConvergence, EstimateDivergesFromDislikedContent) {
    const uint64_t seed = 20260712;
    GeneratedDataset ds = generateDataset(convergenceConfig(), seed);
    applyColdStart(ds.users, globalAveragePreference(ds.hiddenStates));

    LearningConfig learningCfg;
    RankingConfig rankingCfg;
    OnlineUserStateUpdater updater(ds.reels, learningCfg);
    Simulator sim(BehaviourConfig{}, RewardConfig{}, forkRng(seed, "behaviour"),
                  learningCfg.recentWindow, rankingCfg.trendingHalfLifeSeconds);

    const int U = 20;
    const int N = 200;
    double startSum = 0.0;
    double endSum = 0.0;

    for (int u = 0; u < U; ++u) {
        User &user = ds.users[u];
        const HiddenUserState &hidden = ds.hiddenStates[u];
        const std::vector<uint32_t> disliked = reelsByAffinity(ds, hidden, /*ascending=*/true);
        const Embedding centroid = centroidOf(ds, disliked, N); // centroid of the fed content

        startSum += cosine(effectivePreference(user), centroid);
        for (int s = 0; s < N; ++s) {
            Reel &reel = ds.reels[disliked[s]]; // distinct reels, all initially unseen
            const Creator &creator = ds.creators[reel.creatorId.value];
            StepResult step = sim.step(user, hidden, reel, creator);
            updater.apply(user, reel, step.event);
        }
        endSum += cosine(effectivePreference(user), centroid);
    }

    const double startMean = startSum / U;
    const double endMean = endSum / U;
    EXPECT_LT(endMean, startMean - 0.20)
        << "start mean cos-to-disliked=" << startMean << " end mean cos-to-disliked=" << endMean;
}

// (c) Reward improves: a greedy mini-recommender selects the top-feedSize unseen reels by
// dot(estimatedPreference, embedding) (recommender-visible state only). With learning on, the mean
// reward per impression over the last quartile exceeds the first quartile, averaged over users.
TEST(LearningConvergence, GreedyRecommenderRewardImprovesOverTime) {
    const uint64_t seed = 20260713;
    GeneratedDataset ds = generateDataset(convergenceConfig(), seed);
    applyColdStart(ds.users, globalAveragePreference(ds.hiddenStates));

    LearningConfig learningCfg;
    RankingConfig rankingCfg;
    RecommendationConfig recCfg;
    OnlineUserStateUpdater updater(ds.reels, learningCfg);
    Simulator sim(BehaviourConfig{}, RewardConfig{}, forkRng(seed, "behaviour"),
                  learningCfg.recentWindow, rankingCfg.trendingHalfLifeSeconds);

    const int U = 25;
    const uint32_t feedSize = recCfg.feedSize; // 10
    const int rounds = 20;                     // rounds * feedSize = 200 impressions
    const int totalImpr = rounds * static_cast<int>(feedSize);
    const int quartile = totalImpr / 4;

    double firstQSum = 0.0;
    double lastQSum = 0.0;

    for (int u = 0; u < U; ++u) {
        User &user = ds.users[u];
        const HiddenUserState &hidden = ds.hiddenStates[u];

        std::vector<float> rewards;
        rewards.reserve(totalImpr);
        for (int round = 0; round < rounds; ++round) {
            // Greedy top-feedSize unseen by the recommender-visible estimate.
            std::vector<std::pair<float, uint32_t>> scored;
            scored.reserve(ds.reels.size());
            for (uint32_t i = 0; i < ds.reels.size(); ++i) {
                if (user.seenReels.count(ReelId{i})) {
                    continue;
                }
                scored.emplace_back(dot(effectivePreference(user), ds.reels[i].embedding), i);
            }
            const size_t take = std::min<size_t>(feedSize, scored.size());
            std::partial_sort(scored.begin(), scored.begin() + take, scored.end(),
                              [](const auto &x, const auto &y) { return x.first > y.first; });
            for (size_t j = 0; j < take; ++j) {
                Reel &reel = ds.reels[scored[j].second];
                const Creator &creator = ds.creators[reel.creatorId.value];
                StepResult step = sim.step(user, hidden, reel, creator);
                updater.apply(user, reel, step.event);
                rewards.push_back(step.event.reward);
            }
        }

        double firstQ = 0.0;
        double lastQ = 0.0;
        for (int i = 0; i < quartile; ++i) {
            firstQ += rewards[i];
            lastQ += rewards[rewards.size() - 1 - i];
        }
        firstQSum += firstQ / quartile;
        lastQSum += lastQ / quartile;
    }

    const double firstMean = firstQSum / U;
    const double lastMean = lastQSum / U;
    EXPECT_GT(lastMean, firstMean + 0.03)
        << "first-quartile mean reward=" << firstMean << " last-quartile mean reward=" << lastMean;
}

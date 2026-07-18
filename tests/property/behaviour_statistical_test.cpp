#include "rr/simulation/behaviour_model.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "rr/domain/creator.hpp"
#include "rr/domain/reel.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"

using namespace rr;

namespace {

// A small-but-stable dataset for the statistical sweeps (spec: ~2k reels / 500 users / dim 32).
SimulationConfig statConfig() {
    SimulationConfig c;
    c.reels = 2000;
    c.users = 500;
    c.creators = 200;
    c.topics = 32;
    c.dimensions = 32;
    return c;
}

struct Sample {
    float affinity;
    float watchRatio;
    bool instantSkip;
    bool completed;
    bool liked;
};

// Simulate `n` random (user, reel) pairs from `ds`. `pick` chooses pairs; `behaviour` drives the
// stochastic reactions. Both are explicit so the whole sweep is reproducible.
std::vector<Sample> sweep(const GeneratedDataset &ds, const BehaviourModel &model, Rng &pick,
                          Rng &behaviour, int n) {
    std::vector<Sample> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        const auto &h = ds.hiddenStates[pick.uniformInt(ds.hiddenStates.size())];
        const auto &r = ds.reels[pick.uniformInt(ds.reels.size())];
        const auto &c = ds.creators[r.creatorId.value];
        BehaviourOutcome o = model.simulate(h, r, c, behaviour);
        out.push_back({o.baseAffinity, o.watchRatio, o.instantSkip, o.completed, o.liked});
    }
    return out;
}

bool sameOutcome(const BehaviourOutcome &a, const BehaviourOutcome &b) {
    return a.baseAffinity == b.baseAffinity && a.behaviourScore == b.behaviourScore &&
           a.instantSkip == b.instantSkip && a.completed == b.completed && a.rewatch == b.rewatch &&
           a.liked == b.liked && a.shared == b.shared && a.followed == b.followed &&
           a.notInterested == b.notInterested && a.watchRatio == b.watchRatio &&
           a.watchSeconds == b.watchSeconds && a.primaryType == b.primaryType;
}

} // namespace

// (a) Mean watch ratio of the top-affinity decile clearly exceeds the bottom decile.
TEST(BehaviourStatistical, WatchRatioTopDecileBeatsBottomDecile) {
    GeneratedDataset ds = generateDataset(statConfig(), 20250710);
    BehaviourConfig bcfg;
    BehaviourModel model(bcfg);
    Rng pick(1);
    Rng behaviour(2);

    std::vector<Sample> s = sweep(ds, model, pick, behaviour, 60000);
    std::sort(s.begin(), s.end(),
              [](const Sample &a, const Sample &b) { return a.affinity < b.affinity; });

    const size_t d = s.size() / 10;
    double bottom = 0.0;
    double top = 0.0;
    for (size_t i = 0; i < d; ++i) {
        bottom += s[i].watchRatio;
        top += s[s.size() - 1 - i].watchRatio;
    }
    bottom /= static_cast<double>(d);
    top /= static_cast<double>(d);

    EXPECT_GT(top - bottom, 0.4) << "top decile mean=" << top << " bottom decile mean=" << bottom;
}

// (b) Instant-skip rate is monotonically decreasing across affinity quartiles.
TEST(BehaviourStatistical, InstantSkipRateMonotoneDecreasingByAffinityQuartile) {
    GeneratedDataset ds = generateDataset(statConfig(), 20250710);
    BehaviourConfig bcfg;
    BehaviourModel model(bcfg);
    Rng pick(3);
    Rng behaviour(4);

    std::vector<Sample> s = sweep(ds, model, pick, behaviour, 80000);
    std::sort(s.begin(), s.end(),
              [](const Sample &a, const Sample &b) { return a.affinity < b.affinity; });

    const size_t q = s.size() / 4;
    std::array<double, 4> skipRate{};
    for (int quart = 0; quart < 4; ++quart) {
        size_t cnt = 0;
        size_t skips = 0;
        for (size_t i = quart * q; i < (quart + 1) * q; ++i) {
            ++cnt;
            if (s[i].instantSkip)
                ++skips;
        }
        skipRate[quart] = static_cast<double>(skips) / static_cast<double>(cnt);
    }

    for (int quart = 0; quart < 3; ++quart) {
        EXPECT_GT(skipRate[quart], skipRate[quart + 1])
            << "quartile " << quart << " rate=" << skipRate[quart] << " vs " << skipRate[quart + 1];
    }
    // And a meaningful total drop from lowest to highest affinity.
    EXPECT_GT(skipRate[0] - skipRate[3], 0.2);
}

// (c) Like rate among completed watches exceeds the overall rate; non-completed watches never like.
TEST(BehaviourStatistical, LikeRateHigherAmongCompletedWatches) {
    GeneratedDataset ds = generateDataset(statConfig(), 20250710);
    BehaviourConfig bcfg;
    BehaviourModel model(bcfg);
    Rng pick(5);
    Rng behaviour(6);

    std::vector<Sample> s = sweep(ds, model, pick, behaviour, 80000);

    size_t total = s.size();
    size_t completed = 0, likedCompleted = 0, likedTotal = 0, likedNonCompleted = 0;
    for (const Sample &x : s) {
        if (x.liked)
            ++likedTotal;
        if (x.completed) {
            ++completed;
            if (x.liked)
                ++likedCompleted;
        } else if (x.liked) {
            ++likedNonCompleted;
        }
    }
    ASSERT_GT(completed, 0u);
    const double completedLikeRate = static_cast<double>(likedCompleted) / completed;
    const double overallLikeRate = static_cast<double>(likedTotal) / total;

    EXPECT_EQ(likedNonCompleted, 0u); // like requires a completed watch
    EXPECT_GT(completedLikeRate, overallLikeRate + 0.02)
        << "completed=" << completedLikeRate << " overall=" << overallLikeRate;
}

// (d) Determinism: identical seed => bit-identical outcome sequence, swept over >= 20 seeds.
TEST(BehaviourStatistical, DeterministicOutcomeSequenceAcrossSeeds) {
    SimulationConfig cfg;
    cfg.reels = 300;
    cfg.users = 150;
    cfg.creators = 40;
    cfg.topics = 16;
    cfg.dimensions = 32;
    GeneratedDataset ds = generateDataset(cfg, 99);

    BehaviourConfig bcfg;
    BehaviourModel model(bcfg);

    // Fixed pair schedule (same across the two runs of every seed).
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    {
        Rng pick(7);
        for (int i = 0; i < 3000; ++i) {
            pairs.emplace_back(static_cast<uint32_t>(pick.uniformInt(ds.hiddenStates.size())),
                               static_cast<uint32_t>(pick.uniformInt(ds.reels.size())));
        }
    }

    for (int k = 0; k < 24; ++k) {
        const uint64_t seed = 1000ULL + static_cast<uint64_t>(k) * 6151ULL;
        Rng r1(seed);
        Rng r2(seed);
        for (const auto &p : pairs) {
            const auto &h = ds.hiddenStates[p.first];
            const auto &r = ds.reels[p.second];
            const auto &c = ds.creators[r.creatorId.value];
            BehaviourOutcome o1 = model.simulate(h, r, c, r1);
            BehaviourOutcome o2 = model.simulate(h, r, c, r2);
            ASSERT_TRUE(sameOutcome(o1, o2)) << "seed " << seed;
        }
    }
}

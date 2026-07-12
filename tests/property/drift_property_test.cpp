// Property tests for rr::DriftScheduler (Phase 10, TDD 24.3 seed sweep / D7). Over many seeds and
// randomly built valid drift configs the applied preference must stay finite + unit-length (D5)
// and be deterministic (D8); and the cohort test must partition any user population exactly. All
// randomness goes through rr::Rng (std::*_distribution is banned by D8).
#include "rr/simulation/drift_scheduler.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/hidden_user_state.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/topic_generator.hpp"

using namespace rr;

namespace {

constexpr uint32_t kTopics = 16;
constexpr uint32_t kDim = 32;

std::vector<Topic> makeTopics(uint64_t seed) {
    SimulationConfig cfg;
    cfg.topics = kTopics;
    cfg.dimensions = kDim;
    Rng rng = forkRng(seed, "topics");
    return generateTopics(cfg, rng);
}

// Build a random VALID drift event: 2-4 distinct topics, strictly-positive weights, a cohort that
// covers the whole population so the event always applies.
DriftEvent randomEvent(Rng &rng, uint32_t nTopics) {
    DriftEvent e;
    e.atInteraction = static_cast<uint32_t>(rng.uniformInt(1000));
    e.cohortLo = 0.0;
    e.cohortHi = 1.0;
    const uint32_t count = 2 + static_cast<uint32_t>(rng.uniformInt(3)); // 2..4
    std::vector<bool> used(nTopics, false);
    while (e.topicMix.size() < count) {
        const uint32_t t = static_cast<uint32_t>(rng.uniformInt(nTopics));
        if (used[t]) {
            continue;
        }
        used[t] = true;
        e.topicMix.push_back(DriftTopicWeight{t, rng.uniform(0.1, 2.0)});
    }
    return e;
}

HiddenUserState freshHidden(uint32_t userId) {
    HiddenUserState h;
    h.userId = UserId{userId};
    h.hiddenPreference.assign(kDim, 0.0f);
    h.hiddenPreference[0] = 1.0f; // arbitrary valid starting vector
    return h;
}

class DriftSeedSweep : public ::testing::TestWithParam<uint64_t> {};

} // namespace

// Applied preference is finite and unit-length within 1e-4 (D5), for random valid configs over
// generated topic sets, across the seed sweep.
TEST_P(DriftSeedSweep, AppliedPreferenceIsValidUnitVector) {
    const uint64_t seed = GetParam();
    const std::vector<Topic> topics = makeTopics(seed);
    Rng rng = forkRng(seed, "drift-cfg");

    for (int trial = 0; trial < 8; ++trial) {
        DriftEvent e = randomEvent(rng, kTopics);
        DriftScheduler sched(DriftConfig{{e}}, topics);

        for (uint32_t uid : {1u, 2u, 3u, 100u, 5000u}) {
            HiddenUserState h = freshHidden(uid);
            ASSERT_TRUE(sched.maybeApply(h, e.atInteraction)) << "seed " << seed;
            ASSERT_TRUE(isValid(h.hiddenPreference))
                << "seed " << seed << " trial " << trial << " uid " << uid;
            const float norm = std::sqrt(dot(h.hiddenPreference, h.hiddenPreference));
            EXPECT_NEAR(norm, 1.0f, 1e-4f) << "seed " << seed;
            EXPECT_EQ(h.preferredTopics.size(), e.topicMix.size());
        }
    }
}

// Determinism (D8): identical applications produce bit-identical embeddings, whether re-applied
// to a second copy or produced by a second scheduler built from the same config.
TEST_P(DriftSeedSweep, ApplicationIsDeterministic) {
    const uint64_t seed = GetParam();
    const std::vector<Topic> topics = makeTopics(seed);
    Rng rng = forkRng(seed, "drift-cfg");

    for (int trial = 0; trial < 8; ++trial) {
        const DriftEvent e = randomEvent(rng, kTopics);
        const DriftConfig cfg{{e}};
        DriftScheduler schedA(cfg, topics);
        DriftScheduler schedB(cfg, topics); // independent build, same inputs

        for (uint32_t uid : {7u, 42u, 999u, 12345u}) {
            HiddenUserState a = freshHidden(uid);
            HiddenUserState b = freshHidden(uid);
            ASSERT_TRUE(schedA.maybeApply(a, e.atInteraction));
            ASSERT_TRUE(schedB.maybeApply(b, e.atInteraction));
            EXPECT_EQ(a.hiddenPreference, b.hiddenPreference) << "seed " << seed;
            EXPECT_EQ(a.preferredTopics, b.preferredTopics) << "seed " << seed;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SeedSweep, DriftSeedSweep,
                         ::testing::Range<uint64_t>(1, 25)); // 24 seeds (>= 20 per TDD 24.3)

// Cohort partition (seed-independent, driven purely by hash01): a set of disjoint [lo, hi) ranges
// covering [0, 1) assigns every user to exactly one cohort.
TEST(DriftPropertyPartition, DisjointRangesPartitionEveryUser) {
    const std::vector<std::pair<double, double>> quarters = {
        {0.0, 0.25}, {0.25, 0.5}, {0.5, 0.75}, {0.75, 1.0}};

    // everApplies over the union covers all users; inCohort assigns exactly one quarter each.
    std::vector<DriftEvent> events;
    for (const auto &q : quarters) {
        events.push_back(DriftEvent{0, q.first, q.second, {DriftTopicWeight{0, 1.0}}});
    }
    // A single-topic mix over one basis topic keeps construction valid.
    std::vector<Topic> topics;
    {
        Embedding c(4, 0.0f);
        c[0] = 1.0f;
        topics.push_back(Topic{TopicId{0}, std::move(c)});
    }
    DriftScheduler sched(DriftConfig{events}, topics);

    for (uint32_t uid = 0; uid < 10000; ++uid) {
        int hits = 0;
        for (const auto &q : quarters) {
            if (DriftScheduler::inCohort(UserId{uid}, q.first, q.second)) {
                ++hits;
            }
        }
        ASSERT_EQ(hits, 1) << "uid " << uid << " landed in " << hits << " quarters";
        ASSERT_TRUE(sched.everApplies(UserId{uid})) << "uid " << uid;
    }
}

// A [0, 0.5) cohort over 10k sequential UserIds captures ~50% (within +/- 5%).
TEST(DriftPropertyPartition, HalfCohortCapturesAboutHalf) {
    uint32_t captured = 0;
    const uint32_t n = 10000;
    for (uint32_t uid = 0; uid < n; ++uid) {
        if (DriftScheduler::inCohort(UserId{uid}, 0.0, 0.5)) {
            ++captured;
        }
    }
    const double frac = static_cast<double>(captured) / n;
    EXPECT_NEAR(frac, 0.5, 0.05) << "captured " << captured << " of " << n;
}

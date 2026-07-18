// Unit tests for rr::DriftScheduler (Phase 10, TDD 11.4). Covers constructor validation, the
// drift application math, exactly-once event scheduling, [lo, hi) cohort membership with pinned
// hash01 golden values, multi-event ordering/collision semantics, and the unconfigured no-op
// contract. House style mirrors the other tests/unit/*_test.cpp files.
#include "rr/simulation/drift_scheduler.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"

using namespace rr;

namespace {

// Standard-basis topics of a given dimension: topic i's centre is e_i (already unit length). This
// makes the drift math trivially checkable — normalize(sum w_i e_i) has component i proportional to
// w_i, so ratios between components equal ratios between weights.
std::vector<Topic> basisTopics(uint32_t n, uint32_t dim) {
    std::vector<Topic> topics;
    for (uint32_t i = 0; i < n; ++i) {
        Embedding c(dim, 0.0f);
        c[i % dim] = 1.0f;
        topics.push_back(Topic{TopicId{i}, std::move(c)});
    }
    return topics;
}

DriftTopicWeight tw(uint32_t topic, double weight) { return DriftTopicWeight{topic, weight}; }

// A HiddenUserState pre-populated with distinctive sentinel values on every field, so an
// application test can assert exactly which fields change and which are left untouched.
HiddenUserState sentinelHidden(uint32_t userId, uint32_t dim) {
    HiddenUserState h;
    h.userId = UserId{userId};
    h.hiddenPreference.assign(dim, 0.111f);
    h.preferredTopics = {TopicId{97}, TopicId{98}};
    h.preferenceConcentration = 2.5f;
    h.exploreWillingness = 0.33f;
    h.avgSessionLength = 17.0f;
    h.likePropensity = 0.12f;
    h.sharePropensity = 0.04f;
    h.durationTolerance = 0.7f;
    h.preferenceStability = 0.9f;
    return h;
}

// The reference target the scheduler must produce, computed by the identical float operations the
// implementation uses (float accumulation then normalize) so EXPECT_FLOAT_EQ is exact.
Embedding referenceTarget(const std::vector<DriftTopicWeight> &mix,
                          const std::vector<Topic> &topics, uint32_t dim) {
    Embedding target(dim, 0.0f);
    for (const DriftTopicWeight &w : mix) {
        const Embedding &centre = topics[w.topic].centre;
        const float weight = static_cast<float>(w.weight);
        for (uint32_t d = 0; d < dim; ++d) {
            target[d] += weight * centre[d];
        }
    }
    normalize(target);
    return target;
}

DriftConfig oneEvent(uint32_t at, double lo, double hi, std::vector<DriftTopicWeight> mix) {
    DriftEvent e;
    e.atInteraction = at;
    e.cohortLo = lo;
    e.cohortHi = hi;
    e.topicMix = std::move(mix);
    return DriftConfig{{e}};
}

} // namespace

// --- Constructor validation: one throwing case per rule in the frozen header -------------------

TEST(DriftSchedulerTest, ThrowsOnEmptyTopicMix) {
    const auto topics = basisTopics(4, 8);
    DriftConfig cfg = oneEvent(10, 0.0, 1.0, {}); // empty mix
    EXPECT_THROW(DriftScheduler(cfg, topics), std::invalid_argument);
}

TEST(DriftSchedulerTest, ThrowsOnZeroWeight) {
    const auto topics = basisTopics(4, 8);
    DriftConfig cfg = oneEvent(10, 0.0, 1.0, {tw(0, 0.5), tw(1, 0.0)});
    EXPECT_THROW(DriftScheduler(cfg, topics), std::invalid_argument);
}

TEST(DriftSchedulerTest, ThrowsOnNegativeWeight) {
    const auto topics = basisTopics(4, 8);
    DriftConfig cfg = oneEvent(10, 0.0, 1.0, {tw(0, -0.25)});
    EXPECT_THROW(DriftScheduler(cfg, topics), std::invalid_argument);
}

TEST(DriftSchedulerTest, ThrowsOnNonFiniteWeight) {
    const auto topics = basisTopics(4, 8);
    for (double bad :
         {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        DriftConfig cfg = oneEvent(10, 0.0, 1.0, {tw(0, bad)});
        EXPECT_THROW(DriftScheduler(cfg, topics), std::invalid_argument);
    }
}

TEST(DriftSchedulerTest, ThrowsOnUnknownTopicId) {
    const auto topics = basisTopics(4, 8); // ids 0..3 exist
    DriftConfig cfg = oneEvent(10, 0.0, 1.0, {tw(0, 0.5), tw(99, 0.5)});
    EXPECT_THROW(DriftScheduler(cfg, topics), std::invalid_argument);
}

TEST(DriftSchedulerTest, ThrowsOnCohortLoBelowZero) {
    const auto topics = basisTopics(4, 8);
    DriftConfig cfg = oneEvent(10, -0.01, 0.5, {tw(0, 1.0)});
    EXPECT_THROW(DriftScheduler(cfg, topics), std::invalid_argument);
}

TEST(DriftSchedulerTest, ThrowsOnCohortHiAboveOne) {
    const auto topics = basisTopics(4, 8);
    DriftConfig cfg = oneEvent(10, 0.5, 1.01, {tw(0, 1.0)});
    EXPECT_THROW(DriftScheduler(cfg, topics), std::invalid_argument);
}

TEST(DriftSchedulerTest, ThrowsOnCohortLoGreaterEqualHi) {
    const auto topics = basisTopics(4, 8);
    EXPECT_THROW(DriftScheduler(oneEvent(10, 0.5, 0.5, {tw(0, 1.0)}), topics),
                 std::invalid_argument); // lo == hi
    EXPECT_THROW(DriftScheduler(oneEvent(10, 0.7, 0.3, {tw(0, 1.0)}), topics),
                 std::invalid_argument); // lo > hi
}

TEST(DriftSchedulerTest, ValidConfigConstructsAndReportsConfigured) {
    const auto topics = basisTopics(4, 8);
    DriftScheduler sched(oneEvent(10, 0.0, 1.0, {tw(0, 1.0)}), topics);
    EXPECT_TRUE(sched.configured());
}

// --- Application math ---------------------------------------------------------------------------

TEST(DriftSchedulerTest, AppliesNormalizedWeightedMixAndReplacesTopics) {
    const uint32_t dim = 4;
    const auto topics = basisTopics(4, dim);
    const std::vector<DriftTopicWeight> mix = {tw(0, 0.6), tw(1, 0.3), tw(2, 0.1)};
    DriftScheduler sched(oneEvent(5, 0.0, 1.0, mix), topics);

    HiddenUserState h = sentinelHidden(/*userId=*/7, dim);
    const HiddenUserState before = h;
    ASSERT_TRUE(sched.maybeApply(h, 5));

    // hiddenPreference == normalize(sum w*centre), asserted per dimension.
    const Embedding expected = referenceTarget(mix, topics, dim);
    ASSERT_EQ(h.hiddenPreference.size(), expected.size());
    for (uint32_t d = 0; d < dim; ++d) {
        EXPECT_FLOAT_EQ(h.hiddenPreference[d], expected[d]) << "dim " << d;
    }
    // Independent structural checks over the standard basis: unit length, component ratios track
    // weight ratios, and the unused dimension is exactly zero.
    EXPECT_TRUE(isValid(h.hiddenPreference));
    EXPECT_NEAR(h.hiddenPreference[0] / h.hiddenPreference[1], 2.0f, 1e-5f); // 0.6/0.3
    EXPECT_NEAR(h.hiddenPreference[1] / h.hiddenPreference[2], 3.0f, 1e-4f); // 0.3/0.1
    EXPECT_FLOAT_EQ(h.hiddenPreference[3], 0.0f);

    // preferredTopics replaced by the mix's ids in config order.
    EXPECT_EQ(h.preferredTopics, (std::vector<TopicId>{TopicId{0}, TopicId{1}, TopicId{2}}));

    // Every behavioural trait field and the userId are untouched.
    EXPECT_EQ(h.userId, before.userId);
    EXPECT_FLOAT_EQ(h.preferenceConcentration, before.preferenceConcentration);
    EXPECT_FLOAT_EQ(h.exploreWillingness, before.exploreWillingness);
    EXPECT_FLOAT_EQ(h.avgSessionLength, before.avgSessionLength);
    EXPECT_FLOAT_EQ(h.likePropensity, before.likePropensity);
    EXPECT_FLOAT_EQ(h.sharePropensity, before.sharePropensity);
    EXPECT_FLOAT_EQ(h.durationTolerance, before.durationTolerance);
    EXPECT_FLOAT_EQ(h.preferenceStability, before.preferenceStability);
}

// --- Scheduling: exactly-once firing at atInteraction ------------------------------------------

TEST(DriftSchedulerTest, FiresAtExactlyAtInteraction) {
    const auto topics = basisTopics(4, 8);
    DriftScheduler sched(oneEvent(5, 0.0, 1.0, {tw(0, 1.0)}), topics);

    HiddenUserState h = sentinelHidden(7, 8);
    HiddenUserState atMinus1 = h;
    EXPECT_FALSE(sched.maybeApply(atMinus1, 4)); // at-1: no fire
    EXPECT_EQ(atMinus1.preferredTopics, h.preferredTopics);

    HiddenUserState atExact = h;
    EXPECT_TRUE(sched.maybeApply(atExact, 5)); // exact: fires

    HiddenUserState atPlus1 = h;
    EXPECT_FALSE(sched.maybeApply(atPlus1, 6)); // at+1: no fire
    EXPECT_EQ(atPlus1.preferredTopics, h.preferredTopics);
}

TEST(DriftSchedulerTest, EventAtZeroFiresOnFirstCall) {
    const auto topics = basisTopics(4, 8);
    DriftScheduler sched(oneEvent(0, 0.0, 1.0, {tw(0, 1.0)}), topics);
    HiddenUserState h = sentinelHidden(7, 8);
    EXPECT_TRUE(sched.maybeApply(h, 0));
}

TEST(DriftSchedulerTest, AtInteractionBeyondBudgetNeverFires) {
    const auto topics = basisTopics(4, 8);
    DriftScheduler sched(oneEvent(1000000, 0.0, 1.0, {tw(0, 1.0)}), topics);
    HiddenUserState h = sentinelHidden(7, 8);
    const HiddenUserState before = h;
    for (uint32_t t : {0u, 1u, 5u, 200u, 999u, 500000u}) {
        EXPECT_FALSE(sched.maybeApply(h, t)) << "t=" << t;
    }
    EXPECT_EQ(h.preferredTopics, before.preferredTopics);
    EXPECT_EQ(h.hiddenPreference, before.hiddenPreference);
}

// --- Cohort membership: [lo, hi) semantics, golden hash01 tripwire, consistency -----------------

// Pinned hash01 golden values (SplitMix64 finalizer). For the fixed cohort [0.25, 0.5):
//   userId 4  -> hash01 ~= 0.4315 -> IN
//   userId 5  -> hash01 ~= 0.3868 -> IN
//   userId 3  -> hash01 ~= 0.1135 -> OUT
//   userId 0  -> hash01 ~= 0.8833 -> OUT
// These are a cross-platform regression tripwire: if hash01's constants ever change, they break.
TEST(DriftSchedulerTest, CohortHash01GoldenValues) {
    const double lo = 0.25, hi = 0.5;
    EXPECT_TRUE(DriftScheduler::inCohort(UserId{4}, lo, hi));
    EXPECT_TRUE(DriftScheduler::inCohort(UserId{5}, lo, hi));
    EXPECT_FALSE(DriftScheduler::inCohort(UserId{3}, lo, hi));
    EXPECT_FALSE(DriftScheduler::inCohort(UserId{0}, lo, hi));

    // Tighter brackets pin the actual hash01 magnitudes, not just the [0.25,0.5) verdict.
    EXPECT_TRUE(DriftScheduler::inCohort(UserId{0}, 0.88, 0.89)); // ~0.88331
    EXPECT_FALSE(DriftScheduler::inCohort(UserId{0}, 0.87, 0.88));
    EXPECT_TRUE(DriftScheduler::inCohort(UserId{3}, 0.11, 0.12));  // ~0.11345
    EXPECT_TRUE(DriftScheduler::inCohort(UserId{10}, 0.03, 0.04)); // ~0.03331
}

TEST(DriftSchedulerTest, CohortBoundaryIsLoInclusiveHiExclusive) {
    // userId 4's hash01 is ~0.43145581774497377. Anchoring the boundary exactly on it verifies
    // lo-inclusive / hi-exclusive without depending on any float slack.
    const UserId u{4};
    const double h = 0.43145581774497377;
    EXPECT_TRUE(DriftScheduler::inCohort(u, h, 1.0));  // h >= lo (== h): included
    EXPECT_FALSE(DriftScheduler::inCohort(u, 0.0, h)); // h <  hi (== h): excluded
}

TEST(DriftSchedulerTest, EverAppliesMatchesInCohort) {
    const auto topics = basisTopics(4, 8);
    DriftScheduler sched(oneEvent(10, 0.25, 0.5, {tw(0, 1.0)}), topics);
    for (uint32_t uid : {0u, 3u, 4u, 5u, 10u, 42u, 100u, 777u}) {
        EXPECT_EQ(sched.everApplies(UserId{uid}), DriftScheduler::inCohort(UserId{uid}, 0.25, 0.5))
            << "uid=" << uid;
    }
}

// --- Multiple events: collision (last-wins), disjoint cohorts, firstDriftInteraction ------------

TEST(DriftSchedulerTest, SameInteractionOverlappingCohortsLastWins) {
    const uint32_t dim = 4;
    const auto topics = basisTopics(4, dim);
    DriftEvent e0{10, 0.0, 1.0, {tw(0, 1.0)}}; // topic 0
    DriftEvent e1{10, 0.0, 1.0, {tw(1, 1.0)}}; // topic 1, same interaction, overlapping cohort
    DriftScheduler sched(DriftConfig{{e0, e1}}, topics);

    HiddenUserState h = sentinelHidden(7, dim);
    ASSERT_TRUE(sched.maybeApply(h, 10));
    // Last event in config order wins.
    EXPECT_EQ(h.preferredTopics, (std::vector<TopicId>{TopicId{1}}));
    const Embedding expected = referenceTarget({tw(1, 1.0)}, topics, dim);
    EXPECT_EQ(h.hiddenPreference, expected);
}

TEST(DriftSchedulerTest, DisjointCohortsEachUserGetsOwnMix) {
    const uint32_t dim = 4;
    const auto topics = basisTopics(4, dim);
    DriftEvent lowHalf{10, 0.0, 0.5, {tw(0, 1.0)}};  // topic 0 for hash01 < 0.5
    DriftEvent highHalf{10, 0.5, 1.0, {tw(1, 1.0)}}; // topic 1 for hash01 >= 0.5
    DriftScheduler sched(DriftConfig{{lowHalf, highHalf}}, topics);

    // userId 3 -> hash01 ~0.1135 (low half); userId 0 -> hash01 ~0.8833 (high half).
    ASSERT_TRUE(DriftScheduler::inCohort(UserId{3}, 0.0, 0.5));
    ASSERT_TRUE(DriftScheduler::inCohort(UserId{0}, 0.5, 1.0));

    HiddenUserState lowUser = sentinelHidden(3, dim);
    HiddenUserState highUser = sentinelHidden(0, dim);
    ASSERT_TRUE(sched.maybeApply(lowUser, 10));
    ASSERT_TRUE(sched.maybeApply(highUser, 10));

    EXPECT_EQ(lowUser.preferredTopics, (std::vector<TopicId>{TopicId{0}}));
    EXPECT_EQ(highUser.preferredTopics, (std::vector<TopicId>{TopicId{1}}));
    EXPECT_EQ(lowUser.hiddenPreference, referenceTarget({tw(0, 1.0)}, topics, dim));
    EXPECT_EQ(highUser.hiddenPreference, referenceTarget({tw(1, 1.0)}, topics, dim));
}

TEST(DriftSchedulerTest, FirstDriftInteractionReturnsMinimum) {
    const auto topics = basisTopics(4, 8);
    DriftEvent a{30, 0.0, 1.0, {tw(0, 1.0)}};
    DriftEvent b{10, 0.0, 1.0, {tw(1, 1.0)}};
    DriftEvent c{20, 0.0, 1.0, {tw(2, 1.0)}};
    DriftScheduler sched(DriftConfig{{a, b, c}}, topics);
    EXPECT_EQ(sched.firstDriftInteraction(), 10u);
}

// --- Unconfigured: guaranteed no-op -------------------------------------------------------------

TEST(DriftSchedulerTest, UnconfiguredIsNoOp) {
    const auto topics = basisTopics(4, 8);
    DriftScheduler sched(DriftConfig{}, topics); // no events

    EXPECT_FALSE(sched.configured());
    EXPECT_EQ(sched.firstDriftInteraction(), 0u);
    EXPECT_FALSE(sched.everApplies(UserId{0}));
    EXPECT_FALSE(sched.everApplies(UserId{12345}));

    HiddenUserState h = sentinelHidden(7, 8);
    const HiddenUserState before = h;
    for (uint32_t t : {0u, 1u, 10u, 500u}) {
        EXPECT_FALSE(sched.maybeApply(h, t)) << "t=" << t;
    }
    // Nothing mutated.
    EXPECT_EQ(h.hiddenPreference, before.hiddenPreference);
    EXPECT_EQ(h.preferredTopics, before.preferredTopics);
    EXPECT_FLOAT_EQ(h.preferenceConcentration, before.preferenceConcentration);
    EXPECT_FLOAT_EQ(h.preferenceStability, before.preferenceStability);
}

// Phase 23 LearnedRanker unit tests (contracts §5, V2 TDD §4.21). Cold-start fallback EXACTNESS
// (scores byte-identical to a standalone WeightedRanker), explanation well-formedness (all §2 keys
// + learned_value == Σ weighted terms), and the satisfaction-availability rule. Models are built
// with the real Retrainer over a tiny synthetic matrix so the whole learned path is exercised in
// isolation (no runner / no world).

#include "rr/learning_v2/learned_ranker.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/learning_v2/retrainer.hpp"
#include "rr/recommendation/weighted_ranker.hpp"

using namespace rr;
using rr::learning_v2::LearnedModels;
using rr::learning_v2::LearnedRanker;
using rr::learning_v2::Retrainer;
using rr::learning_v2::ShownFeatureRow;

namespace {

Reel makeReel(uint32_t id, uint32_t creator, uint32_t topic) {
    Reel r{};
    r.id = ReelId{id};
    r.creatorId = CreatorId{creator};
    r.primaryTopic = TopicId{topic};
    r.embedding = {1.0f, 0.0f};
    r.intrinsicQuality = 0.4f + 0.1f * static_cast<float>(id % 4);
    r.durationSeconds = 30.0f;
    r.createdAt = 0;
    r.active = true;
    return r;
}

Candidate cand(uint32_t id, float similarity) {
    Candidate c{};
    c.reelId = ReelId{id};
    c.source = CandidateSource::VectorHNSW;
    c.retrievalSimilarity = similarity;
    return c;
}

// A tiny synthetic training matrix: n rows with varied features and label rates chosen so each of
// the six §4.21 targets clears kMinPositivesToTrain (20) at n=120 (shared 40, follow 30, ni 24,
// exit 60, watch 120 linear, satisfaction 120 when withSurvey).
std::vector<ShownFeatureRow> makeMatrix(int n, bool withSurvey) {
    std::vector<ShownFeatureRow> rows;
    rows.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        ShownFeatureRow r;
        for (std::size_t j = 0; j < r.features.size(); ++j) {
            r.features[j] = static_cast<float>((i * 7 + static_cast<int>(j)) % 10) / 10.0f;
        }
        r.watchRatio = static_cast<float>(i % 5) / 5.0f;
        r.shared = (i % 3 == 0) ? 1 : 0;
        r.followed = (i % 4 == 0) ? 1 : 0;
        r.notInterested = (i % 5 == 0) ? 1 : 0;
        r.sessionExit = (i % 2 == 0) ? 1 : 0;
        r.likert = withSurvey ? static_cast<std::uint8_t>(1 + (i % 5)) : 0;
        r.hasFeatures = true;
        r.hasOutcome = true;
        rows.push_back(r);
    }
    return rows;
}

std::vector<Reel> makeReels() {
    return {makeReel(0, 1, 0), makeReel(1, 2, 1), makeReel(2, 3, 0), makeReel(3, 1, 2)};
}

User makeUser() {
    User u{};
    u.sessionPreference = {1.0f, 0.0f};
    u.creatorAffinity[CreatorId{1}] = 0.5f;
    return u;
}

std::vector<Candidate> makeCandidates() {
    return {cand(0, 0.9f), cand(1, 0.3f), cand(2, 0.6f), cand(3, 0.1f)};
}

} // namespace

// Cold-start fallback EXACTNESS (contracts §5): below the data threshold (no ready models) the
// LearnedRanker must serve the hand-tuned WeightedRanker scores byte-for-byte, in the same order —
// only the additive fallback=1 marker distinguishes the explanation map.
TEST(LearnedRankerTest, ColdStartFallbackMatchesWeightedRankerExactly) {
    const std::vector<Reel> reels = makeReels();
    const RankingConfig ranking; // defaults
    const LearningV2ValueWeights weights;

    LearnedRanker learned(reels, ranking, /*contentV2=*/false, /*personalizedDiversity=*/false,
                          weights);
    WeightedRanker baseline(reels, ranking, /*contentV2=*/false, /*personalizedDiversity=*/false);

    const User user = makeUser();
    const std::vector<Candidate> pool = makeCandidates();
    const std::vector<Candidate> learnedOut = learned.rank(user, pool, 100);
    const std::vector<Candidate> baseOut = baseline.rank(user, pool, 100);

    ASSERT_EQ(learnedOut.size(), baseOut.size());
    for (std::size_t i = 0; i < learnedOut.size(); ++i) {
        // Same order (same reel at each slot) and BYTE-identical score.
        EXPECT_EQ(learnedOut[i].reelId.value, baseOut[i].reelId.value);
        EXPECT_EQ(learnedOut[i].rankingScore, baseOut[i].rankingScore);
        // The WeightedRanker's map is present verbatim (its eleven keys sum to the score); the only
        // difference is the additive fallback marker.
        EXPECT_FLOAT_EQ(learnedOut[i].featureContributions.at("fallback"), 1.0f);
        EXPECT_FLOAT_EQ(learnedOut[i].featureContributions.at("similarity"),
                        baseOut[i].featureContributions.at("similarity"));
    }
    EXPECT_EQ(learned.fallbackCalls(), 1u);
    EXPECT_EQ(learned.rankCalls(), 1u);
}

// Explanation well-formedness (contracts §5): every §2 key present, learned_value == Σ weighted
// terms within 1e-6, fallback=0, satisfaction_available=1 (survey trained), feed ordered by value.
TEST(LearnedRankerTest, LearnedExplanationWellFormed) {
    const std::vector<Reel> reels = makeReels();
    const RankingConfig ranking;
    const LearningV2ValueWeights weights;
    LearnedRanker learned(reels, ranking, false, false, weights);

    Retrainer retrainer(/*runSeed=*/123, /*epochs=*/25);
    LearnedModels bundle = retrainer.retrain(makeMatrix(120, /*withSurvey=*/true), /*version=*/1);
    ASSERT_TRUE(bundle.ready);
    ASSERT_TRUE(bundle.satisfaction.has_value());
    learned.setModels(std::move(bundle));

    const std::vector<Candidate> out = learned.rank(makeUser(), makeCandidates(), 100);
    ASSERT_EQ(out.size(), 4u);

    static const std::vector<std::string> kKeys = {
        "predicted_watch",        "predicted_share", "predicted_follow",
        "predicted_satisfaction", "predicted_exit",  "predicted_regret",
        "learned_value",          "fallback",        "satisfaction_available"};
    float prev = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const auto &m = out[i].featureContributions;
        for (const std::string &k : kKeys) {
            EXPECT_TRUE(m.count(k)) << "missing key " << k;
        }
        const float sum = m.at("predicted_watch") + m.at("predicted_share") +
                          m.at("predicted_follow") + m.at("predicted_satisfaction") +
                          m.at("predicted_exit") + m.at("predicted_regret");
        EXPECT_NEAR(m.at("learned_value"), sum, 1e-6f);
        EXPECT_FLOAT_EQ(m.at("learned_value"), out[i].rankingScore);
        EXPECT_FLOAT_EQ(m.at("fallback"), 0.0f);
        EXPECT_FLOAT_EQ(m.at("satisfaction_available"), 1.0f);
        // exit/regret terms enter negatively (mirroring the penalty convention).
        EXPECT_LE(m.at("predicted_exit"), 0.0f);
        EXPECT_LE(m.at("predicted_regret"), 0.0f);
        if (i > 0) {
            EXPECT_LE(out[i].rankingScore, prev); // sorted by value descending
        }
        prev = out[i].rankingScore;
    }
}

// Satisfaction-availability rule (contracts §2): with no survey rows the satisfaction model is not
// trained, so its term contributes 0 and satisfaction_available=0 — the survey-off arm's signal.
TEST(LearnedRankerTest, SatisfactionUnavailableWithoutSurvey) {
    const std::vector<Reel> reels = makeReels();
    LearnedRanker learned(reels, RankingConfig{}, false, false, LearningV2ValueWeights{});

    Retrainer retrainer(123, 25);
    LearnedModels bundle = retrainer.retrain(makeMatrix(120, /*withSurvey=*/false), 1);
    ASSERT_TRUE(bundle.ready);
    EXPECT_FALSE(bundle.satisfaction.has_value()); // no likert rows => skipped
    learned.setModels(std::move(bundle));

    const std::vector<Candidate> out = learned.rank(makeUser(), makeCandidates(), 100);
    for (const Candidate &c : out) {
        EXPECT_FLOAT_EQ(c.featureContributions.at("satisfaction_available"), 0.0f);
        EXPECT_FLOAT_EQ(c.featureContributions.at("predicted_satisfaction"), 0.0f);
    }
    // Serving learned (not fallback) even though one target was skipped.
    EXPECT_EQ(learned.fallbackCalls(), 0u);
}

// The per-version seed derivation is deterministic and version-separating (contracts §3): same
// (seed, version) => same seed; different version => different seed.
TEST(LearnedRankerTest, RetrainVersionSeedDeterministicAndSeparating) {
    EXPECT_EQ(learning_v2::retrainVersionSeed(42, 1), learning_v2::retrainVersionSeed(42, 1));
    EXPECT_NE(learning_v2::retrainVersionSeed(42, 1), learning_v2::retrainVersionSeed(42, 2));
    EXPECT_NE(learning_v2::retrainVersionSeed(42, 1), learning_v2::retrainVersionSeed(43, 1));
}

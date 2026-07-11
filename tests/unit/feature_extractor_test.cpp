// Unit tests for the FeatureExtractor (Phase 6, TDD 14.1/14.3). Each normalization rule is
// checked against hand-computed values (including range edges, absent affinity, the no-history
// neutral duration, and degenerate pools), plus a determinism check. Small hand-built fixtures
// keep every expected value exactly computable.
#include "rr/recommendation/feature_extractor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/interaction.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"

namespace {

// A reel with sane, inert defaults; individual tests override just the fields they exercise.
rr::Reel makeReel(uint32_t id, uint32_t creator = 0, uint32_t topic = 0) {
    rr::Reel r{};
    r.id = rr::ReelId{id};
    r.creatorId = rr::CreatorId{creator};
    r.primaryTopic = rr::TopicId{topic};
    r.embedding = {1.0f, 0.0f};
    r.intrinsicQuality = 0.5f;
    r.durationSeconds = 30.0f;
    r.createdAt = 0;
    r.active = true;
    return r;
}

rr::Candidate cand(uint32_t id, float similarity = 0.0f) {
    rr::Candidate c{};
    c.reelId = rr::ReelId{id};
    c.source = rr::CandidateSource::VectorHNSW;
    c.retrievalSimilarity = similarity;
    return c;
}

rr::InteractionEvent event(uint32_t reelId, uint32_t creator, rr::InteractionType type) {
    rr::InteractionEvent e{};
    e.reelId = rr::ReelId{reelId};
    e.creatorId = rr::CreatorId{creator};
    e.type = type;
    return e;
}

// The single feature vector for a one-candidate pool over `reels`.
rr::FeatureVector extractOne(const std::vector<rr::Reel> &reels, const rr::User &user,
                             const rr::Candidate &c, rr::Timestamp now,
                             const rr::RankingConfig &config = {}) {
    rr::FeatureExtractor extractor(reels, config);
    std::vector<rr::Candidate> pool{c};
    return extractor.extract(user, pool, now).front();
}

} // namespace

TEST(FeatureExtractorTest, SimilarityAffineMap) {
    std::vector<rr::Reel> reels{makeReel(0)};
    rr::User user{};
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0, 0.5f), 0).similarity, 0.75f);
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0, -1.0f), 0).similarity, 0.0f);
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0, 1.0f), 0).similarity, 1.0f);
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0, 0.0f), 0).similarity, 0.5f);
}

TEST(FeatureExtractorTest, SessionTopicAffineMap) {
    // sessionTopic = (cos(sessionPreference, embedding) + 1) / 2, IDENTICAL to the similarity map.
    // makeReel's embedding is {1, 0}, so the cosine is just sessionPreference[0].
    std::vector<rr::Reel> reels{makeReel(0)};
    rr::User user{};

    user.sessionPreference = {1.0f, 0.0f}; // cos 1 -> 1.0
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).sessionTopic, 1.0f);

    user.sessionPreference = {0.0f, 1.0f}; // cos 0 (orthogonal) -> 0.5
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).sessionTopic, 0.5f);

    user.sessionPreference = {-1.0f, 0.0f}; // cos -1 -> 0.0
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).sessionTopic, 0.0f);

    user.sessionPreference = {0.6f, 0.8f}; // cos 0.6 -> (0.6+1)/2 = 0.8
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).sessionTopic, 0.8f);
}

TEST(FeatureExtractorTest, SessionTopicNeutralWithoutSessionVector) {
    // A user that bypassed cold start has an empty session vector; the defensive fallback is the
    // neutral 0.5 (no rr::dot throw on the hot path).
    std::vector<rr::Reel> reels{makeReel(0)};
    rr::User user{}; // sessionPreference empty
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).sessionTopic, 0.5f);
}

TEST(FeatureExtractorTest, QualityUsedAsIs) {
    std::vector<rr::Reel> reels{makeReel(0)};
    reels[0].intrinsicQuality = 0.42f;
    rr::User user{};
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).quality, 0.42f);
}

TEST(FeatureExtractorTest, FreshnessDecaysByHalfLife) {
    rr::RankingConfig config{};
    const auto halfLife = static_cast<rr::Timestamp>(config.freshnessHalfLifeSeconds);
    std::vector<rr::Reel> reels{makeReel(0)};
    rr::User user{};

    // Uploaded "now" => age 0 => 2^0 = 1.0.
    reels[0].createdAt = 1000;
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 1000, config).freshness, 1.0f);

    // One half-life old => 2^-1 = 0.5.
    reels[0].createdAt = 0;
    EXPECT_NEAR(extractOne(reels, user, cand(0), halfLife, config).freshness, 0.5f, 1e-6);
}

TEST(FeatureExtractorTest, TrendingSaturation) {
    rr::RankingConfig config{};
    std::vector<rr::Reel> reels{makeReel(0)};
    rr::User user{};

    // Never interacted => raw 0 => 0.
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 100, config).trending, 0.0f);

    // updatedAt == now => decay w = 1. raw = E/(1+I). E=1, I=0 => raw 1 => 1/(1+1) = 0.5.
    reels[0].trendingUpdatedAt = 100;
    reels[0].trendingEngagement = 1.0;
    reels[0].trendingImpressions = 0.0;
    EXPECT_NEAR(extractOne(reels, user, cand(0), 100, config).trending, 0.5f, 1e-6);

    // E=3, I=0 => raw 3 => 3/4 = 0.75.
    reels[0].trendingEngagement = 3.0;
    EXPECT_NEAR(extractOne(reels, user, cand(0), 100, config).trending, 0.75f, 1e-6);
}

TEST(FeatureExtractorTest, CreatorAffinityLookupAndAbsent) {
    std::vector<rr::Reel> reels{makeReel(0, /*creator=*/7)};
    rr::User user{};

    // Absent key => 0.
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).creatorAffinity, 0.0f);

    // Present => looked-up value.
    user.creatorAffinity[rr::CreatorId{7}] = 0.6f;
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).creatorAffinity, 0.6f);
}

TEST(FeatureExtractorTest, ExplorationConstantZero) {
    std::vector<rr::Reel> reels{makeReel(0)};
    rr::User user{};
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).exploration, 0.0f);
}

TEST(FeatureExtractorTest, DurationMatchNeutralWithoutHistory) {
    std::vector<rr::Reel> reels{makeReel(0)};
    rr::User user{}; // no recent interactions
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).durationMatch, 0.5f);
}

TEST(FeatureExtractorTest, DurationMatchFromCompletedHistory) {
    // reels[0] = history reel (duration 30, completed); reels[1] = exact-match candidate (30);
    // reels[2] = mismatched candidate (45). preferred = 30.
    std::vector<rr::Reel> reels{makeReel(0), makeReel(1), makeReel(2)};
    reels[0].durationSeconds = 30.0f;
    reels[1].durationSeconds = 30.0f;
    reels[2].durationSeconds = 45.0f;
    rr::User user{};
    user.recentInteractions.push_back(event(0, 0, rr::InteractionType::CompleteWatch));

    EXPECT_NEAR(extractOne(reels, user, cand(1), 0).durationMatch, 1.0f, 1e-6);
    // |45 - 30| / 115 = 0.130434..., match = 0.869565...
    EXPECT_NEAR(extractOne(reels, user, cand(2), 0).durationMatch,
                static_cast<float>(1.0 - 15.0 / 115.0), 1e-6);
}

TEST(FeatureExtractorTest, DurationMatchIgnoresNonPositiveHistory) {
    // Only a skip in history => no usable preference => neutral 0.5.
    std::vector<rr::Reel> reels{makeReel(0), makeReel(1)};
    reels[0].durationSeconds = 30.0f;
    reels[1].durationSeconds = 90.0f;
    rr::User user{};
    user.recentInteractions.push_back(event(0, 0, rr::InteractionType::InstantSkip));
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(1), 0).durationMatch, 0.5f);
}

TEST(FeatureExtractorTest, ImpressionCountLogNormalized) {
    std::vector<rr::Reel> reels{makeReel(0)};
    rr::User user{};

    reels[0].impressionCount = 0;
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).impressionCount, 0.0f);

    reels[0].impressionCount = static_cast<uint64_t>(rr::FeatureExtractor::kImpressionLogScale);
    EXPECT_NEAR(extractOne(reels, user, cand(0), 0).impressionCount, 1.0f, 1e-6);

    reels[0].impressionCount = 999;
    const float expected = static_cast<float>(
        std::log1p(999.0) / std::log1p(rr::FeatureExtractor::kImpressionLogScale));
    EXPECT_NEAR(extractOne(reels, user, cand(0), 0).impressionCount, expected, 1e-6);
}

TEST(FeatureExtractorTest, RepetitionFractionOfRecentWindow) {
    // Candidate: creator 5, topic 3. History (4 events):
    //   e0: creator 5 (creator match)         -> match
    //   e1: creator 9, reel with topic 3       -> topic match
    //   e2: creator 9, reel with topic 8       -> no match
    //   e3: creator 9, reel with topic 8       -> no match
    // 2 of 4 => 0.5.
    std::vector<rr::Reel> reels{makeReel(0, 5, 3),  // candidate at index 0
                                makeReel(1, 9, 3),  // e1's reel: topic 3
                                makeReel(2, 9, 8),  // e2's reel: topic 8
                                makeReel(3, 9, 8)}; // e3's reel: topic 8
    rr::User user{};
    user.recentInteractions.push_back(event(9, 5, rr::InteractionType::PartialWatch)); // creator 5
    user.recentInteractions.push_back(event(1, 9, rr::InteractionType::PartialWatch)); // topic 3
    user.recentInteractions.push_back(event(2, 9, rr::InteractionType::PartialWatch)); // none
    user.recentInteractions.push_back(event(3, 9, rr::InteractionType::PartialWatch)); // none

    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).repetition, 0.5f);
}

TEST(FeatureExtractorTest, RepetitionZeroWithoutHistory) {
    std::vector<rr::Reel> reels{makeReel(0)};
    rr::User user{};
    EXPECT_FLOAT_EQ(extractOne(reels, user, cand(0), 0).repetition, 0.0f);
}

TEST(FeatureExtractorTest, PopularityDegeneratePoolIsHalf) {
    // Two identical reels => min == max => neutral 0.5.
    std::vector<rr::Reel> reels{makeReel(0), makeReel(1)};
    for (rr::Reel &r : reels) {
        r.impressionCount = 100;
        r.likeCount = 5;
    }
    rr::User user{};
    rr::FeatureExtractor extractor(reels, {});
    std::vector<rr::Candidate> pool{cand(0), cand(1)};
    const std::vector<rr::FeatureVector> f = extractor.extract(user, pool, 0);
    EXPECT_FLOAT_EQ(f[0].popularity, 0.5f);
    EXPECT_FLOAT_EQ(f[1].popularity, 0.5f);
}

TEST(FeatureExtractorTest, PopularityMinMaxEndpointsAndMiddle) {
    // All impressions equal (100) => the +C*prior/denominator terms cancel in the min-max, so the
    // normalized value is exactly (engagement - minEng)/(maxEng - minEng).
    // engagement = completion + 2*like + 4*share. Shares 0/10/50 => engagement 0/40/200.
    std::vector<rr::Reel> reels{makeReel(0), makeReel(1), makeReel(2)};
    for (rr::Reel &r : reels) {
        r.impressionCount = 100;
    }
    reels[0].shareCount = 0;  // engagement 0   -> min
    reels[1].shareCount = 10; // engagement 40  -> (40-0)/(200-0) = 0.2
    reels[2].shareCount = 50; // engagement 200 -> max
    rr::User user{};
    rr::FeatureExtractor extractor(reels, {});
    std::vector<rr::Candidate> pool{cand(0), cand(1), cand(2)};
    const std::vector<rr::FeatureVector> f = extractor.extract(user, pool, 0);
    EXPECT_FLOAT_EQ(f[0].popularity, 0.0f);
    EXPECT_NEAR(f[1].popularity, 0.2f, 1e-6);
    EXPECT_FLOAT_EQ(f[2].popularity, 1.0f);
}

TEST(FeatureExtractorTest, Deterministic) {
    // Same inputs twice => identical feature vectors, every field.
    std::vector<rr::Reel> reels{makeReel(0, 5, 3), makeReel(1, 9, 3), makeReel(2, 9, 8)};
    reels[0].impressionCount = 100;
    reels[0].shareCount = 3;
    reels[1].impressionCount = 50;
    reels[1].likeCount = 4;
    reels[2].impressionCount = 10;
    reels[0].trendingUpdatedAt = 100;
    reels[0].trendingEngagement = 2.0;
    reels[0].trendingImpressions = 1.0;
    rr::User user{};
    user.sessionPreference = {0.8f, 0.6f};
    user.creatorAffinity[rr::CreatorId{5}] = 0.3f;
    user.recentInteractions.push_back(event(1, 5, rr::InteractionType::CompleteWatch));
    user.recentInteractions.push_back(event(2, 9, rr::InteractionType::Like));

    rr::FeatureExtractor extractor(reels, {});
    std::vector<rr::Candidate> pool{cand(0, 0.4f), cand(1, 0.1f), cand(2, -0.2f)};
    const std::vector<rr::FeatureVector> a = extractor.extract(user, pool, 200);
    const std::vector<rr::FeatureVector> b = extractor.extract(user, pool, 200);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(a[i].similarity, b[i].similarity);
        EXPECT_FLOAT_EQ(a[i].sessionTopic, b[i].sessionTopic);
        EXPECT_FLOAT_EQ(a[i].quality, b[i].quality);
        EXPECT_FLOAT_EQ(a[i].freshness, b[i].freshness);
        EXPECT_FLOAT_EQ(a[i].popularity, b[i].popularity);
        EXPECT_FLOAT_EQ(a[i].trending, b[i].trending);
        EXPECT_FLOAT_EQ(a[i].creatorAffinity, b[i].creatorAffinity);
        EXPECT_FLOAT_EQ(a[i].exploration, b[i].exploration);
        EXPECT_FLOAT_EQ(a[i].durationMatch, b[i].durationMatch);
        EXPECT_FLOAT_EQ(a[i].repetition, b[i].repetition);
        EXPECT_FLOAT_EQ(a[i].impressionCount, b[i].impressionCount);
    }
}

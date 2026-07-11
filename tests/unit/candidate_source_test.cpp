// Unit tests for the vector candidate sources (Phase 5 tasks 1-2). They verify the
// CandidateGenerator contract: correct source label, distance/similarity filled via the D3
// conversion (checked against rr::dot for unit vectors), request.candidateLimit honoured, and
// RAW pass-through (no active/seen/validity filtering — that is the Orchestrator's job).
#include "rr/candidate_sources/exact_candidate_source.hpp"
#include "rr/candidate_sources/hnsw_candidate_source.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/vindex/exact_vector_index.hpp"
#include "rr/vindex/hnsw_vector_index.hpp"

namespace {

rr::Embedding unit(rr::Embedding e) {
    rr::normalize(e);
    return e;
}

// Reels spread around the +x axis so cosine order against query {1,0,0} is deterministic.
std::vector<rr::Embedding> reelEmbeddings() {
    return {unit({1.0f, 0.0f, 0.0f}), unit({0.9f, 0.1f, 0.0f}), unit({0.5f, 0.5f, 0.0f}),
            unit({0.0f, 1.0f, 0.0f}), unit({-1.0f, 0.0f, 0.0f})};
}

rr::User makeUser(rr::Embedding preference, std::vector<uint32_t> seen = {}) {
    rr::User user{};
    user.id = rr::UserId{0};
    user.estimatedPreference = unit(std::move(preference));
    for (uint32_t s : seen) {
        user.seenReels.insert(rr::ReelId{s});
    }
    return user;
}

rr::RecommendationRequest request(std::size_t candidateLimit) {
    rr::RecommendationRequest req{};
    req.userId = rr::UserId{0};
    req.candidateLimit = candidateLimit;
    req.feedSize = candidateLimit;
    return req;
}

// HNSWVectorIndex is non-movable, so indexes are constructed in place and filled through this
// helper rather than returned by value.
template <class Index> void insertAll(Index &index, const std::vector<rr::Embedding> &embeddings) {
    for (std::size_t i = 0; i < embeddings.size(); ++i) {
        index.insert(rr::ReelId{static_cast<uint32_t>(i)}, embeddings[i]);
    }
}

rr::ExactVectorIndex buildExact(const std::vector<rr::Embedding> &embeddings) {
    rr::ExactVectorIndex index(3);
    insertAll(index, embeddings);
    return index;
}

} // namespace

TEST(ExactCandidateSourceTest, FillsSourceDistanceAndSimilarity) {
    const std::vector<rr::Embedding> embeddings = reelEmbeddings();
    rr::ExactVectorIndex index = buildExact(embeddings);
    rr::ExactCandidateSource source(index);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});

    const std::vector<rr::Candidate> candidates = source.generate(user, request(5));
    ASSERT_EQ(candidates.size(), 5u);
    for (const rr::Candidate &c : candidates) {
        EXPECT_EQ(c.source, rr::CandidateSource::VectorExact);
        // D3: similarity is exactly the Euclidean->cosine conversion of the reported distance.
        EXPECT_FLOAT_EQ(c.retrievalSimilarity, rr::similarityFromEuclidean(c.retrievalDistance));
        // ... and for unit vectors that equals the dot product with the query.
        const float expected = rr::dot(user.estimatedPreference, embeddings[c.reelId.value]);
        EXPECT_NEAR(c.retrievalSimilarity, expected, 1e-4f);
    }
    // Ascending distance == descending similarity: nearest reel is id 0 ({1,0,0} == query).
    EXPECT_EQ(candidates.front().reelId, rr::ReelId{0});
}

TEST(ExactCandidateSourceTest, HonoursCandidateLimit) {
    rr::ExactVectorIndex index = buildExact(reelEmbeddings());
    rr::ExactCandidateSource source(index);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});

    const std::vector<rr::Candidate> candidates = source.generate(user, request(2));
    ASSERT_EQ(candidates.size(), 2u);
    EXPECT_EQ(candidates[0].reelId, rr::ReelId{0});
    EXPECT_EQ(candidates[1].reelId, rr::ReelId{1});
}

TEST(ExactCandidateSourceTest, ReturnsRawResultsIgnoringSeen) {
    rr::ExactVectorIndex index = buildExact(reelEmbeddings());
    rr::ExactCandidateSource source(index);
    // Reel 0 is the nearest AND marked seen; a candidate source must still return it (filtering
    // is the Orchestrator's job, TDD 13).
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f}, {0});

    const std::vector<rr::Candidate> candidates = source.generate(user, request(5));
    ASSERT_EQ(candidates.size(), 5u);
    EXPECT_EQ(candidates.front().reelId, rr::ReelId{0});
}

TEST(ExactCandidateSourceTest, ZeroCandidateLimitReturnsEmpty) {
    rr::ExactVectorIndex index = buildExact(reelEmbeddings());
    rr::ExactCandidateSource source(index);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});
    EXPECT_TRUE(source.generate(user, request(0)).empty());
}

TEST(HnswCandidateSourceTest, FillsSourceDistanceAndSimilarity) {
    const std::vector<rr::Embedding> embeddings = reelEmbeddings();
    rr::HNSWVectorIndex index(3, rr::HNSWConfig{}, 123);
    insertAll(index, embeddings);
    rr::HNSWCandidateSource source(index);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});

    const std::vector<rr::Candidate> candidates = source.generate(user, request(5));
    ASSERT_FALSE(candidates.empty());
    for (const rr::Candidate &c : candidates) {
        EXPECT_EQ(c.source, rr::CandidateSource::VectorHNSW);
        EXPECT_FLOAT_EQ(c.retrievalSimilarity, rr::similarityFromEuclidean(c.retrievalDistance));
        const float expected = rr::dot(user.estimatedPreference, embeddings[c.reelId.value]);
        EXPECT_NEAR(c.retrievalSimilarity, expected, 1e-4f);
    }
}

TEST(HnswCandidateSourceTest, HonoursCandidateLimit) {
    rr::HNSWVectorIndex index(3, rr::HNSWConfig{}, 123);
    insertAll(index, reelEmbeddings());
    rr::HNSWCandidateSource source(index);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});
    EXPECT_LE(source.generate(user, request(2)).size(), 2u);
}

TEST(HnswCandidateSourceTest, ZeroCandidateLimitReturnsEmpty) {
    rr::HNSWVectorIndex index(3, rr::HNSWConfig{}, 123);
    insertAll(index, reelEmbeddings());
    rr::HNSWCandidateSource source(index);
    const rr::User user = makeUser({1.0f, 0.0f, 0.0f});
    EXPECT_TRUE(source.generate(user, request(0)).empty());
}

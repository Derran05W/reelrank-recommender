// Unit tests for the Orchestrator (Phase 5 task 2, TDD 7/13). A hand-written fake
// CandidateGenerator lets us feed exact candidate sets and assert the merge/dedup/filter/cap and
// identity-ranking behaviour precisely and deterministically.
#include "rr/recommendation/orchestrator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/recommendation/ranker.hpp"

namespace {

// Returns a fixed candidate list on every call, ignoring the user/request. Lets a test control
// the exact merged input the Orchestrator sees.
class FakeSource final : public rr::CandidateGenerator {
  public:
    explicit FakeSource(std::vector<rr::Candidate> candidates)
        : candidates_(std::move(candidates)) {}

    std::vector<rr::Candidate> generate(const rr::User &,
                                        const rr::RecommendationRequest &) override {
        return candidates_;
    }

  private:
    std::vector<rr::Candidate> candidates_;
};

rr::Candidate cand(uint32_t id, rr::CandidateSource source, float distance, float similarity) {
    rr::Candidate c{};
    c.reelId = rr::ReelId{id};
    c.source = source;
    c.retrievalDistance = distance;
    c.retrievalSimilarity = similarity;
    return c;
}

// n active reels, dense ids, each with a non-empty (dummy) embedding so the validity filter keeps
// them. The Orchestrator scores off the candidates' own similarity, not these embeddings.
std::vector<rr::Reel> makeReels(std::size_t n) {
    std::vector<rr::Reel> reels(n);
    for (std::size_t i = 0; i < n; ++i) {
        reels[i].id = rr::ReelId{static_cast<uint32_t>(i)};
        reels[i].active = true;
        reels[i].embedding = {1.0f, 0.0f};
    }
    return reels;
}

rr::RecommendationRequest request(std::size_t feedSize, std::size_t candidateLimit) {
    rr::RecommendationRequest req{};
    req.userId = rr::UserId{0};
    req.feedSize = feedSize;
    req.candidateLimit = candidateLimit;
    return req;
}

std::vector<uint32_t> feedIds(const rr::RecommendationResponse &response) {
    std::vector<uint32_t> ids;
    for (const rr::RankedReel &r : response.reels) {
        ids.push_back(r.reelId.value);
    }
    return ids;
}

bool hasSource(const rr::RankedReel &r, rr::CandidateSource s) {
    return std::find(r.sources.begin(), r.sources.end(), s) != r.sources.end();
}

// A Ranker that lets the ranker PATH be tested without any real feature math: it stamps a
// distinctive score (100 + reelId) and a single "fake" contribution on each candidate, and returns
// them in ASCENDING retrieval-similarity order — the REVERSE of the orchestrator's identity path,
// so honouring the ranker's order is observable.
class FakeRanker final : public rr::Ranker {
  public:
    std::vector<rr::Candidate> rank(const rr::User &, const std::vector<rr::Candidate> &candidates,
                                    rr::Timestamp) const override {
        std::vector<rr::Candidate> out = candidates;
        for (rr::Candidate &c : out) {
            c.rankingScore = 100.0f + static_cast<float>(c.reelId.value);
            c.featureContributions = {{"fake", c.rankingScore}};
        }
        std::sort(out.begin(), out.end(), [](const rr::Candidate &a, const rr::Candidate &b) {
            return a.retrievalSimilarity < b.retrievalSimilarity; // least relevant first
        });
        return out;
    }
};

const rr::RankedReel &feedById(const rr::RecommendationResponse &response, uint32_t id) {
    for (const rr::RankedReel &r : response.reels) {
        if (r.reelId.value == id) {
            return r;
        }
    }
    ADD_FAILURE() << "reel " << id << " missing from feed";
    return response.reels.front();
}

} // namespace

TEST(OrchestratorTest, DedupPreservesAllSourceLabels) {
    std::vector<rr::Reel> reels = makeReels(3);
    FakeSource hnsw({cand(0, rr::CandidateSource::VectorHNSW, 0.3f, 0.9f)});
    FakeSource exact({cand(0, rr::CandidateSource::VectorExact, 0.3f, 0.9f)});
    rr::Orchestrator orch({&hnsw, &exact}, reels);
    rr::User user{};

    const rr::RecommendationResponse response = orch.recommend(user, request(10, 10));
    ASSERT_EQ(response.reels.size(), 1u);
    const rr::RankedReel &reel = response.reels.front();
    EXPECT_EQ(reel.reelId, rr::ReelId{0});
    EXPECT_EQ(reel.sources.size(), 2u);
    EXPECT_TRUE(hasSource(reel, rr::CandidateSource::VectorHNSW));
    EXPECT_TRUE(hasSource(reel, rr::CandidateSource::VectorExact));
}

TEST(OrchestratorTest, DedupKeepsSmallestDistanceAndPairedSimilarity) {
    std::vector<rr::Reel> reels = makeReels(1);
    // Two views of reel 0; the smaller-distance one (0.2) carries similarity 0.99.
    FakeSource a({cand(0, rr::CandidateSource::VectorHNSW, 0.6f, 0.50f)});
    FakeSource b({cand(0, rr::CandidateSource::VectorExact, 0.2f, 0.99f)});
    rr::Orchestrator orch({&a, &b}, reels);
    rr::User user{};

    const rr::RecommendationResponse response = orch.recommend(user, request(10, 10));
    ASSERT_EQ(response.reels.size(), 1u);
    // score == similarity paired with the smallest distance.
    EXPECT_FLOAT_EQ(response.reels.front().score, 0.99f);
}

TEST(OrchestratorTest, DropsInactiveReels) {
    std::vector<rr::Reel> reels = makeReels(3);
    reels[1].active = false;
    FakeSource src({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f),
                    cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.8f),
                    cand(2, rr::CandidateSource::VectorHNSW, 0.3f, 0.7f)});
    rr::Orchestrator orch({&src}, reels);
    rr::User user{};

    EXPECT_EQ(feedIds(orch.recommend(user, request(10, 10))), (std::vector<uint32_t>{0, 2}));
}

TEST(OrchestratorTest, DropsSeenReels) {
    std::vector<rr::Reel> reels = makeReels(3);
    FakeSource src({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f),
                    cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.8f),
                    cand(2, rr::CandidateSource::VectorHNSW, 0.3f, 0.7f)});
    rr::Orchestrator orch({&src}, reels);
    rr::User user{};
    user.seenReels.insert(rr::ReelId{2});

    EXPECT_EQ(feedIds(orch.recommend(user, request(10, 10))), (std::vector<uint32_t>{0, 1}));
}

TEST(OrchestratorTest, DropsInvalidEmbeddings) {
    std::vector<rr::Reel> reels = makeReels(3);
    reels[0].embedding.clear(); // empty embedding == invalid (TDD 13 step 7)
    FakeSource src({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f),
                    cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.8f)});
    rr::Orchestrator orch({&src}, reels);
    rr::User user{};

    EXPECT_EQ(feedIds(orch.recommend(user, request(10, 10))), (std::vector<uint32_t>{1}));
}

TEST(OrchestratorTest, CapsPoolAtCandidateLimit) {
    std::vector<rr::Reel> reels = makeReels(5);
    FakeSource src({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.90f),
                    cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.80f),
                    cand(2, rr::CandidateSource::VectorHNSW, 0.3f, 0.70f),
                    cand(3, rr::CandidateSource::VectorHNSW, 0.4f, 0.60f),
                    cand(4, rr::CandidateSource::VectorHNSW, 0.5f, 0.50f)});
    rr::Orchestrator orch({&src}, reels);
    rr::User user{};

    const rr::RecommendationResponse response = orch.recommend(user, request(10, 3));
    // candidatesRanked is the capped pool size; only the top-3 by similarity survive the cap.
    EXPECT_EQ(response.candidatesRanked, 3u);
    EXPECT_EQ(feedIds(response), (std::vector<uint32_t>{0, 1, 2}));
}

TEST(OrchestratorTest, IdentityRankingOrdersBySimilarityThenId) {
    std::vector<rr::Reel> reels = makeReels(6);
    // Two reels tie at 0.80 (ids 5 and 2) — the tie breaks by ascending id (2 before 5).
    FakeSource src({cand(5, rr::CandidateSource::VectorHNSW, 0.2f, 0.80f),
                    cand(3, rr::CandidateSource::VectorHNSW, 0.1f, 0.90f),
                    cand(2, rr::CandidateSource::VectorHNSW, 0.2f, 0.80f)});
    rr::Orchestrator orch({&src}, reels);
    rr::User user{};

    const rr::RecommendationResponse response = orch.recommend(user, request(10, 10));
    EXPECT_EQ(feedIds(response), (std::vector<uint32_t>{3, 2, 5}));
    for (std::size_t i = 0; i < response.reels.size(); ++i) {
        EXPECT_EQ(response.reels[i].rank, i); // ranks are 0..n-1 in feed order
    }
}

TEST(OrchestratorTest, TruncatesToFeedSize) {
    std::vector<rr::Reel> reels = makeReels(5);
    FakeSource src({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.90f),
                    cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.80f),
                    cand(2, rr::CandidateSource::VectorHNSW, 0.3f, 0.70f),
                    cand(3, rr::CandidateSource::VectorHNSW, 0.4f, 0.60f)});
    rr::Orchestrator orch({&src}, reels);
    rr::User user{};

    const rr::RecommendationResponse response = orch.recommend(user, request(2, 10));
    EXPECT_EQ(response.reels.size(), 2u);     // truncated to feedSize
    EXPECT_EQ(response.candidatesRanked, 4u); // but the whole pool was ranked
    EXPECT_EQ(feedIds(response), (std::vector<uint32_t>{0, 1}));
}

TEST(OrchestratorTest, CandidateCountsAndLatencyFieldsPopulated) {
    std::vector<rr::Reel> reels = makeReels(3);
    // reel 0 appears from both sources: candidatesRetrieved counts it twice (pre-dedup).
    FakeSource a({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.90f),
                  cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.80f)});
    FakeSource b({cand(0, rr::CandidateSource::VectorExact, 0.1f, 0.90f)});
    rr::Orchestrator orch({&a, &b}, reels);
    rr::User user{};

    const rr::RecommendationResponse response = orch.recommend(user, request(10, 10));
    EXPECT_EQ(response.candidatesRetrieved, 3u); // 2 + 1, pre-dedup
    EXPECT_EQ(response.candidatesRanked, 2u);    // post-dedup pool
    EXPECT_GE(response.retrievalLatencyMs, 0.0);
    EXPECT_GE(response.rankingLatencyMs, 0.0);
    EXPECT_GE(response.rerankingLatencyMs, 0.0);
    EXPECT_GE(response.totalLatencyMs, response.retrievalLatencyMs);
}

TEST(OrchestratorTest, NullptrRankerLeavesContributionsEmpty) {
    std::vector<rr::Reel> reels = makeReels(2);
    FakeSource src({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f),
                    cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.8f)});
    rr::Orchestrator orch({&src}, reels); // nullptr ranker (default)
    rr::User user{};

    const rr::RecommendationResponse response = orch.recommend(user, request(10, 10));
    ASSERT_EQ(response.reels.size(), 2u);
    for (const rr::RankedReel &r : response.reels) {
        EXPECT_TRUE(r.featureContributions.empty());
    }
}

TEST(OrchestratorTest, RankerPathHonoursRankerOrderAndScore) {
    // Identity path would order [0,1,2] by descending similarity; the FakeRanker returns ascending
    // similarity, so the feed must be [2,1,0] and each score == 100 + reelId.
    std::vector<rr::Reel> reels = makeReels(3);
    FakeSource src({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f),
                    cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.8f),
                    cand(2, rr::CandidateSource::VectorHNSW, 0.3f, 0.7f)});
    FakeRanker ranker;
    rr::Orchestrator orch({&src}, reels, &ranker);
    rr::User user{};

    const rr::RecommendationResponse response = orch.recommend(user, request(10, 10));
    EXPECT_EQ(feedIds(response), (std::vector<uint32_t>{2, 1, 0}));
    EXPECT_FLOAT_EQ(response.reels[0].score, 102.0f);
    EXPECT_FLOAT_EQ(response.reels[1].score, 101.0f);
    EXPECT_FLOAT_EQ(response.reels[2].score, 100.0f);
    for (std::size_t i = 0; i < response.reels.size(); ++i) {
        EXPECT_EQ(response.reels[i].rank, i); // ranks 0..n-1 in feed order
    }
    EXPECT_GE(response.rankingLatencyMs, 0.0); // ranking latency populated on the ranker path
    EXPECT_EQ(response.candidatesRanked, 3u);
}

TEST(OrchestratorTest, RankerPathPropagatesContributions) {
    std::vector<rr::Reel> reels = makeReels(2);
    FakeSource src({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f),
                    cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.8f)});
    FakeRanker ranker;
    rr::Orchestrator orch({&src}, reels, &ranker);
    rr::User user{};

    const rr::RecommendationResponse response = orch.recommend(user, request(10, 10));
    const rr::RankedReel &reel1 = feedById(response, 1);
    ASSERT_EQ(reel1.featureContributions.count("fake"), 1u);
    EXPECT_FLOAT_EQ(reel1.featureContributions.at("fake"), 101.0f);
    EXPECT_FLOAT_EQ(reel1.featureContributions.at("fake"), reel1.score);
}

TEST(OrchestratorTest, RankerPathPreservesMultiSourceLabels) {
    // reel 0 is produced by two sources; after ranking the RankedReel must still carry BOTH labels
    // (the Candidate handed to the ranker only carries the first label).
    std::vector<rr::Reel> reels = makeReels(2);
    FakeSource hnsw({cand(0, rr::CandidateSource::VectorHNSW, 0.3f, 0.9f),
                     cand(1, rr::CandidateSource::VectorHNSW, 0.4f, 0.6f)});
    FakeSource exact({cand(0, rr::CandidateSource::VectorExact, 0.3f, 0.9f)});
    FakeRanker ranker;
    rr::Orchestrator orch({&hnsw, &exact}, reels, &ranker);
    rr::User user{};

    const rr::RecommendationResponse response = orch.recommend(user, request(10, 10));
    const rr::RankedReel &reel0 = feedById(response, 0);
    EXPECT_EQ(reel0.sources.size(), 2u);
    EXPECT_TRUE(hasSource(reel0, rr::CandidateSource::VectorHNSW));
    EXPECT_TRUE(hasSource(reel0, rr::CandidateSource::VectorExact));
}

// Unit tests for the Orchestrator (Phase 5 task 2, TDD 7/13). A hand-written fake
// CandidateGenerator lets us feed exact candidate sets and assert the merge/dedup/filter/cap and
// identity-ranking behaviour precisely and deterministically.
#include "rr/recommendation/orchestrator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rr/candidate_sources/exploration_candidate_source.hpp"
#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/recommendation/ranker.hpp"
#include "rr/recommendation/reranker.hpp"

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

// A Ranker that forces every exploration-labeled candidate BELOW every non-exploration candidate
// (non-exploration score 1000-id, exploration score -id), ties by ascending id. Lets the
// guaranteed-slot promotion be observed: exploration items start entirely below the feed cut.
class ExplorationBottomRanker final : public rr::Ranker {
  public:
    std::vector<rr::Candidate> rank(const rr::User &, const std::vector<rr::Candidate> &candidates,
                                    rr::Timestamp) const override {
        std::vector<rr::Candidate> out = candidates;
        for (rr::Candidate &c : out) {
            const bool expl = c.source == rr::CandidateSource::Exploration;
            c.rankingScore = expl ? -static_cast<float>(c.reelId.value)
                                  : (1000.0f - static_cast<float>(c.reelId.value));
        }
        std::sort(out.begin(), out.end(), [](const rr::Candidate &a, const rr::Candidate &b) {
            if (a.rankingScore != b.rankingScore) {
                return a.rankingScore > b.rankingScore;
            }
            return a.reelId.value < b.reelId.value;
        });
        return out;
    }
};

rr::RecommendationRequest explRequest(std::size_t feedSize, std::size_t candidateLimit) {
    rr::RecommendationRequest req{};
    req.userId = rr::UserId{0};
    req.feedSize = feedSize;
    req.candidateLimit = candidateLimit;
    req.enableExploration = true; // let the wired exploration source draw its per-slot gates
    return req;
}

std::size_t explorationCountInFeed(const rr::RecommendationResponse &response) {
    std::size_t n = 0;
    for (const rr::RankedReel &r : response.reels) {
        if (std::find(r.sources.begin(), r.sources.end(), rr::CandidateSource::Exploration) !=
            r.sources.end()) {
            ++n;
        }
    }
    return n;
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

// --- Guaranteed exploration slots (Phase 8, task 3). A real ExplorationCandidateSource over an
// EMPTY reel set contributes ZERO candidates but still draws its feedSize gates, so it is a clean,
// controllable source of lastFiredSlots; the exploration-LABELED candidates are injected by a
// FakeSource. ExplorationBottomRanker forces those below the feed cut so promotion is observable.

TEST(OrchestratorTest, GuaranteePromotesAndIsCappedByGuaranteedSlots) {
    std::vector<rr::Reel> reels = makeReels(13);
    std::vector<rr::Candidate> exploit;
    for (uint32_t i = 0; i < 8; ++i) {
        exploit.push_back(cand(i, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f));
    }
    std::vector<rr::Candidate> expl;
    for (uint32_t i = 8; i < 13; ++i) {
        expl.push_back(cand(i, rr::CandidateSource::Exploration, 0.5f, 0.5f)); // 5 exploration
    }
    FakeSource exploitSrc(exploit);
    FakeSource explSrc(expl);
    std::vector<rr::Reel> emptyReels;
    rr::Rng rng(1);
    rr::ExplorationCandidateSource realExpl(emptyReels, /*epsilon=*/1.0, /*poolCap=*/50,
                                            /*window=*/1e9, &rng);
    ExplorationBottomRanker ranker;
    rr::Orchestrator orch({&exploitSrc, &explSrc, &realExpl}, reels, &ranker, &realExpl,
                          /*guaranteedSlots=*/2);
    rr::User user{};

    const rr::RecommendationResponse resp = orch.recommend(user, explRequest(5, 100));
    EXPECT_EQ(realExpl.lastFiredSlots(), 5u); // epsilon=1 => all feedSize gates fire
    // g = min(fired=5, guaranteedSlots=2, explCount=5) = 2. The two highest-ranked exploration
    // items (ids 8,9) are promoted; the two lowest-ranked non-exploration prefix items (3,4) drop.
    EXPECT_EQ(feedIds(resp), (std::vector<uint32_t>{0, 1, 2, 8, 9}));
    EXPECT_EQ(explorationCountInFeed(resp), 2u);
}

TEST(OrchestratorTest, GuaranteeIsCappedByExplorationCount) {
    std::vector<rr::Reel> reels = makeReels(10);
    std::vector<rr::Candidate> exploit;
    for (uint32_t i = 0; i < 8; ++i) {
        exploit.push_back(cand(i, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f));
    }
    std::vector<rr::Candidate> expl{cand(8, rr::CandidateSource::Exploration, 0.5f, 0.5f),
                                    cand(9, rr::CandidateSource::Exploration, 0.5f, 0.5f)};
    FakeSource exploitSrc(exploit);
    FakeSource explSrc(expl);
    std::vector<rr::Reel> emptyReels;
    rr::Rng rng(2);
    rr::ExplorationCandidateSource realExpl(emptyReels, 1.0, 50, 1e9, &rng);
    ExplorationBottomRanker ranker;
    rr::Orchestrator orch({&exploitSrc, &explSrc, &realExpl}, reels, &ranker, &realExpl,
                          /*guaranteedSlots=*/5);
    rr::User user{};

    const rr::RecommendationResponse resp = orch.recommend(user, explRequest(6, 100));
    // g = min(fired=6, guaranteedSlots=5, explCount=2) = 2 (only two exploration items exist).
    EXPECT_EQ(explorationCountInFeed(resp), 2u);
    EXPECT_EQ(feedIds(resp), (std::vector<uint32_t>{0, 1, 2, 3, 8, 9}));
}

TEST(OrchestratorTest, GuaranteeIsCappedByFiredSlots) {
    std::vector<rr::Reel> reels = makeReels(16);
    std::vector<rr::Candidate> exploit;
    for (uint32_t i = 0; i < 8; ++i) {
        exploit.push_back(cand(i, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f));
    }
    std::vector<rr::Candidate> expl;
    for (uint32_t i = 8; i < 16; ++i) {
        expl.push_back(cand(i, rr::CandidateSource::Exploration, 0.5f, 0.5f)); // 8 exploration
    }
    FakeSource exploitSrc(exploit);
    FakeSource explSrc(expl);
    std::vector<rr::Reel> emptyReels;
    rr::Rng rng(1);
    // epsilon=0.5 with feedSize=8 => a partial, deterministic number of gates fire.
    rr::ExplorationCandidateSource realExpl(emptyReels, 0.5, 50, 1e9, &rng);
    ExplorationBottomRanker ranker;
    rr::Orchestrator orch({&exploitSrc, &explSrc, &realExpl}, reels, &ranker, &realExpl,
                          /*guaranteedSlots=*/100);
    rr::User user{};

    const rr::RecommendationResponse resp = orch.recommend(user, explRequest(8, 100));
    const std::size_t fired = realExpl.lastFiredSlots();
    // guaranteedSlots (100) and explCount (8) both exceed fired (<= feedSize 8), so fired binds:
    // exactly `fired` exploration items are guaranteed into the feed.
    EXPECT_GT(fired, 0u);
    EXPECT_LE(fired, 8u);
    EXPECT_EQ(explorationCountInFeed(resp), fired);
}

TEST(OrchestratorTest, GuaranteeInertWithoutExplorationCandidates) {
    std::vector<rr::Reel> reels = makeReels(8);
    std::vector<rr::Candidate> exploit;
    for (uint32_t i = 0; i < 8; ++i) {
        exploit.push_back(cand(i, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f));
    }
    FakeSource exploitSrc(exploit);
    std::vector<rr::Reel> emptyReels;
    rr::Rng rng(3);
    rr::ExplorationCandidateSource realExpl(emptyReels, 1.0, 50, 1e9, &rng);
    ExplorationBottomRanker ranker;
    rr::Orchestrator orch({&exploitSrc, &realExpl}, reels, &ranker, &realExpl,
                          /*guaranteedSlots=*/2);
    rr::User user{};

    const rr::RecommendationResponse resp = orch.recommend(user, explRequest(5, 100));
    EXPECT_EQ(realExpl.lastFiredSlots(), 5u);    // gates fired...
    EXPECT_EQ(explorationCountInFeed(resp), 0u); // ...but there is nothing to promote
    EXPECT_EQ(feedIds(resp), (std::vector<uint32_t>{0, 1, 2, 3, 4}));
}

TEST(OrchestratorTest, GuaranteeInertWhenExplorationSourceDefaulted) {
    // Defaulted trailing params (no exploration source, guaranteedSlots 0) => byte-for-byte the
    // pre-Phase-8 pipeline: exploration-labeled candidates rank on merit and, being at the bottom,
    // never enter the feed.
    std::vector<rr::Reel> reels = makeReels(10);
    std::vector<rr::Candidate> exploit;
    for (uint32_t i = 0; i < 8; ++i) {
        exploit.push_back(cand(i, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f));
    }
    std::vector<rr::Candidate> expl{cand(8, rr::CandidateSource::Exploration, 0.5f, 0.5f),
                                    cand(9, rr::CandidateSource::Exploration, 0.5f, 0.5f)};
    FakeSource exploitSrc(exploit);
    FakeSource explSrc(expl);
    ExplorationBottomRanker ranker;
    rr::Orchestrator orch({&exploitSrc, &explSrc}, reels, &ranker); // defaulted trailing params
    rr::User user{};

    const rr::RecommendationResponse resp = orch.recommend(user, explRequest(5, 100));
    EXPECT_EQ(explorationCountInFeed(resp), 0u);
    EXPECT_EQ(feedIds(resp), (std::vector<uint32_t>{0, 1, 2, 3, 4}));
}

TEST(OrchestratorTest, GuaranteeIsDeterministic) {
    auto run = [] {
        std::vector<rr::Reel> reels = makeReels(13);
        std::vector<rr::Candidate> exploit;
        for (uint32_t i = 0; i < 8; ++i) {
            exploit.push_back(cand(i, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f));
        }
        std::vector<rr::Candidate> expl;
        for (uint32_t i = 8; i < 13; ++i) {
            expl.push_back(cand(i, rr::CandidateSource::Exploration, 0.5f, 0.5f));
        }
        FakeSource exploitSrc(exploit);
        FakeSource explSrc(expl);
        std::vector<rr::Reel> emptyReels;
        rr::Rng rng(42);
        rr::ExplorationCandidateSource realExpl(emptyReels, 1.0, 50, 1e9, &rng);
        ExplorationBottomRanker ranker;
        rr::Orchestrator orch({&exploitSrc, &explSrc, &realExpl}, reels, &ranker, &realExpl, 3);
        rr::User user{};
        return feedIds(orch.recommend(user, explRequest(5, 100)));
    };
    EXPECT_EQ(run(), run());
}

TEST(OrchestratorTest, GuaranteeCountsMultiLabelReelAsExploration) {
    // Reel 5 is produced by BOTH an exploitation source and the exploration source. The
    // representative label is Exploration, so it is protected AND both labels survive to the feed.
    std::vector<rr::Reel> reels = makeReels(6);
    std::vector<rr::Candidate> exploit;
    for (uint32_t i = 0; i < 6; ++i) {
        exploit.push_back(cand(i, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f));
    }
    std::vector<rr::Candidate> expl{cand(5, rr::CandidateSource::Exploration, 0.5f, 0.5f)};
    FakeSource exploitSrc(exploit);
    FakeSource explSrc(expl);
    std::vector<rr::Reel> emptyReels;
    rr::Rng rng(7);
    rr::ExplorationCandidateSource realExpl(emptyReels, 1.0, 50, 1e9, &rng);
    ExplorationBottomRanker ranker;
    rr::Orchestrator orch({&exploitSrc, &explSrc, &realExpl}, reels, &ranker, &realExpl,
                          /*guaranteedSlots=*/1);
    rr::User user{};

    const rr::RecommendationResponse resp = orch.recommend(user, explRequest(5, 100));
    // g = min(fired=5, guaranteedSlots=1, explCount=1) = 1. Reel 5 (bottom-ranked, exploration
    // representative) is promoted, displacing the lowest-ranked non-exploration prefix item (4).
    EXPECT_EQ(explorationCountInFeed(resp), 1u);
    EXPECT_EQ(feedIds(resp), (std::vector<uint32_t>{0, 1, 2, 3, 5}));
    const rr::RankedReel &reel5 = feedById(resp, 5);
    EXPECT_TRUE(hasSource(reel5, rr::CandidateSource::VectorHNSW));
    EXPECT_TRUE(hasSource(reel5, rr::CandidateSource::Exploration));
}

// --- Diversity re-ranking plug-in (Phase 9, task 4). A stub Reranker whose ONLY observable effect
// is to REVERSE the ranked pool (then truncate to feedSize) lets us prove: the gate is inert unless
// a reranker is wired AND request.enableDiversity is set AND we are on the RANKED path; and that
// the full merged multi-source label set is restored onto the reranked feed (the stub only sees /
// emits the single representative label per reel).

namespace {

class ReverseReranker final : public rr::Reranker {
  public:
    std::vector<rr::RankedReel> rerank(const rr::User &, const std::vector<rr::Candidate> &pool,
                                       std::size_t feedSize) const override {
        std::vector<rr::RankedReel> out;
        const std::size_t n = std::min(feedSize, pool.size());
        for (std::size_t i = 0; i < n; ++i) {
            const rr::Candidate &c = pool[pool.size() - 1 - i]; // reverse of the ranked order
            rr::RankedReel r{};
            r.reelId = c.reelId;
            r.score = c.rankingScore;
            r.rank = i;
            r.sources = {
                c.source}; // single representative label only; orchestrator restores full set
            r.featureContributions = c.featureContributions;
            out.push_back(std::move(r));
        }
        return out;
    }
};

rr::RecommendationRequest diversityRequest(std::size_t feedSize, std::size_t candidateLimit,
                                           bool enableDiversity) {
    rr::RecommendationRequest req{};
    req.userId = rr::UserId{0};
    req.feedSize = feedSize;
    req.candidateLimit = candidateLimit;
    req.enableDiversity = enableDiversity;
    return req;
}

} // namespace

TEST(OrchestratorTest, DiversityGateInertWhenDisabledMatchesNoReranker) {
    // FakeRanker orders ascending similarity => ranked pool [3,2,1,0]. A no-reranker Orchestrator
    // and a reranker-wired one with enableDiversity=false must produce byte-identical feeds (gate
    // off).
    std::vector<rr::Reel> reels = makeReels(4);
    FakeSource src({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f),
                    cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.8f),
                    cand(2, rr::CandidateSource::VectorHNSW, 0.3f, 0.7f),
                    cand(3, rr::CandidateSource::VectorHNSW, 0.4f, 0.6f)});
    FakeRanker ranker;
    ReverseReranker reranker;
    rr::User user{};

    FakeSource srcA(src);
    rr::Orchestrator noReranker({&srcA}, reels, &ranker);
    FakeSource srcB(src);
    rr::Orchestrator withReranker({&srcB}, reels, &ranker, nullptr, 0, &reranker);

    const std::vector<uint32_t> baseline = feedIds(noReranker.recommend(user, request(10, 10)));
    const std::vector<uint32_t> gatedOff =
        feedIds(withReranker.recommend(user, diversityRequest(10, 10, /*enableDiversity=*/false)));
    EXPECT_EQ(baseline, (std::vector<uint32_t>{3, 2, 1, 0}));
    EXPECT_EQ(gatedOff, baseline); // enableDiversity=false => reranker never runs
}

TEST(OrchestratorTest, DiversityRerankerAppliedWhenEnabled) {
    // Same setup; with enableDiversity=true the ReverseReranker reverses [3,2,1,0] -> [0,1,2,3],
    // and the reranking latency field is populated.
    std::vector<rr::Reel> reels = makeReels(4);
    FakeSource src({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f),
                    cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.8f),
                    cand(2, rr::CandidateSource::VectorHNSW, 0.3f, 0.7f),
                    cand(3, rr::CandidateSource::VectorHNSW, 0.4f, 0.6f)});
    FakeRanker ranker;
    ReverseReranker reranker;
    rr::Orchestrator orch({&src}, reels, &ranker, nullptr, 0, &reranker);
    rr::User user{};

    const rr::RecommendationResponse resp =
        orch.recommend(user, diversityRequest(10, 10, /*enableDiversity=*/true));
    EXPECT_EQ(feedIds(resp), (std::vector<uint32_t>{0, 1, 2, 3}));
    for (std::size_t i = 0; i < resp.reels.size(); ++i) {
        EXPECT_EQ(resp.reels[i].rank, i); // ranks are the reranker's 0..n-1
    }
    EXPECT_GE(resp.rerankingLatencyMs, 0.0);
}

TEST(OrchestratorTest, DiversityRerankerRestoresFullMultiSourceLabels) {
    // reel 0 is produced by two sources; after diversity reranking the RankedReel must still carry
    // BOTH labels even though the reranker only ever saw/emitted the single representative label.
    std::vector<rr::Reel> reels = makeReels(2);
    FakeSource hnsw({cand(0, rr::CandidateSource::VectorHNSW, 0.3f, 0.9f),
                     cand(1, rr::CandidateSource::VectorHNSW, 0.4f, 0.6f)});
    FakeSource exact({cand(0, rr::CandidateSource::VectorExact, 0.3f, 0.9f)});
    FakeRanker ranker;
    ReverseReranker reranker;
    rr::Orchestrator orch({&hnsw, &exact}, reels, &ranker, nullptr, 0, &reranker);
    rr::User user{};

    const rr::RecommendationResponse resp =
        orch.recommend(user, diversityRequest(10, 10, /*enableDiversity=*/true));
    ASSERT_EQ(resp.reels.size(), 2u);
    const rr::RankedReel &reel0 = feedById(resp, 0);
    EXPECT_EQ(reel0.sources.size(), 2u);
    EXPECT_TRUE(hasSource(reel0, rr::CandidateSource::VectorHNSW));
    EXPECT_TRUE(hasSource(reel0, rr::CandidateSource::VectorExact));
}

TEST(OrchestratorTest, DiversityRerankerInertOnIdentityPath) {
    // Identity (nullptr ranker) path: diversity reranking is RANKED-path only, so even with a
    // reranker wired and enableDiversity=true the feed stays in identity (descending similarity)
    // order — the ReverseReranker is never invoked.
    std::vector<rr::Reel> reels = makeReels(4);
    FakeSource src({cand(0, rr::CandidateSource::VectorHNSW, 0.1f, 0.9f),
                    cand(1, rr::CandidateSource::VectorHNSW, 0.2f, 0.8f),
                    cand(2, rr::CandidateSource::VectorHNSW, 0.3f, 0.7f),
                    cand(3, rr::CandidateSource::VectorHNSW, 0.4f, 0.6f)});
    ReverseReranker reranker;
    rr::Orchestrator orch({&src}, reels, /*ranker=*/nullptr, nullptr, 0, &reranker);
    rr::User user{};

    const rr::RecommendationResponse resp =
        orch.recommend(user, diversityRequest(10, 10, /*enableDiversity=*/true));
    EXPECT_EQ(feedIds(resp), (std::vector<uint32_t>{0, 1, 2, 3})); // identity order, unreversed
}

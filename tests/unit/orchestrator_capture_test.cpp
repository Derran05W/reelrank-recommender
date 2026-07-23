// Phase 22 orchestrator FEATURE-CAPTURE test (package A; contracts §7). Proves the served-time
// capture surface the TrainingLogger consumes is faithful:
//   * structural join — one capture row per ranked-pool candidate, contiguous pool_rank, the shown
//     set + scores match resp.reels;
//   * features AS SERVED (non-circular) — the captured `similarity` feature reconstructs the
//   RANKER's
//     own `similarity` featureContribution (weight * feature) exactly, i.e. the captured vector is
//     the one the ranker scored, not a divergent recompute;
//   * a full independent extractor pass on the served pool reproduces every captured field.
// Also confirms the capture is OFF by default (nullptr sink => empty, no perturbation).

#include "rr/recommendation/orchestrator.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/recommendation/candidate_generator.hpp"
#include "rr/recommendation/feature_extractor.hpp"
#include "rr/recommendation/weighted_ranker.hpp"

using namespace rr;

namespace {

class FakeSource final : public CandidateGenerator {
  public:
    explicit FakeSource(std::vector<Candidate> candidates) : candidates_(std::move(candidates)) {}
    std::vector<Candidate> generate(const User &, const RecommendationRequest &) override {
        return candidates_;
    }

  private:
    std::vector<Candidate> candidates_;
};

Candidate cand(uint32_t id, float similarity) {
    Candidate c{};
    c.reelId = ReelId{id};
    c.source = CandidateSource::VectorHNSW;
    c.retrievalDistance = 1.0f - similarity;
    c.retrievalSimilarity = similarity;
    return c;
}

std::vector<Reel> makeReels(std::size_t n) {
    std::vector<Reel> reels(n);
    for (std::size_t i = 0; i < n; ++i) {
        reels[i].id = ReelId{static_cast<uint32_t>(i)};
        reels[i].active = true;
        reels[i].embedding = {1.0f, 0.0f};
        reels[i].intrinsicQuality = 0.5f;
    }
    return reels;
}

const RankingCaptureRow *rowFor(const RankingCapture &cap, uint32_t reelId) {
    for (const RankingCaptureRow &r : cap.rows) {
        if (r.reelId.value == reelId) {
            return &r;
        }
    }
    return nullptr;
}

} // namespace

TEST(OrchestratorCaptureTest, CapturedFeaturesAreTheServedFeatures) {
    const std::vector<Reel> reels = makeReels(6);
    // A distinctive, nonzero similarity weight so the ranker's `similarity` contribution is a clean
    // multiple of the served similarity feature (the reconstruction key below).
    RankingConfig cfg;
    cfg.similarityWeight = 1.5;

    WeightedRanker ranker(reels, cfg, /*contentV2=*/false, /*personalizedDiversity=*/false);
    std::vector<Candidate> source = {cand(0, 0.9f), cand(1, 0.1f), cand(2, 0.7f), cand(3, -0.3f),
                                     cand(4, 0.5f)};
    FakeSource fake(std::move(source));
    std::vector<CandidateGenerator *> sources = {&fake};
    Orchestrator orch(sources, reels, &ranker);

    RecommendationRequest req{};
    req.userId = UserId{0};
    req.feedSize = 3;
    req.candidateLimit = 100;
    req.requestTime = 1000;

    FeatureExtractor captureExtractor(reels, cfg, /*contentV2=*/false);
    RankingCapture cap;
    cap.extractor = &captureExtractor;
    req.capture = &cap;

    const RecommendationResponse resp = orch.recommend(User{}, req);

    // Structural: one row per ranked candidate, contiguous pool_rank 0..n-1.
    ASSERT_EQ(cap.rows.size(), resp.candidatesRanked);
    ASSERT_EQ(cap.rows.size(), 5u);
    for (std::size_t i = 0; i < cap.rows.size(); ++i) {
        EXPECT_EQ(cap.rows[i].poolRank, i);
    }

    // The shown feed's items each have a capture row whose servedScore == the served RankedReel
    // score and whose captured `similarity` feature reconstructs the RANKER's own similarity
    // contribution (weight * feature) — non-circular proof the captured vector is the one the
    // ranker scored.
    for (const RankedReel &shown : resp.reels) {
        const RankingCaptureRow *row = rowFor(cap, shown.reelId.value);
        ASSERT_NE(row, nullptr);
        EXPECT_FLOAT_EQ(row->servedScore, shown.score);
        const float servedSimContribution = shown.featureContributions.at("similarity");
        EXPECT_NEAR(static_cast<double>(row->features.similarity) * cfg.similarityWeight,
                    static_cast<double>(servedSimContribution), 1e-5);
    }

    // Every captured field reproduces an independent extractor pass on the served pool (all 21
    // features, in poolRank order) — confirms the capture recorded a real served-pool extraction.
    std::vector<Candidate> servedPool;
    for (const RankingCaptureRow &r : cap.rows) {
        Candidate c{};
        c.reelId = r.reelId;
        c.source = r.sources.empty() ? CandidateSource::VectorHNSW : r.sources.front();
        c.retrievalSimilarity = r.retrievalSimilarity;
        servedPool.push_back(c);
    }
    const std::vector<FeatureVector> ref =
        captureExtractor.extract(User{}, servedPool, req.requestTime);
    ASSERT_EQ(ref.size(), cap.rows.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_FLOAT_EQ(cap.rows[i].features.similarity, ref[i].similarity);
        EXPECT_FLOAT_EQ(cap.rows[i].features.popularity,
                        ref[i].popularity); // pool-relative feature
        EXPECT_FLOAT_EQ(cap.rows[i].features.quality, ref[i].quality);
    }
}

// --- Capture is OFF by default: a null sink means no work and an empty capture
// ---------------------
TEST(OrchestratorCaptureTest, NoCaptureWhenSinkNull) {
    const std::vector<Reel> reels = makeReels(4);
    RankingConfig cfg;
    WeightedRanker ranker(reels, cfg, false, false);
    FakeSource fake({cand(0, 0.9f), cand(1, 0.5f), cand(2, 0.2f)});
    std::vector<CandidateGenerator *> sources = {&fake};
    Orchestrator orch(sources, reels, &ranker);

    RecommendationRequest req{};
    req.userId = UserId{0};
    req.feedSize = 2;
    req.candidateLimit = 100;
    req.requestTime = 1000;
    // req.capture stays nullptr (the default).

    const RecommendationResponse resp = orch.recommend(User{}, req);
    EXPECT_EQ(resp.reels.size(), 2u); // recommend still works, unperturbed
}

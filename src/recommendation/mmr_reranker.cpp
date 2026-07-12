#include "rr/recommendation/mmr_reranker.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"

namespace rr {

MMRReranker::MMRReranker(const std::vector<Reel> &reels, double lambda)
    : reels_(reels), lambda_(lambda) {}

namespace {

// Cosine similarity between two unit embeddings via rr::dot. Empty or mismatched-dimension
// embeddings (an out-of-range reel id or a degenerate reel) contribute 0 — "no similarity" — which
// keeps the pairwise term well-defined without an exception on the hot path.
double cosine(const Embedding &a, const Embedding &b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0;
    }
    return static_cast<double>(dot(a, b));
}

} // namespace

std::vector<RankedReel> MMRReranker::rerank(const User &, const std::vector<Candidate> &pool,
                                            std::size_t feedSize) const {
    const std::size_t n = pool.size();
    const std::size_t want = std::min(feedSize, n);
    std::vector<RankedReel> out;
    if (want == 0) {
        return out;
    }

    // --- Relevance: min-max normalise rankingScore over the pool to [0, 1]. Degenerate all-equal
    // pool => position-based fallback (strictly decreasing in input index), which keeps the
    // incoming relevance order meaningful and reproduces the input order at lambda == 1.
    float minScore = pool.front().rankingScore;
    float maxScore = pool.front().rankingScore;
    for (const Candidate &c : pool) {
        minScore = std::min(minScore, c.rankingScore);
        maxScore = std::max(maxScore, c.rankingScore);
    }
    std::vector<double> relevance(n);
    if (maxScore > minScore) {
        const double range = static_cast<double>(maxScore) - static_cast<double>(minScore);
        for (std::size_t i = 0; i < n; ++i) {
            relevance[i] =
                (static_cast<double>(pool[i].rankingScore) - static_cast<double>(minScore)) / range;
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            relevance[i] = 1.0 - static_cast<double>(i) / static_cast<double>(n);
        }
    }

    // Resolve each candidate's embedding once (empty for an out-of-range id -> cosine 0 elsewhere).
    std::vector<const Embedding *> embeddings(n);
    static const Embedding kEmpty{};
    for (std::size_t i = 0; i < n; ++i) {
        const ReelId id = pool[i].reelId;
        embeddings[i] = id.value < reels_.size() ? &reels_[id.value].embedding : &kEmpty;
    }

    // --- Greedy MMR selection. maxSim[i] is the running max cosine of candidate i to any already-
    // selected item (0 while nothing is selected, so the first pick carries no diversity penalty).
    std::vector<char> selected(n, 0);
    std::vector<double> maxSim(n, 0.0);
    out.reserve(want);
    for (std::size_t picked = 0; picked < want; ++picked) {
        std::size_t best = n;
        double bestMmr = 0.0;
        float bestScore = 0.0f;
        std::uint32_t bestId = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (selected[i]) {
                continue;
            }
            const double mmr = lambda_ * relevance[i] - (1.0 - lambda_) * maxSim[i];
            const float score = pool[i].rankingScore;
            const std::uint32_t id = pool[i].reelId.value;
            // Argmax with the documented tie-break: higher mmr, then higher rankingScore, then
            // smaller ReelId. A total order => deterministic.
            const bool better =
                best == n || mmr > bestMmr ||
                (mmr == bestMmr && (score > bestScore || (score == bestScore && id < bestId)));
            if (better) {
                best = i;
                bestMmr = mmr;
                bestScore = score;
                bestId = id;
            }
        }

        selected[best] = 1;
        Candidate chosen = pool[best];
        RankedReel r{};
        r.reelId = chosen.reelId;
        r.score = chosen.rankingScore;
        r.rank = picked;
        r.sources = {chosen.source};
        r.featureContributions = std::move(chosen.featureContributions);
        out.push_back(std::move(r));

        // Update every remaining candidate's running similarity to the freshly selected item.
        for (std::size_t i = 0; i < n; ++i) {
            if (!selected[i]) {
                maxSim[i] = std::max(maxSim[i], cosine(*embeddings[i], *embeddings[best]));
            }
        }
    }
    return out;
}

} // namespace rr

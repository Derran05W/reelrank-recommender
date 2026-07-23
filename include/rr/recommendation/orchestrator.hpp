#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rr/domain/candidate.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/recommendation.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/recommendation/candidate_generator.hpp"
#include "rr/recommendation/feature_extractor.hpp"
#include "rr/recommendation/ranker.hpp"

namespace rr {

class ExplorationCandidateSource;
class Reranker;

// ================================================================================================
// Phase 22 SERVED-TIME RANKING CAPTURE (contracts docs/design/P22-CONTRACTS.md §7). The additive,
// zero-cost-when-off surface the TrainingLogger consumes: the Orchestrator fills a RankingCapture
// (via RecommendationRequest::capture) with one row per FULL-POOL candidate, best-first, carrying
// the exact served-time signals the log needs — pool rank, served score, exploration provenance,
// retrieval-source union, retrieval similarity, and the per-candidate FeatureVector.
//
// The FeatureVectors are NOT recomputed post-hoc from a different world: the Orchestrator runs the
// runner-supplied `extractor` on the SAME served pool (same reel set, same request time) the ranker
// scored, and because FeatureExtractor is a pure, deterministic function of (reels, RankingConfig,
// contentV2, user, pool, now) — and every pool-relative feature (popularity min-max, pool prior) is
// set-invariant — the captured vectors are BYTE-IDENTICAL to the ones the WeightedRanker computed
// internally. The runner builds `extractor` with the SAME (reels, config.ranking,
// realism.contentV2) the recommender's own ranker used, so the equality holds for every ranked
// recommender.
// ================================================================================================
struct RankingCaptureRow {
    ReelId reelId;
    std::size_t poolRank; // 0-based rank in the full served pool (best-first, post-exploration-
                          // guarantee on the ranked path; retrieval-similarity order on identity)
    float servedScore; // the ranker's rankingScore (ranked path) / retrievalSimilarity (identity)
    bool explorationLabeled; // representative source == Exploration (matches the served exploration
                             // feature + the guaranteed-slot rule — one signal, cannot disagree)
    float retrievalSimilarity;
    std::vector<CandidateSource> sources; // FULL merged retrieval-source union, first-seen order
    FeatureVector features;               // served-time features (see the equality argument above)
};

struct RankingCapture {
    // INPUT (runner-set): the served-time feature extractor to run on the pool. Built by the runner
    // with the SAME reels + RankingConfig + contentV2 as the recommender's ranker, so features are
    // byte-identical to the served ones. When null the Orchestrator captures nothing (the pointer
    // on the request is the on/off flag; this is a belt-and-braces second guard).
    const FeatureExtractor *extractor = nullptr;
    // OUTPUT (Orchestrator-filled): one row per full-pool candidate, in poolRank order.
    std::vector<RankingCaptureRow> rows;
};

// TDD 7 / 13: the recommendation orchestrator. Given a fixed set of candidate sources it runs
// them, merges + deduplicates their output (preserving all source labels), drops ineligible
// reels, caps the pool, applies a ranking stage, and truncates to the requested feed size —
// recording per-stage wall-clock latency into the response.
//
// The ranking stage is pluggable (Phase 6). If `ranker == nullptr` the stage is IDENTITY: the
// capped pool is ordered by descending retrieval similarity (Phase 5 behaviour, byte-for-byte
// unchanged). If a Ranker is supplied, the capped pool is materialized as std::vector<Candidate>
// and handed to `ranker->rank(...)`, which RE-ORDERS it (it never changes membership); each
// resulting RankedReel then carries the ranker's rankingScore and featureContributions while the
// FULL merged multi-source label set is still preserved.
//
// The whole path is DETERMINISTIC and exception-free: no rr::Rng is used, and every ordering
// decision falls back to ascending ReelId so the output is independent of hash-map iteration
// order.
//
// GUARANTEED EXPLORATION SLOTS (Phase 8, TDD 12.7/task 3). When an ExplorationCandidateSource and
// a non-zero guaranteedSlots are supplied (default: neither, so the behaviour below is byte-for-
// byte the pre-Phase-8 pipeline), the orchestrator protects exploration candidates from being
// ranked out entirely. After ranking, with g = min(exploration slots that fired the epsilon gate
// this request [source->lastFiredSlots()], guaranteedSlots, exploration-labeled candidates in the
// ranked pool): if the feed prefix holds fewer than g exploration-labeled items, it promotes the
// highest-ranked exploration-labeled items from below the cut and evicts the lowest-ranked
// non-exploration items from the prefix, preserving relative order otherwise. Deterministic; the
// rule and its label semantics are documented at applyExplorationGuarantee() in the .cpp. Applies
// only on the RANKED path (exploration recommenders always supply a ranker); epsilon = 0 fires no
// slots, so it is structurally inert (the epsilon=0 no-op exit criterion).
//
// EXPLORATION LABEL semantics: a reel is "exploration-labeled" iff its REPRESENTATIVE source
// (Candidate.source) is Exploration. The orchestrator elects Exploration as the representative
// whenever a reel's merged label set contains it (see representativeSource in the .cpp), so this
// single field drives BOTH the guarantee here AND the FeatureExtractor's exploration feature — the
// two can never disagree. Non-exploration recommenders never carry the Exploration label, so this
// is a no-op for them.
//
// DIVERSITY RE-RANKING (Phase 9, TDD 15). An optional trailing Reranker is applied ONLY on the
// RANKED path (diversity recommenders always supply a ranker, mirroring the exploration-guarantee
// precedent) and ONLY when request.enableDiversity is true. It runs inside the reranking Stopwatch
// AFTER applyExplorationGuarantee: the guarantee reorders the pool FIRST, so the constraint
// selection walks the guarantee-promoted order — promoted exploration items are taken unless they
// violate a hard cap. Caps take precedence, so delivered exploration slots may fall below g;
// deterministic and accepted (documented at the call site). When the gate is false (no reranker, or
// enableDiversity false, or the identity path) behaviour is byte-identical to Phase 8. The
// reranker sees only each reel's REPRESENTATIVE single source label; the Orchestrator restores the
// full merged multi-source label set onto every returned RankedReel afterwards.
//
// Ownership: the Orchestrator holds NON-OWNING pointers to its sources, the ranker, the exploration
// source, and the reranker, plus a reference to the live reels vector. The owner (a Recommender)
// must keep them all alive for the Orchestrator's lifetime.
class Orchestrator {
  public:
    Orchestrator(std::vector<CandidateGenerator *> sources, const std::vector<Reel> &reels,
                 const Ranker *ranker = nullptr,
                 const ExplorationCandidateSource *explorationSource = nullptr,
                 uint32_t guaranteedSlots = 0, const Reranker *reranker = nullptr);

    RecommendationResponse recommend(const User &user, const RecommendationRequest &request);

  private:
    // Reorders `ranked` (the full capped pool, best-first) in place so the first `feedSize` items
    // satisfy the guaranteed-exploration-slot rule. No-op unless an exploration source and a
    // non-zero guaranteedSlots were supplied and at least one gate fired this request.
    void applyExplorationGuarantee(std::vector<Candidate> &ranked, std::size_t feedSize) const;

    std::vector<CandidateGenerator *> sources_;
    const std::vector<Reel> &reels_;
    const Ranker *ranker_;
    const ExplorationCandidateSource *explorationSource_;
    uint32_t guaranteedSlots_;
    const Reranker *reranker_;
};

} // namespace rr

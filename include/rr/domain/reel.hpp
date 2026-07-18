#pragma once

#include <cstdint>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/clock.hpp"

namespace rr {

// TDD 8.2.
struct Reel {
    ReelId id;
    CreatorId creatorId;

    Embedding embedding;

    float intrinsicQuality;
    float freshnessScore;
    float durationSeconds;

    TopicId primaryTopic;
    std::vector<TopicId> secondaryTopics;

    Timestamp createdAt;

    uint64_t impressionCount;
    uint64_t completionCount;
    uint64_t likeCount;
    uint64_t shareCount;
    uint64_t skipCount;

    bool active;

    // Trending accumulator (TDD 12.4): exponentially decayed interaction counters maintained by
    // Simulator::step and read via rr::trendingScore (scoring.hpp). On every impression at time t
    // both accumulators are decayed by trendingDecayFactor(trendingUpdatedAt, t, halfLife), then
    // trendingImpressions += 1 and trendingEngagement += the event's engagement increment (same
    // 1/2/4 completion/like/share weights as popularityEngagement). Appended after `active` with
    // defaults so pre-Phase-6 positional aggregate initializers stay valid.
    double trendingEngagement = 0.0;
    double trendingImpressions = 0.0;
    Timestamp trendingUpdatedAt = 0;

    // --- Realism V2 serving-time-visible attributes (V2 TDD 4.1, Phase 13) -------------------
    // Populated by augmentReelsV2 (streams "archetypes"/"reels-v2") only when realism.content_v2
    // is on; under gate-off every field keeps the defaults below, zero V2 draws occur, and no V1
    // consumer reads them (D17). Appended after the trending block with defaults so earlier
    // positional aggregate initializers stay valid.
    //
    // Modality embeddings share simulation.dimensions and are L2-normalized (D3/D5). Only
    // `embedding` (semantic) is ever ANN-indexed (D23) — these ride on the Reel as ranker/
    // behaviour-model features from Phase 14 on.
    Embedding visualStyleEmbedding{};
    Embedding musicEmbedding{};
    Embedding emotionalToneEmbedding{};

    // Content-value scalars, all in [0, 1].
    float usefulness = 0.0f;
    float humour = 0.0f;
    float novelty = 0.0f;
    // V2 name for visual/production polish. Deliberately distinct from V1 `intrinsicQuality`
    // (overall content quality consumed by the V1 behaviour model and ranker): archetypes can
    // push polish independently of substance (e.g. polished-irrelevant), so the two are drawn
    // correlated but not aliased. V1 consumers keep reading intrinsicQuality unchanged.
    float productionQuality = 0.0f;
    float controversy = 0.0f;
    float clickbaitStrength = 0.0f;
    float informationDensity = 0.0f;
    float emotionalIntensity = 0.0f;

    // Language of the reel, drawn from the skewed global distribution over realism.languages.
    LanguageId language{};
};

} // namespace rr

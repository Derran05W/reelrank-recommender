#include "rr/simulation/behaviour_model.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <set>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/hidden_user_state.hpp"
#include "rr/domain/ids.hpp"
#include "rr/domain/reel.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/dataset_generator.hpp"

using namespace rr;

namespace {

// Band edges mirrored from behaviour_model.cpp (which keeps them in an anonymous namespace, repo
// convention: no premature config surface). Kept in sync for the flag-consistency assertions.
constexpr float kInstantSkipRatioHi = 0.05f;
constexpr float kHighRatioLo = 0.80f;
constexpr float kHighRatioHi = 1.20f;
constexpr float kImpressionMaxRatio = 0.02f;

Embedding unitEmbedding(uint32_t dim, uint64_t seed) {
    Rng rng(seed);
    Embedding e(dim);
    for (uint32_t d = 0; d < dim; ++d) {
        e[d] = static_cast<float>(rng.gaussian());
    }
    normalize(e);
    return e;
}

Embedding negate(const Embedding &e) {
    Embedding n(e.size());
    for (size_t i = 0; i < e.size(); ++i) {
        n[i] = -e[i];
    }
    return n;
}

HiddenUserState makeHidden(Embedding pref, float likeProp = 0.1f, float shareProp = 0.05f,
                           float durationTolerance = 0.5f) {
    HiddenUserState h;
    h.userId = UserId{0};
    h.hiddenPreference = std::move(pref);
    h.likePropensity = likeProp;
    h.sharePropensity = shareProp;
    h.durationTolerance = durationTolerance;
    return h;
}

Reel makeReel(Embedding emb, float quality, float durationSeconds) {
    Reel r{};
    r.id = ReelId{0};
    r.creatorId = CreatorId{0};
    r.embedding = std::move(emb);
    r.intrinsicQuality = quality;
    r.durationSeconds = durationSeconds;
    r.active = true;
    return r;
}

Creator makeCreator(Embedding style, float baseQuality = 0.5f) {
    Creator c;
    c.id = CreatorId{0};
    c.styleEmbedding = std::move(style);
    c.baseQuality = baseQuality;
    return c;
}

// Independent reference implementation of the header's collapse priority, used to check the .cpp.
InteractionType referencePrimary(const BehaviourOutcome &o) {
    if (o.notInterested)
        return InteractionType::NotInterested;
    if (o.instantSkip)
        return InteractionType::InstantSkip;
    if (o.shared)
        return InteractionType::Share;
    if (o.followed)
        return InteractionType::FollowCreator;
    if (o.liked)
        return InteractionType::Like;
    if (o.rewatch)
        return InteractionType::Rewatch;
    if (o.completed)
        return InteractionType::CompleteWatch;
    if (o.watchRatio >= kImpressionMaxRatio)
        return InteractionType::PartialWatch;
    return InteractionType::Impression;
}

void checkInvariants(const BehaviourOutcome &o, const HiddenUserState &h, const Reel &r,
                     const BehaviourConfig &cfg) {
    // baseAffinity is exactly the hidden preference vs reel embedding dot product (TDD 10.1).
    EXPECT_FLOAT_EQ(o.baseAffinity, dot(h.hiddenPreference, r.embedding));

    // Engagement requires a completed watch.
    if (o.liked) {
        EXPECT_TRUE(o.completed);
    }
    if (o.shared) {
        EXPECT_TRUE(o.completed);
    }
    if (o.followed) {
        EXPECT_TRUE(o.completed);
    }

    // Instant skip and completion are mutually exclusive; skip => tiny watch ratio.
    if (o.instantSkip) {
        EXPECT_FALSE(o.completed);
        EXPECT_LE(o.watchRatio, kInstantSkipRatioHi);
    }

    // completed <=> high band; rewatch <=> watchRatio > 1.0 (only reachable inside the high band).
    if (o.completed) {
        EXPECT_GE(o.watchRatio, kHighRatioLo);
        EXPECT_LE(o.watchRatio, kHighRatioHi);
    } else {
        EXPECT_LT(o.watchRatio, kHighRatioLo);
    }
    EXPECT_EQ(o.rewatch, o.watchRatio > 1.0f);
    if (o.rewatch) {
        EXPECT_TRUE(o.completed);
    }

    // watchSeconds is exactly watchRatio * duration.
    EXPECT_FLOAT_EQ(o.watchSeconds, o.watchRatio * r.durationSeconds);

    // NotInterested only for very negative z.
    if (o.notInterested) {
        EXPECT_LT(o.behaviourScore, static_cast<float>(cfg.notInterestedZ));
    }

    // Collapsed primaryType matches the documented priority.
    EXPECT_EQ(o.primaryType, referencePrimary(o));
}

} // namespace

// Enumerate many simulated outcomes over a realistic dataset and assert every ground-truth
// invariant plus the primaryType collapse.
TEST(BehaviourModelTest, OutcomeFlagConsistencyOverDataset) {
    SimulationConfig cfg;
    cfg.reels = 400;
    cfg.users = 200;
    cfg.creators = 60;
    cfg.topics = 16;
    cfg.dimensions = 32;
    GeneratedDataset ds = generateDataset(cfg, 4242);

    BehaviourConfig bcfg;
    BehaviourModel model(bcfg);
    Rng rng(9001);
    Rng pick(123);

    for (int i = 0; i < 40000; ++i) {
        const auto &h = ds.hiddenStates[pick.uniformInt(ds.hiddenStates.size())];
        const auto &r = ds.reels[pick.uniformInt(ds.reels.size())];
        const auto &c = ds.creators[r.creatorId.value];
        BehaviourOutcome o = model.simulate(h, r, c, rng);
        checkInvariants(o, h, r, bcfg);
    }
}

// The collapse priority is exercised through every branch: random pairs, deliberately anti-aligned
// pairs (drive NotInterested / InstantSkip / Impression), and deliberately aligned high-propensity
// pairs (drive Share / FollowCreator / Like / Rewatch). All observed primaryTypes must match the
// reference collapse, and each documented type must be reachable.
TEST(BehaviourModelTest, PrimaryTypeCollapsePriorityAndCoverage) {
    const uint32_t dim = 32;
    BehaviourConfig bcfg;
    BehaviourModel model(bcfg);
    Rng rng(555);

    std::set<InteractionType> seen;
    auto run = [&](const HiddenUserState &h, const Reel &r, const Creator &c, int n) {
        for (int i = 0; i < n; ++i) {
            BehaviourOutcome o = model.simulate(h, r, c, rng);
            ASSERT_EQ(o.primaryType, referencePrimary(o));
            seen.insert(o.primaryType);
        }
    };

    Embedding pref = unitEmbedding(dim, 1);

    // Aligned + high propensity + high creator affinity => completed, like/share/follow, rewatch.
    // Propensities sit at the top of the documented trait ranges; anything higher saturates the
    // boosted like/share probabilities to ~1, which would mask the lower-priority primary types
    // (Like / FollowCreator / Rewatch / CompleteWatch behind an always-on Share).
    {
        HiddenUserState h = makeHidden(pref, /*like*/ 0.25f, /*share*/ 0.10f, /*tol*/ 1.0f);
        Reel r = makeReel(pref, /*quality*/ 0.9f, /*duration*/ 10.0f); // reel == preference => a~1
        Creator c = makeCreator(pref, 0.9f);                           // C ~ 1
        run(h, r, c, 20000);
    }
    // Anti-aligned => very negative z => instant skip / not interested; low-affinity partials with
    // near-zero watch ratio surface Impression.
    {
        HiddenUserState h = makeHidden(pref);
        Reel r = makeReel(negate(pref), 0.2f, 20.0f); // a ~ -1
        Creator c = makeCreator(negate(pref), 0.2f);  // C ~ -1
        run(h, r, c, 40000);
    }
    // Near-orthogonal, quality-only lift: partial watches at very-low affinity => Impression /
    // PartialWatch band.
    {
        Embedding ortho = unitEmbedding(dim, 777);
        HiddenUserState h = makeHidden(pref);
        Reel r = makeReel(ortho, 0.5f, 8.0f);
        Creator c = makeCreator(ortho, 0.5f);
        run(h, r, c, 40000);
    }

    for (InteractionType t :
         {InteractionType::Impression, InteractionType::InstantSkip, InteractionType::PartialWatch,
          InteractionType::CompleteWatch, InteractionType::Rewatch, InteractionType::Like,
          InteractionType::Share, InteractionType::FollowCreator, InteractionType::NotInterested}) {
        EXPECT_TRUE(seen.count(t) > 0) << "primaryType " << static_cast<int>(t) << " never fired";
    }
}

// Duration penalty D_v: isolate delta by zeroing every other z term and the noise. Then
// behaviourScore == -D_v, so we can read the penalty directly.
TEST(BehaviourModelTest, DurationPenaltyDirectionAndBounds) {
    const uint32_t dim = 16;
    BehaviourConfig bcfg;
    bcfg.alpha = 0.0;
    bcfg.beta = 0.0;
    bcfg.gamma = 0.0;
    bcfg.delta = 1.0;
    bcfg.noiseStd = 0.0; // deterministic z == -D_v
    BehaviourModel model(bcfg);
    Rng rng(1);

    Embedding pref = unitEmbedding(dim, 2);
    Creator c = makeCreator(unitEmbedding(dim, 3));

    auto penalty = [&](float duration, float tolerance) {
        HiddenUserState h = makeHidden(pref, 0.1f, 0.05f, tolerance);
        Reel r = makeReel(pref, 0.5f, duration);
        BehaviourOutcome o = model.simulate(h, r, c, rng);
        return -o.behaviourScore; // D_v
    };

    // D_v stays within [0, 1].
    for (float dur : {5.0f, 20.0f, 60.0f, 120.0f}) {
        for (float tol : {0.0f, 0.5f, 1.0f}) {
            float d = penalty(dur, tol);
            EXPECT_GE(d, 0.0f);
            EXPECT_LE(d, 1.0f);
        }
    }

    // Longer duration => larger penalty (lower z), at fixed tolerance.
    EXPECT_LT(penalty(10.0f, 0.5f), penalty(60.0f, 0.5f));
    EXPECT_LT(penalty(60.0f, 0.5f), penalty(115.0f, 0.5f));

    // Higher tolerance => smaller penalty (higher z), at fixed duration.
    EXPECT_GT(penalty(100.0f, 0.0f), penalty(100.0f, 0.5f));
    EXPECT_GT(penalty(100.0f, 0.5f), penalty(100.0f, 1.0f));
}

// NotInterested never fires unless z is below the configured threshold.
TEST(BehaviourModelTest, NotInterestedOnlyBelowThreshold) {
    const uint32_t dim = 16;
    BehaviourConfig bcfg;
    BehaviourModel model(bcfg);
    Rng rng(31337);

    Embedding pref = unitEmbedding(dim, 5);
    // Sweep affinity from very negative to very positive by blending +/- preference.
    int negFired = 0;
    for (int k = 0; k < 20000; ++k) {
        // Alternate strongly negative and positive pairs.
        const bool anti = (k % 2 == 0);
        Reel r = makeReel(anti ? negate(pref) : pref, 0.3f, 30.0f);
        Creator c = makeCreator(anti ? negate(pref) : pref, 0.3f);
        HiddenUserState h = makeHidden(pref);
        BehaviourOutcome o = model.simulate(h, r, c, rng);
        if (o.notInterested) {
            ++negFired;
            EXPECT_LT(o.behaviourScore, static_cast<float>(bcfg.notInterestedZ));
        }
    }
    // Sanity: the threshold path is actually reachable on the anti-aligned half.
    EXPECT_GT(negFired, 0);
}

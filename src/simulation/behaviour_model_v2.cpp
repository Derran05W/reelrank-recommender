#include "rr/simulation/behaviour_model_v2.hpp"

#include <algorithm>
#include <cmath>

#include "rr/core/embedding.hpp"
#include "rr/simulation/latent_model.hpp"

// BehaviourModelV2 (V2 TDD 4.3/4.4, Phase 14) — the latent-conditioned observable sampler that
// replaces the V1 BehaviourModel under realism.latent_reactions. Engagement is evidence, not
// truth (V2 TDD 3.2): every observable below is a NOISY, archetype-shaped, deliberately-imperfect
// function of the hidden LatentReaction, never a copy of it.
//
// The tuning surface (channel weights, propensities, wedge gains) lives in BehaviourV2Config; the
// SHAPE constants below are behavioural constants no planned experiment varies, so per D24 they
// are named constants documented here at their definition rather than premature config. All are
// the shipped operating point — the Phase 14/15 statistical signature tests calibrate against the
// config surface, not these.

namespace rr {
namespace {

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

// --- Watch / completion shape (V2 TDD 4.3/4.4; V1 TDD 10.3 lineage) ---------------------------
// The watch propensity is driven PRIMARILY by immediateSatisfaction but also by emotionalValue,
// so emotionally-intense-but-unsatisfying content (ragebait) still gets watched — the "long watch
// time with negative satisfaction" wedge (V2 TDD 3.2). A gaussian noise term keeps watch<->
// satisfaction correlation positive-but-imperfect (the Spearman-in-[0.2,0.8] target). openingHook
// SUPPRESSES instant-skip (strong opening retention); retentionDecay SUPPRESSES completion (early
// abandonment) — together they produce the clickbait signature: skip below population AND
// completion below population, watch ratio landing low-mid (hooked but abandoned).
constexpr double kWatchSatScale = 1.2; // satisfaction[-1,1] -> watch logit
constexpr double kWatchEmoScale = 1.5; // emotionalValue[0,1] -> watch logit (the ragebait hook)
// Arousal compels watching independent of valence (Phase 14 integration calibration): the reel's
// emotionalIntensity enters the watch logit directly, so high-arousal negative-valence content
// (ragebait, intensity ~0.85 vs population ~0.49) sustains above-population watch while its
// latent satisfaction stays negative — the V2 TDD 3.2 wedge. Deliberately weighted alongside a
// REDUCED satisfaction scale (2.4 -> 1.2 at integration) so valence still correlates with watch
// (the Spearman [0.2, 0.8] band) but no longer dominates it.
constexpr double kWatchIntensityScale = 1.7;
constexpr double kWatchNoiseStd = 0.9; // gaussian noise on the logit (imperfect correlation)
// Music-driven completion (V2 TDD 4.4 background_music): the music-channel match also enters the
// watch/completion logit, so centre-aligned users finish musically-matching reels even at weak
// topic match — completions are where rewatches come from.
constexpr double kWatchMusicScale = 0.8;
constexpr double kSkipBias = -0.3;        // baseline instant-skip logit bias
constexpr double kHookSkipSuppress = 3.0; // openingHook[0,1] pushes instant-skip down
// A short reel ends before a skip decision registers, so shortness ALSO suppresses instant-skip;
// together with the completion boost below this lets "completed-because-short" clear the skip gate
// (completed = completionDraw && !instantSkip) instead of being masked by it.
constexpr double kShortSkipSuppress = 1.3;
constexpr double kCompletionBias = 0.30;         // baseline completion logit bias
constexpr double kDecayCompletionSuppress = 3.0; // retentionDecay[0,1] pushes completion down

// --- Watch-ratio bands (V1 TDD 10.4 style), sampled uniformly within [lo, hi) -----------------
// instantSkip forces a tiny ratio; completed maps to a high band whose UPPER bound stretches with
// emotionalValue/satisfaction so emotionally-engaging completed reels rewatch (ratio > 1) and can
// replay whole times (ratio > 2) — the music/comfort rewatch signal. Every other (partial)
// impression draws a band selected by a latent-driven "watch depth" that the opening hook lifts,
// so a hooked-but-abandoned impression lands mid, not near-zero.
constexpr double kInstantSkipLo = 0.00;
constexpr double kInstantSkipHi = 0.05;
constexpr double kCompletedLo = 0.80;
constexpr double kCompletedBaseHi = 1.10; // base upper bound (V1's 1.20-ish high band)
constexpr double kReplayEmoGain = 1.3;    // emotionalValue stretches the completed upper bound
constexpr double kReplaySatGain = 0.6;    // strong satisfaction also stretches it
// Compulsive replay of high-arousal content (rage-rewatching while reading comments) and the
// familiar-song replay (V2 TDD 3.2: "a familiar song may produce a rewatch even when the topic
// is irrelevant"): emotionalIntensity and the MUSIC-channel match (dot(musicPreference,
// musicEmbedding), the modality channels' shared-centre structure) both stretch the completed
// band's upper bound — background_music reels' tight music coherence gives centre-aligned users
// a high match, so their completions rewatch above population even at weak topic match.
constexpr double kReplayIntensityGain = 0.9;
constexpr double kReplayMusicGain = 0.5;
// The familiar-song LOOP (the dominant music-rewatch mechanism): a completed, musically-matched
// impression multiplies its drawn watch ratio by (1 + boost*min(match, cap)) — matched users loop
// the song, pushing ratios across the rewatch boundary and into whole replays. Multiplicative and
// music-specific, so it moves the music cohort's rewatch RATE without inflating the population
// band the way the additive stretches do. Deterministic scaling of the already-drawn ratio: no
// extra rng draw, stream alignment unchanged.
constexpr double kMusicReplayBoost = 0.9;
constexpr double kMusicReplayMatchCap = 0.8;
constexpr double kHookWatchDepth = 0.45; // openingHook lifts the partial-watch depth
// "Can't look away": arousal also lifts how DEEP a non-completed watch runs before abandoning.
constexpr double kDepthIntensityGain = 0.30;
constexpr double kPartialVeryLowLo = 0.00, kPartialVeryLowHi = 0.10;
constexpr double kPartialLowLo = 0.10, kPartialLowHi = 0.40;
constexpr double kPartialMidLo = 0.40, kPartialMidHi = 0.80;
constexpr double kWatchDepthVeryLow = 0.15; // wd below -> very-low band
constexpr double kWatchDepthLow = 0.45;     // wd below -> low band; else mid band

// --- Engagement modulation (V2 TDD 3.2/4.3) ---------------------------------------------------
// like: satisfaction-correlated (affinity boost on the per-user likePropensity) + a social-
// conformity term from VISIBLE popularity counters (reel.likeCount/impressionCount) + DAMPED when
// informationalValue dominates emotionalValue (the "useful content is under-liked" mechanism:
// useful lands below population like rate while its saves run high). share/follow are V1-spirit:
// completed-gated, propensity-modulated, satisfaction-leaning; follow additionally needs high true
// creator affinity. All modulated by the V1 hidden traits (like/sharePropensity).
constexpr double kLikeSatGain = 2.0;        // affinity boost multiplier in [1, 1+gain]
constexpr double kUsefulLikeDampGain = 1.1; // (info - emo) damping strength
constexpr double kShareSatGain = 2.0;
constexpr float kFollowBaseProb = 0.10F;
constexpr float kFollowAffinityGain = 1.0F;
constexpr float kFollowCreatorAffinityThreshold = 0.30F;

// --- New V2 events (V2 TDD 4.3, task 3) -------------------------------------------------------
// comment: base commentPropensity boosted by controversy x emotionalIntensity and by
// clickbaitSusceptibility x clickbaitStrength, plus an emotional term — NOT satisfaction-gated, so
// ragebait comments above population even at negative satisfaction. save: savePropensity boosted
// by informationalValue + satisfaction (useful's co-signal). profile-visit: profileVisitPropensity
// boosted by desireForSimilarContent + creator attachment.
constexpr double kCommentControversyGain = 0.55; // controversy x emotionalIntensity
constexpr double kCommentClickbaitGain = 0.20;   // clickbaitSusceptibility x clickbaitStrength
constexpr double kCommentEmoGain = 0.08;         // mild emotionalValue term
constexpr double kSaveInfoGain = 0.35;           // informationalValue
constexpr double kSaveSatGain = 0.20;            // satisfaction (mapped to [0,1])
constexpr double kProfileDesireGain = 0.30;      // desireForSimilarContent (mapped to [0,1])
constexpr double kProfileCreatorGain = 0.20;     // creator attachment (hidden C term)

// --- Not-interested (V2 TDD 4.3): fires on strongly negative satisfaction / high regret --------
constexpr double kNotInterestedSatOffset = 0.35; // only |negative satisfaction| beyond this counts
constexpr double kNotInterestedSatGain = 0.5;
constexpr double kNotInterestedRegretOffset = 0.5; // only regret beyond this counts
constexpr double kNotInterestedRegretGain = 0.4;

// A non-skip impression whose sampled watch ratio is below this collapses to Impression; at or
// above it collapses to (at least) PartialWatch (mirrors the V1 model's threshold).
constexpr double kImpressionMaxRatio = 0.02;

// --- Phase 16 fatigue modulation shape constants (V2 TDD 4.7) ----------------------------------
// The alpha/beta/gamma repetition/creator/novelty coefficients live on SessionDynamicsConfig
// (tuning surface). These two are the flat drags no experiment varies (D24): general scrolling
// fatigue hits everyone regardless of repetition tolerance, and sustained high-arousal exposure +
// boredom sap emotionalValue. Documented at fatigueSatisfactionDelta / fatigueEmotionalDelta.
constexpr double kGeneralFatigueSatWeight = 0.5; // generalFatigue drag on immediateSatisfaction
constexpr double kEmoFatigueEmoWeight = 0.6;     // emotionalIntensityFatigue drag on emotionalValue
constexpr double kBoredomEmoWeight = 0.4;        // boredom drag on emotionalValue

// Lookup into a const fatigue map with a 0 default (operator[] would insert on a const map).
float fatigueOf(const std::unordered_map<TopicId, float> &m, TopicId k) {
    const auto it = m.find(k);
    return it == m.end() ? 0.0f : it->second;
}
float fatigueOf(const std::unordered_map<CreatorId, float> &m, CreatorId k) {
    const auto it = m.find(k);
    return it == m.end() ? 0.0f : it->second;
}

struct Band {
    double lo;
    double hi;
};

// Watch-ratio band from the outcome category + the latent. completedHi stretches with
// emotionalValue/satisfaction plus the arousal and music-match replay terms (constants above);
// the partial band is chosen from a hook- and arousal-lifted watch depth.
Band selectBand(bool instantSkip, bool completed, double satisfaction, double emotionalValue,
                float openingHook, double emotionalIntensity, double musicMatch) {
    if (instantSkip) {
        return {kInstantSkipLo, kInstantSkipHi};
    }
    if (completed) {
        const double hi = kCompletedBaseHi + kReplayEmoGain * clamp01(emotionalValue) +
                          kReplaySatGain * std::max(0.0, satisfaction) +
                          kReplayIntensityGain * clamp01(emotionalIntensity) +
                          kReplayMusicGain * std::max(0.0, musicMatch);
        return {kCompletedLo, hi};
    }
    const double wd = clamp01(0.5 * (satisfaction + 1.0) + kHookWatchDepth * clamp01(openingHook) +
                              kDepthIntensityGain * clamp01(emotionalIntensity));
    if (wd < kWatchDepthVeryLow) {
        return {kPartialVeryLowLo, kPartialVeryLowHi};
    }
    if (wd < kWatchDepthLow) {
        return {kPartialLowLo, kPartialLowHi};
    }
    return {kPartialMidLo, kPartialMidHi};
}

// Collapse the ground-truth flags into a single InteractionType using the FIXED V1 priority
// (behaviour_outcome.hpp): the V2 comment/save/profile-visit signals ride on their own flags and
// deliberately do NOT participate in this collapse (no new InteractionType enumerators).
InteractionType collapsePrimaryType(const BehaviourOutcome &o) {
    if (o.notInterested) {
        return InteractionType::NotInterested;
    }
    if (o.instantSkip) {
        return InteractionType::InstantSkip;
    }
    if (o.shared) {
        return InteractionType::Share;
    }
    if (o.followed) {
        return InteractionType::FollowCreator;
    }
    if (o.liked) {
        return InteractionType::Like;
    }
    if (o.rewatch) {
        return InteractionType::Rewatch;
    }
    if (o.completed) {
        return InteractionType::CompleteWatch;
    }
    if (o.watchRatio >= static_cast<float>(kImpressionMaxRatio)) {
        return InteractionType::PartialWatch;
    }
    return InteractionType::Impression;
}

} // namespace

BehaviourModelV2::BehaviourModelV2(const BehaviourConfig &v1Config, const BehaviourV2Config &config)
    : v1Config_(v1Config), config_(config) {}

void BehaviourModelV2::configureSessionDynamics(const SessionDynamicsConfig &config) {
    sessionConfig_ = config;
}

// Fatigue drag on immediateSatisfaction (the plan formula; see the header). alpha covers ALL
// repetition channels (topic + duration-format + music-repetition), modulated DOWN by the user's
// repetitionTolerance; beta covers same-creator fatigue, modulated down by creatorLoyalty; a flat
// generalFatigue drag; the novelty term ADDS utility when a novelty-hungry user meets novel
// content. All tolerance factors clamped to [0,1] (traits are sampled in [0,1], userTraitsV2).
double fatigueSatisfactionDelta(const SessionDynamicsConfig &cfg, const HiddenUserState &hidden,
                                const Reel &reel, const HiddenSessionState &session) {
    const double repFactor = clamp01(1.0 - static_cast<double>(hidden.repetitionTolerance));
    const double loyalFactor = clamp01(1.0 - static_cast<double>(hidden.creatorLoyalty));
    const double alphaEff = cfg.topicFatigueWeight * repFactor;
    const double betaEff = cfg.creatorFatigueWeight * loyalFactor;
    const double repetitionFatigue =
        static_cast<double>(fatigueOf(session.topicFatigue, reel.primaryTopic)) +
        static_cast<double>(session.formatFatigue) +
        static_cast<double>(session.musicRepetitionFatigue);
    const double creatorFatigue =
        static_cast<double>(fatigueOf(session.creatorFatigue, reel.creatorId));
    const double noveltyMatch =
        static_cast<double>(session.noveltyNeed) * clamp01(static_cast<double>(reel.novelty));
    return -alphaEff * repetitionFatigue - betaEff * creatorFatigue -
           kGeneralFatigueSatWeight * static_cast<double>(session.generalFatigue) +
           cfg.noveltyMatchWeight * noveltyMatch;
}

// Fatigue drag on emotionalValue: sustained high-arousal exposure and boredom both flatten affect.
double fatigueEmotionalDelta(const SessionDynamicsConfig & /*cfg*/, const HiddenSessionState &s) {
    return -kEmoFatigueEmoWeight * static_cast<double>(s.emotionalIntensityFatigue) -
           kBoredomEmoWeight * static_cast<double>(s.boredom);
}

BehaviourOutcome BehaviourModelV2::simulate(const HiddenUserState &hidden, const Reel &reel,
                                            const HiddenReelState &hiddenReel,
                                            const Creator &creator, Rng &behaviourRng,
                                            Rng &satisfactionRng, LatentReaction &latentOut,
                                            const HiddenSessionState *session) const {
    // Stage 1 — the hidden latent, on the "satisfaction" stream (package A). Its per-call draw
    // count on satisfactionRng is package A's FIXED contract (latent_model.hpp), independent of
    // archetype/branch.
    latentOut = computeLatentReaction(config_, hidden, reel, hiddenReel, creator, satisfactionRng);
    // Stage 1b — Phase 16 fatigue modulation (V2 TDD 4.7), session path ONLY. A fatigued user
    // genuinely enjoys the same reel less: adjust the latent's immediateSatisfaction/emotionalValue
    // HERE, before both the observable sampling below AND the caller's welfare accumulation (so the
    // welfare metrics see the fatigue-adjusted truth — the realism point of repetition). Pure
    // arithmetic on latentOut: zero rng draws, so the pinned draw order below is unchanged.
    // nullptr session (every pre-Phase-16 / gate-off call site) => byte-identical Phase 14.
    if (session != nullptr) {
        const double satAdj = static_cast<double>(latentOut.immediateSatisfaction) +
                              fatigueSatisfactionDelta(sessionConfig_, hidden, reel, *session);
        latentOut.immediateSatisfaction = static_cast<float>(std::clamp(satAdj, -1.0, 1.0));
        const double emoAdj = static_cast<double>(latentOut.emotionalValue) +
                              fatigueEmotionalDelta(sessionConfig_, *session);
        latentOut.emotionalValue = static_cast<float>(clamp01(emoAdj));
    }
    // Stage 2 — the observables, on the "behaviour" stream (owned wholesale here, D19).
    return sampleObservables(hidden, reel, hiddenReel, creator, latentOut, behaviourRng);
}

// PINNED per-impression draw order on `behaviourRng` (V2 owns the "behaviour" stream wholesale
// under realism.latent_reactions, D19; the count need not match V1's but is FIXED and documented
// here). Every draw below is taken in EXACTLY this order:
//   1. gaussian()  -> epsWatch, the noise folded into the watch/completion logit.
//   2. bernoulli() -> instantSkip = P(skip); openingHook suppresses it.
//   3. bernoulli() -> completionDraw = P(complete); retentionDecay suppresses it, short-duration
//                     reels get the config.shortCompletionBoost added (latent-independent);
//                     completed = completionDraw && !instantSkip.
//   4. uniform01() -> position within the category-selected watch-ratio band.
//   5. bernoulli() -> commented   (unconditional; ragebait signature, not satisfaction-gated).
//   6. bernoulli() -> saved       (unconditional; useful signature).
//   7. bernoulli() -> profileVisited (unconditional).
//   8. bernoulli() -> notInterested  (unconditional; probability is ~0 unless satisfaction is
//                     strongly negative or regret is high, so the draw count stays fixed).
//   9. ONLY when completed (the one documented branch, V1 precedent): bernoulli() x3 ->
//      liked, shared, followed.
// => 1 gaussian + 7 unconditional bernoulli/uniform draws, + 3 bernoulli iff completed. (The
// Simulator additionally draws one gaussian from this same stream for session-target sampling on
// the impressions that open/rotate a session — documented at Simulator::stepV2.)
BehaviourOutcome
BehaviourModelV2::sampleObservables(const HiddenUserState &hidden, const Reel &reel,
                                    const HiddenReelState &hiddenReel, const Creator &creator,
                                    const LatentReaction &latent, Rng &behaviourRng) const {
    BehaviourOutcome o{};

    const double s = latent.immediateSatisfaction;        // [-1, 1]
    const double regret = latent.regret;                  // [0, 1]
    const double info = latent.informationalValue;        // [0, 1]
    const double emo = latent.emotionalValue;             // [0, 1]
    const double desire = latent.desireForSimilarContent; // [-1, 1]
    const double satBoost01 = clamp01(0.5 * (s + 1.0));   // satisfaction mapped to [0, 1]

    // True creator affinity C = p_u . styleEmbedding (V1 TDD 10.2 C term). Simulator-side hidden
    // read (D11); never surfaced. The multi-channel base affinity is exposed on the outcome for the
    // evaluation carve-out / tests, exactly like V1 (no recommender consumer reads it).
    const double creatorAffinity = dot(hidden.hiddenPreference, creator.styleEmbedding);
    o.baseAffinity = dot(hidden.hiddenPreference, reel.embedding);

    // Music-channel match for the music-driven completion and familiar-song replay terms
    // (guarded like the latent model: zero when either side's modality embedding is absent or
    // mismatched — only reachable in unit tests that construct partial fixtures; real gate-on
    // datasets always populate both).
    const double musicMatch = (hidden.musicPreference.size() == reel.musicEmbedding.size() &&
                               !reel.musicEmbedding.empty())
                                  ? dot(hidden.musicPreference, reel.musicEmbedding)
                                  : 0.0;

    // --- Watch / completion (draws 1-3) --------------------------------------------------------
    const bool isShort = reel.durationSeconds < config_.shortDurationSeconds;
    const double epsWatch = behaviourRng.gaussian();
    const double baseLogit = kWatchSatScale * s + kWatchEmoScale * clamp01(emo) +
                             kWatchIntensityScale * clamp01(reel.emotionalIntensity) +
                             kWatchMusicScale * std::max(0.0, musicMatch) +
                             kWatchNoiseStd * epsWatch;
    o.behaviourScore = static_cast<float>(baseLogit);

    const double pInstantSkip =
        sigmoid(-baseLogit + kSkipBias - kHookSkipSuppress * clamp01(hiddenReel.openingHook) -
                (isShort ? kShortSkipSuppress : 0.0));
    o.instantSkip = behaviourRng.bernoulli(pInstantSkip);

    // Completed-because-short (V2 TDD 3.2): a latent-INDEPENDENT completion boost for reels shorter
    // than config.shortDurationSeconds.
    const double shortBoost = isShort ? config_.shortCompletionBoost : 0.0;
    const double pComplete =
        clamp01(sigmoid(baseLogit + kCompletionBias -
                        kDecayCompletionSuppress * clamp01(hiddenReel.retentionDecay)) +
                shortBoost);
    const bool completionDraw = behaviourRng.bernoulli(pComplete);
    o.completed = completionDraw && !o.instantSkip;

    // --- Watch ratio / rewatch / replay (draw 4) -----------------------------------------------
    const Band band = selectBand(o.instantSkip, o.completed, s, emo, hiddenReel.openingHook,
                                 reel.emotionalIntensity, musicMatch);
    const double bandPos = behaviourRng.uniform01();
    double drawnRatio = band.lo + bandPos * (band.hi - band.lo);
    if (o.completed) {
        // Familiar-song loop (kMusicReplayBoost above): deterministic multiplicative lift of the
        // drawn ratio for musically-matched completions; no extra rng draw.
        drawnRatio *=
            1.0 + kMusicReplayBoost * std::min(std::max(0.0, musicMatch), kMusicReplayMatchCap);
    }
    o.watchRatio = static_cast<float>(drawnRatio);
    o.watchSeconds = o.watchRatio * reel.durationSeconds;
    o.rewatch = o.watchRatio > 1.0F;
    // Whole EXTRA plays beyond the first: floor(watchRatio) - 1 for watchRatio > 1, else 0. A
    // rewatch in (1, 2) is a started-but-unfinished replay -> replayCount 0; ratio >= 2 -> >= 1.
    o.replayCount = o.watchRatio > 1.0F ? static_cast<uint32_t>(o.watchRatio) - 1U : 0U;

    // --- New V2 events (draws 5-7, unconditional) ----------------------------------------------
    const double pComment =
        clamp01(config_.commentPropensity +
                kCommentControversyGain * reel.controversy * reel.emotionalIntensity +
                kCommentClickbaitGain * hidden.clickbaitSusceptibility * reel.clickbaitStrength +
                kCommentEmoGain * emo);
    o.commented = behaviourRng.bernoulli(pComment);

    const double pSave =
        clamp01(config_.savePropensity + kSaveInfoGain * info + kSaveSatGain * satBoost01);
    o.saved = behaviourRng.bernoulli(pSave);

    const double desire01 = clamp01(0.5 * (desire + 1.0));
    const double pProfileVisit =
        clamp01(config_.profileVisitPropensity + kProfileDesireGain * desire01 +
                kProfileCreatorGain * clamp01(creatorAffinity));
    o.profileVisited = behaviourRng.bernoulli(pProfileVisit);

    // --- Not-interested (draw 8, unconditional; ~0 unless very negative satisfaction / high
    // regret)
    const double pNotInterested =
        clamp01(kNotInterestedSatGain * std::max(0.0, -s - kNotInterestedSatOffset) +
                kNotInterestedRegretGain * std::max(0.0, regret - kNotInterestedRegretOffset));
    o.notInterested = behaviourRng.bernoulli(pNotInterested);

    // --- Like / share / follow (draws 9a-9c, ONLY when completed — V1 precedent) ----------------
    if (o.completed) {
        // Satisfaction-correlated, damped when informationalValue dominates emotionalValue, lifted
        // by VISIBLE popularity (social conformity — uses the observable counters, not the latent).
        const double affinityBoost = 1.0 + kLikeSatGain * satBoost01;
        const double usefulDamp = clamp01(1.0 - kUsefulLikeDampGain * std::max(0.0, info - emo));
        const double visiblePopRate =
            reel.impressionCount > 0
                ? static_cast<double>(reel.likeCount) / static_cast<double>(reel.impressionCount)
                : 0.0;
        const double socialTerm = config_.socialConformityWeight * visiblePopRate;
        const double pLike =
            clamp01(clamp01(hidden.likePropensity * affinityBoost * usefulDamp) + socialTerm);
        o.liked = behaviourRng.bernoulli(pLike);

        const double pShare = clamp01(hidden.sharePropensity * (1.0 + kShareSatGain * satBoost01));
        o.shared = behaviourRng.bernoulli(pShare);

        const double pFollow =
            clamp01(kFollowBaseProb * (1.0 + kFollowAffinityGain * clamp01(creatorAffinity)));
        const bool followDraw = behaviourRng.bernoulli(pFollow);
        o.followed = followDraw && (creatorAffinity > kFollowCreatorAffinityThreshold);
    }

    o.primaryType = collapsePrimaryType(o);
    return o;
}

} // namespace rr

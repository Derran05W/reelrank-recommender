#include "rr/simulation/simulator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace rr {

namespace {

// Fixed per-impression browsing overhead added to the logical clock on every step, on top of the
// (rounded) time spent watching the reel. Models the seconds a user spends scrolling to and
// deciding on the next reel, and guarantees the clock strictly advances even on a zero-watch
// instant skip (D9 — simulated seconds, never wall clock).
constexpr Timestamp kBrowseOverheadSeconds = 2;

// Spread of per-session target lengths around hidden.avgSessionLength, as a fraction of the mean.
// Session lengths vary run-to-run but stay centred on the user's trait.
constexpr double kSessionLengthRelStddev = 0.25;

// Exponential decay factor 2^(-dt/halfLife) (dt in simulated seconds, D9) that brings a reel's
// trending accumulators forward from `updatedAt` to `now`, clamping to 1 when now <= updatedAt
// (TDD 12.4). This is the WRITE side of the trending twin; it intentionally MIRRORS
// rr::trendingDecayFactor (scoring.hpp — the READ side used by rr::trendingScore) rather than
// calling it, so rr_simulation stays free of the rr_recommend/vector-db link dependency (the
// module boundary declared in CMakeLists; inspect_user links rr_simulation alone). The two are
// kept in lockstep by the shared half-life semantics and cross-checked in simulator_test against
// rr::trendingDecayFactor as an independent oracle.
double trendingDecayForward(Timestamp updatedAt, Timestamp now, double halfLifeSeconds) {
    const double dt = now > updatedAt ? static_cast<double>(now - updatedAt) : 0.0;
    return std::exp2(-dt / halfLifeSeconds);
}

// Per-impression gains applied to the recommender-visible User::creatorAffinity estimate (TDD
// 12.6) from the OBSERVABLE outcome flags only — never from HiddenUserState (D11: the ground-truth
// creator affinity C_{u,c} is off-limits here; this is the system's LOGGED estimate). Signals are
// weighted by how strongly they express enjoyment of the creator: an explicit follow most, a plain
// completed watch least, an explicit "not interested" pushes the estimate down. The running
// per-creator affinity is accumulated and then clamped to [0, 1]; that CONTRACT is relied on by
// the sibling ranker package (the creator-affinity ranking feature) and by
// CreatorAffinityCandidateSource, so it must hold on every write.
constexpr float kAffinityFollowedGain = 0.25f;
constexpr float kAffinitySharedGain = 0.15f;
constexpr float kAffinityLikedGain = 0.10f;
constexpr float kAffinityCompletedGain = 0.02f;
constexpr float kAffinityNotInterestedGain = -0.20f;

// --- Phase 16 session-dynamics local helpers --------------------------------------------------
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
float clamp01f(float v) { return std::clamp(v, 0.0f, 1.0f); }

// Duration-format bucket (V2 TDD 4.7 "Format fatigue" on a repeated duration-bucket): short /
// medium / long, edges the named constants in rr::sessionDynamics.
int durationBucket(float seconds) {
    if (seconds < static_cast<float>(sessionDynamics::kDurationBucketShortSeconds)) {
        return 0;
    }
    if (seconds < static_cast<float>(sessionDynamics::kDurationBucketMediumSeconds)) {
        return 1;
    }
    return 2;
}

// Mean regret per reel this session (the exit logit's "recent regret" term / the Regret-exit
// discriminator). 0 on an empty session.
double meanRegret(const HiddenSessionState &s) {
    return s.impressionsThisSession > 0 ? static_cast<double>(s.accumulatedRegret) /
                                              static_cast<double>(s.impressionsThisSession)
                                        : 0.0;
}

} // namespace

Simulator::Simulator(const BehaviourConfig &behaviour, const RewardConfig &reward, Rng rng,
                     uint32_t recentWindow, double trendingHalfLifeSeconds)
    : behaviour_(behaviour), reward_(reward), rng_(std::move(rng)), recentWindow_(recentWindow),
      trendingHalfLifeSeconds_(trendingHalfLifeSeconds) {}

Simulator::Simulator(const BehaviourConfig &behaviour, const BehaviourV2Config &behaviourV2,
                     const RewardConfig &reward, Rng behaviourRng, Rng satisfactionRng,
                     uint32_t recentWindow, double trendingHalfLifeSeconds)
    : behaviour_(behaviour), reward_(reward), rng_(std::move(behaviourRng)),
      recentWindow_(recentWindow), trendingHalfLifeSeconds_(trendingHalfLifeSeconds),
      behaviourV2_(BehaviourModelV2(behaviour, behaviourV2)),
      satisfactionRng_(std::move(satisfactionRng)) {}

// Phase 16 constructor (realism.session_dynamics): additionally owns the exit/interruption streams
// and the session-dynamics coefficients, and hands those coefficients to BehaviourModelV2 for the
// fatigue modulation. stepV2 then runs the hidden-session lifecycle (below).
Simulator::Simulator(const BehaviourConfig &behaviour, const BehaviourV2Config &behaviourV2,
                     const SessionDynamicsConfig &sessionDynamics, const RewardConfig &reward,
                     Rng behaviourRng, Rng satisfactionRng, Rng sessionExitRng, Rng interruptionRng,
                     uint32_t recentWindow, double trendingHalfLifeSeconds)
    : behaviour_(behaviour), reward_(reward), rng_(std::move(behaviourRng)),
      recentWindow_(recentWindow), trendingHalfLifeSeconds_(trendingHalfLifeSeconds),
      behaviourV2_(BehaviourModelV2(behaviour, behaviourV2)),
      satisfactionRng_(std::move(satisfactionRng)), sessionDynamicsEnabled_(true),
      sessionDynamicsConfig_(sessionDynamics), sessionExitRng_(std::move(sessionExitRng)),
      interruptionRng_(std::move(interruptionRng)) {
    behaviourV2_->configureSessionDynamics(sessionDynamicsConfig_);
}

// ===== Phase 16 session-exit + fatigue machinery (V2 TDD 4.7-4.9) =============================
// Pure functions of config + state (+ this impression's reel/latent); NO rng, NO clock read.
// stepV2 composes them under realism.session_dynamics; unit tests drive them directly.

void startSession(const SessionDynamicsConfig &cfg, HiddenSessionState &s, Timestamp now) {
    // Away-time decay 2^(-awayGap/halfLife). awayGap comes from the SHARED logical clock: within
    // the round-robin runner `now` advances across all other users between this user's successive
    // sessions, so the gap = now(this session's first impression) - previousSessionEnd (the last
    // impression finish of the previous session). 0 on the first-ever session (no previous end).
    const double awayGap =
        s.previousSessionEnd == 0
            ? 0.0
            : static_cast<double>(now > s.previousSessionEnd ? now - s.previousSessionEnd : 0);
    const double decay = std::exp2(-awayGap / cfg.awayDecayHalfLifeSeconds);
    // SURVIVING fatigue channels decay but persist (repetition fatigue fades while away, V2 4.7).
    s.generalFatigue = static_cast<float>(s.generalFatigue * decay);
    s.formatFatigue = static_cast<float>(s.formatFatigue * decay);
    s.musicRepetitionFatigue = static_cast<float>(s.musicRepetitionFatigue * decay);
    s.emotionalIntensityFatigue = static_cast<float>(s.emotionalIntensityFatigue * decay);
    for (auto &kv : s.topicFatigue) {
        kv.second = static_cast<float>(kv.second * decay);
    }
    for (auto &kv : s.creatorFatigue) {
        kv.second = static_cast<float>(kv.second * decay);
    }
    // Carried-over satisfaction fades toward neutral at the same away half-life and SEEDS the new
    // session's EMA (startingSatisfaction is the "next-session starting satisfaction" hook, 4.9;
    // decay choice: same half-life as fatigue — long gaps reset mood, short gaps preserve it).
    s.startingSatisfaction = static_cast<float>(s.currentSatisfaction * decay);
    s.currentSatisfaction = s.startingSatisfaction;
    // Per-session accumulators reset. Long-term preferences live on User/HiddenUserState — NOT
    // touched here (Tier 2 acceptance: re-entry keeps prefs, gets a fresh session).
    s.accumulatedRegret = 0.0f;
    s.regretSum = 0.0f;
    s.satisfactionSum = 0.0f;
    s.watchSecondsSum = 0.0f;
    s.noveltyNeed = 0.0f;
    s.boredom = 0.0f;
    s.remainingAttention = 1.0f;
    s.impressionsThisSession = 0;
    s.consecutivePoorReels = 0;
    s.sessionStartTime = now;
    s.lastImpressionTime = now;
    s.lastDurationBucket = -1;
    s.lastMusicEmbedding.clear();
}

void accumulateFatigue(const SessionDynamicsConfig &cfg, HiddenSessionState &s, const Reel &reel,
                       const LatentReaction &latent, float watchSeconds) {
    using namespace sessionDynamics;
    const float w = std::max(0.0f, watchSeconds);
    // Topic + creator fatigue (config increments), each clamped [0,1].
    float &tf = s.topicFatigue[reel.primaryTopic];
    tf = clamp01f(tf + static_cast<float>(cfg.topicFatigueIncrement));
    float &cf = s.creatorFatigue[reel.creatorId];
    cf = clamp01f(cf + static_cast<float>(cfg.creatorFatigueIncrement));
    // Format fatigue on a repeated duration-bucket; always remember this reel's bucket.
    const int bucket = durationBucket(reel.durationSeconds);
    if (s.lastDurationBucket == bucket) {
        s.formatFatigue = clamp01f(s.formatFatigue + static_cast<float>(kFormatFatigueIncrement));
    }
    s.lastDurationBucket = bucket;
    // Music repetition when consecutive reels share a music centre (dot > threshold); remember it.
    if (!s.lastMusicEmbedding.empty() &&
        s.lastMusicEmbedding.size() == reel.musicEmbedding.size() && !reel.musicEmbedding.empty() &&
        static_cast<double>(dot(s.lastMusicEmbedding, reel.musicEmbedding)) >
            kMusicRepeatThreshold) {
        s.musicRepetitionFatigue =
            clamp01f(s.musicRepetitionFatigue + static_cast<float>(kMusicRepetitionIncrement));
    }
    s.lastMusicEmbedding = reel.musicEmbedding;
    // Emotional-intensity fatigue from sustained arousal exposure.
    s.emotionalIntensityFatigue =
        clamp01f(s.emotionalIntensityFatigue + static_cast<float>(kEmotionalIntensityFatigueScale) *
                                                   std::max(0.0f, reel.emotionalIntensity));
    // General scrolling fatigue from P14's fatigueDelta (emitted for exactly this, latent_model).
    s.generalFatigue = clamp01f(s.generalFatigue +
                                static_cast<float>(cfg.generalFatigueScale * latent.fatigueDelta));
    // Attention depletes with watch time.
    s.remainingAttention =
        clamp01f(s.remainingAttention - static_cast<float>(kAttentionDepletionPerWatchSecond) * w);
    // Novelty need rises with topic repetition (the just-updated topicFatigue), falls on novelty.
    s.noveltyNeed = clamp01f(s.noveltyNeed + static_cast<float>(kNoveltyNeedRise) * tf -
                             static_cast<float>(kNoveltyNeedFall) * std::max(0.0f, reel.novelty));
    // Boredom rises when the reel's (adjusted) emotional value is low.
    s.boredom =
        clamp01f(s.boredom +
                 static_cast<float>(kBoredomRise *
                                    std::max(0.0, kBoredomEmoThreshold -
                                                      static_cast<double>(latent.emotionalValue))));
    // Satisfaction EMA over the (fatigue-adjusted) latent — the welfare-visible session mood.
    s.currentSatisfaction = static_cast<float>(
        (1.0 - kSatisfactionEmaAlpha) * static_cast<double>(s.currentSatisfaction) +
        kSatisfactionEmaAlpha * static_cast<double>(latent.immediateSatisfaction));
    // Regret + U_s sums.
    s.accumulatedRegret += latent.regret;
    s.regretSum += latent.regret;
    s.satisfactionSum += latent.immediateSatisfaction;
    s.watchSecondsSum += w;
    // Poor-streak: consecutive reels whose latent satisfaction is below the poor threshold.
    if (static_cast<double>(latent.immediateSatisfaction) < kPoorSatisfactionThreshold) {
        s.consecutivePoorReels += 1;
    } else {
        s.consecutivePoorReels = 0;
    }
    s.impressionsThisSession += 1;
}

double sessionExitProbability(const SessionDynamicsConfig &cfg, const HiddenSessionState &s,
                              bool interruption) {
    const double logit = cfg.exitBias +
                         cfg.exitFatigueWeight * static_cast<double>(s.generalFatigue) +
                         cfg.exitRegretWeight * meanRegret(s) +
                         cfg.exitPoorStreakWeight * static_cast<double>(s.consecutivePoorReels) -
                         cfg.exitSatisfactionWeight * static_cast<double>(s.currentSatisfaction) +
                         cfg.exitInterruptionWeight * (interruption ? 1.0 : 0.0);
    return sigmoid(logit);
}

SessionExitType classifyExit(const SessionDynamicsConfig & /*cfg*/, const HiddenSessionState &s,
                             bool interruption) {
    using namespace sessionDynamics;
    // Order (V2 TDD 4.8): External > Failure (early + poor) > Regret > Fatigue > Satisfied. Never
    // RunEnded (that is the harness's label for sessions still open at run end).
    if (interruption) {
        return SessionExitType::External;
    }
    if (s.impressionsThisSession < kFailureMaxImpressions &&
        static_cast<double>(s.currentSatisfaction) < kFailureSatisfactionThreshold) {
        return SessionExitType::Failure;
    }
    if (meanRegret(s) > kRegretExitMeanThreshold) {
        return SessionExitType::Regret;
    }
    if (static_cast<double>(s.generalFatigue) > kFatigueExitThreshold ||
        static_cast<double>(s.remainingAttention) < kAttentionDepletedThreshold) {
        return SessionExitType::Fatigue;
    }
    return SessionExitType::Satisfied;
}

SessionRecord makeSessionRecord(const SessionDynamicsConfig &cfg, const HiddenSessionState &s,
                                UserId userId, SessionId sessionId, SessionExitType exitType,
                                Timestamp lastImpressionFinish) {
    SessionRecord rec;
    rec.userId = userId;
    rec.sessionId = sessionId;
    rec.exitType = exitType;
    rec.impressions = s.impressionsThisSession;
    rec.durationSeconds = static_cast<float>(
        lastImpressionFinish >= s.sessionStartTime ? lastImpressionFinish - s.sessionStartTime : 0);
    rec.satisfactionSum = s.satisfactionSum;
    rec.regretSum = s.regretSum;
    rec.harmfulFatigue =
        std::max(0.0f, static_cast<float>(static_cast<double>(s.generalFatigue) -
                                          sessionDynamics::kHarmfulFatigueThreshold));
    const double failure = exitType == SessionExitType::Failure ? 1.0 : 0.0;
    rec.sessionUtility =
        static_cast<float>(static_cast<double>(s.satisfactionSum) -
                           cfg.regretLambda * static_cast<double>(s.regretSum) -
                           cfg.fatigueLambda * static_cast<double>(rec.harmfulFatigue) -
                           cfg.failureExitLambda * failure);
    rec.startingSatisfaction = s.startingSatisfaction;
    rec.startTime = s.sessionStartTime;
    rec.endTime = lastImpressionFinish;
    return rec;
}

// Realism V2 step (Phase 14, realism.latent_reactions). Structurally a sibling of step(): the
// SAME reward (unchanged RewardModel on observables), clock advance (watch + browse overhead),
// session resolution (V1 rotation KEPT — P16 replaces it), reel counters/trending, and user
// bookkeeping/creator-affinity update — all byte-for-byte the V1 logic, but driven by
// BehaviourModelV2 (latent on the "satisfaction" stream, observables on the "behaviour" stream
// this model owns wholesale under the gate, D19) and populating the V2 InteractionEvent fields.
// step() is left completely untouched (D17: the V1 path is the golden baseline); the shared logic
// is intentionally duplicated here rather than refactored out of step().
//
// The latent (latentOut) flows ONLY to the harness's welfare accumulation (evaluation carve-out,
// D18); no latent value reaches the event or any recommender-visible structure.
StepResult Simulator::stepV2(User &user, const HiddenUserState &hidden, Reel &reel,
                             const Creator &creator, const StepV2Inputs &v2,
                             LatentReaction &latentOut, SessionRecord *closedSession) {
    assert(behaviourV2_.has_value() && "stepV2 requires the Realism V2 constructor");
    assert(v2.hiddenReel != nullptr && "StepV2Inputs.hiddenReel is required");
    StepResult result{};

    // 0. Phase 16 session lifecycle (realism.session_dynamics ONLY; nullptr session otherwise =>
    //    Phase 14 behaviour byte-identical). A session STARTS on the first-ever impression or the
    //    first after an exit closed one — both leave impressionsThisSession == 0. On start,
    //    startSession away-decays surviving fatigue + carried-over satisfaction and resets the
    //    per-session accumulators.
    HiddenSessionState *session = nullptr;
    if (sessionDynamicsEnabled_) {
        session = &hiddenSessions_[user.id];
        if (session->impressionsThisSession == 0) {
            startSession(sessionDynamicsConfig_, *session, now_);
        }
    }

    // 1. Ground-truth reaction. rng_ is the "behaviour" stream, satisfactionRng_ the "satisfaction"
    //    stream; BehaviourModelV2 draws the latent on the latter and the observables on the former.
    //    Under session dynamics the `session` pointer enables fatigue modulation: it adjusts the
    //    latent BEFORE both the observable sampling AND latentOut (welfare sees the adjusted
    //    truth).
    result.outcome = behaviourV2_->simulate(hidden, reel, *v2.hiddenReel, creator, rng_,
                                            satisfactionRng_, latentOut, session);
    const BehaviourOutcome &outcome = result.outcome;

    // 2. Reward: the UNCHANGED engagement-proxy RewardModel over the observable outcome (no extra
    //    randomness, no latent access).
    const float reward = reward_.reward(outcome);

    // 3. Timestamps + clock. startTimestamp = clock at playback start (BEFORE advancing);
    //    finishTimestamp = start + round(watchSeconds); then advance the clock by the same
    //    watch + browse overhead as V1 step (D9), so the V1 `timestamp` field keeps its V1 meaning
    //    (finish + browse overhead). dwellSeconds = watched seconds + the browse overhead.
    const Timestamp startTs = now_;
    const Timestamp watchedSeconds =
        static_cast<Timestamp>(std::lround(std::max(0.0f, outcome.watchSeconds)));
    const Timestamp finishTs = startTs + watchedSeconds;
    now_ += watchedSeconds + kBrowseOverheadSeconds;

    // 4. Session-id resolution. Gate-off: byte-identical V1 rotation (may draw one gaussian on the
    //    "behaviour" stream via sampleSessionTarget). Gate-on (session dynamics): the exit model
    //    owns session boundaries (step 8) — the V1 rotation DRAW is OWNED-OUT here (D19 wholesale
    //    replacement: the exit model replaces avgSessionLength rotation, so the "behaviour" stream
    //    takes NO rotation gaussian under the gate), and we only open the user's first session id.
    auto [it, inserted] = sessions_.try_emplace(user.id);
    SessionState &vsession = it->second;
    if (sessionDynamicsEnabled_) {
        if (inserted) {
            vsession.sessionId = SessionId{0};
            user.currentSessionLength = 0;
        }
    } else if (inserted) {
        vsession.sessionId = SessionId{0};
        vsession.targetLength = sampleSessionTarget(hidden.avgSessionLength);
        user.currentSessionLength = 0;
    } else if (user.currentSessionLength >= vsession.targetLength) {
        vsession.sessionId = SessionId{vsession.sessionId.value + 1};
        vsession.targetLength = sampleSessionTarget(hidden.avgSessionLength);
        user.currentSessionLength = 0;
    }

    // 5. Assemble the event: the nine V1 observable fields exactly as step(), then the V2
    //    observables (V2 TDD 5). requestId/positionInFeed/provenance/exploration come from the
    //    harness (StepV2Inputs); start/finish/dwell/replay/comment/save/profile come from this
    //    impression. observedExitAfterImpression is set in step 8 iff a probabilistic exit fires.
    //    Every field is an observable — no latent leaks (leak-audit enforced).
    result.event = InteractionEvent{user.id,
                                    reel.id,
                                    reel.creatorId,
                                    outcome.primaryType,
                                    outcome.watchSeconds,
                                    outcome.watchRatio,
                                    reward,
                                    now_,
                                    vsession.sessionId};
    result.event.positionInFeed = v2.positionInFeed;
    result.event.requestId = v2.requestId;
    result.event.requestTimestamp = v2.requestTimestamp;
    result.event.startTimestamp = startTs;
    result.event.finishTimestamp = finishTs;
    result.event.dwellSeconds =
        std::max(0.0f, outcome.watchSeconds) + static_cast<float>(kBrowseOverheadSeconds);
    result.event.replayCount = outcome.replayCount;
    result.event.commented = outcome.commented;
    result.event.saved = outcome.saved;
    result.event.profileVisited = outcome.profileVisited;
    result.event.fromExploration = v2.fromExploration;
    result.event.sourceProvenance = v2.sourceProvenance;

    // 6. Reel ground-truth counters — identical to V1 step.
    reel.impressionCount += 1;
    if (outcome.completed) {
        reel.completionCount += 1;
    }
    if (outcome.liked) {
        reel.likeCount += 1;
    }
    if (outcome.shared) {
        reel.shareCount += 1;
    }
    if (outcome.instantSkip) {
        reel.skipCount += 1;
    }

    // 6b. Trending accumulators — identical to V1 step (same 1/2/4 completion/like/share weights).
    const double decay =
        trendingDecayForward(reel.trendingUpdatedAt, now_, trendingHalfLifeSeconds_);
    reel.trendingEngagement *= decay;
    reel.trendingImpressions *= decay;
    reel.trendingImpressions += 1.0;
    reel.trendingEngagement += (outcome.completed ? 1.0 : 0.0) + (outcome.liked ? 2.0 : 0.0) +
                               (outcome.shared ? 4.0 : 0.0);
    reel.trendingUpdatedAt = now_;

    // 7. Recommender-visible user bookkeeping — identical to V1 step.
    user.seenReels.insert(reel.id);
    user.totalInteractions += 1;
    user.currentSessionLength += 1;
    user.recentInteractions.push_back(result.event);
    while (user.recentInteractions.size() > recentWindow_) {
        user.recentInteractions.pop_front();
    }

    // 7b. Creator-affinity estimate from OBSERVABLE flags only — identical to V1 step.
    const float affinityDelta = (outcome.followed ? kAffinityFollowedGain : 0.0f) +
                                (outcome.shared ? kAffinitySharedGain : 0.0f) +
                                (outcome.liked ? kAffinityLikedGain : 0.0f) +
                                (outcome.completed ? kAffinityCompletedGain : 0.0f) +
                                (outcome.notInterested ? kAffinityNotInterestedGain : 0.0f);
    if (affinityDelta != 0.0f) {
        float &affinity = user.creatorAffinity[reel.creatorId];
        affinity = std::clamp(affinity + affinityDelta, 0.0f, 1.0f);
    }

    // 8. Phase 16 session dynamics: fatigue accumulation, then the probabilistic classified exit
    //    (V2 TDD 4.7/4.8). Fatigue is accumulated AFTER the behaviour sample (draw order above is
    //    unchanged). The two exit draws happen EVERY impression, in a pinned order: the external-
    //    interruption hazard on interruptionRng_, THEN the exit bernoulli on sessionExitRng_ — a
    //    FIXED count of exactly one draw per stream per impression, so both streams stay aligned
    //    regardless of feed content, exit, or interruption (D8/D19).
    if (sessionDynamicsEnabled_) {
        accumulateFatigue(sessionDynamicsConfig_, *session, reel, latentOut, outcome.watchSeconds);
        session->lastImpressionTime = finishTs;

        const bool interruption =
            interruptionRng_.bernoulli(sessionDynamicsConfig_.externalInterruptionHazard);
        const double pExit = sessionExitProbability(sessionDynamicsConfig_, *session, interruption);
        const bool exit = sessionExitRng_.bernoulli(pExit);
        if (exit) {
            // Recommender-visible observable (P14 event field): last event of the session.
            result.event.observedExitAfterImpression = true;
            const SessionExitType type =
                classifyExit(sessionDynamicsConfig_, *session, interruption);
            if (closedSession != nullptr) {
                *closedSession = makeSessionRecord(sessionDynamicsConfig_, *session, user.id,
                                                   vsession.sessionId, type, finishTs);
            }
            // Rotate the V1-visible session id + reset lengths, coherent with V1 bookkeeping (the
            // exit REPLACES avgSessionLength rotation). impressionsThisSession=0 marks the NEXT
            // impression a session start (startSession then away-decays and resets).
            vsession.sessionId = SessionId{vsession.sessionId.value + 1};
            user.currentSessionLength = 0;
            session->previousSessionEnd = finishTs;
            session->impressionsThisSession = 0;
        }
    } else if (closedSession != nullptr) {
        // Gate-off: no session closes in stepV2 (the harness emits RunEnded records at run end).
        *closedSession = SessionRecord{};
    }

    return result;
}

uint32_t Simulator::sampleSessionTarget(float avgSessionLength) {
    // Gaussian around the mean with a proportional spread, clamped to at least one interaction so a
    // session always contains the reel that opened it. Uses rng_ so the sequence is deterministic.
    const double mean = std::max(1.0, static_cast<double>(avgSessionLength));
    const double sampled = mean + rng_.gaussian() * (mean * kSessionLengthRelStddev);
    const long rounded = std::lround(sampled);
    return static_cast<uint32_t>(std::max<long>(1, rounded));
}

StepResult Simulator::step(User &user, const HiddenUserState &hidden, Reel &reel,
                           const Creator &creator) {
    StepResult result{};

    // 1. Ground-truth reaction. HiddenUserState is handed only to the behaviour model here (D11).
    result.outcome = behaviour_.simulate(hidden, reel, creator, rng_);
    const BehaviourOutcome &outcome = result.outcome;

    // 2. Reward: pure function of the outcome, no extra randomness.
    const float reward = reward_.reward(outcome);

    // 3. Advance the logical clock by the watched seconds (rounded) plus browse overhead (D9). The
    //    clock therefore advances by at least kBrowseOverheadSeconds every step.
    const Timestamp watchedSeconds =
        static_cast<Timestamp>(std::lround(std::max(0.0f, outcome.watchSeconds)));
    now_ += watchedSeconds + kBrowseOverheadSeconds;

    // 4. Resolve the session this interaction belongs to. On first sight of the user we open a
    //    session; when the running currentSessionLength has reached the target we rotate to a new
    //    session (advance the id, reset the length, resample the target) BEFORE attributing this
    //    interaction, so this reel opens the new session.
    auto [it, inserted] = sessions_.try_emplace(user.id);
    SessionState &session = it->second;
    if (inserted) {
        session.sessionId = SessionId{0};
        session.targetLength = sampleSessionTarget(hidden.avgSessionLength);
        user.currentSessionLength = 0;
    } else if (user.currentSessionLength >= session.targetLength) {
        session.sessionId = SessionId{session.sessionId.value + 1};
        session.targetLength = sampleSessionTarget(hidden.avgSessionLength);
        user.currentSessionLength = 0;
    }

    // 5. Assemble the recommender-visible event. Every field is an observable interaction signal;
    //    no hidden preference vector or trait leaks through (D11).
    result.event = InteractionEvent{user.id,
                                    reel.id,
                                    reel.creatorId,
                                    outcome.primaryType,
                                    outcome.watchSeconds,
                                    outcome.watchRatio,
                                    reward,
                                    now_,
                                    session.sessionId};

    // 6. Update reel ground-truth counters, consistent with the outcome flags.
    reel.impressionCount += 1;
    if (outcome.completed) {
        reel.completionCount += 1;
    }
    if (outcome.liked) {
        reel.likeCount += 1;
    }
    if (outcome.shared) {
        reel.shareCount += 1;
    }
    if (outcome.instantSkip) {
        reel.skipCount += 1;
    }

    // 6b. Maintain the reel's trending accumulators (TDD 12.4): the exponentially-decayed twin of
    //     the popularity numerator. Decay both accumulators forward to this event's timestamp,
    //     then add exactly the increment the lifetime popularity counters gained for this event
    //     (+1 iff completed, +2 iff liked, +4 iff shared — the same 1/2/4 weights as
    //     popularityEngagement). rr::trendingScore reads these decayed forward again at query time,
    //     so the velocity is correct at any later `now` without a per-query update. Pure
    //     arithmetic on existing state — no rng draw (D8).
    const double decay =
        trendingDecayForward(reel.trendingUpdatedAt, now_, trendingHalfLifeSeconds_);
    reel.trendingEngagement *= decay;
    reel.trendingImpressions *= decay;
    reel.trendingImpressions += 1.0;
    reel.trendingEngagement += (outcome.completed ? 1.0 : 0.0) + (outcome.liked ? 2.0 : 0.0) +
                               (outcome.shared ? 4.0 : 0.0);
    reel.trendingUpdatedAt = now_; // set last, so the decay above used the PREVIOUS update time

    // 7. Update recommender-visible user bookkeeping.
    user.seenReels.insert(reel.id);
    user.totalInteractions += 1;
    user.currentSessionLength += 1;
    user.recentInteractions.push_back(result.event);
    while (user.recentInteractions.size() > recentWindow_) {
        user.recentInteractions.pop_front();
    }

    // 7b. Update the recommender-visible creator-affinity estimate from the OBSERVABLE outcome
    //     flags only (never HiddenUserState — D11). Accumulate the signed gains, then clamp the
    //     running per-creator estimate to [0, 1] (contract for the ranker + creator source). The
    //     map is only touched when some signal fired, so plain skips leave it untouched.
    const float affinityDelta = (outcome.followed ? kAffinityFollowedGain : 0.0f) +
                                (outcome.shared ? kAffinitySharedGain : 0.0f) +
                                (outcome.liked ? kAffinityLikedGain : 0.0f) +
                                (outcome.completed ? kAffinityCompletedGain : 0.0f) +
                                (outcome.notInterested ? kAffinityNotInterestedGain : 0.0f);
    if (affinityDelta != 0.0f) {
        float &affinity = user.creatorAffinity[reel.creatorId];
        affinity = std::clamp(affinity + affinityDelta, 0.0f, 1.0f);
    }

    return result;
}

Timestamp Simulator::now() const { return now_; }

// Phase 18 event-clock bridge: the event-driven runner sets now_ to the popped event's timestamp
// before stepV2 so away-gaps and start/finish timestamps track event time (see the header). Three
// lines by design — no other simulator state is touched; the legacy path never calls it.
void Simulator::syncClock(Timestamp t) { now_ = t; }

std::vector<SessionRecord> Simulator::drainOpenSessions() {
    std::vector<SessionRecord> drained;
    if (!sessionDynamicsEnabled_) {
        return drained;
    }
    // Ascending-UserId order: unordered_map iteration order is not deterministic, and these
    // records feed deterministic CSVs.
    std::vector<UserId> ids;
    ids.reserve(hiddenSessions_.size());
    for (const auto &[id, state] : hiddenSessions_) {
        if (state.impressionsThisSession > 0) {
            ids.push_back(id);
        }
    }
    std::sort(ids.begin(), ids.end(), [](UserId a, UserId b) { return a.value < b.value; });
    drained.reserve(ids.size());
    for (const UserId id : ids) {
        const HiddenSessionState &state = hiddenSessions_.at(id);
        const SessionId sessionId = sessions_[id].sessionId;
        drained.push_back(makeSessionRecord(sessionDynamicsConfig_, state, id, sessionId,
                                            SessionExitType::RunEnded, state.lastImpressionTime));
    }
    hiddenSessions_.clear();
    return drained;
}

} // namespace rr

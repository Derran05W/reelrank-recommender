#pragma once

#include <optional>
#include <unordered_map>
#include <vector>

#include "rr/domain/creator.hpp"
#include "rr/domain/interaction.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/learning/reward_model.hpp"
#include "rr/simulation/behaviour_model.hpp"
#include "rr/simulation/behaviour_model_v2.hpp"
#include "rr/simulation/hidden/hidden_session_state.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"

namespace rr {

// Phase 16 session-dynamics SHAPE constants (V2 TDD 4.7/4.8/4.9) that no planned experiment
// varies, so per D24 they are named constants here rather than config (the tuning surface is
// SessionDynamicsConfig). Exposed in the header (like userTraitsV2's ranges) so the fatigue/exit
// unit tests assert the exact contract. Documented at their use in simulator.cpp.
namespace sessionDynamics {
// --- Fatigue accumulation (4.7) ---------------------------------------------------------------
inline constexpr double kFormatFatigueIncrement = 0.10;   // per repeated duration-bucket
inline constexpr double kMusicRepetitionIncrement = 0.10; // per repeated music centre
inline constexpr double kMusicRepeatThreshold = 0.6;      // dot(prevMusic,curMusic) > => repeat
inline constexpr double kEmotionalIntensityFatigueScale = 0.08; // x reel.emotionalIntensity / reel
inline constexpr double kAttentionDepletionPerWatchSecond = 0.004; // remainingAttention drain
inline constexpr double kSatisfactionEmaAlpha = 0.4;      // weight on the new sample in the EMA
inline constexpr double kPoorSatisfactionThreshold = 0.0; // latent sat below => a "poor" reel
inline constexpr double kNoveltyNeedRise = 0.15;          // x topicFatigue (repetition)
inline constexpr double kNoveltyNeedFall = 0.10;          // x reel.novelty (novel content)
inline constexpr double kBoredomRise = 0.12;              // x (threshold - emotionalValue)+
inline constexpr double kBoredomEmoThreshold = 0.30;
inline constexpr double kDurationBucketShortSeconds = 15.0; // format bucket edges
inline constexpr double kDurationBucketMediumSeconds = 45.0;
// --- Exit classification (4.8 taxonomy) -------------------------------------------------------
inline constexpr uint32_t kFailureMaxImpressions = 5;        // "early" exit ceiling
inline constexpr double kFailureSatisfactionThreshold = 0.0; // currentSatisfaction below => poor
inline constexpr double kRegretExitMeanThreshold = 0.5;      // mean regret/reel above => Regret
inline constexpr double kFatigueExitThreshold = 0.6;         // generalFatigue above => Fatigue
inline constexpr double kAttentionDepletedThreshold = 0.25;  // remainingAttention below => Fatigue
// --- Session utility (4.9) --------------------------------------------------------------------
inline constexpr double kHarmfulFatigueThreshold = 0.5; // generalFatigue beyond this is "harmful"
} // namespace sessionDynamics

// --- Phase 16 session-exit + fatigue machinery, exposed for unit tests (package-A-owned, D18
// simulator/evaluation-side only). Pure functions of config + state (+ this impression's reel /
// latent); NONE draw rng or read a clock. stepV2 composes them under realism.session_dynamics. ---

// Session start (V2 TDD 4.7): away-decay the SURVIVING fatigue channels + carried-over
// satisfaction by 2^(-awayGap/awayDecayHalfLifeSeconds) (awayGap = now - previousSessionEnd, 0 on
// the first-ever session), seed startingSatisfaction/currentSatisfaction from the decayed carry-
// over, and reset the per-session accumulators. Long-term prefs live on User/HiddenUserState and
// are untouched here (Tier 2 acceptance). `now` is the logical clock at this impression's start.
void startSession(const SessionDynamicsConfig &cfg, HiddenSessionState &s, Timestamp now);

// Per-impression fatigue accumulation (V2 TDD 4.7), applied AFTER the behaviour sample from this
// impression's reel and (fatigue-adjusted) latent. Advances every channel + the satisfaction EMA,
// the U_s sums, impression/poor-streak counters, and the format/music repeat trackers.
void accumulateFatigue(const SessionDynamicsConfig &cfg, HiddenSessionState &s, const Reel &reel,
                       const LatentReaction &latent, float watchSeconds);

// P(exit) for this impression (V2 TDD 4.8): sigma(b0 + b1*generalFatigue + b2*meanRegret +
// b3*consecutivePoorReels - b4*currentSatisfaction + b5*interruption). meanRegret =
// accumulatedRegret / max(1, impressionsThisSession).
double sessionExitProbability(const SessionDynamicsConfig &cfg, const HiddenSessionState &s,
                              bool interruption);

// Classify a fired exit into the V2 TDD 4.8 taxonomy from documented named-constant thresholds.
// NEVER returns RunEnded (that label is the harness's, for sessions still open at run end).
SessionExitType classifyExit(const SessionDynamicsConfig &cfg, const HiddenSessionState &s,
                             bool interruption);

// Assemble the completed-session summary (V2 TDD 4.9) at exit: sums, duration, harmfulFatigue,
// U_s = satSum - l1*regretSum - l2*harmfulFatigue - l3*earlyFailureExit, startingSatisfaction.
SessionRecord makeSessionRecord(const SessionDynamicsConfig &cfg, const HiddenSessionState &s,
                                UserId userId, SessionId sessionId, SessionExitType exitType,
                                Timestamp lastImpressionFinish);

// One simulated impression: the ground-truth outcome plus the assembled event. The outcome is
// simulator/evaluation-side only (it derives from hidden state, TDD 18.2); recommenders may see
// the event but never the outcome.
struct StepResult {
    BehaviourOutcome outcome;
    InteractionEvent event;
};

// Per-impression V2 context (Phase 14), threaded by the harness from the feed-serving loop into
// stepV2 so the event can carry the V2 TDD 5 observables: which request served the item, where
// in the feed it sat, its candidate-source provenance, and the reel's hidden archetype state
// (simulator-side input for BehaviourModelV2 — never surfaced on the event).
struct StepV2Inputs {
    const HiddenReelState *hiddenReel = nullptr; // required (points into the dataset's vector)
    uint32_t positionInFeed = 0;
    uint64_t requestId = 0;
    Timestamp requestTimestamp = 0;
    bool fromExploration = false;
    CandidateSource sourceProvenance = CandidateSource::VectorHNSW;
};

// Drives the ground-truth interaction loop (TDD 10 + phase-3 task 5): given a user shown a reel,
// it simulates the reaction, computes the reward, assembles the InteractionEvent, advances the
// logical clock (D9 — simulated seconds, never wall clock), and updates reel counters
// (impressions/completions/likes/shares/skips) and recommender-visible user bookkeeping
// (seenReels, recentInteractions, totalInteractions, currentSessionLength, session rotation
// around hidden.avgSessionLength).
class Simulator {
  public:
    // `rng` is forked by the caller on the "behaviour" stream (D8); the Simulator owns it from
    // here on. `recentWindow` bounds User::recentInteractions (LearningConfig.recentWindow).
    // `trendingHalfLifeSeconds` is the half-life (RankingConfig.trendingHalfLifeSeconds) used to
    // decay each reel's trending accumulators forward on every impression (TDD 12.4).
    Simulator(const BehaviourConfig &behaviour, const RewardConfig &reward, Rng rng,
              uint32_t recentWindow, double trendingHalfLifeSeconds);

    // Realism V2 constructor (Phase 14, realism.latent_reactions): additionally owns the
    // caller-forked "satisfaction" stream and the BehaviourModelV2 parameters; step() keeps its
    // V1 behaviour byte-identically (D17 — the V1 model still serves it), while stepV2() runs
    // the latent-reaction pipeline. PACKAGE-B OWNERSHIP, FROZEN SIGNATURE (the harness calls
    // both constructors).
    Simulator(const BehaviourConfig &behaviour, const BehaviourV2Config &behaviourV2,
              const RewardConfig &reward, Rng behaviourRng, Rng satisfactionRng,
              uint32_t recentWindow, double trendingHalfLifeSeconds);

    // Phase 16 constructor (realism.session_dynamics): additionally owns the session-dynamics
    // coefficients and the caller-forked "session-exit" / "external-interruption" streams
    // (D19). stepV2 then runs the hidden-session lifecycle: fatigue accumulation + modulation,
    // probabilistic classified exit (replacing the V1 avgSessionLength rotation under this
    // gate), away-time decay on session start. PACKAGE-A OWNERSHIP, FROZEN SIGNATURE.
    Simulator(const BehaviourConfig &behaviour, const BehaviourV2Config &behaviourV2,
              const SessionDynamicsConfig &sessionDynamics, const RewardConfig &reward,
              Rng behaviourRng, Rng satisfactionRng, Rng sessionExitRng, Rng interruptionRng,
              uint32_t recentWindow, double trendingHalfLifeSeconds);

    // Simulate `user` being shown `reel`. Mutates `reel` counters and `user` bookkeeping;
    // advances the logical clock. HiddenUserState is read-only here and only ever handed to the
    // BehaviourModel (D11).
    StepResult step(User &user, const HiddenUserState &hidden, Reel &reel, const Creator &creator);

    // Realism V2 step (Phase 14): BehaviourModelV2 (latent on "satisfaction", observables on
    // "behaviour" — which V2 owns wholesale under the gate, D19), V2 event fields populated from
    // `v2` (position/request/provenance) and from the outcome (comment/save/profile-visit/
    // replay/dwell/start/finish). Fills `latentOut` for the harness's welfare accumulation
    // (evaluation carve-out); no latent value reaches the event or any recommender-visible
    // structure (leak-audit enforced). Requires the V2 constructor (asserts in Debug).
    // PACKAGE-B OWNERSHIP, FROZEN SIGNATURE.
    // `closedSession` (Phase 16, evaluation carve-out): under realism.session_dynamics, when this
    // impression fires a probabilistic exit (returned event.observedExitAfterImpression == true),
    // the completed SessionRecord is written here; on a non-exit impression the pointee is left
    // UNCHANGED, so callers gate on observedExitAfterImpression. Gate-off (Phase 14 ctor) never
    // fires an exit and writes a default RunEnded record; still-open sessions at run end are the
    // harness's to emit (RunEnded, excluded from exit-rate denominators).
    StepResult stepV2(User &user, const HiddenUserState &hidden, Reel &reel, const Creator &creator,
                      const StepV2Inputs &v2, LatentReaction &latentOut,
                      SessionRecord *closedSession = nullptr);

    // Phase 16 run-end drain (integration glue): emits a RunEnded SessionRecord for every
    // session still open when the run ends (impressionsThisSession > 0), in ascending-UserId
    // order (deterministic despite the unordered map), and clears the hidden-session table.
    // RunEnded records are excluded from exit-rate denominators (session-health metrics). No
    // rng draw, no clock advance; a no-op (empty result) unless the P16 constructor was used.
    std::vector<SessionRecord> drainOpenSessions();

    // Current logical time in simulated seconds (starts at 0).
    Timestamp now() const;

  private:
    // Per-user session bookkeeping. The recommender-visible User carries currentSessionLength but
    // not the SessionId, so the Simulator owns the id and the (rng-sampled) target length that
    // triggers rotation. avgSessionLength (hidden state) is read here ONLY to derive targetLength;
    // it is never copied into any recommender-visible field (D11).
    struct SessionState {
        SessionId sessionId{0};
        uint32_t targetLength = 1;
    };

    // Draw a per-session target length from hidden.avgSessionLength using rng_ (deterministic, D8).
    uint32_t sampleSessionTarget(float avgSessionLength);

    BehaviourModel behaviour_;
    RewardModel reward_;
    Rng rng_;
    uint32_t recentWindow_;
    double trendingHalfLifeSeconds_;
    Timestamp now_ = 0;
    std::unordered_map<UserId, SessionState> sessions_;

    // Realism V2 state (Phase 14): engaged only via the V2 constructor; V1-constructed
    // simulators carry std::nullopt plus a never-drawn seed-0 placeholder rng (Rng has no
    // default constructor) and never touch either.
    std::optional<BehaviourModelV2> behaviourV2_;
    Rng satisfactionRng_{0};

    // Phase 16 session-dynamics state (engaged only via the P16 constructor; placeholders
    // otherwise). PACKAGE-A OWNERSHIP of the machinery built on these.
    bool sessionDynamicsEnabled_ = false;
    SessionDynamicsConfig sessionDynamicsConfig_{};
    Rng sessionExitRng_{0};
    Rng interruptionRng_{0};
    std::unordered_map<UserId, HiddenSessionState> hiddenSessions_;
};

} // namespace rr

#pragma once

#include <cstdint>
#include <unordered_map>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/clock.hpp"

namespace rr {

// Hidden per-active-session user state (V2 TDD 4.6/4.7, Phase 16), owned SOLELY by the Simulator
// (D18): accumulates while a session runs, RESETS on session start (long-term preferences
// persist — Tier 2 acceptance), decays with away time. Consumed by BehaviourModelV2's fatigue
// modulation and the exit model; flows to evaluation only through SessionRecord (the completed-
// session summary). PACKAGE-A OWNERSHIP of semantics; the field list is the frozen cross-package
// surface.
struct HiddenSessionState {
    // --- V2 TDD 4.6 core -------------------------------------------------------------------
    float currentSatisfaction = 0.0f; // EMA of recent latent satisfaction, [-1, 1]
    float accumulatedRegret = 0.0f;   // running sum of latent regret this session
    float generalFatigue = 0.0f;      // scrolling fatigue, [0, 1]
    float noveltyNeed = 0.0f;         // rises with repetition, [0, 1]
    float boredom = 0.0f;             // rises with low emotional value, [0, 1]
    float remainingAttention = 1.0f;  // depletes with watch time, [0, 1]
    std::unordered_map<TopicId, float> topicFatigue;
    std::unordered_map<CreatorId, float> creatorFatigue;

    // --- V2 TDD 4.7 additional fatigue channels ---------------------------------------------
    float formatFatigue = 0.0f;             // duration-bucket repetition, [0, 1]
    float musicRepetitionFatigue = 0.0f;    // same-music-centre repetition, [0, 1]
    float emotionalIntensityFatigue = 0.0f; // sustained high-arousal exposure, [0, 1]

    // --- Session bookkeeping (exit model + classification inputs) ---------------------------
    uint32_t impressionsThisSession = 0;
    uint32_t consecutivePoorReels = 0; // latent satisfaction below the poor threshold, in a row
    float satisfactionSum = 0.0f;      // Σ satisfaction_t this session (U_s numerator)
    float regretSum = 0.0f;            // Σ regret_t this session
    float watchSecondsSum = 0.0f;      // session watch time (per-minute denominators)
    Timestamp sessionStartTime = 0;
    Timestamp lastImpressionTime = 0;
    // Logical time the PREVIOUS session ended (0 = never): the away-gap driving fatigue decay.
    Timestamp previousSessionEnd = 0;
    // currentSatisfaction carried over (decayed) from the previous session's end — the
    // "next-session starting satisfaction" measurement hook (V2 TDD 4.9).
    float startingSatisfaction = 0.0f;

    // --- Package-A cross-impression scratch (Phase 16) --------------------------------------
    // NOT part of the evaluation/SessionRecord surface (packages B/C never read these): the
    // one-impression memory the format- and music-repetition detectors need (V2 TDD 4.7 lists
    // "Format fatigue" on repeated duration-bucket and "Music repetition" on same-music-centre
    // repetition, neither of which is a function of the CURRENT reel alone). Reset on session
    // start (rr::startSession); mutated after each impression's fatigue accumulation.
    int32_t lastDurationBucket = -1; // duration bucket of the previous reel (-1 = none yet)
    Embedding lastMusicEmbedding{};  // music embedding of the previous reel (empty = none yet)
};

// Why a session ended (V2 TDD 4.8 taxonomy) — hidden simulator-side labels, evaluation-only.
enum class SessionExitType : uint8_t {
    Failure,   // early exit following poor recommendations
    Satisfied, // long productive session ending naturally
    Fatigue,   // acceptable recommendations but depleted attention
    External,  // independent interruption
    Regret,    // departure after ragebait/clickbait or repeated bad content
    RunEnded,  // simulation ended with the session still open (not a real exit; excluded from
               // exit-rate denominators, documented in the session-health metrics)
};

// One COMPLETED session (V2 TDD 4.9), emitted by Simulator::stepV2 when an exit fires (or by
// the harness for still-open sessions at run end). Carries hidden-derived values: evaluation
// carve-out only (D18) — never recommender-visible.
struct SessionRecord {
    UserId userId{};
    SessionId sessionId{};
    SessionExitType exitType = SessionExitType::RunEnded;
    uint32_t impressions = 0;
    float durationSeconds = 0.0f; // last impression finish - session start (time before exit)
    float satisfactionSum = 0.0f;
    float regretSum = 0.0f;
    float harmfulFatigue = 0.0f; // end-of-session general fatigue beyond the harmful threshold
    float sessionUtility = 0.0f; // U_s = satSum - l1*regretSum - l2*harmfulFatigue - l3*failure
    float startingSatisfaction = 0.0f; // what this session STARTED with (carry-over)
    Timestamp startTime = 0;
    Timestamp endTime = 0;
};

} // namespace rr

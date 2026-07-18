#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "rr/infrastructure/config.hpp"                  // SessionDynamicsConfig (U_s lambdas)
#include "rr/simulation/hidden/hidden_session_state.hpp" // SessionRecord + SessionExitType

namespace rr {

// ===========================================================================================
// Session-health metric group (V2 TDD §4.9/§6, Phase 16) — the evaluation carve-out module.
//
// V2 §6's four metric groups are reported as SEPARATE blocks and NO aggregate score is ever
// defined (D22). This header owns the "session health" group: it reduces the stream of COMPLETED
// SessionRecords the exit-aware harness collects (one per probabilistic exit; V2 TDD §4.8) into
// per-round and overall session statistics — session count, time-before-exit (mean/median),
// impressions/session, satisfaction- and regret-per-minute, the exit-type distribution, the
// early-failure-exit and natural-completion rates, the mean session utility U_s (V2 TDD §4.9),
// the §4.9 next-session starting-satisfaction measure, and the harmful-fatigue mean.
//
// Everything here is computed strictly inside the D18 EVALUATION CARVE-OUT: SessionRecord carries
// hidden-derived values (satisfaction/regret sums, harmful fatigue) that never reach any
// recommender-visible structure. The module is a pure reduction (no rng, no clock), so same-seed
// runs produce byte-identical reports (D8).
//
// TWO conventions the whole module is built on, documented once here:
//   1. CLOSED vs OPEN. A SessionRecord with exitType == RunEnded is an OPEN session at run end
//      (the simulation stopped before it exited); it is NOT a real exit. RunEnded records are
//      counted as `openSessions` and EXCLUDED from every mean/rate/median and from every
//      exit-type-share denominator. All aggregate statistics below are over CLOSED sessions.
//   2. PER-MINUTE DENOMINATOR. The session-health group's satisfaction-/regret-per-minute divide
//      by SESSION-DURATION minutes (durationSeconds / 60 — the "time before exit"), which is the
//      only time measure SessionRecord carries. This is DELIBERATELY distinct from the hidden-
//      user-welfare group's satisfaction-per-minute, which divides by WATCH-minutes; the two
//      groups measure different things (V2 §6 keeps the groups separate).
// ===========================================================================================

// Session utility U_s (V2 TDD §4.9): U_s = Σ satisfaction − λ1·Σ regret − λ2·harmfulFatigue −
// λ3·[earlyFailureExit]. The last term is the failure-exit indicator (1 iff exitType == Failure —
// the classification taxonomy's "early exit following poor recommendations", V2 §4.8). Recomputed
// on the evaluation side from the record's components under the reporting lambdas (rather than
// trusting the record's precomputed sessionUtility field) so the eval side owns a single testable
// U_s definition; the two agree by construction when package A uses the same lambdas. Directly
// unit-testable on a constructed record with a known SessionDynamicsConfig.
double sessionUtility(const SessionRecord &rec, const SessionDynamicsConfig &lambdas);

// Value accrued per minute of SESSION time (V2 §4.9 satisfaction-/regret-per-minute): sum divided
// by durationSeconds/60. Returns 0 when there is no session duration (guards divide-by-zero → never
// NaN). Shared by the per-round and overall reductions and directly unit-testable.
double perSessionMinute(double sum, double durationSeconds);

// Exit-type tally (V2 §4.8 taxonomy). Closed = the five real exit kinds; runEnded = open at run end
// (excluded from rate/share denominators, reported separately as open sessions).
struct ExitTypeCounts {
    std::size_t failure = 0;
    std::size_t satisfied = 0;
    std::size_t fatigue = 0;
    std::size_t external = 0;
    std::size_t regret = 0;
    std::size_t runEnded = 0;

    std::size_t closed() const { return failure + satisfied + fatigue + external + regret; }
    std::size_t total() const { return closed() + runEnded; }
    void tally(SessionExitType t);
};

// Per-round session-health point. Every mean/rate is over the round's CLOSED sessions (the sessions
// whose exit fired during this round); RunEnded records add only to `openSessions`.
struct SessionHealthRoundPoint {
    std::size_t round = 0;
    std::size_t sessions = 0;     // closed sessions this round (== exits.closed())
    std::size_t openSessions = 0; // RunEnded records attributed to this round (== exits.runEnded)
    ExitTypeCounts exits;

    double meanDurationSeconds = 0.0;   // mean time-before-exit
    double medianDurationSeconds = 0.0; // median time-before-exit
    double meanImpressions = 0.0;       // mean impressions/session
    double durationMinutes = 0.0;       // Σ durationSeconds / 60 (per-minute denominator)
    double satisfactionPerMinute = 0.0; // Σ satisfactionSum / durationMinutes
    double regretPerMinute = 0.0;       // Σ regretSum / durationMinutes
    double meanSessionUtility = 0.0;    // mean U_s

    double earlyFailureExitRate = 0.0;  // failure / closed
    double naturalCompletionRate = 0.0; // satisfied / closed
    double harmfulFatigueMean = 0.0;    // mean harmfulFatigue over closed sessions

    // §4.9 next-session measure, restricted to this round: mean startingSatisfaction over the
    // round's sessions that are NOT their user's first session (they followed an earlier session).
    double nextSessionStartingSatisfaction = 0.0;
    std::size_t linkedSessions = 0; // denominator of the above
};

// Overall session-health report (the module's reduction; also the ExperimentResult carrier that
// package C's comparison and the P16 statistical tests read). `configured` mirrors
// realism.session_dynamics: false leaves the whole block absent from output (byte-identical to a
// pre-Phase-16 run, D17). All means/rates are over CLOSED sessions (RunEnded excluded).
struct SessionHealthReport {
    bool configured = false;
    std::size_t sessions = 0;     // total closed sessions
    std::size_t openSessions = 0; // total RunEnded (open at run end)
    ExitTypeCounts exits;

    double meanDurationSeconds = 0.0;
    double medianDurationSeconds = 0.0;
    double meanImpressions = 0.0;
    double durationMinutes = 0.0;
    double satisfactionPerMinute = 0.0;
    double regretPerMinute = 0.0;
    double meanSessionUtility = 0.0;

    double earlyFailureExitRate = 0.0;
    double naturalCompletionRate = 0.0;
    double harmfulFatigueMean = 0.0;

    // §4.9 next-session starting satisfaction over the WHOLE run: mean startingSatisfaction over
    // every session that followed an earlier session of the same user (records linked in the
    // chronological order they were collected). `linkedSessions` is the denominator.
    double nextSessionStartingSatisfaction = 0.0;
    std::size_t linkedSessions = 0;

    std::vector<SessionHealthRoundPoint> byRound;

    // Share of CLOSED sessions of a given exit type (0 when no closed sessions). RunEnded is not a
    // closed type; openShare() reports RunEnded over ALL sessions instead.
    double exitShare(SessionExitType t) const;
    double openShare() const; // runEnded / total (0 when no sessions)
};

// The session-health accumulator (the module): fed one SessionRecord per collected session (on
// each probabilistic exit; RunEnded records only when a run-end drain feeds them — see the harness
// note), reduced to a SessionHealthReport. Constructed with the round count (for pre-sized,
// rehash-order-free per-round buckets) and the session-dynamics lambdas (for U_s). Pure reduction —
// no rng, no clock — so two same-seed runs produce byte-identical reports (D8). The next-session
// linkage is accumulated ONLINE, in the deterministic chronological order add() is called, so the
// floating-point sum is byte-identical across same-seed runs regardless of container hashing.
class SessionHealthMetrics {
  public:
    SessionHealthMetrics(std::size_t rounds, SessionDynamicsConfig lambdas);

    // Attribute one collected session to `round` (the round in which its exit fired / it was
    // collected). `round` must be in range. A RunEnded record increments openSessions only; every
    // other type is a closed session that feeds the round's and overall aggregates.
    void add(std::size_t round, const SessionRecord &rec);

    // Reduce to the report. `configured` is left default (false) — the caller sets it from the
    // gate flag. Deterministic.
    SessionHealthReport reduce() const;

    std::size_t sessions() const { return closedSessions_; } // closed count

  private:
    struct RoundAcc {
        ExitTypeCounts exits;
        double durationSum = 0.0;
        double impressionsSum = 0.0;
        double satisfactionSum = 0.0;
        double regretSum = 0.0;
        double utilitySum = 0.0;
        double harmfulFatigueSum = 0.0;
        std::vector<double> durations; // closed-session durations, for the median
        double nextStartSum = 0.0;
        std::size_t linked = 0;
    };

    static SessionHealthRoundPoint reduceRound(std::size_t round, const RoundAcc &acc);

    SessionDynamicsConfig lambdas_;
    std::vector<RoundAcc> rounds_;

    // Overall accumulators (mirror RoundAcc, summed across all rounds; kept separately so the
    // overall median has its own pooled duration list).
    ExitTypeCounts exits_;
    double durationSum_ = 0.0;
    double impressionsSum_ = 0.0;
    double satisfactionSum_ = 0.0;
    double regretSum_ = 0.0;
    double utilitySum_ = 0.0;
    double harmfulFatigueSum_ = 0.0;
    std::vector<double> durations_;
    double nextStartSum_ = 0.0;
    std::size_t linked_ = 0;
    std::size_t closedSessions_ = 0;

    // Per-user "have we seen a prior session?" flag, driving the online next-session linkage. Not
    // iterated (only per-key lookups), so its hash order never affects the deterministic sums.
    std::unordered_map<uint64_t, uint8_t> userHasPrior_;
};

// Median of a value list (V2 §6 time-before-exit median). Sorts a COPY (deterministic), returns the
// middle element for odd counts and the mean of the two middle elements for even counts; 0 for an
// empty list. Exposed for direct unit testing.
double medianOf(std::vector<double> values);

} // namespace rr

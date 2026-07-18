#pragma once

#include "rr/domain/behaviour_outcome.hpp"
#include "rr/domain/creator.hpp"
#include "rr/domain/reel.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/hidden/hidden_reel_state.hpp"
#include "rr/simulation/hidden/hidden_session_state.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/simulation/hidden/latent_reaction.hpp"

namespace rr {

// The Realism V2 synthetic ground-truth user (V2 TDD 4.3/4.4, Phase 14): every impression first
// computes a hidden LatentReaction (rr::computeLatentReaction, stream "satisfaction"), then
// samples OBSERVABLE behaviour conditionally on it (stream "behaviour", which this model owns
// WHOLESALE under realism.latent_reactions — D19; the V1 BehaviourModel is untouched and serves
// all gate-off runs, D17). Engagement is evidence, not truth (V2 TDD 3.2): watch/likes/comments
// correlate with the latent but are noisy, archetype-shaped, and deliberately misleading in the
// documented ways (ragebait watch+comment on negative satisfaction; completed-because-short;
// social-conformity likes from visible popularity counters; clickbait opening-hook retention
// with early abandonment).
//
// PACKAGE-B OWNERSHIP, FROZEN SIGNATURES: package B implements this class in
// behaviour_model_v2.cpp (currently a scaffolding stub that delegates to the V1 model). The
// constructor and simulate() signatures must not change (the Simulator's V2 path calls them);
// package B owns everything else in the file. The pinned per-impression draw order on BOTH
// streams must be documented at the definition.
class BehaviourModelV2 {
  public:
    BehaviourModelV2(const BehaviourConfig &v1Config, const BehaviourV2Config &config);

    // Phase 16 (PACKAGE-A): hand the model the session-dynamics coefficients (alpha/beta/gamma +
    // the U_s/exit tuning surface) used by the fatigue modulation. Called by the Simulator's P16
    // constructor; unset otherwise, and never consulted unless a non-null `session` is passed to
    // simulate() (the P16 path only) — so the frozen two-arg constructor and every gate-off /
    // Phase-14 call site stay byte-identical. Not on the frozen constructor (which the harness
    // and tests call unchanged); a separate setter keeps that signature stable.
    void configureSessionDynamics(const SessionDynamicsConfig &config);

    // Simulate one impression. Fills `latentOut` (hidden side — the caller routes it to welfare
    // accumulation via the evaluation carve-out; it must NEVER reach recommender-visible state)
    // and returns the observable outcome (V1 fields + the V2 comment/save/profile-visit/replay
    // signals). `behaviourRng` is the "behaviour" stream, `satisfactionRng` the "satisfaction"
    // stream, both caller-forked (D8/D19); never calls forkRng, never reads a clock.
    //
    // Two stages: (1) latentOut = rr::computeLatentReaction(...) on `satisfactionRng` (package A;
    // a neutral no-op stub in package B's tree), then (2) sampleObservables(...) on
    // `behaviourRng`. The pinned per-stream draw order is documented at sampleObservables in the
    // .cpp.
    // `session` (Phase 16, nullable): when non-null, fatigue modulation applies (effective
    // utility = base - alpha*topicFatigue - beta*creatorFatigue + gamma*noveltyMatch, per-user
    // tolerance-trait modulated) and the sampled observables reflect the session state.
    // PACKAGE-A OWNERSHIP; nullptr (every pre-Phase-16 call site) preserves Phase 14 behaviour
    // byte-identically.
    BehaviourOutcome simulate(const HiddenUserState &hidden, const Reel &reel,
                              const HiddenReelState &hiddenReel, const Creator &creator,
                              Rng &behaviourRng, Rng &satisfactionRng, LatentReaction &latentOut,
                              const HiddenSessionState *session = nullptr) const;

    // Sample the OBSERVABLE outcome conditionally on an EXPLICITLY supplied latent reaction,
    // drawing only from `behaviourRng` (the "behaviour" stream this model owns wholesale under the
    // gate, D19). This is stage (2) of simulate(): simulate() computes the latent via package A's
    // rr::computeLatentReaction and forwards it here. Exposed so unit tests can drive the
    // latent-conditioned sampling across a synthetic LatentReaction grid — package A's real
    // multi-channel latent is a stub (neutral zero) in package B's tree, so testing the wedges
    // (satisfaction->watch monotonicity, useful-underliked like damping, ragebait comments,
    // emotion-driven rewatch, ...) is only possible by constructing the latent directly. The
    // per-impression draw order on `behaviourRng` is FIXED and documented at the definition
    // (behaviour_model_v2.cpp). Never calls forkRng, never reads a clock, never mutates an input.
    BehaviourOutcome sampleObservables(const HiddenUserState &hidden, const Reel &reel,
                                       const HiddenReelState &hiddenReel, const Creator &creator,
                                       const LatentReaction &latent, Rng &behaviourRng) const;

  private:
    BehaviourConfig v1Config_;
    BehaviourV2Config config_;
    // Phase 16: session-dynamics coefficients for the fatigue modulation (default until
    // configureSessionDynamics runs; inert while `session` is null).
    SessionDynamicsConfig sessionConfig_{};
};

// Phase 16 fatigue modulation (V2 TDD 4.7), the plan formula
//   satisfaction delta = -alpha_eff*(topicFatigue+formatFatigue+musicRepetitionFatigue)
//                        -beta_eff*creatorFatigue - kGeneralFatigue*generalFatigue
//                        +gamma*(noveltyNeed * reel.novelty)
// where alpha_eff = topicFatigueWeight*(1-repetitionTolerance), beta_eff =
// creatorFatigueWeight*(1-creatorLoyalty), gamma = noveltyMatchWeight (per-user tolerance-trait
// modulation, P13 traits; heterogeneity EXPERIMENTS are P17). Free functions so unit tests assert
// the monotonicity + tolerance modulation directly; simulate() applies them to the latent's
// immediateSatisfaction/emotionalValue BEFORE both sampling and latentOut, so welfare metrics see
// the fatigue-adjusted truth (a fatigued user genuinely enjoys the same reel less). Pure
// arithmetic — no rng, so the fixed per-impression draw contract is preserved.
double fatigueSatisfactionDelta(const SessionDynamicsConfig &cfg, const HiddenUserState &hidden,
                                const Reel &reel, const HiddenSessionState &session);
double fatigueEmotionalDelta(const SessionDynamicsConfig &cfg, const HiddenSessionState &session);

} // namespace rr

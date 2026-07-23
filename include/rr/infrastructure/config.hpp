#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "rr/infrastructure/archetype_config.hpp"
#include "rr/infrastructure/cohort_config.hpp"

namespace rr {

// Every struct default-constructs to the TDD 21 suggested values. JSON keys are snake_case
// exactly as the TDD; struct members are camelCase (design decision D10). Unknown JSON keys are
// an error; missing keys keep the default (D6).

struct SimulationConfig {
    uint64_t seed = 42;
    uint32_t users = 10000;
    uint32_t reels = 100000;
    uint32_t creators = 5000;
    uint32_t topics = 32;
    uint32_t dimensions = 64;
    uint32_t interactionsPerUser = 200;
    // Mid-simulation entity injection (Phase 8, TDD 18.5 cold-start metrics): at the START of
    // round new_users_at / new_reels_at (0-based round index in the harness's interleaved loop),
    // inject that many freshly generated users / reels so genuinely-cold entities are measurable.
    // A count of 0 disables the injection (the "_at" value is then ignored). Injected entities are
    // generated on their own named rng streams, so enabling injection never perturbs the original
    // dataset (D8). An addition to the TDD 21 example, mandated by phase-8 task 5.
    uint32_t newUsers = 0;
    uint32_t newUsersAt = 0;
    uint32_t newReels = 0;
    uint32_t newReelsAt = 0;
    // Phase 18 (D20): runner selection. "round_robin" (the default, the legacy ExperimentRunner
    // loop — the D17 golden path, retained permanently) or "event_queue" (the EventDrivenRunner:
    // independent per-user timelines over a deterministic priority queue). Validated at load.
    std::string scheduler = "round_robin";
    // Phase 18 event-mode horizon: the simulation runs until the queue is exhausted or the
    // logical clock passes this many simulated seconds (0 under round_robin — ignored there;
    // event mode REQUIRES > 0, validated at load). Per-user interaction counts become an
    // OUTCOME in event mode; interactions_per_user is ignored there (documented).
    double horizonSeconds = 0.0;
    bool operator==(const SimulationConfig &) const = default;
};

// Phase 18 event-scheduling surface (V2 TDD 4.12; consumed only under
// simulation.scheduler == "event_queue", stream "scheduling" D19). Baseline models — the
// satisfaction-coupled retention arrives in P20.
struct SchedulingConfig {
    // Initial OpenApp times are staggered uniformly over [0, open_stagger_seconds).
    double openStaggerSeconds = 43200.0; // 12 simulated hours
    // Baseline return-delay model: after an exit, the user returns after
    // max(60, gaussian(mean/baselineDailyUsage-scaled, spread)) seconds. Package A documents the
    // exact formula; the per-user mean divides by the P13 baselineDailyUsage trait via the
    // documented mapping so heavy users return sooner.
    double returnDelayMeanSeconds = 21600.0; // 6 simulated hours at baselineDailyUsage == 1
    double returnDelaySpreadRel = 0.5;       // relative gaussian spread
    bool operator==(const SchedulingConfig &) const = default;
};

// Phase 19 serving-strategy surface (V2 TDD 4.13; consumed ONLY by the event runner —
// validated at load to require simulation.scheduler == "event_queue" when non-default). The
// freshness-versus-cost experiment axis.
struct ServingConfig {
    // Ranked reels fetched per RequestFeed. 0 (default) = recommendation.feed_size (the Phase 18
    // baseline behaviour, byte-identical).
    uint32_t prefetchDepth = 0;
    // Fire the next RequestFeed when the timeline's remaining prefetched inventory reaches this
    // count (0 = refill only when empty, the Phase 18 depth-1 semantics).
    uint32_t refillThreshold = 0;
    // Invalidate (drop) the remaining prefetched feed on a MAJOR session-intent change:
    // the user's session-preference vector swings beyond the cosine threshold below since the
    // feed was ranked, or a scheduled drift event fires for the user. Default OFF: already-
    // downloaded reels survive preference-estimate changes (the realistic client cache —
    // "preserve downloaded", V2 4.13).
    bool invalidateOnIntentChange = false;
    double intentSwingCosineThreshold = 0.5;
    bool operator==(const ServingConfig &) const = default;
};

struct RecommendationConfig {
    uint32_t feedSize = 10;
    uint32_t vectorCandidates = 500;
    uint32_t popularCandidates = 100;
    uint32_t freshCandidates = 100;
    uint32_t explorationCandidates = 50;
    // trending_candidates / creator_affinity_candidates are an addition to the TDD 21 example;
    // the counts themselves are mandated by TDD 13 (Trending: 100, Creator affinity: 100).
    uint32_t trendingCandidates = 100;
    uint32_t creatorAffinityCandidates = 100;
    bool operator==(const RecommendationConfig &) const = default;
};

struct HNSWConfig {
    uint32_t m = 16;
    uint32_t efConstruction = 200;
    uint32_t efSearch = 64;
    bool operator==(const HNSWConfig &) const = default;
};

struct RankingConfig {
    double similarityWeight = 0.50;
    double qualityWeight = 0.10;
    double freshnessWeight = 0.08;
    double popularityWeight = 0.07;
    double trendingWeight = 0.08;
    double creatorAffinityWeight = 0.07;
    double explorationWeight = 0.05;
    double repetitionPenalty = 0.15;
    // duration_match_weight / impression_penalty_weight are an addition to the TDD 14.2 example
    // formula: 14.1 mandates duration-preference match and previous-impression count as ranking
    // features, and 14.2 mandates every weight be configuration-driven. Duration match is a
    // bonus; the impression term is a fatigue penalty (subtracted, like repetition_penalty).
    double durationMatchWeight = 0.05;
    double impressionPenaltyWeight = 0.05;
    // Weight on the TDD 14.1 session-topic-similarity feature (cosine between the candidate and
    // the user's session preference vector), active from Phase 7 when session vectors start
    // updating online. An addition to the TDD 14.2 example weights (which predate the feature
    // list's session term); config-driven like every other weight per 14.2.
    double sessionTopicWeight = 0.05;
    // Decay half-lives (simulated seconds) for the two time-sensitive scoring inputs, an addition
    // to the TDD 21 example: freshness decay from createdAt is mandated configurable by TDD 8.2,
    // trending time-decay by TDD 12.4. Consumed by rr::freshnessScore / rr::trendingScore and by
    // Simulator::step's accumulator maintenance.
    double freshnessHalfLifeSeconds = 604800.0; // 7 simulated days
    double trendingHalfLifeSeconds = 21600.0;   // 6 simulated hours

    // --- Realism V2 ranking-feature weights (V2 TDD refs via plan Phase 15 task 3) ------------
    // The V2 features are EXTRACTED only when realism.content_v2 is on (FeatureExtractor gate);
    // every weight below defaults to 0.0 so a gate-on run with default weights ranks EXACTLY
    // like V1 — experiment presets (engagement / satisfaction-proxy arms) opt in per weight,
    // keeping arm comparisons single-variable. clickbait/emotional-intensity are the
    // engagement-arm wedge features; the satisfaction-proxy arm weights clickbait NEGATIVELY.
    double visualMatchWeight = 0.0;
    double musicMatchWeight = 0.0;
    double emotionalMatchWeight = 0.0;
    double clickbaitWeight = 0.0;
    double emotionalIntensityWeight = 0.0;
    double usefulnessWeight = 0.0;
    double productionQualityWeight = 0.0;
    double informationDensityWeight = 0.0;
    double languageMatchWeight = 0.0;
    double savePopularityWeight = 0.0;
    bool operator==(const RankingConfig &) const = default;
};

struct LearningConfig {
    // Master switch for online preference learning (Phase 7): false freezes every user at the
    // cold-start estimate (the pre-Phase-7 behaviour), giving the learning-vs-frozen experiment
    // arm mandated by the phase plan. An addition to the TDD 21 example.
    bool enabled = true;
    double longTermRate = 0.02;
    double sessionRate = 0.15;
    uint32_t recentWindow = 20;
    // TDD 11.3 session-decay lambda (suggested range 0.85-0.95): weight lambda^(n-i) applied to
    // the i-th most recent interaction when recomputing the session vector. An addition to the
    // TDD 21 example, mandated configurable by TDD 11.3.
    double sessionLambda = 0.90;
    // long_term_weight / session_weight are an addition to the TDD 21 example, mandated
    // configurable by TDD 8.3.
    double longTermWeight = 0.65;
    double sessionWeight = 0.35;
    // Realism V2 (Phase 15): EMA rate for the per-modality estimated preferences on User
    // (V2 TDD 5), mirroring the V1 11.2 long-term rule applied to the reel's modality
    // embeddings, driven by the same observable reward. Consumed only when realism.content_v2
    // is on (the modality embeddings exist); gate-off performs no modality updates (D17).
    double modalityRate = 0.02;
    bool operator==(const LearningConfig &) const = default;
};

struct ExplorationConfig {
    bool enabled = true;
    double epsilon = 0.05;
    // Phase 21 exploration time gate (contracts §1). Default -1.0 = NO time gating: behaviour is
    // exactly as before at all times. When >= 0: for any feed request with timestamp t where
    // floor(t / 86400) < enable_at_day, the ExplorationCandidateSource's per-slot epsilon gates use
    // EFFECTIVE epsilon = 0; at/after that simulated day, the configured `epsilon`. The per-slot
    // gates already draw bernoulli(epsilon) UNCONDITIONALLY for every slot, so the draw count and
    // recommender-stream alignment are IDENTICAL to a run without the gate — only the gate OUTCOMES
    // flip, and effective-0 outcomes reproduce today's epsilon=0 behaviour exactly (bernoulli(0)
    // consumes the same uniform01 draw as bernoulli(epsilon)). This lets the exploration-recovery
    // scenario switch exploration on partway through the run on the SAME world/seed. Consumed
    // inside ExplorationCandidateSource via the request timestamp; documented there at the epsilon
    // member.
    double enableAtDay = -1.0;
    // Recency window (simulated seconds) defining a "recently created" reel for the Fresh
    // candidate source (TDD 12.5) and the exploration source's random-fresh mode (TDD 12.7):
    // a reel qualifies at request time t iff createdAt >= t - fresh_window_seconds. An addition
    // to the TDD 21 example; TDD 12.5 defines no window, and experiments need to vary it.
    double freshWindowSeconds = 259200.0; // 3 simulated days
    // Phase-8 task 3: exploration candidates are protected from being ranked out entirely. The
    // final feed carries min(slots that fired the per-slot epsilon gate this request,
    // guaranteed_slots, exploration candidates available in the ranked pool) exploration items;
    // the promotion rule is documented at its implementation in the Orchestrator. epsilon = 0
    // fires no slots, so the guarantee is structurally inert there (the epsilon=0 no-op exit
    // criterion).
    uint32_t guaranteedSlots = 2;
    bool operator==(const ExplorationConfig &) const = default;
};

// TDD 10.2/10.3 behaviour-score parameters: z = alpha*a + beta*Q + gamma*C - delta*D + eps,
// P(complete) = sigmoid(z), P(instantSkip) = sigmoid(-z + skipBias). The TDD mandates these
// configurable but suggests no values; defaults here are calibrated so the Phase 3 statistical
// monotonicity tests pass at TDD-default data-generation settings (tests must not assert these
// exact values — they are tuning surface, not contract).
struct BehaviourConfig {
    double alpha = 4.0;              // weight on base affinity a = p_u . q_v            (TDD 10.1)
    double beta = 1.0;               // weight on intrinsic quality Q_v
    double gamma = 0.5;              // weight on true creator affinity C_{u,c}
    double delta = 1.0;              // weight on duration penalty D_v
    double noiseStd = 0.5;           // stddev of the gaussian noise term eps
    double skipBias = 1.0;           // b in P(instantSkip) = sigmoid(-z + b)
    double notInterestedZ = -2.0;    // NotInterested is possible only when z < this
    double notInterestedProb = 0.05; // ... and then fires with this probability
    bool operator==(const BehaviourConfig &) const = default;
};

// TDD 10.5 reward weights (defaults are the TDD's exact suggested values; the reward itself is
// clamped to [-1, 1]).
struct RewardConfig {
    double watchRatioWeight = 0.45;
    double watchSecondsWeight = 0.15; // applied to log(1 + watchSeconds), normalized
    double likeWeight = 0.15;
    double shareWeight = 0.20;
    double followWeight = 0.15;
    double instantSkipPenalty = 0.35;
    double notInterestedPenalty = 0.75;
    bool operator==(const RewardConfig &) const = default;
};

struct DiversityConfig {
    bool enabled = true;
    uint32_t maxPerCreator = 2;
    uint32_t maxPerTopic = 3;
    double mmrLambda = 0.75;
    // Phase 9: selects between the two diversity re-ranking modes the experiments compare —
    // constraints only (false) vs constraints + MMR ordering within them (true, TDD 15.2). An
    // addition to the TDD 21 example, mandated by the phase-9 experiment arms (hnsw_ranker vs
    // +constraints vs +constraints+MMR).
    bool useMmr = true;
    // Phase 17 personalized-diversity scaling surface (V2 TDD 4.10; consumed only under
    // realism.personalized_diversity): per-user effective caps scale within
    // [cap * personalizedCapScaleMin, cap * personalizedCapScaleMax] as a function of the
    // ESTIMATED tolerances, and the per-user MMR lambda interpolates
    // [personalizedLambdaMin, personalizedLambdaMax]. Package B documents the interpolation at
    // its definition.
    // 0.7 (integration-calibrated from 0.5): the tighter floor over-truncated intolerant users'
    // topic caps at medium scale — see the weighted_ranker.cpp calibration note.
    double personalizedCapScaleMin = 0.7;
    double personalizedCapScaleMax = 2.0;
    double personalizedLambdaMin = 0.60;
    double personalizedLambdaMax = 0.90;
    bool operator==(const DiversityConfig &) const = default;
};

// One weighted topic in a drift event's target mix (TDD 11.4). `topic` is a TopicId value that
// must exist in the generated topic set (validated by DriftScheduler at construction); `weight` is
// a relative weight > 0 — the mix is normalized at application, so weights need not sum to 1.
struct DriftTopicWeight {
    uint32_t topic = 0;
    double weight = 0.0;
    bool operator==(const DriftTopicWeight &) const = default;
};

// One scheduled hidden-preference change (TDD 11.4, Phase 10): when a user in the cohort reaches
// `atInteraction` completed interactions, their hidden preference is rebuilt from `topicMix`
// (interactions with 0-based index >= atInteraction see the new preference). The cohort is the
// deterministic, rng-free user slice with hash01(userId) in [cohortLo, cohortHi) — disjoint
// [lo, hi) ranges express disjoint cohorts drifting to different mixes. Defaults cover the whole
// population.
struct DriftEvent {
    uint32_t atInteraction = 0;
    double cohortLo = 0.0;
    double cohortHi = 1.0;
    std::vector<DriftTopicWeight> topicMix;
    bool operator==(const DriftEvent &) const = default;
};

// Scheduled interest drift (TDD 11.4, Phase 10). Empty `events` = drift disabled: every drift
// branch in the simulator/harness is inert and the run's output is byte-identical to a
// pre-Phase-10 run (the regression contract, same pattern as Phase 8 injection). An addition to
// the TDD 21 example, mandated by phase-10 task 1.
struct DriftConfig {
    std::vector<DriftEvent> events;
    bool operator==(const DriftConfig &) const = default;
};

// Realism V2 gates and parameters (V2 TDD, D17/D24 — Phase 13 onward). Every V2 mechanism ships
// behind this block with defaults that preserve V1 semantics EXACTLY: all gates default false,
// gate-off runs perform zero V2 rng draws and emit byte-identical V1 output (the committed
// tests/golden/v1-baseline/ anchor). Later phases add their gates here (latent_reactions,
// session_dynamics, personalized_diversity, ...); gates that require earlier gates throw at
// config load (fail-fast, D10).
struct RealismConfig {
    // Phase 13 gate: V2 multi-factor content and hidden user channels. On => generateDataset
    // augments reels/users from the new "archetypes"/"reels-v2"/"users-v2" streams (D19).
    bool contentV2 = false;
    // Phase 14 gate: latent reactions + BehaviourModelV2. On => every impression computes a
    // hidden LatentReaction (stream "satisfaction") from which observables are sampled
    // conditionally (BehaviourModelV2 owns stream "behaviour" wholesale under this gate, D19);
    // the V1 BehaviourModel is untouched and serves all gate-off runs. REQUIRES content_v2
    // (throws at config load otherwise, D17).
    bool latentReactions = false;
    // Phase 16 gate: hidden per-session state, fatigue dynamics, and the probabilistic
    // classified exit model (streams "session-exit"/"external-interruption", D19), replacing
    // the V1 avgSessionLength rotation under the gate — session length becomes an OUTCOME of
    // feed quality (V2 TDD 4.6-4.9). REQUIRES latent_reactions (throws at config load
    // otherwise).
    bool sessionDynamics = false;
    // Phase 17 gate: the observables-only ToleranceEstimator + PersonalizedDiversityReranker
    // (per-user caps/lambda/repetition scaling from ESTIMATED tolerances). REQUIRES
    // session_dynamics (the estimator reads exit-after-repetition signals; throws at load
    // otherwise). The fixed-diversity path is untouched (D17).
    bool personalizedDiversity = false;
    // Phase 20 gate: exposure-driven preference evolution + saturation/aversion. On => the
    // PreferenceEvolution component runs once per impression AFTER stepV2, mutating hidden
    // preference channels / exposure accumulators / retention.trust (V2 TDD 4.15/4.16). REQUIRES
    // session_dynamics (throws at config load otherwise, D17). Independent of retention.enabled —
    // either P20 gate may be on alone. Draws ZERO rng (the reserved "preference-evolution" stream
    // stays unused; D19), so a gate-off run is byte-identical to a pre-Phase-20 run.
    bool preferenceEvolution = false;
    // Size of the global language set (V2 TDD 4.1); language ids are 0..languages-1 with the
    // skewed distribution rr::languageWeights. Must be >= 1.
    uint32_t languages = 8;
    // Config-driven archetype catalog (V2 TDD 4.4, D24): defaults to the eight TDD archetypes;
    // configs may replace it wholesale. Must be non-empty.
    std::vector<ArchetypeSpec> archetypes = defaultArchetypeCatalog();
    // Phase 17 trait-cohort mixture (plan task 1): EMPTY (the default) keeps Phase 13's
    // continuous uniform trait sampling byte-identical; non-empty pins/mixes the population
    // across named tolerance-trait profiles (focused / novelty-seeker / creator-loyal /
    // easily-fatigued) for the fixed-vs-personalized experiments.
    std::vector<TraitCohortSpec> cohortMix{};
    bool operator==(const RealismConfig &) const = default;
};

// BehaviourModelV2 parameters (V2 TDD 4.3, Phase 14). The multi-channel base utility is a
// weighted combination of per-channel matches (V2 TDD 4.2/4.3 mandate the channel weights be
// config-driven); the exact formula is documented at rr::computeLatentReaction's definition
// (latent_model.cpp). Observable-sampling propensities for the NEW V2 events (comment / save /
// profile-visit) and the engagement-vs-truth wedge terms (social conformity, completed-because-
// short) are config-driven too — they are the tuning surface the Phase 14 statistical signature
// tests calibrate against. Defaults are the shipped operating point; tests must not assert these
// exact values (tuning surface, not contract).
struct BehaviourV2Config {
    // Channel-match weights (dot products of user preference channels vs reel embeddings).
    double topicWeight = 1.5;
    double visualWeight = 0.6;
    double musicWeight = 0.6;
    double emotionalWeight = 0.5;
    // Scalar content-value terms (each modulated by the matching hidden user preference).
    double usefulnessWeight = 0.8;
    double humourWeight = 0.5;
    double noveltyWeight = 0.4;
    double informationDensityWeight = 0.4;
    // Controversy vs tolerance: penalty beyond tolerance, small boost within for high-tolerance
    // users (V2 TDD 4.3).
    double controversyPenaltyWeight = 0.9;
    double controversyBoostWeight = 0.15;
    // Language mismatch penalty (scaled by 1 - languageMismatchTolerance).
    double languageMismatchPenalty = 0.5;
    // Creator attachment (hidden style affinity — the V1 TDD 10.2 C term, hidden side).
    double creatorAttachmentWeight = 0.4;
    // Gaussian noise std on the latent utility (stream "satisfaction").
    double latentNoiseStd = 0.30;
    // Observable-sampling surface (Phase 14 task 3): base propensities for the new V2 events and
    // the two mandated engagement-vs-truth wedges.
    double commentPropensity = 0.04;
    double savePropensity = 0.05;
    double profileVisitPropensity = 0.03;
    // Like-probability boost from VISIBLE popularity counters (social conformity, V2 TDD 3.2).
    double socialConformityWeight = 0.10;
    // Completed-because-short inflation (V2 TDD 3.2): completion boost for reels shorter than
    // shortDurationSeconds, independent of the latent reaction.
    double shortCompletionBoost = 0.8;
    double shortDurationSeconds = 12.0;
    bool operator==(const BehaviourV2Config &) const = default;
};

// Session-dynamics parameters (V2 TDD 4.6-4.9, Phase 16). All coefficients the TDD mandates
// config-driven: the fatigue modulation of BehaviourModelV2's utility (effective utility =
// base - alpha*topicFatigue - beta*creatorFatigue + gamma*noveltyMatch, coefficients modulated
// per-user by the Phase 13 tolerance traits — heterogeneity measured in P17), the away-time
// fatigue decay, the probabilistic exit logit (4.8), and the session-utility lambdas (4.9).
// Classification thresholds that no experiment varies stay named constants at their definition
// (D24). Defaults are the shipped operating point, calibrated so the Phase 16 statistical tests
// pass at TDD-default data settings; tests must not assert these exact values.
struct SessionDynamicsConfig {
    // Fatigue modulation weights (V2 TDD 4.7 example formula).
    double topicFatigueWeight = 0.6;   // alpha
    double creatorFatigueWeight = 0.4; // beta
    double noveltyMatchWeight = 0.3;   // gamma
    // Per-impression fatigue accumulation scales (package A documents the exact increments).
    double topicFatigueIncrement = 0.12;
    double creatorFatigueIncrement = 0.10;
    double generalFatigueScale = 1.0; // multiplies the latent fatigueDelta contribution
    // Away-time decay: fatigue halves every this many simulated seconds away (V2 TDD 4.7).
    double awayDecayHalfLifeSeconds = 3600.0;
    // Exit logit (V2 TDD 4.8): P(exit) = sigma(b0 + b1*fatigue + b2*recentRegret +
    // b3*consecutivePoorReels - b4*recentSatisfaction + b5*externalInterruption).
    double exitBias = -3.6;              // b0 (baseline exit hazard per impression)
    double exitFatigueWeight = 2.2;      // b1
    double exitRegretWeight = 1.6;       // b2
    double exitPoorStreakWeight = 0.30;  // b3
    double exitSatisfactionWeight = 1.4; // b4
    double exitInterruptionWeight = 6.0; // b5 (an interruption almost always ends the session)
    // Per-impression external-interruption hazard (independent stream, V2 TDD 4.8).
    double externalInterruptionHazard = 0.006;
    // Session-utility lambdas (V2 TDD 4.9): U_s = satSum - l1*regretSum - l2*harmfulFatigue -
    // l3*earlyFailureExit.
    double regretLambda = 1.0;      // l1
    double fatigueLambda = 0.5;     // l2
    double failureExitLambda = 2.0; // l3
    bool operator==(const SessionDynamicsConfig &) const = default;
};

// Exposure-driven preference-evolution parameters (V2 TDD 4.15, Phase 20). Consumed ONLY when
// realism.preference_evolution is on (which requires realism.session_dynamics). eta_evo is the
// per-impression base reinforcement rate η in p' = normalize((1-η_u)·p + η_u·s·v), where s is the
// HIDDEN latent satisfaction (NOT reward) and η_u = eta_evo scaled per user by the P13
// preferencePlasticity trait inside PreferenceEvolution. Every OTHER §4.16 constant
// (exhaustion/burnout/novelty/aversion scales + away-decay half-lives) stays a NAMED CONSTANT in
// preference_evolution.cpp (D24 no-premature-config); only the swept base rate is config surface,
// so this block holds just it.
struct EvolutionConfig {
    double etaEvo = 0.02;
    bool operator==(const EvolutionConfig &) const = default;
};

// Retention / churn parameters (V2 TDD 4.17, Phase 20). Consumed ONLY when retention.enabled is on
// (which requires realism.session_dynamics AND simulation.scheduler == "event_queue"): the
// RetentionModel replaces P18's baseline return-delay consumer under the gate (D19 wholesale
// replacement at the SAME "scheduling"-stream call site). All OTHER §4.17 shape constants (habit
// strengthen/decay rates, the time-of-day curve, trust coupling) stay NAMED CONSTANTS in
// retention_model.cpp (D24); only the churn threshold and hazard floor — the two knobs the
// failure-mode experiments sweep — are config surface.
struct RetentionConfig {
    // Master gate. Load-validation: requires realism.session_dynamics AND scheduler=='event_queue'.
    bool enabled = false;
    // A computed next-return delay strictly greater than this marks the user churned (the caller
    // then schedules NO ReturnToApp). Default 604800 = 7 simulated days.
    double churnDelayThresholdSeconds = 604800.0;
    // Minimum effective per-day return hazard for a non-churned user (bounds delays away from
    // infinity; exact use documented in retention_model.cpp). Default 0.02.
    double hazardFloor = 0.02;
    bool operator==(const RetentionConfig &) const = default;
};

// Explicit-feedback survey sub-block of learning_v2 (V2 TDD 4.19, Phase 22, contracts §1). Emits
// the ONLY hidden-derived training table (survey.csv): a sampled, Likert-quantized noisy
// immediateSatisfaction. OFF by default. When enabled it DRAWS on the D19-pinned
// "explicit-feedback" stream — exactly two draws per surveyed impression (rate bernoulli + gaussian
// noise), zero when disabled — so it never perturbs the V1 behaviour streams.
struct LearningV2SurveyConfig {
    bool enabled = false;
    double sampleRate = 0.02; // fraction of shown impressions that receive a survey draw
    double noiseSd = 0.35;    // gaussian noise sd added to immediateSatisfaction before quantizing
    bool operator==(const LearningV2SurveyConfig &) const = default;
};

// Phase 23 multi-objective value-function weights (V2 TDD §4.21, contracts §1). The learned value
// is  watch·pWatch + share·pShare + follow·pFollow + satisfaction·pSatisfaction − exit·pExit −
// regret·pRegret  (the six §4.21 terms; exit/regret enter negatively). Defaults are the balanced
// operating point (contracts §1); the frontier-sweep experiment arms vary watch/satisfaction
// against each other (renormalized) to expose the engagement↔welfare trade-off. `follow` stays
// small: P22 measured no learnable signal for the ~1.5%-base-rate follow target, so its predictor
// contributes little (finding cited at the default). Consumed only under
// learning_v2.learned_ranker.
struct LearningV2ValueWeights {
    double watch = 0.30;
    double share = 0.15;
    double follow = 0.10;
    double satisfaction = 0.30;
    double exit = 0.10;
    double regret = 0.05;
    bool operator==(const LearningV2ValueWeights &) const = default;
};

// Tier-5 learned-ranking pipeline gates (V2 TDD 4.19/4.22, Phase 22, contracts §1). All defaults
// preserve current behaviour: the whole block is inert until training_log (or survey.enabled) is
// turned on, and BOTH require the event runner (P22 logs the event-driven world only, V2 §9).
// SAMPLING DRAWS NO RNG — request selection is the pinned SplitMix64 hash
// pinnedHash01(requestId ^ salt) < rate (the two salts in learning_v2/training_log_schema.hpp), so
// enabling logging does not perturb any simulation stream (golden-tripwired, D19).
struct LearningV2Config {
    // Master gate for the impression-logging pipeline. Load-validation: requires
    // simulation.scheduler == "event_queue".
    bool trainingLog = false;
    // Fraction of REQUESTS whose SHOWN impressions are logged (features + outcomes join).
    double logSampleRate = 0.25;
    // Fraction of requests whose FULL RANKED POOL is logged (§4.22 eligibility/position-bias
    // support; ~500 rows per sampled request).
    double logPoolSampleRate = 0.01;
    // Rotation threshold for candidates/outcomes part files (`-partNNNN.csv`).
    uint64_t logMaxRowsPerFile = 2000000;
    LearningV2SurveyConfig survey;

    // --- Phase 23 in-loop learned ranking (V2 TDD §4.21, contracts §1) ------------------------
    // Master gate for serving the LearnedRanker (the §4.21 multi-objective value from the P22
    // models, retrained in-loop). Load-validation: requires trainingLog (the ranker trains on the
    // in-run log) — hence transitively event mode. Off by default => byte-identical (D17).
    bool learnedRanker = false;
    // The six §4.21 value-function weights (above).
    LearningV2ValueWeights valueWeights;
    // Simulated hours between in-loop retrains (deterministic schedule; the runner retrains at each
    // crossed boundary once the matrix reaches min_training_rows).
    double retrainEveryHours = 24.0;
    // Cold-start threshold: while the in-memory training matrix holds fewer than this many joined
    // shown rows, the LearnedRanker serves the hand-tuned WeightedRanker scores (fallback=1 in the
    // explanation). The first retrain at/after a boundary once the matrix crosses it swaps in
    // models.
    uint32_t minTrainingRows = 5000;
    // In-loop SGD epochs per retrain (the offline apps/train_models default 200 is unchanged).
    uint32_t retrainEpochs = 50;
    bool operator==(const LearningV2Config &) const = default;
};

// Evaluation-harness parameters (TDD 19 / phase-4 task 5). The oracle exhaustively scores all
// reels by true hidden affinity, so it runs only on a Bernoulli-sampled subset of requests; the
// rate is config-driven and recorded in every experiment's output.
struct EvaluationConfig {
    double oracleSampleRate = 0.05;
    // TDD 18.1 live retrieval metrics (Phase 5): fraction of requests on which the harness
    // compares the recommender's vector index against exact ground truth (Recall@K, distance
    // error). Exact search over the full corpus per sample keeps this a sampled measurement.
    double retrievalSampleRate = 0.02;
    // Phase 21 (contracts §2, D22 additive): emit the per-simulated-day ecosystem_metrics.csv +
    // the `ecosystem` summary block (creator HHI, tail-creator share, per-archetype impression
    // shares, niche-in-cohort match rate — the failure-mode scenario time series). Load-validation:
    // requires simulation.scheduler == "event_queue" (the metrics are per SIMULATED DAY, so they
    // need the event runner's day semantics). Default OFF => byte-identical output for every
    // existing run (D17); the accumulators and file are entirely behind this gate.
    bool ecosystemMetrics = false;
    bool operator==(const EvaluationConfig &) const = default;
};

// TDD 16.1-16.7, plus the Phase 15 evaluation-only oracle arm (V2 TDD 4.4 core experiment):
// OracleSatisfaction ranks the SEMANTIC candidate pool by expected hidden satisfaction. It is
// implemented in evaluation/ (D18: it reads hidden state), so the recommendation-side factory
// REJECTS it — only the ExperimentRunner may construct it, and it is barred from being a
// trainable/serving policy.
enum class RecommendationAlgorithm {
    Random,
    Popularity,
    ExactVector,
    Hnsw,
    HnswRanker,
    HnswRankerDiversity,
    HnswRankerExploration,
    OracleSatisfaction,
    // Phase 23 (contracts §1): the hnsw_ranker pipeline with the LearnedRanker in place of the
    // WeightedRanker (same sources; diversity off by default, as hnsw_ranker). Requires the
    // learning_v2.learned_ranker gate (validated at config load). Constructed by the EVENT RUNNER
    // (it needs the in-loop retrainer + a live handle to the ranker for model hot-swaps); the
    // recommendation-side factory REJECTS it, mirroring the OracleSatisfaction event-only
    // precedent.
    HnswLearnedRanker,
};

struct ExperimentConfig {
    SimulationConfig simulation;
    RecommendationConfig recommendation;
    RecommendationAlgorithm algorithm = RecommendationAlgorithm::Random;
    HNSWConfig hnsw;
    RankingConfig ranking;
    LearningConfig learning;
    ExplorationConfig exploration;
    DiversityConfig diversity;
    DriftConfig drift;
    BehaviourConfig behaviour;
    BehaviourV2Config behaviourV2;
    SessionDynamicsConfig sessionDynamics;
    SchedulingConfig scheduling;
    ServingConfig serving;
    EvolutionConfig evolution;
    RetentionConfig retention;
    LearningV2Config learningV2;
    RewardConfig reward;
    EvaluationConfig evaluation;
    RealismConfig realism;
    // Phase 21 scenario pre-registration carrier (contracts §4). A free-text top-level string,
    // default "", that a scenario config uses to record its pre-registered hypothesis / mechanism /
    // expected signature / verdict criteria BEFORE any run. Additive and purely documentary — no
    // subsystem reads it — but round-tripped through to_json so every experiment's fully-resolved
    // config.json echoes the block it was run under (the audit trail for the failure-mode suite).
    std::string description;
    bool operator==(const ExperimentConfig &) const = default;
};

// Serialization (defined in config.cpp). from_json throws std::invalid_argument on unknown keys.
void to_json(nlohmann::json &j, const SimulationConfig &c);
void from_json(const nlohmann::json &j, SimulationConfig &c);
void to_json(nlohmann::json &j, const RecommendationConfig &c);
void from_json(const nlohmann::json &j, RecommendationConfig &c);
void to_json(nlohmann::json &j, const HNSWConfig &c);
void from_json(const nlohmann::json &j, HNSWConfig &c);
void to_json(nlohmann::json &j, const RankingConfig &c);
void from_json(const nlohmann::json &j, RankingConfig &c);
void to_json(nlohmann::json &j, const LearningConfig &c);
void from_json(const nlohmann::json &j, LearningConfig &c);
void to_json(nlohmann::json &j, const ExplorationConfig &c);
void from_json(const nlohmann::json &j, ExplorationConfig &c);
void to_json(nlohmann::json &j, const DiversityConfig &c);
void from_json(const nlohmann::json &j, DiversityConfig &c);
void to_json(nlohmann::json &j, const DriftTopicWeight &c);
void from_json(const nlohmann::json &j, DriftTopicWeight &c);
void to_json(nlohmann::json &j, const DriftEvent &c);
void from_json(const nlohmann::json &j, DriftEvent &c);
void to_json(nlohmann::json &j, const DriftConfig &c);
void from_json(const nlohmann::json &j, DriftConfig &c);
void to_json(nlohmann::json &j, const BehaviourConfig &c);
void from_json(const nlohmann::json &j, BehaviourConfig &c);
void to_json(nlohmann::json &j, const RewardConfig &c);
void from_json(const nlohmann::json &j, RewardConfig &c);
void to_json(nlohmann::json &j, const EvaluationConfig &c);
void from_json(const nlohmann::json &j, EvaluationConfig &c);
void to_json(nlohmann::json &j, const RealismConfig &c);
void from_json(const nlohmann::json &j, RealismConfig &c);
void to_json(nlohmann::json &j, const BehaviourV2Config &c);
void from_json(const nlohmann::json &j, BehaviourV2Config &c);
void to_json(nlohmann::json &j, const SessionDynamicsConfig &c);
void from_json(const nlohmann::json &j, SessionDynamicsConfig &c);
void to_json(nlohmann::json &j, const SchedulingConfig &c);
void from_json(const nlohmann::json &j, SchedulingConfig &c);
void to_json(nlohmann::json &j, const ServingConfig &c);
void from_json(const nlohmann::json &j, ServingConfig &c);
void to_json(nlohmann::json &j, const EvolutionConfig &c);
void from_json(const nlohmann::json &j, EvolutionConfig &c);
void to_json(nlohmann::json &j, const RetentionConfig &c);
void from_json(const nlohmann::json &j, RetentionConfig &c);
void to_json(nlohmann::json &j, const LearningV2SurveyConfig &c);
void from_json(const nlohmann::json &j, LearningV2SurveyConfig &c);
void to_json(nlohmann::json &j, const LearningV2ValueWeights &c);
void from_json(const nlohmann::json &j, LearningV2ValueWeights &c);
void to_json(nlohmann::json &j, const LearningV2Config &c);
void from_json(const nlohmann::json &j, LearningV2Config &c);
void to_json(nlohmann::json &j, const RecommendationAlgorithm &a);
void from_json(const nlohmann::json &j, RecommendationAlgorithm &a);
void to_json(nlohmann::json &j, const ExperimentConfig &c);
void from_json(const nlohmann::json &j, ExperimentConfig &c);

// Map an algorithm to/from its canonical string. algorithmFromString throws
// std::invalid_argument (listing valid values) on an unknown name.
const char *toString(RecommendationAlgorithm a);
RecommendationAlgorithm algorithmFromString(const std::string &s);

// Load and parse a config file. Throws std::invalid_argument (with the path in the message) on a
// missing/unreadable file or a parse error.
ExperimentConfig loadExperimentConfig(const std::filesystem::path &path);

} // namespace rr

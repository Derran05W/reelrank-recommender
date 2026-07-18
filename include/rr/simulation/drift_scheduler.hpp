#pragma once

#include <cstdint>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/creator.hpp" // Topic
#include "rr/infrastructure/config.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"

namespace rr {

// Scheduled hidden-preference drift (TDD 11.4, Phase 10). The scheduler is a simulation-side
// component: it reads and writes ONLY HiddenUserState (the ground truth the simulator owns, D11)
// and is invoked in the harness loop immediately BEFORE each Simulator::step, mirroring how
// OnlineUserStateUpdater::apply runs immediately after it. It is rng-free and clock-free
// (deterministic by construction, D8): invoking it never perturbs any named rng stream.
//
// Semantics (pinned by the phase plan; both work packages and the tests rely on these exactly):
//  - An event fires for a user when the user's completed-interaction count equals
//    event.atInteraction at the moment maybeApply is called (the harness calls it before every
//    impression, and Simulator::step increments User::totalInteractions by exactly 1 per
//    impression, so every count value is hit exactly once per user — firing is exactly-once
//    without any per-user applied-state). Interactions with 0-based index >= atInteraction are
//    therefore simulated under the new preference (TDD 11.4's "interactions 500+" reading).
//    An atInteraction beyond a user's budget simply never fires (documented, not an error).
//  - Cohort membership is the deterministic hash test hash01(userId) in [cohortLo, cohortHi),
//    where hash01 maps a UserId to [0, 1) via the SplitMix64 finalizer (implementation documented
//    in the .cpp). Membership depends only on the userId value, so disjoint [lo, hi) ranges
//    partition any user population into disjoint cohorts, independent of user count or injection.
//  - Applying an event rebuilds hiddenPreference = normalize(sum_i weight_i * centre_i) over the
//    event's topic mix and replaces preferredTopics with the mix's topic ids (ground truth stays
//    honest for inspection tools). Every other HiddenUserState field (behavioural traits) is
//    untouched. No noise is added: the drifted preference sits exactly on the topic-mix manifold
//    (adding noise would need rng and break stream neutrality; recorded as documented behaviour).
//    preferenceStability is deliberately NOT consulted: scheduled drift models an exogenous,
//    experiment-controlled preference change; the stability trait is reserved for autonomous
//    (self-driven) drift, which is out of scope for Phase 10 (recorded in commit.md).
//  - Events are applied in their config order; if two events fire for the same user at the same
//    interaction, the LAST one in config order wins (documented; the published experiments use
//    disjoint cohorts so this never triggers).
class DriftScheduler {
  public:
    // Validates the drift config against the generated topic set and precomputes each event's
    // normalized target preference. Throws std::invalid_argument (setup error, fail fast per D10)
    // on: an event with an empty topicMix; a weight that is <= 0 or non-finite; a topic id not
    // present in `topics`; cohortLo/cohortHi outside [0, 1] or cohortLo >= cohortHi.
    DriftScheduler(const DriftConfig &config, const std::vector<Topic> &topics);

    // True when at least one drift event is configured. When false, maybeApply is a guaranteed
    // no-op and the harness writes pre-Phase-10 output byte-identically (regression contract).
    bool configured() const { return !events_.empty(); }

    // Deterministic, rng-free cohort membership: hash01(userId) in [cohortLo, cohortHi).
    static bool inCohort(UserId userId, double cohortLo, double cohortHi);

    // True iff at least one configured event's cohort contains this user — the "drifted cohort"
    // the adaptation metrics split on (drifted vs control).
    bool everApplies(UserId userId) const;

    // Earliest configured atInteraction (0 when unconfigured) — the anchor the adaptation
    // metrics use for pre-drift baselines and recovery windows.
    uint32_t firstDriftInteraction() const;

    // Apply every event whose atInteraction == totalInteractions and whose cohort contains
    // hidden.userId, per the semantics above. Returns true iff any event applied.
    bool maybeApply(HiddenUserState &hidden, uint32_t totalInteractions) const;

  private:
    std::vector<DriftEvent> events_;
    // events_[i]'s precomputed target: targets_[i] = normalize(sum weight * topic centre),
    // targetTopics_[i] = the mix's topic ids in config order.
    std::vector<Embedding> targets_;
    std::vector<std::vector<TopicId>> targetTopics_;
};

} // namespace rr

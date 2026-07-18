#include "rr/domain/serialization.hpp"

#include <nlohmann/json.hpp>

namespace rr {

namespace {

// Local InteractionType -> string mapping (D18 leak-audit scope: this is the ONLY place the
// nine InteractionType enumerators get a serialized spelling, snake_case per D6/D10, matching
// the existing rr::toString(RecommendationAlgorithm) convention in config.cpp). The switch is
// exhaustive over all nine enumerators; the trailing return (never reached) exists only to keep
// -Wall/-Wextra/-Werror quiet about a possible fall-through past the switch, same idiom as
// toString(RecommendationAlgorithm).
const char *interactionTypeToString(InteractionType type) {
    switch (type) {
    case InteractionType::Impression:
        return "impression";
    case InteractionType::InstantSkip:
        return "instant_skip";
    case InteractionType::PartialWatch:
        return "partial_watch";
    case InteractionType::CompleteWatch:
        return "complete_watch";
    case InteractionType::Rewatch:
        return "rewatch";
    case InteractionType::Like:
        return "like";
    case InteractionType::Share:
        return "share";
    case InteractionType::FollowCreator:
        return "follow_creator";
    case InteractionType::NotInterested:
        return "not_interested";
    }
    return "impression";
}

} // namespace

// D18 leak-audit schema for InteractionEvent: every InteractionEvent member is observable (V1
// carries no hidden field on this struct), so the emitted key set is a direct 1:1 snake_case
// mapping of the nine members. tests/unit/leak_audit_test.cpp asserts this exact key set — ANY
// field added to InteractionEvent (Phase 14 onward, V2 TDD S5) must be a conscious addition here
// AND to that test's allowlist in the same commit; hidden/latent fields (satisfaction, regret,
// archetype, ...) must never appear (V2 TDD S5: "Do not include hidden satisfaction directly").
void to_json(nlohmann::json &j, const InteractionEvent &e) {
    j = nlohmann::json{
        {"user_id", e.userId.value},
        {"reel_id", e.reelId.value},
        {"creator_id", e.creatorId.value},
        {"type", interactionTypeToString(e.type)},
        {"watch_seconds", e.watchSeconds},
        {"watch_ratio", e.watchRatio},
        {"reward", e.reward},
        {"timestamp", e.timestamp},
        {"session_id", e.sessionId.value},
    };
}

// D18 leak-audit schema for User (the recommender-visible profile — D11 already keeps hidden
// preference off this struct entirely). seen_reels / creator_affinity / recent_interactions are
// emitted as COUNTS, not their contents: the audit cares about which KEYS are observable, and
// serializing (say) a 100k-entry seen-reel set would be both useless for the audit and enormous
// in any training log that reuses this function (D18: "the P22 training log MUST serialize
// events through this function"). Extend both this function and the allowlist test together if a
// field is ever added to User.
void to_json(nlohmann::json &j, const User &u) {
    j = nlohmann::json{
        {"id", u.id.value},
        {"estimated_preference", u.estimatedPreference},
        {"long_term_preference", u.longTermPreference},
        {"session_preference", u.sessionPreference},
        {"seen_reels", u.seenReels.size()},
        {"creator_affinity", u.creatorAffinity.size()},
        {"recent_interactions", u.recentInteractions.size()},
        {"total_interactions", u.totalInteractions},
        {"current_session_length", u.currentSessionLength},
    };
}

} // namespace rr

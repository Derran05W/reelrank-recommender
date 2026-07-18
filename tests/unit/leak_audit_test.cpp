// D18 leak-audit scaffolding (Phase 13, plan task 7): InteractionEvent and User are the two
// recommender-visible structs whose serialized JSON form is a hidden-state leak surface — the
// P22 training log and any future logging path MUST serialize through rr::to_json (declared in
// rr/domain/serialization.hpp), never ad hoc, so THIS test is the single place that has to be
// extended whenever either struct's field list changes.
//
// NOTE FOR PHASE 14 (V2 TDD S5): InteractionEvent gains new fields there (position in feed,
// feed/request id, request timestamp, start/finish timestamps, dwell, replay count,
// comment/save/profile-visit flags, observedExitAfterImpression). Whoever lands that phase MUST
// extend BOTH src/domain/serialization.cpp's to_json(InteractionEvent) AND kEventAllowlist below,
// in the SAME commit as the struct change — that review moment (a new key fails this test until
// consciously allowlisted as observable) is the entire point of the audit, not something to
// silence by widening the allowlist speculatively ahead of time.

#include "rr/domain/serialization.hpp"

#include "rr/domain/interaction.hpp"
#include "rr/domain/user.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <set>
#include <string>
#include <vector>

using namespace rr;

namespace {

// The D18 allowlists: the ONLY keys each schema may ever emit. A field added to the struct
// without a conscious update here (and to src/domain/serialization.cpp) fails
// SchemaMatchesAllowlistExactly with an extra key; a field removed without updating here fails
// with a missing key. Both directions are deliberate — this is not a "no new keys" gate, it is a
// "no UNREVIEWED keys" gate.
const std::set<std::string> kEventAllowlist = {
    "user_id",     "reel_id", "creator_id", "type",       "watch_seconds",
    "watch_ratio", "reward",  "timestamp",  "session_id",
};

const std::set<std::string> kUserAllowlist = {
    "id",
    "estimated_preference",
    "long_term_preference",
    "session_preference",
    "seen_reels",
    "creator_affinity",
    "recent_interactions",
    "total_interactions",
    "current_session_length",
};

// Substrings that must never appear in ANY emitted key, in either schema, at any nesting depth
// (belt-and-braces per plan task 7(c)): these name the simulator-only hidden types/fields V2
// introduces (HiddenReelState's archetypeIndex/satisfactionBias/regretBias, HiddenUserState,
// LatentReaction's regret/satisfaction fields, ...). The exact-allowlist tests below already
// catch any of today's fields leaking; this check additionally guards against a future
// serialization change that adds a plausibly-named-but-forbidden key (e.g. "hidden_archetype")
// that some other allowlist edit accidentally OK'd.
constexpr const char *kForbiddenSubstrings[] = {"hidden", "satisfaction", "regret", "archetype",
                                                "latent"};

// Recursively collects every object key in `node`, at any nesting depth (objects and arrays are
// both walked) — future schema extensions may nest structures, and the substring check should not
// need re-deriving for that; it walks whatever shape to_json happens to produce.
void collectKeys(const nlohmann::json &node, std::vector<std::string> &out) {
    if (node.is_object()) {
        for (auto it = node.begin(); it != node.end(); ++it) {
            out.push_back(it.key());
            collectKeys(it.value(), out);
        }
    } else if (node.is_array()) {
        for (const auto &item : node) {
            collectKeys(item, out);
        }
    }
}

// A fully-populated InteractionEvent (every field set to a concrete, non-default-looking value)
// so a failing test's dumped JSON is meaningful to read, not a wall of zeros.
InteractionEvent fullyPopulatedEvent() {
    InteractionEvent e{};
    e.userId = UserId{7};
    e.reelId = ReelId{99};
    e.creatorId = CreatorId{3};
    e.type = InteractionType::Rewatch;
    e.watchSeconds = 12.5F;
    e.watchRatio = 0.8F;
    e.reward = 0.42F;
    e.timestamp = 123456;
    e.sessionId = SessionId{5};
    return e;
}

// A fully-populated User: non-empty preference vectors, a couple of seen reels / creator
// affinities / recent interactions (so the *_reels/*_affinity/*_interactions COUNT fields are
// exercised at count > 0, not vacuously 0), and non-zero counters.
User fullyPopulatedUser() {
    User u{};
    u.id = UserId{11};
    u.estimatedPreference = {0.1F, 0.2F, 0.3F};
    u.longTermPreference = {0.4F, 0.5F, 0.6F};
    u.sessionPreference = {0.7F, 0.8F, 0.9F};
    u.seenReels = {ReelId{1}, ReelId{2}, ReelId{3}};
    u.creatorAffinity = {{CreatorId{1}, 0.5F}, {CreatorId{2}, 0.25F}};
    u.recentInteractions = {fullyPopulatedEvent(), fullyPopulatedEvent()};
    u.totalInteractions = 42;
    u.currentSessionLength = 6;
    return u;
}

std::set<std::string> topLevelKeys(const nlohmann::json &j) {
    std::set<std::string> keys;
    for (auto it = j.begin(); it != j.end(); ++it) {
        keys.insert(it.key());
    }
    return keys;
}

} // namespace

// (a) InteractionEvent's emitted key set must EXACTLY equal kEventAllowlist: no extra key, no
// missing key.
TEST(LeakAuditTest, InteractionEventSchemaMatchesAllowlistExactly) {
    const InteractionEvent e = fullyPopulatedEvent();
    nlohmann::json j = e;
    const std::set<std::string> keys = topLevelKeys(j);

    EXPECT_EQ(keys, kEventAllowlist)
        << "InteractionEvent's serialized schema no longer matches the D18 leak-audit "
           "allowlist.\n"
        << "A NEW field on InteractionEvent must be CONSCIOUSLY allowlisted here as an "
           "observable (update BOTH src/domain/serialization.cpp's to_json(InteractionEvent) "
           "and kEventAllowlist in this file) before this test may pass again; a field that "
           "disappeared should have its key removed from the allowlist instead. Hidden/latent "
           "simulator-only fields (satisfaction, regret, archetype, fatigue, ...) must NEVER be "
           "added to this schema — V2 TDD S5: \"Do not include hidden satisfaction directly.\"\n"
        << "  emitted JSON: " << j.dump();
}

// (b) same for User.
TEST(LeakAuditTest, UserSchemaMatchesAllowlistExactly) {
    const User u = fullyPopulatedUser();
    nlohmann::json j = u;
    const std::set<std::string> keys = topLevelKeys(j);

    EXPECT_EQ(keys, kUserAllowlist)
        << "User's serialized schema no longer matches the D18 leak-audit allowlist.\n"
        << "A NEW field on User must be CONSCIOUSLY allowlisted here as an observable (update "
           "BOTH src/domain/serialization.cpp's to_json(User) and kUserAllowlist in this file) "
           "before this test may pass again; a field that disappeared should have its key "
           "removed from the allowlist instead. User is the recommender-visible profile (D11) — "
           "hidden/latent simulator-only fields must NEVER be added to this schema.\n"
        << "  emitted JSON: " << j.dump();
}

// (c) belt-and-braces: regardless of the exact-allowlist tests above, no emitted key in either
// schema may contain any of the forbidden hidden-state substrings, at any nesting depth.
TEST(LeakAuditTest, NoEmittedKeyContainsForbiddenHiddenStateSubstrings) {
    const nlohmann::json eventJson = fullyPopulatedEvent();
    const nlohmann::json userJson = fullyPopulatedUser();

    std::vector<std::string> keys;
    collectKeys(eventJson, keys);
    collectKeys(userJson, keys);
    ASSERT_FALSE(keys.empty());

    for (const std::string &key : keys) {
        for (const char *forbidden : kForbiddenSubstrings) {
            EXPECT_EQ(key.find(forbidden), std::string::npos)
                << "emitted key '" << key << "' contains forbidden hidden-state substring '"
                << forbidden
                << "' — hidden/latent simulator-only state must never reach a serialized "
                   "InteractionEvent or User (D18).";
        }
    }
}

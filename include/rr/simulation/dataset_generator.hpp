#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rr/domain/creator.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/simulation/hidden/hidden_reel_state.hpp"
#include "rr/simulation/hidden/hidden_user_state.hpp"
#include "rr/simulation/modality_space.hpp"

namespace rr {

// A full synthetic dataset (TDD §9): topics, creators, reels, and users with their ground-truth
// hidden state, generated together from one master seed. The Realism V2 members (Phase 13) are
// populated only when realism.content_v2 is on: modalitySpaces holds the shared style centres,
// hiddenReelStates is index-aligned with reels (empty under gate-off, D17).
struct GeneratedDataset {
    std::vector<Topic> topics;
    std::vector<Creator> creators;
    std::vector<Reel> reels;
    std::vector<User> users;
    std::vector<HiddenUserState> hiddenStates;
    ModalitySpaces modalitySpaces{};
    std::vector<HiddenReelState> hiddenReelStates{};
};

// Generates topics, then creators, then reels, then users, each on its own named Rng stream
// forked from `seed` ("topics" / "creators" / "reels" / "users", design decision D8). Because the
// streams are independent, regenerating one subsystem (e.g. widening config.reels) never changes
// another (e.g. the generated users) — same seed, same stream name, same output.
//
// The two-argument overload is the V1 entry point: it delegates with a default-constructed
// RealismConfig (content_v2 off), so every pre-Phase-13 call site keeps byte-identical output.
GeneratedDataset generateDataset(const SimulationConfig &config, uint64_t seed);

// Realism V2 entry point (Phase 13). With realism.content_v2 OFF this is byte-identical to the
// two-argument overload and performs ZERO V2 draws (D17). With it ON, the V1 generators run
// unchanged first (V1 fields stay byte-identical), then the V2 augmentation runs on the three
// new independent streams (D19): modality spaces + reel attributes/archetypes on
// "reels-v2"/"archetypes" (augmentReelsV2), user channels/traits on "users-v2"
// (augmentUsersV2, which consumes the modality spaces as data, drawing nothing from any reel
// stream). New streams never perturb V1 streams and vice versa.
GeneratedDataset generateDataset(const SimulationConfig &config, const RealismConfig &realism,
                                 uint64_t seed);

// --- Phase 8 mid-simulation entity injection (TDD 18.5 cold-start metrics) -------------------
//
// Both append helpers draw from NEW, independent named rng streams forked from `masterSeed`
// ("users-injected" / "reels-injected"), never from the four original streams. This is the D8
// injection contract: enabling injection leaves `generateDataset`'s output and every original
// stream byte-identical, and (same masterSeed, same count) always produces identical injected
// entities. Injected entities continue the dense-id sequence (element i keeps id value i), so the
// factory's density check still holds after growth. Both return the index of the first appended
// element (= the size of the vector BEFORE the append), which the harness passes to
// Recommender::onReelsAppended / RetrievalEvaluator::appendReels. A count of 0 appends nothing and
// returns the current size. `config` supplies dimensions/topic count only; its `users`/`reels`
// counts are overridden by `count` internally.

// Append `count` freshly generated users to `ds.users` / `ds.hiddenStates` (index-aligned), using
// the EXISTING topics. The public User's estimated/long-term/session vectors are left as the
// generator leaves them (empty); the caller applies the frozen cold-start prior (TDD 11.1).
std::size_t appendUsers(GeneratedDataset &ds, const SimulationConfig &config, uint64_t masterSeed,
                        uint32_t count);

// Append `count` freshly generated reels to `ds.reels`, using the EXISTING topics/creators. These
// are the genuinely-fresh injected content: every appended reel's createdAt is overwritten to
// `createdAt` (the injection-time logical clock, D9) — the generator's random-window createdAt draw
// is discarded — while counters stay zero and `active` stays true from the normal reel machinery.
std::size_t appendReels(GeneratedDataset &ds, const SimulationConfig &config, uint64_t masterSeed,
                        uint32_t count, Timestamp createdAt);

} // namespace rr

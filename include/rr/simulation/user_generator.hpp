#pragma once

#include <cstdint>
#include <vector>

#include "rr/domain/creator.hpp" // Topic
#include "rr/domain/user.hpp"
#include "rr/infrastructure/config.hpp" // SimulationConfig
#include "rr/infrastructure/random.hpp" // Rng
#include "rr/simulation/hidden/hidden_user_state.hpp"

namespace rr {

// Documented sampling ranges for the per-user behavioural traits (TDD 9.3). These are the single
// source of truth: generateUsers samples within them and the tests verify every generated trait
// falls within them. Each is [lo, hi] with lo inclusive; rng.uniform draws in [lo, hi).
namespace userTraits {

inline constexpr int kMinPreferredTopics = 2;
inline constexpr int kMaxPreferredTopics = 5;

inline constexpr double kConcentrationLo = 0.5;
inline constexpr double kConcentrationHi = 4.0;
inline constexpr double kExploreLo = 0.0;
inline constexpr double kExploreHi = 1.0;
inline constexpr double kSessionLengthLo = 5.0;
inline constexpr double kSessionLengthHi = 40.0;
inline constexpr double kLikePropensityLo = 0.02;
inline constexpr double kLikePropensityHi = 0.25;
inline constexpr double kSharePropensityLo = 0.0;
inline constexpr double kSharePropensityHi = 0.10;
inline constexpr double kDurationToleranceLo = 0.0;
inline constexpr double kDurationToleranceHi = 1.0;
inline constexpr double kStabilityLo = 0.0;
inline constexpr double kStabilityHi = 1.0;

// Topic-mix weights are a_i = base^concentration with base drawn from [kWeightBaseLo, 1.0). The
// positive floor keeps every weight strictly positive (a_i > 0) as TDD 9.3 requires.
inline constexpr double kWeightBaseLo = 0.1;

// Std-dev of the per-dimension additive gaussian noise ε applied before normalization.
inline constexpr double kNoiseScale = 0.05;

} // namespace userTraits

// Index-aligned output of user generation: users[i] and hiddenStates[i] describe the same user,
// with hiddenStates[i].userId == users[i].id == UserId{idOffset + i}.
struct GeneratedUsers {
    std::vector<User> users;
    std::vector<HiddenUserState> hiddenStates;
};

// Deterministically generate config.users users (TDD 9.3). The Rng is already the caller's forked
// "users" stream (design decision D8) — this function consumes it and never calls forkRng, so the
// generated users depend only on this stream, never on how reels/creators are generated.
//
// For each user (id 0..config.users-1): a sparse mix of 2-5 distinct topics (weights shaped by a
// per-user concentration trait) plus gaussian noise, L2-normalized into hiddenPreference; and each
// HiddenUserState behavioural trait sampled within its documented range. The public recommender-
// visible User carries no hidden state (D11) and no estimated-preference state yet (its estimated/
// long-term/session preferences are left empty — cold-start initialization happens in Phase 4).
//
// `idOffset` (Phase 8 mid-simulation injection) shifts the assigned UserId values so appended users
// continue the dense-id sequence past the existing population; it affects ONLY id assignment
// (users[i].id and hiddenStates[i].userId), never any rng draw, so idOffset == 0 (the default)
// reproduces the original byte-identical output.
//
// Throws std::invalid_argument (a setup error, per D10) if config.users > 0 and topics is empty.
GeneratedUsers generateUsers(const SimulationConfig &config, const std::vector<Topic> &topics,
                             Rng &rng, uint32_t idOffset = 0);

} // namespace rr

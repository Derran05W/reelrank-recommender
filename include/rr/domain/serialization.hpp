#pragma once

#include <nlohmann/json_fwd.hpp>

#include "rr/domain/interaction.hpp"
#include "rr/domain/user.hpp"

namespace rr {

// Canonical JSON schemas for the two recommender-visible structs whose serialized form is a
// hidden-state leak surface (D18): InteractionEvent (the observable event record — the Phase 22
// training log MUST serialize events through this function, never ad hoc) and User (the
// recommender-visible profile). The leak-audit test (tests/unit/leak_audit_test.cpp) asserts the
// emitted key sets against an explicit field-name allowlist, so ANY field added to either struct
// shows up as an audit failure until it is consciously allowlisted as observable — that review
// moment is the point of the audit. Keys are snake_case (D6/D10 serialization convention).
//
// PACKAGE-C OWNERSHIP, FROZEN SIGNATURES: package C implements these in
// src/domain/serialization.cpp (currently a scaffolding stub) together with the allowlist test.
void to_json(nlohmann::json &j, const InteractionEvent &e);
void to_json(nlohmann::json &j, const User &u);

} // namespace rr

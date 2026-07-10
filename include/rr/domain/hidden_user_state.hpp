#pragma once

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"

namespace rr {

// Ground-truth user state owned SOLELY by the simulator (design decision D11). The recommender
// must never see this type — the "recommender never accesses hidden preference" property is thus
// a structural, compile-time guarantee. Behavioural traits are added here in Phase 2.
struct HiddenUserState {
    UserId userId;
    Embedding hiddenPreference;
};

} // namespace rr

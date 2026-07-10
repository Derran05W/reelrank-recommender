#pragma once

#include "rr/domain/interaction.hpp"
#include "rr/domain/reel.hpp"
#include "rr/domain/user.hpp"

namespace rr {

// TDD 23.4.
class UserStateUpdater {
  public:
    virtual void apply(User &user, const Reel &reel, const InteractionEvent &interaction) const = 0;

    virtual ~UserStateUpdater() = default;
};

} // namespace rr

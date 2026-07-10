#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace rr {

// Strong 32-bit identifier (design decision D4). Distinct tags produce distinct, non-
// interconvertible types, so a ReelId can never be passed where a UserId is expected.
template <class Tag> struct Id {
    uint32_t value = 0;

    // Defaulting <=> also declares a matching defaulted ==, giving ==, !=, <, <=, >, >=.
    friend auto operator<=>(const Id &, const Id &) = default;
};

struct ReelTag {};
struct UserTag {};
struct CreatorTag {};
struct TopicTag {};
struct SessionTag {};

using ReelId = Id<ReelTag>;
using UserId = Id<UserTag>;
using CreatorId = Id<CreatorTag>;
using TopicId = Id<TopicTag>;
using SessionId = Id<SessionTag>;

} // namespace rr

namespace std {
template <class Tag> struct hash<rr::Id<Tag>> {
    std::size_t operator()(const rr::Id<Tag> &id) const noexcept {
        return std::hash<uint32_t>{}(id.value);
    }
};
} // namespace std

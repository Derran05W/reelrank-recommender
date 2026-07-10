#pragma once

#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"

namespace rr {

// TDD 9.4.
struct Creator {
    CreatorId id;
    Embedding styleEmbedding;
    std::vector<TopicId> topicSpecialties;
    float baseQuality;
};

// TDD 9.1 topic space.
struct Topic {
    TopicId id;
    Embedding centre;
};

} // namespace rr

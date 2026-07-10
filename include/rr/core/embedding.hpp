#pragma once

#include <vector>

namespace rr {

// Semantic vector. Plain std per TDD 8.1 / design decision D5; conversion to vector-db's Vector
// happens only inside the vindex adapters.
using Embedding = std::vector<float>;

// L2-normalize in place. Throws std::invalid_argument if empty, if any component is non-finite,
// or if the norm is below 1e-12. The norm is accumulated in double.
void normalize(Embedding &e);

// Dot product. Throws std::invalid_argument on size mismatch. Accumulated in double, returned as
// float.
float dot(const Embedding &a, const Embedding &b);

// True iff non-empty, all-finite, and unit-length within 1e-4.
bool isValid(const Embedding &e);

// Convert a Euclidean distance between unit vectors into cosine similarity (design decision D3;
// deliberately overrides TDD 12.1's 1/(1+d) suggestion). For unit vectors d^2 = 2 - 2 cos, so
// similarity = 1 - d^2/2.
float similarityFromEuclidean(float d);

} // namespace rr

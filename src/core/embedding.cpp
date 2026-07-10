#include "rr/core/embedding.hpp"

#include <cmath>
#include <stdexcept>

namespace rr {

void normalize(Embedding &e) {
    if (e.empty()) {
        throw std::invalid_argument("rr::normalize: embedding is empty");
    }
    double sumSq = 0.0;
    for (float v : e) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument("rr::normalize: embedding has a non-finite component");
        }
        sumSq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double norm = std::sqrt(sumSq);
    if (norm < 1e-12) {
        throw std::invalid_argument("rr::normalize: embedding norm is too small");
    }
    for (float &v : e) {
        v = static_cast<float>(static_cast<double>(v) / norm);
    }
}

float dot(const Embedding &a, const Embedding &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("rr::dot: embedding size mismatch");
    }
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(acc);
}

bool isValid(const Embedding &e) {
    if (e.empty()) {
        return false;
    }
    double sumSq = 0.0;
    for (float v : e) {
        if (!std::isfinite(v)) {
            return false;
        }
        sumSq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double norm = std::sqrt(sumSq);
    return std::abs(norm - 1.0) <= 1e-4;
}

float similarityFromEuclidean(float d) { return 1.0f - 0.5f * d * d; }

} // namespace rr

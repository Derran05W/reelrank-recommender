#include "rr/simulation/language.hpp"

#include <stdexcept>

namespace rr {

std::vector<double> languageWeights(uint32_t languages) {
    if (languages == 0) {
        throw std::invalid_argument("languageWeights: realism.languages must be >= 1");
    }
    std::vector<double> weights(languages);
    double sum = 0.0;
    for (uint32_t i = 0; i < languages; ++i) {
        weights[i] = 1.0 / static_cast<double>(i + 1);
        sum += weights[i];
    }
    for (double &w : weights) {
        w /= sum;
    }
    return weights;
}

} // namespace rr
